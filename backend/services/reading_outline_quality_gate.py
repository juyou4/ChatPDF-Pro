"""Deterministic quality contract for a persisted thematic reading outline.

This is deliberately stricter than the runtime fail-open policy. Runtime
generation may keep a mostly useful document outline when a provider times out;
the fixed fixture gate instead tells us whether a known-good summary shape has
regressed before it reaches a release.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import combinations
from typing import Any


THEMATIC_ITEM_TYPES = (
    "overview",
    "theme_background",
    "theme_innovation",
    "theme_experiment",
    "theme_conclusion",
)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _ngrams(value: Any, size: int = 3) -> set[str]:
    text = _clean_text(value)
    if not text:
        return set()
    if len(text) <= size:
        return {text}
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def _jaccard(left: Any, right: Any) -> float:
    left_tokens = _ngrams(left)
    right_tokens = _ngrams(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _flatten_sections(nodes: Any) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []

    def walk(values: Any) -> None:
        for value in values if isinstance(values, list) else []:
            if not isinstance(value, dict):
                continue
            flat.append(value)
            walk(value.get("children"))

    walk(nodes)
    return flat


def _theme_texts(items: Iterable[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for item in items:
        summary = str(item.get("summary") or "").strip()
        if summary:
            texts.append(summary)
        study = item.get("study") if isinstance(item.get("study"), dict) else {}
        for finding in study.get("findings") or []:
            if isinstance(finding, dict):
                value = "：".join(
                    part for part in (
                        str(finding.get("label") or "").strip(),
                        str(finding.get("text") or finding.get("summary") or "").strip(),
                    ) if part
                )
            else:
                value = str(finding or "").strip()
            if value:
                texts.append(value)
    return texts


def _issue(kind: str, **detail: Any) -> dict[str, Any]:
    return {"kind": kind, **detail}


def evaluate_reading_outline_quality(
    outline: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate structural and semantic ledgers without calling a model.

    It is intended for fixed fixtures and result artifacts, not as a runtime
    hard failure. The function accepts a small threshold mapping so tests and
    CI can make expectations explicit and replay results exactly.
    """

    config = thresholds if isinstance(thresholds, dict) else {}
    issues: list[dict[str, Any]] = []
    items = [item for item in outline.get("items") or [] if isinstance(item, dict)]
    item_types = tuple(str(item.get("type") or "").strip() for item in items)
    if item_types != THEMATIC_ITEM_TYPES:
        issues.append(_issue(
            "thematic_contract",
            actual=list(item_types),
            expected=list(THEMATIC_ITEM_TYPES),
        ))

    expected_theme_counts = {"theme_innovation": 3, "theme_experiment": 5}
    for item in items:
        limit = expected_theme_counts.get(str(item.get("type") or ""))
        if limit is None:
            continue
        study = item.get("study") if isinstance(item.get("study"), dict) else {}
        finding_count = len(study.get("findings") or [])
        if finding_count > limit:
            issues.append(_issue(
                "too_many_theme_findings",
                theme=item.get("type"),
                actual=finding_count,
                maximum=limit,
            ))

    section_nodes = _flatten_sections(outline.get("section_items"))
    section_ids = [
        str(item.get("source_section_id") or item.get("id") or "").strip()
        for item in section_nodes
        if str(item.get("source_section_id") or item.get("id") or "").strip()
    ]
    duplicate_ids = sorted({item_id for item_id in section_ids if section_ids.count(item_id) > 1})
    if duplicate_ids:
        issues.append(_issue("duplicate_section_ids", ids=duplicate_ids))

    section_summaries = [
        str(item.get("summary") or "").strip()
        for item in section_nodes
        if str(item.get("summary") or "").strip()
    ]
    theme_texts = _theme_texts(items)
    max_replay_similarity = float(config.get("section_replay_similarity") or 0.80)
    replay_count = sum(
        1
        for theme_text in theme_texts
        if any(_jaccard(theme_text, section_summary) >= max_replay_similarity for section_summary in section_summaries)
    )
    replay_ratio = replay_count / len(theme_texts) if theme_texts else 1.0
    maximum_replay_ratio = float(config.get("max_section_replay_ratio") or 0.40)
    if replay_ratio > maximum_replay_ratio:
        issues.append(_issue(
            "section_replay_ratio",
            actual=round(replay_ratio, 4),
            maximum=maximum_replay_ratio,
            replayed_sentence_count=replay_count,
            theme_sentence_count=len(theme_texts),
        ))

    summary_pairs = [
        (left, right, _jaccard(left, right))
        for left, right in combinations(section_summaries, 2)
        if len(_clean_text(left)) >= 20 and len(_clean_text(right)) >= 20
    ]
    near_duplicate_threshold = float(config.get("near_duplicate_similarity") or 0.85)
    near_duplicates = [
        {"similarity": round(similarity, 4), "left": left, "right": right}
        for left, right, similarity in summary_pairs
        if similarity >= near_duplicate_threshold
    ]
    if near_duplicates:
        issues.append(_issue("near_duplicate_section_summaries", pairs=near_duplicates[:5]))

    meta = outline.get("meta") if isinstance(outline.get("meta"), dict) else {}
    section_coverage = meta.get("section_coverage") if isinstance(meta.get("section_coverage"), dict) else {}
    for kind in ("body", "appendix"):
        expected = _int(section_coverage.get(f"{kind}_expected"))
        summarized = _int(section_coverage.get(f"{kind}_summarized"))
        if summarized < expected:
            issues.append(_issue(
                "section_coverage",
                section_kind=kind,
                expected=expected,
                summarized=summarized,
            ))

    overview_coverage = meta.get("overview_coverage") if isinstance(meta.get("overview_coverage"), dict) else {}
    required_slots = _int(overview_coverage.get("required_slot_count"))
    covered_slots = _int(overview_coverage.get("covered_slot_count"))
    missing_slots = [str(value) for value in overview_coverage.get("missing_slots") or [] if str(value).strip()]
    if required_slots <= 0 or covered_slots < required_slots or missing_slots:
        issues.append(_issue(
            "overview_slot_coverage",
            required_slot_count=required_slots,
            covered_slot_count=covered_slots,
            missing_slots=missing_slots,
        ))

    landmark = meta.get("landmark_result_coverage") if isinstance(meta.get("landmark_result_coverage"), dict) else {}
    paper_type = str(overview_coverage.get("paper_type") or "unknown").strip().lower()
    expected_claims = _int(landmark.get("expected_claim_count"))
    covered_claims = _int(landmark.get("covered_claim_count"))
    min_empirical_claims = _int(config.get("min_empirical_landmark_claims"))
    empirical_types = {"empirical_method", "empirical_study", "dataset_or_benchmark"}
    if paper_type in empirical_types and expected_claims < min_empirical_claims:
        issues.append(_issue(
            "empirical_landmark_empty_or_insufficient",
            paper_type=paper_type,
            expected_claim_count=expected_claims,
            minimum=min_empirical_claims,
        ))
    elif expected_claims and covered_claims < expected_claims:
        issues.append(_issue(
            "landmark_coverage",
            expected_claim_count=expected_claims,
            covered_claim_count=covered_claims,
            missing_claim_ids=list(landmark.get("missing_claim_ids") or []),
        ))

    sampling = meta.get("sampling") if isinstance(meta.get("sampling"), dict) else {}
    recommended_sections = _int(sampling.get("review_recommended_sections"))
    max_review_recommended = _int(config.get("max_review_recommended_sections"))
    if recommended_sections > max_review_recommended:
        issues.append(_issue(
            "sampling_review_recommended",
            actual=recommended_sections,
            maximum=max_review_recommended,
        ))

    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "metrics": {
            "thematic_item_types": list(item_types),
            "section_replay_ratio": round(replay_ratio, 4),
            "near_duplicate_pair_count": len(near_duplicates),
            "section_count": len(section_ids),
            "theme_statement_count": len(theme_texts),
            "required_slot_count": required_slots,
            "covered_slot_count": covered_slots,
            "landmark_expected_claim_count": expected_claims,
            "landmark_covered_claim_count": covered_claims,
            "review_recommended_sections": recommended_sections,
        },
        "gate_version": str(config.get("version") or "reading-outline-gate-v1"),
    }
