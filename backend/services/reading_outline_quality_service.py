"""Pure, cache-compatible diagnostics for thematic reading outlines.

The reading-outline cache is intentionally long lived and bound to a parser
identity.  Product-quality diagnostics must therefore be safe to recompute
from an older cache without changing its identity, writing it back, or
triggering another expensive model pass.
"""
from __future__ import annotations

import re
from typing import Any


_EMPIRICAL_PAPER_TYPES = frozenset({
    "empirical_method",
    "empirical_study",
    "dataset_or_benchmark",
})
_SETUP_SECTION_TITLE_RE = re.compile(
    r"(?:datasets?|data\s*set|experimental\s+setup|evaluation\s+setup|"
    r"implementation\s+details|protocol|数据集|实验设置|评测设置|实现细节|协议)",
    re.IGNORECASE,
)


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _normalized_text(value: Any) -> str:
    """Use the same lightweight identity shape as outline claim audits."""

    import re

    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").casefold())


def _derive_landmark_coverage_from_cached_outline(
    *,
    items: Any,
    section_items: Any,
) -> dict[str, Any]:
    """Recover qualitative landmarks from a pre-landmark cache without writes.

    Older outlines predate ``meta.landmark_result_coverage`` for prose claims,
    yet they already persist an exact ``claim_text + evidence_block_id +
    evidence_quote`` contract on every section.  Requiring a rewrite just to
    surface those claims would defeat cache compatibility, so derive a
    conservative response-only diagnostic here.  It intentionally uses only
    direct claims whose source section is explicitly included in the existing
    experiment theme.
    """

    def flatten(nodes: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for node in nodes if isinstance(nodes, list) else []:
            if not isinstance(node, dict):
                continue
            result.append(node)
            result.extend(flatten(node.get("children")))
        return result

    experiment = next(
        (
            item for item in (items if isinstance(items, list) else []) if isinstance(item, dict)
            and str(item.get("type") or "") == "theme_experiment"
        ),
        {},
    )
    source_ids = {
        str(value).strip()
        for value in experiment.get("source_section_ids") or []
        if str(value).strip()
    }
    if not source_ids:
        return {}
    study = experiment.get("study") if isinstance(experiment.get("study"), dict) else {}
    experiment_text = " ".join([
        str(experiment.get("summary") or ""),
        *(str(value or "") for value in study.get("findings") or []),
    ])
    experiment_key = _normalized_text(experiment_text)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section in flatten(section_items):
        section_id = str(section.get("source_section_id") or section.get("id") or "").strip()
        if section_id not in source_ids:
            continue
        if _SETUP_SECTION_TITLE_RE.search(str(section.get("title") or "")):
            continue
        for claim in section.get("prose_claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_text = str(claim.get("claim_text") or "").strip()
            evidence_block_id = str(claim.get("evidence_block_id") or "").strip()
            evidence_quote = str(claim.get("evidence_quote") or "").strip()
            claim_key = _normalized_text(claim_text)
            if not claim_key or not evidence_block_id or not evidence_quote or claim_key in seen:
                continue
            kind = str(claim.get("claim_kind") or "").strip().lower()
            if kind not in {"comparison", "causal", "limitation"}:
                continue
            seen.add(claim_key)
            candidates.append({
                "claim_id": f"legacy-prose:{section_id}:{len(candidates) + 1}",
                "category": "main_result",
                "source_section_id": section_id,
                "claim_kind": kind,
                "evidence_kind": "prose",
                "covered": bool(claim_key in experiment_key),
            })
    if not candidates:
        return {}
    return {
        "expected_claim_count": len(candidates),
        "covered_claim_count": sum(1 for item in candidates if item["covered"]),
        "coverage_ratio": sum(1 for item in candidates if item["covered"]) / len(candidates),
        "missing_claim_ids": [item["claim_id"] for item in candidates if not item["covered"]],
        "claims": candidates,
        "derived_from_cached_prose_claims": True,
    }


def semantic_summary_quality_diagnostics(
    *,
    overview_coverage: dict[str, Any] | None,
    landmark_result_coverage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe thematic-summary gaps without changing structural validity.

    ``section_coverage`` proves that the parser-bound per-section evidence
    review ran.  It does not prove that the reader-facing synthesis exposes
    every important semantic slot.  This helper deliberately returns a
    non-blocking diagnostic so callers may surface the distinction without
    invalidating a healthy cache.
    """

    overview = overview_coverage if isinstance(overview_coverage, dict) else {}
    landmark = (
        landmark_result_coverage
        if isinstance(landmark_result_coverage, dict)
        else {}
    )
    paper_type = str(overview.get("paper_type") or "unknown").strip().lower()
    missing_slots = [
        str(value).strip()
        for value in overview.get("missing_slots") or []
        if str(value).strip()
    ]
    required_slots = _non_negative_int(overview.get("required_slot_count"))
    covered_slots = _non_negative_int(overview.get("covered_slot_count"))
    expected_landmarks = _non_negative_int(landmark.get("expected_claim_count"))
    covered_landmarks = _non_negative_int(landmark.get("covered_claim_count"))
    landmark_empty_for_empirical = bool(
        paper_type in _EMPIRICAL_PAPER_TYPES and expected_landmarks == 0
    )

    issues: list[str] = []
    if missing_slots:
        issues.append("missing_overview_slots:" + ",".join(missing_slots))
    if expected_landmarks and covered_landmarks < expected_landmarks:
        issues.append(
            f"landmark_claims_partial:{covered_landmarks}/{expected_landmarks}"
        )
    if landmark_empty_for_empirical:
        # A zero denominator must not silently become a perfect landmark score.
        # It signals that a numerical/result-oriented paper has no independently
        # selected landmark, not that its parser-bound structure is invalid.
        issues.append("empirical_landmarks_empty")

    return {
        "status": "needs_review" if issues else "healthy",
        "paper_type": paper_type,
        "required_slot_count": required_slots,
        "covered_slot_count": covered_slots,
        "missing_slots": missing_slots,
        "landmark_expected_claim_count": expected_landmarks,
        "landmark_covered_claim_count": covered_landmarks,
        "landmark_empty_for_empirical": landmark_empty_for_empirical,
        "issues": issues,
        "blocking": False,
    }


def _recompute_ledgers_with_current_audit(metadata: dict[str, Any]) -> dict[str, Any]:
    """Re-run the generation-side audits over the cached tree, without writes.

    The import is deferred because ``reading_outline_service`` imports this
    module at top level; at call time it is always fully initialized.  Any
    failure falls back to the persisted-ledger path rather than degrading the
    response.
    """

    section_items = metadata.get("_section_items")
    outline_items = metadata.get("_outline_items")
    if not isinstance(section_items, list) or not isinstance(outline_items, list):
        return {}
    try:
        from services.reading_outline_service import recompute_semantic_ledgers_for_cache

        return recompute_semantic_ledgers_for_cache(section_items, outline_items) or {}
    except Exception:
        return {}


def semantic_summary_quality_from_metadata(meta: Any) -> dict[str, Any]:
    """Return diagnostics for a cached outline under the *current* audit rules.

    The return value is a fresh dictionary.  In particular this function never
    mutates ``meta``: rendering an old summary must not look like a cache write
    or force a different reading-outline source hash on a later request.

    When the cached tree carries per-theme ``source_section_ids`` the two
    ledgers are recomputed with the generation-side audit functions, so an
    outline generated under older slot ownership or before qualitative
    landmark selection is judged by today's contract instead of its stale
    persisted ledger.  For a fresh outline the recomputation is identical to
    what generation just persisted, so the verdict never diverges.
    """

    metadata = meta if isinstance(meta, dict) else {}
    recomputed = _recompute_ledgers_with_current_audit(metadata)
    if recomputed:
        diagnostics = semantic_summary_quality_diagnostics(
            overview_coverage=recomputed.get("overview_coverage"),
            landmark_result_coverage=recomputed.get("landmark_result_coverage"),
        )
        return {
            **diagnostics,
            "ledgers_recomputed_from_cache": True,
        }
    overview = metadata.get("overview_coverage")
    landmark = metadata.get("landmark_result_coverage")
    persisted = metadata.get("semantic_quality")
    has_coverage_ledger = isinstance(overview, dict) or isinstance(landmark, dict)
    if not has_coverage_ledger and not (isinstance(persisted, dict) and persisted):
        # No semantic ledger was ever produced for this outline.  Keep the
        # public result empty/"unknown" rather than turning missing metadata
        # into an implicit healthy quality verdict.
        return {}
    base = (
        dict(persisted)
        if isinstance(persisted, dict) and persisted
        else semantic_summary_quality_diagnostics(
            overview_coverage=overview if isinstance(overview, dict) else {},
            landmark_result_coverage=landmark if isinstance(landmark, dict) else {},
        )
    )
    # A legacy cache can have a persisted v4.16 diagnostic with a zero
    # landmark denominator.  Prefer a conservative derived prose ledger when
    # it proves there were real evidence-bound results; do not mutate the
    # cache or pretend a missing overview slot has been repaired.
    expected = _non_negative_int((landmark or {}).get("expected_claim_count") if isinstance(landmark, dict) else 0)
    derived = (
        _derive_landmark_coverage_from_cached_outline(
            items=metadata.get("_outline_items"),
            section_items=metadata.get("_section_items"),
        )
        if expected == 0
        else {}
    )
    if not derived:
        return base
    issues = [
        issue for issue in base.get("issues") or []
        if issue != "empirical_landmarks_empty"
    ]
    if derived["covered_claim_count"] < derived["expected_claim_count"]:
        issues.append(
            "landmark_claims_partial:"
            f"{derived['covered_claim_count']}/{derived['expected_claim_count']}"
        )
    return {
        **base,
        "status": "needs_review" if issues else "healthy",
        "landmark_expected_claim_count": derived["expected_claim_count"],
        "landmark_covered_claim_count": derived["covered_claim_count"],
        "landmark_empty_for_empirical": False,
        "issues": issues,
        "derived_landmark_result_coverage": derived,
        "blocking": False,
    }
