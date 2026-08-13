"""Render a parse-bound reading outline as a complete chat summary.

The chat route deliberately does not ask a second free-form model to summarize
the document.  ``reading_outline`` has already validated each section against
the active block index; this module only presents that verified structure and
keeps the block-to-citation mapping available to the reader.
"""
from __future__ import annotations

import re
from typing import Any

from services.citation_alignment_service import claim_support_score
from services.full_document_summary_request import (
    is_full_document_section_summary_request,
)
from services.reading_outline_quality_service import semantic_summary_quality_from_metadata


FULL_DOCUMENT_SUMMARY_RETRIEVAL_MODE = "reading_outline_full_document"
# The reading outline is the source contract; this version identifies the
# user-facing projection of that contract.  Changing the projection must not
# invalidate MinerU or the much more expensive section-summary cache.
FULL_DOCUMENT_SUMMARY_RENDER_VERSION = "full-document-summary-v2"

# Reader-facing order: what the paper asks, how it answers, what it measured,
# what that is worth.  ``reading_outline`` already emits themes in this order,
# but a cache is long lived and the renderer is the last place that can keep a
# reordered or hand-edited outline from reaching the user.
_THEME_RENDER_ORDER = (
    "overview",
    "theme_background",
    "theme_innovation",
    "theme_experiment",
    "theme_conclusion",
)
# Mirrors the generation-side ``THEME_SPECS`` point budget.  Enforcing it again
# here bounds the answer even when the cache predates that budget.
_THEME_FINDING_LIMITS = {"theme_innovation": 3, "theme_experiment": 5}
_DEFAULT_THEME_FINDING_LIMIT = 3
# Only treat a section summary as echoed when the overlap is long enough that a
# verbatim match cannot be coincidental.  Measured on normalized characters.
_MIN_SECTION_ECHO_CHARS = 16

def should_render_section_details(question: Any = "") -> bool:
    """Return whether the user explicitly requested a chapter-level view.

    A full-document request alone should produce a semantic synthesis.  The
    structural outline remains available to the reading panel and is only
    projected into chat when the user asks for chapter/subsection detail.
    """

    return is_full_document_section_summary_request(question)


def _clean_text(value: Any, *, limit: int = 1_400) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _as_positive_int(value: Any, default: int = 0) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return default
    return normalized if normalized > 0 else default


def _flatten_block_index(block_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for page_index, raw_page in enumerate(block_index.get("pages") or [], start=1):
        page = raw_page if isinstance(raw_page, dict) else {}
        page_number = _as_positive_int(page.get("page") or page.get("page_num"), page_index)
        for raw_block in page.get("blocks") or []:
            if not isinstance(raw_block, dict):
                continue
            block_id = str(raw_block.get("block_id") or "").strip()
            if not block_id:
                continue
            block = dict(raw_block)
            block.setdefault("page", page_number)
            blocks[block_id] = block
    return blocks


def _ordered_section_entries(nodes: list[Any]) -> list[dict[str, Any]]:
    """Return every structural section in reading order with its tree depth.

    A parent section can have a concise, evidence-bound overview that is not
    repeated by any of its children.  Rendering leaves only therefore makes a
    deeply nested MinerU outline look complete in metadata while silently
    omitting part of the document in the chat answer.
    """
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(raw: Any, depth: int) -> None:
        if not isinstance(raw, dict):
            return
        source_section_id = str(raw.get("source_section_id") or "").strip()
        stable_id = source_section_id or str(raw.get("id") or "").strip()
        if stable_id and stable_id not in seen:
            seen.add(stable_id)
            entries.append({**raw, "_summary_depth": max(0, depth)})
        for child in raw.get("children") or []:
            visit(child, depth + 1)

    for node in nodes or []:
        visit(node, 0)
    return entries


def _theme_identity(item: dict[str, Any]) -> str:
    if str(item.get("id") or "").strip() == "paper_overview":
        return "overview"
    return str(item.get("type") or "").strip()


def _ordered_thematic_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort themes into the fixed reading order, keeping unknown kinds last."""

    def rank(item: dict[str, Any]) -> int:
        identity = _theme_identity(item)
        return (
            _THEME_RENDER_ORDER.index(identity)
            if identity in _THEME_RENDER_ORDER
            else len(_THEME_RENDER_ORDER)
        )

    # ``sorted`` is stable, so unrecognized themes keep their cached order
    # instead of being dropped or shuffled.
    return sorted(items, key=rank)


def _echo_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").casefold())


def _section_summary_echo_keys(section_nodes: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for item in section_nodes:
        key = _echo_key(item.get("summary"))
        if len(key) >= _MIN_SECTION_ECHO_CHARS:
            keys.add(key)
    return keys


def _is_section_echo(text: Any, echo_keys: set[str]) -> bool:
    """Detect a thematic point that restates one section summary verbatim.

    When the model omits a theme, the outline falls back to relabelling section
    summaries as "points", so a high echo count means the synthesis degraded
    into a repackaged chapter walkthrough.  Matching is verbatim-only (one
    string contains the other after normalization) so a genuinely synthesized
    point that merely resembles a section is never flagged.
    """

    body = str(text or "").split("：", 1)[-1]
    key = _echo_key(body)
    if len(key) < _MIN_SECTION_ECHO_CHARS:
        return False
    return any(echo in key or key in echo for echo in echo_keys)


def _normalized_findings(
    item: dict[str, Any],
    *,
    echo_keys: set[str] | None = None,
    drop_echo: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """Normalize thematic findings from old and new outline cache shapes.

    Echoed points are always counted but only dropped when ``drop_echo`` is
    set.  A section restatement is redundant next to a rendered chapter list,
    yet in the thematic projection the chapter list is absent, so the same
    point may be the reader's only access to that conclusion.  Suppressing it
    there would remove information rather than remove duplication.
    """

    study = item.get("study") if isinstance(item.get("study"), dict) else {}
    raw_findings = study.get("findings") if isinstance(study.get("findings"), list) else []
    raw_evidence = (
        study.get("finding_evidence")
        if isinstance(study.get("finding_evidence"), list)
        else []
    )
    evidence_by_text: dict[str, dict[str, Any]] = {}
    for entry in raw_evidence:
        if not isinstance(entry, dict):
            continue
        raw_text = entry.get("text") or entry.get("finding") or entry.get("summary")
        key = re.sub(r"\s+", " ", str(raw_text or "")).strip()
        if key:
            evidence_by_text[key] = entry
    normalized_echo_keys = echo_keys or set()
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    echoed = 0
    considered = 0
    for raw in raw_findings:
        if isinstance(raw, dict):
            label = _clean_text(raw.get("label"), limit=120)
            text = _clean_text(raw.get("text") or raw.get("summary"), limit=420)
            value = f"{label}：{text}" if label and text else (text or label)
        else:
            value = _clean_text(raw, limit=420)
        key = re.sub(r"\s+", " ", value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        considered += 1
        if _is_section_echo(key, normalized_echo_keys):
            echoed += 1
            if drop_echo:
                continue
        evidence = evidence_by_text.get(key) or {}
        evidence_ids = [
            str(block_id).strip()
            for block_id in evidence.get("evidence_block_ids") or []
            if str(block_id).strip()
        ]
        findings.append({"text": key, "evidence_block_ids": evidence_ids})
    limit = _THEME_FINDING_LIMITS.get(_theme_identity(item), _DEFAULT_THEME_FINDING_LIMIT)
    return findings[:limit], echoed, considered


def _section_detail_entries(section_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose a readable structural appendix without duplicating synthesized parents."""

    def has_renderable_descendant(item: dict[str, Any]) -> bool:
        for raw_child in item.get("children") or []:
            if not isinstance(raw_child, dict):
                continue
            kind = str(raw_child.get("section_kind") or "body").strip().lower()
            if kind in {"body", "appendix"} and _clean_text(raw_child.get("summary"), limit=500):
                return True
            if has_renderable_descendant(raw_child):
                return True
        return False

    visible: list[dict[str, Any]] = []
    for item in section_nodes:
        if (
            str(item.get("section_status") or "").strip().lower() == "synthesized"
            and has_renderable_descendant(item)
        ):
            # This parent is only a compressed copy of its children.  Showing
            # it together with every child recreates the old double summary.
            # Keep it when its descendants are not themselves renderable: a
            # synthesized parent is still the only useful fallback in that
            # malformed/partial shape.
            continue
        visible.append(item)
    return visible


def _coverage_payload(
    outline: dict[str, Any],
    *,
    presentation_mode: str,
    visible_section_count: int,
    has_narrative_content: bool,
    citation_count: int,
    structural_coverage: dict[str, list[str]] | None = None,
    section_echo_finding_count: int = 0,
    themes_without_evidence: list[str] | None = None,
    themes_restating_sections: list[str] | None = None,
) -> dict[str, Any]:
    meta = outline.get("meta") if isinstance(outline.get("meta"), dict) else {}
    raw = meta.get("section_coverage") if isinstance(meta.get("section_coverage"), dict) else {}
    body_expected = _as_positive_int(raw.get("body_expected"))
    outlined_body_summarized = _as_positive_int(raw.get("body_summarized"))
    appendix_expected = _as_positive_int(raw.get("appendix_expected"))
    outlined_appendix_summarized = _as_positive_int(raw.get("appendix_summarized"))
    # Structural coverage describes the generation/evidence contract, not how
    # many navigation entries we choose to print in a semantic summary.  The
    # previous implementation clamped this to visible bullets, which forced
    # the chat answer to dump the entire MinerU tree to remain "complete".
    body_summarized = outlined_body_summarized
    appendix_summarized = outlined_appendix_summarized
    rendered_sections = body_summarized + appendix_summarized
    body_complete = body_expected == 0 or body_summarized >= body_expected
    appendix_complete = appendix_expected == 0 or appendix_summarized >= appendix_expected
    status = str(meta.get("generation_status") or "").strip().lower()
    source = str(outline.get("source") or "").strip().lower()
    # Pre-diagnostic caches already contain the two source ledgers.  Derive a
    # response-only diagnostic from them rather than invalidating, rewriting,
    # or regenerating a perfectly reusable outline.
    # The quality helper is intentionally cache-compatible.  It receives the
    # persisted tree only as an ephemeral diagnostic input so old outlines can
    # recover evidence-bound qualitative landmarks without a write or rerun.
    quality_meta = {
        **meta,
        "_outline_items": outline.get("items") or [],
        "_section_items": outline.get("section_items") or [],
    }
    meta_quality = semantic_summary_quality_from_metadata(quality_meta)
    # Two defects are only observable while projecting the outline: a theme
    # whose points merely relabel sections, and a theme whose evidence blocks
    # are all missing from the active index.  Neither invalidates the
    # structural ledger, so they join the same non-blocking diagnostic.
    echoed_findings = max(0, int(section_echo_finding_count or 0))
    ungrounded_themes = [
        str(value).strip()
        for value in themes_without_evidence or []
        if str(value).strip()
    ]
    restating_themes = [
        str(value).strip()
        for value in themes_restating_sections or []
        if str(value).strip()
    ]
    render_issues: list[str] = []
    if restating_themes:
        render_issues.append("themes_restating_sections:" + ",".join(restating_themes))
    if ungrounded_themes:
        render_issues.append("themes_without_evidence:" + ",".join(ungrounded_themes))
    if render_issues:
        meta_quality = {
            **(meta_quality or {}),
            "status": "needs_review",
            "issues": [*(meta_quality.get("issues") or []), *render_issues],
            "blocking": False,
        }
    # Render diagnostics are reported even when the outline carries no semantic
    # ledger: a nonzero echo count is worth observing below the degradation
    # threshold, and an absent ledger must still read as "unknown" rather than
    # as a healthy verdict, which holds because no ``status`` is invented here.
    meta_quality = {
        **(meta_quality or {}),
        "section_echo_finding_count": echoed_findings,
        "themes_restating_sections": restating_themes,
        "themes_without_evidence": ungrounded_themes,
    }
    quality_status = str(meta_quality.get("status") or "").strip().lower()
    complete = bool(
        has_narrative_content
        and body_complete
        and appendix_complete
        and status not in {"partial", "failed", "unavailable"}
    )
    if not has_narrative_content:
        status = "unavailable"
    structural_expected = body_expected + appendix_expected
    structural_summarized = body_summarized + appendix_summarized
    structural_coverage = structural_coverage if isinstance(structural_coverage, dict) else {}
    raw_body_ids = [str(value).strip() for value in structural_coverage.get("body_expected_ids") or [] if str(value).strip()]
    raw_appendix_ids = [str(value).strip() for value in structural_coverage.get("appendix_expected_ids") or [] if str(value).strip()]
    raw_body_covered_ids = [str(value).strip() for value in structural_coverage.get("body_covered_ids") or [] if str(value).strip()]
    raw_appendix_covered_ids = [str(value).strip() for value in structural_coverage.get("appendix_covered_ids") or [] if str(value).strip()]
    return {
        "mode": FULL_DOCUMENT_SUMMARY_RETRIEVAL_MODE,
        "render_version": FULL_DOCUMENT_SUMMARY_RENDER_VERSION,
        "source": source or "fallback",
        "generation_status": status or ("completed" if complete else "partial"),
        "body_expected": body_expected,
        "body_summarized": body_summarized,
        "appendix_expected": appendix_expected,
        "appendix_summarized": appendix_summarized,
        "body_complete": body_complete,
        "appendix_complete": appendix_complete,
        "complete": complete,
        # Legacy clients interpreted this as the parse-bound structural
        # section count.  Preserve that value while `visible_section_count`
        # reports the deliberately smaller chat projection.
        "rendered_section_count": structural_summarized,
        "visible_section_count": max(0, int(visible_section_count or 0)),
        "structural_section_count": structural_summarized,
        "structural_expected_count": structural_expected,
        "presentation_mode": presentation_mode,
        "semantic_quality_status": quality_status or "unknown",
        "semantic_quality": dict(meta_quality),
        "structural_coverage": {
            "body_expected_ids": raw_body_ids,
            "body_covered_ids": raw_body_covered_ids,
            "appendix_expected_ids": raw_appendix_ids,
            "appendix_covered_ids": raw_appendix_covered_ids,
        },
        "citation_count": citation_count,
        "retryable": bool(meta.get("retryable")) or not has_narrative_content,
        "partial_quality_issues": [
            _clean_text(item, limit=160)
            for item in (meta.get("partial_quality_issues") or [])
            if _clean_text(item, limit=160)
        ][:5],
    }


def build_full_document_summary(
    outline: dict[str, Any],
    block_index: dict[str, Any],
    *,
    question: str = "",
    include_section_details: bool | None = None,
) -> dict[str, Any]:
    """Build markdown, citations and coverage metadata from ``reading_outline``.

    Every narrative entry gets a citation to one of its persisted evidence
    blocks.  No prose is generated here, which keeps the answer stable across
    chat model changes and prevents a local retrieval miss from becoming a
    claim about the full document.
    """
    normalized_outline = outline if isinstance(outline, dict) else {}
    block_map = _flatten_block_index(block_index if isinstance(block_index, dict) else {})
    citations: list[dict[str, Any]] = []
    citations_by_block: dict[str, int] = {}

    def citation_suffix(item: dict[str, Any], *, claim_text: Any = "") -> str:
        evidence_ids = [
            str(value).strip()
            for value in (item.get("evidence_block_ids") or [])
            if str(value).strip() in block_map
        ]
        if not evidence_ids:
            first_block = str(item.get("first_block") or "").strip()
            if first_block in block_map:
                evidence_ids = [first_block]
        if not evidence_ids:
            return ""
        normalized_claim = _clean_text(claim_text, limit=700)
        if normalized_claim and len(evidence_ids) > 1:
            # A thematic point can be supported by a different section from
            # the theme-level overview.  Rank only the already-authorized
            # blocks; this never widens the source scope or generates prose.
            evidence_ids.sort(
                key=lambda block_id: claim_support_score(
                    normalized_claim,
                    {"source_text": str(block_map[block_id].get("text") or "")},
                ),
                reverse=True,
            )
        block_id = evidence_ids[0]
        existing = citations_by_block.get(block_id)
        if existing is not None:
            return f" [{existing}]"
        block = block_map[block_id]
        ref = len(citations) + 1
        page = _as_positive_int(block.get("page"), 1)
        text = _clean_text(block.get("text"), limit=1_600)
        source_section_id = str(item.get("source_section_id") or "").strip()
        item_id = str(item.get("id") or "").strip()
        citation = {
            "ref": ref,
            "evidence_id": f"reading-outline:{source_section_id or item_id}:{block_id}",
            "context_id": block_id,
            "block_id": block_id,
            "chunk_id": block_id,
            "group_id": f"reading-outline:{source_section_id or item_id or block_id}",
            "chunk_type": str(block.get("type") or block.get("block_type") or "paragraph"),
            "block_type": str(block.get("block_type") or block.get("type") or "paragraph"),
            "page_range": [page, page],
            "source_text": text,
            "display_text": text,
            "highlight_text": _clean_text(block.get("text"), limit=220),
            "_full_text": text,
            "retrieval_type": FULL_DOCUMENT_SUMMARY_RETRIEVAL_MODE,
            "reading_outline": True,
            "source_section_id": source_section_id,
        }
        citations.append(citation)
        citations_by_block[block_id] = ref
        return f" [{ref}]"

    title = _clean_text(normalized_outline.get("title"), limit=180) or "全文总结"
    parts = [f"## {title}"]

    section_nodes = _ordered_section_entries(
        normalized_outline.get("section_items")
        if isinstance(normalized_outline.get("section_items"), list)
        else []
    )
    section_echo_keys = _section_summary_echo_keys(section_nodes)

    thematic_items = _ordered_thematic_items([
        item
        for item in normalized_outline.get("items") or []
        if isinstance(item, dict)
        and (
            str(item.get("id") or "") == "paper_overview"
            or str(item.get("type") or "").startswith("theme_")
        )
        and _clean_text(item.get("summary"), limit=500)
    ])
    has_thematic_summary = bool(thematic_items)
    show_section_details = (
        should_render_section_details(question)
        if include_section_details is None
        else bool(include_section_details)
    )
    show_section_details = bool(show_section_details or not has_thematic_summary)
    section_echo_finding_count = 0
    themes_without_evidence: list[str] = []
    themes_restating_sections: list[str] = []
    if thematic_items:
        for item in thematic_items:
            heading = _clean_text(item.get("title"), limit=120) or "论文要点"
            summary = _clean_text(item.get("summary"), limit=500)
            theme_suffix = citation_suffix(item, claim_text=summary)
            parts.extend([
                f"### {heading}",
                f"{summary}{theme_suffix}",
            ])
            findings, echoed, considered = _normalized_findings(
                item,
                echo_keys=section_echo_keys,
                drop_echo=show_section_details,
            )
            section_echo_finding_count += echoed
            if considered >= 2 and echoed == considered:
                # Every point in this theme is a section restatement, which is
                # what the fallback payload produces when the model omits the
                # theme.  One echoed point among several is normal: a section
                # conclusion can legitimately be a key result.
                themes_restating_sections.append(_theme_identity(item) or heading)
            finding_suffixes: list[str] = []
            for finding in findings:
                suffix = citation_suffix(
                    {
                        **item,
                        "evidence_block_ids": finding["evidence_block_ids"]
                        or item.get("evidence_block_ids"),
                    },
                    claim_text=finding["text"],
                )
                finding_suffixes.append(suffix)
                parts.append(f"- {finding['text']}{suffix}")
            if not theme_suffix and not any(finding_suffixes):
                # The theme carries no resolvable evidence block at all.  It is
                # still shown, because suppressing it would silently shrink the
                # summary, but the gap must be visible downstream.
                themes_without_evidence.append(
                    _theme_identity(item) or heading
                )
    else:
        overview = next(
            (
                item
                for item in normalized_outline.get("items") or []
                if isinstance(item, dict) and str(item.get("id") or "") == "paper_overview"
            ),
            None,
        )
        if isinstance(overview, dict) and _clean_text(overview.get("summary"), limit=500):
            parts.extend([
                "### 论文要旨",
                f"{_clean_text(overview.get('summary'), limit=500)}"
                f"{citation_suffix(overview, claim_text=overview.get('summary'))}",
            ])

    section_detail_nodes = _section_detail_entries(section_nodes)
    visible_sections = [
        item
        for item in section_detail_nodes
        if str(item.get("section_kind") or "body") in {"body", "appendix"}
        and _clean_text(item.get("summary"), limit=500)
    ]
    if show_section_details and visible_sections:
        parts.append("### 章节梳理")
        for item in visible_sections:
            heading = _clean_text(item.get("title"), limit=140) or "未命名章节"
            summary = _clean_text(item.get("summary"), limit=500)
            depth = max(0, min(3, _as_positive_int(item.get("_summary_depth"), 0)))
            indent = "  " * depth
            parts.append(
                f"{indent}- **{heading}**：{summary}"
                f"{citation_suffix(item, claim_text=summary)}"
            )

    if has_thematic_summary and not show_section_details:
        meta = normalized_outline.get("meta") if isinstance(normalized_outline.get("meta"), dict) else {}
        raw_coverage = meta.get("section_coverage") if isinstance(meta.get("section_coverage"), dict) else {}
        body_expected = _as_positive_int(raw_coverage.get("body_expected"))
        body_summarized = _as_positive_int(raw_coverage.get("body_summarized"))
        appendix_expected = _as_positive_int(raw_coverage.get("appendix_expected"))
        appendix_summarized = _as_positive_int(raw_coverage.get("appendix_summarized"))
        parts.append(
            "*结构覆盖：正文 "
            f"{body_summarized}/{body_expected} 节，附录 {appendix_summarized}/{appendix_expected} 节；"
            "详细章节导航见阅读大纲。*"
        )

    def section_id(item: dict[str, Any]) -> str:
        return str(item.get("source_section_id") or item.get("id") or "").strip()

    structural_coverage = {
        "body_expected_ids": [],
        "body_covered_ids": [],
        "appendix_expected_ids": [],
        "appendix_covered_ids": [],
    }
    for item in section_nodes:
        item_id = section_id(item)
        kind = str(item.get("section_kind") or "body").strip().lower()
        if not item_id or kind not in {"body", "appendix"} or not item.get("evidence_block_ids"):
            continue
        expected_key = f"{kind}_expected_ids"
        covered_key = f"{kind}_covered_ids"
        if item_id not in structural_coverage[expected_key]:
            structural_coverage[expected_key].append(item_id)
        if (
            _clean_text(item.get("summary"), limit=500)
            and str(item.get("section_status") or "").strip().lower() not in {"fallback", "unavailable"}
        ):
            structural_coverage[covered_key].append(item_id)

    has_renderable_narrative = bool(
        thematic_items or (show_section_details and visible_sections)
    )
    if not has_renderable_narrative:
        parts.append("当前解析结果尚未形成可验证的全文结构化总结，请在阅读面板重新生成后再试。")

    answer = "\n\n".join(parts).strip()
    presentation_mode = "thematic" if has_thematic_summary and not show_section_details else "section_detail"
    coverage = _coverage_payload(
        normalized_outline,
        presentation_mode=presentation_mode,
        visible_section_count=len(visible_sections) if show_section_details else 0,
        has_narrative_content=has_renderable_narrative,
        structural_coverage=structural_coverage,
        citation_count=len(citations),
        section_echo_finding_count=section_echo_finding_count,
        themes_without_evidence=themes_without_evidence,
        themes_restating_sections=themes_restating_sections,
    )
    return {
        "answer": answer,
        "citations": citations,
        "coverage": coverage,
    }
