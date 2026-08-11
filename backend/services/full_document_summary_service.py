"""Render a parse-bound reading outline as a complete chat summary.

The chat route deliberately does not ask a second free-form model to summarize
the document.  ``reading_outline`` has already validated each section against
the active block index; this module only presents that verified structure and
keeps the block-to-citation mapping available to the reader.
"""
from __future__ import annotations

import re
from typing import Any


FULL_DOCUMENT_SUMMARY_RETRIEVAL_MODE = "reading_outline_full_document"


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


def _coverage_payload(
    outline: dict[str, Any],
    *,
    rendered_body_sections: int,
    rendered_appendix_sections: int,
    citation_count: int,
) -> dict[str, Any]:
    meta = outline.get("meta") if isinstance(outline.get("meta"), dict) else {}
    raw = meta.get("section_coverage") if isinstance(meta.get("section_coverage"), dict) else {}
    body_expected = _as_positive_int(raw.get("body_expected"))
    outlined_body_summarized = _as_positive_int(raw.get("body_summarized"))
    appendix_expected = _as_positive_int(raw.get("appendix_expected"))
    outlined_appendix_summarized = _as_positive_int(raw.get("appendix_summarized"))
    # The client reads the rendered answer, not the cached outline in
    # isolation. Clamp the coverage ledger to entries actually shown so a
    # malformed nested tree cannot advertise chapters the reply omitted.
    body_summarized = min(outlined_body_summarized, max(0, rendered_body_sections))
    appendix_summarized = min(outlined_appendix_summarized, max(0, rendered_appendix_sections))
    rendered_sections = body_summarized + appendix_summarized
    body_complete = body_expected == 0 or body_summarized >= body_expected
    appendix_complete = appendix_expected == 0 or appendix_summarized >= appendix_expected
    status = str(meta.get("generation_status") or "").strip().lower()
    source = str(outline.get("source") or "").strip().lower()
    complete = bool(
        rendered_sections > 0
        and body_complete
        and appendix_complete
        and status not in {"partial", "failed", "unavailable"}
    )
    if rendered_sections <= 0:
        status = "unavailable"
    return {
        "mode": FULL_DOCUMENT_SUMMARY_RETRIEVAL_MODE,
        "source": source or "fallback",
        "generation_status": status or ("completed" if complete else "partial"),
        "body_expected": body_expected,
        "body_summarized": body_summarized,
        "appendix_expected": appendix_expected,
        "appendix_summarized": appendix_summarized,
        "body_complete": body_complete,
        "appendix_complete": appendix_complete,
        "complete": complete,
        "rendered_section_count": rendered_sections,
        "citation_count": citation_count,
        "retryable": bool(meta.get("retryable")) or rendered_sections <= 0,
        "partial_quality_issues": [
            _clean_text(item, limit=160)
            for item in (meta.get("partial_quality_issues") or [])
            if _clean_text(item, limit=160)
        ][:5],
    }


def build_full_document_summary(
    outline: dict[str, Any],
    block_index: dict[str, Any],
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

    def citation_suffix(item: dict[str, Any]) -> str:
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

    thematic_items = [
        item
        for item in normalized_outline.get("items") or []
        if isinstance(item, dict)
        and (
            str(item.get("id") or "") == "paper_overview"
            or str(item.get("type") or "").startswith("theme_")
        )
        and _clean_text(item.get("summary"), limit=500)
    ]
    if thematic_items:
        for item in thematic_items:
            heading = _clean_text(item.get("title"), limit=120) or "论文要点"
            summary = _clean_text(item.get("summary"), limit=500)
            parts.extend([f"### {heading}", f"{summary}{citation_suffix(item)}"])
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
                f"{_clean_text(overview.get('summary'), limit=500)}{citation_suffix(overview)}",
            ])

    section_nodes = _ordered_section_entries(
        normalized_outline.get("section_items")
        if isinstance(normalized_outline.get("section_items"), list)
        else []
    )
    visible_sections = [
        item
        for item in section_nodes
        if str(item.get("section_kind") or "body") in {"body", "appendix"}
        and _clean_text(item.get("summary"), limit=500)
    ]
    if visible_sections:
        parts.append("### 按章节梳理")
        for item in visible_sections:
            heading = _clean_text(item.get("title"), limit=140) or "未命名章节"
            summary = _clean_text(item.get("summary"), limit=500)
            depth = max(0, min(3, _as_positive_int(item.get("_summary_depth"), 0)))
            indent = "  " * depth
            parts.append(f"{indent}- **{heading}**：{summary}{citation_suffix(item)}")

    if len(parts) == 1:
        parts.append("当前解析结果尚未形成可验证的全文结构化总结，请在阅读面板重新生成后再试。")

    answer = "\n\n".join(parts).strip()
    coverage = _coverage_payload(
        normalized_outline,
        rendered_body_sections=sum(
            1 for item in visible_sections
            if str(item.get("section_kind") or "body") == "body"
        ),
        rendered_appendix_sections=sum(
            1 for item in visible_sections
            if str(item.get("section_kind") or "body") == "appendix"
        ),
        citation_count=len(citations),
    )
    return {
        "answer": answer,
        "citations": citations,
        "coverage": coverage,
    }
