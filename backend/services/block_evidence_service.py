"""Canonical block-index evidence for retrieval and downstream citations.

The reading block index is the authoritative representation of the selected
parse route.  This module keeps retrieval from flattening that representation
into anonymous page text before it reaches the vector index.
"""
from __future__ import annotations

import re
from typing import Any


EVIDENCE_SCHEMA_VERSION = 2

_EXCLUDED_BLOCK_TYPES = frozenset({
    "artifact",
    "image",
    "table_row",
    "table_cell",
})
_EVIDENCE_BLOCK_TYPES = frozenset({
    "heading",
    "paragraph",
    "caption",
    "formula",
    "code",
    "figure",
    "table",
    "visual_enrichment",
})
_SECONDARY_CONTENT_ROLES = frozenset({
    "reference",
    "author",
    "affiliation",
    "contact",
    "publication_header",
})


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        bbox = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _line_rects(block: dict[str, Any]) -> list[list[float]]:
    rects: list[list[float]] = []
    for anchor in block.get("line_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        bbox = _bbox(anchor.get("bbox"))
        if bbox:
            rects.append(bbox)
    return rects[:64]


def _section_paths(block_index: dict[str, Any]) -> dict[str, str]:
    """Turn the flat, level-aware outline into stable readable paths."""
    paths: dict[str, str] = {}
    stack: list[str] = []
    for item in block_index.get("outline") or []:
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("section_id") or "").strip()
        title = _text(item.get("title"))
        if not section_id or not title:
            continue
        try:
            level = max(1, min(6, int(item.get("level") or 1)))
        except (TypeError, ValueError):
            level = 1
        stack = stack[: level - 1]
        stack.append(title)
        paths[section_id] = " > ".join(stack)
    return paths


def _is_retrievable_block(block: dict[str, Any], block_type: str) -> bool:
    if block_type in _EXCLUDED_BLOCK_TYPES:
        return False
    if block.get("repeated_artifact"):
        return False
    if block.get("text_duplicate") or block.get("highlight_of"):
        # Empty-box slots keep geometry for highlight, not a second retrieval copy.
        return False
    if block.get("visual_enhancement"):
        # Visual supplements only reach a public block index after the
        # document-level publish marker is written.  Require its revision here
        # as a second guard so staged VLM output cannot become retrievable.
        if block_type not in {"caption", "visual_enrichment"} or not str(block.get("visual_supplement_revision") or "").strip():
            return False
    if block.get("exclude_from_reading"):
        return False
    text = _text(block.get("text") or block.get("content") or block.get("caption"))
    # Bare labels such as "Figure 1" are navigation noise.  A figure only
    # becomes retrievable when MinerU supplied a substantive caption/description.
    if block_type == "figure":
        return len(text) >= 24
    return bool(text)


def _evidence_text(block: dict[str, Any], block_type: str) -> str:
    text = _text(block.get("text") or block.get("content") or block.get("caption"))
    if block_type == "table":
        text = _text(re.sub(r"<[^>]+>", " ", text))
    return text


def _page_number(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def collect_highlight_slots(block_index: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Map a MinerU source block id to the visual boxes left after stitching."""
    slots: dict[str, list[dict[str, Any]]] = {}
    for page_index, page in enumerate(block_index.get("pages") or []):
        if not isinstance(page, dict):
            continue
        try:
            page_number = max(1, int(page.get("page") or page_index + 1))
        except (TypeError, ValueError):
            page_number = page_index + 1
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            source_id = str(block.get("highlight_of") or block.get("linked_content_id") or "").strip()
            if not source_id or not (block.get("text_duplicate") or block.get("highlight_of")):
                continue
            bbox = _bbox(block.get("bbox"))
            if not bbox:
                continue
            slots.setdefault(source_id, []).append({
                "block_id": str(block.get("block_id") or ""),
                "page": page_number,
                "bbox": bbox,
            })
    return slots


def build_highlight_page_rects(
    *,
    primary_page: int,
    primary_bbox: list[float] | None = None,
    primary_rects: list[list[float]] | None = None,
    slots: list[dict[str, Any]] | None = None,
    existing_page_rects: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Group highlight boxes by page so consumers never paint a foreign-page rect."""
    by_page: dict[int, list[list[float]]] = {}

    def _add(page: int, bbox: list[float] | None) -> None:
        normalized = _bbox(bbox)
        if page <= 0 or not normalized:
            return
        rects = by_page.setdefault(page, [])
        if normalized not in rects:
            rects.append(normalized)

    if isinstance(existing_page_rects, list):
        for item in existing_page_rects:
            if not isinstance(item, dict):
                continue
            page = _page_number(item.get("page"))
            for rect in item.get("rects") or []:
                _add(page, rect)
            _add(page, item.get("bbox"))

    for rect in primary_rects or []:
        _add(primary_page, rect)
    _add(primary_page, primary_bbox)

    for slot in slots or []:
        if not isinstance(slot, dict):
            continue
        _add(_page_number(slot.get("page")), slot.get("bbox"))
        for rect in slot.get("rects") or []:
            _add(_page_number(slot.get("page")), rect)

    return [
        {"page": page, "rects": rects[:64]}
        for page, rects in sorted(by_page.items())
        if rects
    ]


def _apply_highlight_slots(metadata: dict[str, Any], slots: list[dict[str, Any]]) -> None:
    """Keep retrieval identity on the original block; paint the visual slot."""
    if not slots:
        return
    slot_ids = [str(slot.get("block_id") or "") for slot in slots if slot.get("block_id")]
    block_ids = list(metadata.get("block_ids") or [])
    for slot_id in slot_ids:
        if slot_id and slot_id not in block_ids:
            block_ids.append(slot_id)
    metadata["block_ids"] = block_ids
    metadata["highlight_block_ids"] = [
        str(metadata.get("block_id") or ""),
        *slot_ids,
    ]
    primary_page = _page_number(metadata.get("page"))
    page_rects = build_highlight_page_rects(
        primary_page=primary_page,
        primary_bbox=_bbox(metadata.get("bbox")),
        primary_rects=list(metadata.get("rects") or []),
        slots=slots,
        existing_page_rects=metadata.get("page_rects") if isinstance(metadata.get("page_rects"), list) else None,
    )
    if page_rects:
        metadata["page_rects"] = page_rects
        primary_entry = next((item for item in page_rects if item["page"] == primary_page), None)
        if primary_entry:
            metadata["rects"] = list(primary_entry["rects"])
        pages = [_page_number(item.get("page")) for item in page_rects]
        valid_pages = [page for page in pages if page > 0]
        if valid_pages:
            metadata["page_range"] = [min(valid_pages), max(valid_pages)]
    metadata["highlight_slots"] = slots


def build_rag_source_from_block_index(
    block_index: dict[str, Any] | None,
    *,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> dict[str, Any]:
    """Build page text plus identity-preserving evidence inputs from blocks.

    ``evidence_chunks`` deliberately remains pre-split.  The embedding service
    owns model-aware chunk sizes and can split a very long block without losing
    its source identity.
    """
    if not isinstance(block_index, dict):
        return {
            "full_text": "",
            "pages": [],
            "evidence_chunks": [],
            "block_count": 0,
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        }

    section_paths = _section_paths(block_index)
    highlight_slots = collect_highlight_slots(block_index)
    active_generation = str(parse_generation or block_index.get("parse_generation") or "").strip()
    active_source_hash = str(
        document_source_hash or block_index.get("document_source_hash") or ""
    ).strip()
    parser_route = str(block_index.get("parser_route") or "").strip()
    pages: list[dict[str, Any]] = []
    evidence_chunks: list[dict[str, Any]] = []
    block_count = 0

    for page_index, page in enumerate(block_index.get("pages") or []):
        if not isinstance(page, dict):
            continue
        try:
            page_number = max(1, int(page.get("page") or page_index + 1))
        except (TypeError, ValueError):
            page_number = page_index + 1
        try:
            page_width = float(page.get("width_pts") or 612.0)
            page_height = float(page.get("height_pts") or 792.0)
        except (TypeError, ValueError):
            page_width, page_height = 612.0, 792.0

        page_parts: list[str] = []
        seen_page_text: set[str] = set()
        for block_order, block in enumerate(page.get("blocks") or []):
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or block.get("block_type") or "paragraph").strip().lower()
            if not _is_retrievable_block(block, block_type):
                continue
            text = _evidence_text(block, block_type)
            if not text:
                continue
            text_key = re.sub(r"\s+", " ", text).casefold()
            if text_key not in seen_page_text:
                seen_page_text.add(text_key)
                page_parts.append(text)
            block_count += 1

            # Headings serve as section context; body/caption/formula blocks
            # are the retrievable evidence.  Table evidence still travels via
            # the dedicated structured-table path in embedding_service.
            if block_type not in _EVIDENCE_BLOCK_TYPES:
                continue
            block_id = str(block.get("block_id") or f"p{page_number}_b{block_order}").strip()
            if not block_id:
                continue
            section_id = str(block.get("section_id") or "").strip()
            section_path = section_paths.get(section_id, "")
            bbox = _bbox(block.get("bbox"))
            metadata: dict[str, Any] = {
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "evidence_source": "block_index",
                "evidence_id": f"block:{block_id}",
                "block_id": block_id,
                "block_ids": [block_id],
                "section_id": section_id,
                "section_path": section_path,
                "page": page_number,
                "page_range": [page_number, page_number],
                "block_type": block_type,
                "chunk_type": block_type,
                "content_role": str(block.get("content_role") or "body").strip().lower() or "body",
                "reading_order": int(block.get("reading_order") or block_order),
                "parse_generation": active_generation,
                "document_source_hash": active_source_hash,
                "parser_route": parser_route,
                "source": str(block.get("source") or "block_index"),
                "page_size": [page_width, page_height],
                "coordinate_space": "pdf_top_left_points",
            }
            if metadata["content_role"] in _SECONDARY_CONTENT_ROLES:
                metadata["retrieval_tier"] = "secondary"
            else:
                metadata["retrieval_tier"] = "primary"
            if block_type == "table":
                metadata["table_text_fallback"] = True
            for key in (
                "figure_id",
                "purpose",
                "visual_source",
                "visual_supplement_revision",
                "prompt_version",
                "confidence",
            ):
                value = block.get(key)
                if value not in (None, ""):
                    metadata[key] = value
            if block.get("visual_enhancement"):
                metadata["visual_enhancement"] = True
            if bbox:
                metadata["bbox"] = bbox
            rects = _line_rects(block)
            if rects:
                metadata["rects"] = rects
            _apply_highlight_slots(metadata, highlight_slots.get(block_id) or [])
            evidence_chunks.append({
                "text": text,
                "heading": section_path,
                "page": page_number,
                "block_type": block_type,
                "metadata": metadata,
            })

        page_content = "\n\n".join(page_parts).strip()
        if page_content:
            pages.append({
                "page": page_number,
                "page_index": page_number - 1,
                "content": page_content,
                "text": page_content,
                "source": "block_index",
            })

    return {
        "full_text": "\n\n".join(page["content"] for page in pages).strip(),
        "pages": pages,
        "evidence_chunks": evidence_chunks,
        "block_count": block_count,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
    }
