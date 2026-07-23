"""MinerU result adapter for ChatPDF immersive block index.

MinerU pipeline/VLM results already contain page blocks, semantic types and bbox
anchors. This module converts those results into the same block index schema used
by PDF-native parsing so downstream outline, hover translation and highlighting
can stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.block_index_service import (
    BLOCK_INDEX_VERSION,
    _assign_sections,
    _annotate_block_roles,
    _build_outline,
    _exclude_post_reference_template_blocks,
    _limit_text,
    _normalize_bbox,
    stamp_block_index_revision,
)
from services.mineru_text_normalizer import (
    build_mineru_page_ledger,
)

logger = logging.getLogger(__name__)

MINERU_BLOCK_INDEX_SOURCE = "mineru_vlm"
MINERU_RAW_VERSION = 1
MINERU_STRUCTURE_VERSION = 5


def get_mineru_result_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "mineru_results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_mineru_result_path(data_dir: Path | str, doc_id: str) -> Path:
    return get_mineru_result_dir(data_dir) / f"{doc_id}.json"


def save_mineru_result(
    data_dir: Path | str,
    doc_id: str,
    payload: dict[str, Any],
    *,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> None:
    """Persist a raw MinerU result with the parse run that produced it.

    ``doc_id`` is derived from the original PDF bytes, so the same PDF can be
    uploaded again with a different primary parse route.  Raw MinerU output
    must therefore be tied to the parse generation rather than being treated
    as an unqualified document-level cache.
    """
    path = get_mineru_result_path(data_dir, doc_id)
    serializable = {
        "version": MINERU_RAW_VERSION,
        "doc_id": doc_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parse_generation": str(parse_generation or ""),
        "document_source_hash": str(document_source_hash or ""),
        "payload": payload,
    }
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def load_mineru_result(
    data_dir: Path | str,
    doc_id: str,
    *,
    parse_generation: str | None = None,
    document_source_hash: str | None = None,
    require_identity: bool = False,
) -> dict[str, Any] | None:
    """Load raw MinerU output, optionally only for one parse generation.

    Legacy records predate parse manifests and do not carry an identity.  They
    remain readable unless a caller explicitly requests identity validation;
    newly routed documents must use that validation before rebuilding blocks or
    an index from the raw payload.
    """
    path = get_mineru_result_path(data_dir, doc_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != MINERU_RAW_VERSION:
            return None
        stored_generation = str(data.get("parse_generation") or "")
        stored_source_hash = str(data.get("document_source_hash") or "")
        if require_identity and (not stored_generation or not stored_source_hash):
            return None
        if parse_generation is not None and stored_generation != str(parse_generation or ""):
            return None
        if document_source_hash is not None and stored_source_hash != str(document_source_hash or ""):
            return None
        payload = data.get("payload")
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.warning("[MinerUBlockIndex] failed to load %s: %s", path, exc)
        return None


def build_block_index_from_mineru_payload(
    *,
    doc_id: str,
    doc: dict[str, Any],
    payload: dict[str, Any],
    pdf_path: Path | str | None = None,
) -> dict[str, Any]:
    """Convert MinerU JSON payload into ChatPDF block index."""
    page_specs = _load_page_specs(doc, pdf_path)
    payload_page_source_sizes = _mineru_page_source_sizes(payload)
    middle_json = payload.get("middle_json")
    content_list_json = payload.get("content_list_json")
    middle_blocks: dict[int, list[dict[str, Any]]] = {}
    content_blocks: dict[int, list[dict[str, Any]]] = {}
    if isinstance(middle_json, dict):
        middle_blocks = _blocks_from_middle_json(middle_json, page_specs, source_name="middle_json")
    if isinstance(content_list_json, list):
        content_blocks = _blocks_from_content_list(
            content_list_json,
            page_specs,
            source_name="content_list_json",
            page_source_sizes=payload_page_source_sizes,
        )
    elif isinstance(middle_json, list):
        content_blocks = _blocks_from_content_list(
            middle_json,
            page_specs,
            source_name="middle_json_list",
            page_source_sizes=payload_page_source_sizes,
        )
    blocks_by_page = _merge_block_sources(middle_blocks, content_blocks)
    blocks_by_page = {
        page_num: _split_run_in_numbered_heading_blocks(blocks)
        for page_num, blocks in blocks_by_page.items()
    }

    # The PDF metadata may be stale or unavailable. Never drop a page which
    # MinerU actually returned merely because a stored total_pages is smaller.
    observed_page_nums = set(blocks_by_page) | _payload_page_numbers(payload)
    expected_page_count = max(set(page_specs) | observed_page_nums, default=0)
    if expected_page_count > 0:
        page_nums = list(range(1, expected_page_count + 1))
        for page_num in page_nums:
            page_specs.setdefault(page_num, {"width": 612.0, "height": 792.0})
    else:
        page_nums = sorted(observed_page_nums)
    pages: list[dict[str, Any]] = []
    for page_num in page_nums:
        spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
        # MinerU content_list is already in learned reading order.  Do not
        # replace it with a geometric ``top, left`` sort here: that ordering
        # interleaves two-column papers (right-column headings can appear
        # before later left-column headings).  The merge below keeps middle
        # JSON's stronger geometry while restoring content_list's sequence.
        blocks = list(blocks_by_page.get(page_num, []))
        for idx, block in enumerate(blocks):
            block["block_id"] = block.get("block_id") or f"p{page_num}_b{idx}"
            block["reading_order"] = idx
            block["section_id"] = None
        pages.append({
            "page": page_num,
            "width_pts": float(spec.get("width") or 612.0),
            "height_pts": float(spec.get("height") or 792.0),
            "blocks": blocks,
            "layout_regions": [],
        })

    block_count = sum(len(page.get("blocks") or []) for page in pages)
    semantic_block_pages = {
        int(page.get("page") or 0)
        for page in pages
        if any(
            isinstance(block, dict) and block.get("type") != "artifact"
            for block in (page.get("blocks") or [])
        )
    }
    text_pages = {
        int(page.get("page") or 0)
        for page in pages
        if any(
            block.get("type") != "artifact"
            and str(block.get("text") or "").strip() not in {"", "Figure", "Table"}
            for block in (page.get("blocks") or [])
            if isinstance(block, dict)
        )
    }
    body_text_chars = sum(
        len(str(block.get("text") or "").strip())
        for page in pages
        for block in (page.get("blocks") or [])
        if block.get("type") != "artifact"
        and str(block.get("text") or "").strip() not in {"Figure", "Table"}
    )
    page_quality = build_mineru_page_ledger(
        payload,
        expected_page_count=expected_page_count,
        block_pages=semantic_block_pages,
        text_pages=text_pages,
    )
    outline = _build_outline([], pages)
    _assign_sections(pages, outline)
    _annotate_block_roles(pages, outline)
    _exclude_post_reference_template_blocks(pages)
    structure_quality = _mineru_structure_diagnostics(payload, pages, outline)
    data = doc.get("data", {}) if isinstance(doc, dict) else {}
    manifest = data.get("parse_manifest") if isinstance(data, dict) else {}
    parse_identity = {}
    if isinstance(manifest, dict):
        generation = str(manifest.get("generation") or "").strip()
        source_hash = str(manifest.get("source_hash") or "").strip()
        route = str(manifest.get("resolved_route") or "").strip().lower()
        metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
        if generation and source_hash and route:
            parse_identity = {
                "parser_route": route,
                "parse_generation": generation,
                "document_source_hash": source_hash,
                "full_route": bool(metadata.get("full_route")),
            }

    return stamp_block_index_revision({
        "version": BLOCK_INDEX_VERSION,
        "doc_id": doc_id,
        "source": MINERU_BLOCK_INDEX_SOURCE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "pages": pages,
        "outline": outline,
        **parse_identity,
        "mineru_meta": {
            "raw_hash": _payload_hash(payload),
            "has_middle_json": isinstance(middle_json, (dict, list)),
            "has_content_list_json": isinstance(content_list_json, list),
            "zip_entries": payload.get("zip_entries") or [],
            "block_count": block_count,
            "body_text_chars": body_text_chars,
            "semantic_covered_page_count": page_quality.get("covered_page_count", 0),
            "semantic_page_coverage": page_quality.get("coverage", 0.0),
            **structure_quality,
            **page_quality,
        },
    })


def _load_page_specs(doc: dict[str, Any], pdf_path: Path | str | None) -> dict[int, dict[str, float]]:
    specs: dict[int, dict[str, float]] = {}
    pdf_file = Path(pdf_path) if pdf_path else None
    if pdf_file and pdf_file.exists():
        try:
            import fitz

            pdf_doc = fitz.open(str(pdf_file))
            try:
                for idx, page in enumerate(pdf_doc):
                    specs[idx + 1] = {
                        "width": float(page.rect.width),
                        "height": float(page.rect.height),
                    }
            finally:
                pdf_doc.close()
        except Exception as exc:
            logger.warning("[MinerUBlockIndex] failed to read PDF page sizes: %s", exc)

    data = doc.get("data", {}) if isinstance(doc, dict) else {}
    total_pages = int(data.get("total_pages") or len(data.get("pages", []) or []) or len(specs) or 1)
    for idx in range(1, total_pages + 1):
        specs.setdefault(idx, {"width": 612.0, "height": 792.0})
    return specs


def _mineru_page_source_sizes(payload: dict[str, Any]) -> dict[int, tuple[float, float]]:
    """Read declared MinerU page coordinate spaces before heuristic guessing."""
    sizes: dict[int, tuple[float, float]] = {}
    middle = payload.get("middle_json") if isinstance(payload, dict) else None
    if not isinstance(middle, dict):
        return sizes
    for page_info in middle.get("pdf_info") or []:
        if not isinstance(page_info, dict):
            continue
        source_size = _page_source_size(page_info)
        if source_size:
            sizes[_page_num(page_info)] = source_size
    return sizes


def _payload_page_numbers(payload: dict[str, Any]) -> set[int]:
    pages: set[int] = set()
    content = payload.get("content_list_json")
    if isinstance(content, list):
        pages.update(_page_num(item) for item in content if isinstance(item, dict))

    middle = payload.get("middle_json")
    if isinstance(middle, list):
        pages.update(_page_num(item) for item in middle if isinstance(item, dict))
    elif isinstance(middle, dict):
        pdf_info = middle.get("pdf_info")
        if isinstance(pdf_info, list):
            pages.update(_page_num(item) for item in pdf_info if isinstance(item, dict))
    return {page for page in pages if page > 0}


def _iter_raw_mineru_items(payload: dict[str, Any]):
    """Yield the raw items that participate in block-index conversion."""
    middle = payload.get("middle_json")
    if isinstance(middle, dict):
        pdf_info = middle.get("pdf_info")
        if isinstance(pdf_info, list):
            for page_info in pdf_info:
                if not isinstance(page_info, dict):
                    continue
                page_num = _page_num(page_info)
                for key in ("preproc_blocks", "para_blocks", "layout_blocks", "blocks"):
                    for item in page_info.get(key) or []:
                        if isinstance(item, dict):
                            yield page_num, item

    content = payload.get("content_list_json")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                yield _page_num(item), item
    elif isinstance(middle, list):
        for item in middle:
            if isinstance(item, dict):
                yield _page_num(item), item


def _mineru_structure_diagnostics(
    payload: dict[str, Any],
    pages: list[dict[str, Any]],
    outline: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_occurrences = 0
    raw_headings: set[tuple[Any, ...]] = set()
    raw_native_headings: set[tuple[Any, ...]] = set()
    raw_run_in_paths: set[tuple[int, str]] = set()
    for page_num, item in _iter_raw_mineru_items(payload):
        raw_type = _normalize_type(
            item.get("type")
            or item.get("block_type")
            or item.get("category")
            or item.get("category_name")
        )
        text_level = _valid_mineru_text_level(item)
        text = _extract_item_text(item, "paragraph")
        bbox = _item_bbox(item)
        bbox_key = tuple(round(float(value), 3) for value in bbox) if bbox else None
        fingerprint = _block_text_fingerprint(text)
        if raw_type == "text" and text_level is not None:
            raw_occurrences += 1
            raw_headings.add((page_num, fingerprint, bbox_key, text_level))
        elif raw_type in {"title", "heading", "section_header"} and fingerprint:
            raw_native_headings.add((page_num, fingerprint, bbox_key))
        elif raw_type in {
            "text", "plain_text", "paragraph", "para", "body_text", "list", "list_item",
        }:
            run_in = _RUN_IN_NUMBERED_HEADING_RE.match(" ".join(str(text or "").split()))
            if run_in and len(str(text or "")) >= 48:
                tail = str(run_in.group("tail") or "")
                if (
                    _RUN_IN_BODY_START_RE.search(tail)
                    or re.search(
                        r"[。！？](?=(?:根据|我们|本文|该|实验|结果|通过|在|从|对于|此外|因此|同时|随后|然后))",
                        tail,
                    )
                ):
                    raw_run_in_paths.add((page_num, str(run_in.group("number"))))

    emitted_headings = [
        block
        for page in pages
        for block in (page.get("blocks") or [])
        if isinstance(block, dict) and block.get("type") == "heading"
    ]
    emitted_text_level_headings = [
        block
        for block in emitted_headings
        if block.get("heading_source") == "mineru_text_level"
    ]
    intentionally_excluded_text_level_headings = [
        block
        for page in pages
        for block in (page.get("blocks") or [])
        if (
            isinstance(block, dict)
            and block.get("type") == "artifact"
            and block.get("heading_source") == "mineru_text_level"
            and block.get("structure_exclusion_reason") == "post_references_artifact"
        )
    ]
    emitted_native_heading_keys = {
        (
            int(page.get("page") or 1),
            _block_text_fingerprint(block.get("text")),
            tuple(round(float(value), 3) for value in _normalize_bbox(block.get("bbox")))
            if _normalize_bbox(block.get("bbox")) else None,
        )
        for page in pages
        for block in (page.get("blocks") or [])
        if isinstance(block, dict)
        and block.get("heading_source") == "mineru_type"
        and block.get("type") in {"heading", "artifact"}
    }
    emitted_numbered_paths = {
        (int(page.get("page") or 1), str(match.group(1)))
        for page in pages
        for block in (page.get("blocks") or [])
        if isinstance(block, dict)
        and block.get("type") == "heading"
        for match in [re.match(r"^\s*(\d+(?:\.\d+)+)(?:\.)?\s+", str(block.get("text") or ""))]
        if match
    }
    outline_numbered_paths = {
        (int(item.get("page") or 1), str(match.group(1)))
        for item in outline
        if isinstance(item, dict)
        for match in [re.match(r"^\s*(\d+(?:\.\d+)+)(?:\.)?\s+", str(item.get("title") or ""))]
        if match
    }
    raw_heading_count = len(raw_headings)
    raw_native_heading_count = len(raw_native_headings)
    emitted_text_level_heading_count = len(emitted_text_level_headings)
    intentionally_excluded_text_level_heading_count = len(intentionally_excluded_text_level_headings)
    covered_text_level_heading_count = (
        emitted_text_level_heading_count + intentionally_excluded_text_level_heading_count
    )
    missing_run_in_paths = sorted(raw_run_in_paths - outline_numbered_paths)
    return {
        "structure_version": MINERU_STRUCTURE_VERSION,
        "raw_text_level_heading_count": raw_heading_count,
        "raw_text_level_heading_occurrences": raw_occurrences,
        "emitted_heading_count": len(emitted_headings),
        "emitted_text_level_heading_count": emitted_text_level_heading_count,
        "intentionally_excluded_text_level_heading_count": intentionally_excluded_text_level_heading_count,
        "covered_text_level_heading_count": covered_text_level_heading_count,
        "raw_native_heading_count": raw_native_heading_count,
        "covered_native_heading_count": len(raw_native_headings & emitted_native_heading_keys),
        "raw_run_in_heading_count": len(raw_run_in_paths),
        "covered_run_in_heading_count": len(raw_run_in_paths & emitted_numbered_paths),
        "missing_run_in_outline_paths": [".".join((str(page), path)) for page, path in missing_run_in_paths],
        "outline_heading_count": len(outline),
        "structure_degraded": (
            covered_text_level_heading_count < raw_heading_count
            or len(raw_native_headings & emitted_native_heading_keys) < raw_native_heading_count
            or bool(missing_run_in_paths)
        ),
    }


def _blocks_from_middle_json(
    data: dict[str, Any],
    page_specs: dict[int, dict[str, float]],
    *,
    source_name: str,
) -> dict[int, list[dict[str, Any]]]:
    blocks_by_page: dict[int, list[dict[str, Any]]] = {}
    pdf_info = data.get("pdf_info")
    if not isinstance(pdf_info, list):
        return blocks_by_page

    for page_info in pdf_info:
        if not isinstance(page_info, dict):
            continue
        page_num = _page_num(page_info)
        page_source_size = _page_source_size(page_info)
        raw_blocks: list[dict[str, Any]] = []
        for key in ("preproc_blocks", "para_blocks", "layout_blocks", "blocks"):
            value = page_info.get(key)
            if isinstance(value, list):
                raw_blocks.extend(item for item in value if isinstance(item, dict))
        discarded = page_info.get("discarded_blocks")
        if isinstance(discarded, list):
            for item in discarded:
                if isinstance(item, dict):
                    clone = dict(item)
                    clone.setdefault("type", "discarded")
                    raw_blocks.append(clone)

        for raw in raw_blocks:
            block = _convert_mineru_item(raw, page_num, page_specs, page_source_size, source_name=source_name)
            if block:
                blocks_by_page.setdefault(page_num, []).append(block)
    return blocks_by_page


def _blocks_from_content_list(
    items: list[Any],
    page_specs: dict[int, dict[str, float]],
    *,
    source_name: str,
    page_source_sizes: dict[int, tuple[float, float]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    blocks_by_page: dict[int, list[dict[str, Any]]] = {}
    items_by_page: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        page_num = _page_num(item)
        items_by_page.setdefault(page_num, []).append(item)

    for page_num, page_items in items_by_page.items():
        page_spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
        page_source_size = (
            _page_source_size(page_items[0])
            or (page_source_sizes or {}).get(page_num)
            or _infer_page_source_size(
            page_items,
            page_width=float(page_spec.get("width") or 612.0),
            page_height=float(page_spec.get("height") or 792.0),
            )
        )
        for item in page_items:
            block = _convert_mineru_item(
                item,
                page_num,
                page_specs,
                _page_source_size(item) or page_source_size,
                source_name=source_name,
            )
            if block:
                blocks_by_page.setdefault(page_num, []).append(block)
    return blocks_by_page


def _merge_block_sources(
    middle_blocks: dict[int, list[dict[str, Any]]],
    content_blocks: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Union MinerU's structural sources without treating either as complete.

    ``middle.json`` usually has the strongest page geometry while
    ``content_list.json`` is often more complete for textual paragraphs.  A
    page-level fallback loses valid content whenever the preferred source is
    merely partial, so match individual blocks and retain every unmatched one.
    """
    merged: dict[int, list[dict[str, Any]]] = {}
    for page_num in sorted(set(middle_blocks) | set(content_blocks)):
        page_blocks: list[dict[str, Any]] = []
        for candidate in [*(middle_blocks.get(page_num) or []), *(content_blocks.get(page_num) or [])]:
            match_index = next(
                (index for index, existing in enumerate(page_blocks) if _blocks_describe_same_region(existing, candidate)),
                None,
            )
            if match_index is None:
                page_blocks.append(dict(candidate))
            else:
                page_blocks[match_index] = _merge_matching_blocks(page_blocks[match_index], candidate)
        if page_blocks:
            page_content_blocks = content_blocks.get(page_num) or []
            merged[page_num] = (
                _restore_content_list_order(page_blocks, page_content_blocks)
                if page_content_blocks
                else _layout_fallback_reading_order(page_blocks)
            )
    return merged


def _restore_content_list_order(
    page_blocks: list[dict[str, Any]],
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use MinerU's flattened content list as the authoritative reading order.

    ``middle.json`` and ``content_list.json`` are merged per block because each
    can contain details missing from the other.  The former is enumerated by
    structural groups, however, while the latter is MinerU's reading-order
    linearization.  Match the merged blocks back to content_list in that order,
    then keep middle-only blocks afterwards in their original MinerU order.
    """
    if not content_blocks:
        return page_blocks

    result: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for content_block in content_blocks:
        match_index = next(
            (
                index
                for index, existing in enumerate(page_blocks)
                if index not in consumed and _blocks_describe_same_region(existing, content_block)
            ),
            None,
        )
        if match_index is None:
            continue
        result.append(page_blocks[match_index])
        consumed.add(match_index)

    result.extend(
        block
        for index, block in enumerate(page_blocks)
        if index not in consumed
    )
    return result


_RUN_IN_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)+)(?:\.)?\s+(?P<tail>.+)$"
)
_RUN_IN_BODY_START_RE = re.compile(
    r"\s+(?P<body>"
    r"(?:根据|我们|本文|该|实验|结果|通过|在|从|对于|此外|因此|同时|随后|然后|"
    r"The\b|We\b|This\b|Our\b|In\b|For\b|To\b|Results?\b|Experiments?\b)"
    r")",
    re.IGNORECASE,
)


def _split_run_in_numbered_heading_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover a numbered heading MinerU merged into its following paragraph.

    Some MinerU outputs flatten ``4.4 Heading body...`` into one ordinary text
    block. A conservative split only runs for nested numeric paths and a clear
    discourse transition, keeping the body as a separate canonical block.
    """
    expanded: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict) or str(block.get("type") or "") != "paragraph":
            expanded.append(block)
            continue
        text = " ".join(str(block.get("text") or "").split())
        match = _RUN_IN_NUMBERED_HEADING_RE.match(text)
        if not match or len(text) < 48:
            expanded.append(block)
            continue

        tail = str(match.group("tail") or "").strip()
        split_at: int | None = None
        # Prefer an explicit newline from a textual exporter when present.
        raw_text = str(block.get("text") or "")
        newline = raw_text.find("\n")
        if newline > 0:
            first_line = " ".join(raw_text[:newline].split())
            if first_line.startswith(match.group("number")) and 6 <= len(first_line) <= 160:
                split_at = len(first_line)

        if split_at is None:
            body_match = _RUN_IN_BODY_START_RE.search(tail)
            if body_match and body_match.start() >= 6:
                split_at = len(match.group("number")) + 1 + body_match.start()

        if split_at is None:
            sentence_match = re.search(r"(?<=[A-Za-z])\.\s+(?=[A-Z])", text)
            if sentence_match and 10 <= sentence_match.start() <= 160:
                split_at = sentence_match.start() + 1

        if split_at is None:
            chinese_sentence = re.search(
                r"[。！？](?=(?:根据|我们|本文|该|实验|结果|通过|在|从|对于|此外|因此|同时|随后|然后))",
                text,
            )
            if chinese_sentence and 10 <= chinese_sentence.start() <= 160:
                split_at = chinese_sentence.start() + 1

        if split_at is None:
            expanded.append(block)
            continue

        heading_text = text[:split_at].strip(" \t\n:：;；-—。！？")
        body_text = text[split_at:].strip(" \t\n:：;；-—")
        number_match = _RUN_IN_NUMBERED_HEADING_RE.match(heading_text)
        if (
            not number_match
            or len(heading_text) > 180
            or len(body_text) < 24
            or len(heading_text) <= len(number_match.group("number")) + 2
        ):
            expanded.append(block)
            continue

        heading = dict(block)
        heading["type"] = "heading"
        heading["text"] = heading_text
        heading["level"] = max(1, min(4, number_match.group("number").count(".") + 1))
        heading["heading_source"] = "mineru_run_in_numbered"
        heading["run_in_heading_recovered"] = True
        heading.pop("block_id", None)
        heading.pop("reading_order", None)

        body = dict(block)
        body["text"] = body_text
        body["recovered_heading_prefix"] = number_match.group("number")
        body.pop("block_id", None)
        body.pop("reading_order", None)
        expanded.extend((heading, body))
    return expanded


def _layout_fallback_reading_order(page_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover a conservative reading order when MinerU lacks content_list.

    The normal path must retain MinerU's own sequence.  This fallback is only
    for middle.json-only responses, where structural groups are not guaranteed
    to be emitted in reading order.  It follows the same core rule used by
    paper parsers such as RAGFlow: detect a clear two-column split and read
    each column top-to-bottom before advancing to the next column.
    """
    if len(page_blocks) < 4:
        return _geometric_reading_order(page_blocks)

    boxes: list[tuple[int, list[float]]] = []
    for index, block in enumerate(page_blocks):
        bbox = _normalize_bbox(block.get("bbox"))
        if bbox:
            boxes.append((index, bbox))
    if len(boxes) < 4:
        return _geometric_reading_order(page_blocks)

    page_left = min(bbox[0] for _, bbox in boxes)
    page_right = max(bbox[2] for _, bbox in boxes)
    page_span = page_right - page_left
    if page_span <= 0:
        return _geometric_reading_order(page_blocks)

    # A wide spanning block needs a full page-layout model to position safely;
    # do not force it into a column with a simplistic fallback.
    if any((bbox[2] - bbox[0]) >= page_span * 0.68 for _, bbox in boxes):
        return _geometric_reading_order(page_blocks)

    x_positions = sorted((bbox[0], index) for index, bbox in boxes)
    gaps = [
        (x_positions[idx + 1][0] - x_positions[idx][0], idx)
        for idx in range(len(x_positions) - 1)
    ]
    if not gaps:
        return _geometric_reading_order(page_blocks)
    largest_gap, split_index = max(gaps)
    if largest_gap < max(36.0, page_span * 0.12):
        return _geometric_reading_order(page_blocks)

    split_x = (x_positions[split_index][0] + x_positions[split_index + 1][0]) / 2
    left_indices = {index for index, bbox in boxes if (bbox[0] + bbox[2]) / 2 < split_x}
    right_indices = {index for index, _bbox in boxes if index not in left_indices}
    if len(left_indices) < 2 or len(right_indices) < 2:
        return _geometric_reading_order(page_blocks)

    def column_key(index: int) -> tuple[float, float, int]:
        bbox = _normalize_bbox(page_blocks[index].get("bbox")) or [0.0, 0.0, 0.0, 0.0]
        return (bbox[1], bbox[0], index)

    return [
        *(page_blocks[index] for index in sorted(left_indices, key=column_key)),
        *(page_blocks[index] for index in sorted(right_indices, key=column_key)),
    ]


def _geometric_reading_order(page_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last-resort order for responses that expose neither sequence nor columns."""
    return sorted(
        page_blocks,
        key=lambda block: (
            float((_normalize_bbox(block.get("bbox")) or [0, 0, 0, 0])[1]),
            float((_normalize_bbox(block.get("bbox")) or [0, 0, 0, 0])[0]),
        ),
    )


def _blocks_describe_same_region(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_type = str(left.get("type") or "")
    right_type = str(right.get("type") or "")
    if left_type != right_type and {left_type, right_type} != {"heading", "paragraph"}:
        return False
    overlap = _bbox_iou(left.get("bbox"), right.get("bbox"))
    left_text = _block_text_fingerprint(left.get("text"))
    right_text = _block_text_fingerprint(right.get("text"))
    if overlap <= 0 and (not _normalize_bbox(left.get("bbox")) or not _normalize_bbox(right.get("bbox"))):
        # A title without geometry is still structural evidence. When both
        # MinerU sources expose the exact same text on the same page, merge it
        # instead of retaining duplicate phantom headings.
        return bool(left_text and left_text == right_text and len(left_text) >= 8)
    if overlap >= 0.75:
        return True
    # Text identity alone is not enough because papers often repeat captions or
    # short labels. Require meaningful text plus a relaxed geometric overlap.
    return bool(left_text and left_text == right_text and len(left_text) >= 12 and overlap >= 0.3)


def _merge_matching_blocks(preferred: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(preferred)
    preferred_text = str(preferred.get("text") or "")
    incoming_text = str(incoming.get("text") or "")
    if len(incoming_text.strip()) > len(preferred_text.strip()):
        merged["text"] = incoming_text
        merged["mineru_type"] = incoming.get("mineru_type") or merged.get("mineru_type")
    if not merged.get("line_anchors") and incoming.get("line_anchors"):
        merged["line_anchors"] = incoming["line_anchors"]
    heading_blocks = [
        block for block in (preferred, incoming)
        if block.get("type") == "heading"
    ]
    if heading_blocks:
        structural_heading = next(
            (
                block for block in heading_blocks
                if block.get("heading_source") == "mineru_text_level"
            ),
            heading_blocks[0],
        )
        merged["type"] = "heading"
        for key in ("mineru_text_level", "heading_source", "level"):
            if structural_heading.get(key) is not None:
                merged[key] = structural_heading[key]
        if structural_heading.get("mineru_type"):
            merged["mineru_type"] = structural_heading["mineru_type"]
    sources = []
    for value in [*(preferred.get("mineru_sources") or []), *(incoming.get("mineru_sources") or [])]:
        name = str(value or "").strip()
        if name and name not in sources:
            sources.append(name)
    merged["mineru_sources"] = sources
    return merged


def _block_text_fingerprint(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _bbox_iou(left: Any, right: Any) -> float:
    left_box = _normalize_bbox(left)
    right_box = _normalize_bbox(right)
    if not left_box or not right_box:
        return 0.0
    x0 = max(left_box[0], right_box[0])
    y0 = max(left_box[1], right_box[1])
    x1 = min(left_box[2], right_box[2])
    y1 = min(left_box[3], right_box[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left_box[2] - left_box[0]) * max(0.0, left_box[3] - left_box[1])
    right_area = max(0.0, right_box[2] - right_box[0]) * max(0.0, right_box[3] - right_box[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _infer_page_source_size(
    items: list[dict[str, Any]],
    *,
    page_width: float,
    page_height: float,
) -> tuple[float, float] | None:
    max_x = 0.0
    max_y = 0.0
    for item in items:
        bbox = _item_bbox(item)
        if not bbox:
            continue
        max_x = max(max_x, abs(float(bbox[0])), abs(float(bbox[2])))
        max_y = max(max_y, abs(float(bbox[1])), abs(float(bbox[3])))

    if max_x <= 0 or max_y <= 0:
        return None

    # MinerU content_list often omits page_size while using a 0-1000 normalized
    # coordinate system. If any block exceeds the PDF point dimensions, treat the
    # page as normalized instead of scaling every block against its own bbox.
    if (max_x > page_width * 1.05 or max_y > page_height * 1.05) and max_x <= 1100 and max_y <= 1100:
        return 1000.0, 1000.0

    return None


def _convert_mineru_item(
    item: dict[str, Any],
    page_num: int,
    page_specs: dict[int, dict[str, float]],
    page_source_size: tuple[float, float] | None,
    *,
    source_name: str,
) -> dict[str, Any] | None:
    raw_type = _normalize_type(item.get("type") or item.get("block_type") or item.get("category") or item.get("category_name"))
    block_type = _map_mineru_type(raw_type, item)
    if not block_type:
        return None

    text = _extract_item_text(item, block_type)
    if not text and block_type not in {"figure", "table", "artifact"}:
        return None
    if block_type in {"figure", "table"} and not text:
        text = "Figure" if block_type == "figure" else "Table"

    bbox = _item_bbox(item)
    page_spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
    bbox_pts = _bbox_to_page_pts(
        bbox,
        page_width=float(page_spec.get("width") or 612.0),
        page_height=float(page_spec.get("height") or 792.0),
        source_size=page_source_size,
    )
    if not bbox_pts and block_type in {"figure", "table"}:
        return None

    block: dict[str, Any] = {
        "type": block_type,
        # ``text`` is the canonical source used by summaries and retrieval.
        # Do not truncate a long paragraph or code block before it reaches
        # downstream consumers; UI callers can use ``display_text`` instead.
        "text": text,
        "section_id": None,
        "source": MINERU_BLOCK_INDEX_SOURCE,
        "mineru_type": raw_type or block_type,
        "mineru_sources": [source_name],
    }
    if bbox_pts:
        block["bbox"] = bbox_pts
    else:
        block["geometry_missing"] = True
    if len(text) > 2400:
        block["display_text"] = _limit_text(text, 2400)
    line_anchors = _mineru_line_anchors(
        item,
        page_width=float(page_spec.get("width") or 612.0),
        page_height=float(page_spec.get("height") or 792.0),
        source_size=page_source_size,
    )
    if line_anchors:
        block["line_anchors"] = line_anchors
    if block_type == "heading":
        text_level = _valid_mineru_text_level(item)
        if raw_type == "text" and text_level is not None:
            block["mineru_text_level"] = text_level
            block["heading_source"] = "mineru_text_level"
            block["level"] = _infer_mineru_heading_level(text, text_level)
        else:
            block["heading_source"] = "mineru_type"
            block["level"] = _infer_heading_level(text)
    if block_type == "artifact":
        block["layout_excluded_from_outline"] = True
    return block


def _normalize_type(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _map_mineru_type(raw_type: str, item: dict[str, Any]) -> str:
    if raw_type in {"title", "heading", "section_header"}:
        return "heading"
    if raw_type == "text" and _valid_mineru_text_level(item) is not None:
        return "heading"
    if raw_type in {
        "text", "plain_text", "paragraph", "para", "body_text", "list", "list_item",
        "code", "code_body", "ref_text", "reference", "references", "footnote",
    }:
        return "paragraph"
    if raw_type in {"table", "table_body"}:
        return "table"
    if raw_type in {"table_caption", "image_caption", "chart_caption", "caption"}:
        return "caption"
    if raw_type in {"image", "figure", "chart", "image_body", "chart_body"}:
        return "figure"
    if raw_type in {"interline_equation", "equation", "formula", "inline_equation"}:
        return "formula"
    if raw_type in {
        "discarded", "discarded_block", "abandon", "header", "footer", "header_footer",
        "page_number", "page_footnote", "aside_text",
    }:
        return "artifact"
    if item.get("discarded") is True:
        return "artifact"
    if _looks_like_visual_item(item):
        return "figure"
    if _extract_item_text(item, "paragraph"):
        # Preserve forward-compatible textual schema additions. The raw type is
        # retained on the block so diagnostics can flag the schema drift.
        return "paragraph"
    return ""


def _valid_mineru_text_level(item: dict[str, Any]) -> int | None:
    value = item.get("text_level")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if isinstance(value, str) and not re.fullmatch(r"[1-6]", value.strip()):
        return None
    try:
        level = int(value)
    except (TypeError, ValueError):
        return None
    return level if 1 <= level <= 6 else None


def _looks_like_visual_item(item: dict[str, Any]) -> bool:
    return any(
        isinstance(item.get(key), str) and str(item.get(key)).strip()
        for key in (
            "img_path", "image_path", "image_url", "chart_path", "figure_path", "asset_path",
        )
    )


def _extract_item_text(item: dict[str, Any], block_type: str) -> str:
    values: list[str] = []
    for key in (
        "text",
        "content",
        "html",
        "latex",
        "table_body",
        "table_html",
        "img_caption",
        "image_caption",
        "table_caption",
        "caption",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, list):
            joined = _text_from_list(value)
            if joined:
                values.append(joined)

    if not values:
        values.extend(_texts_from_nested(item))

    text = "\n".join(_dedupe(values))
    if block_type == "table" and text:
        return text
    return _clean_text(text)


def _text_from_list(items: list[Any]) -> str:
    parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.extend(_texts_from_nested(item))
    return "\n".join(part for part in parts if part.strip()).strip()


def _texts_from_nested(item: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in ("lines", "spans", "content", "blocks"):
        value = item.get(key)
        if not isinstance(value, list):
            continue
        for child in value:
            if isinstance(child, str) and child.strip():
                parts.append(child.strip())
            elif isinstance(child, dict):
                text = child.get("text") or child.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                parts.extend(_texts_from_nested(child))
    return parts


def _clean_text(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = re.sub(r"\s+", " ", value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _item_bbox(item: dict[str, Any]) -> list[float] | None:
    for key in (
        "bbox",
        "layout_bbox",
        "block_bbox",
        "poly",
        "img_body_bbox",
        "table_body_bbox",
        "span_bbox",
    ):
        bbox = _normalize_bbox(item.get(key))
        if bbox:
            return bbox
    return None


def _bbox_to_page_pts(
    bbox: list[float] | None,
    *,
    page_width: float,
    page_height: float,
    source_size: tuple[float, float] | None,
) -> list[float] | None:
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    max_x = max(abs(x0), abs(x1))
    max_y = max(abs(y0), abs(y1))

    source_w, source_h = source_size or (0.0, 0.0)
    if source_w > 0 and source_h > 0:
        scaled = [
            x0 / source_w * page_width,
            y0 / source_h * page_height,
            x1 / source_w * page_width,
            y1 / source_h * page_height,
        ]
        return _clip_bbox(scaled, page_width, page_height)

    if max_x <= page_width * 1.05 and max_y <= page_height * 1.05:
        return _clip_bbox([x0, y0, x1, y1], page_width, page_height)

    if max_x <= 1100 and max_y <= 1100:
        source_w = 1000.0
        source_h = 1000.0
    else:
        source_w = max_x
        source_h = max_y
    if source_w <= 0 or source_h <= 0:
        return None
    scaled = [
        x0 / source_w * page_width,
        y0 / source_h * page_height,
        x1 / source_w * page_width,
        y1 / source_h * page_height,
    ]
    return _clip_bbox(scaled, page_width, page_height)


def _clip_bbox(bbox: list[float], page_width: float, page_height: float) -> list[float] | None:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(page_width, float(x0)))
    x1 = max(0.0, min(page_width, float(x1)))
    y0 = max(0.0, min(page_height, float(y0)))
    y1 = max(0.0, min(page_height, float(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]


def _mineru_line_text(line: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("text", "content", "latex"):
        value = line.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for span in line.get("spans", []) or []:
        if not isinstance(span, dict):
            continue
        for key in ("text", "content", "latex"):
            value = span.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
                break
    return _clean_text(" ".join(_dedupe(parts)))


def _mineru_line_bbox(line: dict[str, Any]) -> list[float] | None:
    direct = _item_bbox(line)
    if direct:
        return direct
    boxes = [
        _item_bbox(span)
        for span in (line.get("spans", []) or [])
        if isinstance(span, dict)
    ]
    boxes = [box for box in boxes if box]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _iter_mineru_lines(item: dict[str, Any]):
    seen: set[int] = set()

    def visit(node):
        if not isinstance(node, dict):
            return
        lines = node.get("lines")
        if isinstance(lines, list):
            for line in lines:
                if not isinstance(line, dict) or id(line) in seen:
                    continue
                seen.add(id(line))
                yield line
        for key in ("blocks", "content"):
            children = node.get(key)
            if not isinstance(children, list):
                continue
            for child in children:
                if isinstance(child, dict):
                    yield from visit(child)

    yield from visit(item)


def _mineru_line_anchors(
    item: dict[str, Any],
    *,
    page_width: float,
    page_height: float,
    source_size: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for line in _iter_mineru_lines(item):
        text = _mineru_line_text(line)
        bbox = _bbox_to_page_pts(
            _mineru_line_bbox(line),
            page_width=page_width,
            page_height=page_height,
            source_size=source_size,
        )
        if text and bbox:
            anchors.append({"text": _limit_text(text, 800), "bbox": bbox})
    return anchors


def _page_num(item: dict[str, Any]) -> int:
    # MinerU 约定 page_idx 全程 0-based，需无条件 +1；仅当缺失 page_idx 时
    # 才回退到其他候选键（假定为 1-based），避免整份文档页码错位一页。
    if "page_idx" in item:
        try:
            return max(1, int(item["page_idx"]) + 1)
        except (TypeError, ValueError):
            return 1
    value = item.get("page", item.get("page_id", 1))
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _page_source_size(item: dict[str, Any]) -> tuple[float, float] | None:
    for keys in (
        ("width", "height"),
        ("page_width", "page_height"),
        ("img_width", "img_height"),
        ("image_width", "image_height"),
    ):
        if item.get(keys[0]) and item.get(keys[1]):
            try:
                return float(item[keys[0]]), float(item[keys[1]])
            except (TypeError, ValueError):
                pass
    size = item.get("page_size") or item.get("image_size")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            return float(size[0]), float(size[1])
        except (TypeError, ValueError):
            return None
    return None


def _infer_heading_level(text: str) -> int:
    stripped = " ".join(str(text or "").split())
    match = re.match(r"^(\d+(?:\.\d+)*)(?:\.)?\s+", stripped)
    if match:
        return max(1, min(match.group(1).count(".") + 1, 4))
    if re.match(r"^[A-Z]\.\s+", stripped):
        return 2
    return 1


def _infer_mineru_heading_level(text: str, fallback_level: int) -> int:
    """Use explicit title syntax when MinerU's text_level is coarse."""
    stripped = " ".join(str(text or "").split())
    numbered = re.match(r"^(\d+(?:\.\d+)*)(?:\.)?\s+", stripped)
    if numbered:
        return max(1, min(numbered.group(1).count(".") + 1, 4))
    appendix = re.match(
        r"^(?:appendix|appendices|supplementary\s+material)(?:\s+([A-Z])(?:\.(\d+(?:\.\d+)*))?)?(?:[.)]?\s+|$)",
        stripped,
        re.IGNORECASE,
    )
    if appendix:
        appendix_path = appendix.group(2)
        return 1 + len(appendix_path.split(".")) if appendix_path else 1
    if re.match(r"^[IVXLCM]+\.\s+", stripped):
        return 1
    bare_appendix = re.match(r"^[A-Z](?:\.(\d+(?:\.\d+)*))?(?:[.)]?\s+)", stripped)
    if bare_appendix and bare_appendix.group(1):
        appendix_path = bare_appendix.group(1)
        return 1 + len(appendix_path.split(".")) if appendix_path else 1
    if re.match(r"^[A-Z]\.\s+", stripped):
        return 2
    if re.match(
        r"^(?:abstract|introduction|background|related\s+work|method(?:s|ology)?|"
        r"approach|experiments?|evaluation|results?|discussion|conclusion|limitations?|"
        r"implementation|future\s+work|references|acknowledg(?:e)?ments?|"
        r"appendix|supplementary\s+material)\s*$",
        stripped,
        re.IGNORECASE,
    ):
        return 1
    try:
        return max(1, min(int(fallback_level), 4))
    except (TypeError, ValueError):
        return 1


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]
