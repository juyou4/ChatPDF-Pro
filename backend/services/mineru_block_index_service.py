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
    normalize_code_text,
    normalize_formula_markdown,
    normalize_table_text,
)

logger = logging.getLogger(__name__)

MINERU_BLOCK_INDEX_SOURCE = "mineru_vlm"
MINERU_RAW_VERSION = 1
MINERU_STRUCTURE_VERSION = 13
_PDF_CLIP_MIN_CHARS = 8
_VISUAL_CAPTION_LINE_RE = re.compile(
    r"^\s*(?:(?:Fig(?:ure)?|Table|图|表)\s*\.?\s*[A-Za-z0-9IVXLC]+)",
    re.IGNORECASE,
)
_VISUAL_CAPTION_LABEL_RE = re.compile(
    r"^\s*(?P<kind>Fig(?:ure)?|Table|图|表)\s*\.?\s*(?P<label>[A-Za-z0-9IVXLC]+)",
    re.IGNORECASE,
)
_PANEL_CAPTION_RE = re.compile(r"^\s*[\(\[（]\s*[a-zA-Z0-9]\s*[\)\]）]")
_CAPTION_CHILD_TYPES = {
    "image_caption",
    "table_caption",
    "chart_caption",
    "code_caption",
    "caption",
}
_FOOTNOTE_CHILD_TYPES = {
    "image_footnote",
    "table_footnote",
    "chart_footnote",
    "figure_footnote",
}
_BODY_CHILD_TYPES = {
    "image_body",
    "table_body",
    "chart_body",
    "code_body",
}
_TRUSTED_CAPTION_GEOMETRY = {"mineru_layout", "mineru_middle"}
_CAPTION_TEXT_KEYS = (
    "image_caption",
    "img_caption",
    "table_caption",
    "chart_caption",
    "caption",
    "code_caption",
)
_FOOTNOTE_TEXT_KEYS = (
    "image_footnote",
    "table_footnote",
    "chart_footnote",
)
_V2_LIFTED_KEYS = (
    "image_caption",
    "table_caption",
    "chart_caption",
    "code_caption",
    "code_body",
    "code_content",
    "list_items",
    "html",
    "table_body",
    "image_footnote",
    "table_footnote",
    "chart_footnote",
)
_MINERU_PAYLOAD_TEXT_KEYS = (
    "text",
    "content",
    "html",
    "latex",
    "list_items",
    "item_content",
    "code_body",
    "code_content",
    "code_caption",
    "algorithm_content",
    "algorithm_caption",
    "paragraph_content",
    "title_content",
)


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


class MinerUResultUnreadable(RuntimeError):
    """Raised when a stored MinerU payload exists but cannot be read.

    Distinct from ``load_mineru_result`` returning ``None``, which only means
    "no payload for this identity".
    """


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
        # ``None`` above means "no artifact for this identity" — a normal,
        # expected answer. A file that exists but cannot be read is a different
        # event: returning ``None`` for it made a corrupted MinerU payload
        # indistinguishable from "never parsed", so a full-MinerU document could
        # quietly rebuild from something else instead of refusing to publish.
        logger.error("[MinerUBlockIndex] failed to load %s: %s", path, exc)
        raise MinerUResultUnreadable(f"MinerU 原始解析产物不可读: {path}") from exc


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
    layout_json = payload.get("layout_json")
    content_list_json = payload.get("content_list_json")
    content_list_v2_json = payload.get("content_list_v2_json")
    if isinstance(content_list_json, list):
        content_list_json = _flatten_page_nested_content_items(content_list_json)
    if isinstance(content_list_v2_json, list):
        content_list_v2_json = _flatten_page_nested_content_items(content_list_v2_json)
    if isinstance(content_list_json, list) and isinstance(content_list_v2_json, list):
        content_list_json = _enrich_content_list_with_v2(content_list_json, content_list_v2_json)
    elif not isinstance(content_list_json, list) and isinstance(content_list_v2_json, list):
        content_list_json = [
            _flatten_content_list_v2_item(item)
            for item in content_list_v2_json
        ]
    layout_regions = _layout_regions_from_payload(payload)
    geometry_tree, geometry_tree_name = _geometry_tree_from_payload(payload)
    layout_caption_groups = _visual_caption_groups_from_payload(payload)
    middle_blocks: dict[int, list[dict[str, Any]]] = {}
    content_blocks: dict[int, list[dict[str, Any]]] = {}
    if geometry_tree is not None:
        middle_blocks = _blocks_from_middle_json(
            geometry_tree,
            page_specs,
            source_name=geometry_tree_name,
            visual_only=geometry_tree_name == "layout_json",
        )
    if isinstance(content_list_json, list):
        content_blocks = _blocks_from_content_list(
            content_list_json,
            page_specs,
            source_name="content_list_json",
            page_source_sizes=payload_page_source_sizes,
            layout_regions=layout_regions,
            layout_caption_groups=layout_caption_groups,
        )
    elif isinstance(middle_json, list):
        content_blocks = _blocks_from_content_list(
            middle_json,
            page_specs,
            source_name="middle_json_list",
            page_source_sizes=payload_page_source_sizes,
            layout_regions=layout_regions,
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

    _backfill_missing_text_from_pdf(pages, pdf_path)
    _mark_duplicate_backfilled_text(pages)
    _relocate_visual_captions(pages, pdf_path)
    _link_visual_caption_siblings(pages)

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
        and not block.get("text_duplicate")
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
            "has_layout_json": isinstance(layout_json, dict) and isinstance(layout_json.get("pdf_info"), list),
            "geometry_tree": geometry_tree_name,
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
    layout = payload.get("layout_json")
    if isinstance(layout, dict):
        pdf_info = layout.get("pdf_info")
        if isinstance(pdf_info, list):
            pages.update(_page_num(item) for item in pdf_info if isinstance(item, dict))
    return {page for page in pages if page > 0}


def _geometry_tree_from_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Prefer middle.json child boxes; fall back to layout.json when middle is absent."""
    if not isinstance(payload, dict):
        return None, ""
    middle = payload.get("middle_json")
    if isinstance(middle, dict):
        pdf_info = middle.get("pdf_info")
        if isinstance(pdf_info, list) and pdf_info:
            return middle, "middle_json"
    layout = payload.get("layout_json")
    if isinstance(layout, dict):
        pdf_info = layout.get("pdf_info")
        if isinstance(pdf_info, list) and pdf_info:
            return layout, "layout_json"
    return None, ""


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

    # 标题/大纲对齐只作诊断，**不**拒绝发布。页覆盖和正文完整时，下游按 MinerU
    # 版面块 ingest；``structure_degraded`` 交给大纲恢复器补导航锚点。
    # ``_build_outline`` 会给刻意丢弃的标题打 ``structure_exclusion_reason``，
    # 所以"既没进 outline 又没有排除理由"才算静默损失。
    # MinerU 未标任何标题时比较是 ``0 < 0``，outline 可能塌成「全文」，同样只记诊断。
    outline_block_ids = {
        str(item.get("first_block") or "")
        for item in outline
        if isinstance(item, dict) and str(item.get("first_block") or "")
    }
    silently_dropped_headings = [
        block
        for block in emitted_headings
        if not str(block.get("structure_exclusion_reason") or "").strip()
        and str(block.get("block_id") or "") not in outline_block_ids
    ]
    outline_is_fallback_only = (
        len(outline) == 1
        and str((outline[0] or {}).get("source") or "").strip().lower() == "fallback"
        if outline and isinstance(outline[0], dict)
        else False
    )
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
        "emitted_heading_block_count": len(emitted_headings),
        "silently_dropped_heading_count": len(silently_dropped_headings),
        "outline_is_fallback_only": outline_is_fallback_only,
        "flat_structure_without_headings": not emitted_headings,
        "structure_degraded": (
            covered_text_level_heading_count < raw_heading_count
            or len(raw_native_headings & emitted_native_heading_keys) < raw_native_heading_count
            or bool(missing_run_in_paths)
            or bool(silently_dropped_headings)
        ),
    }


def _page_structure_blocks(page_info: dict[str, Any], *, prefer_para: bool) -> list[dict[str, Any]]:
    para = page_info.get("para_blocks")
    if prefer_para and isinstance(para, list) and para:
        return [item for item in para if isinstance(item, dict)]
    raw_blocks: list[dict[str, Any]] = []
    keys = ("preproc_blocks", "layout_blocks", "blocks") if prefer_para else (
        "preproc_blocks",
        "para_blocks",
        "layout_blocks",
        "blocks",
    )
    for key in keys:
        value = page_info.get(key)
        if isinstance(value, list):
            raw_blocks.extend(item for item in value if isinstance(item, dict))
    return raw_blocks


def _blocks_from_middle_json(
    data: dict[str, Any],
    page_specs: dict[int, dict[str, float]],
    *,
    source_name: str,
    visual_only: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    blocks_by_page: dict[int, list[dict[str, Any]]] = {}
    pdf_info = data.get("pdf_info")
    if not isinstance(pdf_info, list):
        return blocks_by_page
    geometry_source = "mineru_layout" if source_name == "layout_json" else "mineru_middle"

    for page_info in pdf_info:
        if not isinstance(page_info, dict):
            continue
        page_num = _page_num(page_info)
        page_source_size = _page_source_size(page_info)
        raw_blocks = _page_structure_blocks(page_info, prefer_para=visual_only)
        if not visual_only:
            discarded = page_info.get("discarded_blocks")
            if isinstance(discarded, list):
                for item in discarded:
                    if isinstance(item, dict):
                        clone = dict(item)
                        clone.setdefault("type", "discarded")
                        raw_blocks.append(clone)

        for raw in raw_blocks:
            parent_type = _normalize_type(raw.get("type") or raw.get("block_type"))
            if visual_only and parent_type not in {"image", "figure", "chart", "table"}:
                continue
            for expanded in _expand_grouped_mineru_block(raw):
                block = _convert_mineru_item(
                    expanded,
                    page_num,
                    page_specs,
                    page_source_size,
                    source_name=source_name,
                )
                if block:
                    mineru_type = _normalize_type(block.get("mineru_type"))
                    if block.get("type") == "caption" and mineru_type in (
                        _CAPTION_CHILD_TYPES | _FOOTNOTE_CHILD_TYPES
                    ):
                        block.setdefault("caption_geometry_source", geometry_source)
                    blocks_by_page.setdefault(page_num, []).append(block)
    return blocks_by_page


def _blocks_from_content_list(
    items: list[Any],
    page_specs: dict[int, dict[str, float]],
    *,
    source_name: str,
    page_source_sizes: dict[int, tuple[float, float]] | None = None,
    layout_regions: list[dict[str, Any]] | None = None,
    layout_caption_groups: list[dict[str, Any]] | None = None,
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
        item_declared_size = next(
            (_page_source_size(item) for item in page_items if _page_source_size(item)),
            None,
        )
        inferred_page_source_size = _infer_page_source_size(
            page_items,
            page_width=float(page_spec.get("width") or 612.0),
            page_height=float(page_spec.get("height") or 792.0),
        )
        page_source_size = (
            item_declared_size
            or (page_source_sizes or {}).get(page_num)
            or inferred_page_source_size
        )
        # A content-list-only page may use PDF points or a 0-1000 coordinate
        # system.  When every coordinate happens to fit inside the PDF page,
        # magnitude alone cannot distinguish the two.  Do not silently turn
        # that ambiguity into a crop anchor; textual structure remains usable.
        page_geometry_uncertain = bool(
            page_source_size is None
            and any(_item_bbox(item) for item in page_items)
        )
        for item in page_items:
            flat_item = _flatten_content_list_v2_item(item) if isinstance(item.get("content"), dict) else item
            item_source_size = _page_source_size(flat_item) or page_source_size
            geometry_uncertain = bool(
                page_geometry_uncertain and _page_source_size(flat_item) is None
            )
            block = _convert_mineru_item(
                flat_item,
                page_num,
                page_specs,
                item_source_size,
                source_name=source_name,
                geometry_uncertain=geometry_uncertain,
            )
            if block:
                blocks_by_page.setdefault(page_num, []).append(block)
            caption_siblings = _caption_siblings_from_visual_item(
                flat_item,
                page_num=page_num,
                page_items=page_items,
                page_specs=page_specs,
                page_source_size=item_source_size,
                source_name=source_name,
                geometry_uncertain=geometry_uncertain,
                layout_regions=layout_regions or [],
                layout_caption_groups=layout_caption_groups or [],
            )
            occupied_caption_boxes = []
            for caption_block, caption_source_bbox in caption_siblings:
                blocks_by_page.setdefault(page_num, []).append(caption_block)
                if caption_source_bbox:
                    occupied_caption_boxes.append(caption_source_bbox)
            footnote_block = _footnote_sibling_from_visual_item(
                flat_item,
                page_num=page_num,
                page_items=page_items,
                page_specs=page_specs,
                page_source_size=item_source_size,
                source_name=source_name,
                geometry_uncertain=geometry_uncertain,
                layout_regions=layout_regions or [],
                occupied_boxes=occupied_caption_boxes,
                layout_caption_groups=layout_caption_groups or [],
            )
            if footnote_block:
                blocks_by_page.setdefault(page_num, []).append(footnote_block)
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
# 这条规则找的是"正文句子的开头"。英文句首必然大写，所以**不能**加 IGNORECASE：
# 加了之后 ``The\b`` 会匹配正文里任意一个小写 the，于是以小数/版本号/金额开头的
# 句子会被当成 run-in 标题切成"假标题 + 残句"——
# ``3.14 is approximately the ratio of ...`` 会在大纲里变出一条 ``3.14 is approximately``。
# 中文分支本来就没有大小写，不受影响。
_RUN_IN_BODY_START_RE = re.compile(
    r"\s+(?P<body>"
    r"(?:根据|我们|本文|该|实验|结果|通过|在|从|对于|此外|因此|同时|随后|然后|"
    r"The\b|We\b|This\b|Our\b|In\b|For\b|To\b|Results?\b|Experiments?\b)"
    r")"
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
    for middle.json-only responses.  OpenDataLoader-style XY-Cut: peel
    spanning figures/tables first, then read each remaining band as
    left-then-right columns.  Do not dump a whole page to Y→X just because
    one block is wide.
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

    spanning = [
        (index, bbox)
        for index, bbox in boxes
        if (bbox[2] - bbox[0]) >= page_span * 0.68
    ]
    if spanning:
        return _banded_column_reading_order(page_blocks, boxes, page_span, spanning)
    return _order_column_band(page_blocks, [index for index, _ in boxes], page_span)


def _order_column_band(
    page_blocks: list[dict[str, Any]],
    indices: list[int],
    page_span: float,
) -> list[dict[str, Any]]:
    """Read one vertical band: two columns when the gutter is clear, else Y→X."""
    boxes = [
        (index, bbox)
        for index in indices
        if (bbox := _normalize_bbox(page_blocks[index].get("bbox")))
    ]
    if len(boxes) < 4:
        return _geometric_reading_order([page_blocks[index] for index in indices])

    x_positions = sorted((bbox[0], index) for index, bbox in boxes)
    gaps = [
        (x_positions[idx + 1][0] - x_positions[idx][0], idx)
        for idx in range(len(x_positions) - 1)
    ]
    if not gaps:
        return _geometric_reading_order([page_blocks[index] for index in indices])
    largest_gap, split_index = max(gaps)
    if largest_gap < max(36.0, page_span * 0.12):
        return _geometric_reading_order([page_blocks[index] for index in indices])

    split_x = (x_positions[split_index][0] + x_positions[split_index + 1][0]) / 2
    left_indices = {index for index, bbox in boxes if (bbox[0] + bbox[2]) / 2 < split_x}
    right_indices = {index for index, _bbox in boxes if index not in left_indices}
    if len(left_indices) < 2 or len(right_indices) < 2:
        return _geometric_reading_order([page_blocks[index] for index in indices])

    def column_key(index: int) -> tuple[float, float, int]:
        bbox = _normalize_bbox(page_blocks[index].get("bbox")) or [0.0, 0.0, 0.0, 0.0]
        return (bbox[1], bbox[0], index)

    return [
        *(page_blocks[index] for index in sorted(left_indices, key=column_key)),
        *(page_blocks[index] for index in sorted(right_indices, key=column_key)),
    ]


def _banded_column_reading_order(
    page_blocks: list[dict[str, Any]],
    boxes: list[tuple[int, list[float]]],
    page_span: float,
    spanning: list[tuple[int, list[float]]],
) -> list[dict[str, Any]]:
    """Emit spanning blocks in Y order; columnize the bands between them."""
    box_map = {index: bbox for index, bbox in boxes}
    remaining = {index for index, _ in boxes}
    result: list[dict[str, Any]] = []
    cursor_y = -1e9
    for span_index, span_bbox in sorted(spanning, key=lambda item: (item[1][1], item[1][0], item[0])):
        band = [
            index
            for index in remaining
            if index != span_index and box_map[index][1] < span_bbox[1] - 2 and box_map[index][1] >= cursor_y
        ]
        if band:
            result.extend(_order_column_band(page_blocks, band, page_span))
            remaining.difference_update(band)
        if span_index in remaining:
            result.append(page_blocks[span_index])
            remaining.discard(span_index)
        cursor_y = span_bbox[3]
    if remaining:
        result.extend(_order_column_band(page_blocks, list(remaining), page_span))
    boxed = {index for index, _ in boxes}
    result.extend(
        page_blocks[index]
        for index in range(len(page_blocks))
        if index not in boxed and page_blocks[index] not in result
    )
    return result


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
    preferred_geom = str(preferred.get("caption_geometry_source") or "")
    incoming_geom = str(incoming.get("caption_geometry_source") or "")
    if preferred_geom in _TRUSTED_CAPTION_GEOMETRY:
        merged["caption_geometry_source"] = preferred_geom
        if preferred.get("bbox"):
            merged["bbox"] = preferred["bbox"]
    elif incoming_geom in _TRUSTED_CAPTION_GEOMETRY:
        merged["caption_geometry_source"] = incoming_geom
        if incoming.get("bbox"):
            merged["bbox"] = incoming["bbox"]
    elif incoming_geom and not preferred_geom:
        merged["caption_geometry_source"] = incoming_geom
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
    geometry_uncertain: bool = False,
) -> dict[str, Any] | None:
    raw_type = _normalize_type(item.get("type") or item.get("block_type") or item.get("category") or item.get("category_name"))
    block_type = _map_mineru_type(raw_type, item)
    # MinerU 很少漏块。适配器只做字段映射；没有对上类型名时，有框或有载荷就留下。
    if not block_type:
        if _looks_like_code_item(item):
            block_type = "code"
        elif _item_bbox(item) or _harvest_mineru_text(item):
            block_type = "paragraph"
        else:
            return None

    text = _extract_item_text(item, block_type)
    bbox = _item_bbox(item)
    text_missing = False
    if not text and block_type not in {"figure", "table", "artifact"}:
        # MinerU content_list keeps empty text boxes after column/page wraps.
        # Dropping them erases the only geometry for those regions.
        if not bbox:
            return None
        text_missing = True
    if block_type in {"figure", "table"} and not text:
        text = "Figure" if block_type == "figure" else "Table"

    # 目录页条目（"引言 ······ 3"）在检索里是纯噪声：它和真正讲引言的正文
    # 共享全部关键词，却不含任何内容。归为 artifact 而不是丢弃，block_id 与
    # 页面定位都保留，阅读面板仍能显示。
    toc_entry = block_type == "paragraph" and _looks_like_toc_entry(text)
    if toc_entry:
        block_type = "artifact"

    page_spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
    geometry_uncertain = bool(geometry_uncertain and bbox)
    bbox_pts = None
    if not geometry_uncertain:
        bbox_pts = _bbox_to_page_pts(
            bbox,
            page_width=float(page_spec.get("width") or 612.0),
            page_height=float(page_spec.get("height") or 792.0),
            source_size=page_source_size,
        )
    if not bbox_pts and block_type in {"figure", "table"}:
        if not geometry_uncertain:
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
        if geometry_uncertain:
            block["geometry_uncertain"] = True
            block["geometry_reason"] = "content_list_coordinate_space_ambiguous"
    if text_missing:
        block["text_missing"] = True
    if len(text) > 2400:
        block["display_text"] = _limit_text(text, 2400)
    line_anchors = []
    if not geometry_uncertain:
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
        if toc_entry:
            block["structure_exclusion_reason"] = "toc_dot_leader"
    return block


# 目录条目的点引导符：标题与页码之间那串填充点。要求至少 4 个填充符并以页码
# 收尾，避免把正文里偶然出现的省略号加数字（"……见第 3 节"）误判成目录。
_TOC_DOT_LEADER_RE = re.compile(r"[.．·・∙•…]{4,}\s*\d{1,4}\s*$")


def _looks_like_toc_entry(text: str) -> bool:
    """判断一段文本是否为目录条目。

    目录条目与真正讲该章节的正文共享全部关键词却不含内容，是检索里的纯噪声。
    单行必须命中；多行要求过半命中，这样一整块目录会被识别，而夹带一行类似
    格式的正文段落不会。
    """
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return False
    matched = sum(1 for line in lines if _TOC_DOT_LEADER_RE.search(line))
    if len(lines) == 1:
        return matched == 1
    return matched * 2 >= len(lines)


def _normalize_type(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _map_mineru_type(raw_type: str, item: dict[str, Any]) -> str:
    if raw_type in {"title", "heading", "section_header"}:
        return "heading"
    if raw_type == "text" and _valid_mineru_text_level(item) is not None:
        return "heading"
    if raw_type in {
        "text", "plain_text", "paragraph", "para", "body_text", "list", "list_item",
        "ref_text", "reference", "references", "footnote",
    }:
        return "paragraph"
    if raw_type in {"code", "code_body", "algorithm"} or _looks_like_code_item(item):
        return "code"
    if raw_type in {"table", "table_body"}:
        return "table"
    if raw_type in {
        "table_caption", "image_caption", "chart_caption", "caption",
        # 图/表脚注（"* p<0.05"、数据来源、样本量说明）属于视觉块附属说明，
        # 按 caption 归到 ROLE_CAPTION，不要当成正文段落或写进 figure/table 本体。
        "table_footnote",
        "image_footnote",
        "chart_footnote",
        "figure_footnote",
    }:
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
    if _harvest_mineru_text(item) or _item_bbox(item):
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


def _looks_like_code_item(item: dict[str, Any]) -> bool:
    if _normalize_type(item.get("sub_type")) == "algorithm":
        return True
    sources = [item]
    nested = item.get("content")
    if isinstance(nested, dict):
        sources.append(nested)
    for source in sources:
        if _normalize_type(source.get("sub_type")) == "algorithm":
            return True
        for key in ("code_body", "code_content", "algorithm_content"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, list) and any(str(part or "").strip() for part in value):
                return True
    return False


def _collect_payload_text_values(value: Any, *, depth: int = 0) -> list[str]:
    if depth > 6:
        return []
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        parts: list[str] = []
        for child in value:
            parts.extend(_collect_payload_text_values(child, depth=depth + 1))
        return parts
    if isinstance(value, dict):
        parts = []
        for key in _MINERU_PAYLOAD_TEXT_KEYS:
            if key in value:
                parts.extend(_collect_payload_text_values(value.get(key), depth=depth + 1))
        parts.extend(_texts_from_nested(value))
        return parts
    return []


def _harvest_mineru_text(item: dict[str, Any]) -> str:
    """Read MinerU payload fields instead of requiring a ``text`` key."""
    return _clean_text("\n".join(_dedupe(_collect_payload_text_values(item))))


def _extract_item_text(item: dict[str, Any], block_type: str) -> str:
    caption_keys = _CAPTION_TEXT_KEYS + _FOOTNOTE_TEXT_KEYS
    if block_type == "figure":
        keys = ("text", "content")
    elif block_type == "table":
        keys = ("text", "content", "html", "latex", "table_body", "table_html")
    elif block_type == "caption":
        keys = caption_keys + ("text", "content")
    elif block_type == "code":
        keys = ("code_body", "code_content", "algorithm_content", "text", "content")
    else:
        keys = (
            "text",
            "content",
            "list_items",
            "item_content",
            "html",
            "latex",
            "table_body",
            "table_html",
            *caption_keys,
        )
    values: list[str] = []
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, list):
            joined = _text_from_list(value)
            if joined:
                values.append(joined)
        elif isinstance(value, dict):
            values.extend(_collect_payload_text_values(value))

    if not values and block_type not in {"figure", "table"}:
        values.extend(_collect_payload_text_values(item))

    text = "\n".join(_dedupe(values))
    if block_type == "table" and text:
        # 表格块以前直接返回原始 HTML，于是 ``<table><tr><td>`` 会一路进到
        # 大纲与速览的 prompt，再被 800/900 字的截断从标签中间切开，摘要里就
        # 出现 ``<td`` 和半截标签。渲染成 markdown 与 RAG 侧口径一致；
        # 解析不出来时退回按普通文本清洗，而不是把标签原样放出去。
        # caption 由 sibling 承担，这里不再把图注写进表体。
        return normalize_table_text(text, caption="")
    if block_type == "formula" and text:
        return normalize_formula_markdown(text)
    if block_type == "code" and text:
        return normalize_code_text(text)
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
    poly = _poly_to_bbox(item.get("poly"))
    if poly:
        return poly
    for key in (
        "bbox",
        "layout_bbox",
        "block_bbox",
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


def _poly_to_bbox(poly: Any) -> list[float] | None:
    if isinstance(poly, (list, tuple)) and len(poly) >= 8:
        try:
            xs = [float(poly[index]) for index in range(0, 8, 2)]
            ys = [float(poly[index]) for index in range(1, 8, 2)]
        except (TypeError, ValueError):
            return None
        return _normalize_bbox([min(xs), min(ys), max(xs), max(ys)])
    return _normalize_bbox(poly)


def _expand_grouped_mineru_block(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Split middle.json image/table groups so captions keep their own bbox."""
    children = raw.get("blocks")
    if not isinstance(children, list):
        return [raw]
    captions: list[dict[str, Any]] = []
    footnotes: list[dict[str, Any]] = []
    bodies: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        kind = _normalize_type(child.get("type") or child.get("block_type"))
        if kind in _CAPTION_CHILD_TYPES:
            captions.append(child)
        elif kind in _FOOTNOTE_CHILD_TYPES:
            footnotes.append(child)
        elif kind in _BODY_CHILD_TYPES:
            bodies.append(child)
        else:
            leftovers.append(child)
    if not captions and not bodies and not footnotes:
        return [raw]

    parent_type = _normalize_type(raw.get("type") or raw.get("block_type"))
    visual_type = "image" if parent_type in {"image", "figure", "chart"} else parent_type or "image"
    caption_text = _visual_caption_text(raw)
    if not caption_text:
        caption_text = "\n".join(
            part for child in captions if (part := _extract_item_text(child, "caption"))
        ).strip()
    footnote_text = _visual_footnote_text(raw)
    if not footnote_text:
        footnote_text = "\n".join(
            part for child in footnotes if (part := _extract_item_text(child, "caption"))
        ).strip()

    expanded: list[dict[str, Any]] = []
    sources = bodies or [raw]
    for body in sources:
        clone = dict(raw)
        clone["type"] = visual_type
        clone["bbox"] = body.get("bbox") or raw.get("bbox")
        clone["blocks"] = leftovers
        for key in _CAPTION_TEXT_KEYS + _FOOTNOTE_TEXT_KEYS:
            clone.pop(key, None)
        expanded.append(clone)
    for caption in captions:
        child = dict(caption)
        if len(captions) == 1 and caption_text and not _extract_item_text(child, "caption"):
            child["text"] = caption_text
        expanded.append(child)
    for footnote in footnotes:
        child = dict(footnote)
        if len(footnotes) == 1 and footnote_text and not _extract_item_text(child, "caption"):
            child["text"] = footnote_text
        expanded.append(child)
    return expanded


def _flatten_page_nested_content_items(raw: list[Any]) -> list[dict[str, Any]]:
    """Flatten official MinerU ``List[List[item]]`` page arrays into one item list."""
    if not isinstance(raw, list):
        return []
    if raw and all(isinstance(item, dict) for item in raw):
        return [item for item in raw if isinstance(item, dict)]
    items: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, dict):
            items.append(entry)
            continue
        if not isinstance(entry, list):
            continue
        for item in entry:
            if not isinstance(item, dict):
                continue
            clone = dict(item)
            if clone.get("page_idx") is None and clone.get("page_no") is None and clone.get("page") is None:
                clone["page_idx"] = index
            items.append(clone)
    return items


def _flatten_content_list_v2_item(item: dict[str, Any]) -> dict[str, Any]:
    """Lift CONTENT_LIST_V2 ``content`` fields into the v1 item shape."""
    clone = dict(item)
    content = item.get("content")
    if not isinstance(content, dict):
        return clone
    if content.get("paragraph_content") and not clone.get("text"):
        clone["text"] = content.get("paragraph_content")
    if content.get("title_content") and not clone.get("text"):
        clone["text"] = content.get("title_content")
        clone.setdefault("type", "title")
    for key in _V2_LIFTED_KEYS:
        if key in content and key not in clone:
            clone[key] = content[key]
    image_source = content.get("image_source")
    if isinstance(image_source, dict) and image_source.get("path"):
        clone.setdefault("img_path", image_source.get("path"))
    return clone


_V2_TYPE_UPGRADES = {
    "code",
    "code_body",
    "algorithm",
    "list",
    "list_item",
    "title",
    "heading",
    "section_header",
}


def _copy_v2_fields_onto_v1(v1_item: dict[str, Any], v2_item: dict[str, Any]) -> None:
    for key in _V2_LIFTED_KEYS:
        if v2_item.get(key) and not v1_item.get(key):
            v1_item[key] = v2_item[key]
    if v2_item.get("text") and not str(v1_item.get("text") or "").strip():
        v1_item["text"] = v2_item["text"]
    v1_type = _normalize_type(v1_item.get("type"))
    v2_type = _normalize_type(v2_item.get("type"))
    if v1_type in {"text", "plain_text", "paragraph", "para"} and v2_type in _V2_TYPE_UPGRADES:
        v1_item["type"] = v2_item.get("type") or v2_type


def _enrich_content_list_with_v2(
    v1_items: list[Any],
    v2_items: list[Any],
) -> list[dict[str, Any]]:
    """Keep v1 reading order; copy v2 types/captions onto overlapping boxes."""
    flat_v1 = _flatten_page_nested_content_items(v1_items)
    flat_v2 = [
        _flatten_content_list_v2_item(item)
        for item in _flatten_page_nested_content_items(v2_items)
    ]
    if not flat_v2:
        return list(flat_v1)
    enriched = [dict(item) for item in flat_v1 if isinstance(item, dict)]
    used: set[int] = set()
    for item in enriched:
        best: tuple[float, int, dict[str, Any]] | None = None
        for index, other in enumerate(flat_v2):
            if index in used or other.get("page_idx") != item.get("page_idx"):
                continue
            overlap = _bbox_iou(item.get("bbox"), other.get("bbox"))
            if overlap < 0.45:
                continue
            if best is None or overlap > best[0]:
                best = (overlap, index, other)
        if best is None:
            continue
        _, index, other = best
        used.add(index)
        _copy_v2_fields_onto_v1(item, other)
    return enriched


def _visual_field_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    values: list[str] = []
    sources = [item]
    nested = item.get("content")
    if isinstance(nested, dict):
        sources.append(nested)
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
            elif isinstance(value, list):
                joined = _text_from_list(value)
                if joined:
                    values.append(joined)
    return "\n".join(_dedupe(values)).strip()


def _visual_caption_text(item: dict[str, Any]) -> str:
    return _visual_field_text(item, _CAPTION_TEXT_KEYS)


def _visual_footnote_text(item: dict[str, Any]) -> str:
    return _visual_field_text(item, _FOOTNOTE_TEXT_KEYS)


def _iter_caption_strings(value: Any) -> list[str]:
    parts: list[str] = []
    if isinstance(value, str) and value.strip():
        parts.extend(_split_joined_caption_text(value.strip()))
    elif isinstance(value, list):
        for entry in value:
            if isinstance(entry, str) and entry.strip():
                parts.extend(_split_joined_caption_text(entry.strip()))
            elif isinstance(entry, dict):
                text = str(entry.get("text") or entry.get("content") or "").strip()
                if not text:
                    text = _text_from_list([entry])
                if text:
                    parts.extend(_split_joined_caption_text(text))
    return parts


def _split_joined_caption_text(text: str) -> list[str]:
    """Keep panel labels and Figure/Table lines as separate caption siblings."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) >= 2:
        parts: list[str] = []
        buffer: list[str] = []
        for line in lines:
            if _VISUAL_CAPTION_LINE_RE.search(line) and buffer:
                parts.append(" ".join(buffer))
                buffer = [line]
            else:
                buffer.append(line)
        if buffer:
            parts.append(" ".join(buffer))
        return [part for part in parts if part]
    match = re.search(
        r"(?=\b(?:Fig(?:ure)?|Table|图|表)\s*\.?\s*[A-Za-z0-9IVXLC]+)",
        text,
        re.IGNORECASE,
    )
    if match and match.start() >= 8 and _PANEL_CAPTION_RE.search(text):
        head = text[:match.start()].strip(" ,;，；")
        tail = text[match.start():].strip()
        if head and tail:
            return [head, tail]
    return [text]


def _visual_caption_parts(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    sources = [item]
    nested = item.get("content")
    if isinstance(nested, dict):
        sources.append(nested)
    for source in sources:
        for key in _CAPTION_TEXT_KEYS:
            values.extend(_iter_caption_strings(source.get(key)))
    return _dedupe(values)


def _caption_part_kind(text: str) -> str:
    if _VISUAL_CAPTION_LINE_RE.search(text):
        return "figure"
    if _PANEL_CAPTION_RE.search(text):
        return "panel"
    return "other"


def _visual_caption_label(text: str) -> tuple[str, str] | None:
    match = _VISUAL_CAPTION_LABEL_RE.match(str(text or ""))
    if not match:
        return None
    kind = str(match.group("kind") or "").lower()
    if kind.startswith("fig") or kind == "图":
        kind = "figure"
    elif kind.startswith("table") or kind == "表":
        kind = "table"
    return kind, str(match.group("label") or "").lower()


def _caption_text_matches_pdf_candidate(caption_text: str, pdf_text: str) -> bool:
    caption = " ".join(str(caption_text or "").split())
    pdf = " ".join(str(pdf_text or "").split())
    if not caption or not pdf:
        return False
    caption_label = _visual_caption_label(caption)
    pdf_label = _visual_caption_label(pdf)
    if caption_label and pdf_label:
        return caption_label == pdf_label
    if _PANEL_CAPTION_RE.search(caption) and pdf_label:
        return False
    return _text_mostly_covered(pdf, caption) or _text_mostly_covered(caption, pdf)


def _visual_caption_groups_from_tree(
    data: dict[str, Any],
    *,
    geometry_source: str,
) -> list[dict[str, Any]]:
    """Read image/table caption and footnote child boxes from a MinerU geometry tree."""
    if not isinstance(data, dict):
        return []
    groups: list[dict[str, Any]] = []
    for page_info in data.get("pdf_info") or []:
        if not isinstance(page_info, dict):
            continue
        page_num = _page_num(page_info)
        for raw in _page_structure_blocks(page_info, prefer_para=True):
            kind = _normalize_type(raw.get("type") or raw.get("block_type"))
            if kind not in {"image", "figure", "chart", "table"}:
                continue
            children = raw.get("blocks")
            if not isinstance(children, list):
                continue
            body_box = _normalize_bbox(raw.get("bbox"))
            caption_boxes: list[list[float]] = []
            footnote_boxes: list[list[float]] = []
            for child in children:
                if not isinstance(child, dict):
                    continue
                child_kind = _normalize_type(child.get("type") or child.get("block_type"))
                child_box = _normalize_bbox(child.get("bbox"))
                if child_kind in _BODY_CHILD_TYPES and child_box:
                    body_box = child_box
                elif child_kind in _CAPTION_CHILD_TYPES and child_box:
                    caption_boxes.append(child_box)
                elif child_kind in _FOOTNOTE_CHILD_TYPES and child_box:
                    footnote_boxes.append(child_box)
            if body_box and (caption_boxes or footnote_boxes):
                groups.append({
                    "page": page_num,
                    "body_bbox": body_box,
                    "caption_bboxes": caption_boxes,
                    "footnote_bboxes": footnote_boxes,
                    "geometry_source": geometry_source,
                    "page_source_size": _page_source_size(page_info),
                })
    return groups


def _visual_caption_groups_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tree, name = _geometry_tree_from_payload(payload)
    if tree is None:
        return []
    geometry_source = "mineru_middle" if name == "middle_json" else "mineru_layout"
    return _visual_caption_groups_from_tree(tree, geometry_source=geometry_source)


def _layout_visual_caption_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Compatibility wrapper: harvest caption children from the trusted geometry tree."""
    return _visual_caption_groups_from_payload(payload)


def _group_box_page_pts(
    box: Any,
    group: dict[str, Any],
    *,
    page_num: int,
    page_specs: dict[int, dict[str, float]],
) -> list[float] | None:
    spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
    return _bbox_to_page_pts(
        _normalize_bbox(box),
        page_width=float(spec.get("width") or 612.0),
        page_height=float(spec.get("height") or 792.0),
        source_size=group.get("page_source_size"),
    )


def _match_visual_caption_group(
    body_pts: list[float] | None,
    *,
    page_num: int,
    groups: list[dict[str, Any]] | None,
    page_specs: dict[int, dict[str, float]],
) -> dict[str, Any] | None:
    if not body_pts:
        return None
    for group in groups or []:
        if int(group.get("page") or 0) != int(page_num):
            continue
        group_body = _group_box_page_pts(
            group.get("body_bbox"),
            group,
            page_num=page_num,
            page_specs=page_specs,
        )
        if not group_body or _bbox_iou(body_pts, group_body) < 0.45:
            continue
        return group
    return None


def _item_bbox_page_pts(
    item: dict[str, Any],
    *,
    page_num: int,
    page_specs: dict[int, dict[str, float]],
    page_source_size: tuple[float, float] | None,
) -> list[float] | None:
    spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
    return _bbox_to_page_pts(
        _item_bbox(item),
        page_width=float(spec.get("width") or 612.0),
        page_height=float(spec.get("height") or 792.0),
        source_size=page_source_size,
    )


def _assign_layout_caption_boxes(
    parts: list[str],
    caption_boxes: list[list[float]],
    body_box: list[float] | None,
) -> list[list[float] | None]:
    remaining = [box for box in caption_boxes if _normalize_bbox(box)]
    assigned: list[list[float] | None] = [None] * len(parts)
    for index, part in enumerate(parts):
        if _caption_part_kind(part) != "figure" or not remaining:
            continue
        assigned[index] = max(remaining, key=lambda box: (box[2] - box[0], box[1]))
        remaining = [box for box in remaining if box is not assigned[index]]
    body = _normalize_bbox(body_box)
    for index, part in enumerate(parts):
        if assigned[index] is not None or not remaining:
            continue
        if body:
            assigned[index] = max(
                remaining,
                key=lambda box: max(0.0, min(body[2], box[2]) - max(body[0], box[0])),
            )
        else:
            assigned[index] = remaining[0]
        remaining = [box for box in remaining if box is not assigned[index]]
    return assigned


def _emit_caption_sibling_block(
    *,
    caption_type: str,
    text: str,
    caption_bbox: list[float],
    item: dict[str, Any],
    page_num: int,
    page_specs: dict[int, dict[str, float]],
    convert_source_size: tuple[float, float] | None,
    source_name: str,
    geometry_uncertain: bool,
    geometry_source: str,
    pin_bbox: bool = False,
) -> dict[str, Any] | None:
    caption_item = {
        "type": caption_type,
        "text": text,
        "bbox": caption_bbox,
        "page_idx": item.get("page_idx", page_num - 1),
    }
    block = _convert_mineru_item(
        caption_item,
        page_num,
        page_specs,
        convert_source_size,
        source_name=source_name,
        geometry_uncertain=geometry_uncertain and geometry_source not in _TRUSTED_CAPTION_GEOMETRY,
    )
    if not block:
        return None
    if pin_bbox:
        block["bbox"] = [round(float(value), 3) for value in caption_bbox]
    block["caption_geometry_source"] = geometry_source
    return block


def _last_caption_sibling_for_part(
    siblings: list[tuple[dict[str, Any], list[float] | None]],
    part: str,
) -> dict[str, Any] | None:
    kind = _caption_part_kind(part)
    for block, _bbox in reversed(siblings):
        if _caption_part_kind(str(block.get("text") or "")) == kind:
            return block
    return siblings[-1][0] if siblings else None


def _caption_siblings_from_visual_item(
    item: dict[str, Any],
    *,
    page_num: int,
    page_items: list[dict[str, Any]],
    page_specs: dict[int, dict[str, float]],
    page_source_size: tuple[float, float] | None,
    source_name: str,
    geometry_uncertain: bool,
    layout_regions: list[dict[str, Any]],
    layout_caption_groups: list[dict[str, Any]] | None = None,
) -> list[tuple[dict[str, Any], list[float] | None]]:
    raw_type = _normalize_type(item.get("type") or item.get("block_type"))
    if raw_type not in {"image", "figure", "chart", "table"} and not _looks_like_visual_item(item):
        return []
    parts = _visual_caption_parts(item)
    body_pts = _item_bbox_page_pts(
        item,
        page_num=page_num,
        page_specs=page_specs,
        page_source_size=page_source_size,
    )
    group = _match_visual_caption_group(
        body_pts,
        page_num=page_num,
        groups=layout_caption_groups,
        page_specs=page_specs,
    )
    layout_boxes = [
        box
        for box in (
            _group_box_page_pts(
                raw_box,
                group,
                page_num=page_num,
                page_specs=page_specs,
            )
            for raw_box in ((group or {}).get("caption_bboxes") or [])
        )
        if box
    ] if group else []
    caption_type = (
        "image_caption"
        if raw_type in {"image", "figure", "chart"} or _looks_like_visual_item(item)
        else "table_caption"
    )
    spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
    page_size = (
        float(spec.get("width") or 612.0),
        float(spec.get("height") or 792.0),
    )
    trusted_source = str((group or {}).get("geometry_source") or "mineru_layout")
    siblings: list[tuple[dict[str, Any], list[float] | None]] = []
    if layout_boxes:
        assigned_layout = _assign_layout_caption_boxes(parts, layout_boxes, body_pts)
        used_boxes: list[list[float]] = []
        for part, layout_box in zip(parts, assigned_layout):
            if not layout_box:
                continue
            block = _emit_caption_sibling_block(
                caption_type=caption_type,
                text=part,
                caption_bbox=layout_box,
                item=item,
                page_num=page_num,
                page_specs=page_specs,
                convert_source_size=page_size,
                source_name=source_name,
                geometry_uncertain=False,
                geometry_source=trusted_source,
                pin_bbox=True,
            )
            if not block:
                continue
            siblings.append((block, layout_box))
            used_boxes.append(layout_box)
        for part, layout_box in zip(parts, assigned_layout):
            if layout_box:
                continue
            target = _last_caption_sibling_for_part(siblings, part)
            if target is None:
                continue
            existing = str(target.get("text") or "").strip()
            target["text"] = f"{existing}\n{part}".strip() if existing else part
        for layout_box in layout_boxes:
            if any(layout_box is used for used in used_boxes):
                continue
            block = _emit_caption_sibling_block(
                caption_type=caption_type,
                text="",
                caption_bbox=layout_box,
                item=item,
                page_num=page_num,
                page_specs=page_specs,
                convert_source_size=page_size,
                source_name=source_name,
                geometry_uncertain=False,
                geometry_source=trusted_source,
                pin_bbox=True,
            )
            if block:
                siblings.append((block, layout_box))
        return siblings
    if not parts:
        return []
    occupied: list[list[float]] = []
    for part in parts:
        caption_bbox, geometry_source = _infer_caption_bbox(
            _item_bbox(item),
            page_num=page_num,
            page_items=page_items,
            layout_regions=layout_regions,
            prefer="above" if raw_type == "table" else "below",
            exclude_boxes=occupied,
        )
        if not caption_bbox:
            continue
        block = _emit_caption_sibling_block(
            caption_type=caption_type,
            text=part,
            caption_bbox=caption_bbox,
            item=item,
            page_num=page_num,
            page_specs=page_specs,
            convert_source_size=page_source_size,
            source_name=source_name,
            geometry_uncertain=geometry_uncertain,
            geometry_source=geometry_source,
        )
        if not block:
            continue
        siblings.append((block, caption_bbox))
        occupied.append(caption_bbox)
    return siblings


def _footnote_sibling_from_visual_item(
    item: dict[str, Any],
    *,
    page_num: int,
    page_items: list[dict[str, Any]],
    page_specs: dict[int, dict[str, float]],
    page_source_size: tuple[float, float] | None,
    source_name: str,
    geometry_uncertain: bool,
    layout_regions: list[dict[str, Any]],
    occupied_boxes: list[Any] | None = None,
    layout_caption_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    raw_type = _normalize_type(item.get("type") or item.get("block_type"))
    if raw_type not in {"image", "figure", "chart", "table"} and not _looks_like_visual_item(item):
        return None
    footnote_text = _visual_footnote_text(item)
    if not footnote_text:
        return None
    body_pts = _item_bbox_page_pts(
        item,
        page_num=page_num,
        page_specs=page_specs,
        page_source_size=page_source_size,
    )
    group = _match_visual_caption_group(
        body_pts,
        page_num=page_num,
        groups=layout_caption_groups,
        page_specs=page_specs,
    )
    footnote_boxes = [
        box
        for box in (
            _group_box_page_pts(
                raw_box,
                group,
                page_num=page_num,
                page_specs=page_specs,
            )
            for raw_box in ((group or {}).get("footnote_bboxes") or [])
        )
        if box
    ] if group else []
    spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
    page_size = (
        float(spec.get("width") or 612.0),
        float(spec.get("height") or 792.0),
    )
    if footnote_boxes:
        footnote_bbox = footnote_boxes[0]
        geometry_source = str((group or {}).get("geometry_source") or "mineru_layout")
        convert_source_size = page_size
        pin_bbox = True
        uncertain = False
    else:
        footnote_bbox, geometry_source = _infer_caption_bbox(
            _item_bbox(item),
            page_num=page_num,
            page_items=page_items,
            layout_regions=layout_regions,
            prefer="below",
            region_type="footnote",
            exclude_boxes=occupied_boxes or [],
        )
        convert_source_size = page_source_size
        pin_bbox = False
        uncertain = geometry_uncertain
    if not footnote_bbox:
        return None
    footnote_type = (
        "image_footnote"
        if raw_type in {"image", "figure", "chart"} or _looks_like_visual_item(item)
        else "table_footnote"
    )
    return _emit_caption_sibling_block(
        caption_type=footnote_type,
        text=footnote_text,
        caption_bbox=footnote_bbox,
        item=item,
        page_num=page_num,
        page_specs=page_specs,
        convert_source_size=convert_source_size,
        source_name=source_name,
        geometry_uncertain=uncertain,
        geometry_source=geometry_source,
        pin_bbox=pin_bbox,
    )


def _infer_caption_bbox(
    figure_bbox: list[float] | None,
    *,
    page_num: int,
    page_items: list[dict[str, Any]],
    layout_regions: list[dict[str, Any]],
    prefer: str = "below",
    region_type: str = "caption",
    exclude_boxes: list[Any] | None = None,
) -> tuple[list[float] | None, str]:
    """Place a caption/footnote sibling. Tables prefer the gap above; figures prefer below."""
    fig = _normalize_bbox(figure_bbox)
    if not fig:
        return None, ""
    excluded = [_normalize_bbox(box) for box in (exclude_boxes or [])]
    excluded = [box for box in excluded if box]

    region_boxes = []
    for region in layout_regions:
        if int(region.get("page") or 0) != int(page_num):
            continue
        if str(region.get("type") or "") != region_type:
            continue
        box = _normalize_bbox(region.get("bbox"))
        if not box:
            continue
        if any(_bbox_iou(box, other) >= 0.45 for other in excluded):
            continue
        region_boxes.append({"bbox": box})
    picked = _pick_adjacent_caption_box(fig, prefer, region_boxes)
    if picked:
        return picked["bbox"], "mineru_layout"

    above_bottom: float | None = None
    below_top: float | None = None
    for item in page_items:
        box = _normalize_bbox(_item_bbox(item))
        if not box or _bbox_iou(box, fig) >= 0.75:
            continue
        if box[3] <= fig[1] + 2:
            above_bottom = box[3] if above_bottom is None else max(above_bottom, box[3])
        if box[1] >= fig[3] - 2:
            below_top = box[1] if below_top is None else min(below_top, box[1])

    above_gap = [fig[0], above_bottom, fig[2], fig[1]] if above_bottom is not None and fig[1] > above_bottom + 2 else None
    below_gap = [fig[0], fig[3], fig[2], below_top] if below_top is not None and below_top > fig[3] + 2 else None
    above_gap = _shrink_gap_away_from_boxes(above_gap, excluded, "above")
    below_gap = _shrink_gap_away_from_boxes(below_gap, excluded, "below")
    if prefer == "above" and above_gap:
        return above_gap, "gap"
    if prefer == "below" and below_gap:
        return below_gap, "gap"
    if above_gap:
        return above_gap, "gap"
    if below_gap:
        return below_gap, "gap"

    strip = max(12.0, (fig[3] - fig[1]) * 0.08)
    if prefer == "above":
        top = fig[1]
        for box in excluded:
            if box[1] < fig[1] + 2:
                top = min(top, box[1])
        return [fig[0], max(0.0, top - strip), fig[2], top], "gap"
    bottom = fig[3]
    for box in excluded:
        if box[3] > fig[3] - 2:
            bottom = max(bottom, box[3])
    return [fig[0], bottom, fig[2], bottom + strip], "gap"


def _shrink_gap_away_from_boxes(
    gap: list[float] | None,
    exclude_boxes: list[list[float]],
    side: str,
) -> list[float] | None:
    box = _normalize_bbox(gap)
    if not box:
        return None
    for other in exclude_boxes:
        x_overlap = min(box[2], other[2]) - max(box[0], other[0])
        y_overlap = min(box[3], other[3]) - max(box[1], other[1])
        if x_overlap <= 0 or y_overlap <= 0:
            continue
        if side == "below" and other[3] > box[1]:
            box = [box[0], max(box[1], other[3]), box[2], box[3]]
        if side == "above" and other[1] < box[3]:
            box = [box[0], box[1], box[2], min(box[3], other[1])]
        if box[2] <= box[0] or box[3] <= box[1] + 2:
            return None
    return box


def _caption_blocked_by_body(
    body: list[float],
    caption: list[float],
    blockers: list[list[float]],
) -> bool:
    for other in blockers:
        x_overlap = min(body[2], caption[2], other[2]) - max(body[0], caption[0], other[0])
        if x_overlap <= 0:
            continue
        if caption[1] >= body[3] - 2 and other[1] >= body[3] - 8 and other[3] <= caption[1] + 8:
            return True
        if caption[3] <= body[1] + 2 and other[3] <= body[1] + 8 and other[1] >= caption[3] - 8:
            return True
    return False


def _pick_adjacent_caption_box(
    body_bbox: list[float],
    prefer: str,
    captions: list[dict[str, Any]],
    *,
    max_above: float = 160.0,
    max_below: float = 180.0,
    blocker_boxes: list[list[float]] | None = None,
) -> dict[str, Any] | None:
    body = _normalize_bbox(body_bbox)
    if not body:
        return None
    body_width = max(body[2] - body[0], 1.0)
    blockers = [box for box in (blocker_boxes or []) if _normalize_bbox(box)]
    best: tuple[float, dict[str, Any]] | None = None
    for caption in captions:
        cap_bbox = _normalize_bbox(caption.get("bbox"))
        if not cap_bbox:
            continue
        if _caption_blocked_by_body(body, cap_bbox, blockers):
            continue
        overlap = max(0.0, min(body[2], cap_bbox[2]) - max(body[0], cap_bbox[0]))
        overlap_ratio = overlap / max(min(body_width, cap_bbox[2] - cap_bbox[0]), 1.0)
        if overlap_ratio < 0.08:
            continue
        above_gap = body[1] - cap_bbox[3]
        below_gap = cap_bbox[1] - body[3]
        sides: list[tuple[float, str]] = []
        if -12.0 <= above_gap <= max_above:
            sides.append((max(0.0, above_gap), "above"))
        if -12.0 <= below_gap <= max_below:
            sides.append((max(0.0, below_gap), "below"))
        if not sides:
            continue
        gap, position = min(sides, key=lambda item: item[0])
        orientation_penalty = 0.0
        if prefer == "above" and position != "above":
            orientation_penalty = 30.0
        if prefer == "below" and position != "below":
            orientation_penalty = 20.0
        kind = str(caption.get("kind") or "")
        kind_penalty = 0.0 if not kind or kind == prefer or (
            kind == "table" and prefer == "above"
        ) or (
            kind == "figure" and prefer == "below"
        ) else 80.0
        score = gap + kind_penalty + orientation_penalty - overlap_ratio * 20.0
        if best is None or score < best[0]:
            best = (score, caption)
    return best[1] if best else None


def _layout_regions_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for key in ("model_json", "layout_json"):
        regions.extend(_parse_mineru_layout_regions(payload.get(key)))
    return regions


def _parse_mineru_layout_regions(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list) and data and isinstance(data[0], dict) and "layout_dets" in data[0]:
        regions: list[dict[str, Any]] = []
        for page in data:
            if not isinstance(page, dict):
                continue
            info = page.get("page_info") if isinstance(page.get("page_info"), dict) else {}
            try:
                page_idx = int(info.get("page_no", 0))
            except (TypeError, ValueError):
                page_idx = 0
            for item in page.get("layout_dets") or []:
                if not isinstance(item, dict):
                    continue
                kind = {
                    4: "caption",
                    6: "caption",
                    3: "figure",
                    5: "table",
                    7: "footnote",
                    101: "footnote",
                }.get(item.get("category_id"))
                bbox = _poly_to_bbox(item.get("poly")) or _item_bbox(item)
                if not kind or not bbox:
                    continue
                regions.append({"page": page_idx + 1, "type": kind, "bbox": bbox})
        return regions
    if isinstance(data, list):
        regions = []
        for item in data:
            if not isinstance(item, dict):
                continue
            kind = {
                "figure_caption": "caption",
                "image_caption": "caption",
                "table_caption": "caption",
                "caption": "caption",
                "figure": "figure",
                "image": "figure",
                "table": "table",
                "table_footnote": "footnote",
                "image_footnote": "footnote",
                "figure_footnote": "footnote",
                "chart_footnote": "footnote",
            }.get(_normalize_type(item.get("type") or item.get("category") or item.get("category_name")))
            bbox = _poly_to_bbox(item.get("poly")) or _item_bbox(item)
            if not kind or not bbox:
                continue
            regions.append({"page": _page_num(item), "type": kind, "bbox": bbox})
        return regions
    return []


def _clean_pdf_clip_text(text: str, *, keep_newlines: bool = False) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"(\w)-\n(\w)", r"\1\2", value)
    if keep_newlines:
        value = re.sub(r"[ \t]+", " ", value)
        return re.sub(r"\n{3,}", "\n\n", value).strip()
    return re.sub(r"\s+", " ", value).strip()


def _extract_pdf_clip_text(page: Any, bbox: list[float], *, keep_newlines: bool = False) -> str:
    try:
        import fitz
    except Exception:
        return ""
    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    try:
        raw = page.get_textbox(rect) or page.get_text("text", clip=rect, sort=True)
    except Exception:
        return ""
    return _clean_pdf_clip_text(str(raw or ""), keep_newlines=keep_newlines)


def _backfill_missing_text_from_pdf(pages: list[dict[str, Any]], pdf_path: Path | str | None) -> None:
    pdf_file = Path(pdf_path) if pdf_path else None
    if not pdf_file or not pdf_file.exists():
        return
    try:
        import fitz

        pdf_doc = fitz.open(str(pdf_file))
    except Exception as exc:
        logger.warning("[MinerUBlockIndex] failed to open PDF for clip backfill: %s", exc)
        return

    filled_count = 0
    try:
        for page in pages:
            page_num = int(page.get("page") or 0)
            if page_num < 1 or page_num > pdf_doc.page_count:
                continue
            for block in page.get("blocks") or []:
                if not isinstance(block, dict) or not block.get("text_missing"):
                    continue
                bbox = _normalize_bbox(block.get("bbox"))
                if not bbox:
                    continue
                keep_newlines = str(block.get("type") or "") == "code" or str(
                    block.get("mineru_type") or ""
                ) in {"code", "code_body", "algorithm"}
                filled = _extract_pdf_clip_text(
                    pdf_doc[page_num - 1],
                    bbox,
                    keep_newlines=keep_newlines,
                )
                if len(filled) < _PDF_CLIP_MIN_CHARS:
                    continue
                block["text"] = filled
                block.pop("text_missing", None)
                block["text_source"] = "pdf_clip"
                if len(filled) > 2400:
                    block["display_text"] = _limit_text(filled, 2400)
                filled_count += 1
    finally:
        pdf_doc.close()

    if filled_count:
        logger.info("[MinerUBlockIndex] backfilled %s empty text box(es) from PDF clips", filled_count)


def _hyphen_join_fingerprint(value: Any) -> str:
    fingerprint = _block_text_fingerprint(value)
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", fingerprint)


def _text_mostly_covered(inner: str, outer: str) -> bool:
    inner_fp = _hyphen_join_fingerprint(inner)
    outer_fp = _hyphen_join_fingerprint(outer)
    if len(inner_fp) < 8 or len(outer_fp) < 8:
        return False
    if inner_fp in outer_fp:
        return True
    # Clip often starts mid-word after MinerU kept the hyphen prefix elsewhere.
    for cut in range(1, 16):
        trimmed = inner_fp[cut:].lstrip(" ,.;:")
        if len(trimmed) >= 24 and trimmed in outer_fp:
            return True
    trimmed = re.sub(r"^[a-z]{1,16}[.,;:]?\s+", "", inner_fp)
    return len(trimmed) >= 24 and trimmed in outer_fp


def _mark_duplicate_backfilled_text(pages: list[dict[str, Any]]) -> None:
    """Keep clip geometry when MinerU already stitched the same sentence elsewhere."""
    page_bodies: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        page_num = int(page.get("page") or 0)
        page_bodies[page_num] = [
            block
            for block in (page.get("blocks") or [])
            if isinstance(block, dict)
            and block.get("type") in {"paragraph", "heading", "code"}
            and str(block.get("text") or "").strip()
        ]

    for page in pages:
        page_num = int(page.get("page") or 0)
        neighbors = [
            *(page_bodies.get(page_num) or []),
            *(page_bodies.get(page_num - 1) or []),
        ]
        for block in page.get("blocks") or []:
            if not isinstance(block, dict) or block.get("text_source") != "pdf_clip":
                continue
            filled = str(block.get("text") or "")
            source = next(
                (
                    other
                    for other in neighbors
                    if other is not block
                    and other.get("text_source") != "pdf_clip"
                    and _text_mostly_covered(filled, str(other.get("text") or ""))
                ),
                next(
                    (
                        other
                        for other in neighbors
                        if other is not block and _text_mostly_covered(filled, str(other.get("text") or ""))
                    ),
                    None,
                ),
            )
            if source is None:
                continue
            block["text_duplicate"] = True
            source_id = str(source.get("block_id") or "").strip()
            if source_id:
                block["highlight_of"] = source_id
                block["linked_content_id"] = source_id


def _pdf_caption_candidates(page: Any, page_num: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return candidates
    for block in text_dict.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        for line in block.get("lines") or []:
            if not isinstance(line, dict):
                continue
            text = "".join(
                str(span.get("text") or "")
                for span in (line.get("spans") or [])
                if isinstance(span, dict)
            )
            text = re.sub(r"\s+", " ", text).strip()
            if not text or not _VISUAL_CAPTION_LINE_RE.search(text):
                continue
            bbox = _normalize_bbox(line.get("bbox"))
            if not bbox:
                continue
            kind = "table" if re.match(r"^\s*(table|表)\b", text, re.IGNORECASE) else "figure"
            candidates.append({"bbox": bbox, "text": text, "kind": kind, "page": page_num})
    return candidates


def _iter_pdf_line_boxes(page: Any) -> list[list[float]]:
    lines: list[list[float]] = []
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return lines
    for block in text_dict.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        for line in block.get("lines") or []:
            if not isinstance(line, dict):
                continue
            line_box = _normalize_bbox(line.get("bbox"))
            if line_box:
                lines.append(line_box)
    return lines


def _line_has_column_sibling(line_box: list[float], lines: list[list[float]]) -> bool:
    for other in lines:
        if other is line_box:
            continue
        if abs(other[1] - line_box[1]) > 8:
            continue
        if other[2] < line_box[0] - 8 or other[0] > line_box[2] + 8:
            return True
    return False


def _expand_pdf_caption_bbox(
    page: Any,
    seed_bbox: list[float],
    body_bbox: list[float],
    *,
    other_bodies: list[list[float]] | None = None,
    stop_boxes: list[list[float]] | None = None,
) -> list[float]:
    """Grow a Figure/Table label to its wrapped lines, but not into two-column body."""
    seed = _normalize_bbox(seed_bbox) or seed_bbox
    body = _normalize_bbox(body_bbox) or body_bbox
    above = seed[3] <= body[1] + 8
    y0 = seed[1] if above else max(body[3], seed[1] - 4)
    y1 = body[1] if above else seed[3] + 72
    if not above:
        below_stops = [
            box[1]
            for box in (stop_boxes or [])
            if _normalize_bbox(box) and box[1] > seed[1] + 6
        ]
        if below_stops:
            y1 = min(y1, min(below_stops) - 1)

    lines = _iter_pdf_line_boxes(page)
    accepted = [seed]
    prev_bottom = seed[3]
    for line_box in sorted(lines, key=lambda box: (box[1], box[0])):
        if line_box[3] < y0 - 2 or line_box[1] > y1:
            continue
        if _bbox_iou(line_box, seed) >= 0.7:
            continue
        if _bbox_iou(line_box, body) > 0.15:
            continue
        if any(_bbox_iou(line_box, other) > 0.15 for other in (other_bodies or [])):
            continue
        overlap = min(seed[2], line_box[2]) - max(seed[0], line_box[0])
        if overlap < 8:
            continue
        if line_box[1] > prev_bottom + 14:
            break
        if _line_has_column_sibling(line_box, lines):
            break
        accepted.append(line_box)
        prev_bottom = max(prev_bottom, line_box[3])
    grown = [
        min(box[0] for box in accepted),
        min(box[1] for box in accepted),
        max(box[2] for box in accepted),
        max(box[3] for box in accepted),
    ]
    return _clip_caption_before_blocks(grown, stop_boxes or [])


def _clip_caption_before_blocks(box: list[float], stop_boxes: list[list[float]]) -> list[float]:
    """Keep a spanning caption from swallowing the two-column body under it."""
    clipped = list(box)
    cuts: list[float] = []
    for stop in stop_boxes:
        other = _normalize_bbox(stop)
        if not other:
            continue
        overlap = min(clipped[2], other[2]) - max(clipped[0], other[0])
        if overlap < 12:
            continue
        if other[1] > clipped[1] + 8:
            cuts.append(other[1] - 1)
    if cuts:
        clipped[3] = min(clipped[3], min(cuts))
    if clipped[3] <= clipped[1]:
        return [round(v, 3) for v in box]
    return [round(v, 3) for v in clipped]


def _is_visual_footnote_block(block: dict[str, Any]) -> bool:
    return _normalize_type(block.get("mineru_type")) in _FOOTNOTE_CHILD_TYPES


def _bind_caption_to_body(caption: dict[str, Any], body: dict[str, Any]) -> None:
    body_id = str(body.get("block_id") or "").strip()
    if not body_id:
        return
    caption["linked_content_id"] = caption.get("linked_content_id") or body_id
    caption.setdefault("linked_content_ids", [])
    if body_id not in caption["linked_content_ids"]:
        caption["linked_content_ids"].append(body_id)
    caption["figure_id"] = str(body.get("figure_id") or body_id)


def _link_visual_caption_siblings(pages: list[dict[str, Any]]) -> None:
    """Bind figure/table bodies to caption siblings without copying caption text."""
    for page in pages:
        blocks = [block for block in (page.get("blocks") or []) if isinstance(block, dict)]
        bodies = [
            block
            for block in blocks
            if block.get("type") in {"figure", "table"} and _normalize_bbox(block.get("bbox"))
        ]
        captions = [
            block
            for block in blocks
            if block.get("type") == "caption"
            and not block.get("visual_enhancement")
            and not _is_visual_footnote_block(block)
            and _normalize_bbox(block.get("bbox"))
        ]
        used: set[int] = set()
        for body in bodies:
            body_id = str(body.get("block_id") or "").strip()
            if not body_id:
                continue
            body["figure_id"] = str(body.get("figure_id") or body_id)
            prefer = "above" if body.get("type") == "table" else "below"
            unused = [
                {
                    "bbox": caption.get("bbox"),
                    "kind": "table" if str(caption.get("mineru_type") or "").startswith("table") else "figure",
                    "block": caption,
                }
                for caption in captions
                if id(caption) not in used
            ]
            blockers = [
                other.get("bbox")
                for other in bodies
                if other is not body and _normalize_bbox(other.get("bbox"))
            ]
            picked = _pick_adjacent_caption_box(
                body.get("bbox"),
                prefer,
                unused,
                blocker_boxes=blockers,
            )
            if not picked:
                continue
            caption = picked.get("block") if isinstance(picked, dict) else None
            if not isinstance(caption, dict):
                continue
            used.add(id(caption))
            _bind_caption_to_body(caption, body)
            body["caption_block_id"] = str(caption.get("block_id") or "")
        _link_stacked_visual_bodies(blocks)
        _link_row_visual_bodies(blocks)
        _link_visual_footnote_siblings(blocks)


def _link_visual_footnote_siblings(blocks: list[dict[str, Any]]) -> None:
    """Attach figure/table footnotes without replacing the real caption sibling."""
    bodies = [
        block
        for block in blocks
        if block.get("type") in {"figure", "table"} and _normalize_bbox(block.get("bbox"))
    ]
    footnotes = [
        block
        for block in blocks
        if block.get("type") == "caption"
        and _is_visual_footnote_block(block)
        and _normalize_bbox(block.get("bbox"))
    ]
    used: set[int] = set()
    for body in bodies:
        unused = [
            {"bbox": footnote.get("bbox"), "block": footnote}
            for footnote in footnotes
            if id(footnote) not in used
        ]
        picked = _pick_adjacent_caption_box(body.get("bbox"), "below", unused)
        if not picked:
            continue
        footnote = picked.get("block") if isinstance(picked, dict) else None
        if not isinstance(footnote, dict):
            continue
        used.add(id(footnote))
        _bind_caption_to_body(footnote, body)


def _link_stacked_visual_bodies(blocks: list[dict[str, Any]]) -> None:
    """Share one Figure N caption across stacked/side-by-side chart bodies."""
    bodies = [
        block
        for block in blocks
        if block.get("type") == "figure" and _normalize_bbox(block.get("bbox"))
    ]
    if len(bodies) < 2:
        return
    captions = {
        str(block.get("block_id") or ""): block
        for block in blocks
        if block.get("type") == "caption"
    }
    linked = [block for block in bodies if block.get("caption_block_id")]
    for body in bodies:
        if body.get("caption_block_id"):
            continue
        body_box = _normalize_bbox(body.get("bbox"))
        if not body_box:
            continue
        partner = None
        best_gap = 80.0
        for other in linked:
            other_box = _normalize_bbox(other.get("bbox"))
            if not other_box:
                continue
            overlap = max(0.0, min(body_box[2], other_box[2]) - max(body_box[0], other_box[0]))
            width = min(body_box[2] - body_box[0], other_box[2] - other_box[0])
            if width <= 0 or overlap / width < 0.45:
                continue
            gap = min(
                (
                    value
                    for value in (other_box[1] - body_box[3], body_box[1] - other_box[3])
                    if value >= 0
                ),
                default=None,
            )
            if gap is not None and gap < best_gap:
                best_gap = gap
                partner = other
        if partner is None:
            continue
        figure_id = str(partner.get("figure_id") or partner.get("block_id") or "")
        caption_id = str(partner.get("caption_block_id") or "")
        body["figure_id"] = figure_id
        body["caption_block_id"] = caption_id
        body_id = str(body.get("block_id") or "")
        caption = captions.get(caption_id)
        if caption and body_id:
            caption.setdefault("linked_content_ids", [])
            if body_id not in caption["linked_content_ids"]:
                caption["linked_content_ids"].append(body_id)


def _is_figure_level_caption(block: dict[str, Any]) -> bool:
    return bool(_VISUAL_CAPTION_LINE_RE.search(str(block.get("text") or "")))


def _horizontal_figure_row(
    body: dict[str, Any],
    bodies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    body_box = _normalize_bbox(body.get("bbox"))
    if not body_box:
        return [body]
    row = [body]
    seen = {id(body)}
    queue = [body]
    while queue:
        current = queue.pop()
        current_box = _normalize_bbox(current.get("bbox"))
        if not current_box:
            continue
        current_height = max(current_box[3] - current_box[1], 1.0)
        for other in bodies:
            if id(other) in seen:
                continue
            other_box = _normalize_bbox(other.get("bbox"))
            if not other_box:
                continue
            overlap = max(0.0, min(current_box[3], other_box[3]) - max(current_box[1], other_box[1]))
            height = min(current_height, other_box[3] - other_box[1])
            if height <= 0 or overlap / height < 0.45:
                continue
            gap = min(
                (
                    value
                    for value in (other_box[0] - current_box[2], current_box[0] - other_box[2])
                    if value >= -8.0
                ),
                default=None,
            )
            if gap is None or gap > 80.0:
                continue
            seen.add(id(other))
            row.append(other)
            queue.append(other)
    return row


def _link_row_visual_bodies(blocks: list[dict[str, Any]]) -> None:
    """Share one Figure N caption across side-by-side panel bodies."""
    bodies = [
        block
        for block in blocks
        if block.get("type") == "figure" and _normalize_bbox(block.get("bbox"))
    ]
    if len(bodies) < 2:
        return
    figure_captions = [
        block
        for block in blocks
        if block.get("type") == "caption"
        and not block.get("visual_enhancement")
        and not _is_visual_footnote_block(block)
        and _is_figure_level_caption(block)
        and _normalize_bbox(block.get("bbox"))
    ]
    if not figure_captions:
        return
    seen_rows: set[tuple[int, ...]] = set()
    for body in bodies:
        row = _horizontal_figure_row(body, bodies)
        if len(row) < 2:
            continue
        row_key = tuple(sorted(id(member) for member in row))
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        row_box = None
        for member in row:
            member_box = _normalize_bbox(member.get("bbox"))
            if not member_box:
                continue
            if row_box is None:
                row_box = list(member_box)
            else:
                row_box = [
                    min(row_box[0], member_box[0]),
                    min(row_box[1], member_box[1]),
                    max(row_box[2], member_box[2]),
                    max(row_box[3], member_box[3]),
                ]
        if not row_box:
            continue
        shared = _pick_adjacent_caption_box(row_box, "below", figure_captions)
        if not shared:
            continue
        figure_id = str(shared.get("figure_id") or row[0].get("figure_id") or row[0].get("block_id") or "")
        caption_id = str(shared.get("block_id") or "")
        row_ids = {str(member.get("block_id") or "") for member in row}
        for member in row:
            member["figure_id"] = figure_id
            if caption_id:
                member["caption_block_id"] = caption_id
            _bind_caption_to_body(shared, member)
        for block in blocks:
            if block.get("type") != "caption":
                continue
            linked = {
                str(block.get("linked_content_id") or ""),
                *(str(item) for item in (block.get("linked_content_ids") or [])),
            }
            if linked & row_ids:
                block["figure_id"] = figure_id


def _body_has_trusted_caption_children(
    body: dict[str, Any],
    captions: list[dict[str, Any]],
    prefer: str,
) -> bool:
    trusted = [
        {"bbox": caption.get("bbox"), "block": caption}
        for caption in captions
        if str(caption.get("caption_geometry_source") or "") in _TRUSTED_CAPTION_GEOMETRY
        and _normalize_bbox(caption.get("bbox"))
    ]
    return bool(_pick_adjacent_caption_box(body.get("bbox"), prefer, trusted))


def _relocate_visual_captions(pages: list[dict[str, Any]], pdf_path: Path | str | None) -> None:
    """Snap inferred caption siblings onto PDF Table/Figure lines when available."""
    pdf_file = Path(pdf_path) if pdf_path else None
    if not pdf_file or not pdf_file.exists():
        return
    try:
        import fitz

        pdf_doc = fitz.open(str(pdf_file))
    except Exception as exc:
        logger.warning("[MinerUBlockIndex] failed to open PDF for caption relocate: %s", exc)
        return

    try:
        for page in pages:
            page_num = int(page.get("page") or 0)
            if page_num < 1 or page_num > pdf_doc.page_count:
                continue
            pdf_page = pdf_doc[page_num - 1]
            candidates = _pdf_caption_candidates(pdf_page, page_num)
            if not candidates:
                continue
            blocks = list(page.get("blocks") or [])
            bodies = [
                block
                for block in blocks
                if isinstance(block, dict)
                and block.get("type") in {"figure", "table"}
                and not block.get("visual_enhancement")
                and _normalize_bbox(block.get("bbox"))
            ]
            captions = [
                block
                for block in blocks
                if isinstance(block, dict)
                and block.get("type") == "caption"
                and not block.get("visual_enhancement")
                and not _is_visual_footnote_block(block)
                and _normalize_bbox(block.get("bbox"))
            ]
            stop_boxes = [
                block.get("bbox")
                for block in blocks
                if isinstance(block, dict)
                and block.get("type") in {"paragraph", "heading", "code"}
                and _normalize_bbox(block.get("bbox"))
            ]
            other_bodies = [body.get("bbox") for body in bodies if _normalize_bbox(body.get("bbox"))]
            used_candidates: set[int] = set()
            for body in bodies:
                prefer = "above" if body.get("type") == "table" else "below"
                if _body_has_trusted_caption_children(body, captions, prefer):
                    continue
                unused = [
                    candidate
                    for candidate in candidates
                    if id(candidate) not in used_candidates
                ]
                hit = _pick_adjacent_caption_box(body["bbox"], prefer, unused)
                if not hit:
                    continue
                grown = _expand_pdf_caption_bbox(
                    pdf_page,
                    hit["bbox"],
                    body["bbox"],
                    other_bodies=[box for box in other_bodies if box is not body.get("bbox")],
                    stop_boxes=stop_boxes,
                )
                target = _pick_adjacent_caption_box(
                    body["bbox"],
                    prefer,
                    [{"bbox": caption.get("bbox"), "block": caption} for caption in captions],
                )
                caption = target.get("block") if isinstance(target, dict) else None
                if isinstance(caption, dict) and str(caption.get("caption_geometry_source") or "") in _TRUSTED_CAPTION_GEOMETRY:
                    continue
                if isinstance(caption, dict) and not _caption_text_matches_pdf_candidate(
                    str(caption.get("text") or ""),
                    str(hit.get("text") or ""),
                ):
                    continue
                used_candidates.add(id(hit))
                if isinstance(caption, dict):
                    caption["bbox"] = grown
                    caption["caption_geometry_source"] = "pdf_caption"
                    if hit.get("text") and len(str(hit.get("text") or "")) > 8:
                        caption["text"] = str(caption.get("text") or hit["text"])
                    continue
                caption_item = {
                    "type": "table_caption" if body.get("type") == "table" else "image_caption",
                    "text": str(hit.get("text") or ""),
                    "bbox": grown,
                    "page_idx": page_num - 1,
                }
                created = _convert_mineru_item(
                    caption_item,
                    page_num,
                    {page_num: {"width": float(page.get("width_pts") or 612.0), "height": float(page.get("height_pts") or 792.0)}},
                    None,
                    source_name="pdf_caption",
                )
                if created:
                    created["caption_geometry_source"] = "pdf_caption"
                    created["block_id"] = f"p{page_num}_b{len(blocks)}"
                    created["reading_order"] = len(blocks)
                    blocks.append(created)
                    captions.append(created)
            page["blocks"] = blocks
    finally:
        pdf_doc.close()
