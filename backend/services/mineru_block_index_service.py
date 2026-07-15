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
    _build_outline,
    _limit_text,
    _normalize_bbox,
)

logger = logging.getLogger(__name__)

MINERU_BLOCK_INDEX_SOURCE = "mineru_vlm"
MINERU_RAW_VERSION = 1


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
    blocks_by_page: dict[int, list[dict[str, Any]]] = {}

    middle_json = payload.get("middle_json")
    content_list_json = payload.get("content_list_json")

    if isinstance(middle_json, dict):
        blocks_by_page = _blocks_from_middle_json(middle_json, page_specs)
    if not blocks_by_page and isinstance(content_list_json, list):
        blocks_by_page = _blocks_from_content_list(content_list_json, page_specs)
    if not blocks_by_page and isinstance(middle_json, list):
        blocks_by_page = _blocks_from_content_list(middle_json, page_specs)

    if not blocks_by_page:
        raise ValueError("MinerU 结果中没有可转换的版面块")

    page_nums = sorted(set(page_specs.keys()) | set(blocks_by_page.keys()))
    pages: list[dict[str, Any]] = []
    for page_num in page_nums:
        spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
        blocks = sorted(
            blocks_by_page.get(page_num, []),
            key=lambda b: (
                float((b.get("bbox") or [0, 0, 0, 0])[1]),
                float((b.get("bbox") or [0, 0, 0, 0])[0]),
            ),
        )
        for idx, block in enumerate(blocks):
            block["block_id"] = block.get("block_id") or f"p{page_num}_b{idx}"
            block["section_id"] = None
        pages.append({
            "page": page_num,
            "width_pts": float(spec.get("width") or 612.0),
            "height_pts": float(spec.get("height") or 792.0),
            "blocks": blocks,
            "layout_regions": [],
        })

    outline = _build_outline([], pages)
    _assign_sections(pages, outline)
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

    return {
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
        },
    }


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


def _blocks_from_middle_json(data: dict[str, Any], page_specs: dict[int, dict[str, float]]) -> dict[int, list[dict[str, Any]]]:
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
            block = _convert_mineru_item(raw, page_num, page_specs, page_source_size)
            if block:
                blocks_by_page.setdefault(page_num, []).append(block)
    return blocks_by_page


def _blocks_from_content_list(items: list[Any], page_specs: dict[int, dict[str, float]]) -> dict[int, list[dict[str, Any]]]:
    blocks_by_page: dict[int, list[dict[str, Any]]] = {}
    items_by_page: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        page_num = _page_num(item)
        items_by_page.setdefault(page_num, []).append(item)

    for page_num, page_items in items_by_page.items():
        page_spec = page_specs.get(page_num, {"width": 612.0, "height": 792.0})
        page_source_size = _page_source_size(page_items[0]) or _infer_page_source_size(
            page_items,
            page_width=float(page_spec.get("width") or 612.0),
            page_height=float(page_spec.get("height") or 792.0),
        )
        for item in page_items:
            block = _convert_mineru_item(item, page_num, page_specs, _page_source_size(item) or page_source_size)
            if block:
                blocks_by_page.setdefault(page_num, []).append(block)
    return blocks_by_page


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
    if not bbox_pts:
        return None

    block: dict[str, Any] = {
        "type": block_type,
        "bbox": bbox_pts,
        "text": _limit_text(text, 2400),
        "section_id": None,
        "source": MINERU_BLOCK_INDEX_SOURCE,
        "mineru_type": raw_type or block_type,
    }
    if block_type == "heading":
        block["level"] = _infer_heading_level(text)
    if block_type == "artifact":
        block["layout_excluded_from_outline"] = True
    return block


def _normalize_type(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _map_mineru_type(raw_type: str, item: dict[str, Any]) -> str:
    if raw_type in {"title", "heading", "header"}:
        return "heading"
    if raw_type in {"text", "plain_text", "paragraph", "para"}:
        return "paragraph"
    if raw_type in {"table"}:
        return "table"
    if raw_type in {"table_caption", "image_caption", "caption"}:
        return "caption"
    if raw_type in {"image", "figure"}:
        return "figure"
    if raw_type in {"interline_equation", "equation", "formula", "inline_equation"}:
        return "formula"
    if raw_type in {"discarded", "discarded_block", "abandon", "footer", "header_footer", "page_number"}:
        return "artifact"
    if item.get("discarded") is True:
        return "artifact"
    return ""


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
    match = re.match(r"^(\d+(?:\.\d+)*)\s+", stripped)
    if match:
        return max(1, min(match.group(1).count(".") + 1, 4))
    if re.match(r"^[A-Z]\.\s+", stripped):
        return 2
    return 1


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]
