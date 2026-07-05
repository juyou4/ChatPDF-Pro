"""Document block index for immersive reading.

Builds a stable page/block/outline structure from the original PDF when
available, with a text-only fallback for imported non-PDF documents.
"""
from __future__ import annotations

import json
import logging
import re
import io
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)

BLOCK_INDEX_VERSION = 6

_RE_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.?\s+)(.+)$")
_RE_ROMAN_HEADING = re.compile(r"^\s*([IVXLCM]+)\.\s+(.+)$")
_RE_ALPHA_HEADING = re.compile(r"^\s*([A-Z])\.\s+(.+)$")
_RE_CANONICAL_HEADING = re.compile(
    r"^\s*(abstract|introduction|background|related\s+work|method|methods|methodology|"
    r"approach|experiments?|evaluation|results?|discussion|conclusion|limitations?|"
    r"preliminar(?:y|ies)|implementation|future\s+work|references|acknowledg(?:e)?ments?|"
    r"appendix|supplementary\s+material)\s*$",
    re.IGNORECASE,
)
_RE_CAPTION = re.compile(
    r"^\s*(fig(?:ure)?\.?|图|table|表)\s*([0-9]+|[ivxlcdm]+)\b",
    re.IGNORECASE,
)
_RE_TABLE_LABEL = re.compile(r"^\s*(table|表)\s*([0-9]+|[ivxlcdm]+)\s*[:.\-]?\s*$", re.IGNORECASE)
_RE_DECIMAL_TOKEN = re.compile(r"(?<![A-Za-z])[-+]?\d+\.\d+(?![A-Za-z])")
_RE_PUBLICATION_HEADER_CUE = re.compile(
    r"\b(vol\.?|no\.?|pp\.?|transactions?|journal|proceedings|conference|copyright|"
    r"authorized|licensed|downloaded|doi|issn|isbn)\b",
    re.IGNORECASE,
)
_RE_ALGORITHM_LINE = re.compile(
    r"^\s*(input|output|require|ensure|initialize|initialise|update|repeat|return|for\s+each|for\s+"
    r"|while\s+|if\s+|else\b|end\b|stage\s*\d+)\b",
    re.IGNORECASE,
)


def get_block_index_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "block_indexes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_block_index_path(data_dir: Path | str, doc_id: str) -> Path:
    return get_block_index_dir(data_dir) / f"{doc_id}.json"


def load_block_index(data_dir: Path | str, doc_id: str) -> dict[str, Any] | None:
    path = get_block_index_path(data_dir, doc_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != BLOCK_INDEX_VERSION:
            return None
        return data
    except Exception as exc:
        logger.warning("[BlockIndex] Failed to load %s: %s", path, exc)
        return None


def save_block_index(data_dir: Path | str, doc_id: str, index: dict[str, Any]) -> None:
    path = get_block_index_path(data_dir, doc_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("[BlockIndex] Failed to save %s: %s", path, exc)


def ensure_block_index(
    *,
    doc_id: str,
    doc: dict[str, Any],
    data_dir: Path | str,
    pdf_path: Path | str | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    if not force_rebuild:
        cached = load_block_index(data_dir, doc_id)
        if cached:
            return cached

    index = build_block_index(doc_id=doc_id, doc=doc, pdf_path=pdf_path)
    save_block_index(data_dir, doc_id, index)
    return index


def build_block_index(
    *,
    doc_id: str,
    doc: dict[str, Any],
    pdf_path: Path | str | None = None,
) -> dict[str, Any]:
    data = doc.get("data", {}) if isinstance(doc, dict) else {}
    pages: list[dict[str, Any]] = []
    toc_items: list[dict[str, Any]] = []
    source = "fallback"

    pdf_file = Path(pdf_path) if pdf_path else None
    if pdf_file and pdf_file.exists():
        try:
            pages, toc_items = _build_pages_from_pdf(pdf_file)
            source = "pdf_native"
        except Exception as exc:
            logger.warning("[BlockIndex] PDF build failed for %s: %s", doc_id, exc)

    if not pages:
        pages = _build_pages_from_text(data.get("pages", []))
        source = "text_fallback"

    _mark_repeated_page_artifacts(pages)
    _inject_visual_blocks(pages, data)
    outline = _build_outline(toc_items, pages)
    _assign_sections(pages, outline)

    return {
        "version": BLOCK_INDEX_VERSION,
        "doc_id": doc_id,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "pages": pages,
        "outline": outline,
    }


def _build_pages_from_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import fitz

    pdf_doc = fitz.open(str(pdf_path))
    try:
        toc_items = [
            {"level": int(level), "title": str(title).strip(), "page": int(page)}
            for level, title, page in pdf_doc.get_toc(simple=True)
            if str(title).strip() and int(page) > 0
        ]

        pages: list[dict[str, Any]] = []
        for page_idx in range(len(pdf_doc)):
            page = pdf_doc[page_idx]
            text_dict = page.get_text("dict") or {}
            layout_regions = _detect_page_layout_regions(page)
            page_blocks = _extract_page_text_blocks(
                text_dict=text_dict,
                page_number=page_idx + 1,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                layout_regions=layout_regions,
            )
            pages.append({
                "page": page_idx + 1,
                "width_pts": float(page.rect.width),
                "height_pts": float(page.rect.height),
                "blocks": page_blocks,
                "layout_regions": layout_regions,
            })
        return pages, toc_items
    finally:
        pdf_doc.close()


def _extract_page_text_blocks(
    text_dict: dict[str, Any],
    page_number: int,
    page_width: float = 612.0,
    page_height: float = 792.0,
    layout_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw_blocks = text_dict.get("blocks", []) if isinstance(text_dict, dict) else []
    font_sizes = _collect_font_sizes(raw_blocks)
    median_font = median(font_sizes) if font_sizes else 10.0
    body_font_key = _dominant_page_body_font_key(raw_blocks, median_font)

    blocks: list[dict[str, Any]] = []
    seq = 0
    for raw in raw_blocks:
        if raw.get("type") != 0:
            continue
        lines = raw.get("lines", []) or []
        for group_lines, forced_heading in _split_text_block_line_groups(
            lines,
            median_font=median_font,
            body_font_key=body_font_key,
            page_width=page_width,
        ):
            text = _block_text(group_lines)
            if len(text) < 2:
                continue

            bbox = _merge_bboxes(
                span.get("bbox")
                for line in group_lines
                for span in (line.get("spans", []) or [])
            ) or _normalize_bbox(raw.get("bbox"))
            if not bbox:
                continue

            max_font = max(_collect_font_sizes([{"lines": group_lines}]) or [median_font])
            is_bold = _line_is_bold(group_lines)
            block_type, heading_level = _classify_text_block(
                text,
                max_font,
                median_font,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                lines=group_lines,
            )
            if forced_heading and block_type != "heading" and _looks_like_heading_text(text):
                block_type = "heading"
                heading_level = _embedded_heading_level(text)
            block = {
                "block_id": f"p{page_number}_b{seq}",
                "type": block_type,
                "bbox": bbox,
                "text": _limit_text(text, 2400),
                "section_id": None,
                "font_size": round(max_font, 2),
                "is_bold": is_bold,
                "line_count": len([line for line in group_lines if line.get("spans")]),
                "font_key": _dominant_font_key(group_lines),
            }
            if forced_heading:
                block["split_from_mixed_block"] = True
            if heading_level:
                block["level"] = heading_level
            blocks.append(block)
            seq += 1

    _demote_adjacent_table_titles(blocks)
    _apply_layout_regions(blocks, layout_regions or [], page_width, page_height)
    return blocks


def _build_pages_from_text(source_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for page_idx, page_data in enumerate(source_pages or []):
        page_num = int(page_data.get("page") or page_idx + 1)
        text = str(page_data.get("content") or page_data.get("text") or "")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks = []
        y = 72.0
        available_height = 660.0
        step = max(32.0, available_height / max(len(paragraphs), 1))
        for idx, paragraph in enumerate(paragraphs):
            block_type, heading_level = _classify_text_block(paragraph, 12.0, 10.0)
            block = {
                "block_id": f"p{page_num}_b{idx}",
                "type": block_type,
                "bbox": [54.0, y, 558.0, min(760.0, y + max(22.0, step * 0.75))],
                "text": _limit_text(paragraph, 2400),
                "section_id": None,
            }
            if heading_level:
                block["level"] = heading_level
            blocks.append(block)
            y += step
        pages.append({
            "page": page_num,
            "width_pts": 612.0,
            "height_pts": 792.0,
            "blocks": blocks,
        })
    return pages


def _mark_repeated_page_artifacts(pages: list[dict[str, Any]]) -> None:
    """Demote repeated top/bottom blocks such as journal headers and footers."""
    candidates: dict[str, list[tuple[int, float, dict[str, Any]]]] = {}
    for page in pages:
        page_num = int(page.get("page") or 1)
        page_height = float(page.get("height_pts") or 792.0)
        for block in page.get("blocks", []) or []:
            text = str(block.get("text") or "").strip()
            bbox = _normalize_bbox(block.get("bbox"))
            if not text or not bbox:
                continue
            y_mid = ((bbox[1] + bbox[3]) / 2.0) / max(page_height, 1.0)
            if 0.12 < y_mid < 0.88:
                continue
            norm = _normalize_repeated_artifact_text(text)
            if len(norm) < 10:
                continue
            candidates.setdefault(norm, []).append((page_num, y_mid, block))

    for entries in candidates.values():
        page_nums = {page_num for page_num, _y_mid, _block in entries}
        if len(page_nums) < 2:
            continue
        entries = sorted(entries, key=lambda item: item[1])
        clusters: list[list[tuple[int, float, dict[str, Any]]]] = []
        for entry in entries:
            if not clusters or abs(clusters[-1][-1][1] - entry[1]) > 0.018:
                clusters.append([entry])
            else:
                clusters[-1].append(entry)
        for cluster in clusters:
            if len({page_num for page_num, _y_mid, _block in cluster}) < 2:
                continue
            for _page_num, _y_mid, block in cluster:
                block["type"] = "artifact"
                block.pop("level", None)


def _normalize_repeated_artifact_text(text: str) -> str:
    value = " ".join(str(text or "").lower().split())
    value = re.sub(r"\b\d+\b", "#", value)
    value = re.sub(r"[^a-z\u4e00-\u9fff#]+", " ", value)
    return " ".join(value.split())


def _inject_visual_blocks(pages: list[dict[str, Any]], data: dict[str, Any]) -> None:
    page_map = {int(page.get("page", 0)): page for page in pages}
    existing_ids: set[str] = {
        str(block.get("block_id"))
        for page in pages
        for block in page.get("blocks", [])
        if block.get("block_id")
    }

    for fig in data.get("logical_figures", []) or []:
        if not isinstance(fig, dict):
            continue
        page_num = int(fig.get("page_idx", 0)) + 1
        bbox = _normalize_bbox(fig.get("full_bbox_page_pts")) or _normalize_bbox(fig.get("body_bbox_page_pts"))
        if not bbox or page_num not in page_map:
            continue
        block_id = str(fig.get("figure_id") or f"figure_p{page_num}_{len(existing_ids)}")
        _append_visual_block(
            page_map[page_num],
            existing_ids,
            block_id=block_id,
            block_type="figure",
            bbox=bbox,
            text=str(fig.get("caption_text") or fig.get("caption") or "Figure").strip(),
        )

    for fig in data.get("figures", []) or []:
        if not isinstance(fig, dict):
            continue
        page_num = int(fig.get("page") or 0)
        if page_num not in page_map:
            continue
        bbox = (
            _normalize_bbox(fig.get("group_bbox"))
            or _normalize_bbox(fig.get("figure_bbox"))
            or _normalize_bbox(fig.get("caption_bbox"))
            or _normalize_bbox(fig.get("bbox"))
        )
        if not bbox:
            continue
        block_id = str(fig.get("figure_id") or f"figure_p{page_num}_{len(existing_ids)}")
        _append_visual_block(
            page_map[page_num],
            existing_ids,
            block_id=block_id,
            block_type="figure",
            bbox=bbox,
            text=str(fig.get("caption") or fig.get("label") or "Figure").strip(),
        )

    for img in data.get("images", []) or []:
        if not isinstance(img, dict):
            continue
        page_num = int(img.get("page") or 0)
        bbox = _normalize_bbox(img.get("bbox"))
        if not bbox or page_num not in page_map:
            continue
        block_id = str(img.get("id") or f"image_p{page_num}_{len(existing_ids)}")
        _append_visual_block(
            page_map[page_num],
            existing_ids,
            block_id=block_id,
            block_type="figure",
            bbox=bbox,
            text="Image",
        )

    for page in pages:
        page["blocks"] = sorted(
            page.get("blocks", []),
            key=lambda b: (float((b.get("bbox") or [0, 0, 0, 0])[1]), float((b.get("bbox") or [0, 0, 0, 0])[0])),
        )


def _append_visual_block(
    page: dict[str, Any],
    existing_ids: set[str],
    *,
    block_id: str,
    block_type: str,
    bbox: list[float],
    text: str,
) -> None:
    if block_id in existing_ids:
        return
    existing_ids.add(block_id)
    page.setdefault("blocks", []).append({
        "block_id": block_id,
        "type": block_type,
        "bbox": bbox,
        "text": _limit_text(text or block_type, 1000),
        "section_id": None,
    })


def _demote_adjacent_table_titles(blocks: list[dict[str, Any]]) -> None:
    """将 TABLE 标签后的表题降级为 caption，避免进入章节大纲。"""
    for idx, block in enumerate(blocks[:-1]):
        text = str(block.get("text") or "").strip()
        if not _RE_TABLE_LABEL.match(text):
            continue

        next_block = blocks[idx + 1]
        next_text = str(next_block.get("text") or "").strip()
        if not next_text or len(next_text) > 220:
            continue
        if _looks_like_numeric_measurement_line(next_text):
            continue
        if not _looks_like_table_title(next_text):
            continue

        label_bbox = _normalize_bbox(block.get("bbox"))
        title_bbox = _normalize_bbox(next_block.get("bbox"))
        if label_bbox and title_bbox:
            y_gap = title_bbox[1] - label_bbox[3]
            if y_gap < -4 or y_gap > 42:
                continue

        next_block["type"] = "caption"
        next_block["caption_role"] = "table_title"
        next_block.pop("level", None)


def _build_outline(toc_items: list[dict[str, Any]], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outline: list[dict[str, Any]] = []
    if toc_items:
        for idx, item in enumerate(toc_items):
            page_num = item.get("page", 1)
            first_block = _first_block_id_on_page(pages, page_num)
            outline.append({
                "section_id": f"s{idx + 1}",
                "title": _limit_text(item.get("title", ""), 180),
                "level": max(1, min(int(item.get("level", 1)), 6)),
                "page": page_num,
                "first_block": first_block,
                "source": "toc",
            })
        return outline

    for page in pages:
        page_num = int(page.get("page", 1))
        page_width = float(page.get("width_pts") or 612.0)
        page_height = float(page.get("height_pts") or 792.0)
        for block in page.get("blocks", []):
            if block.get("type") != "heading":
                continue
            title = str(block.get("text") or "").strip()
            if not title or len(title) > 180:
                continue
            if _looks_like_layout_noise(title, _normalize_bbox(block.get("bbox")), page_width, page_height):
                continue
            outline.append({
                "section_id": f"s{len(outline) + 1}",
                "title": title,
                "level": max(1, min(int(block.get("level") or 2), 6)),
                "page": page_num,
                "first_block": block.get("block_id"),
                "source": "heading",
            })

    if outline:
        return outline[:80]

    first_page = pages[0].get("page", 1) if pages else 1
    return [{
        "section_id": "s1",
        "title": "全文",
        "level": 1,
        "page": first_page,
        "first_block": _first_block_id_on_page(pages, first_page),
        "source": "fallback",
    }]


def _assign_sections(pages: list[dict[str, Any]], outline: list[dict[str, Any]]) -> None:
    anchors: list[tuple[int, int, str]] = []
    block_order: dict[str, tuple[int, int]] = {}
    for page in pages:
        page_num = int(page.get("page", 1))
        for idx, block in enumerate(page.get("blocks", [])):
            block_id = block.get("block_id")
            if block_id:
                block_order[str(block_id)] = (page_num, idx)

    for item in outline:
        block_id = item.get("first_block")
        if block_id and str(block_id) in block_order:
            page_num, block_idx = block_order[str(block_id)]
        else:
            page_num, block_idx = int(item.get("page", 1)), 0
        anchors.append((page_num, block_idx, str(item.get("section_id"))))

    anchors.sort(key=lambda x: (x[0], x[1]))
    if not anchors:
        return

    current = anchors[0][2]
    anchor_pos = 0
    for page in pages:
        page_num = int(page.get("page", 1))
        for idx, block in enumerate(page.get("blocks", [])):
            while anchor_pos + 1 < len(anchors) and (page_num, idx) >= (anchors[anchor_pos + 1][0], anchors[anchor_pos + 1][1]):
                anchor_pos += 1
                current = anchors[anchor_pos][2]
            block["section_id"] = current


def _first_block_id_on_page(pages: list[dict[str, Any]], page_num: int) -> str | None:
    for page in pages:
        if int(page.get("page", 0)) != int(page_num):
            continue
        for block in page.get("blocks", []):
            if block.get("block_id"):
                return block["block_id"]
    return None


def _detect_page_layout_regions(page: Any) -> list[dict[str, Any]]:
    """YOLO 可用时检测页面版面区域，失败静默降级到纯文本路径。"""
    try:
        from PIL import Image
        from services.layout_service import detect_layout, is_available, pixel_bbox_to_page_pts

        if not is_available():
            return []

        scale = 2.0
        import fitz
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        detections = detect_layout(image, conf=0.18)
        regions = []
        for det in detections or []:
            bbox = det.get("bbox")
            if not bbox:
                continue
            page_bbox = _normalize_bbox(pixel_bbox_to_page_pts(
                bbox,
                pix.width,
                pix.height,
                float(page.rect.width),
                float(page.rect.height),
            ))
            if not page_bbox:
                continue
            regions.append({
                "category_id": int(det.get("category_id", -1)),
                "category_name": str(det.get("category_name") or ""),
                "bbox": page_bbox,
                "score": float(det.get("score") or 0),
            })
        return regions
    except Exception as exc:
        logger.debug("[BlockIndex] DocLayout-YOLO layout detection skipped: %s", exc)
        return []


def _apply_layout_regions(
    blocks: list[dict[str, Any]],
    layout_regions: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> None:
    if not blocks or not layout_regions:
        return

    title_regions = [r for r in layout_regions if r.get("category_name") == "Title"]
    table_regions = [r for r in layout_regions if r.get("category_name") == "TableBody"]
    image_regions = [r for r in layout_regions if r.get("category_name") == "ImageBody"]
    excluded_regions = table_regions + image_regions

    for block in blocks:
        bbox = _normalize_bbox(block.get("bbox"))
        text = str(block.get("text") or "").strip()
        if not bbox or not text:
            continue

        excluded = _best_overlapping_region(bbox, excluded_regions)
        if excluded and _block_should_demote_for_layout(block, excluded):
            block["layout_region"] = excluded.get("category_name")
            block["layout_score"] = excluded.get("score")
            block["layout_excluded_from_outline"] = True
            block.pop("level", None)
            if _looks_like_region_label_noise(text) or block.get("type") == "heading":
                block["type"] = "artifact"
            elif excluded.get("category_name") == "TableBody":
                block["type"] = "table"
            elif excluded.get("category_name") == "ImageBody":
                block["type"] = "figure"
            continue

        title = _best_overlapping_region(bbox, title_regions)
        if title and _block_should_promote_for_layout(block, title, page_width, page_height):
            block["layout_region"] = "Title"
            block["layout_score"] = title.get("score")
            block["type"] = "heading"
            block["level"] = _layout_title_level(text, bbox, page_height)
            block["layout_promoted"] = True


def _best_overlapping_region(
    bbox: list[float],
    regions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for region in regions:
        region_bbox = _normalize_bbox(region.get("bbox"))
        if not region_bbox:
            continue
        score = _bbox_intersection_ratio(bbox, region_bbox)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, region)
    return best[1] if best else None


def _block_should_demote_for_layout(block: dict[str, Any], region: dict[str, Any]) -> bool:
    bbox = _normalize_bbox(block.get("bbox"))
    region_bbox = _normalize_bbox(region.get("bbox"))
    if not bbox or not region_bbox:
        return False
    coverage = _bbox_intersection_ratio(bbox, region_bbox)
    center_inside = _bbox_center_inside(bbox, region_bbox)
    text = str(block.get("text") or "").strip()
    if coverage >= 0.45 or center_inside:
        return True
    return block.get("type") == "heading" and coverage >= 0.25 and _looks_like_region_label_noise(text)


def _block_should_promote_for_layout(
    block: dict[str, Any],
    region: dict[str, Any],
    page_width: float,
    page_height: float,
) -> bool:
    bbox = _normalize_bbox(block.get("bbox"))
    region_bbox = _normalize_bbox(region.get("bbox"))
    if not bbox or not region_bbox:
        return False
    text = str(block.get("text") or "").strip()
    if not _looks_like_embedded_heading_line(text):
        return False
    if _looks_like_layout_noise(text, bbox, page_width, page_height):
        return False
    coverage = _bbox_intersection_ratio(bbox, region_bbox)
    return coverage >= 0.18 or _bbox_center_inside(bbox, region_bbox)


def _layout_title_level(text: str, bbox: list[float], page_height: float) -> int:
    explicit = _embedded_heading_level(text)
    if explicit == 1:
        return 1
    if bbox and bbox[1] < page_height * 0.20 and len(text.split()) >= 5:
        return 1
    return explicit or 2


def _looks_like_region_label_noise(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if _looks_like_algorithm_line(stripped) or _looks_like_numeric_measurement_line(stripped):
        return True
    if _RE_CAPTION.match(stripped):
        return False
    words = stripped.split()
    if len(words) <= 4:
        return True
    if len(stripped) <= 40 and not stripped.endswith("."):
        return True
    return False


def _bbox_intersection_ratio(inner: list[float], outer: list[float]) -> float:
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return intersection / area


def _bbox_center_inside(inner: list[float], outer: list[float]) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _split_text_block_line_groups(
    lines: list[dict[str, Any]],
    *,
    median_font: float,
    body_font_key: str,
    page_width: float,
) -> list[tuple[list[dict[str, Any]], bool]]:
    valid_lines = [line for line in lines or [] if _line_text(line)]
    if len(valid_lines) <= 1:
        return [(valid_lines, False)] if valid_lines else []

    groups: list[tuple[list[dict[str, Any]], bool]] = []
    idx = 0
    while idx < len(valid_lines) - 1:
        line = valid_lines[idx]
        remaining = valid_lines[idx + 1:]
        if not _line_should_split_as_heading(
            line,
            remaining,
            median_font=median_font,
            body_font_key=body_font_key,
            page_width=page_width,
        ):
            break
        groups.append(([line], True))
        idx += 1

    if not groups:
        return [(valid_lines, False)]
    if idx < len(valid_lines):
        groups.append((valid_lines[idx:], False))
    return groups


def _line_should_split_as_heading(
    line: dict[str, Any],
    remaining_lines: list[dict[str, Any]],
    *,
    median_font: float,
    body_font_key: str,
    page_width: float,
) -> bool:
    text = _line_text(line)
    if not text or not _looks_like_embedded_heading_line(text):
        return False
    if _looks_like_layout_noise(text, _line_bbox(line), page_width, 792.0):
        return False

    remaining_text = " ".join(_line_text(item) for item in remaining_lines if _line_text(item))
    if len(remaining_text) < 24:
        return False

    if _RE_CANONICAL_HEADING.match(text) or _RE_NUMBERED_HEADING.match(text) or _RE_ROMAN_HEADING.match(text) or _RE_ALPHA_HEADING.match(text):
        return True

    line_font = _dominant_font_key([line])
    next_font = _dominant_font_key(remaining_lines[: min(3, len(remaining_lines))])
    font_differs_from_body = bool(line_font and body_font_key and line_font != body_font_key)
    font_differs_from_next = bool(line_font and next_font and line_font != next_font)
    size = _line_max_font(line, median_font)
    size_signal = median_font > 0 and size >= median_font * 1.08
    weight_signal = _line_is_bold([line])
    return (font_differs_from_body or font_differs_from_next or size_signal or weight_signal) and _looks_like_heading_text(text)


def _looks_like_embedded_heading_line(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped or len(stripped) > 140:
        return False
    if _RE_CAPTION.match(stripped) or _looks_like_numeric_measurement_line(stripped) or _looks_like_algorithm_line(stripped):
        return False
    if _RE_CANONICAL_HEADING.match(stripped) or _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return True
    words = stripped.split()
    if not 1 <= len(words) <= 12:
        return False
    return _looks_like_heading_text(stripped)


def _embedded_heading_level(text: str) -> int:
    stripped = " ".join(str(text or "").split())
    numbered = _RE_NUMBERED_HEADING.match(stripped)
    if numbered:
        return max(1, min(numbered.group(1).count(".") + 1, 4))
    if _RE_ROMAN_HEADING.match(stripped) or _RE_CANONICAL_HEADING.match(stripped):
        return 1
    if _RE_ALPHA_HEADING.match(stripped):
        return 2
    return 2


def _classify_text_block(
    text: str,
    max_font: float,
    median_font: float,
    *,
    bbox: list[float] | None = None,
    page_width: float = 612.0,
    page_height: float = 792.0,
    lines: list[dict[str, Any]] | None = None,
) -> tuple[str, int | None]:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return "paragraph", None
    if _RE_CAPTION.match(stripped):
        return "caption", None
    if _looks_like_numeric_measurement_line(stripped):
        return "paragraph", None
    if _looks_like_algorithm_line(stripped):
        return "paragraph", None

    heading_level = _heading_level(
        stripped,
        max_font,
        median_font,
        bbox=bbox,
        page_width=page_width,
        page_height=page_height,
        lines=lines,
    )
    if heading_level:
        return "heading", heading_level
    return "paragraph", None


def _heading_level(
    text: str,
    max_font: float,
    median_font: float,
    *,
    bbox: list[float] | None = None,
    page_width: float = 612.0,
    page_height: float = 792.0,
    lines: list[dict[str, Any]] | None = None,
) -> int | None:
    if len(text) > 180:
        return None
    if _looks_like_algorithm_line(text):
        return None
    if _looks_like_layout_noise(text, bbox, page_width, page_height):
        return None

    numbered = _RE_NUMBERED_HEADING.match(text)
    if numbered:
        number = numbered.group(1)
        if re.match(r"^\d{3,}(?:\.|$)", number) and _looks_like_publication_header_footer(text):
            return None
        return max(1, min(number.count(".") + 1, 4))

    if _RE_ROMAN_HEADING.match(text):
        return 1

    if _RE_ALPHA_HEADING.match(text):
        return 2

    if _RE_CANONICAL_HEADING.match(text):
        return 1

    alpha = re.sub(r"[^A-Za-z]", "", text)
    if (
        3 <= len(alpha) <= 80
        and alpha.isupper()
        and len(text.split()) <= 10
        and not re.match(r"^\s*\d", text)
    ):
        return 1

    if (
        median_font > 0
        and max_font >= median_font * 1.42
        and _looks_like_standalone_heading(text, bbox, page_width, lines)
    ):
        return 1 if max_font >= median_font * 1.55 else 2

    if (
        _line_is_bold(lines)
        and _looks_like_standalone_heading(text, bbox, page_width, lines)
        and _looks_like_heading_text(text)
    ):
        return 2

    return None


def _looks_like_layout_noise(
    text: str,
    bbox: list[float] | None,
    page_width: float,
    page_height: float,
) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if re.fullmatch(r"[\d.,:/()%+\-=×xX<>~]+", compact):
        return True
    if _looks_like_numeric_measurement_line(text):
        return True
    if _looks_like_algorithm_line(text):
        return True
    if re.match(r"^\s*\d{3,}\s+[A-Z]", text) and _looks_like_publication_header_footer(text):
        return True
    digit_ratio = sum(ch.isdigit() for ch in compact) / max(len(compact), 1)
    symbol_ratio = sum(not ch.isalnum() for ch in compact) / max(len(compact), 1)
    if digit_ratio > 0.45 and symbol_ratio > 0.12:
        return True
    if bbox:
        x0, y0, x1, y1 = bbox
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        if x0 < page_width * 0.03 or x1 > page_width * 0.97:
            return True
        if y0 < page_height * 0.08 and _looks_like_publication_header_footer(text):
            return True
        if y1 > page_height * 0.94 and _looks_like_publication_header_footer(text):
            return True
        if height > page_height * 0.28 and width < page_width * 0.10:
            return True
    return False


def _looks_like_publication_header_footer(text: str) -> bool:
    compact = " ".join(str(text or "").split())
    if not compact:
        return False
    if re.match(r"^\s*\d+\s+[A-Z]", compact):
        return True
    return bool(_RE_PUBLICATION_HEADER_CUE.search(compact))


def _looks_like_algorithm_line(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    if _RE_ALGORITHM_LINE.match(stripped):
        return True
    if re.search(r"\b(stage|step)\s*\d+\b", stripped, re.IGNORECASE) and re.search(r"\b(update|initialize|input|output)\b", stripped, re.IGNORECASE):
        return True
    return False


def _looks_like_numeric_measurement_line(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    if _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return False
    decimal_tokens = _RE_DECIMAL_TOKEN.findall(stripped)
    if not decimal_tokens:
        return False
    alpha_tokens = re.findall(r"[A-Za-z]{2,}", stripped)
    number_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", stripped)
    if len(alpha_tokens) <= 2 and (len(number_tokens) >= 2 or len(stripped.split()) <= 8):
        return True
    compact = re.sub(r"\s+", "", stripped)
    digit_ratio = sum(ch.isdigit() for ch in compact) / max(len(compact), 1)
    return len(alpha_tokens) <= 3 and digit_ratio > 0.35 and len(stripped.split()) <= 12


def _looks_like_table_title(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    words = stripped.split()
    if not 2 <= len(words) <= 18:
        return False
    alpha = re.sub(r"[^A-Za-z]", "", stripped)
    if alpha and alpha.isupper() and len(alpha) >= 6:
        return True
    if stripped.endswith("."):
        return False
    return _looks_like_heading_text(stripped)


def _looks_like_standalone_heading(
    text: str,
    bbox: list[float] | None,
    page_width: float,
    lines: list[dict[str, Any]] | None,
) -> bool:
    words = text.split()
    if not 1 <= len(words) <= 12:
        return False
    if "\n" in text and len([line for line in text.splitlines() if line.strip()]) > 2:
        return False
    if not bbox:
        return True

    x0, _y0, x1, _y1 = bbox
    width = max(1.0, x1 - x0)
    center = (x0 + x1) / 2
    is_centered = abs(center - page_width / 2) <= page_width * 0.18
    is_near_text_start = page_width * 0.06 <= x0 <= page_width * 0.28
    is_not_tiny_label = width >= page_width * 0.10
    if not is_not_tiny_label:
        return False

    boldish = False
    for line in lines or []:
        for span in line.get("spans", []) or []:
            font = str(span.get("font") or "").lower()
            flags = int(span.get("flags") or 0)
            if flags & 16 or _font_has_weight_signal(font):
                boldish = True
                break
        if boldish:
            break
    return (is_centered or is_near_text_start) and (boldish or len(words) <= 6)


def _dominant_page_body_font_key(raw_blocks: list[dict[str, Any]], median_font: float) -> str:
    weighted: dict[str, int] = {}
    fallback: dict[str, int] = {}
    for block in raw_blocks or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                key = _normalize_font_key(span.get("font"))
                if not key:
                    continue
                text = str(span.get("text") or "")
                weight = max(1, len(text.strip()))
                fallback[key] = fallback.get(key, 0) + weight
                try:
                    size = float(span.get("size", 0) or 0)
                except (TypeError, ValueError):
                    size = 0
                if median_font > 0 and 0 < size <= median_font * 1.15:
                    weighted[key] = weighted.get(key, 0) + weight
    pool = weighted or fallback
    if not pool:
        return ""
    return max(pool.items(), key=lambda item: item[1])[0]


def _dominant_font_key(lines: list[dict[str, Any]] | None) -> str:
    weighted: dict[str, int] = {}
    for line in lines or []:
        for span in line.get("spans", []) or []:
            key = _normalize_font_key(span.get("font"))
            if not key:
                continue
            text = str(span.get("text") or "")
            weighted[key] = weighted.get(key, 0) + max(1, len(text.strip()))
    if not weighted:
        return ""
    return max(weighted.items(), key=lambda item: item[1])[0]


def _normalize_font_key(font: Any) -> str:
    value = str(font or "").strip().lower()
    if not value:
        return ""
    if "+" in value:
        value = value.split("+", 1)[1]
    return re.sub(r"\s+", "", value)


def _font_has_weight_signal(font: Any) -> bool:
    value = str(font or "").lower()
    return bool(re.search(r"(bold|semibold|semi-bold|demibold|demi-bold|extrabold|extra-bold|black|heavy|medium)", value))


def _line_text(line: dict[str, Any]) -> str:
    pieces = []
    for span in line.get("spans", []) or []:
        text = str(span.get("text") or "")
        if text:
            pieces.append(text)
    return _clean_pdf_text(" ".join("".join(pieces).split()))


def _line_bbox(line: dict[str, Any]) -> list[float] | None:
    return _merge_bboxes(span.get("bbox") for span in line.get("spans", []) or [])


def _line_max_font(line: dict[str, Any], fallback: float) -> float:
    sizes = []
    for span in line.get("spans", []) or []:
        try:
            size = float(span.get("size", 0) or 0)
        except (TypeError, ValueError):
            continue
        if size > 0:
            sizes.append(size)
    return max(sizes or [fallback])


def _line_is_bold(lines: list[dict[str, Any]] | None) -> bool:
    for line in lines or []:
        for span in line.get("spans", []) or []:
            font = str(span.get("font") or "").lower()
            flags = int(span.get("flags") or 0)
            if flags & 16:
                return True
            if _font_has_weight_signal(font):
                return True
    return False


def _looks_like_heading_text(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped or stripped.endswith((".", ",", ";", ":")):
        return False
    if _looks_like_algorithm_line(stripped):
        return False
    if _RE_CANONICAL_HEADING.match(stripped):
        return True
    if _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return True
    if _RE_CAPTION.match(stripped):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", stripped)
    if not words:
        return False
    small_words = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "via", "with"}
    content_words = [word for word in words if word.lower() not in small_words]
    if not content_words:
        return False
    capitalized = sum(1 for word in content_words if word[:1].isupper())
    if capitalized / max(len(content_words), 1) >= 0.6:
        return True
    if len(content_words) <= 4 and stripped[:1].isupper():
        return True
    return False


def _block_text(lines: list[dict[str, Any]]) -> str:
    line_texts: list[str] = []
    for line in lines:
        pieces = []
        for span in line.get("spans", []) or []:
            text = str(span.get("text") or "")
            if text:
                pieces.append(text)
        line_text = " ".join("".join(pieces).split())
        if line_text:
            line_texts.append(line_text)
    return _clean_pdf_text("\n".join(line_texts))


def _clean_pdf_text(text: Any) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", value)
    return value.strip()


def _collect_font_sizes(raw_blocks: list[dict[str, Any]]) -> list[float]:
    sizes: list[float] = []
    for block in raw_blocks:
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                try:
                    size = float(span.get("size", 0))
                except (TypeError, ValueError):
                    continue
                if size > 0:
                    sizes.append(size)
    return sizes


def _normalize_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def _merge_bboxes(bboxes: Any) -> list[float] | None:
    valid = [_normalize_bbox(bbox) for bbox in bboxes]
    valid = [bbox for bbox in valid if bbox]
    if not valid:
        return None
    return [
        round(min(b[0] for b in valid), 2),
        round(min(b[1] for b in valid), 2),
        round(max(b[2] for b in valid), 2),
        round(max(b[3] for b in valid), 2),
    ]


def _limit_text(text: Any, max_len: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len].rstrip() + "..."
