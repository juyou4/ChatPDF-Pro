"""
ODL (OpenDataLoader PDF) 解析服务 — 入库前去脏层

职责
----
- 调用 opendataloader_pdf.convert() 把 PDF 解析为结构化 JSON
- 按元素类型过滤脏块，仅保留对检索有价值的内容
- 表格转 Markdown，保留数值结构
- 返回与现有 pages 格式完全兼容的数据结构，可直接替换 pdfplumber/pymupdf 输出

两条线分工（架构约定）
-----------------------
- ODL（本模块）：入库前去脏，文本清洗与结构提取
  - 剔除：header / footer / caption / image
  - 保留：paragraph / heading / list / text block / table
- DocLayout-YOLO（layout_service.py）：速览功能的图表 bbox 收紧，与本模块完全解耦

安装要求
--------
pip install opendataloader-pdf    # 包含已打包的 JAR
Java 11+ 需在 PATH 中可用

注意：若从源码克隆（opendataloader-pdf 目录），需先 mvn package 再 pip install -e。
推荐直接 pip install opendataloader-pdf 从 PyPI 获取含 JAR 的发行版。
"""
import html
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from services.document_geometry import visual_geometry

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 元素类型分类
# ─────────────────────────────────────────────────────────────────────────────

# 直接跳过，不进入文本索引（脏块）
_SKIP_TYPES = frozenset({"header", "footer", "image", "caption"})
_SOFT_KEEP_TYPES = frozenset()

# 直接保留为纯文本的类型
_TEXT_TYPES = frozenset({"paragraph", "list item"})

# 需要递归处理子元素的容器类型
_CONTAINER_TYPES = frozenset({"text block", "list"})

# 标题：保留并加层级前缀，利于结构感知分块
_HEADING_TYPE = "heading"

# 表格：转换为 Markdown 管道表格
_TABLE_TYPE = "table"
_CAPTION_TYPE = "caption"

_TABLE_REF_RE = re.compile(r"(?:Table|TABLE|表)\s*\.?\s*(\d+(?:\.\d+)?)")

# ─────────────────────────────────────────────────────────────────────────────
# 可用性检测（带缓存）
# ─────────────────────────────────────────────────────────────────────────────

_odl_available: Optional[bool] = None  # None = 未检测


def is_odl_available() -> bool:
    """检测 opendataloader_pdf 和 Java 是否同时可用（结果缓存，只检测一次）"""
    global _odl_available
    if _odl_available is not None:
        return _odl_available

    # 1. 检测 Python 包是否已安装
    try:
        import opendataloader_pdf  # noqa: F401
    except ImportError:
        logger.info("[ODL] opendataloader_pdf 未安装，跳过 ODL 解析路径")
        _odl_available = False
        return False

    # 2. 检测 Java 是否在 PATH 中
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.info("[ODL] Java 返回非零退出码，跳过 ODL 解析路径")
            _odl_available = False
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.info(f"[ODL] Java 不可用（{e}），跳过 ODL 解析路径")
        _odl_available = False
        return False

    logger.info("[ODL] opendataloader_pdf + Java 均可用，ODL 解析路径已启用")
    _odl_available = True
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 元素文本提取
# ─────────────────────────────────────────────────────────────────────────────

def _extract_table_matrix(element: dict) -> list[list[str]]:
    """提取 table 元素中的二维单元格文本矩阵，并展开 row/column span。"""
    rows_data = element.get("rows", [])
    if not rows_data:
        return []

    matrix: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}

    for row_index, row_obj in enumerate(rows_data, start=1):
        if not isinstance(row_obj, dict):
            continue
        cells = row_obj.get("cells", [])
        row_cells: dict[int, str] = {
            col_idx: text
            for (pending_row, col_idx), text in list(pending.items())
            if pending_row == row_index
        }
        for key in [key for key in pending if key[0] == row_index]:
            pending.pop(key, None)

        cursor = 1
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            explicit_col = _normalize_optional_int(cell.get("column number")) or _normalize_optional_int(cell.get("col idx"))
            col_idx = explicit_col or cursor
            while col_idx in row_cells:
                col_idx += 1

            text = _extract_table_cell_text(cell)
            row_span = _normalize_optional_int(cell.get("row span")) or _normalize_optional_int(cell.get("row_span")) or 1
            col_span = _normalize_optional_int(cell.get("column span")) or _normalize_optional_int(cell.get("col_span")) or 1

            for dx in range(max(1, col_span)):
                row_cells[col_idx + dx] = text
            for dy in range(1, max(1, row_span)):
                for dx in range(max(1, col_span)):
                    pending[(row_index + dy, col_idx + dx)] = text
            cursor = col_idx + max(1, col_span)

        if row_cells:
            max_col = max(row_cells)
            matrix.append([row_cells.get(col_idx, "") for col_idx in range(1, max_col + 1)])

    return matrix


def _detect_odl_header_depth(matrix: list[list[str]]) -> int:
    if len(matrix) < 2:
        return 1
    first = [str(cell or "").strip() for cell in matrix[0]]
    second = [str(cell or "").strip() for cell in matrix[1]]
    first_values = [cell for cell in first if cell]
    second_values = [cell for cell in second if cell]
    if not first_values or not second_values:
        return 1
    repeated_parent = len(set(first_values)) < len(first_values)
    second_numeric = sum(
        1
        for cell in second_values
        if re.fullmatch(r"[-+−]?\d+(?:[.,]\d+)?%?", cell)
    )
    second_numeric_ratio = second_numeric / len(second_values)
    return 2 if repeated_parent and second_numeric_ratio < 0.45 else 1


def _odl_column_header_paths(matrix: list[list[str]]) -> list[str]:
    if not matrix:
        return []
    header_depth = max(1, min(_detect_odl_header_depth(matrix), len(matrix)))
    header_rows = matrix[:header_depth]
    max_cols = max(len(row) for row in matrix)
    paths: list[str] = []
    for col_idx in range(max_cols):
        parts: list[str] = []
        for row in header_rows:
            part = str(row[col_idx] if col_idx < len(row) else "").strip()
            if part and (not parts or parts[-1] != part):
                parts.append(part)
        paths.append(" ".join(parts).strip() or f"Column {col_idx + 1}")
    return paths


def _extract_table_cell_text(cell: dict) -> str:
    """抽取单元格文本，兼容 kids / content 两种 ODL 结构。"""
    if not isinstance(cell, dict):
        return ""

    kids = cell.get("kids", [])
    if kids:
        return " ".join(_extract_element_text(k) for k in kids).strip()
    return str(cell.get("content", "") or "").strip()


def _table_to_markdown(element: dict) -> str:
    """把 ODL table 元素转换为 Markdown 管道表格"""
    matrix = _extract_table_matrix(element)
    if not matrix:
        return ""

    row_texts = []
    header_done = False

    for row in matrix:
        cell_texts = [cell.replace("|", "\\|") for cell in row]
        row_texts.append("| " + " | ".join(cell_texts) + " |")

        # 第一行之后插入分隔行
        if not header_done and row_texts:
            separator = "| " + " | ".join("---" for _ in cell_texts) + " |"
            row_texts.append(separator)
            header_done = True

    return "\n".join(row_texts)


def _table_to_html(element: dict) -> str:
    """把 ODL table 元素转换为简洁 HTML，供 typed metadata 保留。"""
    matrix = _extract_table_matrix(element)
    if not matrix:
        return ""

    lines = ["<table>"]
    for row_idx, row in enumerate(matrix):
        tag = "th" if row_idx == 0 else "td"
        cells = "".join(f"<{tag}>{html.escape(cell or '')}</{tag}>" for cell in row)
        lines.append(f"  <tr>{cells}</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _extract_table_cell_text(cell: dict) -> str:
    """提取 table cell 的纯文本。"""
    if not isinstance(cell, dict):
        return ""
    kids = cell.get("kids", [])
    if kids:
        return " ".join(_extract_element_text(k) for k in kids).strip()
    return (cell.get("content") or "").strip()


def _normalize_optional_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        int_value = int(value)
        return int_value if int_value > 0 else None
    return None


def _normalize_bbox(value) -> list:
    if not isinstance(value, (list, tuple)):
        return []
    bbox = []
    for item in value:
        if isinstance(item, (int, float)):
            bbox.append(item)
    return bbox if len(bbox) >= 4 else []


def _merge_bboxes(bboxes: list[list]) -> list:
    valid = [bbox for bbox in bboxes if isinstance(bbox, list) and len(bbox) >= 4]
    if not valid:
        return []
    xs0 = [bbox[0] for bbox in valid]
    ys0 = [bbox[1] for bbox in valid]
    xs1 = [bbox[2] for bbox in valid]
    ys1 = [bbox[3] for bbox in valid]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def _build_evidence_unit_id(
    bundle_key: str,
    unit_type: str,
    source_label: str,
    row_idx: int,
    col_idx: Optional[int] = None,
) -> str:
    parts = [bundle_key or "bundle", unit_type, source_label or "source", f"r{row_idx}"]
    if col_idx is not None:
        parts.append(f"c{col_idx}")
    return "::".join(parts)


def _build_table_evidence_units(
    bundle_key: str,
    table_id: str,
    table_caption: str,
    table_header: str,
    grouped_items: list[tuple[int, dict, str]],
    page_sizes: dict[int, list[float]] | None = None,
) -> list[dict]:
    evidence_units: list[dict] = []
    page_sizes = page_sizes or {}

    for _, element, _caption in grouped_items:
        if not isinstance(element, dict):
            continue

        matrix = _extract_table_matrix(element)
        header_paths = _odl_column_header_paths(matrix)
        header_depth = _detect_odl_header_depth(matrix)
        page_num = _normalize_optional_int(element.get("page number")) or 1
        source_id = _normalize_optional_int(element.get("id"))
        source_label = f"source:{source_id}" if source_id is not None else f"page:{page_num}"
        table_bbox = _normalize_bbox(element.get("bounding box") or element.get("bbox"))
        page_size = page_sizes.get(page_num, [])

        for row_index, row_obj in enumerate(element.get("rows", []), start=1):
            if not isinstance(row_obj, dict):
                continue

            row_idx = _normalize_optional_int(row_obj.get("row number")) or _normalize_optional_int(row_obj.get("row idx")) or row_index
            explicit_row_bbox = _normalize_bbox(row_obj.get("bounding box") or row_obj.get("bbox"))
            cell_bboxes: list[list] = []
            cell_units: list[dict] = []
            cell_texts: list[str] = []
            is_header_row = row_index == 1

            for col_index, cell in enumerate(row_obj.get("cells", []), start=1):
                if not isinstance(cell, dict):
                    continue

                col_idx = _normalize_optional_int(cell.get("column number")) or _normalize_optional_int(cell.get("col idx")) or col_index
                cell_text = _extract_table_cell_text(cell)
                cell_texts.append(cell_text)

                cell_bbox = _normalize_bbox(cell.get("bounding box") or cell.get("bbox"))
                cell_geometry = visual_geometry(
                    cell_bbox,
                    coordinate_space="pdf_bottom_left_points",
                    page_size=page_size,
                )
                cell_crop_eligible = bool(cell_geometry.get("visual_bbox"))
                if cell_crop_eligible:
                    cell_bboxes.append(cell_bbox)

                cell_row_idx = _normalize_optional_int(cell.get("row number")) or row_idx
                cell_row_span = _normalize_optional_int(cell.get("row span")) or _normalize_optional_int(cell.get("row_span"))
                cell_col_span = _normalize_optional_int(cell.get("column span")) or _normalize_optional_int(cell.get("col_span"))
                header_path = header_paths[col_idx - 1] if col_idx and col_idx - 1 < len(header_paths) else f"Column {col_idx}"

                cell_units.append({
                    "evidence_unit_id": _build_evidence_unit_id(bundle_key, "table_cell", source_label, row_idx, col_idx),
                    "evidence_unit_type": "table_cell",
                    "table_bundle_id": bundle_key,
                    "table_id": table_id,
                    "table_caption": table_caption,
                    "table_header": table_header,
                    "source_id": source_id,
                    "page": page_num,
                    "row_idx": cell_row_idx,
                    "row_number": cell_row_idx,
                    "col_idx": col_idx,
                    "column_number": col_idx,
                    "col_id": header_path,
                    "column_header": header_path,
                    "header_path": header_path,
                    "row_span": cell_row_span,
                    "col_span": cell_col_span,
                    "bbox": cell_bbox,
                    "bounding_box": cell_bbox,
                    **cell_geometry,
                    "geometry_source": "odl_cell" if cell_crop_eligible else "table_fallback",
                    "visual_crop_eligible": cell_crop_eligible,
                    "cell_text": cell_text,
                    "content": cell_text,
                    "is_header_row": row_index <= header_depth,
                    "source": "odl",
                })

            row_text = " | ".join(text for text in cell_texts if text).strip()
            explicit_row_geometry = visual_geometry(
                explicit_row_bbox,
                coordinate_space="pdf_bottom_left_points",
                page_size=page_size,
            )
            cells_cover_row = bool(cell_units) and len(cell_bboxes) == len(cell_units)
            inferred_row_bbox = _merge_bboxes(cell_bboxes) if cells_cover_row else []
            row_bbox = (
                explicit_row_bbox if explicit_row_geometry.get("visual_bbox")
                else inferred_row_bbox or table_bbox
            )
            row_geometry = visual_geometry(
                row_bbox,
                coordinate_space="pdf_bottom_left_points",
                page_size=page_size,
            )
            row_crop_eligible = bool(explicit_row_geometry.get("visual_bbox")) or (
                cells_cover_row and bool(row_geometry.get("visual_bbox"))
            )
            row_id = next((text for text in cell_texts if text), "")
            row_unit = {
                "evidence_unit_id": _build_evidence_unit_id(bundle_key, "table_row", source_label, row_idx),
                "evidence_unit_type": "table_row",
                "table_bundle_id": bundle_key,
                "table_id": table_id,
                "table_caption": table_caption,
                "table_header": table_header,
                "source_id": source_id,
                "page": page_num,
                "row_idx": row_idx,
                "row_number": row_idx,
                "bbox": row_bbox,
                "bounding_box": row_bbox,
                **row_geometry,
                "geometry_source": "odl_row" if explicit_row_geometry.get("visual_bbox") else (
                    "odl_cells_merged" if row_crop_eligible else "table_fallback"
                ),
                "visual_crop_eligible": row_crop_eligible,
                "row_id": row_id,
                "row_text": row_text,
                "row_numbers": " ".join(text for text in cell_texts[1:] if text),
                "content": row_text,
                "is_header_row": row_index <= header_depth,
                "cell_count": len(cell_units),
                "cell_evidence_unit_ids": [
                    cell.get("evidence_unit_id")
                    for cell in cell_units
                    if cell.get("evidence_unit_id")
                ],
                "cell_evidence_units": cell_units,
                "source": "odl",
            }
            evidence_units.append(row_unit)

    return evidence_units


def _extract_table_reference(caption: str) -> str:
    """从 caption 中提取规范化的 Table X 引用。"""
    if not caption:
        return ""
    match = _TABLE_REF_RE.search(caption)
    if not match:
        return ""
    return f"Table {match.group(1)}"


def _normalize_table_bundle_key(value: str) -> str:
    sample = re.sub(r"\s+", " ", (value or "").strip().lower())
    return sample[:240]


def _build_caption_lookup(elements: list) -> tuple[dict[int, list[str]], dict[int, list[dict]]]:
    linked: dict[int, list[str]] = {}
    orphan_by_page: dict[int, list[dict]] = {}
    for element in elements:
        if not isinstance(element, dict) or element.get("type") != _CAPTION_TYPE:
            continue
        content = (element.get("content") or "").strip()
        if not content:
            continue
        linked_id = element.get("linked content id")
        if isinstance(linked_id, int):
            linked.setdefault(linked_id, []).append(content)
            continue
        page_num = int(element.get("page number", 1) or 1)
        orphan_by_page.setdefault(page_num, []).append({
            "content": content,
            "bounding_box": element.get("bounding box") or [],
        })
    return linked, orphan_by_page


def _pick_nearest_page_caption(table_element: dict, page_captions: list[dict]) -> str:
    if not page_captions:
        return ""
    table_bbox = table_element.get("bounding box") or []
    if len(table_bbox) < 4:
        return page_captions[0].get("content", "")

    table_bottom = float(table_bbox[1])
    best_content = ""
    best_distance = None
    for caption in page_captions:
        bbox = caption.get("bounding_box") or []
        if len(bbox) >= 4:
            distance = abs(table_bottom - float(bbox[3]))
        else:
            distance = 0.0
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_content = caption.get("content", "")
    return best_content


def _resolve_table_chain_key(
    element: dict,
    index_in_doc: int,
    tables_by_id: dict[int, dict],
    captions_by_linked_id: dict[int, list[str]],
    orphan_captions_by_page: dict[int, list[dict]],
) -> tuple[str, str]:
    current = element
    visited: set[int] = set()

    while True:
        prev_id = current.get("previous table id")
        if not isinstance(prev_id, int) or prev_id in visited or prev_id not in tables_by_id:
            break
        visited.add(prev_id)
        current = tables_by_id[prev_id]

    root_id = current.get("id")
    root_caption = ""
    if isinstance(root_id, int):
        root_caption = " ".join(captions_by_linked_id.get(root_id, [])).strip()
    if not root_caption:
        page_num = int(current.get("page number", 1) or 1)
        root_caption = _pick_nearest_page_caption(
            current,
            orphan_captions_by_page.get(page_num, []),
        )

    if isinstance(root_id, int):
        return f"id:{root_id}", root_caption
    if root_caption:
        return f"caption:{_normalize_table_bundle_key(root_caption)}", root_caption

    page_num = int(element.get("page number", 1) or 1)
    return f"page:{page_num}:table:{index_in_doc}", root_caption


def _build_structured_table_bundles(elements: list, page_sizes: dict[int, list[float]] | None = None) -> list[dict]:
    """从 ODL 顶层元素中提取并合并结构化表格 bundle。"""
    page_sizes = page_sizes or {}
    captions_by_linked_id, orphan_captions_by_page = _build_caption_lookup(elements)
    table_elements = [
        element for element in elements
        if isinstance(element, dict) and element.get("type") == _TABLE_TYPE
    ]
    if not table_elements:
        return []

    tables_by_id = {
        element.get("id"): element
        for element in table_elements
        if isinstance(element.get("id"), int)
    }
    grouped: dict[str, list[tuple[int, dict, str]]] = {}
    for index_in_doc, element in enumerate(table_elements):
        group_key, fallback_caption = _resolve_table_chain_key(
            element,
            index_in_doc,
            tables_by_id,
            captions_by_linked_id,
            orphan_captions_by_page,
        )
        element_id = element.get("id")
        caption = ""
        if isinstance(element_id, int):
            caption = " ".join(captions_by_linked_id.get(element_id, [])).strip()
        if not caption:
            caption = fallback_caption
        if not caption:
            page_num = int(element.get("page number", 1) or 1)
            caption = _pick_nearest_page_caption(
                element,
                orphan_captions_by_page.get(page_num, []),
            )
        grouped.setdefault(group_key, []).append((index_in_doc, element, caption))

    bundles = []
    for group_key, grouped_items in grouped.items():
        grouped_items.sort(key=lambda item: (int(item[1].get("page number", 1) or 1), item[0]))
        captions: list[str] = []
        pages: list[int] = []
        markdown_parts: list[str] = []
        html_parts: list[str] = []
        source_ids: list[int] = []
        bboxes: list[list[float]] = []
        headers: list[str] = []
        previous_ids: list[int] = []
        next_ids: list[int] = []

        for _, element, caption in grouped_items:
            markdown = _table_to_markdown(element)
            if not markdown.strip():
                continue
            html_table = _table_to_html(element)
            matrix = _extract_table_matrix(element)
            header_paths = _odl_column_header_paths(matrix)
            header = " | ".join(header_paths).strip() if header_paths else ""
            page_num = int(element.get("page number", 1) or 1)
            pages.append(page_num)
            markdown_parts.append(markdown)
            if html_table:
                html_parts.append(html_table)
            if caption and caption not in captions:
                captions.append(caption)
            if header and header not in headers:
                headers.append(header)
            if isinstance(element.get("id"), int):
                source_ids.append(int(element["id"]))
            bbox = element.get("bounding box") or []
            if bbox:
                bboxes.append(bbox)
            if isinstance(element.get("previous table id"), int):
                previous_ids.append(int(element["previous table id"]))
            if isinstance(element.get("next table id"), int):
                next_ids.append(int(element["next table id"]))

        if not markdown_parts:
            continue

        pages = sorted(set(pages))
        bundle_caption = captions[0] if captions else ""
        table_id = _extract_table_reference(bundle_caption)
        if not table_id:
            if source_ids:
                table_id = f"odl-table-{source_ids[0]}"
            elif pages:
                table_id = f"page-{pages[0]}-table"
            else:
                table_id = group_key

        table_body_markdown = "\n\n".join(markdown_parts)
        evidence_units = _build_table_evidence_units(
            group_key,
            table_id,
            bundle_caption,
            headers[0] if headers else "",
            grouped_items,
            page_sizes,
        )
        bundle_geometry = visual_geometry(
            bboxes[0] if bboxes else [],
            coordinate_space="pdf_bottom_left_points",
            page_size=page_sizes.get(pages[0], []) if pages else [],
        )
        row_cell_geometry_available = any(
            row.get("visual_crop_eligible") is True
            or any(
                cell.get("visual_crop_eligible") is True
                for cell in row.get("cell_evidence_units") or []
                if isinstance(cell, dict)
            )
            for row in evidence_units
            if isinstance(row, dict)
        )
        bundle = {
            "bundle_id": group_key,
            "evidence_unit_id": f"{group_key}::table_bundle",
            "table_id": table_id,
            "table_caption": bundle_caption,
            "table_header": headers[0] if headers else "",
            "table_body_markdown": table_body_markdown,
            "table_markdown": table_body_markdown,
            "html_table": "\n".join(html_parts),
            "table_footnote": "",
            "page_start": pages[0] if pages else 1,
            "page_end": pages[-1] if pages else 1,
            "pages": pages,
            "bounding_box": bboxes[0] if bboxes else [],
            "bounding_boxes": bboxes,
            **bundle_geometry,
            "row_cell_geometry_available": row_cell_geometry_available,
            "source_ids": sorted(set(source_ids)),
            "previous_table_ids": sorted(set(previous_ids)),
            "next_table_ids": sorted(set(next_ids)),
            "source": "odl",
            "evidence_units": evidence_units,
        }
        bundle_lines = ["[Structured Table Bundle]"]
        if bundle_caption:
            bundle_lines.extend(["", bundle_caption])
        if table_id:
            bundle_lines.extend(["", "[Table ID]", table_id])
        if headers:
            bundle_lines.extend(["", "[Header]", headers[0]])
        if table_body_markdown:
            bundle_lines.extend(["", "[Body]", table_body_markdown])
        bundle["bundle_text"] = "\n".join(bundle_lines).strip()
        bundles.append(bundle)

    bundles.sort(key=lambda bundle: (bundle.get("page_start", 1), bundle.get("table_id", "")))
    return bundles


def _attach_structured_bundles_to_pages(
    pages: list[dict],
    structured_table_bundles: list[dict],
) -> tuple[list[dict], str]:
    """把结构化 table bundles 挂回页面，并把 bundle 文本注入 page/full_text。"""
    if not pages:
        return [], ""

    bundles_by_page: dict[int, list[dict]] = {}
    for bundle in structured_table_bundles or []:
        if not isinstance(bundle, dict):
            continue
        page_numbers = bundle.get("pages") or []
        if not isinstance(page_numbers, list):
            page_numbers = []
        fallback_page = bundle.get("page_start", 1)
        if not page_numbers and isinstance(fallback_page, int):
            page_numbers = [fallback_page]
        for page_num in page_numbers:
            if not isinstance(page_num, int) or page_num <= 0:
                continue
            bundles_by_page.setdefault(page_num, []).append(dict(bundle))

    full_text_parts: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_num = int(page.get("page", 1) or 1)
        page_bundles = bundles_by_page.get(page_num, [])
        page["table_bundles"] = page_bundles

        page_text = page.get("text", page.get("content", "")) or ""
        for bundle in page_bundles:
            bundle_text = (bundle.get("bundle_text") or "").strip()
            table_body = (bundle.get("table_body_markdown") or "").strip()
            if not bundle_text:
                continue
            if table_body and table_body in page_text:
                page_text = page_text.replace(table_body, bundle_text, 1)
                continue
            if bundle_text not in page_text:
                page_text = f"{page_text.rstrip()}\n\n{bundle_text}".strip()
        page["text"] = page_text
        page["content"] = page_text
        if page_text.strip():
            full_text_parts.append(page_text)

    return pages, "\n\n".join(full_text_parts)


def _extract_element_text(element: dict) -> str:
    """
    递归提取单个元素的文本内容。

    过滤规则
    --------
    - header / footer / image → 返回空字符串（脏块）
    - paragraph / list item → 直接返回 content
    - heading → 返回带 # 前缀的 content（保留层级语义）
    - table → 转 Markdown
    - text block / list → 递归处理 kids
    - caption → 软保留，返回 content
    - 其他 → 尝试返回 content，没有则返回空
    """
    if not isinstance(element, dict):
        return ""

    elem_type = element.get("type", "")

    # 脏块：直接跳过
    if elem_type in _SKIP_TYPES:
        return ""

    # 软保留类型
    if elem_type in _SOFT_KEEP_TYPES:
        return element.get("content", "").strip()

    # 纯文本类型
    if elem_type in _TEXT_TYPES:
        return element.get("content", "").strip()

    # 标题：加 Markdown 前缀保留层级
    if elem_type == _HEADING_TYPE:
        level = element.get("heading level", 2)
        prefix = "#" * max(1, min(level, 6))
        content = element.get("content", "").strip()
        return f"{prefix} {content}" if content else ""

    # 表格 → Markdown
    if elem_type == _TABLE_TYPE:
        return _table_to_markdown(element)

    # 容器类型：递归 kids
    if elem_type in _CONTAINER_TYPES or "kids" in element:
        kids = element.get("kids", [])
        parts = [_extract_element_text(k) for k in kids]
        return "\n".join(p for p in parts if p.strip())

    # 兜底：直接取 content（如 textBlock 变体等）
    return element.get("content", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# 页面重建
# ─────────────────────────────────────────────────────────────────────────────

def _build_pages_from_elements(elements: list, total_pages: int) -> tuple[list, str]:
    """
    将 ODL 顶层元素列表按页号聚合，重建 pages 列表和 full_text。

    返回与现有 extract_text_from_pdf 兼容的格式：
    pages = [{"page": int, "text": str, "content": str, "source": "odl"}, ...]
    full_text = str
    """
    page_text_parts: dict[int, list[str]] = {}

    for element in elements:
        if not isinstance(element, dict):
            continue
        page_num = element.get("page number", 1)
        text = _extract_element_text(element)
        if text.strip():
            page_text_parts.setdefault(page_num, []).append(text)

    pages = []
    full_text_parts = []

    # 按页号顺序输出，确保空页也生成占位
    for page_num in range(1, total_pages + 1):
        parts = page_text_parts.get(page_num, [])
        page_text = "\n\n".join(parts)
        pages.append({
            "page": page_num,
            "text": page_text,
            "content": page_text,
            "source": "odl",
        })
        if page_text.strip():
            full_text_parts.append(page_text)

    full_text = "\n\n".join(full_text_parts)
    return pages, full_text


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def _pdf_page_sizes(pdf_path: str) -> dict[int, list[float]]:
    """Read PDF point dimensions once for ODL's bottom-left coordinates."""
    try:
        import fitz

        document = fitz.open(pdf_path)
        try:
            return {
                page_index + 1: [float(page.rect.width), float(page.rect.height)]
                for page_index, page in enumerate(document)
            }
        finally:
            document.close()
    except Exception as exc:
        logger.warning("[ODL] 无法读取 PDF 页面尺寸，视觉裁剪将跳过 ODL bbox: %s", exc)
        return {}


def parse_pdf_odl(pdf_path: str) -> Optional[dict]:
    """
    用 OpenDataLoader PDF 解析指定 PDF 文件，返回去脏后的结构化数据。

    参数
    ----
    pdf_path : str
        已保存到磁盘的 PDF 文件绝对路径

    返回
    ----
    与 extract_text_from_pdf() 输出兼容的 dict，包含：
    {
        "full_text": str,
        "pages": [...],
        "total_pages": int,
        "extraction_method": "odl",
        "odl_element_count": int,   # 原始元素总数（含被过滤的）
        "odl_kept_count": int,       # 实际保留到索引的元素数
    }
    若 ODL 不可用或解析失败，返回 None（由调用方降级到 pdfplumber）。
    """
    if not is_odl_available():
        return None

    import opendataloader_pdf  # noqa: F401（已在 is_odl_available 确认可用）

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 调用 ODL 转换
            opendataloader_pdf.convert(
                input_path=pdf_path,
                output_dir=tmpdir,
                format="json",
                reading_order="xycut",   # XY-Cut++ 修正多栏阅读顺序
                quiet=True,
            )

            # 读取 JSON 输出
            pdf_stem = Path(pdf_path).stem
            json_path = Path(tmpdir) / f"{pdf_stem}.json"
            if not json_path.exists():
                logger.warning(f"[ODL] 未找到输出 JSON: {json_path}")
                return None

            with open(json_path, encoding="utf-8") as f:
                doc = json.load(f)

        elements: list = doc.get("kids", [])
        total_pages: int = doc.get("number of pages", 1)

        if not elements:
            logger.warning("[ODL] 解析结果为空，降级到 pdfplumber")
            return None

        # 统计被过滤的硬脏块数
        raw_count = len(elements)
        dirty_count = sum(
            1 for e in elements
            if isinstance(e, dict) and e.get("type") in _SKIP_TYPES
        )
        soft_keep_count = sum(
            1 for e in elements
            if isinstance(e, dict) and e.get("type") in _SOFT_KEEP_TYPES
        )

        structured_table_bundles = _build_structured_table_bundles(elements, _pdf_page_sizes(pdf_path))
        # 重建页面
        pages, full_text = _build_pages_from_elements(elements, total_pages)
        pages, full_text = _attach_structured_bundles_to_pages(pages, structured_table_bundles)

        if not full_text.strip():
            logger.warning("[ODL] 提取文本为空，降级到 pdfplumber")
            return None

        kept_count = raw_count - dirty_count
        logger.info(
            f"[ODL] 解析完成: {total_pages} 页, "
            f"{raw_count} 个元素 → 保留 {kept_count} 个（硬过滤 {dirty_count} 个, 软保留 caption {soft_keep_count} 个）, "
            f"全文 {len(full_text)} 字符"
        )

        return {
            "full_text": full_text,
            "pages": pages,
            "total_pages": total_pages,
            "extraction_method": "odl",
            "odl_element_count": raw_count,
            "odl_kept_count": kept_count,
            "odl_soft_kept_caption_count": soft_keep_count,
            # 以下字段保持与 extract_text_from_pdf 输出格式一致
            "images": [],
            "image_count": 0,
            "ocr_used": False,
            "ocr_backend": None,
            "extraction_quality": "odl_clean",
            "structured_table_bundles": structured_table_bundles,
            "structured_table_count": len(structured_table_bundles),
        }

    except Exception as e:
        logger.warning(f"[ODL] 解析失败，降级到 pdfplumber: {e}")
        return None
