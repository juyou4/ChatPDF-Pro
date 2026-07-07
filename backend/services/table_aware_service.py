"""
表格感知服务 — PDF 表格检测与 Markdown 转换

参考 ragflow rag/nlp/__init__.py 的 tokenize_table 和 attach_media_context 策略：
- 使用 PyMuPDF find_tables() 检测页面中的表格区域
- 将表格转换为结构化 Markdown 格式
- 替换原始页面文本中的表格区域，保留上下文
- 自动检测表格标题（Table X / 表 X）并作为前缀

设计：
- 表格转 Markdown 后，structure_aware_split 的 _find_protected_regions 会自动识别
  并将其作为受保护区域，不会被分块切割
- 表格 Markdown 前缀包含 [TABLE] 标记，便于检索时识别
"""
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _clean_table_cell(value) -> str:
    text = str(value) if value is not None else ""
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _normalize_bundle_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9._:-]+", "-", str(value or "").strip().lower()).strip("-")
    return key or "table"


def _extract_table_id_from_caption(caption: str) -> str:
    match = re.search(r"\b(?:Table|TABLE|表)\s*\.?\s*([A-Za-z0-9IVXLC]+(?:\.\d+)?)", caption or "", re.IGNORECASE)
    if match:
        return f"Table {match.group(1)}"
    return ""


def _normalize_table_data(table_data: list) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table_data or []:
        if not isinstance(row, (list, tuple)):
            row = [row]
        cells = [_clean_table_cell(cell) for cell in row]
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    max_cols = max(len(row) for row in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append("")
    return rows


def _normalize_cell_bboxes(raw_bboxes, row_count: int, col_count: int) -> list[list[list]]:
    if row_count <= 0 or col_count <= 0:
        return []

    def _box(value) -> list:
        return _normalize_bbox(value)

    matrix: list[list[list]] = [[[] for _ in range(col_count)] for _ in range(row_count)]
    if isinstance(raw_bboxes, list) and raw_bboxes and all(isinstance(row, list) for row in raw_bboxes):
        # Already matrix-shaped.
        if len(raw_bboxes) == row_count and any(isinstance(cell, (list, tuple)) for row in raw_bboxes for cell in row):
            for r_idx, row in enumerate(raw_bboxes[:row_count]):
                for c_idx, cell in enumerate(row[:col_count]):
                    matrix[r_idx][c_idx] = _box(cell)
            return matrix
        # Flat row-major list.
        if len(raw_bboxes) >= row_count * col_count:
            for r_idx in range(row_count):
                for c_idx in range(col_count):
                    matrix[r_idx][c_idx] = _box(raw_bboxes[r_idx * col_count + c_idx])
            return matrix
    return []


def _is_numericish_cell(value: str) -> bool:
    text = _clean_table_cell(value)
    if not text:
        return False
    return bool(re.fullmatch(r"[-+−]?\d+(?:[.,]\d+)?%?", text.replace(",", "")))


def _row_numeric_ratio(row: list[str]) -> float:
    cells = [_clean_table_cell(cell) for cell in row]
    cells = [cell for cell in cells if cell]
    if not cells:
        return 0.0
    return sum(1 for cell in cells if _is_numericish_cell(cell)) / len(cells)


def _detect_header_row_count(rows: list[list[str]]) -> int:
    """Best-effort header depth for compact scientific tables.

    PyMuPDF does not expose rowspan/colspan, so merged parent headers often show
    up as blanks in the first row. Treat a short non-numeric second row as a
    child-header row, but keep ordinary ``Method | All`` tables at depth 1.
    """
    if len(rows) < 2:
        return 1 if rows else 0
    first = rows[0]
    second = rows[1]
    first_nonempty = [_clean_table_cell(cell) for cell in first if _clean_table_cell(cell)]
    second_nonempty = [_clean_table_cell(cell) for cell in second if _clean_table_cell(cell)]
    if not first_nonempty:
        return 1
    first_has_blank = any(not _clean_table_cell(cell) for cell in first)
    first_numeric_ratio = _row_numeric_ratio(first)
    second_numeric_ratio = _row_numeric_ratio(second)
    headerish_second = second_nonempty and second_numeric_ratio < 0.45
    if first_has_blank and headerish_second:
        return 2
    if first_numeric_ratio < 0.15 and headerish_second and len(first_nonempty) <= max(2, len(first) // 2):
        return 2
    return 1


def _expand_table_header_paths(rows: list[list[str]], header_row_count: Optional[int] = None) -> list[str]:
    """Flatten one or two header rows into stable column header paths."""
    if not rows:
        return []
    header_depth = max(1, min(int(header_row_count or _detect_header_row_count(rows) or 1), len(rows)))
    header_rows = rows[:header_depth]
    max_cols = max(len(row) for row in rows)
    propagated_rows: list[list[str]] = []
    for row_idx, row in enumerate(header_rows):
        propagated: list[str] = []
        last = ""
        for col_idx in range(max_cols):
            cell = _clean_table_cell(row[col_idx] if col_idx < len(row) else "")
            if cell:
                last = cell
                propagated.append(cell)
                continue
            # Parent header rows use horizontal propagation for merged cells.
            # The final header row keeps blanks so an upper-level title can stand alone.
            propagated.append(last if row_idx < header_depth - 1 else "")
        propagated_rows.append(propagated)

    headers: list[str] = []
    for col_idx in range(max_cols):
        parts: list[str] = []
        for row in propagated_rows:
            part = _clean_table_cell(row[col_idx] if col_idx < len(row) else "")
            if part and (not parts or parts[-1] != part):
                parts.append(part)
        headers.append(" ".join(parts).strip() or f"Column {col_idx + 1}")
    return headers


def _build_table_evidence_units(
    rows: list[list[str]],
    *,
    bundle_id: str,
    table_id: str,
    caption: str,
    page_num: int,
    bbox: list,
    cell_bboxes: Optional[list[list[list]]] = None,
    source: str,
) -> list[dict]:
    evidence_units: list[dict] = []
    header_row_count = _detect_header_row_count(rows)
    header_paths = _expand_table_header_paths(rows, header_row_count)
    for row_idx, row in enumerate(rows, start=1):
        row_text = " | ".join(cell for cell in row).strip()
        if not row_text:
            continue
        row_id = next((_clean_table_cell(cell) for cell in row if _clean_table_cell(cell)), f"row {row_idx}")
        cell_units: list[dict] = []
        row_cell_bboxes: list[list] = []
        for col_idx, cell in enumerate(row, start=1):
            header_path = header_paths[col_idx - 1] if col_idx - 1 < len(header_paths) else f"Column {col_idx}"
            cell_bbox = []
            if cell_bboxes and row_idx - 1 < len(cell_bboxes) and col_idx - 1 < len(cell_bboxes[row_idx - 1]):
                cell_bbox = _normalize_bbox(cell_bboxes[row_idx - 1][col_idx - 1])
            if not cell_bbox:
                cell_bbox = bbox
            if cell_bbox:
                row_cell_bboxes.append(cell_bbox)
            cell_units.append({
                "evidence_unit_id": f"{bundle_id}::table_cell::r{row_idx}::c{col_idx}",
                "evidence_unit_type": "table_cell",
                "table_bundle_id": bundle_id,
                "table_id": table_id,
                "table_caption": caption,
                "table_header": " | ".join(header_paths).strip(),
                "page": page_num,
                "row_idx": row_idx,
                "row_number": row_idx,
                "col_idx": col_idx,
                "column_number": col_idx,
                "col_id": header_path,
                "column_header": header_path,
                "header_path": header_path,
                "row_span": 1,
                "col_span": 1,
                "bbox": cell_bbox,
                "bounding_box": cell_bbox,
                "cell_text": cell,
                "content": cell,
                "is_header_row": row_idx <= header_row_count,
                "source": source,
            })
        row_bbox = _merge_bboxes(row_cell_bboxes) or bbox
        evidence_units.append({
            "evidence_unit_id": f"{bundle_id}::table_row::r{row_idx}",
            "evidence_unit_type": "table_row",
            "table_bundle_id": bundle_id,
            "table_id": table_id,
            "table_caption": caption,
            "table_header": " | ".join(header_paths).strip(),
            "page": page_num,
            "row_idx": row_idx,
            "row_number": row_idx,
            "bbox": row_bbox,
            "bounding_box": row_bbox,
            "content": row_text,
            "row_id": row_id,
            "row_text": row_text,
            "raw_row_text": row_text,
            "row_numbers": " ".join(_clean_table_cell(cell) for cell in row[1:] if _clean_table_cell(cell)),
            "cell_count": len(row),
            "is_header_row": row_idx <= header_row_count,
            "cell_evidence_units": cell_units,
            "source": source,
        })
    return evidence_units


def build_structured_table_bundle(table_info: dict, index_in_doc: int = 1) -> dict:
    """Build a Chatpdf structured table bundle from PyMuPDF/native table data."""
    if not isinstance(table_info, dict):
        return {}
    rows = _normalize_table_data(table_info.get("data") or [])
    if not rows:
        return {}

    page_num = int(table_info.get("page") or 1)
    caption = _clean_table_cell(table_info.get("caption") or "")
    table_id = _extract_table_id_from_caption(caption) or f"pdf-native-page-{page_num}-table-{index_in_doc}"
    bundle_id = f"pdf-native:p{page_num}:table:{_normalize_bundle_key(table_id)}:{index_in_doc}"
    bbox_raw = table_info.get("bbox") or []
    bbox = [float(value) for value in bbox_raw[:4]] if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4 else []
    cell_bboxes = _normalize_cell_bboxes(table_info.get("cell_bboxes") or [], len(rows), max(len(row) for row in rows) if rows else 0)
    source = str(table_info.get("source") or "pymupdf_table").strip() or "pymupdf_table"
    table_markdown = str(table_info.get("markdown") or _table_to_markdown(rows, caption) or "").strip()
    header_paths = _expand_table_header_paths(rows)
    header = " | ".join(header_paths).strip() if header_paths else (" | ".join(rows[0]).strip() if rows else "")
    evidence_units = _build_table_evidence_units(
        rows,
        bundle_id=bundle_id,
        table_id=table_id,
        caption=caption,
        page_num=page_num,
        bbox=bbox,
        cell_bboxes=cell_bboxes,
        source=source,
    )
    if not evidence_units:
        return {}

    bundle_lines = ["[Structured Table Bundle]"]
    if caption:
        bundle_lines.extend(["", caption])
    if table_id:
        bundle_lines.extend(["", "[Table ID]", table_id])
    if header:
        bundle_lines.extend(["", "[Header]", header])
    if table_markdown:
        bundle_lines.extend(["", "[Body]", table_markdown])

    return {
        "bundle_id": bundle_id,
        "evidence_unit_id": f"{bundle_id}::table_bundle",
        "table_id": table_id,
        "table_caption": caption,
        "table_header": header,
        "table_body_markdown": table_markdown,
        "table_markdown": table_markdown,
        "html_table": "",
        "table_footnote": "",
        "page_start": page_num,
        "page_end": page_num,
        "pages": [page_num],
        "bounding_box": bbox,
        "bounding_boxes": [bbox] if bbox else [],
        "source_ids": [f"pdf-native:{page_num}:{index_in_doc}"],
        "previous_table_ids": [],
        "next_table_ids": [],
        "source": source,
        "evidence_units": evidence_units,
        "bundle_text": "\n".join(bundle_lines).strip(),
    }


def _merge_bboxes(bboxes: list[list]) -> list:
    valid = [box for box in bboxes if isinstance(box, list) and len(box) >= 4]
    if not valid:
        return []
    return [
        min(float(box[0]) for box in valid),
        min(float(box[1]) for box in valid),
        max(float(box[2]) for box in valid),
        max(float(box[3]) for box in valid),
    ]


def _bbox_horizontal_overlap_ratio(a: list, b: list) -> float:
    if not (isinstance(a, list) and isinstance(b, list) and len(a) >= 4 and len(b) >= 4):
        return 0.0
    left = max(float(a[0]), float(b[0]))
    right = min(float(a[2]), float(b[2]))
    overlap = max(0.0, right - left)
    width = max(1.0, min(abs(float(a[2]) - float(a[0])), abs(float(b[2]) - float(b[0]))))
    return overlap / width


def _bbox_vertical_gap(a: list, b: list) -> float:
    if not (isinstance(a, list) and isinstance(b, list) and len(a) >= 4 and len(b) >= 4):
        return 1_000_000.0
    if float(a[3]) < float(b[1]):
        return float(b[1]) - float(a[3])
    if float(b[3]) < float(a[1]):
        return float(a[1]) - float(b[3])
    return 0.0


def _normalize_bbox(value) -> list:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return []
    return []


_TABLE_CAPTION_LINE_RE = re.compile(
    r"\b(?:Table|TABLE|表)\s*\.?\s*[A-Za-z0-9IVXLC]+(?:\.\d+)?\s*[.:：]?\s*.+|\b(?:Table|TABLE|表)\s*\.?\s*[A-Za-z0-9IVXLC]+(?:\.\d+)?\b",
    re.IGNORECASE,
)


def extract_table_caption_candidates_from_text_dict(text_dict: dict, page_num: int) -> list[dict]:
    """Extract table-caption line candidates with PDF bboxes from PyMuPDF dict text.

    This is intentionally geometry-only and template-agnostic: any short line
    containing a Table/Table-like label can later be bound to the nearest table
    bbox by overlap and vertical distance.
    """
    if not isinstance(text_dict, dict):
        return []
    candidates: list[dict] = []
    seen: set[str] = set()
    for block in text_dict.get("blocks", []) or []:
        if not isinstance(block, dict):
            continue
        for line in block.get("lines", []) or []:
            if not isinstance(line, dict):
                continue
            spans = line.get("spans") or []
            text = _clean_table_cell("".join(str(span.get("text") or "") for span in spans if isinstance(span, dict)))
            if not text or len(text) > 220 or not _TABLE_CAPTION_LINE_RE.search(text):
                continue
            bboxes = [
                _normalize_bbox(span.get("bbox"))
                for span in spans
                if isinstance(span, dict) and _normalize_bbox(span.get("bbox"))
            ]
            bbox = _merge_bboxes(bboxes) if bboxes else _normalize_bbox(line.get("bbox") or block.get("bbox"))
            key = f"{page_num}:{text.casefold()}:{bbox}"
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "text": text,
                "caption": text,
                "table_caption": text,
                "table_id": _extract_table_id_from_caption(text),
                "page": int(page_num or 1),
                "bbox": bbox,
                "bounding_box": bbox,
            })
    return candidates


def _header_tokens(header: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9\u4e00-\u9fff]+", str(header or "").casefold())
        if token and len(token) >= 2
    }


def _headers_compatible(left: str, right: str) -> bool:
    left_tokens = _header_tokens(left)
    right_tokens = _header_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    return overlap >= max(1, min(len(left_tokens), len(right_tokens)) // 2)


def _is_pdf_native_generated_table_id(table_id: str) -> bool:
    return bool(re.match(r"^pdf-native-page-\d+-table-\d+$", str(table_id or "").strip()))


def _markdown_escape_cell(value: str) -> str:
    return _clean_table_cell(value).replace("|", "\\|")


def _bundle_rows_to_markdown(header: str, row_units: list[dict], caption: str = "") -> str:
    header_cells = [_markdown_escape_cell(cell) for cell in str(header or "").split("|")]
    header_cells = [cell for cell in header_cells if cell]
    lines: list[str] = []
    if caption:
        lines.extend([f"[TABLE] {caption}", ""])
    if header_cells:
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
    for row in row_units:
        if row.get("is_header_row"):
            continue
        row_text = _clean_table_cell(row.get("row_text") or row.get("content") or "")
        if not row_text:
            continue
        cells = [_markdown_escape_cell(cell) for cell in row_text.split("|")]
        if header_cells:
            while len(cells) < len(header_cells):
                cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines).strip()


def _rewrite_pdf_native_evidence_units(
    rows: list[dict],
    *,
    bundle_id: str,
    table_id: str,
    caption: str,
    header: str,
) -> list[dict]:
    rewritten: list[dict] = []
    row_idx = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_idx += 1
        row_copy = dict(row)
        row_copy.update({
            "evidence_unit_id": f"{bundle_id}::table_row::r{row_idx}",
            "table_bundle_id": bundle_id,
            "table_id": table_id,
            "table_caption": caption,
            "table_header": header,
            "row_idx": row_idx,
            "row_number": row_idx,
        })
        cell_units: list[dict] = []
        for col_idx, cell in enumerate(row.get("cell_evidence_units") or [], start=1):
            if not isinstance(cell, dict):
                continue
            cell_copy = dict(cell)
            cell_copy.update({
                "evidence_unit_id": f"{bundle_id}::table_cell::r{row_idx}::c{col_idx}",
                "table_bundle_id": bundle_id,
                "table_id": table_id,
                "table_caption": caption,
                "table_header": header,
                "row_idx": row_idx,
                "row_number": row_idx,
                "col_idx": col_idx,
                "column_number": col_idx,
            })
            cell_units.append(cell_copy)
        row_copy["cell_evidence_units"] = cell_units
        rewritten.append(row_copy)
    return rewritten


def _rebuild_pdf_native_bundle_text(bundle: dict) -> str:
    lines = ["[Structured Table Bundle]"]
    if bundle.get("table_caption"):
        lines.extend(["", bundle["table_caption"]])
    if bundle.get("table_id"):
        lines.extend(["", "[Table ID]", bundle["table_id"]])
    if bundle.get("table_header"):
        lines.extend(["", "[Header]", bundle["table_header"]])
    if bundle.get("table_body_markdown"):
        lines.extend(["", "[Body]", bundle["table_body_markdown"]])
    return "\n".join(lines).strip()


def _apply_caption_to_pdf_native_bundle(bundle: dict, caption: str) -> dict:
    caption = _clean_table_cell(caption)
    if not caption:
        return bundle
    updated = dict(bundle)
    table_id = _extract_table_id_from_caption(caption) or _clean_table_cell(updated.get("table_id") or "")
    header = _clean_table_cell(updated.get("table_header") or "")
    bundle_id = _clean_table_cell(updated.get("bundle_id") or f"pdf-native:{table_id}")
    updated["table_caption"] = caption
    updated["table_id"] = table_id
    updated["bundle_id"] = bundle_id
    updated["evidence_unit_id"] = f"{bundle_id}::table_bundle"
    rows = [row for row in updated.get("evidence_units") or [] if isinstance(row, dict)]
    updated["evidence_units"] = _rewrite_pdf_native_evidence_units(
        rows,
        bundle_id=bundle_id,
        table_id=table_id,
        caption=caption,
        header=header,
    )
    table_body = _bundle_rows_to_markdown(header, updated["evidence_units"], caption)
    if table_body:
        updated["table_body_markdown"] = table_body
        updated["table_markdown"] = table_body
    updated["bundle_text"] = _rebuild_pdf_native_bundle_text(updated)
    return updated


def bind_nearest_table_captions(
    bundles: list[dict],
    caption_candidates: list[dict],
    *,
    max_vertical_gap: float = 90.0,
    min_horizontal_overlap: float = 0.25,
) -> list[dict]:
    """Bind separated table captions to nearest table bundles by page geometry."""
    if not bundles or not caption_candidates:
        return bundles or []
    candidates = [c for c in caption_candidates if isinstance(c, dict) and _clean_table_cell(c.get("caption") or c.get("text") or c.get("table_caption") or "")]
    if not candidates:
        return bundles

    used: set[int] = set()
    rebound: list[dict] = []
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        caption = _clean_table_cell(bundle.get("table_caption") or "")
        table_id = _clean_table_cell(bundle.get("table_id") or "")
        if caption and not _is_pdf_native_generated_table_id(table_id):
            rebound.append(bundle)
            continue

        page = int(bundle.get("page_start") or (bundle.get("pages") or [1])[0] or 1)
        bbox = _normalize_bbox(bundle.get("bounding_box") or [])
        best: tuple[float, int, dict] | None = None
        for idx, candidate in enumerate(candidates):
            if idx in used:
                continue
            if int(candidate.get("page") or page) != page:
                continue
            cap_bbox = _normalize_bbox(candidate.get("bbox") or candidate.get("bounding_box") or [])
            if not bbox or not cap_bbox:
                continue
            overlap = _bbox_horizontal_overlap_ratio(bbox, cap_bbox)
            if overlap < min_horizontal_overlap:
                continue
            gap = _bbox_vertical_gap(bbox, cap_bbox)
            if gap > max_vertical_gap:
                continue
            # Captions immediately above/below a table are best; higher overlap
            # breaks ties so multi-column pages bind to the right table.
            score = gap - overlap * 12.0
            if best is None or score < best[0]:
                best = (score, idx, candidate)
        if best is None:
            rebound.append(bundle)
            continue
        _score, candidate_idx, candidate = best
        used.add(candidate_idx)
        rebound.append(_apply_caption_to_pdf_native_bundle(
            bundle,
            str(candidate.get("caption") or candidate.get("text") or candidate.get("table_caption") or ""),
        ))
    return rebound


def merge_pdf_native_structured_table_bundles(bundles: list[dict]) -> list[dict]:
    """Merge pdf_native table bundles across pages and bind continuation captions.

    RAGFlow's parser keeps table chains together before chunking. PyMuPDF only
    gives per-page tables, so this applies a conservative continuation rule:
    same normalized Table ID always merges; otherwise an uncaptioned table on the
    next page may inherit the previous table only when headers and x-overlap match.
    """
    valid = [dict(bundle) for bundle in bundles or [] if isinstance(bundle, dict)]
    if not valid:
        return []
    valid.sort(key=lambda item: (int(item.get("page_start") or 1), str(item.get("bundle_id") or "")))

    merged: list[dict] = []
    for bundle in valid:
        caption = _clean_table_cell(bundle.get("table_caption") or "")
        table_id = _clean_table_cell(bundle.get("table_id") or "")
        header = _clean_table_cell(bundle.get("table_header") or "")
        page_start = int(bundle.get("page_start") or (bundle.get("pages") or [1])[0] or 1)
        current_bbox = bundle.get("bounding_box") or []

        target_idx: Optional[int] = None
        for idx in range(len(merged) - 1, -1, -1):
            prev = merged[idx]
            prev_table_id = _clean_table_cell(prev.get("table_id") or "")
            prev_header = _clean_table_cell(prev.get("table_header") or "")
            prev_page_end = int(prev.get("page_end") or page_start)
            same_explicit_id = (
                table_id
                and prev_table_id
                and table_id == prev_table_id
                and not _is_pdf_native_generated_table_id(table_id)
            )
            continuation = (
                not caption
                and _is_pdf_native_generated_table_id(table_id)
                and page_start <= prev_page_end + 1
                and _headers_compatible(prev_header, header)
                and _bbox_horizontal_overlap_ratio(prev.get("bounding_box") or [], current_bbox) >= 0.45
            )
            if same_explicit_id or continuation:
                target_idx = idx
                break
        if target_idx is None:
            merged.append(bundle)
            continue

        target = merged[target_idx]
        inherited_caption = _clean_table_cell(target.get("table_caption") or caption)
        inherited_table_id = _clean_table_cell(target.get("table_id") or table_id)
        inherited_header = _clean_table_cell(target.get("table_header") or header)
        target["table_caption"] = inherited_caption
        target["table_id"] = inherited_table_id
        target["table_header"] = inherited_header

        pages = sorted({
            int(page)
            for page in [*(target.get("pages") or []), *(bundle.get("pages") or [])]
            if isinstance(page, int) and page > 0
        })
        if not pages:
            pages = [page_start]
        target["pages"] = pages
        target["page_start"] = pages[0]
        target["page_end"] = pages[-1]
        target["bounding_boxes"] = [
            box
            for box in [*(target.get("bounding_boxes") or []), *(bundle.get("bounding_boxes") or [])]
            if isinstance(box, list) and len(box) >= 4
        ]
        target["bounding_box"] = _merge_bboxes(target["bounding_boxes"]) or target.get("bounding_box") or []
        target["source_ids"] = list(dict.fromkeys([*(target.get("source_ids") or []), *(bundle.get("source_ids") or [])]))
        target["previous_table_ids"] = list(dict.fromkeys([*(target.get("previous_table_ids") or []), bundle.get("table_id")]))
        target["next_table_ids"] = list(dict.fromkeys([*(target.get("next_table_ids") or []), *(bundle.get("next_table_ids") or [])]))

        target_rows = [row for row in target.get("evidence_units") or [] if isinstance(row, dict)]
        incoming_rows = [row for row in bundle.get("evidence_units") or [] if isinstance(row, dict)]
        if target_rows and incoming_rows and _headers_compatible(inherited_header, header):
            incoming_rows = [row for row in incoming_rows if not row.get("is_header_row")]
        combined_rows = target_rows + incoming_rows
        bundle_id = _clean_table_cell(target.get("bundle_id") or f"pdf-native:{inherited_table_id or target_idx}")
        target["bundle_id"] = bundle_id
        target["evidence_unit_id"] = f"{bundle_id}::table_bundle"
        target["evidence_units"] = _rewrite_pdf_native_evidence_units(
            combined_rows,
            bundle_id=bundle_id,
            table_id=inherited_table_id,
            caption=inherited_caption,
            header=inherited_header,
        )
        table_body = _bundle_rows_to_markdown(inherited_header, target["evidence_units"], inherited_caption)
        target["table_body_markdown"] = table_body
        target["table_markdown"] = table_body
        target["bundle_text"] = _rebuild_pdf_native_bundle_text(target)

    merged.sort(key=lambda bundle: (int(bundle.get("page_start") or 1), str(bundle.get("table_id") or "")))
    return merged


def _table_to_markdown(table_data: list, caption: str = "") -> str:
    """将表格数据转换为 Markdown 格式

    Args:
        table_data: PyMuPDF find_tables() 返回的二维列表
                    每个元素为一行，每行为一个单元格列表
        caption: 可选的表格标题

    Returns:
        Markdown 格式的表格字符串
    """
    if not table_data or not table_data[0]:
        return ""

    rows = []
    for row in table_data:
        # 清理单元格内容：去除换行、多余空白
        cells = []
        for cell in row:
            cell_text = str(cell) if cell is not None else ""
            cell_text = cell_text.replace("\n", " ").replace("|", "\\|").strip()
            cells.append(cell_text)
        rows.append(cells)

    if not rows:
        return ""

    # 统一列数（取最大列数）
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    # 构建 Markdown 表格
    lines = []

    if caption:
        lines.append(f"[TABLE] {caption}")
        lines.append("")

    # 表头
    lines.append("| " + " | ".join(rows[0]) + " |")
    # 分隔行
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    # 数据行
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _find_table_caption(page_text: str, table_bbox: tuple, page_height: float) -> str:
    """检测表格附近的标题文本（Table X / 表 X）

    搜索策略：在表格上方 3 行内查找 Table/表 编号模式

    Args:
        page_text: 页面完整文本
        table_bbox: 表格的 (x0, y0, x1, y1) 坐标
        page_height: 页面高度

    Returns:
        检测到的标题字符串，未找到返回空字符串
    """
    # 从页面文本中查找所有 Table/表 标题
    caption_pattern = re.compile(
        r'(?:Table|TABLE|表)\s*\.?\s*(\d+(?:\.\d+)?)\s*[.:：]?\s*(.*?)$',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(caption_pattern.finditer(page_text))
    if not matches:
        return ""

    # 取最接近表格 bbox 上方的标题
    # 简单策略：返回文本中最后一个在表格区域之前的 Table 标题
    table_y0 = table_bbox[1] if table_bbox else 0
    best_match = None

    for m in matches:
        # 粗略估算：匹配位置在文本中的比例 ≈ 在页面中的 Y 位置比例
        text_ratio = m.start() / max(len(page_text), 1)
        est_y = text_ratio * page_height

        if est_y <= table_y0 + 20:  # 允许小偏差
            best_match = m

    if best_match:
        num = best_match.group(1)
        desc = best_match.group(2).strip()
        if desc:
            return f"Table {num}: {desc}"
        return f"Table {num}"

    return ""


def _plain_table_text_to_markdown(region_text: str, caption: str = "") -> str:
    """把 bbox 裁出的表格区域文本包装成弱结构 Markdown。

    YOLO fallback 只能定位表格区域，不能恢复单元格结构；这里保留原始行，交给
    后续 structured/page-text bundle 回补逻辑继续处理。
    """
    lines = [re.sub(r"\s+", " ", line).strip() for line in (region_text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    parts = ["[TABLE_YOLO_FALLBACK]"]
    if caption:
        parts.append(caption)
    parts.extend(lines)
    return "\n".join(parts)


def _extract_pymupdf_cell_bboxes(table, row_count: int, col_count: int) -> list[list[list]]:
    """Best-effort cell bbox extraction from PyMuPDF table objects."""
    if row_count <= 0 or col_count <= 0:
        return []
    raw_cells = getattr(table, "cells", None)
    matrix = _normalize_cell_bboxes(raw_cells, row_count, col_count)
    if matrix:
        return matrix
    rows_obj = getattr(table, "rows", None)
    row_items = getattr(rows_obj, "rows", rows_obj)
    if not isinstance(row_items, (list, tuple)):
        return []
    raw_matrix: list[list] = []
    for row in row_items[:row_count]:
        cells = getattr(row, "cells", row)
        if not isinstance(cells, (list, tuple)):
            raw_matrix.append([])
            continue
        raw_matrix.append([
            getattr(cell, "bbox", cell)
            for cell in cells[:col_count]
        ])
    return _normalize_cell_bboxes(raw_matrix, row_count, col_count)


def _extract_tables_with_yolo_fallback(page, page_text: str, page_num: int) -> List[dict]:
    """当 PyMuPDF find_tables() 未命中时，用 DocLayout-YOLO 定位表格区域。

    该路径只作为弱 fallback：YOLO 给 bbox，文本仍由 PyMuPDF 从 bbox 内抽取。
    不做 OCR、不下载模型失败重试，任何异常都静默降级为空列表。
    """
    try:
        from services.layout_service import get_table_bboxes, pixel_bbox_to_page_pts
    except Exception as import_err:
        logger.debug(f"[Table] 页面 {page_num} YOLO fallback 不可用: {import_err}")
        return []

    try:
        import fitz

        zoom = 2.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        from PIL import Image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        detections = get_table_bboxes(image, conf=0.18)
    except Exception as detect_err:
        logger.debug(f"[Table] 页面 {page_num} YOLO 表格检测失败: {detect_err}")
        return []

    if not detections:
        return []

    results: List[dict] = []
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    seen: set[str] = set()
    for idx, det in enumerate(detections, start=1):
        bbox_px = det.get("bbox") if isinstance(det, dict) else None
        if not isinstance(bbox_px, list) or len(bbox_px) < 4:
            continue
        try:
            bbox_pts = pixel_bbox_to_page_pts(
                bbox_px,
                image.width,
                image.height,
                page_width,
                page_height,
            )
            rect = fitz.Rect(*bbox_pts[:4])
            region_text = page.get_textbox(rect)
        except Exception as text_err:
            logger.debug(f"[Table] 页面 {page_num} YOLO bbox 文本抽取失败: {text_err}")
            continue

        normalized = re.sub(r"\s+", " ", region_text or "").strip()
        if len(normalized) < 20 or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())

        caption = _find_table_caption(page_text, tuple(bbox_pts[:4]), page_height)
        md = _plain_table_text_to_markdown(region_text, caption)
        if not md:
            continue
        table_info = {
            "markdown": md,
            "data": [[line] for line in [ln for ln in region_text.splitlines() if ln.strip()]],
            "bbox": tuple(bbox_pts[:4]),
            "caption": caption,
            "page": page_num,
            "rows": max(1, len([line for line in region_text.splitlines() if line.strip()])),
            "cols": 0,
            "source": "doclayout_yolo_fallback",
            "layout_score": det.get("score"),
        }
        table_info["structured_bundle"] = build_structured_table_bundle(
            table_info,
            index_in_doc=idx,
        )
        results.append(table_info)

    if results:
        logger.info(f"[Table] 页面 {page_num} YOLO fallback 检测到 {len(results)} 个表格区域")
    return results


def extract_tables_from_page(page, page_text: str, page_num: int) -> List[dict]:
    """从 PyMuPDF 页面对象中提取表格并转换为 Markdown

    Args:
        page: PyMuPDF 页面对象 (fitz.Page)
        page_text: 该页的原始文本
        page_num: 页码（1-indexed）

    Returns:
        表格信息列表，每项包含:
        - markdown: Markdown 格式的表格
        - bbox: 表格坐标 (x0, y0, x1, y1)
        - caption: 表格标题
        - page: 页码
    """
    try:
        tables = page.find_tables()
    except Exception as e:
        logger.debug(f"[Table] 页面 {page_num} 表格检测失败: {e}")
        return []

    if not tables or not tables.tables:
        return _extract_tables_with_yolo_fallback(page, page_text, page_num)

    results = []
    page_height = page.rect.height

    for i, table in enumerate(tables.tables):
        try:
            # 提取表格数据
            data = table.extract()
            if not data or len(data) < 2:
                # 少于 2 行的不算有效表格
                continue

            # 检查表格是否有足够内容（过滤空表格）
            non_empty_cells = sum(
                1 for row in data for cell in row
                if cell is not None and str(cell).strip()
            )
            total_cells = sum(len(row) for row in data)
            if total_cells == 0 or non_empty_cells / total_cells < 0.3:
                continue

            bbox = table.bbox  # (x0, y0, x1, y1)
            cell_bboxes = _extract_pymupdf_cell_bboxes(
                table,
                len(data),
                max(len(row) for row in data) if data else 0,
            )

            # 检测表格标题
            caption = _find_table_caption(page_text, bbox, page_height)

            # 转换为 Markdown
            md = _table_to_markdown(data, caption)
            if md:
                table_info = {
                    "markdown": md,
                    "data": data,
                    "bbox": bbox,
                    "cell_bboxes": cell_bboxes,
                    "caption": caption,
                    "page": page_num,
                    "rows": len(data),
                    "cols": len(data[0]) if data else 0,
                    "source": "pymupdf_table",
                }
                table_info["structured_bundle"] = build_structured_table_bundle(
                    table_info,
                    index_in_doc=i + 1,
                )
                results.append(table_info)

        except Exception as e:
            logger.debug(f"[Table] 页面 {page_num} 表格 {i} 提取失败: {e}")
            continue

    if results:
        logger.info(f"[Table] 页面 {page_num} 检测到 {len(results)} 个表格")

    if results:
        return results

    return _extract_tables_with_yolo_fallback(page, page_text, page_num)


def inject_tables_into_text(page_text: str, tables: List[dict]) -> str:
    """将检测到的 Markdown 表格注入页面文本末尾

    策略：在页面文本末尾追加表格的 Markdown 格式。
    不尝试替换原文中的表格区域（因为坐标→文本位置映射不精确），
    而是追加到末尾，让 structure_aware_split 的表格保护机制处理。

    Args:
        page_text: 原始页面文本
        tables: extract_tables_from_page 返回的表格列表

    Returns:
        注入表格后的页面文本
    """
    if not tables:
        return page_text

    parts = [page_text.rstrip()]

    for t in tables:
        md = t["markdown"]
        # 避免重复：如果原文已包含 Markdown 表格（| 分隔），跳过
        first_data_line = ""
        for line in md.split("\n"):
            if line.startswith("|") and "---" not in line:
                first_data_line = line
                break
        if first_data_line and first_data_line in page_text:
            continue

        parts.append("")
        parts.append(md)

    return "\n".join(parts)
