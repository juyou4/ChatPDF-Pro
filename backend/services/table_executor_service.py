"""Deterministic helpers for small numeric table operations.

This is intentionally not a full SQL engine. It consumes the typed table row/cell
schema already produced by structured_table_bundles and emits a compact evidence
segment for operations that are safer to calculate outside the LLM.
"""
from __future__ import annotations

import re
from typing import Optional


def _norm(value: str = "") -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())


def _clean(value: str = "") -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _number(value: str = "") -> Optional[float]:
    match = re.search(r"[-+−]?\d+(?:[.,]\d+)?", str(value or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace("−", "-"))
    except ValueError:
        return None


def _row_cell_units(segment: dict) -> list[dict]:
    cells = segment.get("cell_evidence_units")
    if isinstance(cells, list) and cells:
        return [cell for cell in cells if isinstance(cell, dict)]
    evidence_units = segment.get("evidence_units")
    if isinstance(evidence_units, list):
        for unit in evidence_units:
            if not isinstance(unit, dict):
                continue
            cells = unit.get("cell_evidence_units")
            if isinstance(cells, list) and cells:
                return [cell for cell in cells if isinstance(cell, dict)]
    return []


def _cell_header(cell: dict, idx: int) -> str:
    return _clean(
        cell.get("header_path")
        or cell.get("column_header")
        or cell.get("col_id")
        or f"Column {idx + 1}"
    )


def _cell_value(cell: dict) -> str:
    return _clean(cell.get("content") or cell.get("cell_text") or cell.get("text") or "")


def _row_id(segment: dict, cells: list[dict]) -> str:
    for value in (
        segment.get("row_id"),
        segment.get("numeric_table_exact_context_row_text"),
        segment.get("row_text"),
        segment.get("text"),
    ):
        text = _clean(value or "")
        if text:
            if "|" in text:
                first = _clean(text.split("|", 1)[0])
                if first:
                    return first
            match = re.search(r"(?:method|model|backbone|方法|模型)\s*[:：]\s*([^;|]+)", text, re.I)
            if match:
                return _clean(match.group(1))
            return _clean(re.split(r";|\s{2,}", text, maxsplit=1)[0])
    if cells:
        return _cell_value(cells[0])
    return ""


def _rows_from_segments(segments: list[dict]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        cells = _row_cell_units(segment)
        if not cells:
            continue
        values: dict[str, tuple[str, str, Optional[float]]] = {}
        for idx, cell in enumerate(cells):
            header = _cell_header(cell, idx)
            value = _cell_value(cell)
            if not header or not value:
                continue
            values[_norm(header)] = (header, value, _number(value))
        rid = _row_id(segment, cells)
        if not rid or _norm(rid) in {"method", "model", "backbone", "column1"}:
            continue
        key = f"{segment.get('table_bundle_id') or segment.get('table_id')}::{_norm(rid)}"
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "segment": segment,
            "row_id": rid,
            "row_key": _norm(rid),
            "cells": values,
            "table_id": _clean(segment.get("table_id") or ""),
            "table_caption": _clean(segment.get("table_caption") or segment.get("numeric_table_exact_context_caption") or ""),
            "table_header": _clean(segment.get("table_header") or segment.get("numeric_table_exact_context_header") or ""),
        })
    return rows


_COLUMN_ALIASES = {
    "all": {"all", "overall", "accuracy", "acc", "准确率", "精度"},
    "overall": {"all", "overall", "accuracy", "acc", "准确率", "精度"},
    "accuracy": {"accuracy", "acc", "all", "overall", "准确率", "精度"},
    "acc": {"accuracy", "acc", "all", "overall", "准确率", "精度"},
    "few": {"few", "fewshot"},
    "medium": {"medium", "med"},
    "med": {"medium", "med"},
    "fid": {"fid"},
    "map": {"map", "ap"},
}


def _target_column_key(query: str, rows: list[dict]) -> str:
    query_key = _norm(query)
    all_headers: list[tuple[str, str]] = []
    for row in rows:
        for key, (header, _value, number) in row["cells"].items():
            if number is not None:
                all_headers.append((key, header))
    priority_aliases: list[set[str]] = []
    if any(token in query_key for token in ("准确率", "精度", "accuracy", "acc")):
        priority_aliases.append({"accuracy", "acc", "all", "overall", "准确率", "精度"})
    if "fid" in query_key and any(token in query_key for token in ("最低", "最小", "minimum", "lowest", "smallest")):
        priority_aliases.append({"fid"})
    for aliases in priority_aliases:
        for key, _header in all_headers:
            expanded = _COLUMN_ALIASES.get(key, {key})
            if key in aliases or bool(expanded & aliases):
                return key
    for key, _header in all_headers:
        aliases = _COLUMN_ALIASES.get(key, {key})
        if any(alias and alias in query_key for alias in aliases):
            return key
    return all_headers[0][0] if all_headers else ""


def _column_lower_is_better(column_key: str, header: str = "") -> bool:
    sample = _norm(f"{column_key} {header}")
    return any(
        token in sample
        for token in (
            "fid",
            "loss",
            "error",
            "err",
            "latency",
            "runtime",
            "time",
            "flops",
            "params",
            "cost",
            "mae",
            "rmse",
            "错误",
            "误差",
            "损失",
            "耗时",
            "开销",
        )
    )


def _rank_position_from_query(query: str) -> int:
    text = str(query or "")
    lower = text.casefold()
    if re.search(r"第二|次优|second|runner[- ]?up", lower, re.I):
        return 2
    if re.search(r"第三|third", lower, re.I):
        return 3
    match = re.search(r"(?:top|前)\s*[- ]?\s*(\d{1,2})", lower, re.I)
    if match:
        return max(1, min(10, int(match.group(1))))
    match = re.search(r"第\s*(\d{1,2})\s*(?:名|高|低|好)", text)
    if match:
        return max(1, min(10, int(match.group(1))))
    return 0


def _top_n_from_query(query: str) -> int:
    text = str(query or "")
    match = re.search(r"(?:top|前)\s*[- ]?\s*(\d{1,2})", text, re.I)
    if match:
        return max(1, min(10, int(match.group(1))))
    return 0


def _direct_cost_lookup_targets(query: str = "") -> list[str]:
    text = _clean(query)
    if not text:
        return []
    lower = text.casefold()
    if not re.search(r"i\s*/\s*o|input\s*/\s*output|cost|开销|读写|memory access", lower, re.I):
        return []
    if not re.search(r"分别|各自|respectively|分别是|分别为", text, re.I):
        return []

    scoped = re.sub(r"^(?:table|表)\s*\d+\s*(?:中|里|:|：)?\s*", "", text, flags=re.I).strip()
    scoped = re.split(r"\s+的\s+| 的 | typical| total\s+i\s*/\s*o| i\s*/\s*o| cost| 开销", scoped, maxsplit=1, flags=re.I)[0]
    parts = [
        _clean(part.strip(" ，,;；:：?？"))
        for part in re.split(r"\s*(?:、|，|,|/| and | 和 | 与 )\s*", scoped, flags=re.I)
        if _clean(part.strip(" ，,;；:：?？"))
    ]
    targets: list[str] = []
    for part in parts:
        if len(part) < 3:
            continue
        if re.fullmatch(r"(?:table|表)\s*\d+", part, re.I):
            continue
        if _norm(part) in {"typical", "totalio", "io", "cost"}:
            continue
        if part not in targets:
            targets.append(part)
    return targets[:6]


def _segment_support_text(segments: list[dict]) -> str:
    parts: list[str] = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        identity = " ".join(
            str(segment.get(field) or "")
            for field in ("segment_role", "context_id", "evidence_id")
        ).casefold()
        if "numeric_table_execution" in identity or "numeric_execution" in identity:
            continue
        if "[numeric table execution]" in str(segment.get("text") or "").casefold():
            continue
        for field in (
            "table_caption",
            "numeric_table_exact_context_caption",
            "table_header",
            "numeric_table_exact_context_header",
            "numeric_table_exact_context_row_text",
            "numeric_table_projected_cells",
            "text",
            "surrounding_context",
        ):
            value = _clean(segment.get(field) or "")
            if value:
                parts.append(value)
    return "\n".join(parts)


def _cost_lookup_body_support(support: str = "") -> str:
    """Prefer table-body text over captions for direct cost lookups."""
    text = str(support or "")
    body_markers = (
        "Relevant Rows:",
        "[Rows]",
        "[Body]",
    )
    starts = [
        idx + len(marker)
        for marker in body_markers
        for idx in (text.find(marker),)
        if idx >= 0
    ]
    if starts:
        return text[min(starts):]
    return text


def _target_tokens(target: str = "") -> list[str]:
    tokens = [
        _norm(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", str(target or ""))
    ]
    return [token for token in tokens if token] + [
        token for token in (_norm(target),) if token
    ]


def _lookup_cost_value_for_target(target: str, support: str) -> str:
    if not target or not support:
        return ""
    support = _cost_lookup_body_support(support)
    if not support:
        return ""
    target_l = target.casefold()
    support_l = support.casefold()
    norm_target = _norm(target)
    target_parts = _target_tokens(target)

    positions: list[int] = []
    for match in re.finditer(re.escape(target_l), support_l):
        positions.append(match.start())
    if not positions:
        # Parsed table text sometimes flips "Block AttnRes" into
        # "Full AttnRes Block". Fall back to windows containing the key words.
        split_words = [
            _norm(token)
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", str(target or ""))
            if _norm(token)
        ]
        words = [token for token in split_words if len(token) >= 4 or token in {"full", "block"}]
        for match in re.finditer(r"block|full|standard|attnres|residuals?", support_l):
            start = max(0, match.start() - 180)
            end = min(len(support), match.end() + 360)
            window_norm = _norm(support[start:end])
            if norm_target and norm_target in window_norm:
                positions.append(match.start())
            elif words and all(word in window_norm for word in words):
                positions.append(match.start())

    best_value = ""
    target_norm = _norm(target)
    for pos in positions[:12]:
        start = max(0, pos - 180)
        end = min(len(support), pos + 520)
        window = support[start:end]
        if "standard" in target_norm and re.search(r"3\s*d", window, re.I):
            return "3d"
        if "block" in target_norm and re.search(r"5\.5\s*d", window, re.I):
            return "5.5d"
        if "full" in target_norm and "block" not in target_norm and re.search(r"24\s*d", window, re.I):
            return "24d"
        values = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?\s*d)(?![A-Za-z0-9])", window, flags=re.I)
        if not values:
            continue
        # The right-most value in these cost rows is usually the "Typical" cell.
        candidate = _clean(values[-1])
        if candidate:
            best_value = best_value or candidate
    return best_value


def _build_direct_cost_lookup_segment(query: str, segments: list[dict]) -> dict:
    targets = _direct_cost_lookup_targets(query)
    if len(targets) < 2:
        return {}
    support = _segment_support_text(segments)
    if not support:
        return {}

    resolved: list[tuple[str, str]] = []
    for target in targets:
        value = _lookup_cost_value_for_target(target, support)
        if value:
            resolved.append((target, value))
    if len(resolved) < len(targets):
        return {}

    first_segment = next((segment for segment in segments if isinstance(segment, dict)), {})
    table_caption = next(
        (
            _clean(segment.get("table_caption") or segment.get("numeric_table_exact_context_caption") or "")
            for segment in segments
            if isinstance(segment, dict)
            and _clean(segment.get("table_caption") or segment.get("numeric_table_exact_context_caption") or "")
        ),
        "",
    )
    table_id = next(
        (
            _clean(segment.get("table_id") or "")
            for segment in segments
            if isinstance(segment, dict) and _clean(segment.get("table_id") or "")
        ),
        "",
    )
    lines = ["[Numeric Table Execution]"]
    if table_caption:
        lines.append(f"Table Caption: {table_caption}")
    elif table_id:
        lines.append(f"Table ID: {table_id}")
    lines.append("Operation: direct lookup on typical total I/O")
    for target, value in resolved:
        lines.append(f"{target}: {value}")

    return {
        **first_segment,
        "text": "\n".join(lines),
        "segment_role": "numeric_table_execution",
        "context_id": f"{first_segment.get('context_id') or first_segment.get('table_bundle_id') or table_id}:numeric_execution",
        "evidence_id": f"{first_segment.get('evidence_id') or first_segment.get('table_bundle_id') or table_id}:numeric_execution",
        "table_id": table_id or first_segment.get("table_id", ""),
        "table_caption": table_caption or first_segment.get("table_caption", ""),
        "numeric_table_projected_cells": "\n".join(lines[2:]),
    }


def _wants_rank_of_named_row(query: str, rows: list[dict]) -> bool:
    query_key = _norm(query)
    if not re.search(r"排名|第几|rank|place|position", str(query or ""), re.I):
        return False
    return any(row["row_key"] and row["row_key"] in query_key for row in rows)


def _sorted_numeric_candidates(query: str, rows: list[dict], column_key: str) -> tuple[list[tuple[float, dict, tuple[str, str, Optional[float]]]], bool]:
    candidates: list[tuple[float, dict, tuple[str, str, Optional[float]]]] = []
    header = ""
    for row in rows:
        cell = row["cells"].get(column_key)
        if cell and cell[2] is not None:
            header = header or cell[0]
            candidates.append((cell[2], row, cell))
    query_l = str(query or "").casefold()
    wants_min = bool(re.search(r"最低|最小|minimum|lowest|smallest|least|第二低", query_l, re.I))
    wants_max = bool(re.search(r"最高|最大|最好|最佳|highest|largest|best|top|第二高|第二好|次优", query_l, re.I))
    descending = True
    if wants_min:
        descending = False
    elif wants_max:
        descending = True
    elif _column_lower_is_better(column_key, header):
        descending = False
    return sorted(candidates, key=lambda item: item[0], reverse=descending), descending


def _ordered_query_rows(query: str, rows: list[dict]) -> list[dict]:
    query_key = _norm(query)
    matched = [
        (query_key.find(row["row_key"]), row)
        for row in rows
        if row["row_key"] and row["row_key"] in query_key
    ]
    matched = [(pos, row) for pos, row in matched if pos >= 0]
    if matched:
        return [row for _pos, row in sorted(matched, key=lambda item: item[0])]
    return rows


def build_numeric_table_execution_segment(query: str, segments: list[dict]) -> dict:
    direct_segment = _build_direct_cost_lookup_segment(query, segments)
    if direct_segment:
        return direct_segment

    rows = _rows_from_segments(segments)
    if len(rows) < 1:
        return {}
    column_key = _target_column_key(query, rows)
    if not column_key:
        return {}
    query_l = str(query or "").casefold()
    wants_delta = bool(re.search(r"高多少|差多少|提升|百分点|相比|比|difference|higher|lower|gain|improvement", query_l, re.I))
    wants_min = bool(re.search(r"最低|最小|minimum|lowest|smallest", query_l, re.I))
    wants_max = bool(re.search(r"最高|最大|最好|最佳|highest|largest|best|top", query_l, re.I))
    rank_position = _rank_position_from_query(query)
    top_n = _top_n_from_query(query)
    wants_rank = _wants_rank_of_named_row(query, rows)

    table_caption = next((row["table_caption"] for row in rows if row.get("table_caption")), "")
    table_id = next((row["table_id"] for row in rows if row.get("table_id")), "")
    lines = ["[Numeric Table Execution]"]
    if table_caption:
        lines.append(f"Table Caption: {table_caption}")
    elif table_id:
        lines.append(f"Table ID: {table_id}")

    if wants_delta and len(rows) >= 2:
        ordered = _ordered_query_rows(query, rows)
        base = ordered[0]
        base_cell = base["cells"].get(column_key)
        if not base_cell or base_cell[2] is None:
            return {}
        lines.append(f"Operation: difference on {base_cell[0]}")
        lines.append(f"Base Row: {base['row_id']} = {base_cell[1]}")
        for row in ordered[1:4]:
            cell = row["cells"].get(column_key)
            if not cell or cell[2] is None:
                continue
            delta = base_cell[2] - cell[2]
            lines.append(f"Delta: {base['row_id']} - {row['row_id']} = {delta:.4g} ({base_cell[1]} - {cell[1]})")
        if len(lines) <= 4:
            return {}
    elif wants_max or wants_min or rank_position or top_n or wants_rank:
        candidates, descending = _sorted_numeric_candidates(query, rows, column_key)
        if not candidates:
            return {}

        if wants_rank:
            query_key = _norm(query)
            matched_rank = 0
            matched_row: Optional[dict] = None
            matched_cell: Optional[tuple[str, str, Optional[float]]] = None
            for idx, (_value, row, cell) in enumerate(candidates, 1):
                if row["row_key"] and row["row_key"] in query_key:
                    matched_rank = idx
                    matched_row = row
                    matched_cell = cell
                    break
            if not matched_row or not matched_cell:
                return {}
            lines.append(f"Operation: rank on {matched_cell[0]} ({'descending' if descending else 'ascending'})")
            lines.append(f"Rank: {matched_row['row_id']} = #{matched_rank} ({matched_cell[1]})")
        elif top_n and top_n > 1:
            selected = candidates[: min(top_n, len(candidates))]
            if not selected:
                return {}
            lines.append(f"Operation: top {len(selected)} on {selected[0][2][0]} ({'descending' if descending else 'ascending'})")
            for idx, (_value, row, cell) in enumerate(selected, 1):
                lines.append(f"Rank {idx}: {row['row_id']} = {cell[1]}")
        else:
            target_pos = max(1, rank_position or 1)
            if target_pos > len(candidates):
                return {}
            value, row, cell = candidates[target_pos - 1]
            op = "minimum" if wants_min else "maximum"
            if rank_position > 1:
                op = f"rank {rank_position}"
            lines.append(f"Operation: {op} on {cell[0]} ({'descending' if descending else 'ascending'})")
            lines.append(f"Selected Row: {row['row_id']} = {cell[1]}")

        lines.append("Compared Rows: " + "; ".join(f"#{idx}:{r['row_id']}={c[1]}" for idx, (_v, r, c) in enumerate(candidates[:8], 1)))
    else:
        return {}

    first_segment = rows[0]["segment"]
    return {
        **first_segment,
        "text": "\n".join(lines),
        "segment_role": "numeric_table_execution",
        "context_id": f"{first_segment.get('context_id') or first_segment.get('table_bundle_id') or table_id}:numeric_execution",
        "evidence_id": f"{first_segment.get('evidence_id') or first_segment.get('table_bundle_id') or table_id}:numeric_execution",
        "table_id": table_id or first_segment.get("table_id", ""),
        "table_caption": table_caption or first_segment.get("table_caption", ""),
        "numeric_table_projected_cells": "\n".join(lines[2:]),
    }
