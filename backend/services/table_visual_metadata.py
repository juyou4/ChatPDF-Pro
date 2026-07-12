"""Stable, read-only metadata for visual table verification.

The visual-verification cache must identify a concrete table occurrence, not
only a human-facing label such as ``Table 3``. These helpers derive that
identity from the table's location, content and provenance. They deliberately
return new values only and never modify parser-owned bundle dictionaries.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


TABLE_VISUAL_METADATA_VERSION = 1
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_table_visual_identity(bundle: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build a deterministic, JSON-safe identity payload for a table bundle.

    The payload intentionally excludes generated IDs and other mutable runtime
    fields. A parser can therefore keep owning its native data while callers
    derive the same visual-verification identity from an equivalent bundle.
    """
    source = bundle if isinstance(bundle, Mapping) else {}
    table_body = _first_text(source, "table_body_markdown", "table_markdown")
    html_table = _normalized_text(source.get("html_table"))
    fallback_text = ""
    if not table_body and not html_table:
        fallback_text = _normalized_text(source.get("bundle_text"))

    return {
        "version": TABLE_VISUAL_METADATA_VERSION,
        "location": {
            "pages": _normalized_pages(source),
            "bboxes": _normalized_bboxes(source),
        },
        "table": {
            "table_id": _normalized_text(source.get("table_id")),
            "caption": _normalized_text(source.get("table_caption")),
            "header": _normalized_text(source.get("table_header")),
            "body_markdown": table_body,
            "html_table": html_table,
            "footnote": _normalized_text(source.get("table_footnote")),
            "fallback_text": fallback_text,
            "evidence": _normalized_evidence(source.get("evidence_units")),
        },
        "source": {
            "source": _normalized_text(source.get("source")),
            "selected_source": _normalized_text(source.get("selected_source")),
            "source_ids": _normalized_source_ids(source),
        },
    }


def derive_table_source_hash(bundle: Mapping[str, Any] | None) -> str:
    """Return a full source hash suitable for cache invalidation."""
    payload = normalize_table_visual_identity(bundle)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def derive_table_instance_id(bundle: Mapping[str, Any] | None) -> str:
    """Return a compact stable identifier for one concrete table occurrence."""
    source_hash = derive_table_source_hash(bundle)
    return f"table-v{TABLE_VISUAL_METADATA_VERSION}-{source_hash[:24]}"


def build_table_visual_metadata(bundle: Mapping[str, Any] | None) -> dict[str, str]:
    """Return the derived fields to attach to a copied bundle or chunk metadata."""
    source_hash = derive_table_source_hash(bundle)
    return {
        "table_instance_id": f"table-v{TABLE_VISUAL_METADATA_VERSION}-{source_hash[:24]}",
        "table_source_hash": source_hash,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    text = _CONTROL_RE.sub("", str(value))
    return _WHITESPACE_RE.sub(" ", text).strip()


def _first_text(source: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        text = _normalized_text(source.get(key))
        if text:
            return text
    return ""


def _normalized_pages(source: Mapping[str, Any]) -> list[int]:
    pages: list[int] = []
    raw_pages = source.get("pages")
    if isinstance(raw_pages, (list, tuple, set)):
        pages.extend(_coerce_positive_int(value) for value in raw_pages)

    page_start = _coerce_positive_int(source.get("page_start"))
    page_end = _coerce_positive_int(source.get("page_end"))
    if page_start:
        pages.append(page_start)
    if page_end:
        pages.append(page_end)

    page = _coerce_positive_int(source.get("page"))
    if page:
        pages.append(page)
    if not pages:
        page_index = _coerce_nonnegative_int(source.get("page_index"))
        if page_index is not None:
            pages.append(page_index + 1)

    return sorted({page for page in pages if page > 0})


def _coerce_positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _coerce_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _normalized_bboxes(source: Mapping[str, Any]) -> list[list[float]]:
    bboxes: list[list[float]] = []
    for key in ("bounding_box", "bbox", "table_bbox"):
        bboxes.extend(_iter_normalized_bboxes(source.get(key)))
    bboxes.extend(_iter_normalized_bboxes(source.get("bounding_boxes")))

    unique = {tuple(bbox) for bbox in bboxes}
    return [list(bbox) for bbox in sorted(unique)]


def _iter_normalized_bboxes(value: Any) -> list[list[float]]:
    bbox = _normalize_bbox(value)
    if bbox:
        return [bbox]
    if not isinstance(value, (list, tuple)):
        return []
    return [bbox for item in value if (bbox := _normalize_bbox(item))]


def _normalize_bbox(value: Any) -> list[float]:
    raw_values: list[Any]
    if isinstance(value, Mapping):
        raw_values = [value.get(key) for key in ("x0", "y0", "x1", "y1")]
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        raw_values = list(value[:4])
    else:
        return []

    values: list[float] = []
    for raw in raw_values:
        try:
            coordinate = float(raw)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(coordinate):
            return []
        coordinate = round(coordinate, 3)
        values.append(0.0 if coordinate == 0 else coordinate)
    if values[2] <= values[0] or values[3] <= values[1]:
        return []
    return values


def _normalized_source_ids(source: Mapping[str, Any]) -> list[str]:
    raw_values = source.get("source_ids")
    if isinstance(raw_values, (list, tuple, set)):
        values = [_normalized_text(value) for value in raw_values]
    elif raw_values not in (None, ""):
        values = [_normalized_text(raw_values)]
    else:
        values = []

    for key in ("source_id", "page_uid", "page_uids"):
        value = source.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(_normalized_text(item) for item in value)
        elif value not in (None, ""):
            values.append(_normalized_text(value))
    if not any(values):
        fallback_bundle_id = _normalized_text(source.get("bundle_id") or source.get("table_bundle_id"))
        if fallback_bundle_id:
            values.append(fallback_bundle_id)
    return sorted({value for value in values if value})


def _normalized_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []

    normalized: list[dict[str, Any]] = []
    for unit in value:
        if not isinstance(unit, Mapping):
            continue
        cells = unit.get("cell_evidence_units")
        normalized_cells = []
        if isinstance(cells, (list, tuple)):
            for cell in cells:
                if not isinstance(cell, Mapping):
                    continue
                normalized_cells.append({
                    "header": _first_text(cell, "header_path", "header", "column", "col_id"),
                    "content": _first_text(cell, "content", "text", "cell_text", "value"),
                })
        record = {
            "type": _normalized_text(unit.get("evidence_unit_type")),
            "row_id": _normalized_text(unit.get("row_id")),
            "row_number": _coerce_nonnegative_int(unit.get("row_number")),
            "text": _first_text(unit, "row_text", "content", "raw_row_text", "table_row_boundary_text"),
            "cells": normalized_cells,
        }
        if any(record[key] for key in ("type", "row_id", "text", "cells")) or record["row_number"] is not None:
            normalized.append(record)
    return normalized
