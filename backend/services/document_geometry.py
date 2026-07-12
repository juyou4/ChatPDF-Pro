"""Canonical geometry helpers shared by parser adapters and visual verification."""
from __future__ import annotations

from typing import Any


CANONICAL_COORDINATE_SPACE = "pdf_top_left_points"


def normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        x0, y0, x1, y1 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return []
    if x1 <= x0 or y1 <= y0:
        return []
    return [x0, y0, x1, y1]


def normalize_page_size(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return []
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return []
    return [width, height] if width > 0 and height > 0 else []


def to_pdf_top_left_points(
    bbox: Any,
    *,
    coordinate_space: str,
    page_size: Any,
) -> list[float]:
    """Convert a known parser bbox into the PyMuPDF crop coordinate system."""
    raw = normalize_bbox(bbox)
    size = normalize_page_size(page_size)
    if not raw or not size:
        return []
    width, height = size
    source = str(coordinate_space or "").strip().lower()
    x0, y0, x1, y1 = raw
    if source == "normalized_0_1000":
        x0, x1 = x0 / 1000.0 * width, x1 / 1000.0 * width
        y0, y1 = y0 / 1000.0 * height, y1 / 1000.0 * height
    elif source == "pdf_bottom_left_points":
        y0, y1 = height - y1, height - y0
    elif source != CANONICAL_COORDINATE_SPACE:
        return []
    x0, x1 = max(0.0, min(width, x0)), max(0.0, min(width, x1))
    y0, y1 = max(0.0, min(height, y0)), max(0.0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return []
    return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]


def visual_geometry(
    bbox: Any,
    *,
    coordinate_space: str,
    page_size: Any,
    rotation: int = 0,
) -> dict[str, Any]:
    """Return additive geometry fields without changing parser-owned bboxes."""
    raw_bbox = normalize_bbox(bbox)
    normalized_size = normalize_page_size(page_size)
    return {
        "raw_bbox": raw_bbox,
        "bbox_coordinate_space": str(coordinate_space or ""),
        "page_size": normalized_size,
        "rotation": int(rotation or 0),
        "visual_bbox": to_pdf_top_left_points(
            raw_bbox,
            coordinate_space=coordinate_space,
            page_size=normalized_size,
        ),
        "visual_coordinate_space": CANONICAL_COORDINATE_SPACE,
    }
