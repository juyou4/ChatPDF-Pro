"""Pure page/region risk scoring for optional visual enrichment."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


_VISUAL_NUMBER_PATTERN = r"[a-z]?\d+(?:[.-]\d+)?(?:[a-z]|\s*\([a-z]\))?"
_VISUAL_REFERENCE_RE = re.compile(
    rf"(?:\b(?:figure|fig\.?|table)\s*{_VISUAL_NUMBER_PATTERN}(?![a-z0-9])"
    rf"|(?:图|表)\s*{_VISUAL_NUMBER_PATTERN}(?![a-z0-9]))",
    re.IGNORECASE,
)
_FIGURE_REFERENCE_RE = re.compile(
    rf"(?:\b(?:figure|fig\.?)\s*({_VISUAL_NUMBER_PATTERN})(?![a-z0-9])"
    rf"|图\s*({_VISUAL_NUMBER_PATTERN})(?![a-z0-9]))",
    re.IGNORECASE,
)
_GARBLED_RE = re.compile(r"(?:\ufffd|\(cid:\d+\)|\bcid:\d+\b|[\x00-\x08\x0b\x0c\x0e-\x1f])", re.IGNORECASE)


@dataclass(frozen=True)
class VisualRiskAssessment:
    should_enrich: bool
    score: float
    level: str
    reasons: tuple[str, ...]
    page: int
    bbox_hash: str
    source: str
    structure_confidence: float | None
    ocr_confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def assess_visual_region_risk(
    *,
    page: int,
    bbox: Any,
    source: str = "",
    caption: str = "",
    page_text: str = "",
    structure_confidence: Any = None,
    ocr_confidence: Any = None,
    query: str = "",
    text_evidence: str = "",
    threshold: float = 0.45,
) -> VisualRiskAssessment:
    """Score whether one image region needs a VLM rather than text-only reuse."""
    safe_source = str(source or "").strip().lower()
    safe_caption = _clean_text(caption)
    safe_page_text = _clean_text(page_text, limit=12000)
    structure_score = _confidence(structure_confidence)
    ocr_score = _confidence(ocr_confidence)
    reasons: list[str] = []
    score = 0.0

    if safe_source in {"yolo", "doclayout_yolo", "layout_yolo"}:
        reasons.append("yolo_only_detection")
        score += 0.55
    elif safe_source in {"fallback", "caption_only", "unknown"}:
        reasons.append("weak_structure_source")
        score += 0.30

    if not safe_caption:
        reasons.append("missing_caption")
        score += 0.50
    elif _caption_is_weak(safe_caption):
        reasons.append("weak_caption")
        score += 0.24

    if structure_score is not None and structure_score < 0.62:
        reasons.append("low_structure_confidence")
        score += 0.32

    if ocr_score is not None and ocr_score < 0.65:
        reasons.append("low_ocr_confidence")
        score += 0.38

    if len(safe_page_text) < 80:
        reasons.append("low_page_text_density")
        score += 0.46

    combined_text = f"{safe_caption}\n{safe_page_text}".strip()
    if combined_text and _looks_garbled(combined_text):
        reasons.append("garbled_text")
        score += 0.48

    reference = _first_visual_reference(query)
    if reference:
        haystack = _normalize_reference_text(text_evidence)
        if _normalize_reference_text(reference) not in haystack:
            reasons.append("explicit_visual_reference_without_text_evidence")
            score += 0.55

    normalized_score = round(min(1.0, score), 3)
    should_enrich = normalized_score >= max(0.0, min(1.0, float(threshold)))
    level = "high" if normalized_score >= 0.7 else "medium" if should_enrich else "low"
    return VisualRiskAssessment(
        should_enrich=should_enrich,
        score=normalized_score,
        level=level,
        reasons=tuple(reasons),
        page=max(0, int(page or 0)),
        bbox_hash=_bbox_hash(bbox),
        source=safe_source,
        structure_confidence=structure_score,
        ocr_confidence=ocr_score,
    )


def assess_page_risk(
    *,
    page: int,
    bbox: Any,
    page_text: str = "",
    ocr_confidence: Any = None,
    query: str = "",
    text_evidence: str = "",
    threshold: float = 0.45,
) -> VisualRiskAssessment:
    """评估整页是否需要受限的视觉文字恢复或结构复核。"""
    safe_page_text = _clean_text(page_text, limit=12000)
    ocr_score = _confidence(ocr_confidence)
    reasons: list[str] = []
    score = 0.0

    if not safe_page_text:
        reasons.append("missing_page_text")
        score += 0.75
    elif len(safe_page_text) < 80:
        reasons.append("very_low_page_text_density")
        score += 0.55
    elif len(safe_page_text) < 200:
        reasons.append("low_page_text_density")
        score += 0.25

    if ocr_score is not None and ocr_score < 0.65:
        reasons.append("low_ocr_confidence")
        score += 0.40
    if safe_page_text and _looks_garbled(safe_page_text):
        reasons.append("garbled_text")
        score += 0.55

    reference = _first_visual_reference(query)
    if reference and _normalize_reference_text(reference) not in _normalize_reference_text(text_evidence):
        reasons.append("explicit_visual_reference_without_text_evidence")
        score += 0.55

    normalized_score = round(min(1.0, score), 3)
    bounded_threshold = max(0.0, min(1.0, float(threshold)))
    should_enrich = normalized_score >= bounded_threshold
    level = "high" if normalized_score >= 0.7 else "medium" if should_enrich else "low"
    return VisualRiskAssessment(
        should_enrich=should_enrich,
        score=normalized_score,
        level=level,
        reasons=tuple(reasons),
        page=max(0, int(page or 0)),
        bbox_hash=_bbox_hash(bbox),
        source="page",
        structure_confidence=None,
        ocr_confidence=ocr_score,
    )


def extract_figure_reference(query: str) -> dict[str, str]:
    """返回普通问答中明确出现的 Figure/图号引用。"""
    match = _FIGURE_REFERENCE_RE.search(str(query or ""))
    if not match:
        return {}
    number = str(match.group(1) or match.group(2) or "").strip().lower()
    return {
        "label": str(match.group(0) or "").strip(),
        "number": number,
    }


def assess_figure_risk(
    figure: Any,
    *,
    page_text: str = "",
    query: str = "",
    text_evidence: str = "",
    threshold: float = 0.45,
) -> VisualRiskAssessment:
    """Adapt a LogicalFigureSchema or legacy figure mapping to the risk gate."""
    page_idx = _value(figure, "page_idx", None)
    if page_idx is None:
        page = _int(_value(figure, "page", 0) or _value(figure, "page_num", 0))
    else:
        page = _int(page_idx) + 1
    bbox = (
        _value(figure, "full_bbox_page_pts", None)
        or _value(figure, "body_bbox_page_pts", None)
        or _value(figure, "group_bbox", None)
        or _value(figure, "figure_bbox", None)
        or _value(figure, "bbox", None)
    )
    caption = _value(figure, "caption_text", "") or _value(figure, "caption", "")
    source = _value(figure, "source", "")
    metadata = _value(figure, "source_metadata", {})
    if isinstance(metadata, Mapping):
        source = source or metadata.get("adapter") or ""
    structure_confidence = _value(figure, "confidence", None)
    ocr_confidence = _find_confidence(metadata, (
        "ocr_confidence",
        "text_confidence",
        "recognition_confidence",
    ))
    return assess_visual_region_risk(
        page=page,
        bbox=bbox,
        source=str(source or ""),
        caption=str(caption or ""),
        page_text=page_text,
        structure_confidence=structure_confidence,
        ocr_confidence=ocr_confidence,
        query=query,
        text_evidence=text_evidence,
        threshold=threshold,
    )


def page_text_for_risk(data: Mapping[str, Any] | None, page_number: int) -> str:
    """Read bounded page text from common local/MinerU normalized page shapes."""
    if not isinstance(data, Mapping) or page_number <= 0:
        return ""
    candidates: list[Any] = [data.get("pages")]
    ocr_result = data.get("ocr_result")
    if isinstance(ocr_result, Mapping):
        candidates.append(ocr_result.get("pages"))
    for pages in candidates:
        if not isinstance(pages, list):
            continue
        for index, item in enumerate(pages, start=1):
            if not isinstance(item, Mapping):
                continue
            item_page = _int(item.get("page") or item.get("page_num") or item.get("page_number") or index)
            if item_page != page_number:
                continue
            text = item.get("text") or item.get("content") or item.get("markdown") or ""
            if isinstance(text, list):
                text = " ".join(str(part or "") for part in text)
            return _clean_text(text, limit=12000)
    return ""


def _value(item: Any, key: str, default: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _clean_text(value: Any, limit: int = 4000) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _find_confidence(value: Any, keys: tuple[str, ...]) -> float | None:
    if not isinstance(value, Mapping):
        return None
    for key in keys:
        if key in value:
            found = _confidence(value.get(key))
            if found is not None:
                return found
    for nested in value.values():
        found = _find_confidence(nested, keys)
        if found is not None:
            return found
    return None


def _caption_is_weak(caption: str) -> bool:
    normalized = re.sub(r"^(?:figure|fig\.?|图|table|表)\s*[a-z]?\d*\s*[:：.-]?", "", caption, flags=re.IGNORECASE)
    return len(normalized.strip()) < 18


def _looks_garbled(text: str) -> bool:
    if _GARBLED_RE.search(text):
        return True
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    suspicious = sum(1 for char in compact if not (char.isprintable() and (char.isalnum() or char in "，。；：,.!?%+-_/()[]{}'\"")))
    return suspicious / len(compact) > 0.22


def _first_visual_reference(query: str) -> str:
    match = _VISUAL_REFERENCE_RE.search(str(query or ""))
    return match.group(0) if match else ""


def _normalize_reference_text(value: str) -> str:
    return re.sub(r"[\s.:：。_\-()（）\[\]]+", "", str(value or "").lower())


def _bbox_hash(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return ""
    try:
        normalized = [round(float(item), 4) for item in value]
    except (TypeError, ValueError):
        return ""
    raw = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
