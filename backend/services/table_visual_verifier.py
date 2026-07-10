"""Selective visual verification for numeric-table QA.

The normal RAG path should stay text-first.  This module only adds a visual
cell-extraction pass when the text table evidence looks risky, mirroring the
PaperQA idea of deferring to the table image when markdown is malformed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from services.chat_service import call_ai_api

logger = logging.getLogger(__name__)

_VISUAL_CACHE: dict[str, dict] = {}
_VALID_MODES = {"off", "auto", "always"}
_TABLE_REF_RE = re.compile(r"\btable\s*\.?\s*(\d+[a-z]?)\b|表\s*\.?\s*(\d+[a-z]?)", re.IGNORECASE)
_COLUMN_PLACEHOLDER_RE = re.compile(r"\bcolumn\s*\d+\b|列\s*\d+", re.IGNORECASE)


def resolve_visual_mode(custom_params: Optional[dict]) -> str:
    """Return off/auto/always for numeric-table visual verification."""
    value = None
    if isinstance(custom_params, dict):
        for key in (
            "numeric_table_visual_verification",
            "table_visual_verification",
            "visual_table_verification",
        ):
            if key in custom_params:
                value = custom_params.get(key)
                break
    if value is None:
        return "auto"
    normalized = str(value).strip().lower()
    aliases = {
        "true": "always",
        "1": "always",
        "yes": "always",
        "on": "always",
        "enabled": "always",
        "false": "off",
        "0": "off",
        "no": "off",
        "disabled": "off",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in _VALID_MODES else "auto"


def looks_vision_capable_model(provider: str = "", model: str = "") -> bool:
    provider_l = str(provider or "").lower()
    model_l = str(model or "").lower()
    if provider_l in {"gemini", "openai", "openai_native", "anthropic", "moonshot", "grok", "xai"}:
        return True
    return bool(
        re.search(
            r"vision|visual|\bvl\b|vlm|gpt-4o|gpt-5|o3|o4|gemini|claude-3|"
            r"qwen.*vl|internvl|glm-4v|kimi.*vision|moonshot.*vision|grok",
            model_l,
        )
    )


def should_verify_numeric_table_visual(
    *,
    query: str,
    segments: list[dict],
    mode: str = "auto",
    provider: str = "",
    model: str = "",
) -> tuple[bool, list[str]]:
    """Deterministic risk gate for table-image verification."""
    mode = mode if mode in _VALID_MODES else "auto"
    reasons: list[str] = []
    if mode == "off":
        return False, ["mode_off"]
    if not segments:
        return False, ["no_segments"]
    if mode == "auto" and not looks_vision_capable_model(provider, model):
        return False, ["model_not_vision_capable"]
    if mode == "always":
        return True, ["mode_always"]

    table_refs = _extract_table_refs(query)
    if table_refs:
        joined = _segments_text(segments)
        if not any(ref in joined for ref in table_refs):
            reasons.append("explicit_table_ref_missing_in_text_evidence")

    row_keys = _extract_query_row_keys(query)
    if row_keys:
        joined_norm = _norm(_segments_text(segments))
        missing = [key for key in row_keys if key not in joined_norm]
        if missing:
            reasons.append("explicit_row_key_missing")

    headers = "\n".join(
        str(segment.get("numeric_table_exact_context_header") or segment.get("table_header") or "")
        for segment in segments
        if isinstance(segment, dict)
    )
    if _header_has_placeholders_or_duplicates(headers):
        reasons.append("header_ambiguous")

    max_cell_count = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        cells = segment.get("cell_evidence_units") or []
        if isinstance(cells, list):
            max_cell_count = max(max_cell_count, len(cells))
    if max_cell_count >= 10:
        reasons.append("wide_table_row")

    if _has_overpacked_metric_cells(segments):
        reasons.append("overpacked_metric_cell")

    has_execution = any(
        str(segment.get("segment_role") or "") == "numeric_table_execution"
        or "[numeric table execution]" in str(segment.get("text") or "").lower()
        for segment in segments
        if isinstance(segment, dict)
    )
    if not has_execution:
        reasons.append("no_deterministic_numeric_execution")

    return bool(reasons), reasons


async def maybe_verify_numeric_table_visual(
    *,
    query: str,
    doc_id: str,
    doc_data: dict,
    pdf_path: Path | None,
    segments: list[dict],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
    custom_params: Optional[dict] = None,
) -> tuple[dict, dict]:
    mode = resolve_visual_mode(custom_params)
    diagnostics = {
        "enabled": mode != "off",
        "mode": mode,
        "triggered": False,
        "reasons": [],
        "skipped_reason": "",
    }
    should_verify, reasons = should_verify_numeric_table_visual(
        query=query,
        segments=segments,
        mode=mode,
        provider=provider,
        model=model,
    )
    diagnostics["reasons"] = reasons
    if not should_verify:
        diagnostics["skipped_reason"] = reasons[0] if reasons else "not_risky"
        return {}, diagnostics
    if not api_key:
        diagnostics["skipped_reason"] = "missing_api_key"
        return {}, diagnostics
    if not pdf_path or not Path(pdf_path).exists():
        diagnostics["skipped_reason"] = "missing_pdf"
        return {}, diagnostics

    target = _select_target_table(query, segments, doc_data)
    if not target:
        diagnostics["skipped_reason"] = "no_table_target"
        return {}, diagnostics
    page = target.get("page")
    if not isinstance(page, int) or page <= 0:
        diagnostics["skipped_reason"] = "missing_page"
        return {}, diagnostics

    cache_key = _cache_key(doc_id, query, target, provider, model)
    cached = _VISUAL_CACHE.get(cache_key)
    if cached:
        diagnostics.update({"triggered": True, "cache_hit": True, **cached.get("diagnostics", {})})
        return dict(cached.get("segment") or {}), diagnostics

    try:
        image_b64, crop_meta = render_table_crop_base64(pdf_path, page, target.get("bbox"))
    except Exception as exc:
        diagnostics["skipped_reason"] = f"render_failed:{type(exc).__name__}"
        logger.debug("[TableVisual] render failed: %s", exc)
        return {}, diagnostics

    prompt_text = _build_visual_prompt(query, target, segments)
    messages = [
        {
            "role": "system",
            "content": (
                "You verify numeric answers from academic paper tables. "
                "Extract only cells visible in the table image. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]
    diagnostics.update({"triggered": True, "page": page, "crop": crop_meta})
    try:
        response = await call_ai_api(
            messages,
            api_key,
            model,
            provider,
            endpoint=endpoint,
            max_tokens=700,
            temperature=0,
            purpose="numeric_table_visual_verification",
        )
    except Exception as exc:
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return {}, diagnostics
    if isinstance(response, dict) and response.get("error"):
        diagnostics["error"] = str(response.get("error"))
        return {}, diagnostics

    content = _extract_response_content(response)
    parsed = _parse_json_object(content)
    if not parsed:
        diagnostics["error"] = "invalid_json"
        diagnostics["raw_preview"] = content[:300]
        return {}, diagnostics

    segment = _build_visual_segment(
        parsed,
        query=query,
        target=target,
        crop_meta=crop_meta,
        response=response,
    )
    diagnostics["confidence"] = segment.get("visual_confidence")
    diagnostics["used_provider"] = response.get("_used_provider")
    diagnostics["used_model"] = response.get("_used_model")
    _VISUAL_CACHE[cache_key] = {"segment": segment, "diagnostics": diagnostics}
    return segment, diagnostics


def render_table_crop_base64(pdf_path: Path, page_number: int, bbox: Optional[list[float]], *, dpi: int = 180) -> tuple[str, dict]:
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[max(0, int(page_number) - 1)]
        page_rect = page.rect
        if bbox and len(bbox) >= 4:
            rect = fitz.Rect(*[float(v) for v in bbox[:4]])
            x_pad = max(18.0, rect.width * 0.08)
            y_pad = max(24.0, rect.height * 0.12)
            clip = fitz.Rect(
                max(page_rect.x0, rect.x0 - x_pad),
                max(page_rect.y0, rect.y0 - y_pad),
                min(page_rect.x1, rect.x1 + x_pad),
                min(page_rect.y1, rect.y1 + y_pad),
            )
        else:
            clip = page_rect
        pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False, annots=False)
        return base64.b64encode(pix.tobytes("png")).decode("ascii"), {
            "page": int(page_number),
            "bbox": [round(float(v), 2) for v in [clip.x0, clip.y0, clip.x1, clip.y1]],
            "dpi": dpi,
            "width": pix.width,
            "height": pix.height,
        }
    finally:
        doc.close()


def _build_visual_prompt(query: str, target: dict, segments: list[dict]) -> str:
    text_evidence = _segments_text(segments)[:5000]
    return f"""用户问题:
{query}

目标表格:
- table_ref: {target.get('table_ref') or ''}
- table_id: {target.get('table_id') or ''}
- caption: {target.get('caption') or ''}

文本检索证据（可能有错位、漏行或表头错误，仅作参考）:
{text_evidence}

请直接查看图片中的表格，抽取回答该问题所需的行和单元格。
只返回 JSON，不要 Markdown，不要解释。格式:
{{
  "table_id": "Table 4",
  "matched_row": "ID D / method name / row label",
  "cells": {{"column name": "visible value"}},
  "confidence": 0.0,
  "notes": "short note when row/column is uncertain",
  "supports_text_result": true,
  "corrected_answer_hint": "short factual hint"
}}"""


def _build_visual_segment(
    parsed: dict,
    *,
    query: str,
    target: dict,
    crop_meta: dict,
    response: dict,
) -> dict:
    cells = parsed.get("cells") if isinstance(parsed.get("cells"), dict) else {}
    cell_lines = [f"{key} = {value}" for key, value in cells.items() if str(key).strip()]
    matched_row = str(parsed.get("matched_row") or "").strip()
    table_id = str(parsed.get("table_id") or target.get("table_id") or target.get("table_ref") or "").strip()
    confidence = _coerce_float(parsed.get("confidence"), 0.0)
    notes = str(parsed.get("notes") or "").strip()
    hint = str(parsed.get("corrected_answer_hint") or "").strip()
    lines = ["[Numeric Table Visual Verification]"]
    if table_id:
        lines.append(f"Table ID: {table_id}")
    if target.get("caption"):
        lines.append(f"Table Caption: {target.get('caption')}")
    if matched_row:
        lines.append(f"Matched Row: {matched_row}")
    if cell_lines:
        lines.append("Visual Cells:")
        lines.extend(f"- {line}" for line in cell_lines)
    if hint:
        lines.append(f"Corrected Answer Hint: {hint}")
    if notes:
        lines.append(f"Notes: {notes}")
    lines.append(f"Confidence: {confidence:.2f}")
    page = int(crop_meta.get("page") or target.get("page") or 0)
    return {
        "text": "\n".join(lines),
        "segment_role": "numeric_table_visual_verification",
        "context_id": f"{target.get('context_id') or target.get('table_id') or target.get('table_ref') or 'table'}:visual_verification",
        "evidence_id": f"{target.get('evidence_id') or target.get('table_id') or target.get('table_ref') or 'table'}:visual_verification",
        "table_id": table_id,
        "table_caption": target.get("caption") or "",
        "page_range": [page, page] if page else [],
        "bbox": crop_meta.get("bbox") or target.get("bbox") or [],
        "visual_cells": cells,
        "visual_matched_row": matched_row,
        "visual_confidence": confidence,
        "visual_supports_text_result": bool(parsed.get("supports_text_result")),
        "visual_notes": notes,
        "visual_answer_hint": hint,
        "visual_query": query,
        "used_provider": response.get("_used_provider"),
        "used_model": response.get("_used_model"),
        "synthetic_description": True,
    }


def _select_target_table(query: str, segments: list[dict], doc_data: dict) -> dict:
    refs = _extract_table_refs(query)
    candidates: list[dict] = []
    for bundle in (doc_data or {}).get("structured_table_bundles") or []:
        if isinstance(bundle, dict):
            candidates.append(_target_from_record(bundle))
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        candidates.append(_target_from_record(segment))
    candidates = [candidate for candidate in candidates if candidate.get("page")]
    if refs:
        for candidate in candidates:
            joined = " ".join(str(candidate.get(key) or "") for key in ("table_ref", "table_id", "caption", "text")).lower()
            if any(ref in joined for ref in refs):
                return candidate
    return candidates[0] if candidates else {}


def _target_from_record(record: dict) -> dict:
    page = _first_page(record)
    bbox = _record_bbox(record)
    caption = str(record.get("numeric_table_exact_context_caption") or record.get("table_caption") or "").strip()
    table_id = str(record.get("table_id") or record.get("table_bundle_id") or "").strip()
    table_ref = next(iter(_extract_table_refs(f"{table_id} {caption}")), "")
    return {
        "page": page,
        "bbox": bbox,
        "caption": caption,
        "table_id": table_id,
        "table_ref": table_ref,
        "text": str(record.get("text") or record.get("bundle_text") or ""),
        "context_id": record.get("context_id") or record.get("bundle_id") or "",
        "evidence_id": record.get("evidence_id") or "",
    }


def _record_bbox(record: dict) -> list[float]:
    for key in ("bbox", "table_bbox", "bounding_box"):
        bbox = _normalize_bbox(record.get(key))
        if bbox:
            return bbox
    bboxes: list[list[float]] = []
    for key in ("bounding_boxes", "table_bboxes"):
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                bbox = _normalize_bbox(item)
                if bbox:
                    bboxes.append(bbox)
    for unit_key in ("cell_evidence_units", "evidence_units"):
        units = record.get(unit_key)
        if isinstance(units, list):
            for unit in units:
                if not isinstance(unit, dict):
                    continue
                for key in ("bbox", "cell_bbox", "row_bbox", "bounding_box"):
                    bbox = _normalize_bbox(unit.get(key))
                    if bbox:
                        bboxes.append(bbox)
    return _merge_bboxes(bboxes)


def _first_page(record: dict) -> int:
    for key in ("page", "page_start"):
        try:
            page = int(record.get(key) or 0)
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            return page
    for key in ("page_range", "pages"):
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                try:
                    page = int(item)
                except (TypeError, ValueError):
                    continue
                if page > 0:
                    return page
    return 0


def _normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        x0, y0, x1, y1 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return []
    if x1 <= x0 or y1 <= y0:
        return []
    return [x0, y0, x1, y1]


def _merge_bboxes(bboxes: list[list[float]]) -> list[float]:
    valid = [bbox for bbox in bboxes if bbox and len(bbox) >= 4]
    if not valid:
        return []
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


def _extract_response_content(response: dict) -> str:
    choices = response.get("choices") if isinstance(response, dict) else []
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    return str((message or {}).get("content") or "")


def _parse_json_object(text: str) -> dict:
    sample = str(text or "").strip()
    if not sample:
        return {}
    sample = re.sub(r"^```(?:json)?\s*|\s*```$", "", sample, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(sample)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", sample, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _cache_key(doc_id: str, query: str, target: dict, provider: str, model: str) -> str:
    raw = json.dumps(
        {
            "doc_id": doc_id,
            "query": re.sub(r"\s+", " ", query or "").strip().lower(),
            "page": target.get("page"),
            "bbox": target.get("bbox"),
            "table": target.get("table_id") or target.get("table_ref"),
            "provider": provider,
            "model": model,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_table_refs(text: str) -> set[str]:
    refs: set[str] = set()
    for match in _TABLE_REF_RE.finditer(str(text or "")):
        token = (match.group(1) or match.group(2) or "").strip().lower()
        if token:
            refs.add(f"table {token}")
            refs.add(f"table-{token}")
            refs.add(f"表 {token}")
    return refs


def _extract_query_row_keys(query: str) -> set[str]:
    keys: set[str] = set()
    sample = str(query or "")
    patterns = (
        r"\bID\s*[:：]?\s*([A-Za-z0-9]+)\b",
        r"\bRow\s*[:：]?\s*([A-Za-z0-9]+)\b",
        r"(?:配置|实验)\s*ID\s*[:：]?\s*([A-Za-z0-9]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, sample, re.IGNORECASE):
            value = _norm(match.group(1))
            if value and len(value) <= 8:
                keys.add(value)
                if "id" in pattern.lower():
                    keys.add(f"id{value}")
                if "row" in pattern.lower():
                    keys.add(f"row{value}")
    return keys


def _header_has_placeholders_or_duplicates(header: str) -> bool:
    sample = str(header or "")
    if _COLUMN_PLACEHOLDER_RE.search(sample):
        return True
    cells = [_norm(part) for part in re.split(r"\||;|,", sample) if _norm(part)]
    if len(cells) < 4:
        return False
    repeated = len(cells) - len(set(cells))
    blankish = sum(1 for cell in cells if cell in {"", "none", "null", "na"})
    return repeated >= 2 or blankish >= 2


def _has_overpacked_metric_cells(segments: list[dict]) -> bool:
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        cells = segment.get("cell_evidence_units") or []
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            value = str(cell.get("content") or cell.get("cell_text") or "")
            if len(re.findall(r"[-+]?\d+(?:\.\d+)?", value)) >= 3:
                return True
    return False


def _segments_text(segments: list[dict]) -> str:
    parts: list[str] = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        for key in (
            "text",
            "table_id",
            "table_caption",
            "numeric_table_exact_context_caption",
            "table_header",
            "numeric_table_exact_context_header",
            "numeric_table_exact_context_row_text",
            "numeric_table_projected_cells",
        ):
            value = str(segment.get(key) or "").strip()
            if value:
                parts.append(value)
    return "\n".join(parts).lower()


def _norm(value: str = "") -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fffωα]+", "", str(value or "").casefold())


def _coerce_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


__all__ = [
    "maybe_verify_numeric_table_visual",
    "render_table_crop_base64",
    "resolve_visual_mode",
    "should_verify_numeric_table_visual",
    "looks_vision_capable_model",
]
