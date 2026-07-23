"""Post-retrieval visual verification for risky numeric table answers.

This module deliberately sits *after* the normal text retrieval path.  It
never changes MinerU output, table bundles, or vector indexes.  A visual result
is useful to the answer only when it can be reconciled with the already
retrieved table/cell evidence; otherwise it is retained as a diagnostic.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from services.ai_cache_state import LEGACY_AI_CACHE_GENERATION, load_ai_cache_generation
from services.chat_service import call_ai_api as _default_call_ai_api
from services.document_parse_state import matches_parse_generation, read_parse_manifest
from services.visual_enrichment_service import (
    VisualTaskBudgetExceeded,
    VisualTaskPolicy,
    VisualTaskTimeoutError,
    execute_visual_task,
)
from services.visual_model_service import (
    VisualEnrichmentPolicy,
    VisualModelConfig,
    call_visual_model,
    resolve_visual_model_config,
)
from services.visual_risk_service import assess_visual_region_risk

try:
    from services.table_visual_metadata import build_table_visual_metadata
except ImportError:  # Kept for old documents while rolling out the metadata helper.
    build_table_visual_metadata = None  # type: ignore

logger = logging.getLogger(__name__)

# Keep the historic module-level hook available for downstream integrations and
# existing tests.  Normal execution still enters through ``call_visual_model``.
call_ai_api = _default_call_ai_api

_VALID_MODES = {"off", "auto", "always"}
_TERMINAL_STATES = {"confirmed", "conflict", "indeterminate", "failed", "stale"}
_TASK_STATES = {"queued", "running", *_TERMINAL_STATES}
_CACHE_SCHEMA_VERSION = 6
_TASK_STORE_VERSION = 1
_VISUAL_PROMPT_VERSION = 3
_VISUAL_CROP_VERSION = 2
_VISUAL_PURPOSE = "numeric_table_verification"
_DEFAULT_CONCURRENCY = 2
_DEFAULT_FAILURE_COOLDOWN_S = 300.0
_DEFAULT_STALE_TASK_S = 600.0
_DEFAULT_INDETERMINATE_TTL_S = 86400.0
_VISUAL_CACHE: dict[str, dict] = {}
_VISUAL_PENDING: set[str] = set()
_TASK_WRITE_LOCK = threading.Lock()
_AI_CACHE_IDENTITY_UNAVAILABLE = "ai_cache_identity_unavailable"


class _TableVisualIdentityUnavailable(RuntimeError):
    """Raised when a visual cache identity cannot be read safely."""

    def __init__(self, doc_id: str, cause: Exception):
        super().__init__("AI cache generation identity is unavailable")
        self.doc_id = str(doc_id or "")
        self.cause_type = type(cause).__name__


async def _call_table_visual_model(
    *,
    messages: list[dict[str, Any]],
    config: VisualModelConfig,
    middlewares: list[Any] | None,
    max_tokens: int,
    temperature: float,
    purpose: str = "numeric_table_visual_verification",
) -> Any:
    """Use the shared VLM gateway while retaining the legacy injection hook."""
    if call_ai_api is not _default_call_ai_api:
        return await call_ai_api(
            messages,
            config.api_key,
            config.model,
            config.provider,
            endpoint=config.endpoint,
            middlewares=middlewares,
            max_tokens=max_tokens,
            temperature=temperature,
            purpose=purpose,
        )
    return await call_visual_model(
        messages=messages,
        config=config,
        middlewares=middlewares,
        max_tokens=max_tokens,
        temperature=temperature,
        purpose=purpose,
    )

_TABLE_REF_RE = re.compile(r"\btable\s*\.?\s*(\d+[a-z]?)\b|表\s*\.?\s*(\d+[a-z]?)", re.IGNORECASE)
_COLUMN_PLACEHOLDER_RE = re.compile(r"\bcolumn\s*\d+\b|列\s*\d+", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z][a-z0-9._/%-]{1,}|[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?%?", re.IGNORECASE)


class VisualTableResponse(BaseModel):
    """The narrow response contract requested from the vision model."""

    table_id: str = ""
    matched_row: str = ""
    cells: dict[str, str] = Field(default_factory=dict)
    confidence: float | None = None
    notes: str = ""
    supports_text_result: bool | None = None
    corrected_answer_hint: str = ""

    @field_validator("table_id", "matched_row", "notes", "corrected_answer_hint", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @field_validator("cells", mode="before")
    @classmethod
    def _coerce_cells(cls, value: Any) -> dict[str, str]:
        if isinstance(value, list):
            value = {
                item.get("column") or item.get("key") or "": item.get("value") or item.get("content") or ""
                for item in value
                if isinstance(item, dict)
            }
        if not isinstance(value, dict):
            raise ValueError("cells must be an object")
        result: dict[str, str] = {}
        for key, item in value.items():
            clean_key = re.sub(r"\s+", " ", str(key or "")).strip()
            clean_value = re.sub(r"\s+", " ", str(item or "")).strip()
            if clean_key and clean_value:
                result[clean_key] = clean_value
        return result

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc


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


_VISION_CAPABLE_MODEL_RULES: dict[str, re.Pattern[str]] = {
    # Keep these provider/model prefixes aligned with
    # frontend/src/utils/visionDetectorUtils.js.  A provider alone never
    # proves image-input support: users may configure text-only, embedding, or
    # legacy models under the same provider.
    "openai": re.compile(r"^(gpt-4o|gpt-4-turbo|gpt-4\.1|gpt-5|o3|o4)", re.IGNORECASE),
    "openai_native": re.compile(r"^(gpt-4o|gpt-4-turbo|gpt-4\.1|gpt-5|o3|o4)", re.IGNORECASE),
    "anthropic": re.compile(r"^(claude-3|claude-(fable|mythos|sonnet|opus|haiku))", re.IGNORECASE),
    "gemini": re.compile(r"^gemini-(2|[3-9])", re.IGNORECASE),
    "qwen": re.compile(r"^(qwen-vl|qwen-max)", re.IGNORECASE),
    "aliyun": re.compile(r"^(qwen-vl|qwen-max|qwen3\.[567]|qwen3\.5-omni)", re.IGNORECASE),
    "grok": re.compile(r"^(grok-vision|grok-4)", re.IGNORECASE),
    "xai": re.compile(r"^(grok-vision|grok-4)", re.IGNORECASE),
    "minimax": re.compile(r"^(minimax-m3|abab6\.5)", re.IGNORECASE),
    "doubao": re.compile(r"^(doubao-1\.5-pro|doubao-seed)", re.IGNORECASE),
    "moonshot": re.compile(r"^(moonshot-v1|kimi-k2)", re.IGNORECASE),
}


def looks_vision_capable_model(
    provider: str = "",
    model: str = "",
    *,
    capability_hint: Optional[bool] = None,
) -> bool:
    """Return whether a selected model can accept images.

    The desktop client sends ``visual_enabled`` after consulting its model
    metadata, including explicit ``vision`` tags for custom providers.  That
    hint is authoritative when present.  API callers without that metadata use
    the same conservative provider-plus-model matrix as the client.
    """
    if capability_hint is not None:
        return bool(capability_hint)

    provider_l = str(provider or "").strip().lower()
    model_l = str(model or "").strip().lower()
    rule = _VISION_CAPABLE_MODEL_RULES.get(provider_l)
    if rule and rule.search(model_l):
        return True

    # Preserve the client-side fallback for dynamically registered VLMs while
    # avoiding broad "visual"/"vlm" keyword guesses that mark text models as
    # image-capable.
    return "vision" in model_l or "-vl" in model_l


def _visual_capability_hint(custom_params: Optional[dict]) -> Optional[bool]:
    """Extract an explicit client capability decision without coercing absence."""
    if not isinstance(custom_params, dict) or "visual_enabled" not in custom_params:
        return None
    value = custom_params.get("visual_enabled")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def should_verify_numeric_table_visual(
    *,
    query: str,
    segments: list[dict],
    mode: str = "auto",
    provider: str = "",
    model: str = "",
    capability_hint: Optional[bool] = None,
) -> tuple[bool, list[str]]:
    """Deterministic risk gate for table-image verification."""
    mode = mode if mode in _VALID_MODES else "auto"
    reasons: list[str] = []
    if mode == "off":
        return False, ["mode_off"]
    if not looks_vision_capable_model(provider, model, capability_hint=capability_hint):
        return False, ["model_not_vision_capable"]
    if mode == "always":
        return True, ["mode_always"]
    if not segments:
        return False, ["no_segments"]

    table_refs = _extract_table_refs(query)
    if table_refs:
        joined = _segments_text(segments)
        if not any(_table_ref_in_text(ref, joined) for ref in table_refs):
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
    background: bool = False,
    visual_config: VisualModelConfig | None = None,
    visual_policy: VisualEnrichmentPolicy | None = None,
) -> tuple[dict, dict]:
    """Run or enqueue a post-retrieval visual check.

    A non-confirmed result intentionally returns no context segment.  The
    caller can still expose its diagnostic/task status without allowing the
    VLM to alter the answer or citations.
    """
    resolved_visual_config = (
        visual_policy.select(risk_level="medium", purpose=_VISUAL_PURPOSE)
        if visual_policy is not None
        else visual_config or resolve_visual_model_config(
            primary_provider=provider,
            primary_model=model,
            primary_api_key=api_key,
            primary_endpoint=endpoint,
        )
    )
    provider = resolved_visual_config.provider
    model = resolved_visual_config.model
    api_key = resolved_visual_config.api_key
    endpoint = resolved_visual_config.endpoint
    mode = resolve_visual_mode(custom_params)
    capability_hint = _visual_capability_hint(custom_params)
    if visual_policy is not None and resolved_visual_config.source == "local_tier":
        capability_hint = True
    diagnostics: dict[str, Any] = {
        "enabled": mode != "off",
        "mode": mode,
        "triggered": False,
        "reasons": [],
        "skipped_reason": "",
        "state": "not_started",
        "verdict": "indeterminate",
        "visual_model": resolved_visual_config.public_metadata(),
        "visual_policy": visual_policy.public_metadata() if visual_policy is not None else {},
    }
    should_verify, reasons = should_verify_numeric_table_visual(
        query=query,
        segments=segments,
        mode=mode,
        provider=provider,
        model=model,
        capability_hint=capability_hint,
    )
    diagnostics["reasons"] = reasons
    if not should_verify:
        diagnostics.update({"skipped_reason": reasons[0] if reasons else "not_risky", "state": "skipped"})
        return {}, diagnostics
    if not resolved_visual_config.can_call:
        diagnostics.update({"skipped_reason": "missing_visual_model", "state": "skipped"})
        return {}, diagnostics
    if not pdf_path or not Path(pdf_path).exists():
        diagnostics.update({"skipped_reason": "missing_pdf", "state": "skipped"})
        return {}, diagnostics

    target = _select_target_table(query, segments, doc_data)
    if not target:
        diagnostics.update({"skipped_reason": "no_table_target", "state": "indeterminate"})
        return {}, diagnostics
    if target.get("selection_ambiguous"):
        diagnostics.update({
            "triggered": True,
            "state": "indeterminate",
            "verdict": "indeterminate",
            "skipped_reason": "ambiguous_table_target",
            "candidate_count": target.get("candidate_count", 0),
            "selection_score": target.get("selection_score", 0),
        })
        return {}, diagnostics
    page = target.get("page")
    if not isinstance(page, int) or page <= 0:
        diagnostics.update({"skipped_reason": "missing_page", "state": "indeterminate"})
        return {}, diagnostics

    target = dict(target)
    target["requested_rows"] = sorted(_extract_query_row_keys(query))
    target["requested_columns"] = _extract_requested_columns(query, target)
    target["selected_row"] = _select_target_row(target, query)
    if isinstance(target.get("selected_row"), dict):
        row_page = _first_page(target["selected_row"])
        if row_page > 0:
            target["page"] = row_page
            page = row_page
    _ensure_target_metadata(target)
    _attach_document_parse_identity(target, doc_data=doc_data, doc_id=doc_id)
    target_bboxes = target.get("bboxes")
    risk_bbox = target.get("bbox")
    if not risk_bbox and isinstance(target_bboxes, list) and target_bboxes:
        risk_bbox = target_bboxes[0]
    region_risk = assess_visual_region_risk(
        page=int(target.get("page") or 0),
        bbox=risk_bbox,
        source=str(target.get("selected_source") or target.get("source") or ""),
        caption=str(target.get("caption") or target.get("table_caption") or ""),
        page_text=_segments_text(segments),
        structure_confidence=target.get("confidence"),
        ocr_confidence=target.get("ocr_confidence"),
        query=query,
        text_evidence=_segments_text(segments),
    )
    if visual_policy is not None:
        resolved_visual_config = visual_policy.select(
            risk_level=region_risk.level,
            purpose=_VISUAL_PURPOSE,
        )
        provider = resolved_visual_config.provider
        model = resolved_visual_config.model
        api_key = resolved_visual_config.api_key
        endpoint = resolved_visual_config.endpoint
        if not resolved_visual_config.can_call:
            diagnostics.update({
                "skipped_reason": "visual_policy_model_unavailable",
                "state": "skipped",
                "visual_risk": region_risk.to_dict(),
                "visual_model": resolved_visual_config.public_metadata(),
            })
            return {}, diagnostics
    diagnostics.update({
        "table_id": target.get("table_id") or target.get("table_ref") or "",
        "table_caption": target.get("caption") or "",
        "table_instance_id": target.get("table_instance_id") or "",
        "visual_risk": region_risk.to_dict(),
        "visual_model": resolved_visual_config.public_metadata(),
    })

    visual_cache_identity = (
        f"{visual_policy.identity}:{resolved_visual_config.identity}"
        if visual_policy is not None
        else resolved_visual_config.identity
    )
    try:
        current_ai_cache_generation = _load_table_visual_ai_cache_generation(doc_id)
    except _TableVisualIdentityUnavailable as exc:
        diagnostics.update(_identity_unavailable_diagnostics(diagnostics, exc))
        return {}, diagnostics
    cache_key = _cache_key(
        doc_id,
        query,
        target,
        provider,
        model,
        visual_cache_identity,
        current_ai_cache_generation,
    )
    task_id = _task_id(cache_key)
    existing = _load_task_record(cache_key)
    if _task_matches_request(
        existing,
        doc_id=doc_id,
        target=target,
        provider=provider,
        model=model,
        visual_model_identity=visual_cache_identity,
        ai_cache_generation=current_ai_cache_generation,
    ):
        state = str(existing.get("state") or "")
        current_manifest = read_parse_manifest(doc_data, doc_id=doc_id)
        try:
            identity_matches = _record_matches_parse_identity(
                existing,
                current_manifest,
                current_ai_cache_generation=current_ai_cache_generation,
            )
        except _TableVisualIdentityUnavailable as exc:
            diagnostics.update(_identity_unavailable_diagnostics(diagnostics, exc))
            return {}, diagnostics
        if not identity_matches:
            existing = _mark_task_stale(
                cache_key,
                existing,
                current_manifest=current_manifest,
                current_ai_cache_generation=current_ai_cache_generation,
                diagnostics=diagnostics,
            )
            diagnostics.update(_task_diagnostics(existing, cache_hit=True))
            return {}, diagnostics
        if state in {"queued", "running"}:
            if _task_is_stale(existing):
                _VISUAL_PENDING.discard(cache_key)
                diagnostics["stale_task_recovered"] = True
            else:
                diagnostics.update(_task_diagnostics(existing, cache_hit=True))
                return {}, diagnostics
        if state == "failed" and _failure_cooldown_active(existing):
            diagnostics.update(_task_diagnostics(existing, cache_hit=True))
            diagnostics["skipped_reason"] = "failure_cooldown"
            return {}, diagnostics
        if state == "indeterminate" and _indeterminate_cache_active(existing):
            diagnostics.update(_task_diagnostics(existing, cache_hit=True))
            diagnostics["skipped_reason"] = "indeterminate_cache"
            return {}, diagnostics
        if state in _TERMINAL_STATES:
            diagnostics.update(_task_diagnostics(existing, cache_hit=True))
            if state == "confirmed":
                return dict(existing.get("segment") or {}), diagnostics
            return {}, diagnostics

    request_info = _request_info(
        doc_id,
        query,
        target,
        provider,
        model,
        visual_cache_identity,
        current_ai_cache_generation,
    )
    if background and mode == "auto":
        record = _create_or_update_task(cache_key, request_info, state="queued", diagnostics=diagnostics)
        diagnostics.update(_task_diagnostics(record))
        diagnostics.update({"triggered": True, "background": True, "pending": True, "skipped_reason": "background_pending"})
        _schedule_visual_background_task(
            cache_key=cache_key,
            query=query,
            target=target,
            pdf_path=Path(pdf_path),
            segments=[dict(segment) for segment in (segments or []) if isinstance(segment, dict)],
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            custom_params=dict(custom_params or {}),
            diagnostics=dict(diagnostics),
            request_info=request_info,
            current_document_data=doc_data,
            visual_config=resolved_visual_config,
            visual_policy=visual_policy,
        )
        return {}, diagnostics

    return await _run_visual_verification(
        cache_key=cache_key,
        query=query,
        target=target,
        pdf_path=Path(pdf_path),
        segments=segments,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        custom_params=custom_params,
        diagnostics=diagnostics,
        request_info=request_info,
        current_document_data=doc_data,
        visual_config=resolved_visual_config,
        visual_policy=visual_policy,
    )


def rank_table_candidates(query: str, segments: list[dict], doc_data: dict) -> list[dict]:
    """Rank evidence-backed tables without falling back to the first candidate."""
    bundle_records = [
        record for record in (doc_data or {}).get("structured_table_bundles") or []
        if isinstance(record, dict)
    ]
    bundles_by_key: dict[str, dict] = {}
    candidates: list[dict] = []
    for record in bundle_records:
        target = _target_from_record(record, origin="bundle")
        if not target.get("page"):
            continue
        candidates.append(target)
        for key in _candidate_join_keys(target):
            bundles_by_key.setdefault(key, target)

    for record in segments or []:
        if not isinstance(record, dict):
            continue
        segment_target = _target_from_record(record, origin="segment")
        if not segment_target.get("page"):
            continue
        matched = next((bundles_by_key[key] for key in _candidate_join_keys(segment_target) if key in bundles_by_key), None)
        if matched:
            _merge_candidate_evidence(matched, segment_target)
        else:
            candidates.append(segment_target)

    deduped: dict[str, dict] = {}
    for candidate in candidates:
        _ensure_target_metadata(candidate)
        key = str(candidate.get("table_instance_id") or "") or _candidate_fingerprint(candidate)
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = candidate
        else:
            _merge_candidate_evidence(previous, candidate)

    refs = _extract_table_refs(query)
    requested_rows = _extract_query_row_keys(query)
    for candidate in deduped.values():
        score, reasons = _score_table_candidate(query, candidate, refs, requested_rows)
        candidate["selection_score"] = score
        candidate["selection_reasons"] = reasons
    return sorted(
        deduped.values(),
        key=lambda item: (
            -float(item.get("selection_score") or 0),
            -float(item.get("source_quality") or 0),
            int(item.get("page") or 10**9),
            str(item.get("table_instance_id") or ""),
        ),
    )


def _select_target_table(query: str, segments: list[dict], doc_data: dict) -> dict:
    ranked = rank_table_candidates(query, segments, doc_data)
    if not ranked:
        return {}
    selected = dict(ranked[0])
    selected["candidate_count"] = len(ranked)
    if len(ranked) > 1:
        margin = float(selected.get("selection_score") or 0) - float(ranked[1].get("selection_score") or 0)
        # An explicit table reference is a strong enough discriminator.  In its
        # absence a near tie is intentionally indeterminate instead of a guess.
        selected["selection_ambiguous"] = not _extract_table_refs(query) and margin < 8
        selected["selection_margin"] = round(margin, 2)
    else:
        selected["selection_ambiguous"] = False
    return selected


def _target_from_record(record: dict, *, origin: str = "") -> dict:
    page = _first_page(record)
    bbox = _record_bbox(record)
    caption = str(record.get("numeric_table_exact_context_caption") or record.get("table_caption") or "").strip()
    table_id = str(record.get("table_id") or record.get("table_bundle_id") or record.get("bundle_id") or "").strip()
    table_ref = next(iter(sorted(_extract_table_refs(f"{table_id} {caption}"))), "")
    evidence_units = record.get("evidence_units") if isinstance(record.get("evidence_units"), list) else []
    direct_cells = record.get("cell_evidence_units") if isinstance(record.get("cell_evidence_units"), list) else []
    if direct_cells:
        evidence_units = [
            *evidence_units,
            {
                "evidence_unit_type": "table_row",
                "content": record.get("numeric_table_exact_context_row_text") or record.get("row_text") or record.get("text") or "",
                "row_id": record.get("row_id") or "",
                "bbox": _record_bbox(record),
                "page": page,
                "cell_evidence_units": direct_cells,
            },
        ]
    return {
        "page": page,
        "page_end": _last_page(record, page),
        "pages": _record_pages(record, page),
        "bbox": bbox,
        "bboxes": _record_bboxes(record),
        "caption": caption,
        "table_id": table_id,
        "table_ref": table_ref,
        "table_header": str(record.get("numeric_table_exact_context_header") or record.get("table_header") or "").strip(),
        "text": str(record.get("text") or record.get("bundle_text") or record.get("table_body_markdown") or ""),
        "context_id": record.get("context_id") or record.get("bundle_id") or record.get("table_bundle_id") or "",
        "evidence_id": record.get("evidence_id") or record.get("evidence_unit_id") or "",
        "table_bundle_id": record.get("table_bundle_id") or record.get("bundle_id") or "",
        "table_instance_id": record.get("table_instance_id") or "",
        "table_source_hash": record.get("table_source_hash") or "",
        "source": record.get("source") or record.get("selected_source") or "",
        "source_quality": _source_quality(record),
        "evidence_units": evidence_units,
        "cell_evidence_units": direct_cells,
        "origin": origin,
    }


def _merge_candidate_evidence(target: dict, supplement: dict) -> None:
    for key in ("caption", "table_header", "text", "table_id", "table_ref", "table_bundle_id", "context_id", "evidence_id"):
        if not target.get(key) and supplement.get(key):
            target[key] = supplement[key]
    if not target.get("bbox") and supplement.get("bbox"):
        target["bbox"] = supplement["bbox"]
    target["source_quality"] = max(float(target.get("source_quality") or 0), float(supplement.get("source_quality") or 0))
    seen_units = {_unit_fingerprint(unit) for unit in target.get("evidence_units") or [] if isinstance(unit, dict)}
    for unit in supplement.get("evidence_units") or []:
        if not isinstance(unit, dict):
            continue
        fingerprint = _unit_fingerprint(unit)
        if fingerprint not in seen_units:
            target.setdefault("evidence_units", []).append(unit)
            seen_units.add(fingerprint)
    if supplement.get("cell_evidence_units"):
        target.setdefault("cell_evidence_units", []).extend(
            cell for cell in supplement["cell_evidence_units"] if isinstance(cell, dict)
        )


def _candidate_join_keys(candidate: dict) -> list[str]:
    values = (
        candidate.get("table_instance_id"),
        candidate.get("table_bundle_id"),
        candidate.get("context_id"),
        candidate.get("table_id"),
    )
    return [f"{index}:{_norm(str(value))}" for index, value in enumerate(values) if _norm(str(value))]


def _candidate_fingerprint(candidate: dict) -> str:
    raw = json.dumps(
        {
            "page": candidate.get("page"),
            "bbox": candidate.get("bbox"),
            "table": candidate.get("table_id"),
            "caption": candidate.get("caption"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _score_table_candidate(query: str, candidate: dict, refs: set[str], requested_rows: set[str]) -> tuple[float, list[str]]:
    joined = " ".join(
        str(candidate.get(key) or "")
        for key in ("table_ref", "table_id", "caption", "table_header", "text")
    )
    joined_norm = _norm(joined)
    score = 0.0
    reasons: list[str] = []
    matched_refs = [ref for ref in refs if _table_ref_in_text(ref, joined)]
    if matched_refs:
        score += 100.0
        reasons.append("explicit_table_ref")
    elif refs:
        score -= 20.0
    query_terms = {term for term in _query_terms(query) if term not in {"table", "表格", "多少", "what", "which"}}
    caption_terms = set(_query_terms(f"{candidate.get('caption') or ''} {candidate.get('table_id') or ''}"))
    overlap = len(query_terms & caption_terms)
    if overlap:
        score += min(20.0, overlap * 3.0)
        reasons.append("caption_match")
    matched_rows = [key for key in requested_rows if _candidate_matches_row_key(candidate, key)]
    if matched_rows:
        score += min(35.0, len(matched_rows) * 20.0)
        reasons.append("row_key_match")
    columns = _extract_requested_columns(query, candidate)
    if columns:
        score += min(20.0, len(columns) * 5.0)
        reasons.append("column_match")
    if candidate.get("origin") == "segment":
        score += 18.0
        reasons.append("retrieval_context")
    if candidate.get("evidence_units"):
        score += 6.0
    score += min(8.0, float(candidate.get("source_quality") or 0))
    return round(score, 2), reasons


def _source_quality(record: dict) -> float:
    try:
        explicit = float(record.get("table_selector_score"))
        if explicit > 0:
            return min(10.0, explicit)
    except (TypeError, ValueError):
        pass
    source = str(record.get("selected_source") or record.get("source") or "").lower()
    if "mineru" in source:
        return 8.0
    if source in {"odl", "pdf_native"}:
        return 7.0 if source == "odl" else 5.0
    return 3.0


def _ensure_target_metadata(target: dict) -> None:
    if target.get("table_instance_id") and target.get("table_source_hash"):
        return
    source_record = {
        "table_instance_id": target.get("table_instance_id"),
        "table_source_hash": target.get("table_source_hash"),
        "bundle_id": target.get("table_bundle_id"),
        "table_id": target.get("table_id"),
        "table_caption": target.get("caption"),
        "table_header": target.get("table_header"),
        "table_body_markdown": target.get("text"),
        "page_start": target.get("page"),
        "page_end": target.get("page_end"),
        "pages": target.get("pages"),
        "bounding_box": target.get("bbox"),
        "bounding_boxes": target.get("bboxes"),
        "evidence_units": target.get("evidence_units"),
        "source": target.get("source"),
    }
    metadata: dict[str, str] = {}
    if callable(build_table_visual_metadata):
        try:
            metadata = build_table_visual_metadata(source_record) or {}
        except Exception as exc:
            logger.debug("[TableVisual] metadata derivation failed: %s", exc)
    if not metadata:
        identity = json.dumps(source_record, ensure_ascii=False, sort_keys=True, default=str)
        metadata = {
            "table_instance_id": f"table-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}",
            "table_source_hash": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        }
    if not target.get("table_instance_id"):
        target["table_instance_id"] = metadata.get("table_instance_id") or ""
    if not target.get("table_source_hash"):
        target["table_source_hash"] = metadata.get("table_source_hash") or ""


def _attach_document_parse_identity(target: dict, *, doc_data: dict, doc_id: str) -> None:
    """Bind one visual check to the active primary-parser generation."""
    manifest = read_parse_manifest(doc_data if isinstance(doc_data, dict) else {}, doc_id=doc_id)
    target["parse_route"] = str(manifest.get("resolved_route") or "")
    target["parse_generation"] = str(manifest.get("generation") or "")
    target["document_source_hash"] = str(manifest.get("source_hash") or "")


def _target_bbox_hash(target: dict) -> str:
    payload = {
        "page": target.get("page"),
        "page_end": target.get("page_end"),
        "bbox": target.get("bbox"),
        "bboxes": target.get("bboxes"),
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _request_info(
    doc_id: str,
    query: str,
    target: dict,
    provider: str,
    model: str,
    visual_model_identity: str = "",
    ai_cache_generation: str = "",
) -> dict:
    return {
        "doc_id": doc_id,
        "document_source_hash": target.get("document_source_hash") or "",
        "parse_route": target.get("parse_route") or "",
        "parse_generation": target.get("parse_generation") or "",
        "ai_cache_generation": str(
            ai_cache_generation or _load_table_visual_ai_cache_generation(doc_id)
        ),
        "page": target.get("page") or 0,
        "bbox_hash": _target_bbox_hash(target),
        "purpose": _VISUAL_PURPOSE,
        "table_instance_id": target.get("table_instance_id") or "",
        "table_source_hash": target.get("table_source_hash") or "",
        "requested_rows": target.get("requested_rows") or [],
        "requested_columns": target.get("requested_columns") or [],
        "query_hash": _short_hash(_normal_query(query)),
        "provider": provider,
        "model": model,
        "visual_model_identity": visual_model_identity,
        "prompt_version": _VISUAL_PROMPT_VERSION,
        "crop_version": _VISUAL_CROP_VERSION,
    }


def _cache_key(
    doc_id: str,
    query: str,
    target: dict,
    provider: str,
    model: str,
    visual_model_identity: str = "",
    ai_cache_generation: str = "",
) -> str:
    _ensure_target_metadata(target)
    raw = json.dumps(
        {
            "schema_version": _CACHE_SCHEMA_VERSION,
            **_request_info(
                doc_id,
                query,
                target,
                provider,
                model,
                visual_model_identity,
                ai_cache_generation,
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _task_id(cache_key: str) -> str:
    return f"tv_{cache_key}"


def _cache_key_from_task_id(task_id: str) -> str:
    value = str(task_id or "").strip().lower()
    if not value.startswith("tv_"):
        return ""
    candidate = value[3:]
    return candidate if re.fullmatch(r"[a-f0-9]{64}", candidate) else ""


def _visual_cache_dir() -> Path:
    root = os.getenv("CHATPDF_TABLE_VISUAL_CACHE_DIR", "")
    if root:
        return Path(root)
    try:
        from runtime_mode import runtime
        if runtime.is_desktop:
            return Path(runtime.data_dir) / "table_visual_cache"
    except Exception:
        pass
    return Path(__file__).resolve().parents[2] / "data" / "table_visual_cache"


def _table_visual_data_dir() -> Path:
    """Return the application's canonical data root, independent of cache placement."""
    from runtime_mode import runtime
    return Path(runtime.data_dir)


def _load_table_visual_ai_cache_generation(doc_id: str) -> str:
    safe_doc_id = str(doc_id or "").strip()
    if not safe_doc_id:
        return LEGACY_AI_CACHE_GENERATION
    try:
        return str(load_ai_cache_generation(_table_visual_data_dir(), safe_doc_id) or LEGACY_AI_CACHE_GENERATION)
    except Exception as exc:
        logger.debug("[TableVisual] ai cache generation unavailable for %s: %s", safe_doc_id, exc)
        raise _TableVisualIdentityUnavailable(safe_doc_id, exc) from exc


def _identity_unavailable_diagnostics(
    diagnostics: Optional[dict],
    error: _TableVisualIdentityUnavailable,
) -> dict:
    """Describe a fail-closed identity read without mutating a cached task."""
    return {
        **_safe_diagnostics(diagnostics or {}),
        "state": "failed",
        "verdict": "indeterminate",
        "pending": False,
        "stale": False,
        "skipped_reason": _AI_CACHE_IDENTITY_UNAVAILABLE,
        "error": _AI_CACHE_IDENTITY_UNAVAILABLE,
        "error_code": _AI_CACHE_IDENTITY_UNAVAILABLE,
        "error_message": "AI cache identity is unavailable; cached visual results were not used.",
        "retryable": True,
        "identity_available": False,
        "identity_error_type": error.cause_type,
    }


def _identity_unavailable_record(
    cache_key: str,
    record: dict,
    error: _TableVisualIdentityUnavailable,
    diagnostics: Optional[dict] = None,
) -> dict:
    """Build a transient fail-closed record; never persist it over the cached task."""
    saved = record.get("diagnostics") if isinstance(record.get("diagnostics"), dict) else {}
    return {
        **dict(record or {}),
        "task_id": str((record or {}).get("task_id") or _task_id(cache_key)),
        "state": "failed",
        "segment": {},
        "diagnostics": _identity_unavailable_diagnostics(
            {**_safe_diagnostics(saved), **_safe_diagnostics(diagnostics or {})},
            error,
        ),
    }


def _visual_cache_path(cache_key: str) -> Path:
    safe_key = re.sub(r"[^a-f0-9]", "", str(cache_key).lower())[:64] or "cache"
    return _visual_cache_dir() / f"{safe_key}.json"


def _load_task_record(cache_key: str) -> dict:
    cached = _VISUAL_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return dict(cached)
    cache_path = _visual_cache_path(cache_key)
    if not cache_path.exists():
        return {}
    try:
        record = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(record, dict) or record.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return {}
    if record.get("store_version") not in (None, _TASK_STORE_VERSION):
        return {}
    _VISUAL_CACHE[cache_key] = dict(record)
    return dict(record)


def _persist_task_record(cache_key: str, record: dict) -> dict:
    safe_record = dict(record)
    safe_record["schema_version"] = _CACHE_SCHEMA_VERSION
    safe_record["store_version"] = _TASK_STORE_VERSION
    safe_record["task_id"] = _task_id(cache_key)
    safe_record["updated_at"] = time.time()
    with _TASK_WRITE_LOCK:
        _VISUAL_CACHE[cache_key] = dict(safe_record)
        cache_path = _visual_cache_path(cache_key)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(safe_record, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(cache_path)
        except Exception as exc:
            logger.debug("[TableVisual] task write failed: %s", exc)
    return dict(safe_record)


def clear_table_visual_verification_cache(doc_id: str) -> dict[str, int | str]:
    safe_doc_id = str(doc_id or "").strip()
    if not safe_doc_id:
        return {
            "doc_id": "",
            "memory_records_removed": 0,
            "disk_records_removed": 0,
        }

    with _TASK_WRITE_LOCK:
        memory_keys = [
            key for key, record in list(_VISUAL_CACHE.items())
            if isinstance(record, dict) and str(record.get("doc_id") or "") == safe_doc_id
        ]
        for key in memory_keys:
            _VISUAL_CACHE.pop(key, None)
            _VISUAL_PENDING.discard(key)

        disk_removed = 0
        cache_dir = _visual_cache_dir()
        if cache_dir.exists():
            for path in cache_dir.glob("*.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(record, dict) or str(record.get("doc_id") or "") != safe_doc_id:
                    continue
                cache_key = _cache_key_from_task_id(record.get("task_id") or "") or path.stem.lower()
                _VISUAL_PENDING.discard(cache_key)
                try:
                    path.unlink()
                    disk_removed += 1
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    logger.debug("[TableVisual] cache file delete failed for %s: %s", path, exc)

    return {
        "doc_id": safe_doc_id,
        "memory_records_removed": len(memory_keys),
        "disk_records_removed": disk_removed,
    }


def _create_or_update_task(cache_key: str, request_info: dict, *, state: str, diagnostics: Optional[dict] = None) -> dict:
    previous = _load_task_record(cache_key)
    now = time.time()
    record = {
        "created_at": previous.get("created_at", now),
        "state": state if state in _TASK_STATES else "failed",
        "attempts": int(previous.get("attempts") or 0),
        **request_info,
        "diagnostics": dict(diagnostics or previous.get("diagnostics") or {}),
        "segment": dict(previous.get("segment") or {}) if state == "confirmed" else {},
    }
    if state == "running":
        record["attempts"] += 1
        record["started_at"] = now
    if state in _TERMINAL_STATES:
        record["completed_at"] = now
    return _persist_task_record(cache_key, record)


def _complete_task(
    cache_key: str,
    request_info: dict,
    *,
    state: str,
    diagnostics: dict,
    segment: Optional[dict] = None,
) -> dict:
    previous = _load_task_record(cache_key)
    record = {
        "created_at": previous.get("created_at", time.time()),
        "state": state if state in _TERMINAL_STATES else "failed",
        "attempts": int(previous.get("attempts") or 0),
        **request_info,
        "diagnostics": _safe_diagnostics(diagnostics),
        "segment": dict(segment or {}) if state == "confirmed" else {},
        "completed_at": time.time(),
    }
    if state == "failed":
        record["retry_after"] = time.time() + _resolve_failure_cooldown_seconds(None)
    return _persist_task_record(cache_key, record)


def _record_matches_parse_identity(
    record: dict,
    current_manifest: dict,
    *,
    current_ai_cache_generation: Optional[str] = None,
) -> bool:
    generation = str((record or {}).get("parse_generation") or "")
    source_hash = str((record or {}).get("document_source_hash") or "")
    if not generation or not source_hash:
        return False
    if not matches_parse_generation(
        current_manifest,
        generation=generation,
        source_hash=source_hash,
    ):
        return False
    expected_ai_cache_generation = str(
        (record or {}).get("ai_cache_generation") or LEGACY_AI_CACHE_GENERATION
    )
    if current_ai_cache_generation is None:
        current_ai_cache_generation = _load_table_visual_ai_cache_generation(
            str((record or {}).get("doc_id") or "")
        )
    return bool(
        current_ai_cache_generation
        and expected_ai_cache_generation == current_ai_cache_generation
    )


def _stale_task_diagnostics(
    record: dict,
    current_manifest: dict,
    diagnostics: Optional[dict] = None,
    current_ai_cache_generation: Optional[str] = None,
) -> dict:
    saved = record.get("diagnostics") if isinstance(record.get("diagnostics"), dict) else {}
    expected_identity = {
        "parse_generation": str(record.get("parse_generation") or ""),
        "document_source_hash": str(record.get("document_source_hash") or ""),
        "parse_route": str(record.get("parse_route") or ""),
        "ai_cache_generation": str(record.get("ai_cache_generation") or LEGACY_AI_CACHE_GENERATION),
    }
    identity_error: _TableVisualIdentityUnavailable | None = None
    if current_ai_cache_generation is None:
        try:
            current_ai_cache_generation = _load_table_visual_ai_cache_generation(
                str(record.get("doc_id") or "")
            )
        except _TableVisualIdentityUnavailable as exc:
            identity_error = exc
    current_identity = {
        "parse_generation": str(current_manifest.get("generation") or ""),
        "document_source_hash": str(current_manifest.get("source_hash") or ""),
        "parse_route": str(current_manifest.get("resolved_route") or ""),
        "ai_cache_generation": str(current_ai_cache_generation or ""),
    }
    if identity_error is not None:
        current_identity.update({
            "ai_cache_generation_status": "unavailable",
            "ai_cache_generation_error_type": identity_error.cause_type,
        })
    return {
        **_safe_diagnostics(saved),
        **_safe_diagnostics(diagnostics or {}),
        "state": "stale",
        "verdict": "indeterminate",
        "pending": False,
        "stale": True,
        "stale_reason": "parse_identity_mismatch",
        "error": "parse_identity_stale",
        "error_code": "parse_identity_stale",
        "error_message": "The document parse or AI cache generation changed; this visual result is no longer valid.",
        "retryable": True,
        "expected_parse_identity": expected_identity,
        "current_parse_identity": current_identity,
    }


def _mark_task_stale(
    cache_key: str,
    record: dict,
    *,
    current_manifest: dict,
    diagnostics: Optional[dict] = None,
    current_ai_cache_generation: Optional[str] = None,
) -> dict:
    now = time.time()
    stale_record = {
        **dict(record or {}),
        "state": "stale",
        "segment": {},
        "diagnostics": _stale_task_diagnostics(
            record,
            current_manifest,
            diagnostics,
            current_ai_cache_generation,
        ),
        "completed_at": now,
        "stale_at": now,
    }
    stale_record.pop("retry_after", None)
    return _persist_task_record(cache_key, stale_record)


def _complete_task_for_current_parse(
    cache_key: str,
    request_info: dict,
    *,
    current_document_data: dict,
    state: str,
    diagnostics: dict,
    segment: Optional[dict] = None,
) -> dict:
    doc_id = str(request_info.get("doc_id") or "")
    current_manifest = read_parse_manifest(
        current_document_data if isinstance(current_document_data, dict) else {},
        doc_id=doc_id,
    )
    try:
        identity_matches = _record_matches_parse_identity(request_info, current_manifest)
    except _TableVisualIdentityUnavailable as exc:
        return _identity_unavailable_record(
            cache_key,
            request_info,
            exc,
            diagnostics,
        )
    if not identity_matches:
        return _mark_task_stale(
            cache_key,
            request_info,
            current_manifest=current_manifest,
            diagnostics=diagnostics,
        )
    record = _complete_task(
        cache_key,
        request_info,
        state=state,
        diagnostics=diagnostics,
        segment=segment,
    )
    latest_manifest = read_parse_manifest(
        current_document_data if isinstance(current_document_data, dict) else {},
        doc_id=doc_id,
    )
    try:
        identity_matches = _record_matches_parse_identity(record, latest_manifest)
    except _TableVisualIdentityUnavailable as exc:
        return _identity_unavailable_record(
            cache_key,
            record,
            exc,
            diagnostics,
        )
    if not identity_matches:
        return _mark_task_stale(
            cache_key,
            record,
            current_manifest=latest_manifest,
            diagnostics=diagnostics,
        )
    return record


def _task_matches_request(
    record: dict,
    *,
    doc_id: str,
    target: dict,
    provider: str,
    model: str,
    visual_model_identity: str = "",
    ai_cache_generation: str = "",
) -> bool:
    if not record:
        return False
    expected_ai_cache_generation = str(
        ai_cache_generation or _load_table_visual_ai_cache_generation(doc_id)
    )
    return (
        record.get("doc_id") == doc_id
        and record.get("document_source_hash") == target.get("document_source_hash")
        and record.get("parse_route") == target.get("parse_route")
        and record.get("parse_generation") == target.get("parse_generation")
        and str(record.get("ai_cache_generation") or LEGACY_AI_CACHE_GENERATION) == expected_ai_cache_generation
        and record.get("page") == (target.get("page") or 0)
        and record.get("bbox_hash") == _target_bbox_hash(target)
        and record.get("purpose") == _VISUAL_PURPOSE
        and record.get("table_instance_id") == target.get("table_instance_id")
        and record.get("table_source_hash") == target.get("table_source_hash")
        and record.get("provider") == provider
        and record.get("model") == model
        and record.get("visual_model_identity") == visual_model_identity
        and record.get("prompt_version") == _VISUAL_PROMPT_VERSION
        and record.get("crop_version") == _VISUAL_CROP_VERSION
    )


def _failure_cooldown_active(record: dict) -> bool:
    try:
        return float(record.get("retry_after") or 0) > time.time()
    except (TypeError, ValueError):
        return False


def _task_is_stale(record: dict) -> bool:
    """Allow a new process to reclaim a persisted task left in progress."""
    if str(record.get("state") or "") not in {"queued", "running"}:
        return False
    try:
        updated_at = float(record.get("updated_at") or record.get("started_at") or record.get("created_at") or 0)
    except (TypeError, ValueError):
        return True
    return updated_at <= 0 or (time.time() - updated_at) >= _resolve_stale_task_seconds()


def _indeterminate_cache_active(record: dict) -> bool:
    if str(record.get("state") or "") != "indeterminate":
        return False
    try:
        completed_at = float(record.get("completed_at") or record.get("updated_at") or 0)
    except (TypeError, ValueError):
        return False
    return completed_at > 0 and (time.time() - completed_at) < _resolve_indeterminate_ttl_seconds()


def _task_diagnostics(record: dict, *, cache_hit: bool = False) -> dict:
    saved = record.get("diagnostics") if isinstance(record.get("diagnostics"), dict) else {}
    state = str(record.get("state") or "indeterminate")
    result = {
        **_safe_diagnostics(saved),
        "task_id": record.get("task_id") or "",
        "state": state,
        "verdict": state if state in {"confirmed", "conflict", "indeterminate"} else saved.get("verdict", "indeterminate"),
        "table_instance_id": record.get("table_instance_id") or "",
        "cache_hit": bool(cache_hit),
    }
    if state in {"queued", "running"}:
        result["pending"] = True
    return result


def get_table_visual_verification_status(
    doc_id: str,
    task_id: str,
    *,
    current_parse_manifest: Optional[dict] = None,
) -> dict:
    """Return a public status record, fencing obsolete parse generations."""
    cache_key = _cache_key_from_task_id(task_id)
    if not cache_key:
        return {}
    record = _load_task_record(cache_key)
    if not record or record.get("doc_id") != doc_id:
        return {}

    identity_current = True
    try:
        current_ai_cache_generation = _load_table_visual_ai_cache_generation(doc_id)
    except _TableVisualIdentityUnavailable as exc:
        record = _identity_unavailable_record(cache_key, record, exc)
        identity_current = False
    else:
        ai_cache_current = (
            str(record.get("ai_cache_generation") or LEGACY_AI_CACHE_GENERATION)
            == current_ai_cache_generation
        )
        parse_current = (
            True
            if not isinstance(current_parse_manifest, dict)
            else _record_matches_parse_identity(
                record,
                current_parse_manifest,
                current_ai_cache_generation=current_ai_cache_generation,
            )
        )
        if not (ai_cache_current and parse_current):
            record = _mark_task_stale(
                cache_key,
                record,
                current_manifest=current_parse_manifest or {},
                current_ai_cache_generation=current_ai_cache_generation,
            )
            identity_current = False

    diagnostics = _task_diagnostics(record, cache_hit=True)
    state = str(record.get("state") or "indeterminate")
    error_code = str(diagnostics.get("error_code") or "")
    error = None
    if error_code:
        error = {
            "code": error_code,
            "message": str(diagnostics.get("error_message") or diagnostics.get("error") or ""),
            "retryable": bool(diagnostics.get("retryable")),
        }
    return {
        "task_id": record.get("task_id") or task_id,
        "doc_id": doc_id,
        "table_instance_id": record.get("table_instance_id") or "",
        "state": state,
        "verdict": diagnostics.get("verdict", "indeterminate"),
        "consumable": bool(identity_current and state == "confirmed"),
        "stale": state == "stale",
        "parse_generation": record.get("parse_generation") or "",
        "ai_cache_generation": record.get("ai_cache_generation") or LEGACY_AI_CACHE_GENERATION,
        "document_source_hash": record.get("document_source_hash") or "",
        "attempts": int(record.get("attempts") or 0),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "completed_at": record.get("completed_at"),
        "retry_after": record.get("retry_after"),
        "error": error,
        "diagnostics": diagnostics,
    }


def _schedule_visual_background_task(
    *,
    cache_key: str,
    query: str,
    target: dict,
    pdf_path: Path,
    segments: list[dict],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
    custom_params: dict,
    diagnostics: dict,
    request_info: dict,
    current_document_data: dict,
    visual_config: VisualModelConfig | None = None,
    visual_policy: VisualEnrichmentPolicy | None = None,
) -> None:
    if cache_key in _VISUAL_PENDING:
        return
    _VISUAL_PENDING.add(cache_key)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _VISUAL_PENDING.discard(cache_key)
        return
    loop.create_task(
        _run_visual_verification_background(
            cache_key=cache_key,
            query=query,
            target=target,
            pdf_path=pdf_path,
            segments=segments,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            custom_params={**dict(custom_params or {}), "_numeric_table_visual_background_task": True},
            diagnostics=diagnostics,
            request_info=request_info,
            current_document_data=current_document_data,
            visual_config=visual_config,
            visual_policy=visual_policy,
        )
    )


async def _run_visual_verification_background(**kwargs) -> None:
    cache_key = str(kwargs.get("cache_key") or "")
    try:
        _, diagnostics = await _run_visual_verification(**kwargs)
        if diagnostics.get("state") in {"failed", "conflict", "indeterminate"}:
            logger.debug("[TableVisual] background task finished: %s", diagnostics.get("state"))
    except Exception as exc:
        request_info = kwargs.get("request_info") if isinstance(kwargs.get("request_info"), dict) else {}
        error_message = f"{type(exc).__name__}: {exc}"
        diagnostics = {
            "enabled": True,
            "triggered": True,
            "state": "failed",
            "verdict": "indeterminate",
            "error": error_message,
            "error_code": str(getattr(exc, "code", "") or "visual_task_failed"),
            "error_message": error_message,
            "retryable": True,
        }
        _complete_task_for_current_parse(
            cache_key,
            request_info,
            current_document_data=kwargs.get("current_document_data") or {},
            state="failed",
            diagnostics=diagnostics,
        )
        logger.debug("[TableVisual] background verification failed: %s", exc)
    finally:
        _VISUAL_PENDING.discard(cache_key)


async def _run_visual_verification(
    *,
    cache_key: str,
    query: str,
    target: dict,
    pdf_path: Path,
    segments: list[dict],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
    custom_params: Optional[dict],
    diagnostics: dict,
    request_info: dict,
    current_document_data: dict,
    visual_config: VisualModelConfig | None = None,
    visual_policy: VisualEnrichmentPolicy | None = None,
) -> tuple[dict, dict]:
    resolved_visual_config = visual_config or resolve_visual_model_config(
        primary_provider=provider,
        primary_model=model,
        primary_api_key=api_key,
        primary_endpoint=endpoint,
    )
    diagnostics.update({
        "triggered": True,
        "state": "running",
        "task_id": _task_id(cache_key),
        "table_instance_id": target.get("table_instance_id") or "",
        "visual_model": resolved_visual_config.public_metadata(),
    })
    _create_or_update_task(cache_key, request_info, state="running", diagnostics=diagnostics)

    def _finish(
        state: str,
        *,
        segment: Optional[dict] = None,
    ) -> dict:
        return _complete_task_for_current_parse(
            cache_key,
            request_info,
            current_document_data=current_document_data,
            state=state,
            diagnostics=diagnostics,
            segment=segment,
        )

    current_manifest = read_parse_manifest(current_document_data, doc_id=str(request_info.get("doc_id") or ""))
    try:
        identity_matches = _record_matches_parse_identity(request_info, current_manifest)
    except _TableVisualIdentityUnavailable as exc:
        record = _identity_unavailable_record(
            cache_key,
            request_info,
            exc,
            diagnostics,
        )
        return {}, _task_diagnostics(record)
    if not identity_matches:
        record = _mark_task_stale(
            cache_key,
            request_info,
            current_manifest=current_manifest,
            diagnostics=diagnostics,
        )
        return {}, _task_diagnostics(record)

    if _unsafe_cross_page_target(target):
        diagnostics.update({"state": "indeterminate", "verdict": "indeterminate", "rejected_reason": "cross_page_table_ambiguous"})
        record = _finish("indeterminate")
        return {}, _task_diagnostics(record)
    try:
        crops = render_table_crops_base64(pdf_path, target)
    except Exception as exc:
        diagnostics.update({"state": "indeterminate", "verdict": "indeterminate", "skipped_reason": f"render_failed:{type(exc).__name__}"})
        record = _finish("indeterminate")
        logger.debug("[TableVisual] render failed: %s", exc)
        return {}, _task_diagnostics(record)
    if not crops:
        diagnostics.update({"state": "indeterminate", "verdict": "indeterminate", "skipped_reason": "no_rendered_crop"})
        record = _finish("indeterminate")
        return {}, _task_diagnostics(record)

    crop_meta = [{key: value for key, value in crop.items() if key != "image_b64"} for crop in crops]
    diagnostics.update({"page": target.get("page"), "crops": crop_meta, "crop_count": len(crops)})
    prompt_text = _build_visual_prompt(query, target, segments)
    user_content: list[dict] = [{"type": "text", "text": prompt_text}]
    for crop in crops:
        user_content.append({"type": "text", "text": f"Image role: {crop['role']} (page {crop['page']})."})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop['image_b64']}"}})
    messages = [
        {
            "role": "system",
            "content": (
                "You verify numeric answers from academic paper tables. Read only visible cells. "
                "Return exactly one JSON object matching the requested schema; do not infer missing values."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    timeout_s = _resolve_visual_timeout_seconds(custom_params)
    retries = _resolve_visual_retry_retries(custom_params)
    retry_delay = _resolve_visual_retry_delay(custom_params)
    visual_task_policy = VisualTaskPolicy(
        timeout_seconds=timeout_s,
        max_retries=retries,
        retry_delay_seconds=retry_delay,
        concurrency=_resolve_visual_concurrency(custom_params),
        document_budget=(
            visual_policy.document_budget
            if visual_policy is not None
            else _resolve_visual_document_budget(custom_params)
        ),
    )

    async def _invoke_visual_model() -> Any:
        return await _call_table_visual_model(
            messages=messages,
            config=resolved_visual_config,
            middlewares=None,
            max_tokens=700,
            temperature=0,
            purpose="numeric_table_visual_verification",
        )

    try:
        response = await execute_visual_task(
            task_id=_task_id(cache_key),
            document_id=str(request_info.get("doc_id") or ""),
            parse_generation=str(request_info.get("parse_generation") or ""),
            purpose=_VISUAL_PURPOSE,
            operation=_invoke_visual_model,
            policy=visual_task_policy,
            metadata={
                "provider": resolved_visual_config.provider,
                "model": resolved_visual_config.model,
                "source": resolved_visual_config.source,
                "page": target.get("page"),
                "bbox_hash": request_info.get("bbox_hash"),
                "route": request_info.get("parse_route"),
                "prompt_version": _VISUAL_PROMPT_VERSION,
            },
        )
    except VisualTaskTimeoutError:
        error_message = f"visual task timed out after {timeout_s:.0f}s"
        diagnostics.update({
            "state": "failed",
            "verdict": "indeterminate",
            "error": error_message,
            "error_code": "visual_timeout",
            "error_message": error_message,
            "retryable": True,
            "rejected_reason": "timeout",
        })
        record = _finish("failed")
        return {}, _task_diagnostics(record)
    except VisualTaskBudgetExceeded as exc:
        error_message = str(exc)
        diagnostics.update({
            "state": "indeterminate",
            "verdict": "indeterminate",
            "error": error_message,
            "error_code": "visual_budget_exhausted",
            "error_message": error_message,
            "retryable": True,
            "rejected_reason": "document_visual_budget_exhausted",
            "skipped_reason": "document_visual_budget_exhausted",
        })
        record = _finish("indeterminate")
        return {}, _task_diagnostics(record)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        diagnostics.update({
            "state": "failed",
            "verdict": "indeterminate",
            "error": error_message,
            "error_code": str(getattr(exc, "code", "") or "visual_task_failed"),
            "error_message": error_message,
            "retryable": True,
        })
        record = _finish("failed")
        return {}, _task_diagnostics(record)
    if isinstance(response, dict) and response.get("error"):
        error_message = str(response.get("error"))
        diagnostics.update({
            "state": "failed",
            "verdict": "indeterminate",
            "error": error_message,
            "error_code": "visual_upstream_error",
            "error_message": error_message,
            "retryable": True,
        })
        record = _finish("failed")
        return {}, _task_diagnostics(record)

    content = _extract_response_content(response)
    parsed = _parse_visual_response(content)
    diagnostics["used_provider"] = response.get("_used_provider") if isinstance(response, dict) else ""
    diagnostics["used_model"] = response.get("_used_model") if isinstance(response, dict) else ""
    if parsed is None:
        diagnostics.update({
            "state": "indeterminate",
            "verdict": "indeterminate",
            "error": "invalid_schema",
            "error_code": "invalid_visual_schema",
            "error_message": "visual response did not match the table verification schema",
            "retryable": True,
            "raw_preview": content[:300],
        })
        record = _finish("indeterminate")
        return {}, _task_diagnostics(record)

    verdict, verdict_details = _evaluate_visual_result(
        parsed,
        target,
        query,
        minimum_confidence=_resolve_visual_min_confidence(custom_params),
    )
    diagnostics.update({
        "state": verdict,
        "verdict": verdict,
        "confidence": parsed.confidence,
        "verification": verdict_details,
    })
    if verdict != "confirmed":
        diagnostics["rejected_reason"] = verdict_details.get("reason") or verdict
        record = _finish(verdict)
        return {}, _task_diagnostics(record)

    segment = _build_visual_segment(
        parsed,
        query=query,
        target=target,
        crop_meta=crop_meta,
        response=response if isinstance(response, dict) else {},
        verified_cells=verdict_details.get("verified_cells") or [],
    )
    record = _finish("confirmed", segment=segment)
    task_diagnostics = _task_diagnostics(record)
    if str(record.get("state") or "") != "confirmed":
        return {}, task_diagnostics
    return segment, task_diagnostics


def _resolve_visual_timeout_seconds(custom_params: Optional[dict]) -> float:
    value = None
    if isinstance(custom_params, dict):
        value = custom_params.get("numeric_table_visual_timeout_s") or custom_params.get("table_visual_timeout_s")
    if value in (None, ""):
        default_timeout = "75" if isinstance(custom_params, dict) and custom_params.get("_numeric_table_visual_background_task") else "35"
        value = os.getenv("CHATPDF_TABLE_VISUAL_TIMEOUT_S", default_timeout)
    try:
        return max(8.0, min(90.0, float(value)))
    except (TypeError, ValueError):
        return 35.0


def _resolve_visual_concurrency(custom_params: Optional[dict]) -> int:
    value = custom_params.get("numeric_table_visual_concurrency") if isinstance(custom_params, dict) else None
    if value in (None, ""):
        value = os.getenv("CHATPDF_TABLE_VISUAL_CONCURRENCY", str(_DEFAULT_CONCURRENCY))
    try:
        return max(1, min(4, int(value)))
    except (TypeError, ValueError):
        return _DEFAULT_CONCURRENCY


def _resolve_visual_retry_retries(custom_params: Optional[dict]) -> int:
    value = custom_params.get("numeric_table_visual_retries") if isinstance(custom_params, dict) else None
    if value in (None, ""):
        value = os.getenv("CHATPDF_TABLE_VISUAL_RETRIES", "1")
    try:
        return max(0, min(3, int(value)))
    except (TypeError, ValueError):
        return 1


def _resolve_visual_retry_delay(custom_params: Optional[dict]) -> float:
    value = custom_params.get("numeric_table_visual_retry_delay") if isinstance(custom_params, dict) else None
    if value in (None, ""):
        value = os.getenv("CHATPDF_TABLE_VISUAL_RETRY_DELAY", "0.4")
    try:
        return max(0.0, min(5.0, float(value)))
    except (TypeError, ValueError):
        return 0.4


def _resolve_visual_document_budget(custom_params: Optional[dict]) -> int:
    value = custom_params.get("visual_document_budget") if isinstance(custom_params, dict) else None
    if value in (None, ""):
        value = os.getenv("CHATPDF_VISUAL_DOCUMENT_BUDGET", "16")
    try:
        return max(1, min(1000, int(value)))
    except (TypeError, ValueError):
        return 16


def _resolve_visual_min_confidence(custom_params: Optional[dict]) -> float:
    value = None
    if isinstance(custom_params, dict):
        for key in ("numeric_table_visual_min_confidence", "table_visual_min_confidence"):
            if key in custom_params and custom_params.get(key) not in (None, ""):
                value = custom_params.get(key)
                break
    if value in (None, ""):
        value = os.getenv("CHATPDF_TABLE_VISUAL_MIN_CONFIDENCE", "0.75")
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.75


def _resolve_failure_cooldown_seconds(custom_params: Optional[dict]) -> float:
    value = custom_params.get("numeric_table_visual_failure_cooldown_s") if isinstance(custom_params, dict) else None
    if value in (None, ""):
        value = os.getenv("CHATPDF_TABLE_VISUAL_FAILURE_COOLDOWN_S", str(_DEFAULT_FAILURE_COOLDOWN_S))
    try:
        return max(30.0, min(3600.0, float(value)))
    except (TypeError, ValueError):
        return _DEFAULT_FAILURE_COOLDOWN_S


def _resolve_stale_task_seconds() -> float:
    try:
        value = float(os.getenv("CHATPDF_TABLE_VISUAL_STALE_TASK_S", str(_DEFAULT_STALE_TASK_S)))
        return max(30.0, min(86400.0, value))
    except (TypeError, ValueError):
        return _DEFAULT_STALE_TASK_S


def _resolve_indeterminate_ttl_seconds() -> float:
    try:
        value = float(os.getenv("CHATPDF_TABLE_VISUAL_INDETERMINATE_TTL_S", str(_DEFAULT_INDETERMINATE_TTL_S)))
        return max(60.0, min(604800.0, value))
    except (TypeError, ValueError):
        return _DEFAULT_INDETERMINATE_TTL_S


def render_table_crop_base64(pdf_path: Path, page_number: int, bbox: Optional[list[float]], *, dpi: int = 180) -> tuple[str, dict]:
    """Compatibility wrapper returning an overview crop for existing callers."""
    target = {"page": page_number, "bbox": bbox or []}
    crops = render_table_crops_base64(pdf_path, target, dpi=dpi, include_detail=False)
    if not crops:
        raise ValueError("No crop rendered")
    crop = crops[0]
    return crop["image_b64"], {key: value for key, value in crop.items() if key != "image_b64"}


def render_table_crops_base64(
    pdf_path: Path,
    target: dict,
    *,
    dpi: int = 180,
    include_detail: bool = True,
) -> list[dict]:
    """Create overview, header, and target-row crops when their geometry exists."""
    import fitz

    page_number = int(target.get("page") or 0)
    if page_number <= 0:
        raise ValueError("Missing page")
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[max(0, page_number - 1)]
        page_rect = page.rect
        table_bbox = _bbox_to_page_rect(target.get("bbox"), page_rect)
        clips: list[tuple[str, Any]] = []
        overview = _padded_rect(table_bbox or page_rect, page_rect, left_ratio=0.14, right_ratio=0.10, vertical_ratio=0.10)
        clips.append(("overview", overview))
        if include_detail and table_bbox:
            header_height = max(30.0, table_bbox.height * 0.30)
            header = fitz.Rect(table_bbox.x0, table_bbox.y0, table_bbox.x1, min(table_bbox.y1, table_bbox.y0 + header_height))
            clips.append(("header", _padded_rect(header, page_rect, left_ratio=0.05, right_ratio=0.05, vertical_ratio=0.18)))
            selected_row = target.get("selected_row") if isinstance(target.get("selected_row"), dict) else {}
            row_bbox = _bbox_to_page_rect(_record_bbox(selected_row), page_rect) if selected_row else None
            if row_bbox:
                clips.append(("target_row", _padded_rect(row_bbox, page_rect, left_ratio=0.12, right_ratio=0.12, vertical_ratio=0.30)))

        output: list[dict] = []
        seen: set[tuple[int, int, int, int]] = set()
        for role, clip in clips:
            key = tuple(round(float(value), 1) for value in (clip.x0, clip.y0, clip.x1, clip.y1))
            if key in seen:
                continue
            seen.add(key)
            pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False, annots=False)
            output.append({
                "role": role,
                "page": page_number,
                "bbox": [round(float(value), 2) for value in (clip.x0, clip.y0, clip.x1, clip.y1)],
                "dpi": dpi,
                "width": pix.width,
                "height": pix.height,
                "rotation": int(page.rotation or 0),
                "image_b64": base64.b64encode(pix.tobytes("png")).decode("ascii"),
            })
        return output
    finally:
        doc.close()


def _bbox_to_page_rect(bbox: Any, page_rect: Any):
    import fitz

    normalized = _normalize_bbox(bbox)
    if not normalized:
        return None
    x0, y0, x1, y1 = normalized
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0, x1 = x0 * page_rect.width, x1 * page_rect.width
        y0, y1 = y0 * page_rect.height, y1 * page_rect.height
    rect = fitz.Rect(x0, y0, x1, y1)
    rect = rect & page_rect
    return rect if rect.width > 1 and rect.height > 1 else None


def _padded_rect(rect: Any, page_rect: Any, *, left_ratio: float, right_ratio: float, vertical_ratio: float):
    import fitz

    return fitz.Rect(
        max(page_rect.x0, rect.x0 - max(18.0, rect.width * left_ratio)),
        max(page_rect.y0, rect.y0 - max(12.0, rect.height * vertical_ratio)),
        min(page_rect.x1, rect.x1 + max(18.0, rect.width * right_ratio)),
        min(page_rect.y1, rect.y1 + max(12.0, rect.height * vertical_ratio)),
    )


def _build_visual_prompt(query: str, target: dict, segments: list[dict]) -> str:
    requested_columns = ", ".join(target.get("requested_columns") or []) or "all columns needed for the answer"
    requested_rows = ", ".join(target.get("requested_rows") or []) or "row implied by the question"
    return f"""Question:
{query}

Target table identity:
- table_instance_id: {target.get('table_instance_id') or ''}
- table_ref: {target.get('table_ref') or ''}
- table_id: {target.get('table_id') or ''}
- caption: {target.get('caption') or ''}
- requested rows: {requested_rows}
- requested columns: {requested_columns}

Read the supplied table crops. Return JSON only, with this exact shape:
{{
  "table_id": "visible table label or empty",
  "matched_row": "visible row label or empty",
  "cells": {{"visible column label": "visible value"}},
  "confidence": 0.0,
  "notes": "short uncertainty note",
  "supports_text_result": true,
  "corrected_answer_hint": "leave empty unless directly visible"
}}
Do not use values from any text context. Read the supplied image crops directly.
Do not invent a cell, header, row label, or value that is not visible."""


def _parse_visual_response(text: str) -> VisualTableResponse | None:
    parsed = _parse_json_object(text)
    if not parsed:
        return None
    try:
        response = VisualTableResponse.model_validate(parsed)
    except ValidationError:
        return None
    return response if response.cells else None


def _evaluate_visual_result(
    parsed: VisualTableResponse,
    target: dict,
    query: str,
    *,
    minimum_confidence: float = 0.75,
) -> tuple[str, dict]:
    expected = _expected_table_cells(target, query)
    details: dict[str, Any] = {
        "matched_columns": [],
        "matched_values": [],
        "verified_cells": [],
        "expected_columns": sorted(expected["values"].keys()),
    }
    if parsed.supports_text_result is False:
        details["reason"] = "text_result_not_supported"
        return "indeterminate", details
    if parsed.confidence is None or parsed.confidence < minimum_confidence:
        details["reason"] = "confidence_below_threshold"
        details["minimum_confidence"] = minimum_confidence
        return "indeterminate", details
    expected_refs = _extract_table_refs(f"{target.get('table_id') or ''} {target.get('caption') or ''}")
    parsed_refs = _extract_table_refs(parsed.table_id)
    if parsed_refs and expected_refs and not (parsed_refs & expected_refs):
        details["reason"] = "table_identity_mismatch"
        return "conflict", details

    requested_rows = set(target.get("requested_rows") or _extract_query_row_keys(query))
    row_candidates = expected["row_labels"]
    if requested_rows and parsed.matched_row:
        reported_row = {"row_id": parsed.matched_row}
        expected_match = any(_row_matches_key(reported_row, key) for key in requested_rows)
        candidate_match = any(_labels_match(parsed.matched_row, candidate) for candidate in row_candidates)
        if not expected_match and not candidate_match:
            details["reason"] = "row_identity_mismatch"
            return "conflict", details
    elif requested_rows and not parsed.matched_row:
        details["reason"] = "missing_matched_row"
        return "indeterminate", details

    conflict_values: list[dict] = []
    for visual_column, visual_value in parsed.cells.items():
        matched_column = _match_expected_column(visual_column, expected["values"].keys())
        if not matched_column:
            continue
        details["matched_columns"].append(matched_column)
        expected_values = expected["values"].get(matched_column) or set()
        if expected_values and any(_cell_values_match(visual_value, value) for value in expected_values):
            details["matched_values"].append({"column": matched_column, "value": visual_value})
            details["verified_cells"].append({
                "column": visual_column,
                "expected_column": matched_column,
                "value": visual_value,
            })
        elif expected_values:
            conflict_values.append({"column": matched_column, "visual": visual_value, "text": sorted(expected_values)[:3]})
    if conflict_values:
        details["conflicting_values"] = conflict_values
        details["reason"] = "cell_value_mismatch"
        return "conflict", details

    requested_columns = target.get("requested_columns") or []
    required_columns = {
        _match_expected_column(column, expected["values"].keys())
        for column in requested_columns
    }
    if requested_columns and ("" in required_columns or not required_columns):
        details["reason"] = "requested_columns_missing_text_evidence"
        return "indeterminate", details
    # A question such as "ID Ours 的 Accuracy" names the ID column only to
    # locate the row. Once that row has been strictly matched above, it is not
    # an additional numeric answer cell the VLM must repeat.
    if requested_rows:
        selected = target.get("selected_row") if isinstance(target.get("selected_row"), dict) else {}
        selected_cells = selected.get("cell_evidence_units") if isinstance(selected.get("cell_evidence_units"), list) else []
        for index, cell in enumerate(selected_cells):
            if not isinstance(cell, dict):
                continue
            cell_value = str(cell.get("content") or cell.get("text") or cell.get("cell_text") or "").strip()
            if not any(_row_matches_key({"row_id": cell_value}, key) for key in requested_rows):
                continue
            column = str(cell.get("column_name") or cell.get("header") or cell.get("column") or "").strip()
            if not column:
                headers = _split_table_cells(target.get("table_header") or "")
                column = headers[index] if index < len(headers) else ""
            matched_identity_column = _match_expected_column(column, expected["values"].keys())
            if matched_identity_column:
                required_columns.discard(matched_identity_column)
    verified_columns = {cell["expected_column"] for cell in details["verified_cells"]}
    missing_columns = sorted(required_columns - verified_columns)
    if missing_columns:
        details["missing_columns"] = missing_columns
        details["reason"] = "requested_columns_not_fully_verified"
        return "indeterminate", details
    if not details["matched_values"]:
        details["reason"] = "no_cell_evidence_alignment"
        return "indeterminate", details
    return "confirmed", details


def _expected_table_cells(target: dict, query: str) -> dict:
    headers = _split_table_cells(target.get("table_header") or "")
    values: dict[str, set[str]] = {}
    row_labels: list[str] = []
    rows = [unit for unit in target.get("evidence_units") or [] if isinstance(unit, dict)]
    selected = target.get("selected_row") if isinstance(target.get("selected_row"), dict) else None
    if selected:
        rows = [selected]
    elif target.get("cell_evidence_units"):
        rows.append({"cell_evidence_units": target.get("cell_evidence_units"), "content": target.get("text") or ""})
    for row in rows:
        if bool(row.get("is_header_row")):
            continue
        cells = row.get("cell_evidence_units") if isinstance(row.get("cell_evidence_units"), list) else []
        if not cells:
            cells = [{"content": value} for value in _split_table_cells(row.get("content") or row.get("row_text") or "")]
        row_label = str(row.get("row_id") or row.get("row_text") or row.get("content") or "").strip()
        if cells and isinstance(cells[0], dict):
            row_label = str(cells[0].get("content") or cells[0].get("text") or row_label).strip()
        if row_label:
            row_labels.append(row_label)
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            column = str(cell.get("column_name") or cell.get("header") or cell.get("column") or "").strip()
            if not column and index < len(headers):
                column = headers[index]
            if not column:
                column = f"column {index + 1}"
            value = str(cell.get("content") or cell.get("text") or cell.get("cell_text") or "").strip()
            if value:
                values.setdefault(_norm(column), set()).add(value)
    return {"values": values, "row_labels": row_labels}


def _build_visual_segment(
    parsed: VisualTableResponse,
    *,
    query: str,
    target: dict,
    crop_meta: list[dict],
    response: dict,
    verified_cells: list[dict],
) -> dict:
    cells = {
        str(cell.get("column") or ""): str(cell.get("value") or "")
        for cell in verified_cells
        if isinstance(cell, dict) and str(cell.get("column") or "").strip() and str(cell.get("value") or "").strip()
    }
    cell_lines = [f"{key} = {value}" for key, value in cells.items() if str(key).strip()]
    table_id = parsed.table_id or str(target.get("table_id") or target.get("table_ref") or "").strip()
    lines = ["[Numeric Table Visual Verification: confirmed]"]
    if table_id:
        lines.append(f"Table ID: {table_id}")
    if target.get("caption"):
        lines.append(f"Table Caption: {target.get('caption')}")
    if parsed.matched_row:
        lines.append(f"Matched Row: {parsed.matched_row}")
    if cell_lines:
        lines.append("Verified Cells:")
        lines.extend(f"- {line}" for line in cell_lines)
    if parsed.notes:
        lines.append(f"Notes: {parsed.notes}")
    page = int(target.get("page") or 0)
    return {
        "text": "\n".join(lines),
        "segment_role": "numeric_table_visual_verification",
        "context_id": f"{target.get('table_instance_id') or target.get('context_id') or 'table'}:visual_verification",
        "evidence_id": f"{target.get('table_instance_id') or target.get('evidence_id') or 'table'}:visual_verification",
        "table_id": table_id,
        "table_bundle_id": target.get("table_bundle_id") or "",
        "table_instance_id": target.get("table_instance_id") or "",
        "table_source_hash": target.get("table_source_hash") or "",
        "table_caption": target.get("caption") or "",
        "page_range": [page, page] if page else [],
        "bbox": target.get("bbox") or [],
        "visual_crops": crop_meta,
        "visual_cells": cells,
        "visual_matched_row": parsed.matched_row,
        "visual_confidence": parsed.confidence,
        "visual_verdict": "confirmed",
        "visual_notes": parsed.notes,
        "visual_query": query,
        "used_provider": response.get("_used_provider"),
        "used_model": response.get("_used_model"),
        "synthetic_description": True,
    }


def _select_target_row(target: dict, query: str) -> dict:
    requested = _extract_query_row_keys(query)
    candidates = [unit for unit in target.get("evidence_units") or [] if isinstance(unit, dict) and not unit.get("is_header_row")]
    if not candidates:
        return {}
    scored: list[tuple[int, dict]] = []
    for row in candidates:
        score = sum(20 for key in requested if _row_matches_key(row, key))
        if row.get("cell_evidence_units"):
            score += 2
        scored.append((score, row))
    best_score, best = max(scored, key=lambda item: item[0])
    return dict(best) if best_score > 0 else {}


def _unsafe_cross_page_target(target: dict) -> bool:
    pages = [page for page in target.get("pages") or [] if isinstance(page, int) and page > 0]
    if not pages:
        page = target.get("page")
        page_end = target.get("page_end")
        pages = [page, page_end] if isinstance(page, int) and isinstance(page_end, int) else []
    # The current structured bundle format does not prove that a later page's
    # header and table bbox are a continuation of the first page.  Treating a
    # selected row as sufficient would silently create a cross-page merge, so
    # leave these cases to the original structured evidence instead.
    return len(set(pages)) > 1


def _extract_requested_columns(query: str, target: Optional[dict] = None) -> list[str]:
    headers = _split_table_cells((target or {}).get("table_header") or "")
    if not headers:
        return []
    query_norm = _norm(query)
    normalized_headers = [(header, _norm(header)) for header in headers if _norm(header)]
    requested: list[str] = []
    aliases = {"accuracy": "acc", "precision": "prec", "recall": "rec", "performance": "perf"}
    query_with_aliases = query_norm + "".join(aliases.get(term, "") for term in _query_terms(query))
    for header, normalized in normalized_headers:
        if len(normalized) >= 2 and normalized in query_with_aliases:
            requested.append(header)
    return list(dict.fromkeys(requested))


def _split_table_cells(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if "|" in text:
        cells = [re.sub(r"\s+", " ", part.replace("\\|", "|")).strip() for part in text.strip("|").split("|")]
        return [cell for cell in cells if cell and not re.fullmatch(r":?-{2,}:?", cell)]
    return [part.strip() for part in re.split(r"\t|;", text) if part.strip()]


def _match_expected_column(value: str, expected_columns: Any) -> str:
    value_norm = _norm(value)
    if not value_norm:
        return ""
    for column in expected_columns:
        if _labels_match(value, column):
            return str(column)
    return ""


def _labels_match(left: str, right: str) -> bool:
    left_norm, right_norm = _norm(left), _norm(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if min(len(left_norm), len(right_norm)) < 3:
        return False
    return left_norm in right_norm or right_norm in left_norm


def _cell_values_match(left: str, right: str) -> bool:
    left_norm = _norm_numeric_value(left)
    right_norm = _norm_numeric_value(right)
    if left_norm == right_norm:
        return True
    left_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", str(left or "").replace(",", ""))
    right_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", str(right or "").replace(",", ""))
    return bool(left_numbers and right_numbers and left_numbers == right_numbers)


def _norm_numeric_value(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").replace(",", "").replace("％", "%")).casefold()


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


def _record_bbox(record: Any) -> list[float]:
    if not isinstance(record, dict):
        return []
    # MinerU row/cell units without explicit geometry must never turn a whole
    # table bbox into a misleading "target row" crop.
    if (
        str(record.get("evidence_unit_type") or "").lower() in {"table_row", "table_cell"}
        and record.get("visual_crop_eligible") is False
    ):
        return []
    for key in ("visual_bbox",):
        bbox = _normalize_bbox(record.get(key))
        if bbox:
            return bbox
    bboxes: list[list[float]] = []
    for key in ("visual_bounding_boxes",):
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                bbox = _normalize_bbox(item)
                if bbox:
                    bboxes.append(bbox)
    if bboxes:
        return _merge_bboxes(bboxes)
    # Never reinterpret parser-declared normalized or bottom-left coordinates as
    # PyMuPDF top-left points. Legacy structured artifacts fall back to a page
    # overview until they are rebuilt with visual geometry.
    coordinate_space = str(record.get("bbox_coordinate_space") or "").strip()
    source = str(record.get("source") or "").strip().lower()
    if coordinate_space and coordinate_space != "pdf_top_left_points":
        return []
    if not coordinate_space and source in {"mineru", "odl"}:
        return []
    for key in ("bbox", "table_bbox", "bounding_box", "cell_bbox", "row_bbox"):
        bbox = _normalize_bbox(record.get(key))
        if bbox:
            return bbox
    for unit_key in ("cell_evidence_units", "evidence_units"):
        units = record.get(unit_key)
        if isinstance(units, list):
            for unit in units:
                if not isinstance(unit, dict):
                    continue
                bbox = _record_bbox(unit)
                if bbox:
                    bboxes.append(bbox)
    return _merge_bboxes(bboxes)


def _record_bboxes(record: dict) -> list[list[float]]:
    bboxes: list[list[float]] = []
    for key in ("visual_bounding_boxes",):
        for item in record.get(key) or []:
            bbox = _normalize_bbox(item)
            if bbox:
                bboxes.append(bbox)
    if not bboxes:
        coordinate_space = str(record.get("bbox_coordinate_space") or "").strip()
        source = str(record.get("source") or "").strip().lower()
        if not coordinate_space and source not in {"mineru", "odl"} or coordinate_space == "pdf_top_left_points":
            for key in ("bounding_boxes", "table_bboxes"):
                for item in record.get(key) or []:
                    bbox = _normalize_bbox(item)
                    if bbox:
                        bboxes.append(bbox)
    bbox = _record_bbox(record)
    if bbox and bbox not in bboxes:
        bboxes.insert(0, bbox)
    return bboxes


def _record_pages(record: dict, fallback: int) -> list[int]:
    values: list[int] = []
    for key in ("pages", "page_range", "table_pages"):
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                try:
                    page = int(item)
                except (TypeError, ValueError):
                    continue
                if page > 0 and page not in values:
                    values.append(page)
    if not values and fallback > 0:
        values.append(fallback)
    page_end = _last_page(record, fallback)
    if page_end > 0 and page_end not in values:
        values.append(page_end)
    return values


def _first_page(record: dict) -> int:
    for key in ("page", "page_start"):
        try:
            page = int(record.get(key) or 0)
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            return page
    for key in ("page_range", "pages", "table_pages"):
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


def _last_page(record: dict, fallback: int = 0) -> int:
    for key in ("page_end",):
        try:
            page = int(record.get(key) or 0)
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            return page
    pages = _record_pages_no_end(record)
    return max(pages) if pages else fallback


def _record_pages_no_end(record: dict) -> list[int]:
    values: list[int] = []
    for key in ("page_range", "pages", "table_pages"):
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                try:
                    page = int(item)
                except (TypeError, ValueError):
                    continue
                if page > 0:
                    values.append(page)
    return values


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


def _extract_table_refs(text: str) -> set[str]:
    refs: set[str] = set()
    for match in _TABLE_REF_RE.finditer(str(text or "")):
        token = (match.group(1) or match.group(2) or "").strip().lower()
        if token:
            refs.add(f"table {token}")
            refs.add(f"table-{token}")
            refs.add(f"表 {token}")
    return refs


def _table_ref_in_text(reference: str, text: str) -> bool:
    ref = _norm(reference)
    return bool(ref and ref in _norm(text))


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
            if value and len(value) <= 16:
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


def _query_terms(text: str) -> set[str]:
    return {_norm(match.group(0)) for match in _WORD_RE.finditer(str(text or "")) if _norm(match.group(0))}


def _row_text(row: dict) -> str:
    values = [str(row.get(key) or "").strip() for key in ("row_id", "row_text", "content")]
    cells = row.get("cell_evidence_units") if isinstance(row.get("cell_evidence_units"), list) else []
    values.extend(str(cell.get("content") or cell.get("text") or "").strip() for cell in cells if isinstance(cell, dict))
    return " ".join(value for value in values if value)


def _candidate_matches_row_key(candidate: dict, key: str) -> bool:
    for row in candidate.get("evidence_units") or []:
        if isinstance(row, dict) and _row_matches_key(row, key):
            return True
    return False


def _row_matches_key(row: dict, key: str) -> bool:
    """Match a query row key against structured row/cell boundaries, not prose."""
    key_norm = _norm(key)
    if not key_norm:
        return False
    cells = row.get("cell_evidence_units") if isinstance(row.get("cell_evidence_units"), list) else []
    labels = [str(row.get(field) or "").strip() for field in ("row_id", "row_text")]
    labels.extend(
        str(cell.get("content") or cell.get("text") or cell.get("cell_text") or "").strip()
        for cell in cells
        if isinstance(cell, dict)
    )
    # ID B / Row B is represented as both a compact key (idb) and B.  Only
    # accept B when it is an exact structured cell/row label; it must not match
    # the letter inside an unrelated caption such as "Appendix".
    compact_suffix = key_norm
    if key_norm.startswith("id"):
        compact_suffix = key_norm[2:]
    elif key_norm.startswith("row"):
        compact_suffix = key_norm[3:]
    for label in labels:
        label_norm = _norm(label)
        if label_norm == key_norm or (compact_suffix and label_norm == compact_suffix):
            return True
    return False


def _unit_fingerprint(unit: dict) -> str:
    raw = json.dumps(
        {
            "id": unit.get("evidence_unit_id"),
            "row": unit.get("row_id") or unit.get("row_text") or unit.get("content"),
            "bbox": _record_bbox(unit),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_diagnostics(diagnostics: dict) -> dict:
    return {key: value for key, value in (diagnostics or {}).items() if key not in {"raw_preview"}}


def _normal_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "")).strip().casefold()


def _short_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:24]


def _norm(value: str = "") -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fffωα]+", "", str(value or "").casefold())
