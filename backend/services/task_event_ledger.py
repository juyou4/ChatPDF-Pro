"""Privacy-safe append-only phase events for durable background tasks.

The existing document job store keeps only the latest state.  This module
stores a small, replayable event stream keyed by task id while deliberately
excluding credentials, URLs, file paths, document text, and model payloads.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from services.document_job_store import load_document_job, persist_document_job


TASK_EVENT_LEDGER_SCHEMA_VERSION = 1
TASK_EVENT_JOB_TYPE = "task_event_ledger"

_LEDGER_LOCK = threading.RLock()
_EVENT_STAGES = {
    "queued",
    "upload",
    "submit_mineru",
    "poll",
    "download",
    "normalize",
    "publish_block_index",
    "build_rag",
    "rag_index",
    "awaiting_rag_index",
    "publish_visual_assets",
    "downstream_ai",
    "ready",
    "failed",
    "cancelled",
    "restart_recovery",
    "task_failed",
    "stalled",
    "embedding",
    "unclassified",
}
_STATUS_ALIASES = {
    "ready": "succeeded",
    "completed": "succeeded",
    "success": "succeeded",
    "partial_ready": "partial",
    "pending": "queued",
    "processing": "running",
}
_STAGE_ALIASES = {
    "waiting_for_slot": "queued",
    "waiting_for_document_lock": "queued",
    "requesting_upload": "upload",
    "uploading": "upload",
    "submitting": "submit_mineru",
    "resuming": "poll",
    "resuming_result_download": "download",
    "polling": "poll",
    "downloading": "download",
    "retrying_download": "download",
    "building_index": "normalize",
    "normalizing": "normalize",
    "publishing": "publish_block_index",
    "publishing_block_index": "publish_block_index",
    "building_rag_index": "build_rag",
    "rag_index_failed": "rag_index",
    "awaiting_rag_index": "rag_index",
    "embedding": "embedding",
    "embedding_preflight": "embedding",
    "stalled": "stalled",
    "upgrading_local_rag_index": "build_rag",
    "publishing_visual_assets": "publish_visual_assets",
    "generating": "downstream_ai",
    "verifying": "downstream_ai",
    "parse_identity": "downstream_ai",
    "completed": "ready",
    "partial_ready": "ready",
    "restart_recovery": "restart_recovery",
    "failed": "failed",
    "cancelled": "cancelled",
    "queued": "queued",
    "running": "downstream_ai",
    "ready": "ready",
}
_ERROR_CODE_PATTERNS = (
    ("mineru 服务余额", "mineru_quota_exhausted"),
    ("mineru 请求过于频繁", "mineru_rate_limited"),
    ("mineru 拒绝了当前 pdf", "mineru_file_rejected"),
    ("mineru 服务暂时不可用", "mineru_service_unavailable"),
    ("insufficient balance", "embedding_quota_exhausted"),
    ("insufficient_balance", "embedding_quota_exhausted"),
    ("余额不足", "embedding_quota_exhausted"),
    ("额度不足", "embedding_quota_exhausted"),
    ("quota exceeded", "embedding_quota_exhausted"),
    ("payment required", "embedding_quota_exhausted"),
    ("embedding api 返回 http 402", "embedding_quota_exhausted"),
    ("embedding api returned http 402", "embedding_quota_exhausted"),
    ("embedding api 返回 http 429", "embedding_rate_limited"),
    ("embedding api 返回 http 401", "embedding_auth_failed"),
    ("embedding api 返回 http 403", "embedding_auth_failed"),
    ("embedding api 返回 http 404", "embedding_model_unavailable"),
    ("embedding api returned http 429", "embedding_rate_limited"),
    ("embedding api returned http 401", "embedding_auth_failed"),
    ("embedding api returned http 403", "embedding_auth_failed"),
    ("embedding api returned http 404", "embedding_model_unavailable"),
    ("问答索引任务长时间没有进展", "rag_index_stalled"),
    ("问答索引长时间没有进展", "rag_index_stalled"),
    ("credential", "missing_credentials"),
    ("token", "missing_credentials"),
    ("api key", "missing_credentials"),
    ("timeout", "timeout"),
    ("timed out", "timeout"),
    ("rate limit", "rate_limited"),
    ("queue", "queue_full"),
    ("quality", "quality_gate_failed"),
    ("identity", "identity_mismatch"),
    ("parse", "parse_failed"),
    ("download", "download_failed"),
    ("network", "network_error"),
    ("服务重启", "worker_interrupted"),
    ("restart", "worker_interrupted"),
    ("cancel", "cancelled"),
)
_SAFE_REASON_CODES = {
    "quality_gate_failed",
    "partial_parse",
    "parse_failed",
    "worker_interrupted",
    "claim_support_shortfall",
    "partial_quality",
    "degraded_result",
    "generation_failed",
    "missing_credentials",
    "timeout",
    "rate_limited",
    "queue_full",
    "identity_mismatch",
    "download_failed",
    "network_error",
    "cancelled",
    "restart_recovery",
    "redacted",
    "embedding_quota_exhausted",
    "embedding_rate_limited",
    "embedding_auth_failed",
    "embedding_model_unavailable",
    "embedding_network_error",
    "embedding_failed",
    "rag_index_stalled",
    "rag_index_network_error",
    "rag_index_quality_failed",
    "rag_index_storage_failed",
    "rag_index_failed",
    "mineru_auth_failed",
    "mineru_quota_exhausted",
    "mineru_rate_limited",
    "mineru_file_rejected",
    "mineru_service_unavailable",
    "mineru_endpoint_invalid",
    "mineru_response_invalid",
    "mineru_result_expired",
    "mineru_network_error",
    "mineru_download_failed",
    "mineru_remote_failed",
    "mineru_quality_failed",
    "mineru_stalled",
    "mineru_failed",
    "task_start_failed",
    "downstream_task_stalled",
}


def normalize_event_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    normalized = _STATUS_ALIASES.get(normalized, normalized)
    return normalized if normalized in {
        "queued", "running", "verifying", "publishing", "succeeded",
        "partial", "degraded", "failed", "cancelled",
    } else "running"


def normalize_event_stage(value: Any, *, status: Any = "") -> str:
    normalized = str(value or "").strip().lower()
    normalized = _STAGE_ALIASES.get(normalized, normalized)
    if normalized in _EVENT_STAGES:
        return normalized
    status_value = normalize_event_status(status)
    if status_value == "failed":
        return "failed"
    if status_value == "cancelled":
        return "cancelled"
    if status_value == "succeeded":
        return "ready"
    return "downstream_ai"


def classify_error_code(error: Any) -> str:
    text = str(error or "").strip().lower()
    if not text:
        return ""
    for needle, code in _ERROR_CODE_PATTERNS:
        if needle in text:
            return code
    return "task_failed"


def normalize_diagnostic_reason(value: Any) -> str:
    """Return a short code, never an arbitrary error/response string."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if any(token in text for token in ("http://", "https://", "api_key", "token", "secret", "password", "\\", "/")):
        return "redacted"
    normalized = re.sub(r"[^a-z0-9_\-]+", "_", text).strip("_")
    if normalized in _SAFE_REASON_CODES:
        return normalized
    # Unknown free-form reasons can contain document titles or model output;
    # keep the event useful without persisting that content.
    return "unclassified"


def sanitize_task_shortfall(value: Any) -> dict[str, Any]:
    """Keep only structured, non-content shortfall information."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("kind", "code", "stage"):
        text = normalize_diagnostic_reason(value.get(key))
        if text:
            result[key] = text
    if "retryable" in value:
        result["retryable"] = bool(value.get("retryable"))
    for key in ("count", "expected", "actual", "unresolved_count"):
        try:
            if value.get(key) is not None:
                result[key] = max(0, int(value.get(key)))
        except (TypeError, ValueError):
            continue
    for key in ("failed_pages", "missing_pages"):
        values = value.get(key)
        if isinstance(values, (list, tuple)):
            numbers: list[int] = []
            for item in list(values)[:100]:
                try:
                    number = int(item)
                except (TypeError, ValueError):
                    continue
                if number > 0:
                    numbers.append(number)
            if numbers:
                result[key] = sorted(set(numbers))
    reasons = value.get("reasons")
    if isinstance(reasons, (list, tuple)):
        result["reasons"] = [
            normalize_diagnostic_reason(item)
            for item in list(reasons)[:12]
            if normalize_diagnostic_reason(item)
        ]
    return result


def _clean_identity(identity: Mapping[str, Any] | None) -> dict[str, str]:
    identity = identity if isinstance(identity, Mapping) else {}
    route = str(identity.get("route") or identity.get("parser_route") or "").strip().lower()
    generation = str(identity.get("generation") or identity.get("parse_generation") or "").strip()
    source_hash = str(identity.get("source_hash") or identity.get("document_source_hash") or "").strip()
    cleaned = {}
    if route:
        cleaned["route"] = route[:40]
    if generation:
        cleaned["generation"] = generation[:160]
    if source_hash:
        cleaned["source_hash"] = source_hash[:160]
    return cleaned


def _event_signature(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("sequence"),
        event.get("stage"),
        event.get("status"),
        event.get("attempt", 0),
        event.get("error_code", ""),
        event.get("degraded_reason", ""),
    )


def _event_id(task_id: str, event: Mapping[str, Any]) -> str:
    payload = "|".join(str(item) for item in _event_signature(event))
    return hashlib.sha1(f"{task_id}|{payload}".encode("utf-8")).hexdigest()[:20]


def append_task_event(
    data_dir: Path | str,
    *,
    task_id: str,
    stage: str,
    status: str,
    identity: Mapping[str, Any] | None = None,
    sequence: int | None = None,
    duration_ms: int | None = None,
    attempt: int = 0,
    error_code: str = "",
    degraded_reason: str = "",
    occurred_at: float | None = None,
    shortfall: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one event, treating retries of the same sequence as idempotent."""
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        raise ValueError("task_id is required")
    normalized_status = normalize_event_status(status)
    normalized_stage = normalize_event_stage(stage, status=normalized_status)
    now = float(time.time() if occurred_at is None else occurred_at)
    with _LEDGER_LOCK:
        record = load_document_job(data_dir, TASK_EVENT_JOB_TYPE, normalized_task_id)
        events = [item for item in record.get("events") or [] if isinstance(item, dict)]
        last = events[-1] if events else None
        next_sequence = (int(last.get("sequence") or 0) + 1) if last else 1
        normalized_sequence = next_sequence if sequence is None else int(sequence)
        if normalized_sequence <= 0:
            raise ValueError("event sequence must be positive")
        event = {
            "task_id": normalized_task_id,
            "sequence": normalized_sequence,
            "stage": normalized_stage,
            "status": normalized_status,
            "duration_ms": max(0, int(duration_ms)) if duration_ms is not None else None,
            "attempt": max(0, int(attempt or 0)),
            "error_code": normalize_diagnostic_reason(error_code) or classify_error_code(error_code),
            "degraded_reason": normalize_diagnostic_reason(degraded_reason),
            "route": _clean_identity(identity).get("route", ""),
            "generation": _clean_identity(identity).get("generation", ""),
            "source_hash": _clean_identity(identity).get("source_hash", ""),
            "timestamp": now,
        }
        if event["duration_ms"] is None and last:
            try:
                event["duration_ms"] = max(0, int((now - float(last.get("timestamp") or now)) * 1000))
            except (TypeError, ValueError):
                event["duration_ms"] = 0
        if event["duration_ms"] is None:
            event["duration_ms"] = 0
        if last and normalized_sequence != int(last.get("sequence") or 0) + 1:
            same = next((item for item in events if int(item.get("sequence") or 0) == normalized_sequence), None)
            if same and _event_signature(same) == _event_signature(event):
                return dict(same)
            raise ValueError("event sequence must be append-only")
        event["event_id"] = _event_id(normalized_task_id, event)
        duplicate = next((item for item in events if item.get("event_id") == event["event_id"]), None)
        if duplicate:
            return dict(duplicate)
        events.append(event)
        saved = {
            "schema_version": TASK_EVENT_LEDGER_SCHEMA_VERSION,
            "task_id": normalized_task_id,
            "events": events[-200:],
            "updated_at": now,
        }
        cleaned_identity = _clean_identity(identity)
        if cleaned_identity:
            saved["identity"] = cleaned_identity
        clean_shortfall = sanitize_task_shortfall(shortfall)
        if clean_shortfall:
            saved["shortfall"] = clean_shortfall
        persist_document_job(data_dir, TASK_EVENT_JOB_TYPE, normalized_task_id, saved)
        return dict(event)


def set_task_shortfall(
    data_dir: Path | str,
    *,
    task_id: str,
    shortfall: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return {}
    clean_shortfall = sanitize_task_shortfall(shortfall)
    with _LEDGER_LOCK:
        record = load_document_job(data_dir, TASK_EVENT_JOB_TYPE, normalized_task_id)
        if not record:
            record = {
                "schema_version": TASK_EVENT_LEDGER_SCHEMA_VERSION,
                "task_id": normalized_task_id,
                "events": [],
            }
        if clean_shortfall:
            record["shortfall"] = clean_shortfall
        else:
            record.pop("shortfall", None)
        record["updated_at"] = time.time()
        persist_document_job(data_dir, TASK_EVENT_JOB_TYPE, normalized_task_id, record)
        return clean_shortfall


def get_task_event_ledger(data_dir: Path | str, task_id: str) -> dict[str, Any]:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return {}
    with _LEDGER_LOCK:
        record = load_document_job(data_dir, TASK_EVENT_JOB_TYPE, normalized_task_id)
    if not record:
        return {}
    events = [dict(item) for item in record.get("events") or [] if isinstance(item, dict)]
    return {
        "schema_version": int(record.get("schema_version") or TASK_EVENT_LEDGER_SCHEMA_VERSION),
        "task_id": normalized_task_id,
        "events": events,
        "shortfall": sanitize_task_shortfall(record.get("shortfall")),
        "updated_at": record.get("updated_at") or 0,
    }


__all__ = [
    "TASK_EVENT_JOB_TYPE",
    "TASK_EVENT_LEDGER_SCHEMA_VERSION",
    "append_task_event",
    "classify_error_code",
    "get_task_event_ledger",
    "normalize_event_stage",
    "normalize_event_status",
    "normalize_diagnostic_reason",
    "sanitize_task_shortfall",
    "set_task_shortfall",
]
