"""Durable lifecycle records for document-scoped downstream AI work.

Parsing has a durable task model already.  Overview, reading-outline and
section-outline generation used to keep their progress only in a request or an
in-memory task registry, which made an app restart look like an indefinitely
running job.  This module deliberately stores only resumable public state and
identity metadata: credentials and endpoints never reach disk.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from services.document_job_store import load_document_job, persist_document_job
from services.task_event_ledger import (
    append_task_event,
    classify_error_code,
    get_task_event_ledger,
    sanitize_task_shortfall,
)


DOWNSTREAM_TASK_SCHEMA_VERSION = 1
DOWNSTREAM_TASK_STATUSES = {
    "queued",
    "running",
    "verifying",
    "publishing",
    "succeeded",
    "partial",
    "degraded",
    "failed",
    "cancelled",
}
ACTIVE_DOWNSTREAM_TASK_STATUSES = {"queued", "running", "verifying", "publishing"}
TERMINAL_DOWNSTREAM_TASK_STATUSES = DOWNSTREAM_TASK_STATUSES - ACTIVE_DOWNSTREAM_TASK_STATUSES

_SENSITIVE_METADATA_TOKENS = ("api_key", "token", "secret", "password", "authorization", "endpoint", "host")
_UNSET = object()
_WORKER_INSTANCE_ID = uuid.uuid4().hex


def _read_stall_seconds() -> float:
    """Read the upper bound for a silent downstream task.

    A task may legitimately spend several minutes in one model call, so this
    is deliberately much longer than the normal polling interval.  The value
    remains configurable for desktop/CI environments where a shorter bound is
    useful for detecting a dead worker quickly.
    """
    try:
        value = float(os.environ.get("CHATPDF_DOWNSTREAM_TASK_STALL_SECONDS", "1800"))
    except (TypeError, ValueError):
        value = 1800.0
    return max(30.0, min(value, 24 * 60 * 60))


DOWNSTREAM_TASK_STALL_SECONDS = _read_stall_seconds()


def _record_age_seconds(record: Mapping[str, Any]) -> float | None:
    value = record.get("updated_at") or record.get("created_at")
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    return max(0.0, time.time() - timestamp)


def _recover_active_record(
    data_dir: Path | str,
    *,
    purpose: str,
    doc_id: str,
    record: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Persist one terminal outcome for an active record that cannot proceed."""
    recovered = dict(record)
    is_stalled = reason == "stalled"
    if is_stalled:
        recovered.update({
            "status": "failed",
            "stage": "stalled",
            "error": "任务长时间没有进展，已停止等待；请检查模型服务/网络后重试",
            "error_code": "downstream_task_stalled",
            "stalled": True,
        })
        shortfall = {
            "kind": "downstream_ai",
            "code": "downstream_task_stalled",
            "stage": "stalled",
            "retryable": True,
        }
    else:
        recovered.update({
            "status": "failed",
            "stage": "restart_recovery",
            "error": "服务重启导致下游 AI 任务中断，请重新执行",
            "error_code": "worker_interrupted",
            "recovered_after_restart": True,
        })
        shortfall = {
            "kind": "restart_recovery",
            "code": "worker_interrupted",
            "stage": "restart_recovery",
            "retryable": True,
        }
    recovered["retryable"] = True
    recovered["active"] = False
    recovered["terminal"] = True
    recovered["shortfall"] = sanitize_task_shortfall(shortfall)
    recovered["updated_at"] = time.time()
    persist_document_job(data_dir, downstream_task_job_type(purpose), doc_id, recovered)
    try:
        identity = recovered.get("identity") if isinstance(recovered.get("identity"), Mapping) else {}
        append_task_event(
            data_dir,
            task_id=str(recovered.get("task_id") or ""),
            stage=str(recovered.get("stage") or "failed"),
            status="failed",
            identity={"route": "downstream", **dict(identity)},
            error_code=str(recovered.get("error_code") or "downstream_task_stalled"),
            shortfall=recovered.get("shortfall"),
        )
    except Exception:
        pass
    return recovered


def build_downstream_task_identity(
    *,
    doc_id: str,
    parse_generation: str,
    document_source_hash: str,
    block_index_revision: str,
    provider: str,
    model: str,
    prompt_version: str,
) -> dict[str, str]:
    """Build the immutable identity a published downstream result must own."""
    return {
        "doc_id": str(doc_id or "").strip(),
        "parse_generation": str(parse_generation or "").strip(),
        "document_source_hash": str(document_source_hash or "").strip(),
        "block_index_revision": str(block_index_revision or "").strip(),
        "provider": str(provider or "").strip().lower(),
        "model": str(model or "").strip(),
        "prompt_version": str(prompt_version or "").strip(),
    }


def downstream_task_job_type(purpose: str) -> str:
    return f"ai_{str(purpose or 'downstream').strip().lower() or 'downstream'}"


def create_downstream_task(
    data_dir: Path | str,
    *,
    purpose: str,
    doc_id: str,
    identity: Mapping[str, Any],
    task_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the latest durable run for one document/purpose pair."""
    now = time.time()
    record = {
        "schema_version": DOWNSTREAM_TASK_SCHEMA_VERSION,
        "task_id": str(task_id or uuid.uuid4()),
        "purpose": str(purpose or "downstream").strip().lower() or "downstream",
        "doc_id": str(doc_id or "").strip(),
        "status": "queued",
        "stage": "queued",
        "retryable": True,
        "active": True,
        "terminal": False,
        "identity": _clean_identity(identity),
        "metadata": _sanitize_metadata(metadata or {}),
        "created_at": now,
        "updated_at": now,
        "worker_instance_id": _WORKER_INSTANCE_ID,
    }
    persist_document_job(data_dir, downstream_task_job_type(record["purpose"]), record["doc_id"], record)
    try:
        append_task_event(
            data_dir,
            task_id=record["task_id"],
            stage="queued",
            status="queued",
            identity={"route": "downstream", **record["identity"]},
            shortfall=record.get("shortfall"),
        )
    except Exception:
        # The latest task state remains authoritative if event diagnostics
        # cannot be written (for example, a read-only data directory).
        pass
    return record


def transition_downstream_task(
    data_dir: Path | str,
    *,
    purpose: str,
    doc_id: str,
    task_id: str,
    status: str,
    stage: str = "",
    error: str | None = None,
    retryable: bool | None = None,
    result: Any = _UNSET,
    shortfall: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically move a durable task forward if it still owns the record.

    A newly started task replaces the record for the same document/purpose.
    The task id fence prevents an older worker from overwriting that newer
    state after it completes.
    """
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in DOWNSTREAM_TASK_STATUSES:
        raise ValueError(f"Unsupported downstream task status: {status}")

    normalized_purpose = str(purpose or "downstream").strip().lower() or "downstream"
    normalized_doc_id = str(doc_id or "").strip()
    record = load_document_job(data_dir, downstream_task_job_type(normalized_purpose), normalized_doc_id)
    if not record or str(record.get("task_id") or "") != str(task_id or ""):
        return dict(record or {})

    current_status = str(record.get("status") or "").strip().lower()
    if current_status in TERMINAL_DOWNSTREAM_TASK_STATUSES and normalized_status in ACTIVE_DOWNSTREAM_TASK_STATUSES:
        return dict(record)

    record["status"] = normalized_status
    record["stage"] = str(stage or normalized_status).strip() or normalized_status
    record["updated_at"] = time.time()
    # Stamp ownership on records created by older builds when they are touched
    # by the current worker.  A later process can then distinguish a live task
    # from a restart orphan without relying only on elapsed time.
    record.setdefault("worker_instance_id", _WORKER_INSTANCE_ID)
    record["active"] = normalized_status in ACTIVE_DOWNSTREAM_TASK_STATUSES
    record["terminal"] = normalized_status in TERMINAL_DOWNSTREAM_TASK_STATUSES
    if error is not None:
        record["error"] = str(error or "")[:1000]
        record["error_code"] = classify_error_code(error)
    elif normalized_status in ACTIVE_DOWNSTREAM_TASK_STATUSES or normalized_status in {"succeeded"}:
        record["error"] = ""
        record.pop("error_code", None)
    if retryable is not None:
        record["retryable"] = bool(retryable)
    if result is not _UNSET:
        record["result"] = result
    if shortfall is not None:
        clean_shortfall = sanitize_task_shortfall(shortfall)
        if clean_shortfall:
            record["shortfall"] = clean_shortfall
        else:
            record.pop("shortfall", None)
    elif normalized_status in ACTIVE_DOWNSTREAM_TASK_STATUSES or normalized_status == "succeeded":
        record.pop("shortfall", None)
    persist_document_job(data_dir, downstream_task_job_type(normalized_purpose), normalized_doc_id, record)
    try:
        identity = record.get("identity") if isinstance(record.get("identity"), Mapping) else {}
        append_task_event(
            data_dir,
            task_id=str(task_id or ""),
            stage=stage or normalized_status,
            status=normalized_status,
            identity={"route": "downstream", **dict(identity)},
            error_code=classify_error_code(error),
            degraded_reason=(record.get("shortfall") or {}).get("code", "")
            if isinstance(record.get("shortfall"), Mapping)
            else "",
            shortfall=record.get("shortfall"),
        )
    except Exception:
        pass
    return record


def get_downstream_task(
    data_dir: Path | str,
    *,
    purpose: str,
    doc_id: str,
    recover_interrupted: bool = True,
) -> dict[str, Any]:
    """Read the latest state and convert a stranded active record to retryable failure."""
    normalized_purpose = str(purpose or "downstream").strip().lower() or "downstream"
    normalized_doc_id = str(doc_id or "").strip()
    record = load_document_job(data_dir, downstream_task_job_type(normalized_purpose), normalized_doc_id)
    if not record:
        return {}
    record = dict(record)
    if recover_interrupted and str(record.get("status") or "").strip().lower() in ACTIVE_DOWNSTREAM_TASK_STATUSES:
        worker_instance = str(record.get("worker_instance_id") or "").strip()
        is_foreign_worker = bool(worker_instance and worker_instance != _WORKER_INSTANCE_ID)
        # Records written before the worker marker was introduced are treated
        # as restart leftovers. New records from this process are never
        # mistaken for interrupted work while their request is still active.
        if is_foreign_worker or not worker_instance:
            return _recover_active_record(
                data_dir,
                purpose=normalized_purpose,
                doc_id=normalized_doc_id,
                record=record,
                reason="restart",
            )
        age = _record_age_seconds(record)
        if age is not None and age >= DOWNSTREAM_TASK_STALL_SECONDS:
            return _recover_active_record(
                data_dir,
                purpose=normalized_purpose,
                doc_id=normalized_doc_id,
                record=record,
                reason="stalled",
            )
    return record


def get_downstream_task_events(
    data_dir: Path | str,
    *,
    task_id: str,
) -> dict[str, Any]:
    """Return only the sanitized event envelope for a public status route."""
    return get_task_event_ledger(data_dir, task_id)


def downstream_task_identity_matches(record: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    """Require every present identity fence to match exactly."""
    saved = record.get("identity") if isinstance(record, Mapping) else {}
    if not isinstance(saved, Mapping):
        return False
    expected = _clean_identity(identity)
    return all(str(saved.get(key) or "") == value for key, value in expected.items())


def _clean_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value or "").strip()
        for key, value in dict(identity or {}).items()
        if str(key or "").strip()
    }


def _sanitize_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep small public metadata, dropping credentials and network locations."""
    cleaned: dict[str, Any] = {}
    for raw_key, raw_value in dict(value or {}).items():
        key = str(raw_key or "").strip()
        if not key or any(token in key.lower() for token in _SENSITIVE_METADATA_TOKENS):
            continue
        if isinstance(raw_value, Mapping):
            cleaned[key] = _sanitize_metadata(raw_value)
        elif isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
            cleaned[key] = str(raw_value)[:320] if isinstance(raw_value, str) else raw_value
        elif isinstance(raw_value, (list, tuple)):
            cleaned[key] = [
                str(item)[:160] if isinstance(item, str) else item
                for item in list(raw_value)[:20]
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
    return cleaned


__all__ = [
    "ACTIVE_DOWNSTREAM_TASK_STATUSES",
    "DOWNSTREAM_TASK_SCHEMA_VERSION",
    "DOWNSTREAM_TASK_STATUSES",
    "DOWNSTREAM_TASK_STALL_SECONDS",
    "TERMINAL_DOWNSTREAM_TASK_STATUSES",
    "build_downstream_task_identity",
    "create_downstream_task",
    "downstream_task_identity_matches",
    "get_downstream_task_events",
    "downstream_task_job_type",
    "get_downstream_task",
    "transition_downstream_task",
]
