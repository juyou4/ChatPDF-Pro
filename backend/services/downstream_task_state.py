"""Durable lifecycle records for document-scoped downstream AI work.

Parsing has a durable task model already.  Overview, reading-outline and
section-outline generation used to keep their progress only in a request or an
in-memory task registry, which made an app restart look like an indefinitely
running job.  This module deliberately stores only resumable public state and
identity metadata: credentials and endpoints never reach disk.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from services.document_job_store import load_document_job, persist_document_job


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
        "identity": _clean_identity(identity),
        "metadata": _sanitize_metadata(metadata or {}),
        "created_at": now,
        "updated_at": now,
    }
    persist_document_job(data_dir, downstream_task_job_type(record["purpose"]), record["doc_id"], record)
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
    if error is not None:
        record["error"] = str(error or "")[:1000]
    if retryable is not None:
        record["retryable"] = bool(retryable)
    if result is not _UNSET:
        record["result"] = result
    persist_document_job(data_dir, downstream_task_job_type(normalized_purpose), normalized_doc_id, record)
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
        record["status"] = "failed"
        record["stage"] = "restart_recovery"
        record["error"] = "服务重启导致下游 AI 任务中断，请重新执行"
        record["retryable"] = True
        record["recovered_after_restart"] = True
        record["updated_at"] = time.time()
        persist_document_job(data_dir, downstream_task_job_type(normalized_purpose), normalized_doc_id, record)
    return record


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
    "TERMINAL_DOWNSTREAM_TASK_STATUSES",
    "build_downstream_task_identity",
    "create_downstream_task",
    "downstream_task_identity_matches",
    "downstream_task_job_type",
    "get_downstream_task",
    "transition_downstream_task",
]
