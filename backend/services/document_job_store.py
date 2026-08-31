"""Small durable job store for document-scoped background work."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def document_job_path(data_dir: Path | str, job_type: str, doc_id: str) -> Path:
    safe_type = _safe_path_part(job_type, "job")
    safe_doc_id = _safe_path_part(doc_id, "document")
    return Path(data_dir) / "document_jobs" / safe_type / f"{safe_doc_id}.json"


def load_document_job(data_dir: Path | str, job_type: str, doc_id: str) -> dict[str, Any]:
    path = document_job_path(data_dir, job_type, doc_id)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def persist_document_job(data_dir: Path | str, job_type: str, doc_id: str, record: dict[str, Any]) -> Path:
    path = document_job_path(data_dir, job_type, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".{uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


def recover_interrupted_document_job(
    data_dir: Path | str,
    job_type: str,
    doc_id: str,
    record: dict[str, Any],
    *,
    updated_at: str,
) -> dict[str, Any]:
    """Mark an in-memory-only job as failed after a process restart."""
    recovered = dict(record)
    if recovered.get("status") in {"queued", "running"}:
        recovered.update({
            "status": "failed",
            "stage": "restart_recovery",
            "error": "服务重启导致任务中断，请重新执行",
            "updated_at": updated_at,
            "recovered_after_restart": True,
            "remote_state": "failed",
        })
        for key in (
            "remote_progress_percent",
            "remote_progress_source",
            "remote_pages_completed",
            "remote_pages_total",
            "poll_attempt",
            "poll_total",
        ):
            recovered.pop(key, None)
        persist_document_job(data_dir, job_type, doc_id, recovered)
    return recovered


def _safe_path_part(value: Any, fallback: str) -> str:
    text = "".join(char for char in str(value or "") if char.isalnum() or char in {"-", "_", "."})
    return text[:128] or fallback
