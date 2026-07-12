import json
from pathlib import Path

import pytest

import routes.document_routes as document_routes
from services.document_job_store import persist_document_job


@pytest.fixture
def isolated_jobs(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    (data_dir / "docs").mkdir(parents=True)
    monkeypatch.setattr(document_routes, "DATA_DIR", data_dir)
    monkeypatch.setattr(document_routes, "documents_store", {"doc-1": {"filename": "paper.pdf", "data": {}}})
    document_routes._DEEP_PARSE_TASKS.clear()
    document_routes._DEEP_PARSE_CANCEL_EVENTS.clear()
    yield data_dir
    document_routes._DEEP_PARSE_TASKS.clear()
    document_routes._DEEP_PARSE_CANCEL_EVENTS.clear()


def test_startup_resumes_only_remote_job_with_batch_id(monkeypatch, isolated_jobs):
    persist_document_job(
        isolated_jobs,
        document_routes._DEEP_PARSE_JOB_TYPE,
        "doc-1",
        {"doc_id": "doc-1", "status": "running", "batch_id": "batch-1", "data_id": "data-1", "access_mode": "direct"},
    )
    captured = {}

    class ImmediateThread:
        def __init__(self, *, target, args, **_kwargs):
            captured["target"] = target
            captured["args"] = args

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(document_routes.threading, "Thread", ImmediateThread)

    resumed = document_routes.resume_pending_mineru_deep_parse_jobs()

    assert resumed == [{"doc_id": "doc-1", "batch_id": "batch-1"}]
    assert captured["started"] is True
    assert captured["args"][0] == "doc-1"
    assert captured["args"][2]["data_id"] == "data-1"


def test_cancel_sends_remote_cancel_and_persists_result(monkeypatch, isolated_jobs):
    class Adapter:
        def cancel_batch(self, batch_id, *, data_id=""):
            assert (batch_id, data_id) == ("batch-1", "data-1")
            return {"attempted": True, "state": "sent"}

    monkeypatch.setattr(document_routes, "_load_online_ocr_config", lambda _name: {"access_mode": "direct"})
    monkeypatch.setattr(document_routes, "_make_mineru_adapter", lambda *_args: Adapter())
    event = document_routes.threading.Event()
    document_routes._DEEP_PARSE_CANCEL_EVENTS["doc-1"] = event
    document_routes._set_deep_parse_status(
        "doc-1", "running", stage="polling", batch_id="batch-1", data_id="data-1", access_mode="direct"
    )

    status = document_routes._cancel_mineru_deep_parse("doc-1")

    assert event.is_set()
    assert status["status"] == "cancelled"
    assert status["remote_cancel"] == {"attempted": True, "state": "sent"}
    saved = json.loads(
        (isolated_jobs / "document_jobs" / document_routes._DEEP_PARSE_JOB_TYPE / "doc-1.json").read_text(encoding="utf-8")
    )
    assert saved["remote_cancel"]["state"] == "sent"
