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

    # 深度解析已从「每任务直启线程」改为固定 worker 池 + 队列，Thread 不再携带
    # 任务参数。恢复的观察点因此移到入队接缝：只有 _enqueue_mineru_deep_parse
    # 返回 True，任务才算恢复成功。
    def fake_enqueue(doc_id, cancel_event, remote_job, parse_generation, full_route_options):
        captured["doc_id"] = doc_id
        captured["remote_job"] = remote_job
        captured["parse_generation"] = parse_generation
        return True

    monkeypatch.setattr(document_routes, "_enqueue_mineru_deep_parse", fake_enqueue)

    resumed = document_routes.resume_pending_mineru_deep_parse_jobs()

    assert resumed == [{"doc_id": "doc-1", "batch_id": "batch-1"}]
    assert captured["doc_id"] == "doc-1"
    # 远端身份（batch/data id）必须原样带回队列，轮询才接得上原任务。
    assert captured["remote_job"]["batch_id"] == "batch-1"
    assert captured["remote_job"]["data_id"] == "data-1"


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


@pytest.mark.parametrize("access_mode", ["direct", "worker"])
def test_completed_remote_download_can_resume_without_reupload(access_mode):
    assert document_routes._can_resume_direct_mineru_result_download({
        "status": "failed",
        "stage": "download_failed",
        "error_code": "mineru_download_failed",
        "access_mode": access_mode,
        "batch_id": "batch-1",
    }) is True
