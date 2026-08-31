from services.document_job_store import (
    load_document_job,
    persist_document_job,
    recover_interrupted_document_job,
)
from services import downstream_task_state


def test_running_document_job_is_recovered_after_restart(tmp_path):
    record = {"doc_id": "doc-1", "status": "running", "stage": "uploading"}
    persist_document_job(tmp_path, "mineru_deep_parse", "doc-1", record)

    recovered = recover_interrupted_document_job(
        tmp_path,
        "mineru_deep_parse",
        "doc-1",
        load_document_job(tmp_path, "mineru_deep_parse", "doc-1"),
        updated_at="2026-07-12T00:00:00",
    )

    assert recovered["status"] == "failed"
    assert recovered["stage"] == "restart_recovery"
    assert recovered["recovered_after_restart"] is True
    assert recovered["remote_state"] == "failed"
    assert "remote_progress_percent" not in recovered
    assert load_document_job(tmp_path, "mineru_deep_parse", "doc-1")["status"] == "failed"


def _downstream_identity():
    return downstream_task_state.build_downstream_task_identity(
        doc_id="doc-ai",
        parse_generation="parse-1",
        document_source_hash="source-1",
        block_index_revision="blocks-1",
        provider="openai",
        model="gpt-test",
        prompt_version="v1",
    )


def test_live_downstream_task_is_not_mistaken_for_restart(tmp_path):
    task = downstream_task_state.create_downstream_task(
        tmp_path,
        purpose="reading_outline",
        doc_id="doc-ai",
        identity=_downstream_identity(),
    )
    downstream_task_state.transition_downstream_task(
        tmp_path,
        purpose="reading_outline",
        doc_id="doc-ai",
        task_id=task["task_id"],
        status="running",
        stage="generating",
    )

    current = downstream_task_state.get_downstream_task(
        tmp_path,
        purpose="reading_outline",
        doc_id="doc-ai",
    )

    assert current["status"] == "running"
    assert current["active"] is True
    assert current["terminal"] is False


def test_foreign_downstream_worker_is_recovered_as_terminal(tmp_path):
    task = downstream_task_state.create_downstream_task(
        tmp_path,
        purpose="section_outline",
        doc_id="doc-ai",
        identity=_downstream_identity(),
    )
    task.update({
        "status": "running",
        "stage": "generating",
        "worker_instance_id": "previous-process",
    })
    persist_document_job(tmp_path, "ai_section_outline", "doc-ai", task)

    recovered = downstream_task_state.get_downstream_task(
        tmp_path,
        purpose="section_outline",
        doc_id="doc-ai",
    )

    assert recovered["status"] == "failed"
    assert recovered["stage"] == "restart_recovery"
    assert recovered["error_code"] == "worker_interrupted"
    assert recovered["active"] is False
    assert recovered["terminal"] is True
    assert recovered["retryable"] is True


def test_silent_downstream_task_is_terminalized(monkeypatch, tmp_path):
    task = downstream_task_state.create_downstream_task(
        tmp_path,
        purpose="overview",
        doc_id="doc-ai",
        identity=_downstream_identity(),
    )
    task.update({
        "status": "running",
        "stage": "generating",
        "updated_at": 1,
    })
    persist_document_job(tmp_path, "ai_overview", "doc-ai", task)
    monkeypatch.setattr(downstream_task_state, "DOWNSTREAM_TASK_STALL_SECONDS", 1)

    recovered = downstream_task_state.get_downstream_task(
        tmp_path,
        purpose="overview",
        doc_id="doc-ai",
    )

    assert recovered["status"] == "failed"
    assert recovered["stage"] == "stalled"
    assert recovered["error_code"] == "downstream_task_stalled"
    assert recovered["shortfall"]["code"] == "downstream_task_stalled"
