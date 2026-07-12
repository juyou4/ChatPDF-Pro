from services.document_job_store import (
    load_document_job,
    persist_document_job,
    recover_interrupted_document_job,
)


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
    assert load_document_job(tmp_path, "mineru_deep_parse", "doc-1")["status"] == "failed"
