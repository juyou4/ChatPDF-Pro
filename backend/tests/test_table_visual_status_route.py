from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import routes.document_routes as document_routes


@pytest.mark.parametrize("state", ["queued", "running", "confirmed", "conflict", "indeterminate", "failed"])
def test_table_visual_status_route_is_document_scoped(monkeypatch, state):
    app = FastAPI()
    app.include_router(document_routes.router)
    monkeypatch.setattr(document_routes, "documents_store", {"doc-a": {"data": {}}})
    monkeypatch.setattr(
        document_routes,
        "get_table_visual_verification_status",
        lambda doc_id, task_id: {
            "doc_id": doc_id,
            "task_id": task_id,
            "table_instance_id": "table-v1-a",
            "state": state,
            "verdict": state if state in {"confirmed", "conflict", "indeterminate"} else "indeterminate",
            "diagnostics": {"state": state},
        } if task_id == "tv_known" else {},
    )

    with TestClient(app) as client:
        response = client.get("/documents/doc-a/table-visual-verifications/tv_known")
        assert response.status_code == 200
        assert response.json()["state"] == state
        assert client.get("/documents/doc-a/table-visual-verifications/tv_missing").status_code == 404
        assert client.get("/documents/not-found/table-visual-verifications/tv_known").status_code == 404
