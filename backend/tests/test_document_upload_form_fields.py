"""上传路由参数解析回归测试

验证 multipart/form-data 中的 embedding 配置字段会被后端正确读取，
避免 FastAPI 回退到默认 local-minilm 导致桌面版误判为本地模型。
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# 将 backend 目录加入导入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from routes.document_routes import router
import routes.document_routes as document_routes_module


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def isolated_storage(monkeypatch, tmp_path: Path):
    docs_dir = tmp_path / "docs"
    vectors_dir = tmp_path / "vectors"
    uploads_dir = tmp_path / "uploads"
    docs_dir.mkdir()
    vectors_dir.mkdir()
    uploads_dir.mkdir()

    monkeypatch.setattr(document_routes_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(document_routes_module, "VECTOR_STORE_DIR", vectors_dir)
    monkeypatch.setattr(document_routes_module, "UPLOAD_DIR", uploads_dir)


def test_upload_reads_embedding_fields_from_form(client, monkeypatch, isolated_storage):
    """桌面模式下，云端 embedding 的表单字段应透传到 create_index。"""
    monkeypatch.setattr(document_routes_module.runtime, "CHATPDF_MODE", "desktop")

    monkeypatch.setattr(
        document_routes_module,
        "extract_text_from_pdf",
        lambda *args, **kwargs: {
            "full_text": "hello world",
            "total_pages": 1,
            "pages": [{"page_num": 1, "text": "hello world"}],
            "ocr_used": False,
        },
    )
    monkeypatch.setattr(document_routes_module, "generate_doc_id", lambda _: "doc-form-ok")

    captured = {}

    def fake_create_index(doc_id, full_text, vector_store_dir, embedding_model, api_key, api_host, pages=None):
        captured["doc_id"] = doc_id
        captured["embedding_model"] = embedding_model
        captured["api_key"] = api_key
        captured["api_host"] = api_host
        captured["pages"] = pages

    monkeypatch.setattr(document_routes_module, "create_index", fake_create_index)

    resp = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 mock", "application/pdf")},
        data={
            "embedding_model": "silicon:BAAI/bge-m3",
            "embedding_api_key": "sk-test-123",
            "embedding_api_host": "https://api.siliconflow.cn",
            "enable_ocr": "never",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc-form-ok"

    assert captured["doc_id"] == "doc-form-ok"
    assert captured["embedding_model"] == "silicon:BAAI/bge-m3"
    assert captured["api_key"] == "sk-test-123"
    assert captured["api_host"] == "https://api.siliconflow.cn"
    assert captured["pages"] == [{"page_num": 1, "text": "hello world"}]


def test_upload_blocks_local_embedding_in_desktop_mode(client, monkeypatch, isolated_storage):
    """桌面模式下 local embedding 应被明确拦截。"""
    monkeypatch.setattr(document_routes_module.runtime, "CHATPDF_MODE", "desktop")

    resp = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 mock", "application/pdf")},
        data={"embedding_model": "local:all-MiniLM-L6-v2"},
    )

    assert resp.status_code == 400
    assert "桌面版不支持本地 Embedding 模型" in resp.json()["detail"]


def test_upload_returns_400_when_embedding_model_is_invalid(client, monkeypatch, isolated_storage):
    """向量索引阶段的模型错误应返回 400，避免被包装成 500。"""
    monkeypatch.setattr(document_routes_module.runtime, "CHATPDF_MODE", "desktop")

    monkeypatch.setattr(
        document_routes_module,
        "extract_text_from_pdf",
        lambda *args, **kwargs: {
            "full_text": "hello world",
            "total_pages": 1,
            "pages": [{"page_num": 1, "text": "hello world"}],
            "ocr_used": False,
        },
    )
    monkeypatch.setattr(document_routes_module, "generate_doc_id", lambda _: "doc-model-invalid")
    monkeypatch.setattr(
        document_routes_module,
        "create_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Embedding模型 'BAAI/bge-m3' 不存在或未开通。")
        ),
    )

    resp = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 mock", "application/pdf")},
        data={
            "embedding_model": "silicon:BAAI/bge-m3",
            "embedding_api_key": "sk-test-123",
            "embedding_api_host": "https://api.siliconflow.cn",
            "enable_ocr": "never",
        },
    )

    assert resp.status_code == 400
    assert "Embedding模型" in resp.json()["detail"]
