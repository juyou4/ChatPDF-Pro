"""上传路由参数解析回归测试

验证 multipart/form-data 中的 embedding 配置字段会被后端正确读取，
避免 FastAPI 回退到默认 local-minilm 导致桌面版误判为本地模型。
"""

import json
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

    def fake_create_index(
        doc_id,
        full_text,
        vector_store_dir,
        embedding_model,
        api_key,
        api_host,
        pages=None,
        structured_table_bundles=None,
        summary_api_key=None,
    ):
        captured["doc_id"] = doc_id
        captured["embedding_model"] = embedding_model
        captured["api_key"] = api_key
        captured["api_host"] = api_host
        captured["pages"] = pages
        captured["structured_table_bundles"] = structured_table_bundles
        captured["summary_api_key"] = summary_api_key

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
    assert captured["pages"] == [{"page_num": 1, "text": "hello world", "content": "hello world"}]
    assert captured["structured_table_bundles"] is None
    assert captured["summary_api_key"] == "sk-test-123"


def test_upload_prefers_api_key_for_semantic_group_summary(client, monkeypatch, isolated_storage):
    """显式传入 api_key 时，应优先用于语义意群摘要生成。"""
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
    monkeypatch.setattr(document_routes_module, "generate_doc_id", lambda _: "doc-summary-key")

    captured = {}

    def fake_create_index(
        doc_id,
        full_text,
        vector_store_dir,
        embedding_model,
        api_key,
        api_host,
        pages=None,
        structured_table_bundles=None,
        summary_api_key=None,
    ):
        captured["api_key"] = api_key
        captured["structured_table_bundles"] = structured_table_bundles
        captured["summary_api_key"] = summary_api_key

    monkeypatch.setattr(document_routes_module, "create_index", fake_create_index)

    resp = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 mock", "application/pdf")},
        data={
            "embedding_model": "silicon:BAAI/bge-m3",
            "embedding_api_key": "sk-embed-123",
            "api_key": "sk-chat-456",
            "enable_ocr": "never",
        },
    )

    assert resp.status_code == 200
    assert captured["api_key"] == "sk-embed-123"
    assert captured["structured_table_bundles"] is None
    assert captured["summary_api_key"] == "sk-chat-456"


def test_upload_passes_odl_structured_table_bundles_to_index(client, monkeypatch, isolated_storage):
    """ODL 覆盖层应保留结构化 table bundles，并透传给向量索引。"""
    monkeypatch.setattr(document_routes_module.runtime, "CHATPDF_MODE", "desktop")

    monkeypatch.setattr(
        document_routes_module,
        "extract_text_from_pdf",
        lambda *args, **kwargs: {
            "full_text": "raw pdfplumber text",
            "total_pages": 1,
            "pages": [{"page": 1, "text": "raw pdfplumber text", "content": "raw pdfplumber text"}],
            "ocr_used": False,
        },
    )
    monkeypatch.setattr(document_routes_module, "generate_doc_id", lambda _: "doc-odl-bundle")

    import services.odl_parser_service as odl_parser_service

    monkeypatch.setattr(odl_parser_service, "is_odl_available", lambda: True)
    monkeypatch.setattr(
        odl_parser_service,
        "parse_pdf_odl",
        lambda _path: {
            "full_text": "clean odl text",
            "total_pages": 1,
            "pages": [{
                "page": 1,
                "text": "clean odl text",
                "content": "clean odl text",
                "source": "odl",
                "table_bundles": [
                    {
                        "bundle_id": "id:42",
                        "table_id": "Table 7",
                        "table_caption": "Table 7: Main results",
                        "table_header": "Method | All",
                        "table_body_markdown": "| Method | All |\n| --- | --- |\n| Ours | 55.5 |",
                        "table_markdown": "| Method | All |\n| --- | --- |\n| Ours | 55.5 |",
                        "bundle_text": "[Structured Table Bundle]\n\nTable 7: Main results",
                        "evidence_units": [
                            {
                                "evidence_unit_id": "id:42::table_row::source:42::r1",
                                "evidence_unit_type": "table_row",
                                "table_bundle_id": "id:42",
                                "table_id": "Table 7",
                                "table_caption": "Table 7: Main results",
                                "source_id": 42,
                                "page": 1,
                                "row_idx": 1,
                                "bbox": [1, 2, 3, 4],
                                "content": "Method\u0000 | All",
                                "cell_count": 2,
                                "cell_evidence_units": [
                                    {
                                        "evidence_unit_id": "id:42::table_cell::source:42::r1::c1",
                                        "evidence_unit_type": "table_cell",
                                        "table_bundle_id": "id:42",
                                        "table_id": "Table 7",
                                        "table_caption": "Table 7: Main results",
                                        "source_id": 42,
                                        "page": 1,
                                        "row_idx": 1,
                                        "col_idx": 1,
                                        "row_span": 1,
                                        "col_span": 1,
                                        "bbox": [1, 2, 2, 4],
                                        "content": "Method\u0000",
                                        "source": "odl",
                                    },
                                    {
                                        "evidence_unit_id": "id:42::table_cell::source:42::r1::c2",
                                        "evidence_unit_type": "table_cell",
                                        "table_bundle_id": "id:42",
                                        "table_id": "Table 7",
                                        "table_caption": "Table 7: Main results",
                                        "source_id": 42,
                                        "page": 1,
                                        "row_idx": 1,
                                        "col_idx": 2,
                                        "row_span": 1,
                                        "col_span": 1,
                                        "bbox": [2, 2, 3, 4],
                                        "content": "All",
                                        "source": "odl",
                                    },
                                ],
                                "source": "odl",
                            }
                        ],
                    }
                ],
            }],
            "extraction_method": "odl",
            "odl_element_count": 8,
            "odl_kept_count": 5,
            "odl_soft_kept_caption_count": 1,
            "extraction_quality": "odl_clean",
            "structured_table_bundles": [
                {
                    "bundle_id": "id:42",
                    "table_id": "Table 7",
                    "table_caption": "Table 7: Main results",
                    "table_header": "Method | All",
                    "table_body_markdown": "| Method | All |\n| --- | --- |\n| Ours | 55.5 |",
                    "html_table": "<table><tr><th>Method</th><th>All</th></tr></table>",
                    "page_start": 1,
                    "page_end": 1,
                    "pages": [1],
                    "source_ids": [42],
                    "source": "odl",
                    "evidence_units": [
                        {
                            "evidence_unit_id": "id:42::table_row::source:42::r1",
                            "evidence_unit_type": "table_row",
                            "table_bundle_id": "id:42",
                            "table_id": "Table 7",
                            "table_caption": "Table 7: Main results",
                            "source_id": 42,
                            "page": 1,
                            "row_idx": 1,
                            "bbox": [1, 2, 3, 4],
                            "content": "Method\u0000 | All",
                            "cell_count": 2,
                            "cell_evidence_units": [
                                {
                                    "evidence_unit_id": "id:42::table_cell::source:42::r1::c1",
                                    "evidence_unit_type": "table_cell",
                                    "table_bundle_id": "id:42",
                                    "table_id": "Table 7",
                                    "table_caption": "Table 7: Main results",
                                    "source_id": 42,
                                    "page": 1,
                                    "row_idx": 1,
                                    "col_idx": 1,
                                    "row_span": 1,
                                    "col_span": 1,
                                    "bbox": [1, 2, 2, 4],
                                    "content": "Method\u0000",
                                    "source": "odl",
                                },
                                {
                                    "evidence_unit_id": "id:42::table_cell::source:42::r1::c2",
                                    "evidence_unit_type": "table_cell",
                                    "table_bundle_id": "id:42",
                                    "table_id": "Table 7",
                                    "table_caption": "Table 7: Main results",
                                    "source_id": 42,
                                    "page": 1,
                                    "row_idx": 1,
                                    "col_idx": 2,
                                    "row_span": 1,
                                    "col_span": 1,
                                    "bbox": [2, 2, 3, 4],
                                    "content": "All",
                                    "source": "odl",
                                },
                            ],
                            "source": "odl",
                        }
                    ],
                }
            ],
            "structured_table_count": 1,
        },
    )

    captured = {}

    def fake_create_index(
        doc_id,
        full_text,
        vector_store_dir,
        embedding_model,
        api_key,
        api_host,
        pages=None,
        structured_table_bundles=None,
        summary_api_key=None,
    ):
        captured["doc_id"] = doc_id
        captured["full_text"] = full_text
        captured["pages"] = pages
        captured["structured_table_bundles"] = structured_table_bundles
        captured["summary_api_key"] = summary_api_key

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
    assert body["doc_id"] == "doc-odl-bundle"
    assert body["extraction_method"] == "odl"
    assert body["structured_table_count"] == 1

    assert captured["doc_id"] == "doc-odl-bundle"
    assert captured["full_text"] == "clean odl text"
    assert captured["pages"][0]["text"] == "clean odl text"
    assert captured["pages"][0]["table_bundles"][0]["table_id"] == "Table 7"
    assert captured["pages"][0]["table_bundles"][0]["evidence_units"][0]["content"] == "Method | All"
    assert captured["pages"][0]["table_bundles"][0]["evidence_units"][0]["cell_evidence_units"][0]["content"] == "Method"
    assert captured["structured_table_bundles"][0]["table_id"] == "Table 7"
    assert captured["structured_table_bundles"][0]["table_caption"] == "Table 7: Main results"
    assert captured["structured_table_bundles"][0]["evidence_units"][0]["content"] == "Method | All"
    assert captured["structured_table_bundles"][0]["evidence_units"][0]["cell_evidence_units"][0]["content"] == "Method"

    saved_doc = json.loads((document_routes_module.DOCS_DIR / "doc-odl-bundle.json").read_text(encoding="utf-8"))
    saved_bundle = saved_doc["data"]["structured_table_bundles"][0]
    assert saved_bundle["evidence_units"][0]["content"] == "Method | All"
    assert saved_bundle["evidence_units"][0]["cell_evidence_units"][0]["content"] == "Method"


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
