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
from tests.vector_index_fixtures import write_published_vector_index


def _persist_captured_index(doc_id, full_text, vector_store_dir, embedding_model, kwargs, *, api_host=""):
    """后台索引质量门会验证 create_index 的真实产物，替身必须照做。

    持久化的 embedding 身份必须与请求身份一致，否则语义组构建会因
    「Embedding 配置与文档索引不一致」被 409 拒绝。
    """
    # create_index 现在收到的是去 provider 前缀的裸模型名，provider 经
    # 独立 kwarg 传入；持久化身份必须按这套拆分，否则语义组构建 409。
    model = str(embedding_model or "")
    provider = str(kwargs.get("embedding_provider") or "local")
    write_published_vector_index(
        Path(vector_store_dir),
        doc_id,
        chunks=[full_text or "chunk"],
        index_source=str(kwargs.get("index_source") or "pdf_native"),
        index_meta=kwargs.get("index_meta") or {},
        embedding_model=model,
        embedding_provider=provider,
        embedding_api_host="" if provider == "local" else str(api_host or ""),
    )


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def isolated_storage(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    docs_dir = tmp_path / "docs"
    vectors_dir = tmp_path / "vectors"
    uploads_dir = tmp_path / "uploads"
    data_dir.mkdir()
    docs_dir.mkdir()
    vectors_dir.mkdir()
    uploads_dir.mkdir()

    # DATA_DIR 不隔离时，上传会把解析清单/任务账本写进真实数据目录，
    # 同一 doc_id 的下一个测试就会撞上陈旧身份而 400。
    monkeypatch.setattr(document_routes_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(document_routes_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(document_routes_module, "VECTOR_STORE_DIR", vectors_dir)
    monkeypatch.setattr(document_routes_module, "UPLOAD_DIR", uploads_dir)
    monkeypatch.setattr(document_routes_module, "documents_store", {})
    document_routes_module._DOCUMENT_INDEX_STATUS.clear()
    document_routes_module._DEEP_PARSE_TASKS.clear()
    document_routes_module._DEEP_PARSE_CANCEL_EVENTS.clear()


def _run_upload_index_queue_synchronously(monkeypatch):
    """测试环境中同步执行后台索引，保留 /upload 的排队返回语义。"""

    def sync_queue(doc_id, embedding_model, embedding_api_key, embedding_api_host, summary_api_key):
        document_routes_module._build_document_indexes(
            doc_id,
            embedding_model,
            embedding_api_key,
            embedding_api_host,
            summary_api_key,
        )
        return document_routes_module._get_document_index_status(doc_id)

    monkeypatch.setattr(document_routes_module, "_queue_document_indexes", sync_queue)


def test_upload_reads_embedding_fields_from_form(client, monkeypatch, isolated_storage):
    """桌面模式下，云端 embedding 的表单字段应透传到 create_index。"""
    monkeypatch.setattr(document_routes_module.runtime, "CHATPDF_MODE", "desktop")
    _run_upload_index_queue_synchronously(monkeypatch)

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
        **kwargs,
    ):
        captured["doc_id"] = doc_id
        captured["embedding_model"] = embedding_model
        captured["api_key"] = api_key
        captured["api_host"] = api_host
        captured["pages"] = pages
        captured["structured_table_bundles"] = structured_table_bundles
        captured["summary_api_key"] = summary_api_key
        _persist_captured_index(doc_id, full_text, vector_store_dir, embedding_model, kwargs, api_host=api_host)

    monkeypatch.setattr(document_routes_module, "create_index", fake_create_index)

    resp = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 mock", "application/pdf")},
        data={
            "embedding_model": "silicon:BAAI/bge-m3",
            # 索引构建要求完整、由用户显式选定的 embedding 身份：索引只在其构建时
            # 的向量空间里有意义，provider 不允许被推断。
            "embedding_provider": "silicon",
            "embedding_api_key": "sk-test-123",
            "embedding_api_host": "https://api.siliconflow.cn",
            # MinerU 现在是 PDF 的缺省主解析路线，这些用例考的是本地解析链路，
            # 必须显式声明，否则会走进异步的 waiting_for_mineru 分支。
            "parse_route": "local",
            "enable_ocr": "never",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc-form-ok"

    assert captured["doc_id"] == "doc-form-ok"
    # 索引构建入口现在接收去 provider 前缀的裸模型名，provider 单独传递，
    # api_host 会被规范化补上 /v1 版本路径。
    assert captured["embedding_model"] == "BAAI/bge-m3"
    assert captured["api_key"] == "sk-test-123"
    assert captured["api_host"] == "https://api.siliconflow.cn/v1"
    # 上传链路会把页面统一成规范化 schema（page/page_index/source 等字段），
    # 这里只锁定对检索有意义的内容字段。
    assert captured["pages"][0]["page"] == 1
    assert captured["pages"][0]["text"] == "hello world"
    assert captured["pages"][0]["content"] == "hello world"
    assert captured["structured_table_bundles"] is None
    # embedding key 不再被静默复用为意群摘要 key：embedding 专用凭证往往无法
    # 调用 chat completions，未显式提供 api_key 时改走确定性截断。
    assert captured["summary_api_key"] is None


def test_upload_prefers_api_key_for_semantic_group_summary(client, monkeypatch, isolated_storage):
    """显式传入 api_key 时，应优先用于语义意群摘要生成。"""
    monkeypatch.setattr(document_routes_module.runtime, "CHATPDF_MODE", "desktop")
    _run_upload_index_queue_synchronously(monkeypatch)

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
        **kwargs,
    ):
        captured["api_key"] = api_key
        captured["structured_table_bundles"] = structured_table_bundles
        captured["summary_api_key"] = summary_api_key
        _persist_captured_index(doc_id, full_text, vector_store_dir, embedding_model, kwargs, api_host=api_host)

    monkeypatch.setattr(document_routes_module, "create_index", fake_create_index)

    resp = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 mock", "application/pdf")},
        data={
            "embedding_model": "silicon:BAAI/bge-m3",
            "embedding_provider": "silicon",
            "embedding_api_key": "sk-embed-123",
            # 远程 embedding provider 现在要求完整显式身份（含 api_host），
            # 缺失会在上传入口被 400 拒绝而不是静默回退。
            "embedding_api_host": "https://api.siliconflow.cn",
            "api_key": "sk-chat-456",
            # MinerU 现在是 PDF 的缺省主解析路线，这些用例考的是本地解析链路，
            # 必须显式声明，否则会走进异步的 waiting_for_mineru 分支。
            "parse_route": "local",
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
    _run_upload_index_queue_synchronously(monkeypatch)

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
        **kwargs,
    ):
        captured["doc_id"] = doc_id
        captured["full_text"] = full_text
        captured["pages"] = pages
        captured["structured_table_bundles"] = structured_table_bundles
        captured["summary_api_key"] = summary_api_key
        _persist_captured_index(doc_id, full_text, vector_store_dir, embedding_model, kwargs, api_host=api_host)

    monkeypatch.setattr(document_routes_module, "create_index", fake_create_index)

    resp = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 mock", "application/pdf")},
        data={
            "embedding_model": "silicon:BAAI/bge-m3",
            "embedding_api_key": "sk-test-123",
            "embedding_api_host": "https://api.siliconflow.cn",
            "embedding_provider": "silicon",
            # MinerU 现在是 PDF 的缺省主解析路线，这些用例考的是本地解析链路，
            # 必须显式声明，否则会走进异步的 waiting_for_mineru 分支。
            "parse_route": "local",
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
    # 结构化表格保持在独立 bundle 合同里透传，不再内嵌进规范化页面对象；
    # 这防止可见文档、检索与下游概览退化成三份各自展开的表格文本。
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
        data={
            "embedding_model": "local:all-MiniLM-L6-v2",
            "embedding_provider": "local",
        },
    )

    assert resp.status_code == 400
    assert "桌面版不支持本地 Embedding 模型" in resp.json()["detail"]


def test_upload_returns_400_when_embedding_model_is_invalid(client, monkeypatch, isolated_storage):
    """后台向量索引阶段的模型错误应写入索引状态，上传本身不阻塞。"""
    monkeypatch.setattr(document_routes_module.runtime, "CHATPDF_MODE", "desktop")
    _run_upload_index_queue_synchronously(monkeypatch)

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
            "embedding_provider": "silicon",
            # MinerU 现在是 PDF 的缺省主解析路线，这些用例考的是本地解析链路，
            # 必须显式声明，否则会走进异步的 waiting_for_mineru 分支。
            "parse_route": "local",
            "enable_ocr": "never",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc-model-invalid"
    assert body["indexing_status"] == "failed"
    status = document_routes_module._get_document_index_status("doc-model-invalid")
    assert status["status"] == "failed"
    assert "Embedding模型" in status["error"]
