"""MinerU 问答索引重建 / 回滚回归测试。

索引质量门要求真实可读的 FAISS 产物和一整套解析 + embedding/build 身份，
占位字节的假索引会被拒。本文件与 `test_document_upload_form_fields.py`
共用 `tests/vector_index_fixtures.py` 里的「已发布索引」夹具，一次性产出
身份完整、向量真实、代次自洽的索引对。
"""

import json
import os
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.document_routes as document_routes
from services.block_index_service import BLOCK_INDEX_VERSION
from services.semantic_group_store import semantic_group_paths
from tests.vector_index_fixtures import (
    default_semantic_identity,
    make_fake_create_index,
    write_published_vector_index,
)


@pytest.fixture
def isolated_document_routes(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    docs_dir = data_dir / "docs"
    vectors_dir = data_dir / "vector_stores"
    uploads_dir = data_dir / "uploads"
    for path in (docs_dir, vectors_dir, uploads_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(document_routes, "DATA_DIR", data_dir)
    monkeypatch.setattr(document_routes, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(document_routes, "VECTOR_STORE_DIR", vectors_dir)
    monkeypatch.setattr(document_routes, "UPLOAD_DIR", uploads_dir)
    monkeypatch.setattr(document_routes, "documents_store", {})
    document_routes._DOCUMENT_INDEX_STATUS.clear()
    document_routes._DEEP_PARSE_TASKS.clear()
    document_routes._DEEP_PARSE_CANCEL_EVENTS.clear()
    yield data_dir, vectors_dir


def _write_vector_pair(
    vectors_dir: Path,
    doc_id: str,
    *,
    source: str = "pdf_native",
    chunks=None,
    index_meta: dict | None = None,
):
    # 已发布索引必须真实可读且向量数与分块对齐：状态/推荐逻辑现在会实际打开
    # FAISS 文件核验，占位字节会被当作损坏索引处理。
    write_published_vector_index(
        vectors_dir,
        doc_id,
        chunks=list(chunks or ["old chunk"]),
        index_source=source,
        index_meta=index_meta,
    )


def _document_bound_index_meta(doc_id: str, *, block_index_hash: str = "") -> dict:
    """把索引身份绑定到文档当前（合成）解析代次。

    ``ready`` 现在要求索引与活动解析身份匹配（含已发布块树的 revision
    hash）；随机身份的索引会被判为过期产物而不再参与推荐判定。
    """
    manifest = document_routes._read_document_parse_manifest(
        doc_id,
        document_routes.documents_store.get(doc_id),
    )
    meta = {
        "parse_generation": str(manifest.get("generation") or ""),
        "document_source_hash": str(manifest.get("source_hash") or ""),
    }
    if block_index_hash:
        meta["block_index_hash"] = block_index_hash
    return meta


def _write_mineru_payload(data_dir: Path, doc_id: str):
    result_dir = data_dir / "mineru_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "content_list_json": [
            {"type": "header", "text": "8 S. Liu et al.", "page_idx": 0},
            {"type": "text", "text": "1 Introduction\nGrounding DINO text.", "page_idx": 0},
            {
                "type": "table",
                "page_idx": 0,
                "table_caption": "Table 1: Results",
                "table_body": "<table><tr><td>Model</td><td>Acc</td></tr><tr><td>A</td><td>90</td></tr></table>",
                "bbox": [10, 20, 300, 400],
            },
        ]
    }
    with open(result_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump({"version": 1, "doc_id": doc_id, "payload": payload}, f, ensure_ascii=False)


def _write_semantic_group_artifacts(data_dir: Path, doc_id: str):
    # 恢复语义组备份前会完整校验产物（groups 非空、pkl 对齐、FAISS 可读且
    # 向量数一致），损坏的备份宁可降级也不发布，所以夹具必须真实自洽。
    import faiss
    import numpy as np

    groups_dir = data_dir / "semantic_groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    identity = default_semantic_identity(doc_id)
    (groups_dir / f"{doc_id}.json").write_text(
        json.dumps({
            "schema_version": 1,
            "doc_id": doc_id,
            "groups": [{"group_id": "group-0", "title": "旧意群", "chunk_ids": [0]}],
            **identity,
        }),
        encoding="utf-8",
    )
    group_index = faiss.IndexFlatIP(8)
    group_vector = np.ones((1, 8), dtype="float32")
    faiss.normalize_L2(group_vector)
    group_index.add(group_vector)
    faiss.write_index(group_index, str(groups_dir / f"{doc_id}_groups.index"))
    with open(groups_dir / f"{doc_id}_groups.pkl", "wb") as f:
        pickle.dump({"digest_texts": ["old"], "group_ids": ["group-0"], **identity}, f)


def test_read_vector_index_meta_defaults_old_pkl_to_pdf_native(isolated_document_routes):
    _data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-old"
    (vectors_dir / f"{doc_id}.index").write_bytes(b"fake-index")
    with open(vectors_dir / f"{doc_id}.pkl", "wb") as f:
        pickle.dump({"chunks": ["legacy"], "embedding_model": "local-minilm"}, f)

    meta = document_routes._read_vector_index_meta(doc_id)

    assert meta["index_source"] == "pdf_native"
    assert meta["chunk_count"] == 1


def test_deep_parse_status_recommends_rag_rebuild_after_mineru_parse_with_pdf_native_index(isolated_document_routes):
    data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-recommend-rag"
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        # _write_mineru_payload 的条目全在 page_idx 0，即单页文档。质量门按
        # 「已解析页数 / total_pages」算覆盖率，夹具必须与载荷自洽，否则不是
        # 页数未知（expected_page_count_unknown）就是覆盖率不足。
        "data": {"full_text": "old text", "total_pages": 1, "pages": [{"page": 1, "content": "old"}]},
    }
    bound_meta = _document_bound_index_meta(doc_id, block_index_hash="test-block-rev-1")
    _write_vector_pair(
        vectors_dir,
        doc_id,
        source="pdf_native",
        chunks=["old native chunk"],
        index_meta=bound_meta,
    )
    block_index_dir = data_dir / "block_indexes"
    block_index_dir.mkdir(parents=True, exist_ok=True)
    (block_index_dir / f"{doc_id}.json").write_text(
        json.dumps(
            {
                "version": BLOCK_INDEX_VERSION,
                "source": document_routes.MINERU_BLOCK_INDEX_SOURCE,
                # 统一准入规则：已发布块树与向量索引必须携带同一解析身份和
                # 块树 revision hash，否则索引被判为过期而不参与推荐。
                "parse_generation": bound_meta["parse_generation"],
                "document_source_hash": bound_meta["document_source_hash"],
                "block_index_hash": bound_meta["block_index_hash"],
                "pages": [{"page": 1, "blocks": [{"type": "paragraph", "text": "MinerU text"}]}],
                "outline": [{"title": "Introduction", "page": 1}],
            }
        ),
        encoding="utf-8",
    )

    status = document_routes._get_deep_parse_status(doc_id)

    assert status["active_mineru"] is True
    assert status["rag_index"]["index_source"] == "pdf_native"
    assert status["recommend_deep_parse"] is False
    assert status["recommend_rag_index_rebuild"] is True
    assert "问答索引仍使用本地 PDF 解析" in status["recommend_rag_index_reason"]


def test_deep_parse_status_does_not_recommend_rag_rebuild_when_index_is_mineru(isolated_document_routes):
    data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-no-recommend-rag"
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        "data": {"full_text": "mineru text", "total_pages": 1, "pages": [{"page": 1, "content": "mineru"}]},
    }
    _write_vector_pair(vectors_dir, doc_id, source="mineru", chunks=["mineru chunk"])
    block_index_dir = data_dir / "block_indexes"
    block_index_dir.mkdir(parents=True, exist_ok=True)
    (block_index_dir / f"{doc_id}.json").write_text(
        json.dumps(
            {
                "version": BLOCK_INDEX_VERSION,
                "source": document_routes.MINERU_BLOCK_INDEX_SOURCE,
                "pages": [{"page": 1, "blocks": [{"type": "paragraph", "text": "MinerU text"}]}],
                "outline": [{"title": "Introduction", "page": 1}],
            }
        ),
        encoding="utf-8",
    )

    status = document_routes._get_deep_parse_status(doc_id)

    assert status["active_mineru"] is True
    assert status["rag_index"]["index_source"] == "mineru"
    assert status["recommend_rag_index_rebuild"] is False
    assert status["recommend_rag_index_reason"] == ""


def test_assess_recommends_rag_publish_when_ready_parse_has_stale_index(isolated_document_routes):
    _data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-stale-rag-publish"
    manifest = document_routes.build_parse_manifest(
        doc_id=doc_id,
        route="mineru",
        source_hash="src-new",
        generation="gen-new",
        status=document_routes.PARSE_STATUS_READY,
        resolved_route="mineru",
        stage="ready",
        metadata={"full_route": True},
    )
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        "data": {
            "full_text": "mineru text",
            "total_pages": 1,
            "pages": [{"page": 1, "content": "mineru"}],
            "parse_manifest": manifest,
        },
    }
    _write_vector_pair(
        vectors_dir,
        doc_id,
        source="mineru",
        chunks=["stale mineru chunk"],
        index_meta={
            "parse_generation": "gen-old",
            "document_source_hash": "src-old",
        },
    )

    rag_index = document_routes._get_rag_index_status(doc_id)
    result = document_routes._assess_deep_parse_recommendation(
        doc_id,
        True,
        None,
        rag_index=rag_index,
    )

    assert rag_index["ready"] is False
    assert rag_index["matches_active_parse"] is False
    assert result["recommend_rag_index_rebuild"] is True
    assert "尚未按当前解析结果发布" in result["recommend_rag_index_reason"]


def test_rebuild_mineru_rag_index_replaces_after_temp_success_and_can_rollback(monkeypatch, isolated_document_routes):
    data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-rag"
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        "data": {
            "full_text": "1 Introduction\nGrounding DINO text.\nTable 1 Results A 90",
            "total_pages": 1,
            "pages": [{"page": 1, "content": "old"}],
        },
    }
    _write_vector_pair(vectors_dir, doc_id, source="pdf_native", chunks=["old native chunk"])
    _write_mineru_payload(data_dir, doc_id)
    _write_semantic_group_artifacts(data_dir, doc_id)

    captured = {}

    def fake_create_index(
        doc_id_arg,
        full_text,
        vector_store_dir,
        embedding_model,
        api_key,
        api_host,
        pages=None,
        structured_table_bundles=None,
        summary_api_key=None,
        index_source="pdf_native",
        index_meta=None,
        build_semantic_groups=True,
        **_kwargs,
    ):
        captured["full_text"] = full_text
        captured["pages"] = pages
        captured["structured_table_bundles"] = structured_table_bundles
        captured["build_semantic_groups"] = build_semantic_groups
        write_published_vector_index(
            Path(vector_store_dir),
            doc_id_arg,
            chunks=[full_text, "[Structured Table Bundle]\n[Body]\n| Model | Acc |\n| --- | --- |\n| A | 90 |"],
            chunk_metadata=[
                {},
                {
                    "structured_table_bundle": True,
                    "table_body_markdown": structured_table_bundles[0]["table_body_markdown"],
                    "evidence_units": structured_table_bundles[0]["evidence_units"],
                    "source": "mineru",
                },
            ],
            index_source=index_source,
            index_meta=index_meta or {},
            embedding_model=embedding_model,
        )

    monkeypatch.setattr(document_routes, "create_index", fake_create_index)
    fake_embed_fn = object()

    def fake_get_embedding_function(model, api_key=None, base_url=None, **_kwargs):
        captured["embedding_request"] = {"model": model, "api_key": api_key, "base_url": base_url}
        return fake_embed_fn

    monkeypatch.setattr(document_routes, "get_embedding_function", fake_get_embedding_function)

    def fake_build_semantic_group_index(
        doc_id,
        chunks,
        pages,
        embed_fn,
        api_key=None,
        model="gpt-4o-mini",
        provider="openai",
        endpoint="",
        output_dir=None,
        raise_on_error=False,
        **_kwargs,
    ):
        captured["semantic_group_rebuild"] = {
            "doc_id": doc_id,
            "chunks": chunks,
            "pages": pages,
            "embed_fn": embed_fn,
            "api_key": api_key,
            "model": model,
            "provider": provider,
            "endpoint": endpoint,
            "current_index_source": document_routes._read_vector_index_meta(doc_id).get("index_source"),
        }
        return {"status": "disabled", "group_count": 0, "paths": []}

    monkeypatch.setattr(
        document_routes,
            "_build_semantic_group_index",
            fake_build_semantic_group_index,
    )

    result = document_routes._rebuild_mineru_rag_index(
        doc_id,
        embedding_model="local-minilm",
        embedding_api_key="embed-key",
        embedding_api_host=None,
        summary_api_key="summary-key",
        summary_model="deepseek-chat",
        summary_provider="deepseek",
        summary_api_host="https://api.deepseek.com/v1",
    )

    assert result["status"] == "ready"
    assert result["rag_index"]["index_source"] == "mineru"
    assert result["rag_index"]["table_chunk_count"] == 1
    assert result["rag_index"]["can_rollback"] is True
    assert result["backup"]["semantic_groups"]["backed_up"] is True
    assert result["backup"]["semantic_group_cleanup"]["removed"]
    assert captured["build_semantic_groups"] is False
    assert result["semantic_group_rebuild"]["queued"] is False
    assert result["semantic_group_rebuild"]["status"] == "disabled"
    assert result["semantic_group_rebuild"]["chunk_count"] == 2
    # local provider 的 embedding 身份经核验后带上 provider 前缀，并且刻意
    # 不把请求里的 embedding key/host 透传给本地推理路径。
    assert captured["embedding_request"] == {
        "model": "local:local-minilm",
        "api_key": None,
        "base_url": "",
    }
    assert captured["semantic_group_rebuild"]["doc_id"] == doc_id
    assert captured["semantic_group_rebuild"]["chunks"] == [
        captured["full_text"],
        "[Structured Table Bundle]\n[Body]\n| Model | Acc |\n| --- | --- |\n| A | 90 |",
    ]
    assert captured["semantic_group_rebuild"]["pages"] == captured["pages"]
    assert captured["semantic_group_rebuild"]["embed_fn"] is fake_embed_fn
    assert captured["semantic_group_rebuild"]["api_key"] == "summary-key"
    assert captured["semantic_group_rebuild"]["model"] == "deepseek-chat"
    assert captured["semantic_group_rebuild"]["provider"] == "deepseek"
    assert captured["semantic_group_rebuild"]["endpoint"] == "https://api.deepseek.com/v1"
    assert captured["semantic_group_rebuild"]["current_index_source"] == "pdf_native"
    assert "8 S. Liu et al." not in captured["full_text"]
    assert captured["pages"][0]["page"] == 1
    assert captured["structured_table_bundles"][0]["source"] == "mineru"

    with open(vectors_dir / f"{doc_id}.pkl", "rb") as f:
        current = pickle.load(f)
    assert current["index_source"] == "mineru"
    assert current["previous_index_source"] == "pdf_native"
    assert current["chunk_metadata"][1]["structured_table_bundle"] is True
    assert current["chunk_metadata"][1]["evidence_units"]

    switched_doc = document_routes.documents_store[doc_id]
    assert switched_doc["data"]["rag_index_source"] == "mineru"
    assert switched_doc["data"]["full_text"] == captured["full_text"]
    assert switched_doc["data"]["pages"][0]["source"] == "mineru"
    assert switched_doc["data"]["structured_table_bundles"][0]["source"] == "mineru"
    saved_doc = json.loads((data_dir / "docs" / f"{doc_id}.json").read_text(encoding="utf-8"))
    assert saved_doc["data"]["rag_index_source"] == "mineru"
    groups_dir = data_dir / "semantic_groups"
    assert not (groups_dir / f"{doc_id}.json").exists()
    assert not (groups_dir / f"{doc_id}_groups.pkl").exists()
    assert (groups_dir / f"{doc_id}.pdf_native.bak.semantic.json").exists()

    rollback = document_routes._restore_vector_index_backup(doc_id, "pdf_native")
    assert rollback["index_source"] == "pdf_native"
    assert rollback["document_restore"]["restored"] is True
    assert rollback["semantic_group_restore"]["restored"] is True
    with open(vectors_dir / f"{doc_id}.pkl", "rb") as f:
        restored = pickle.load(f)
    assert restored["chunks"] == ["old native chunk"]
    assert document_routes.documents_store[doc_id]["data"]["full_text"] == "1 Introduction\nGrounding DINO text.\nTable 1 Results A 90"
    restored_paths = semantic_group_paths(groups_dir, doc_id)
    assert restored_paths["json"].exists()
    with open(restored_paths["pkl"], "rb") as f:
        restored_group_meta = pickle.load(f)
    assert restored_group_meta["group_ids"] == ["group-0"]


def test_prepare_semantic_rebuild_does_not_reuse_embedding_key_for_summary(monkeypatch, isolated_document_routes):
    _data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-semantic-no-summary"
    temp_dir = vectors_dir / "_tmp" / f"{doc_id}.mineru"
    temp_dir.mkdir(parents=True)
    identity_meta = {
        "parse_generation": "gen-semantic-1",
        "document_source_hash": "hash-semantic-1",
        "block_index_hash": "block-hash-1",
    }
    write_published_vector_index(
        temp_dir,
        doc_id,
        chunks=["mineru chunk"],
        index_source="mineru",
        index_meta=identity_meta,
    )

    fake_embed_fn = object()
    monkeypatch.setattr(
        document_routes,
        "get_embedding_function",
        lambda *_args, **_kwargs: fake_embed_fn,
    )

    prepared = document_routes._prepare_semantic_group_rebuild(
        doc_id,
        temp_dir,
        embedding_model="local-minilm",
        embedding_api_key="embed-key",
        embedding_api_host=None,
        summary_api_key=None,
        # semantic 重建现在必须显式绑定来源与解析身份，防止从错误代次的
        # staging 产物生成意群索引。
        expected_source="mineru",
        expected_parse_generation=identity_meta["parse_generation"],
        expected_document_source_hash=identity_meta["document_source_hash"],
        expected_block_index_hash=identity_meta["block_index_hash"],
    )

    assert prepared["chunks"] == ["mineru chunk"]
    assert prepared["embed_fn"] is fake_embed_fn
    assert prepared["api_key"] is None


def test_rebuild_quality_failure_keeps_old_index(monkeypatch, isolated_document_routes):
    data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-fail"
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        "data": {"full_text": "x" * 500, "total_pages": 1, "pages": [{"page": 1, "content": "old"}]},
    }
    _write_vector_pair(vectors_dir, doc_id, source="pdf_native", chunks=["old chunk"])
    _write_mineru_payload(data_dir, doc_id)

    def bad_normalize(_payload, **_kwargs):
        return {
            "full_text": "too short",
            "pages": [{"page": 1, "content": "too short"}],
            "structured_table_bundles": [],
            "quality_report": {},
            "source_hash": "bad",
            "index_source": "mineru",
            "normalizer_version": "test",
        }

    monkeypatch.setattr(document_routes, "normalize_mineru_for_rag", bad_normalize)

    with pytest.raises(RuntimeError):
        document_routes._rebuild_mineru_rag_index(
            doc_id,
            embedding_model="local-minilm",
            embedding_api_key=None,
            embedding_api_host=None,
            summary_api_key=None,
        )

    with open(vectors_dir / f"{doc_id}.pkl", "rb") as f:
        current = pickle.load(f)
    assert current["index_source"] == "pdf_native"
    assert current["chunks"] == ["old chunk"]
    assert not (vectors_dir / f"{doc_id}.pdf_native.bak.pkl").exists()


def test_semantic_rebuild_prepare_failure_keeps_old_index(monkeypatch, isolated_document_routes):
    data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-semantic-prepare-fail"
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        "data": {
            "full_text": "1 Introduction\nGrounding DINO text.\nTable 1 Results A 90",
            "total_pages": 1,
            "pages": [{"page": 1, "content": "old"}],
        },
    }
    _write_vector_pair(vectors_dir, doc_id, source="pdf_native", chunks=["old native chunk"])
    _write_mineru_payload(data_dir, doc_id)
    _write_semantic_group_artifacts(data_dir, doc_id)

    captured = {}

    monkeypatch.setattr(
        document_routes,
        "create_index",
        make_fake_create_index(captured),
    )

    def fail_get_embedding_function(*_args, **_kwargs):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(document_routes, "get_embedding_function", fail_get_embedding_function)

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        document_routes._rebuild_mineru_rag_index(
            doc_id,
            embedding_model="local-minilm",
            embedding_api_key=None,
            embedding_api_host=None,
            summary_api_key=None,
        )

    assert captured["build_semantic_groups"] is False
    with open(vectors_dir / f"{doc_id}.pkl", "rb") as f:
        current = pickle.load(f)
    assert current["index_source"] == "pdf_native"
    assert current["chunks"] == ["old native chunk"]
    assert not (vectors_dir / f"{doc_id}.pdf_native.bak.pkl").exists()
    groups_dir = data_dir / "semantic_groups"
    assert (groups_dir / f"{doc_id}.json").exists()
    assert (groups_dir / f"{doc_id}_groups.pkl").exists()
    assert document_routes.documents_store[doc_id]["data"].get("rag_index_source") != "mineru"


def test_temp_index_html_failure_keeps_old_index(monkeypatch, isolated_document_routes):
    data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-html-fail"
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        "data": {
            "full_text": "1 Introduction Grounding DINO text. Table 1 Results A 90",
            "total_pages": 1,
            "pages": [],
        },
    }
    _write_vector_pair(vectors_dir, doc_id, source="pdf_native", chunks=["old chunk"])
    _write_mineru_payload(data_dir, doc_id)

    def fake_create_index(doc_id_arg, _full_text, vector_store_dir, *_args, index_source="mineru", index_meta=None, **_kwargs):
        out_dir = Path(vector_store_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{doc_id_arg}.index").write_bytes(b"bad-index")
        with open(out_dir / f"{doc_id_arg}.pkl", "wb") as f:
            pickle.dump(
                {
                    "chunks": ["<table><tr><td>bad</td></tr></table>"],
                    "embedding_model": "local-minilm",
                    "chunk_metadata": [{}],
                    "index_source": index_source,
                    "source_hash": index_meta.get("source_hash", ""),
                    "rebuilt_at": index_meta.get("rebuilt_at", ""),
                    "previous_index_source": index_meta.get("previous_index_source", ""),
                    "normalizer_version": index_meta.get("normalizer_version", ""),
                },
                f,
            )

    monkeypatch.setattr(document_routes, "create_index", fake_create_index)

    with pytest.raises(RuntimeError, match="质量门失败"):
        document_routes._rebuild_mineru_rag_index(
            doc_id,
            embedding_model="local-minilm",
            embedding_api_key=None,
            embedding_api_host=None,
            summary_api_key=None,
        )

    with open(vectors_dir / f"{doc_id}.pkl", "rb") as f:
        current = pickle.load(f)
    assert current["index_source"] == "pdf_native"
    assert current["chunks"] == ["old chunk"]


def test_rebuild_rejects_another_inflight_document_operation(isolated_document_routes):
    doc_id = "doc-operation-lock"
    operation_lock = document_routes._get_document_operation_lock(doc_id)
    assert operation_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="正在执行"):
            document_routes._rebuild_mineru_rag_index(
                doc_id,
                embedding_model="local-minilm",
                embedding_api_key=None,
                embedding_api_host=None,
                summary_api_key=None,
            )
    finally:
        operation_lock.release()


def test_startup_recovery_restores_all_rag_artifacts_before_marking_rolled_back(
    monkeypatch,
    isolated_document_routes,
):
    data_dir, _vectors_dir = isolated_document_routes
    doc_id = "doc-recovery"
    pending = data_dir / "rag_transactions" / "pending"
    pending.mkdir(parents=True)
    journal_path = pending / f"{doc_id}.json"
    journal_path.write_text(
        json.dumps({"doc_id": doc_id, "source": "pdf_native", "state": "document_swapped"}),
        encoding="utf-8",
    )
    # 回滚快照必须能证明自己属于当前解析代次；缺身份的 legacy 快照会被判
    # superseded 而不是投机恢复。夹具直接绑定当前 manifest 的合成代次。
    current_manifest = document_routes._read_document_parse_manifest(doc_id, None)
    manifest = {
        "semantic_groups": {"backed_up": True},
        "parse_generation": str(current_manifest.get("generation") or ""),
        "document_source_hash": str(current_manifest.get("source_hash") or ""),
    }
    calls = []
    monkeypatch.setattr(document_routes, "_load_complete_rag_backup_manifest", lambda *_args: manifest)

    def fake_restore_vector_index_backup(doc_id_arg, source_arg):
        # 文档与意群的恢复已收拢进这个单一入口，恢复结果随快照一并返回，
        # recover 只在三者全部成功后才把事务标记为 rolled_back。
        calls.append((doc_id_arg, source_arg))
        return {
            "restored": True,
            "document_restore": {"restored": True},
            "semantic_group_restore": {"restored": True},
        }

    monkeypatch.setattr(
        document_routes,
        "_restore_vector_index_backup",
        fake_restore_vector_index_backup,
    )

    recovered = document_routes.recover_pending_rag_transactions()

    assert calls == [(doc_id, "pdf_native")]
    assert recovered[0]["state"] == "rolled_back"
    assert recovered[0]["document"]["restored"] is True
    assert recovered[0]["semantic_groups"]["restored"] is True
    assert json.loads(journal_path.read_text(encoding="utf-8"))["state"] == "rolled_back"
