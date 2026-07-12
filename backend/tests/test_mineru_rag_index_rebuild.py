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


def _write_vector_pair(vectors_dir: Path, doc_id: str, *, source: str = "pdf_native", chunks=None):
    (vectors_dir / f"{doc_id}.index").write_bytes(b"fake-index")
    with open(vectors_dir / f"{doc_id}.pkl", "wb") as f:
        pickle.dump(
            {
                "chunks": chunks or ["old chunk"],
                "embedding_model": "local-minilm",
                "chunk_metadata": [{}],
                "index_source": source,
            },
            f,
        )


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
    groups_dir = data_dir / "semantic_groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    (groups_dir / f"{doc_id}.json").write_text(
        json.dumps({"schema_version": 1, "doc_id": doc_id, "groups": []}),
        encoding="utf-8",
    )
    (groups_dir / f"{doc_id}_groups.index").write_bytes(b"old-group-index")
    with open(groups_dir / f"{doc_id}_groups.pkl", "wb") as f:
        pickle.dump({"digest_texts": ["old"], "group_ids": ["group-0"]}, f)


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
        "data": {"full_text": "old text", "total_pages": 8, "pages": [{"page": 1, "content": "old"}]},
    }
    _write_vector_pair(vectors_dir, doc_id, source="pdf_native", chunks=["old native chunk"])
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
    assert status["rag_index"]["index_source"] == "pdf_native"
    assert status["recommend_deep_parse"] is False
    assert status["recommend_rag_index_rebuild"] is True
    assert "问答索引仍使用本地 PDF 解析" in status["recommend_rag_index_reason"]


def test_deep_parse_status_does_not_recommend_rag_rebuild_when_index_is_mineru(isolated_document_routes):
    data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-no-recommend-rag"
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        "data": {"full_text": "mineru text", "total_pages": 8, "pages": [{"page": 1, "content": "mineru"}]},
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


def test_rebuild_mineru_rag_index_replaces_after_temp_success_and_can_rollback(monkeypatch, isolated_document_routes):
    data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-rag"
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        "data": {
            "full_text": "1 Introduction\nGrounding DINO text.\nTable 1 Results A 90",
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
    ):
        captured["full_text"] = full_text
        captured["pages"] = pages
        captured["structured_table_bundles"] = structured_table_bundles
        captured["build_semantic_groups"] = build_semantic_groups
        out_dir = Path(vector_store_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{doc_id_arg}.index").write_bytes(b"mineru-index")
        with open(out_dir / f"{doc_id_arg}.pkl", "wb") as f:
            pickle.dump(
                {
                    "chunks": [full_text, "[Structured Table Bundle]\n[Body]\n| Model | Acc |\n| --- | --- |\n| A | 90 |"],
                    "embedding_model": embedding_model,
                    "chunk_metadata": [
                        {},
                        {
                            "structured_table_bundle": True,
                            "table_body_markdown": structured_table_bundles[0]["table_body_markdown"],
                            "evidence_units": structured_table_bundles[0]["evidence_units"],
                            "source": "mineru",
                        },
                    ],
                    "index_source": index_source,
                    "source_hash": index_meta["source_hash"],
                    "rebuilt_at": index_meta["rebuilt_at"],
                    "previous_index_source": index_meta["previous_index_source"],
                    "normalizer_version": index_meta["normalizer_version"],
                },
                f,
            )

    monkeypatch.setattr(document_routes, "create_index", fake_create_index)
    fake_embed_fn = object()

    def fake_get_embedding_function(model, api_key=None, base_url=None):
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
    assert captured["embedding_request"] == {
        "model": "local-minilm",
        "api_key": "embed-key",
        "base_url": None,
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
    with open(temp_dir / f"{doc_id}.pkl", "wb") as f:
        pickle.dump(
            {
                "chunks": ["mineru chunk"],
                "embedding_model": "local-minilm",
                "chunk_metadata": [{}],
                "index_source": "mineru",
            },
            f,
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
    )

    assert prepared["chunks"] == ["mineru chunk"]
    assert prepared["embed_fn"] is fake_embed_fn
    assert prepared["api_key"] is None


def test_rebuild_quality_failure_keeps_old_index(monkeypatch, isolated_document_routes):
    data_dir, vectors_dir = isolated_document_routes
    doc_id = "doc-fail"
    document_routes.documents_store[doc_id] = {
        "filename": "paper.pdf",
        "data": {"full_text": "x" * 500, "pages": [{"page": 1, "content": "old"}]},
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
        *_args,
        pages=None,
        structured_table_bundles=None,
        index_source="pdf_native",
        index_meta=None,
        build_semantic_groups=True,
        **_kwargs,
    ):
        captured["build_semantic_groups"] = build_semantic_groups
        out_dir = Path(vector_store_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{doc_id_arg}.index").write_bytes(b"mineru-index")
        with open(out_dir / f"{doc_id_arg}.pkl", "wb") as f:
            pickle.dump(
                {
                    "chunks": [full_text],
                    "embedding_model": embedding_model,
                    "chunk_metadata": [{}],
                    "index_source": index_source,
                    "source_hash": index_meta["source_hash"],
                    "rebuilt_at": index_meta["rebuilt_at"],
                    "previous_index_source": index_meta["previous_index_source"],
                    "normalizer_version": index_meta["normalizer_version"],
                },
                f,
            )

    monkeypatch.setattr(document_routes, "create_index", fake_create_index)

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
        "data": {"full_text": "1 Introduction Grounding DINO text. Table 1 Results A 90", "pages": []},
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
    manifest = {"semantic_groups": {"backed_up": True}}
    calls = []
    monkeypatch.setattr(document_routes, "_load_complete_rag_backup_manifest", lambda *_args: manifest)
    monkeypatch.setattr(
        document_routes,
        "_restore_vector_index_backup",
        lambda *_args: calls.append("vector") or {"restored": True},
    )
    monkeypatch.setattr(
        document_routes,
        "_restore_document_backup",
        lambda *_args: calls.append("document") or {"restored": True},
    )
    monkeypatch.setattr(
        document_routes,
        "_restore_semantic_group_backup",
        lambda *_args: calls.append("semantic") or {"restored": True},
    )

    recovered = document_routes.recover_pending_rag_transactions()

    assert calls == ["vector", "document", "semantic"]
    assert recovered[0]["state"] == "rolled_back"
    assert json.loads(journal_path.read_text(encoding="utf-8"))["state"] == "rolled_back"
