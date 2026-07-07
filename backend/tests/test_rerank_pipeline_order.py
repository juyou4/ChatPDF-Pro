"""rerank 管线顺序回归测试"""

import os
import re
import pickle
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embedding_service import (
    _apply_page_provenance,
    _apply_numeric_table_same_bundle_hard_gate,
    _apply_query_intent_boost,
    _apply_numeric_table_boost,
    _augment_with_table_chunks,
    _build_fallback_citation_from_result,
    _build_retrieval_diagnostics,
    _build_multi_row_bundle_context_text,
    _build_query_focused_table_row,
    _build_context_text_for_result,
    _cleanup_numeric_table_context_entries,
    _dedupe_numeric_table_evidence_units,
    _ensure_numeric_table_evidence_slots,
    _extract_table_header_snippet,
    _extract_plain_table_rows,
    _expand_numeric_table_evidence_units,
    _finalize_with_optional_rerank,
    _finalize_without_rerank,
    _focus_mode_compress,
    _prioritize_numeric_table_results,
    _is_numeric_table_explicit_comparator_query,
    _unified_post_clean,
    search_document_chunks,
)
from services.query_rewriter import QueryRewriter


EMBED_DIM = 32


@pytest.fixture(scope="module")
def vector_store_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_id = "rerank-order-doc"

        index = faiss.IndexFlatIP(EMBED_DIM)
        vectors = np.random.randn(4, EMBED_DIM).astype("float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)
        faiss.write_index(index, os.path.join(tmpdir, f"{doc_id}.index"))

        data = {
            "chunks": [
                "这是第一段测试文本。",
                "这是第二段测试文本。",
                "这是第三段测试文本。",
                "这是第四段测试文本。",
            ],
            "chunk_headings": ["摘要", "方法", "实验结果", "References"],
            "chunk_pages": [1, 2, 3, 4],
            "chunk_types": ["text", "text", "table", "text"],
            "embedding_model": "local-minilm",
            "parent_chunks": [],
            "child_to_parent": {},
        }
        with open(os.path.join(tmpdir, f"{doc_id}.pkl"), "wb") as f:
            pickle.dump(data, f)

        yield tmpdir


def _make_mock_embed_fn():
    def embed_fn(texts):
        vecs = np.random.randn(len(texts), EMBED_DIM).astype("float32")
        faiss.normalize_L2(vecs)
        return vecs

    return embed_fn


def _make_result(chunk: str, similarity: float) -> dict:
    return {
        "chunk": chunk,
        "page": 1,
        "score": similarity,
        "similarity": similarity,
        "similarity_percent": round(similarity * 100, 2),
        "snippet": chunk[:80],
        "highlights": [],
        "reranked": False,
    }


def _make_numeric_candidate(
    chunk: str,
    similarity: float,
    *,
    page: int = 1,
    chunk_type: str = "text",
    block_type: str | None = None,
    **extra,
) -> dict:
    candidate = _make_result(chunk, similarity)
    candidate.update(
        {
            "page": page,
            "chunk_type": chunk_type,
            "block_type": block_type or chunk_type,
        }
    )
    candidate.update(extra)
    return candidate


@pytest.mark.parametrize("use_hybrid", [False, True])
def test_rerank_runs_after_candidate_augmentation(vector_store_dir, use_hybrid):
    """最终 rerank 应看到意群融合和表格补充后的候选集。"""
    order = []

    def fake_group_search(**kwargs):
        order.append("group")
        return kwargs["chunk_results"] + [_make_result("group extra chunk", 0.74)]

    def fake_table(results, *_args, **_kwargs):
        order.append("table")
        return results + [_make_result("table extra chunk", 0.73)]

    def fake_clean(results, _query, _top_k):
        order.append("clean")
        return results

    def fake_rerank(_query, candidates, *_args, **_kwargs):
        order.append("rerank")
        chunks = {item["chunk"] for item in candidates}
        assert "group extra chunk" in chunks
        assert "table extra chunk" in chunks
        return candidates

    with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
         patch("services.embedding_service._merge_with_group_search", side_effect=fake_group_search), \
         patch("services.embedding_service._augment_with_table_chunks", side_effect=fake_table), \
         patch("services.embedding_service._unified_post_clean", side_effect=fake_clean), \
         patch("services.embedding_service._apply_rerank", side_effect=fake_rerank), \
         patch("services.embedding_service._query_vector_cache") as mock_cache, \
         patch("services.chunk_expander.expand_context_chunks", side_effect=lambda results, *_args, **_kwargs: results):

        mock_cache.get.return_value = None

        if use_hybrid:
            with patch("services.bm25_service.bm25_search", return_value=[]), \
                 patch("services.hybrid_search.hybrid_search_merge", side_effect=lambda vec, bm25, **kw: vec[:kw.get("top_k", 10)]):
                _results, timings = search_document_chunks(
                    doc_id="rerank-order-doc",
                    query="测试查询",
                    vector_store_dir=vector_store_dir,
                    pages=[{"page": 1, "text": "页面文本"}],
                    top_k=3,
                    use_hybrid=True,
                    use_rerank=True,
                )
        else:
            _results, timings = search_document_chunks(
                doc_id="rerank-order-doc",
                query="测试查询",
                vector_store_dir=vector_store_dir,
                pages=[{"page": 1, "text": "页面文本"}],
                top_k=3,
                use_hybrid=False,
                use_rerank=True,
            )

    assert order == ["group", "table", "clean", "rerank"]
    assert "rerank_ms" in timings


def test_numeric_table_evidence_need_can_trigger_conditional_rerank(vector_store_dir):
    called = {"rerank": False}

    def fake_rerank(_query, candidates, *_args, **_kwargs):
        called["rerank"] = True
        assert all("block_type" in item for item in candidates)
        assert all("section_path" in item for item in candidates)
        return candidates

    with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
         patch("services.embedding_service._apply_rerank", side_effect=fake_rerank), \
         patch("services.embedding_service._query_vector_cache") as mock_cache, \
         patch("services.embedding_service._merge_with_group_search", side_effect=lambda **kwargs: kwargs["chunk_results"]), \
         patch("services.embedding_service._augment_with_table_chunks", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.embedding_service._unified_post_clean", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.chunk_expander.expand_context_chunks", side_effect=lambda results, *_args, **_kwargs: results):

        mock_cache.get.return_value = None
        with patch.object(
            search_document_chunks.__globals__["_rag_config_singleton"],
            "enable_conditional_rerank",
            True,
        ), patch.object(
            search_document_chunks.__globals__["_rag_config_singleton"],
            "conditional_rerank_evidence_needs",
            "numeric_table,section_explanation",
            create=True,
        ):
            results, _timings = search_document_chunks(
                doc_id="rerank-order-doc",
                query="Many/Medium/Few 上分别提升多少？",
                vector_store_dir=vector_store_dir,
                pages=[{"page": 1, "text": "页面文本"}],
                top_k=3,
                use_hybrid=False,
                use_rerank=False,
                reranker_model="test-reranker",
            )

    assert called["rerank"] is True
    assert results
    assert all("block_type" in item for item in results)
    assert all("section_path" in item for item in results)


def test_numeric_table_conditional_rerank_uses_default_model_when_unspecified(vector_store_dir):
    called = {"rerank": False}

    def fake_rerank(_query, candidates, reranker_model, rerank_provider, *_args, **_kwargs):
        called["rerank"] = True
        assert reranker_model is None
        assert rerank_provider is None
        return candidates

    with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
         patch("services.embedding_service._apply_rerank", side_effect=fake_rerank), \
         patch("services.embedding_service._query_vector_cache") as mock_cache, \
         patch("services.embedding_service._merge_with_group_search", side_effect=lambda **kwargs: kwargs["chunk_results"]), \
         patch("services.embedding_service._augment_with_table_chunks", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.embedding_service._unified_post_clean", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.chunk_expander.expand_context_chunks", side_effect=lambda results, *_args, **_kwargs: results):

        mock_cache.get.return_value = None
        with patch.object(
            search_document_chunks.__globals__["_rag_config_singleton"],
            "enable_conditional_rerank",
            True,
        ), patch.object(
            search_document_chunks.__globals__["_rag_config_singleton"],
            "conditional_rerank_evidence_needs",
            "numeric_table,section_explanation",
            create=True,
        ):
            results, _timings = search_document_chunks(
                doc_id="rerank-order-doc",
                query="Many/Medium/Few 上分别提升多少？",
                vector_store_dir=vector_store_dir,
                pages=[{"page": 1, "text": "页面文本"}],
                top_k=3,
                use_hybrid=False,
                use_rerank=False,
            )

    assert called["rerank"] is True
    assert results


def test_rerank_floor_filters_low_score_candidates():
    candidates = [
        {
            "chunk": "高相关证据",
            "similarity": 0.91,
            "rerank_score": 0.92,
            "combined_score": 0.91,
        },
        {
            "chunk": "低相关噪声",
            "similarity": 0.07,
            "rerank_score": 0.04,
            "combined_score": 0.04,
        },
    ]

    with patch("services.embedding_service._apply_rerank", return_value=candidates), \
         patch.object(
            _finalize_with_optional_rerank.__globals__["_rag_config_singleton"],
            "rerank_score_min",
            0.08,
            create=True,
         ), \
         patch.object(
            _finalize_with_optional_rerank.__globals__["_rag_config_singleton"],
            "rerank_score_min_keep",
            1,
            create=True,
         ):
        final = _finalize_with_optional_rerank(
            query="测试问题",
            results=candidates,
            top_k=2,
            use_rerank=True,
            reranker_model="test-reranker",
            rerank_provider="local",
            rerank_api_key=None,
            rerank_endpoint=None,
            timings={},
        )

    assert [item["chunk"] for item in final] == ["高相关证据"]


def test_numeric_table_priority_runs_before_topk_when_rerank_disabled():
    candidates = [
        {
            "chunk": "DiffuLT achieves state-of-the-art results on CIFAR10-LT.",
            "raw_chunk_text": "DiffuLT achieves state-of-the-art results on CIFAR10-LT.",
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.81,
            "similarity_percent": 81.0,
        },
        {
            "chunk": "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "similarity": 0.39,
            "similarity_percent": 39.0,
        },
    ]

    with patch.object(
        _finalize_with_optional_rerank.__globals__["_rag_config_singleton"],
        "enable_focus_mode",
        False,
        create=True,
    ):
        final = _finalize_with_optional_rerank(
            query="表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？",
            results=candidates,
            top_k=1,
            use_rerank=False,
            reranker_model=None,
            rerank_provider=None,
            rerank_api_key=None,
            rerank_endpoint=None,
            timings={},
        )

    assert len(final) == 1
    assert final[0]["chunk_type"] == "table_row"
    assert "ResNet-50" in final[0]["chunk"]


def test_numeric_table_priority_prefers_table_row_with_caption_header_over_narrative_chunk():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    candidates = [
        {
            "chunk": (
                "DiffuLT achieves strong ImageNet-LT performance with ResNet-50 and improves the All metric "
                "over prior work, while Table 8 summarizes the comparison against cRT, RIDE(3 experts), and ADRW."
            ),
            "raw_chunk_text": (
                "DiffuLT achieves strong ImageNet-LT performance with ResNet-50 and improves the All metric "
                "over prior work, while Table 8 summarizes the comparison against cRT, RIDE(3 experts), and ADRW."
            ),
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.83,
            "similarity_percent": 83.0,
        },
        {
            "chunk": "DiffuLT | All=56.4 | cRT=47.3 | RIDE(3 experts)=54.9 | ADRW=54.1",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "DiffuLT",
            "similarity": 0.41,
            "similarity_percent": 41.0,
        },
    ]

    with patch.object(
        _finalize_with_optional_rerank.__globals__["_rag_config_singleton"],
        "enable_focus_mode",
        False,
        create=True,
    ):
        final = _finalize_with_optional_rerank(
            query=query,
            results=candidates,
            top_k=1,
            use_rerank=False,
            reranker_model=None,
            rerank_provider=None,
            rerank_api_key=None,
            rerank_endpoint=None,
            timings={},
        )

    assert len(final) == 1
    assert final[0]["chunk_type"] == "table_row"
    assert final[0]["row_id"] == "DiffuLT"


@pytest.mark.parametrize(
    ("query", "target_table", "target_row", "wrong_table", "wrong_row"),
    [
        (
            "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？",
            "Table 1",
            "CBDM",
            "Table 3",
            "AID",
        ),
        (
            "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？",
            "Table 3",
            "AID",
            "Table 1",
            "CBDM",
        ),
    ],
)
def test_numeric_table_priority_hard_gates_explicit_table_number(
    query,
    target_table,
    target_row,
    wrong_table,
    wrong_row,
):
    candidates = [
        {
            "chunk": f"{wrong_row} | ResNet-50 | metric=0.99",
            "raw_chunk_text": f"{wrong_row} 0.99 0.88",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": f"{wrong_table}: distractor",
            "table_header": "Model Metric Accuracy",
            "table_id": wrong_table,
            "row_id": wrong_row,
            "similarity": 0.95,
            "similarity_percent": 95.0,
        },
        {
            "chunk": f"{target_row} | ResNet-50 | metric=0.72",
            "raw_chunk_text": f"{target_row} 0.72 0.88",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": f"{target_table}: target",
            "table_header": "Model Metric Accuracy",
            "table_id": target_table,
            "row_id": target_row,
            "similarity": 0.74,
            "similarity_percent": 74.0,
        },
    ]

    final = _prioritize_numeric_table_results(candidates, query)

    assert final[0]["table_id"] == target_table
    assert final[0]["row_id"] == target_row


def test_numeric_table_priority_demotes_row_without_explicit_target_table_match():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    candidates = [
        {
            "chunk": "final line of table1. ID | FID=2.75×10−4 | Acc=40",
            "raw_chunk_text": "final line of table1. ID 2.75×10−4 40",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "ID",
            "similarity": 0.97,
            "similarity_percent": 97.0,
            "numeric_table_priority": 14.5,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
        },
        {
            "chunk": "CBDM (τ=1) | FID=5.86 | Acc=46.6",
            "raw_chunk_text": "CBDM (τ=1) 5.86 46.6",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 1: FID of different generation models and their classifiers' accuracy.",
            "table_header": "Model FID Acc. (%)",
            "table_id": "Table 1",
            "row_id": "CBDM (τ=1)",
            "similarity": 0.55,
            "similarity_percent": 55.0,
            "numeric_table_priority": 8.0,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
        },
    ]

    final = _prioritize_numeric_table_results(candidates, query)

    assert final[0]["table_id"] == "Table 1"
    assert final[0]["row_id"] == "CBDM (τ=1)"


def test_numeric_table_boost_demotes_inline_table_mentions_for_strict_metric_queries():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    candidates = [
        {
            "chunk": "final line of table1. ID | FID=2.75×10−4 | Acc=40",
            "raw_chunk_text": "final line of table1. ID 2.75×10−4 40",
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.97,
            "similarity_percent": 97.0,
        },
        {
            "chunk": "Table 1: FID of different generation models and their classifiers' accuracy. CBDM (τ=1) 5.86 46.6",
            "raw_chunk_text": "Table 1: FID of different generation models and their classifiers' accuracy. CBDM (τ=1) 5.86 46.6",
            "chunk_type": "table",
            "block_type": "table",
            "table_caption": "Table 1: FID of different generation models and their classifiers' accuracy.",
            "table_id": "Table 1",
            "similarity": 0.52,
            "similarity_percent": 52.0,
        },
    ]

    adjusted = _apply_numeric_table_boost(candidates, query)

    assert adjusted[0].get("table_id") == "Table 1"


def test_numeric_table_slot_reservation_ignores_inline_table_mentions_for_strict_metric_queries():
    query = "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？"
    results = [
        {
            "chunk": "final line of table3. AID | Acc=45.2 | ΔAcc/||D_gen||=5.78×10^-4",
            "raw_chunk_text": "final line of table3. AID 45.2 5.78×10^-4",
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.96,
            "similarity_percent": 96.0,
            "numeric_table_priority": 13.8,
            "numeric_table_anchor_hits": ["Table 3", "AID", "Acc"],
        },
        {
            "chunk": "Table 3: Quantities and classifier enhancement.",
            "raw_chunk_text": "Table 3: Quantities and classifier enhancement.",
            "chunk_type": "caption",
            "block_type": "caption",
            "table_caption": "Table 3: Quantities and classifier enhancement.",
            "table_id": "Table 3",
            "similarity": 0.51,
            "similarity_percent": 51.0,
            "numeric_table_priority": 8.2,
            "numeric_table_anchor_hits": ["Table 3"],
        },
        {
            "chunk": "AID | Acc=45.2 | ΔAcc/||D_gen||=5.78×10^-4",
            "raw_chunk_text": "AID 45.2 5.78×10^-4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 3: Quantities and classifier enhancement.",
            "table_header": "Type Acc ΔAcc/||D_gen||",
            "table_id": "Table 3",
            "row_id": "AID",
            "similarity": 0.34,
            "similarity_percent": 34.0,
            "numeric_table_priority": 7.4,
            "numeric_table_anchor_hits": ["Table 3", "AID", "Acc"],
        },
    ]

    final = _ensure_numeric_table_evidence_slots(results, query, top_k=2)

    assert any(
        item.get("chunk_type") == "table_row" and item.get("table_id") == "Table 3"
        for item in final
    )
    assert all("final line of table3" not in (item.get("chunk") or "").lower() for item in final)


def test_numeric_table_expansion_creates_table_row_evidence():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    results = [
        {
            "chunk": (
                "ResNet-10 ResNet-50 All All Many Med. Few CE 34.8 41.6 64.0 33.8 5.8 "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
                "DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9"
            ),
            "raw_chunk_text": (
                "Table 8: Results on ImageNet-LT. We deploy ResNet-10 and ResNet-50 as classifier backbones. "
                "ResNet-10 ResNet-50 All All Many Med. Few CE 34.8 41.6 64.0 33.8 5.8 "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
                "DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9"
            ),
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "section_path": "Experiments",
            "chunk_heading": "Table 8",
            "similarity": 0.42,
            "similarity_percent": 42.0,
            "group_id": "group-8",
        }
    ]

    expanded = _expand_numeric_table_evidence_units(
        results,
        query,
        include_rerank_text=True,
        doc_title="DiffuLT",
    )

    row_items = [item for item in expanded if item.get("chunk_type") == "table_row"]
    assert row_items
    assert any(item.get("row_id") == "DiffuLT" for item in row_items)
    assert any(item.get("table_id") == "Table 8" for item in row_items)
    diffult_row = next(item for item in row_items if item.get("row_id") == "DiffuLT")
    assert "ResNet-50" in diffult_row.get("chunk", "")
    assert "All=56.4" in diffult_row.get("chunk", "")
    assert "Many=63.3" in diffult_row.get("chunk", "")
    assert "Med.=55.6" in diffult_row.get("chunk", "")
    assert "Few=39.4" in diffult_row.get("chunk", "")
    assert all(item.get("table_caption") for item in row_items)
    assert all(item.get("rerank_text") for item in row_items)


def test_numeric_table_expansion_filters_weak_rows():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    results = [
        {
            "chunk": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
                "SAM Rangwani et al. [2022] 53.1 62.0 52.1 34.8"
            ),
            "raw_chunk_text": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
                "SAM Rangwani et al. [2022] 53.1 62.0 52.1 34.8"
            ),
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "section_path": "Experiments",
            "chunk_heading": "Table 8",
            "similarity": 0.42,
            "similarity_percent": 42.0,
            "group_id": "group-8",
        }
    ]

    expanded = _expand_numeric_table_evidence_units(
        results,
        query,
        include_rerank_text=False,
        doc_title="DiffuLT",
    )

    row_items = [item for item in expanded if item.get("chunk_type") == "table_row"]
    row_ids = {item.get("row_id") for item in row_items}
    normalized_row_ids = {str(item.get("row_id") or "").lower().replace(" ", "") for item in row_items}
    assert "DiffuLT" in row_ids
    assert "cRT" in row_ids
    assert "ADRW" in row_ids
    assert any(item in row_ids for item in {"RIDE (3 experts)", "RIDE(3 experts)"})
    assert "CE" not in row_ids
    assert all("diffult+ride" not in row_id for row_id in normalized_row_ids)
    assert all(not row_id.startswith("diffult(2)") for row_id in normalized_row_ids)
    assert all(not row_id.startswith("diffult(3)") for row_id in normalized_row_ids)
    assert "SAM" not in row_ids
    assert len(row_items) <= 4
    rows_by_id = {item.get("row_id"): item for item in row_items}
    assert "ResNet-50" in rows_by_id["DiffuLT"]["chunk"]
    assert "All=56.4" in rows_by_id["DiffuLT"]["chunk"]
    assert "All=47.3" in rows_by_id["cRT"]["chunk"]
    assert "All=54.1" in rows_by_id["ADRW"]["chunk"]
    ride_row = next(
        item for item in row_items if item.get("row_id") in {"RIDE (3 experts)", "RIDE(3 experts)"}
    )
    assert "All=54.9" in ride_row["chunk"]


def test_plain_table_row_anchor_ignores_expert_count_digits():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    text = (
        "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
        "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
        "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
        "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
        "DiffuLT 50.4 56.4 63.3 55.6 39.4"
    )
    hints = QueryRewriter().extract_numeric_table_hints(query)

    rows = _extract_plain_table_rows(text, hints)
    ride_row = next(item for item in rows if item["row_id"] == "RIDE(3 experts)")
    focused = _build_query_focused_table_row(ride_row, hints)

    assert ride_row["row_numbers"] == "45.9 54.9 66.2 51.7 34.9"
    assert focused["text"] == "RIDE(3 experts) | ResNet-50 | All=54.9"


def test_plain_table_row_anchor_recovers_table1_fid_and_acc_rows():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    text = (
        "Table 1: FID of different generation models and their corresponding classifiers' accuracy. "
        "Model FID Acc. (%) Baseline - 38.3 DDPM 39.1 21.2 CBDM (τ = 3) 7.42 44.8 "
        "CBDM (τ = 2) 6.82 46.0 CBDM (τ = 1) 5.86 46.6"
    )
    hints = QueryRewriter().extract_numeric_table_hints(query)

    assert "FID" in hints["columns"]
    assert "Acc" in hints["columns"]
    assert "FID" not in hints["methods"]

    rows = _extract_plain_table_rows(text, hints)
    cbdm_row = next(
        item for item in rows
        if "5.86" in item["row_numbers"] and "46.6" in item["row_numbers"]
    )
    focused = _build_query_focused_table_row(cbdm_row, hints)

    assert "FID=5.86" in focused["text"]
    assert "Acc=46.6" in focused["text"]


def test_plain_table_row_anchor_recovers_compact_table1_rows_from_side_by_side_caption_chunk():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    text = (
        "Table1: FIDof different generationmodels Table2: Percentageofdifferenttypesofgener- "
        "andtheircorrespondingclassifiers'accuracy. atedsamplesforeachmodel. "
        "Model FID Acc.(%) Model p p p ID AID OOD Baseline - 38.3 "
        "DDPM 39.1 21.2 39.7 DDPM 7.76 43.8 CBDM(τ =3) 38.6 29.1 32.3 "
        "CBDM(τ =3) 7.42 44.8 CBDM(τ =2) 6.82 46.0 CBDM(τ =2) 40.2 33.5 26.3 "
        "CBDM(τ =1) 5.86 46.6 CBDM(τ =1) 44.8 36.3 18.9"
    )
    hints = QueryRewriter().extract_numeric_table_hints(query)

    rows = _extract_plain_table_rows(text, hints)
    focused_rows = {
        row["row_id"]: _build_query_focused_table_row(row, hints)["text"]
        for row in rows
        if _build_query_focused_table_row(row, hints)["text"]
    }

    assert focused_rows["CBDM(τ =1)"] == "CBDM(τ =1) | FID=5.86 | Acc=46.6"
    assert all("40.2" not in value for value in focused_rows.values())


def test_numeric_table_expansion_does_not_promote_explicit_table1_rows_without_header_support():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    polluted_chunk = (
        "classifier achieves an accuracy of 46.6%, marking an 8.3% increase over the baseline, "
        "as noted in the final line of table1. ID 2.75 × 10^-4 40 AID 5.78 × 10^-4 80"
    )
    results = [
        {
            "chunk": polluted_chunk,
            "raw_chunk_text": polluted_chunk,
            "chunk_type": "table",
            "block_type": "table",
            "page": 5,
            "similarity": 0.91,
            "similarity_percent": 91.0,
        }
    ]

    expanded = _expand_numeric_table_evidence_units(results, query)

    assert not any(item.get("chunk_type") == "table_row" for item in expanded)


def test_plain_table_row_anchor_skips_inline_table_reference_with_multiline_numeric_tail():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    text = (
        "classifier achieves an accuracy of 46.6%, marking an 8.3% increase over the baseline, "
        "as noted in the final line of table1.\n"
        "ID 21,511 44.2 2.75×10−4\n"
        "40 33.2 29.7\n"
        "AID 11,886 45.2 5.78×10−4 80 35.7 32.5\n"
        "OOD 5,756 36.2 −3.61×10−4 100 39.1 32.8"
    )
    hints = QueryRewriter().extract_numeric_table_hints(query)

    rows = _extract_plain_table_rows(text, hints)

    assert rows == []


def test_plain_table_row_anchor_recovers_table3_acc_and_delta_rows():
    query = "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？"
    text = (
        "[TABLE] Table 3: Quantities, overall classifier enhancement, and corresponding results for tail classes. "
        "| Group | ||Dgen|| | Acc. (%) | ΔAcc/||Dgen|| |\n"
        "| --- | --- | --- | --- |\n"
        "| ID | 21,511 | 44.2 | 2.75 × 10^-4 |\n"
        "| AID | 11,886 | 45.2 | 5.78 × 10^-4 |\n"
        "| OOD | 5,756 | 36.2 | -3.61 × 10^-4 |"
    )
    hints = QueryRewriter().extract_numeric_table_hints(query)

    assert "Acc" in hints["columns"]
    assert "D_gen" not in hints["methods"]
    assert any(column in hints["columns"] for column in {"||D_gen||", "ΔAcc/||D_gen||"})

    rows = _extract_plain_table_rows(text, hints)
    aid_row = next(item for item in rows if item["row_id"] == "AID")
    focused = _build_query_focused_table_row(aid_row, hints)

    assert "11,886" in aid_row["row_numbers"]
    assert "5.78" in aid_row["row_numbers"]
    assert "AID" in focused["text"]
    assert "Acc=45.2" in focused["text"]
    assert "ΔAcc/||D_gen||" in focused["text"]


def test_plain_table_row_anchor_recovers_compact_table3_rows_from_side_by_side_caption_chunk():
    query = "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？"
    text = (
        "Table3:Quantities,overallclassifierenhancement,and Table4: Diffusiontrainedwithvarying "
        "averageimprovementpersamplefordifferentgroups proportionsofheadclassdataandthe "
        "ofdatageneratedbydiffusionmodel. correspondingresultsfortailclasses. "
        "Group ∥D gen∥ Acc.(%) ∆Acc/∥D gen∥ p h p AID Acc t(%) "
        "Baseline - 38.3 - - 25.0 ID 21,511 44.2 2.75×10−4 0 25.8 26.0 "
        "AID 11,886 45.2 5.78×10−4 40 33.2 29.7 OOD 5,756 36.2 −3.61×10−4 80 35.7 32.5"
    )
    hints = QueryRewriter().extract_numeric_table_hints(query)

    rows = _extract_plain_table_rows(text, hints)
    focused_rows = {
        row["row_id"]: _build_query_focused_table_row(row, hints)["text"]
        for row in rows
        if _build_query_focused_table_row(row, hints)["text"]
    }

    assert focused_rows["AID"] == "AID | ||D_gen||=11,886 | Acc=45.2 | ΔAcc/||D_gen||=5.78×10−4"
    assert "40" not in focused_rows
    assert "80" not in focused_rows


def test_numeric_table_expansion_scopes_multi_table_chunk_to_requested_table():
    query = "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？"
    mixed_chunk = (
        "Table 1: FID of different generation models and their corresponding classifiers' accuracy. "
        "Model FID Acc. (%) CBDM (τ = 1) 5.86 46.6 "
        "Table 3: Quantities, overall classifier enhancement, and corresponding results for tail classes.\n"
        "| Group | ||Dgen|| | Acc. (%) | ΔAcc/||Dgen|| |\n"
        "| --- | --- | --- | --- |\n"
        "| ID | 21,511 | 44.2 | 2.75 × 10^-4 |\n"
        "| AID | 11,886 | 45.2 | 5.78 × 10^-4 |\n"
        "| OOD | 5,756 | 36.2 | -3.61 × 10^-4 |"
    )
    results = [
        {
            "chunk": mixed_chunk,
            "raw_chunk_text": mixed_chunk,
            "chunk_type": "table",
            "block_type": "table",
            "page": 5,
            "similarity": 0.86,
            "similarity_percent": 86.0,
        }
    ]

    expanded = _expand_numeric_table_evidence_units(results, query)
    row_items = [item for item in expanded if item.get("chunk_type") == "table_row"]

    assert row_items
    assert any(item.get("row_id") == "AID" for item in row_items)
    assert all(item.get("table_id") == "Table 3" for item in row_items if item.get("table_id"))
    assert all(item.get("row_id") != "CBDM (τ = 1)" for item in row_items)


def test_table_header_snippet_recovers_flat_backbone_header():
    text = (
        "Table 8: Results on ImageNet-LT. We deploy ResNet-10 and ResNet-50 as classifier "
        "backbones. Top-performing results are highlighted in bold, with second-best outcomes "
        "underlined. ResNet-10 ResNet-50 All All Many Med. Few CE 34.8 41.6 64.0 33.8 5.8"
    )

    header = _extract_table_header_snippet(text)

    assert header == "ResNet-10 ResNet-50 All All Many Med. Few"


def test_query_focused_table_row_can_recover_resnet50_tail_values_without_header():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    hints = QueryRewriter().extract_numeric_table_hints(query)
    unit = {
        "row_id": "DiffuLT",
        "row_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
        "row_numbers": "50.4 56.4 63.3 55.6 39.4",
        "table_caption": "Table 8: Results on ImageNet-LT.",
        "table_id": "Table 8",
        "table_header": "",
    }

    focused = _build_query_focused_table_row(unit, hints)

    assert focused["matched_backbone"] == "ResNet-50"
    assert focused["text"] == "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4"


def test_numeric_table_expansion_does_not_use_caption_only_method_hits():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    results = [
        {
            "chunk": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4"
            ),
            "raw_chunk_text": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4"
            ),
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "section_path": "Experiments",
            "chunk_heading": "Table 8",
            "similarity": 0.42,
            "similarity_percent": 42.0,
            "group_id": "group-8",
        },
        {
            "chunk": (
                "[TABLE] DiffuLT supplementary overview\n\n"
                "| Method | ResNet-50 | All | Many | Med. | Few |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
                "| our methods. CIFAR100-LT CIFAR10-LTMethod | 10 | 100 | 50 | 10 | 5 |"
            ),
            "raw_chunk_text": (
                "[TABLE] DiffuLT supplementary overview\n\n"
                "| Method | ResNet-50 | All | Many | Med. | Few |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
                "| our methods. CIFAR100-LT CIFAR10-LTMethod | 10 | 100 | 50 | 10 | 5 |"
            ),
            "page": 14,
            "chunk_type": "table",
            "block_type": "table",
            "section_path": "Appendix",
            "chunk_heading": "Supplementary",
            "similarity": 0.41,
            "similarity_percent": 41.0,
            "group_id": "group-14",
        },
    ]

    expanded = _expand_numeric_table_evidence_units(
        results,
        query,
        include_rerank_text=False,
        doc_title="DiffuLT",
    )

    row_items = [item for item in expanded if item.get("chunk_type") == "table_row"]
    assert row_items
    assert {item.get("row_id") for item in row_items} == {"DiffuLT"}
    assert all("our methods." not in item.get("chunk", "") for item in row_items)


def test_plain_table_row_anchor_keeps_hyphen_placeholder_in_same_row():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    text = (
        "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
        "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
        "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
        "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
        "DiffuLT 50.4 56.4 63.3 55.6 39.4"
    )
    hints = QueryRewriter().extract_numeric_table_hints(query)

    rows = _extract_plain_table_rows(text, hints)
    adrw_row = next(item for item in rows if item["row_id"] == "ADRW")
    focused = _build_query_focused_table_row(adrw_row, hints)

    assert adrw_row["row_numbers"] == "- 54.1 62.9 52.6 37.1"
    assert focused["text"] == "ADRW | ResNet-50 | All=54.1"


def test_plain_table_row_anchor_supports_compact_author_suffix_for_table8_comparators():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    text = (
        "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
        "cRTKangetal.[2019] 41.8 47.3 58.8 44.0 26.1 "
        "RIDE(3experts)Wangetal.[2020] 45.9 54.9 66.2 51.7 34.9 "
        "ADRWWangetal.[2024b] - 54.1 62.9 52.6 37.1 "
        "DiffuLT 50.4 56.4 63.3 55.6 39.4"
    )
    hints = QueryRewriter().extract_numeric_table_hints(query)

    rows = _extract_plain_table_rows(text, hints)
    row_ids = {row["row_id"] for row in rows}

    assert {"cRT", "RIDE(3 experts)", "ADRW", "DiffuLT"} <= row_ids


def test_plain_table_row_anchor_skips_appendix_header_like_rows():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    text = (
        "t-LT to test the robustness of methods. ResNet-10 ResNet-50 "
        "DiffuLT(1) 50.4 56.4 DiffuLT(2) 50.4 56.5 DiffuLT(3) 50.5 56.5 "
        "our methods. CIFAR100-LT CIFAR10-LTMethod 100 50 10 100 50 10"
    )
    hints = QueryRewriter().extract_numeric_table_hints(query)

    rows = _extract_plain_table_rows(text, hints)

    assert not any("our methods." in item["row_id"].lower() for item in rows)
    assert not any("cifar100-lt" in item["row_id"].lower() for item in rows)


def test_numeric_table_expansion_keeps_competitor_rows_for_second_best_queries():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    results = [
        {
            "chunk": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "CE 34.8 41.6 64.0 33.8 5.8 "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
                "DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9"
            ),
            "raw_chunk_text": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "CE 34.8 41.6 64.0 33.8 5.8 "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
                "DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9"
            ),
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "section_path": "Experiments",
            "chunk_heading": "Table 8",
            "similarity": 0.42,
            "similarity_percent": 42.0,
            "group_id": "group-8",
        }
    ]

    expanded = _expand_numeric_table_evidence_units(
        results,
        query,
        include_rerank_text=False,
        doc_title="DiffuLT",
    )

    row_items = [item for item in expanded if item.get("chunk_type") == "table_row"]
    row_ids = {item.get("row_id") for item in row_items}
    assert "DiffuLT" in row_ids
    assert len(row_items) >= 2
    assert any(item in row_ids for item in {"cRT", "ADRW", "RIDE (3 experts)", "RIDE(3 experts)"})


def test_numeric_table_expansion_prefers_structured_bundle_body_rows_for_second_best_queries():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    structured_chunk = (
        "[Structured Table Bundle]\n\n"
        "Table 8: Results on ImageNet-LT. We deploy ResNet-10 and ResNet-50 as classifier backbones.\n\n"
        "[Hints]\n"
        "table_id=Table 8; page=9; bundle_id=manual:table8\n\n"
        "[Header]\n"
        "Method | ResNet-10 All | ResNet-50 All | Many | Med. | Few\n\n"
        "[Body]\n"
        "Table 8: Results on ImageNet-LT. We deploy ResNet-10 and ResNet-50 as classifier backbones.\n"
        "Top-performingresultsarehighlightedinbold,withsecond-bestoutcomesunderlined.\n"
        "ResNet-10 ResNet-50\n"
        "All All Many Med. Few\n"
        "CE 34.8 41.6 64.0 33.8 5.8\n"
        "cRTKangetal.[2019] 41.8 47.3 58.8 44.0 26.1\n"
        "RIDE(3experts)Wangetal.[2020] 45.9 54.9 66.2 51.7 34.9\n"
        "ADRWWangetal.[2024b] - 54.1 62.9 52.6 37.1\n"
        "DiffuLT 50.4 56.4 63.3 55.6 39.4\n"
        "DiffuLT+RIDE(3experts) 51.1 56.9 64.1 55.8 39.9\n"
    )
    results = [
        {
            "chunk": structured_chunk,
            "raw_chunk_text": structured_chunk,
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "section_path": "Experiments",
            "chunk_heading": "Table 8",
            "similarity": 0.42,
            "similarity_percent": 42.0,
            "table_id": "Table 8",
            "table_caption": "Table 8: Results on ImageNet-LT. We deploy ResNet-10 and ResNet-50 as classifier backbones.",
            "table_header": "Method | ResNet-10 All | ResNet-50 All | Many | Med. | Few",
            "table_body_markdown": (
                "Table 8: Results on ImageNet-LT. We deploy ResNet-10 and ResNet-50 as classifier backbones.\n"
                "Top-performingresultsarehighlightedinbold,withsecond-bestoutcomesunderlined.\n"
                "ResNet-10 ResNet-50\n"
                "All All Many Med. Few\n"
                "CE 34.8 41.6 64.0 33.8 5.8\n"
                "cRTKangetal.[2019] 41.8 47.3 58.8 44.0 26.1\n"
                "RIDE(3experts)Wangetal.[2020] 45.9 54.9 66.2 51.7 34.9\n"
                "ADRWWangetal.[2024b] - 54.1 62.9 52.6 37.1\n"
                "DiffuLT 50.4 56.4 63.3 55.6 39.4\n"
                "DiffuLT+RIDE(3experts) 51.1 56.9 64.1 55.8 39.9"
            ),
            "structured_table_bundle": True,
        }
    ]

    expanded = _expand_numeric_table_evidence_units(
        results,
        query,
        include_rerank_text=False,
        doc_title="DiffuLT",
    )

    row_items = [item for item in expanded if item.get("chunk_type") == "table_row"]

    assert any(item.get("row_id") == "DiffuLT" for item in row_items)
    assert any("58.8" in (item.get("chunk") or "") for item in row_items)
    assert any("66.2" in (item.get("chunk") or "") for item in row_items)
    assert len(row_items) >= 4


def test_numeric_table_expansion_named_comparison_skips_composite_and_unrequested_rows():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    results = [
        {
            "chunk": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "CE 34.8 41.6 64.0 33.8 5.8 "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4"
            ),
            "raw_chunk_text": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "CE 34.8 41.6 64.0 33.8 5.8 "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4"
            ),
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "section_path": "Experiments",
            "chunk_heading": "Table 8",
            "similarity": 0.43,
            "similarity_percent": 43.0,
            "group_id": "group-8",
        }
    ]

    expanded = _expand_numeric_table_evidence_units(
        results,
        query,
        include_rerank_text=False,
        doc_title="DiffuLT",
    )

    row_items = [item for item in expanded if item.get("chunk_type") == "table_row"]
    row_ids = {item.get("row_id") for item in row_items}

    assert "DiffuLT" in row_ids
    assert "cRT" in row_ids
    assert "ADRW" in row_ids
    assert any(item in row_ids for item in {"RIDE (3 experts)", "RIDE(3 experts)"})
    assert "CE" not in row_ids
    assert not any("DiffuLT+RIDE" in (item.get("row_id") or "").replace(" ", "") for item in row_items)


def test_numeric_table_expansion_prefers_dataset_matched_rows_from_page_content():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    page_content = (
        "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
        "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
        "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
        "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
        "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
        "Table9: Ablation experiments on CIFAR100-LT. Weight Acc.(%) 0.1 49.2 0.3 51.5 "
        "Table10: Performance with different alpha on CIFAR100-LT. alpha Acc.(%) 0.1 49.7 0.5 49.5"
    )
    results = [
        {
            "chunk": page_content,
            "raw_chunk_text": page_content,
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "section_path": "Experiments",
            "chunk_heading": "Page 9",
            "similarity": 0.42,
            "similarity_percent": 42.0,
            "group_id": "group-9",
        }
    ]

    expanded = _expand_numeric_table_evidence_units(
        results,
        query,
        include_rerank_text=False,
        doc_title="DiffuLT",
    )

    row_items = [item for item in expanded if item.get("chunk_type") == "table_row"]
    assert row_items
    assert all(item.get("table_id") == "Table 8" for item in row_items)
    assert {"DiffuLT", "cRT"} <= {item.get("row_id") for item in row_items}


def test_numeric_table_augment_can_backfill_global_anchor_table_chunk():
    results = [
        {
            "chunk": "方法章节摘要，不包含表格证据。",
            "page": 2,
            "similarity": 0.61,
            "similarity_percent": 61.0,
            "score": 0.61,
            "snippet": "方法章节摘要，不包含表格证据。",
            "highlights": [],
            "reranked": False,
        }
    ]
    target_chunk = (
        "ResNet-10 ResNet-50 All All Many Med. Few "
        "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
        "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
        "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
        "DiffuLT 50.4 56.4 63.3 55.6 39.4"
    )
    chunks = [
        results[0]["chunk"],
        target_chunk,
    ]
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"

    def fake_find_page(chunk_text, _pages, page_index=None):
        return 9 if chunk_text == target_chunk else 2

    with patch("services.embedding_service._find_page_for_chunk", side_effect=fake_find_page):
        augmented = _augment_with_table_chunks(
            results,
            chunks,
            pages=[{"page": 2, "text": "方法章节摘要，不包含表格证据。"}],
            page_index={},
            query=query,
            evidence_need=["numeric_table"],
            max_augment=3,
        )

    augmented_chunks = [item for item in augmented if item.get("table_augmented")]
    assert len(augmented_chunks) == 1
    assert augmented_chunks[0]["chunk"] == target_chunk
    assert augmented_chunks[0]["page"] == 9
    assert augmented_chunks[0]["table_augmented_scope"] == "global_anchor"
    assert "ResNet-50" in augmented_chunks[0]["numeric_table_anchor_hits"]
    assert "DiffuLT" in augmented_chunks[0]["numeric_table_anchor_hits"]


def test_numeric_table_augment_explicit_table_label_still_scans_global_anchor_with_local_pollution():
    query = "In Table 3, which generated sample type has the largest average gain per sample, and what are its ΔAcc/||D_gen|| and accuracy values?"
    polluted_local_chunk = (
        "Figure 2 caption mixed with Table 3 header. Table 3: Quantities, overall classifier "
        "enhancement, and Table 4 notes are merged on the same page."
    )
    target_chunk = (
        "[TABLE] Table 3: Quantities, overall classifier enhancement.\n"
        "| Type | Acc | ΔAcc/||D_gen|| |\n"
        "| --- | --- | --- |\n"
        "| ID | 44.2 | 2.75×10^-4 |\n"
        "| AID | 45.2 | 5.78×10^-4 |\n"
        "| OOD | 36.2 | -3.61×10^-4 |"
    )
    results = [
        {
            "chunk": polluted_local_chunk,
            "page": 5,
            "similarity": 0.67,
            "similarity_percent": 67.0,
            "score": 0.67,
            "snippet": polluted_local_chunk[:80],
            "highlights": [],
            "reranked": False,
        }
    ]

    def fake_find_page(chunk_text, _pages, page_index=None):
        return 9 if chunk_text == target_chunk else 5

    with patch("services.embedding_service._find_page_for_chunk", side_effect=fake_find_page):
        augmented = _augment_with_table_chunks(
            results,
            chunks=[polluted_local_chunk, target_chunk],
            pages=[{"page": 5, "text": polluted_local_chunk}],
            page_index={},
            query=query,
            evidence_need=["numeric_table"],
            max_augment=4,
        )

    augmented_chunks = [item for item in augmented if item.get("table_augmented")]
    assert any(item.get("chunk") == target_chunk for item in augmented_chunks)
    assert any(item.get("table_augmented_scope") == "global_anchor" for item in augmented_chunks)


def test_numeric_table_augment_can_add_page_content_candidate():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    results = [
        {
            "chunk": "ImageNet-LT 结果在表格中汇总。",
            "page": 9,
            "similarity": 0.63,
            "similarity_percent": 63.0,
            "score": 0.63,
            "snippet": "ImageNet-LT 结果在表格中汇总。",
            "highlights": [],
            "reranked": False,
        }
    ]
    page_content = (
        "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
        "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
        "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
        "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
        "DiffuLT 50.4 56.4 63.3 55.6 39.4"
    )

    augmented = _augment_with_table_chunks(
        results,
        chunks=["普通叙述块"],
        pages=[{"page": 9, "content": page_content}],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
    )

    page_augmented = [item for item in augmented if item.get("table_augmented_scope") == "page_content"]
    assert len(page_augmented) == 1
    assert page_augmented[0]["page"] == 9
    assert "Table 8: Results on ImageNet-LT." in page_augmented[0]["chunk"]


def test_table_row_context_text_keeps_caption_and_header():
    item = {
        "chunk_type": "table_row",
        "block_type": "table_row",
        "table_caption": "Table 8: Results on ImageNet-LT.",
        "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
        "chunk": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
    }

    context_text = _build_context_text_for_result(item)

    assert "Table 8: Results on ImageNet-LT." in context_text
    assert "ResNet-10 ResNet-50 All All Many Med. Few" in context_text
    assert "DiffuLT 50.4 56.4 63.3 55.6 39.4" in context_text


def test_table_row_context_text_prefers_boundary_text_over_wide_slice():
    item = {
        "chunk_type": "table_row",
        "block_type": "table_row",
        "table_caption": "Table 8: Results on ImageNet-LT.",
        "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
        "table_row_boundary_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
        "chunk": (
            "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
            "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
            "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
            "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
            "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
            "Table 9: Another table should not leak into the bundle. 1 2 3 4 5"
        ),
    }

    context_text = _build_context_text_for_result(item)

    assert "DiffuLT 50.4 56.4 63.3 55.6 39.4" in context_text
    assert "Table 9: Another table should not leak into the bundle." not in context_text


def test_structured_bundle_context_text_keeps_multi_row_body_for_second_best_queries():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    item = {
        "chunk_type": "table",
        "block_type": "table",
        "structured_table_bundle": True,
        "table_caption": "Table 8: Results on ImageNet-LT.",
        "table_header": "Method | All | Many | Med. | Few",
        "numeric_table_exact_context_row_text": "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
        "evidence_units": [
            {"evidence_unit_type": "table_row", "row_id": "CE", "content": "CE | All=41.6 | Many=64.0 | Med.=33.8 | Few=5.8"},
            {"evidence_unit_type": "table_row", "row_id": "cRT", "content": "cRT | All=47.3 | Many=58.8 | Med.=44.0 | Few=26.1"},
            {"evidence_unit_type": "table_row", "row_id": "RIDE(3experts)", "content": "RIDE(3experts) | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9"},
            {"evidence_unit_type": "table_row", "row_id": "ADRW", "content": "ADRW | All=54.1 | Many=62.9 | Med.=52.6 | Few=37.1"},
            {"evidence_unit_type": "table_row", "row_id": "DiffuLT", "content": "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4"},
            {"evidence_unit_type": "table_row", "row_id": "DiffuLT+RIDE(3experts)", "content": "DiffuLT+RIDE(3experts) | All=56.9 | Many=64.1 | Med.=55.8 | Few=39.9"},
        ],
    }

    context_text = _build_context_text_for_result(item, query=query)

    assert "Table 8: Results on ImageNet-LT." in context_text
    assert "cRT | All=47.3 | Many=58.8 | Med.=44.0 | Few=26.1" in context_text
    assert "RIDE(3experts) | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9" in context_text
    assert "ADRW | All=54.1 | Many=62.9 | Med.=52.6 | Few=37.1" in context_text
    assert "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4" in context_text
    assert "DiffuLT+RIDE(3experts)" not in context_text


def test_fallback_citation_from_result_prefers_exact_numeric_table_row():
    item = {
        "chunk_type": "table_row",
        "block_type": "table_row",
        "page": 4,
        "group_id": "table-1-row",
        "table_caption": "Table 1: FID of different generation models.",
        "table_header": "Model FID Acc.(%)",
        "table_row_boundary_text": "CBDM (τ=1) 5.86 46.6",
        "table_row_raw_text": "CBDM (τ=1) 5.86 46.6",
        "chunk": (
            "Table 1: FID of different generation models. Model FID Acc.(%) "
            "CBDM (τ=2) 6.82 46.0 CBDM (τ=1) 5.86 46.6"
        ),
        "cell_evidence_units": [
            {"content": "CBDM (τ=1)"},
            {"content": "5.86"},
            {"content": "46.6"},
        ],
    }

    citation = _build_fallback_citation_from_result(
        item,
        1,
        "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？",
    )

    assert citation is not None
    assert citation["source_text"].startswith("Table 1: FID of different generation models.")
    assert citation["display_text"] == "CBDM (τ=1) 5.86 46.6"
    assert citation["highlight_text"] == "CBDM (τ=1) 5.86 46.6"
    assert citation["cell_evidence_units"][1]["content"] == "5.86"


def test_fallback_citation_from_result_uses_multi_row_bundle_context_for_second_best_queries():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    item = {
        "chunk_type": "table",
        "block_type": "table",
        "page": 9,
        "group_id": "table-8-bundle",
        "structured_table_bundle": True,
        "table_caption": "Table 8: Results on ImageNet-LT.",
        "table_header": "Method | All | Many | Med. | Few",
        "numeric_table_exact_context_row_text": "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
        "evidence_units": [
            {"evidence_unit_type": "table_row", "row_id": "CE", "content": "CE | All=41.6 | Many=64.0 | Med.=33.8 | Few=5.8"},
            {"evidence_unit_type": "table_row", "row_id": "cRT", "content": "cRT | All=47.3 | Many=58.8 | Med.=44.0 | Few=26.1"},
            {"evidence_unit_type": "table_row", "row_id": "RIDE(3experts)", "content": "RIDE(3experts) | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9"},
            {"evidence_unit_type": "table_row", "row_id": "ADRW", "content": "ADRW | All=54.1 | Many=62.9 | Med.=52.6 | Few=37.1"},
            {"evidence_unit_type": "table_row", "row_id": "DiffuLT", "content": "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4"},
        ],
    }

    citation = _build_fallback_citation_from_result(item, 1, query)

    assert citation is not None
    assert "RIDE(3experts) | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9" in citation["source_text"]
    assert "ADRW | All=54.1 | Many=62.9 | Med.=52.6 | Few=37.1" in citation["source_text"]
    assert citation["display_text"] == "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4"


def test_cleanup_numeric_table_context_entries_keeps_same_table_comparator_rows_and_drops_narrative():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    narrative = _make_numeric_candidate(
        "This narrative paragraph mentions DiffuLT and Table 8 but does not contain any exact comparator values.",
        0.97,
        page=9,
        chunk_type="text",
        block_type="text",
        table_id="Table 8",
        table_caption="Table 8: Results on ImageNet-LT.",
        numeric_table_keep_support=True,
    )
    diffult = _make_numeric_candidate(
        "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
        0.46,
        page=9,
        chunk_type="table_row",
        block_type="table_row",
        table_id="Table 8",
        table_caption="Table 8: Results on ImageNet-LT.",
        table_header="Method | All | Many | Med. | Few",
        row_id="DiffuLT",
        table_row_evidence=True,
        table_row_slice_kind="exact",
        table_row_boundary_text="DiffuLT 56.4 63.3 55.6 39.4",
        table_row_raw_text="DiffuLT 56.4 63.3 55.6 39.4",
    )
    ride = _make_numeric_candidate(
        "RIDE(3 experts) | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9",
        0.45,
        page=9,
        chunk_type="table_row",
        block_type="table_row",
        table_id="Table 8",
        table_caption="Table 8: Results on ImageNet-LT.",
        table_header="Method | All | Many | Med. | Few",
        row_id="RIDE(3 experts)",
        table_row_evidence=True,
        table_row_slice_kind="exact",
        table_row_boundary_text="RIDE(3 experts) 54.9 66.2 51.7 34.9",
        table_row_raw_text="RIDE(3 experts) 54.9 66.2 51.7 34.9",
    )
    other_table = _make_numeric_candidate(
        "Table 9: Another benchmark. DiffuLT 10.0 20.0 30.0 40.0",
        0.44,
        page=10,
        chunk_type="table_row",
        block_type="table_row",
        table_id="Table 9",
        table_caption="Table 9: Another benchmark.",
        table_header="Method | All | Many | Med. | Few",
        row_id="DiffuLT",
        table_row_evidence=True,
        table_row_slice_kind="exact",
        table_row_boundary_text="DiffuLT 10.0 20.0 30.0 40.0",
        table_row_raw_text="DiffuLT 10.0 20.0 30.0 40.0",
    )

    layered = _cleanup_numeric_table_context_entries(
        [
            (narrative, _build_context_text_for_result(narrative, query=query)),
            (diffult, _build_context_text_for_result(diffult, query=query)),
            (ride, _build_context_text_for_result(ride, query=query)),
            (other_table, _build_context_text_for_result(other_table, query=query)),
        ],
        query,
    )

    texts = [entry["text"] for entry in layered]
    roles = [entry["context_role"] for entry in layered]

    assert roles[0] == "anchor"
    assert sum(1 for entry in layered if entry["context_role"] == "anchor") >= 1
    assert "DiffuLT 56.4 63.3 55.6 39.4" in texts[0]
    assert "RIDE(3 experts) 54.9 66.2 51.7 34.9" in texts[0]
    assert all("This narrative paragraph" not in text for text in texts)
    assert all("Table 9: Another benchmark." not in text for text in texts)


def test_cleanup_numeric_table_context_entries_keeps_one_explanatory_background_with_rows():
    query = "Table 3 中 Baseline 和 AttnRes 在 MMLU、GPQA-Diamond、HumanEval、C-Eval 上分别是多少？"
    explanatory = _make_numeric_candidate(
        "The benchmark evaluates reasoning, knowledge, and coding abilities across multiple datasets before reporting the table.",
        0.80,
        page=10,
        chunk_type="text",
        block_type="text",
    )
    off_topic = _make_numeric_candidate(
        "Appendix implementation notes mention a UI demo and unrelated configuration details.",
        0.79,
        page=2,
        chunk_type="text",
        block_type="text",
    )
    baseline = _make_numeric_candidate(
        "Baseline | MMLU=60.1 | GPQA-Diamond=32.0 | HumanEval=45.0 | C-Eval=70.0",
        0.72,
        page=10,
        chunk_type="table_row",
        block_type="table_row",
        table_id="Table 3",
        table_caption="Table 3: Benchmark results.",
        table_header="Method | MMLU | GPQA-Diamond | HumanEval | C-Eval",
        row_id="Baseline",
        table_row_evidence=True,
        table_row_slice_kind="exact",
        table_row_boundary_text="Baseline 60.1 32.0 45.0 70.0",
        table_row_raw_text="Baseline 60.1 32.0 45.0 70.0",
    )
    attnres = _make_numeric_candidate(
        "AttnRes | MMLU=63.5 | GPQA-Diamond=34.2 | HumanEval=48.1 | C-Eval=72.4",
        0.71,
        page=10,
        chunk_type="table_row",
        block_type="table_row",
        table_id="Table 3",
        table_caption="Table 3: Benchmark results.",
        table_header="Method | MMLU | GPQA-Diamond | HumanEval | C-Eval",
        row_id="AttnRes",
        table_row_evidence=True,
        table_row_slice_kind="exact",
        table_row_boundary_text="AttnRes 63.5 34.2 48.1 72.4",
        table_row_raw_text="AttnRes 63.5 34.2 48.1 72.4",
    )

    layered = _cleanup_numeric_table_context_entries(
        [
            (baseline, _build_context_text_for_result(baseline, query=query)),
            (attnres, _build_context_text_for_result(attnres, query=query)),
            (explanatory, _build_context_text_for_result(explanatory, query=query)),
            (off_topic, _build_context_text_for_result(off_topic, query=query)),
        ],
        query,
    )

    roles = [entry["context_role"] for entry in layered]
    texts = [entry["text"] for entry in layered]
    assert roles.count("background") == 1
    assert any("benchmark evaluates reasoning" in text for text in texts)
    assert all("UI demo" not in text for text in texts)


def test_cleanup_numeric_table_context_entries_prefers_same_table_exact_row_over_wrong_table_winner_block():
    query = "实验结果表中哪个方法在 Few-shot 子集上取得最高准确率？具体数值是多少？"
    wrong_table = _make_numeric_candidate(
        "Table 1: FID of different generation models. DDPM 7.76 43.8",
        0.94,
        page=4,
        chunk_type="table",
        block_type="table",
        table_id="Table 1",
        table_caption="Table 1: FID of different generation models.",
        table_header="Model FID Acc.(%)",
        numeric_table_exact_context_row_text="DDPM | FID=7.76 | Acc=43.8",
        numeric_table_exact_context_caption="Table 1: FID of different generation models.",
        numeric_table_exact_context_header="Model FID Acc.(%)",
    )
    right_table = _make_numeric_candidate(
        "DiffuLT | Many=69.0 | Med.=51.6 | Few=29.7",
        0.72,
        page=8,
        chunk_type="table_row",
        block_type="table_row",
        table_id="Table 7",
        table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
        table_header="Method | Many | Med. | Few",
        row_id="DiffuLT",
        table_row_evidence=True,
        table_row_slice_kind="exact",
        table_row_boundary_text="DiffuLT 69.0 51.6 29.7",
        table_row_raw_text="DiffuLT 69.0 51.6 29.7",
    )

    layered = _cleanup_numeric_table_context_entries(
        [
            (wrong_table, _build_context_text_for_result(wrong_table, query=query)),
            (right_table, _build_context_text_for_result(right_table, query=query)),
        ],
        query,
    )

    assert layered[0]["context_role"] == "anchor"
    assert "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets." in layered[0]["text"]
    assert "DiffuLT | Acc=51.6 | Few=29.7" in layered[0]["text"]
    assert all("Table 1: FID of different generation models." not in entry["text"] for entry in layered)


def test_cleanup_numeric_table_context_entries_keeps_cost_anchor_ahead_of_unrelated_table_row():
    query = "这篇论文的额外开销、推理时间和 FLOPs 分别是多少？"
    cost_anchor = _make_numeric_candidate(
        "Training time is about 24 hours on CIFAR100-LT and approximately six days on ImageNet-LT, with no extra overhead.",
        0.96,
        page=5,
        chunk_type="text",
        block_type="text",
    )
    unrelated_row = _make_numeric_candidate(
        "CBDM (τ=3) 7.42 44.8",
        0.90,
        page=4,
        chunk_type="table_row",
        block_type="table_row",
        table_id="Table 1",
        table_caption="Table 1: FID of different generation models.",
        table_header="Model FID Acc.(%)",
        row_id="CBDM(τ=3)",
        table_row_evidence=True,
        table_row_slice_kind="exact",
        table_row_boundary_text="CBDM (τ=3) 7.42 44.8",
        table_row_raw_text="CBDM (τ=3) 7.42 44.8",
    )
    broad_narrative = _make_numeric_candidate(
        "The method improves performance on several long-tailed benchmarks and includes an ablation discussion.",
        0.84,
        page=5,
        chunk_type="text",
        block_type="text",
    )

    layered = _cleanup_numeric_table_context_entries(
        [
            (cost_anchor, _build_context_text_for_result(cost_anchor, query=query)),
            (unrelated_row, _build_context_text_for_result(unrelated_row, query=query)),
            (broad_narrative, _build_context_text_for_result(broad_narrative, query=query)),
        ],
        query,
    )

    assert layered[0]["context_role"] == "anchor"
    assert "24 hours" in layered[0]["text"]
    assert all("CBDM (τ=3)" not in entry["text"] for entry in layered)


def test_cleanup_numeric_table_context_entries_keeps_same_table_bundle_projection_for_comparator_rows():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    bundle = _make_numeric_candidate(
        "[Structured Table Bundle] Table 8 ...",
        0.98,
        page=9,
        chunk_type="table",
        block_type="table",
        structured_table_bundle=True,
        table_id="Table 8",
        table_caption="Table 8: Results on ImageNet-LT.",
        table_header="ResNet-10 ResNet-50 All All Many Med. Few",
        evidence_units=[
            {
                "evidence_unit_type": "table_row",
                "row_id": "DiffuLT",
                "row_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
                "row_numbers": "50.4 56.4 63.3 55.6 39.4",
            },
            {
                "evidence_unit_type": "table_row",
                "row_id": "cRT",
                "row_text": "cRT 41.8 47.3 58.8 44.0 26.1",
                "row_numbers": "41.8 47.3 58.8 44.0 26.1",
            },
            {
                "evidence_unit_type": "table_row",
                "row_id": "RIDE (3 experts)",
                "row_text": "RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
                "row_numbers": "45.9 54.9 66.2 51.7 34.9",
            },
            {
                "evidence_unit_type": "table_row",
                "row_id": "ADRW",
                "row_text": "ADRW 54.1 62.9 52.6 37.1",
                "row_numbers": "54.1 62.9 52.6 37.1",
            },
        ],
    )
    rows = [
        _make_numeric_candidate(
            "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            0.49,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 8",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            row_id="DiffuLT",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
            table_row_raw_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
        ),
        _make_numeric_candidate(
            "cRT | All=47.3 | Many=58.8 | Med.=44.0 | Few=26.1",
            0.48,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 8",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            row_id="cRT",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="cRT 41.8 47.3 58.8 44.0 26.1",
            table_row_raw_text="cRT 41.8 47.3 58.8 44.0 26.1",
        ),
        _make_numeric_candidate(
            "RIDE (3 experts) | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9",
            0.47,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 8",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            row_id="RIDE (3 experts)",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            table_row_raw_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
        ),
        _make_numeric_candidate(
            "ADRW | All=54.1 | Many=62.9 | Med.=52.6 | Few=37.1",
            0.46,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 8",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            row_id="ADRW",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="ADRW 54.1 62.9 52.6 37.1",
            table_row_raw_text="ADRW 54.1 62.9 52.6 37.1",
        ),
    ]

    layered = _cleanup_numeric_table_context_entries(
        [
            (bundle, _build_context_text_for_result(bundle, query=query)),
            *((
                row,
                _build_context_text_for_result(row, query=query),
            ) for row in rows),
        ],
        query,
    )

    row_ids = [
        entry["item"].get("row_id")
        for entry in layered
        if entry["item"].get("chunk_type") == "table_row"
    ]

    assert {"DiffuLT", "cRT", "RIDE (3 experts)", "ADRW"} <= set(row_ids)


@pytest.mark.parametrize(
    ("query", "winner_row", "winner_raw", "caption", "header", "table_id", "focus_columns", "expected_fragments"),
    [
        (
            "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？",
            "CBDM (τ=1) | FID=5.86 | Acc=46.6",
            "CBDM (τ=1) 5.86 46.6",
            "Table 1: FID of different generation models and their classifiers' accuracy.",
            "Model FID Acc.(%)",
            "Table 1",
            ["FID", "Acc"],
            ["CBDM (τ=1)", "5.86", "46.6"],
        ),
        (
            "In Table 3, which generated sample type has the largest average gain per sample, and what are its ΔAcc/||D_gen|| and accuracy values?",
            "AID | Acc=45.2 | ΔAcc/||D_gen||=5.78×10^-4",
            "AID 45.2 5.78×10^-4",
            "Table 3: Quantities and classifier enhancement.",
            "Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||",
            "Table 3",
            ["ΔAcc/||D_gen||", "Acc"],
            ["AID", "45.2", "5.78×10^-4"],
        ),
    ],
)
def test_same_bundle_hard_gate_propagates_exact_row_to_support_context(
    query,
    winner_row,
    winner_raw,
    caption,
    header,
    table_id,
    focus_columns,
    expected_fragments,
):
    filtered = _apply_numeric_table_same_bundle_hard_gate(
        [
            _make_numeric_candidate(
                "Narrative summary about mixed page content.",
                0.99,
                page=5,
                chunk_type="text",
                block_type="text",
                table_augmented_scope="page_content",
                numeric_table_priority=13.0,
                numeric_table_anchor_hits=[table_id],
            ),
            _make_numeric_candidate(
                winner_row,
                0.48,
                page=5,
                chunk_type="table_row",
                block_type="table_row",
                table_caption=caption,
                table_header=header,
                table_id=table_id,
                row_id=winner_row.split("|", 1)[0].strip(),
                table_focus_columns=focus_columns,
                table_row_evidence=True,
                table_row_slice_kind="exact",
                table_row_boundary_text=winner_raw,
                table_row_raw_text=winner_raw,
                raw_chunk_text=winner_raw,
                numeric_table_priority=11.6,
                numeric_table_anchor_hits=[table_id],
            ),
            _make_numeric_candidate(
                caption,
                0.44,
                page=5,
                chunk_type="caption",
                block_type="caption",
                table_caption=caption,
                table_header=header,
                table_id=table_id,
                raw_chunk_text=caption,
                numeric_table_priority=10.2,
                numeric_table_anchor_hits=[table_id],
            ),
            _make_numeric_candidate(
                f"{table_id}: structured table support",
                0.42,
                page=5,
                chunk_type="table",
                block_type="table",
                table_caption=caption,
                table_header=header,
                table_id=table_id,
                raw_chunk_text=f"{table_id}: structured table support",
                numeric_table_priority=10.0,
                numeric_table_anchor_hits=[table_id],
            ),
        ],
        query,
    )

    support_contexts = [
        _build_context_text_for_result(item)
        for item in filtered
        if item.get("chunk_type") in {"caption", "table"}
    ]

    assert support_contexts
    for fragment in expected_fragments:
        assert any(fragment in context_text for context_text in support_contexts)


def test_retrieval_diagnostics_treats_cost_queries_as_numeric_table():
    diagnostics = _build_retrieval_diagnostics(
        [
            _make_numeric_candidate(
                "Training time is about 24 hours on CIFAR100-LT and approximately six days on ImageNet-LT.",
                0.65,
                page=6,
                chunk_type="text",
                block_type="text",
                raw_chunk_text="Training time is about 24 hours on CIFAR100-LT and approximately six days on ImageNet-LT.",
            )
        ],
        "DiffuLT 的训练时间和额外推理开销分别是多少？",
    )

    assert diagnostics["numeric_table_query"] is True
    assert diagnostics["numeric_table_hit_quality"] == 1.0


def test_numeric_table_focused_row_prefers_boundary_text_over_wide_slice():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    hints = QueryRewriter().extract_numeric_table_hints(query)
    unit = {
        "row_id": "DiffuLT",
        "table_row_boundary_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
        "row_text": (
            "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
            "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
            "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
            "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
            "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
            "Table 9: Another table should not leak into the bundle. 1 2 3 4 5"
        ),
        "table_caption": "Table 8: Results on ImageNet-LT.",
        "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
    }

    focused = _build_query_focused_table_row(unit, hints)

    assert focused["text"] == "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4"


def test_numeric_table_dedup_replaces_original_table_chunk_with_row_evidence():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    results = [
        {
            "chunk": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few ... DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "raw_chunk_text": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few ... DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "group_id": "group-8",
            "similarity": 0.5,
        },
        {
            "chunk": "Table 8: Results on ImageNet-LT.",
            "raw_chunk_text": "Table 8: Results on ImageNet-LT.",
            "page": 9,
            "chunk_type": "caption",
            "block_type": "caption",
            "group_id": "group-8",
            "similarity": 0.4,
        },
        {
            "chunk": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "page": 9,
            "chunk_type": "table_row",
            "block_type": "table_row",
            "group_id": "group-8",
            "table_id": "Table 8",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "similarity": 0.7,
        },
    ]

    deduped = _dedupe_numeric_table_evidence_units(results, query)

    chunk_types = [item.get("chunk_type") for item in deduped]
    assert chunk_types == ["table_row"]


def test_conditional_rerank_bypasses_when_numeric_table_bundle_is_already_stable():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    candidates = [
        {
            "chunk": "RIDE(3 experts) | ResNet-50 | All=54.9",
            "raw_chunk_text": "RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "RIDE(3 experts)",
            "table_focus_backbone": "ResNet-50",
            "table_focus_columns": ["All"],
            "page": 9,
            "similarity": 0.91,
        },
        {
            "chunk": "DiffuLT | ResNet-50 | All=56.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "DiffuLT",
            "table_focus_backbone": "ResNet-50",
            "table_focus_columns": ["All"],
            "page": 9,
            "similarity": 0.9,
        },
        {
            "chunk": "cRT | ResNet-50 | All=47.3",
            "raw_chunk_text": "cRT 41.8 47.3 58.8 44.0 26.1",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "cRT",
            "table_focus_backbone": "ResNet-50",
            "table_focus_columns": ["All"],
            "page": 9,
            "similarity": 0.89,
        },
        {
            "chunk": "ADRW | ResNet-50 | All=54.1",
            "raw_chunk_text": "ADRW - 54.1 62.9 52.6 37.1",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "ADRW",
            "table_focus_backbone": "ResNet-50",
            "table_focus_columns": ["All"],
            "page": 9,
            "similarity": 0.88,
        },
        {
            "chunk": "our methods. CIFAR100-LT CIFAR10-LTMethod | ResNet-50 | All=10",
            "raw_chunk_text": "our methods. CIFAR100-LT CIFAR10-LTMethod 100 50 10 100 50 10",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "our methods. CIFAR100-LT CIFAR10-LTMethod",
            "table_focus_backbone": "ResNet-50",
            "table_focus_columns": ["All"],
            "page": 14,
            "similarity": 0.84,
        },
    ]

    with patch.object(
        _finalize_with_optional_rerank.__globals__["_rag_config_singleton"],
        "enable_focus_mode",
        False,
        create=True,
    ), patch(
        "services.embedding_service._apply_rerank",
        side_effect=AssertionError("stable numeric_table bundle should bypass conditional rerank"),
    ):
        final = _finalize_with_optional_rerank(
            query=query,
            results=candidates,
            top_k=4,
            use_rerank=True,
            reranker_model="test-reranker",
            rerank_provider="local",
            rerank_api_key=None,
            rerank_endpoint=None,
            timings={},
            conditional_rerank_active=True,
        )

    assert [item["row_id"] for item in final[:4]] == [
        "RIDE(3 experts)",
        "DiffuLT",
        "cRT",
        "ADRW",
    ]


def test_conditional_rerank_bypass_expands_topk_for_explicit_comparator_queries():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    candidates = [
        {
            "chunk": "RIDE(3 experts) | ResNet-50 | All=54.9",
            "raw_chunk_text": "RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "RIDE(3 experts)",
            "table_focus_backbone": "ResNet-50",
            "table_focus_columns": ["All"],
            "page": 9,
            "similarity": 0.91,
        },
        {
            "chunk": "DiffuLT | ResNet-50 | All=56.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "DiffuLT",
            "table_focus_backbone": "ResNet-50",
            "table_focus_columns": ["All"],
            "page": 9,
            "similarity": 0.9,
        },
        {
            "chunk": "cRT | ResNet-50 | All=47.3",
            "raw_chunk_text": "cRT 41.8 47.3 58.8 44.0 26.1",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "cRT",
            "table_focus_backbone": "ResNet-50",
            "table_focus_columns": ["All"],
            "page": 9,
            "similarity": 0.89,
        },
        {
            "chunk": "ADRW | ResNet-50 | All=54.1",
            "raw_chunk_text": "ADRW - 54.1 62.9 52.6 37.1",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "row_id": "ADRW",
            "table_focus_backbone": "ResNet-50",
            "table_focus_columns": ["All"],
            "page": 9,
            "similarity": 0.88,
        },
    ]

    with patch.object(
        _finalize_with_optional_rerank.__globals__["_rag_config_singleton"],
        "enable_focus_mode",
        False,
        create=True,
    ), patch(
        "services.embedding_service._apply_rerank",
        side_effect=AssertionError("stable numeric_table bundle should bypass conditional rerank"),
    ):
        final = _finalize_with_optional_rerank(
            query=query,
            results=candidates,
            top_k=3,
            use_rerank=True,
            reranker_model="test-reranker",
            rerank_provider="local",
            rerank_api_key=None,
            rerank_endpoint=None,
            timings={},
            conditional_rerank_active=True,
        )

    assert [item["row_id"] for item in final[:4]] == [
        "RIDE(3 experts)",
        "DiffuLT",
        "cRT",
        "ADRW",
    ]


def test_conditional_rerank_bypass_keeps_best_row_for_best_table_query():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    table1_caption = "Table 1: FID of different generation models and their classifiers' accuracy."
    table1_header = "Model FID Acc. (%)"
    candidates = [
        _make_numeric_candidate(
            "DDPM | FID=7.76 | Acc=43.8",
            0.98,
            page=4,
            chunk_type="table_row",
            block_type="table_row",
            table_caption=table1_caption,
            table_header=table1_header,
            table_id="Table 1",
            table_bundle_id="page-text:table 1",
            row_id="DDPM",
            table_focus_columns=["FID", "Acc"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            raw_chunk_text="DDPM 7.76 43.8",
        ),
        _make_numeric_candidate(
            "CBDM(τ =1) | FID=5.86 | Acc=46.6",
            0.72,
            page=4,
            chunk_type="table_row",
            block_type="table_row",
            table_caption=table1_caption,
            table_header=table1_header,
            table_id="Table 1",
            table_bundle_id="page-text:table 1",
            row_id="CBDM(τ =1)",
            table_focus_columns=["FID", "Acc"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            raw_chunk_text="CBDM(τ =1) 5.86 46.6",
        ),
        _make_numeric_candidate(
            "CBDM(τ =2) | FID=6.82 | Acc=46.0",
            0.71,
            page=4,
            chunk_type="table_row",
            block_type="table_row",
            table_caption=table1_caption,
            table_header=table1_header,
            table_id="Table 1",
            table_bundle_id="page-text:table 1",
            row_id="CBDM(τ =2)",
            table_focus_columns=["FID", "Acc"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            raw_chunk_text="CBDM(τ =2) 6.82 46.0",
        ),
        _make_numeric_candidate(
            table1_caption,
            0.66,
            page=4,
            chunk_type="table",
            block_type="table",
            table_caption=table1_caption,
            table_header=table1_header,
            table_id="Table 1",
            table_bundle_id="page-text:table 1",
            raw_chunk_text=table1_caption,
        ),
    ]

    with patch.object(
        _finalize_with_optional_rerank.__globals__["_rag_config_singleton"],
        "enable_focus_mode",
        False,
        create=True,
    ), patch(
        "services.embedding_service._apply_rerank",
        side_effect=AssertionError("best-row numeric_table bundle should bypass conditional rerank"),
    ):
        final = _finalize_with_optional_rerank(
            query=query,
            results=candidates,
            top_k=3,
            use_rerank=True,
            reranker_model="test-reranker",
            rerank_provider="local",
            rerank_api_key=None,
            rerank_endpoint=None,
            timings={},
            conditional_rerank_active=True,
        )

    row_ids = [item["row_id"] for item in final if item.get("chunk_type") == "table_row"]
    assert row_ids == ["CBDM(τ =1)"]
    assert any(item.get("table_bundle_id") == "page-text:table 1" for item in final)


def test_numeric_table_slot_reservation_keeps_table_row_inside_final_topk():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    ordered = [
        {
            "chunk": "DiffuLT achieves state-of-the-art results on ImageNet-LT and Table 8 summarizes the gains.",
            "raw_chunk_text": "DiffuLT achieves state-of-the-art results on ImageNet-LT and Table 8 summarizes the gains.",
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.91,
            "numeric_table_priority": 1.2,
        },
        {
            "chunk": "The experiments section highlights strong overall improvements across long-tailed benchmarks.",
            "raw_chunk_text": "The experiments section highlights strong overall improvements across long-tailed benchmarks.",
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.88,
            "numeric_table_priority": 0.9,
        },
        {
            "chunk": "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "DiffuLT",
            "similarity": 0.37,
            "numeric_table_priority": 12.5,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "DiffuLT", "Many", "Med.", "Few"],
        },
    ]

    final = _ensure_numeric_table_evidence_slots(ordered, query, top_k=2)

    assert len(final) == 2
    assert any(item.get("chunk_type") == "table_row" for item in final)
    assert final[0].get("chunk_type") == "table_row"


def test_numeric_table_slot_reservation_ignores_narrative_support_lookalikes():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    ordered = [
        {
            "chunk": (
                "Table 8 reports that DiffuLT with ResNet-50 performs strongly on ImageNet-LT, "
                "including All, Many, Med., and Few subsets."
            ),
            "raw_chunk_text": (
                "Table 8 reports that DiffuLT with ResNet-50 performs strongly on ImageNet-LT, "
                "including All, Many, Med., and Few subsets."
            ),
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.94,
            "numeric_table_priority": 12.4,
            "numeric_table_anchor_hits": ["Table 8", "ImageNet-LT", "ResNet-50", "DiffuLT", "Many"],
        },
        {
            "chunk": (
                "The experimental summary reiterates the Table 8 ResNet-50 gains for DiffuLT on "
                "All, Many, Med., and Few without showing the original row values."
            ),
            "raw_chunk_text": (
                "The experimental summary reiterates the Table 8 ResNet-50 gains for DiffuLT on "
                "All, Many, Med., and Few without showing the original row values."
            ),
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.91,
            "numeric_table_priority": 11.9,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "DiffuLT", "Med.", "Few"],
        },
        {
            "chunk": "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "DiffuLT",
            "similarity": 0.38,
            "numeric_table_priority": 10.8,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "DiffuLT", "Many", "Med.", "Few"],
        },
        {
            "chunk": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
            "raw_chunk_text": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
            "chunk_type": "table",
            "block_type": "table",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "similarity": 0.33,
            "numeric_table_priority": 9.7,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "Many", "Med.", "Few"],
        },
    ]

    final = _ensure_numeric_table_evidence_slots(ordered, query, top_k=2)

    assert len(final) == 2
    assert final[0].get("chunk_type") == "table_row"
    assert {item.get("chunk_type") for item in final} == {"table_row", "table"}


def test_numeric_table_slot_reservation_applies_after_rerank():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    reranked = [
        {
            "chunk": (
                "Table 8 reports that DiffuLT with ResNet-50 performs strongly on ImageNet-LT, "
                "including All, Many, Med., and Few subsets."
            ),
            "raw_chunk_text": (
                "Table 8 reports that DiffuLT with ResNet-50 performs strongly on ImageNet-LT, "
                "including All, Many, Med., and Few subsets."
            ),
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.94,
            "rerank_score": 0.94,
            "numeric_table_priority": 12.4,
            "numeric_table_anchor_hits": ["Table 8", "ImageNet-LT", "ResNet-50", "DiffuLT", "Many"],
        },
        {
            "chunk": (
                "The experimental summary reiterates the Table 8 ResNet-50 gains for DiffuLT on "
                "All, Many, Med., and Few without showing the original row values."
            ),
            "raw_chunk_text": (
                "The experimental summary reiterates the Table 8 ResNet-50 gains for DiffuLT on "
                "All, Many, Med., and Few without showing the original row values."
            ),
            "chunk_type": "text",
            "block_type": "text",
            "similarity": 0.91,
            "rerank_score": 0.91,
            "numeric_table_priority": 11.9,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "DiffuLT", "Med.", "Few"],
        },
        {
            "chunk": "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "DiffuLT",
            "similarity": 0.38,
            "rerank_score": 0.38,
            "numeric_table_priority": 10.8,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "DiffuLT", "Many", "Med.", "Few"],
        },
        {
            "chunk": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
            "raw_chunk_text": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
            "chunk_type": "table",
            "block_type": "table",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "similarity": 0.33,
            "rerank_score": 0.33,
            "numeric_table_priority": 9.7,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "Many", "Med.", "Few"],
        },
    ]

    with patch("services.embedding_service._apply_rerank", return_value=reranked), \
         patch("services.embedding_service._apply_evidence_gate", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._apply_group_post_cap", side_effect=lambda results, top_k: (results, None)), \
         patch.object(
             _finalize_with_optional_rerank.__globals__["_rag_config_singleton"],
             "enable_focus_mode",
             False,
             create=True,
         ):
        final = _finalize_with_optional_rerank(
            query=query,
            results=reranked,
            top_k=2,
            use_rerank=True,
            reranker_model="test-reranker",
            rerank_provider="local",
            rerank_api_key=None,
            rerank_endpoint=None,
            timings={},
        )

    assert len(final) == 2
    assert final[0].get("chunk_type") == "table_row"
    assert {item.get("chunk_type") for item in final} == {"table_row", "table"}


def test_numeric_table_slot_reservation_demotes_page_content_pollution_after_exact_row():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    candidates = [
        {
            "chunk": (
                "Table5: FID of diffusion model. Table6: retained samples statistics. "
                "CBDM All 39,153 46.6 DDPM 7.76 39.1 21.2 39.7 43.8"
            ),
            "raw_chunk_text": (
                "Table5: FID of diffusion model. Table6: retained samples statistics. "
                "CBDM All 39,153 46.6 DDPM 7.76 39.1 21.2 39.7 43.8"
            ),
            "chunk_type": "table",
            "block_type": "table",
            "page": 9,
            "similarity": 0.95,
            "table_augmented": True,
            "table_augmented_scope": "page_content",
            "numeric_table_priority": 14.2,
            "numeric_table_anchor_hits": ["ResNet-50", "Many", "Med.", "Few"],
        },
        {
            "chunk": "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "DiffuLT",
            "page": 9,
            "similarity": 0.40,
            "numeric_table_priority": 12.8,
            "numeric_table_anchor_hits": ["Table 8", "ImageNet-LT", "ResNet-50", "DiffuLT", "Many", "Med.", "Few"],
        },
        {
            "chunk": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
            "raw_chunk_text": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
            "chunk_type": "table",
            "block_type": "table",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "page": 9,
            "similarity": 0.35,
            "numeric_table_priority": 9.8,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "Many", "Med.", "Few"],
        },
    ]

    ordered = _prioritize_numeric_table_results(candidates, query)
    final = _ensure_numeric_table_evidence_slots(ordered, query, top_k=2)

    assert len(final) == 2
    assert final[0].get("chunk_type") == "table_row"
    assert {item.get("chunk_type") for item in final} == {"table_row", "table"}
    assert all(item.get("table_id") == "Table 8" for item in final if item.get("table_id"))
    assert all(item.get("table_augmented_scope") != "page_content" for item in final)


def test_numeric_table_slot_reservation_keeps_requested_comparator_rows():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    results = [
        {
            "chunk": "DiffuLT | ResNet-50 | All=56.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "DiffuLT",
            "page": 9,
            "similarity": 0.41,
            "numeric_table_priority": 14.1,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "DiffuLT", "All"],
        },
        {
            "chunk": "Table 8 summary paragraph mentioning DiffuLT and comparator gains.",
            "raw_chunk_text": "Table 8 summary paragraph mentioning DiffuLT and comparator gains.",
            "chunk_type": "text",
            "block_type": "text",
            "page": 9,
            "similarity": 0.95,
            "numeric_table_priority": 12.7,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "DiffuLT", "cRT"],
        },
        {
            "chunk": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
            "raw_chunk_text": "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
            "chunk_type": "table",
            "block_type": "table",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "page": 9,
            "similarity": 0.39,
            "numeric_table_priority": 10.2,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "All"],
        },
        {
            "chunk": "cRT | ResNet-50 | All=47.3",
            "raw_chunk_text": "cRT 41.8 47.3 58.8 44.0 26.1",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "cRT",
            "page": 9,
            "similarity": 0.36,
            "numeric_table_priority": 9.8,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "cRT", "All"],
        },
        {
            "chunk": "RIDE (3 experts) | ResNet-50 | All=54.9",
            "raw_chunk_text": "RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "RIDE (3 experts)",
            "page": 9,
            "similarity": 0.35,
            "numeric_table_priority": 9.7,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "RIDE (3 experts)", "All"],
        },
        {
            "chunk": "ADRW | ResNet-50 | All=54.1",
            "raw_chunk_text": "ADRW - 54.1 62.9 52.6 37.1",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "ADRW",
            "page": 9,
            "similarity": 0.34,
            "numeric_table_priority": 9.6,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "ADRW", "All"],
        },
    ]

    final = _ensure_numeric_table_evidence_slots(results, query, top_k=4)

    assert {item.get("row_id") for item in final if item.get("row_id")} == {
        "DiffuLT",
        "cRT",
        "RIDE (3 experts)",
        "ADRW",
    }


def test_numeric_table_priority_hard_gates_page_content_noise_when_exact_rows_exist():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    candidates = [
        {
            "chunk": "DiffuLT(1) 51.5 56.3 63.8 84.7 86.9 90.7",
            "raw_chunk_text": "DiffuLT(1) 51.5 56.3 63.8 84.7 86.9 90.7",
            "chunk_type": "text",
            "block_type": "text",
            "page": 9,
            "similarity": 0.95,
            "similarity_percent": 95.0,
            "numeric_table_priority": 12.7,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "DiffuLT", "cRT"],
            "table_augmented_scope": "page_content",
        },
        {
            "chunk": "DiffuLT | ResNet-50 | All=56.4",
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "DiffuLT",
            "page": 9,
            "similarity": 0.41,
            "similarity_percent": 41.0,
            "numeric_table_priority": 14.1,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "DiffuLT", "All"],
            "table_row_slice_kind": "exact",
        },
        {
            "chunk": "Table 8: Results on ImageNet-LT.",
            "raw_chunk_text": "Table 8: Results on ImageNet-LT.",
            "chunk_type": "caption",
            "block_type": "caption",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_id": "Table 8",
            "page": 9,
            "similarity": 0.5,
            "similarity_percent": 50.0,
            "numeric_table_priority": 10.2,
            "numeric_table_anchor_hits": ["Table 8", "ResNet-50", "All"],
        },
    ]

    final = _ensure_numeric_table_evidence_slots(candidates, query, top_k=2)

    assert len(final) == 2
    assert final[0].get("chunk_type") == "table_row"
    assert final[0].get("row_id") == "DiffuLT"
    assert all(item.get("chunk_type") != "text" for item in final)
    assert all(item.get("table_augmented_scope") != "page_content" for item in final)


def test_numeric_table_expansion_second_best_query_skips_composite_target_method_rows():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    results = [
        {
            "chunk": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
                "DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9"
            ),
            "raw_chunk_text": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
                "DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9"
            ),
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "section_path": "Experiments",
            "chunk_heading": "Table 8",
            "similarity": 0.42,
            "similarity_percent": 42.0,
            "group_id": "group-8",
        }
    ]

    expanded = _expand_numeric_table_evidence_units(results, query)
    row_ids = {item.get("row_id") for item in expanded if item.get("chunk_type") == "table_row"}

    assert "DiffuLT" in row_ids
    assert "cRT" in row_ids
    assert "RIDE (3 experts)" in row_ids
    assert "ADRW" in row_ids
    assert "DiffuLT + RIDE (3 experts)" not in row_ids


def test_numeric_table_slot_reservation_prefers_table_rows_for_best_few_query():
    query = "实验结果表中哪个方法在 Few-shot 子集上取得最高准确率？具体数值是多少？"
    results = [
        {
            "chunk": "Few-shot summary paragraph saying DiffuLT improves by 33.6% over baseline.",
            "raw_chunk_text": "Few-shot summary paragraph saying DiffuLT improves by 33.6% over baseline.",
            "chunk_type": "text",
            "block_type": "text",
            "page": 8,
            "similarity": 0.97,
            "numeric_table_priority": 13.6,
            "numeric_table_anchor_hits": ["Few", "DiffuLT"],
        },
        {
            "chunk": "DiffuLT | Few=29.7",
            "raw_chunk_text": "DiffuLT 69.0 51.6 29.7",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            "table_header": "Method Many Med. Few",
            "table_id": "Table 7",
            "row_id": "DiffuLT",
            "page": 8,
            "similarity": 0.42,
            "numeric_table_priority": 9.2,
            "numeric_table_anchor_hits": ["Few", "DiffuLT"],
            "table_focus_columns": ["Few"],
        },
        {
            "chunk": "CSA | Few=18.2",
            "raw_chunk_text": "CSA 64.3 49.7 18.2",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            "table_header": "Method Many Med. Few",
            "table_id": "Table 7",
            "row_id": "CSA",
            "page": 8,
            "similarity": 0.39,
            "numeric_table_priority": 8.8,
            "numeric_table_anchor_hits": ["Few"],
            "table_focus_columns": ["Few"],
        },
        {
            "chunk": "RIDE (3 experts) | Few=23.9",
            "raw_chunk_text": "RIDE (3 experts) 68.1 49.2 23.9",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            "table_header": "Method Many Med. Few",
            "table_id": "Table 7",
            "row_id": "RIDE (3 experts)",
            "page": 8,
            "similarity": 0.38,
            "numeric_table_priority": 8.7,
            "numeric_table_anchor_hits": ["Few", "RIDE (3 experts)"],
            "table_focus_columns": ["Few"],
        },
    ]

    config = SimpleNamespace(
        enable_focus_mode=False,
        focus_mode_window_size=1,
        focus_mode_max_sentences=2,
        focus_mode_min_chars=80,
    )
    ordered = _prioritize_numeric_table_results(results, query)
    filtered = _apply_numeric_table_same_bundle_hard_gate(ordered, query)
    final = _finalize_without_rerank(filtered, query, top_k=3, config=config)

    assert all(item.get("chunk_type") == "table_row" for item in filtered)
    assert {item.get("row_id") for item in filtered} == {
        "DiffuLT",
        "CSA",
        "RIDE (3 experts)",
    }
    assert final[0].get("row_id") == "DiffuLT"


def test_numeric_table_slot_reservation_keeps_best_model_bundle_for_explicit_table_query():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    results = [
        {
            "chunk": "Narrative summary about improved FID and classifier accuracy.",
            "raw_chunk_text": "Narrative summary about improved FID and classifier accuracy.",
            "chunk_type": "text",
            "block_type": "text",
            "page": 5,
            "similarity": 0.98,
            "numeric_table_priority": 13.2,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
        },
        {
            "chunk": "CBDM (τ=1) | FID=5.86 | Acc=46.6",
            "raw_chunk_text": "CBDM (τ=1) 5.86 46.6",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 1: FID of different generation models and their classifiers' accuracy.",
            "table_header": "Model FID Acc. (%)",
            "table_id": "Table 1",
            "row_id": "CBDM (τ=1)",
            "page": 5,
            "similarity": 0.41,
            "numeric_table_priority": 8.4,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
            "table_focus_columns": ["FID", "Acc"],
        },
        {
            "chunk": "CBDM (τ=2) | FID=6.82 | Acc=46.0",
            "raw_chunk_text": "CBDM (τ=2) 6.82 46.0",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 1: FID of different generation models and their classifiers' accuracy.",
            "table_header": "Model FID Acc. (%)",
            "table_id": "Table 1",
            "row_id": "CBDM (τ=2)",
            "page": 5,
            "similarity": 0.39,
            "numeric_table_priority": 8.1,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
            "table_focus_columns": ["FID", "Acc"],
        },
        {
            "chunk": "DDPM | FID=7.76 | Acc=43.8",
            "raw_chunk_text": "DDPM 7.76 43.8",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 1: FID of different generation models and their classifiers' accuracy.",
            "table_header": "Model FID Acc. (%)",
            "table_id": "Table 1",
            "row_id": "DDPM",
            "page": 5,
            "similarity": 0.37,
            "numeric_table_priority": 7.8,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
            "table_focus_columns": ["FID", "Acc"],
        },
    ]

    ordered = _prioritize_numeric_table_results(results, query)
    final = _ensure_numeric_table_evidence_slots(ordered, query, top_k=3)

    assert all(item.get("chunk_type") == "table_row" for item in final)
    assert final[0].get("row_id") == "CBDM (τ=1)"
    assert {item.get("row_id") for item in final} == {
        "CBDM (τ=1)",
        "CBDM (τ=2)",
        "DDPM",
    }


def test_numeric_table_slot_reservation_keeps_target_table_row_for_explicit_table_query():
    query = "In Table 3, which generated sample type has the largest average gain per sample, and what are its ΔAcc/||D_gen|| and accuracy values?"
    results = [
        {
            "chunk": "Figure 3 explanation mixed with appendix statistics.",
            "raw_chunk_text": "Figure 3 explanation mixed with appendix statistics.",
            "chunk_type": "text",
            "block_type": "text",
            "page": 5,
            "similarity": 0.92,
            "numeric_table_priority": 11.4,
            "numeric_table_anchor_hits": ["Table 3", "AID"],
        },
        {
            "chunk": "Table 3 mixed caption without row values.",
            "raw_chunk_text": "Table 3 mixed caption without row values.",
            "chunk_type": "caption",
            "block_type": "caption",
            "table_caption": "Table 3: Quantities and classifier enhancement.",
            "table_id": "Table 3",
            "page": 5,
            "similarity": 0.47,
            "numeric_table_priority": 8.4,
            "numeric_table_anchor_hits": ["Table 3"],
        },
        {
            "chunk": "AID | Acc=45.2 | ΔAcc/||D_gen||=5.78×10^-4",
            "raw_chunk_text": "AID 45.2 5.78×10^-4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 3: Quantities and classifier enhancement.",
            "table_header": "ID AID OOD Acc ΔAcc/||D_gen||",
            "table_id": "Table 3",
            "row_id": "AID",
            "page": 5,
            "similarity": 0.33,
            "numeric_table_priority": 7.5,
            "numeric_table_anchor_hits": ["Table 3", "AID"],
        },
    ]

    final = _ensure_numeric_table_evidence_slots(results, query, top_k=2)

    assert any(
        item.get("chunk_type") == "table_row" and item.get("table_id") == "Table 3"
        for item in final
    )


def test_numeric_table_row_extraction_keeps_compact_comparator_bundle_and_stops_at_next_table():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    hints = QueryRewriter().extract_numeric_table_hints(query)
    text = (
        "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
        "cRTKangetal.[2019] 41.8 47.3 58.8 44.0 26.1 "
        "RIDE(3experts)Wangetal.[2020] 45.9 54.9 66.2 51.7 34.9 "
        "ADRWWangetal.[2024b] - 54.1 62.9 52.6 37.1 "
        "DiffuLT 50.4 56.4 63.3 55.6 39.4 "
        "Table 9: Another table should not leak into the bundle. 1 2 3 4 5"
    )

    rows = _extract_plain_table_rows(text, hints)
    row_texts = [row.get("row_text", "") for row in rows]
    row_ids = {row.get("row_id") for row in rows}

    assert any("41.8" in row_text and "47.3" in row_text for row_text in row_texts)
    assert any("45.9" in row_text and "54.9" in row_text for row_text in row_texts)
    assert any("54.1" in row_text and "37.1" in row_text for row_text in row_texts)
    assert any("50.4" in row_text and "56.4" in row_text for row_text in row_texts)
    assert "cRT" in row_ids
    assert "ADRW" in row_ids
    assert any(item in row_ids for item in {"RIDE(3experts)", "RIDE (3experts)"})
    assert not any("Table 9" in row_text for row_text in row_texts)


def test_numeric_table_priority_prefers_highest_accuracy_row_for_table1_alias():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    hints = QueryRewriter().extract_numeric_table_hints(query)
    results = [
        {
            "chunk": "Narrative summary about improved FID and classifier accuracy.",
            "raw_chunk_text": "Narrative summary about improved FID and classifier accuracy.",
            "chunk_type": "text",
            "block_type": "text",
            "page": 5,
            "similarity": 0.98,
            "numeric_table_priority": 13.2,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
        },
        {
            "chunk": "CBDM (τ=2) | FID=6.82 | Acc=46.0",
            "raw_chunk_text": "CBDM (τ=2) 6.82 46.0",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table1: FID of different generation models and their classifiers' accuracy.",
            "table_header": "Model FID Acc. (%)",
            "table_id": "Table1",
            "row_id": "CBDM (τ=2)",
            "page": 5,
            "similarity": 0.39,
            "numeric_table_priority": 8.1,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
            "table_focus_columns": ["FID", "Acc"],
        },
        {
            "chunk": "DDPM | FID=7.76 | Acc=43.8",
            "raw_chunk_text": "DDPM 7.76 43.8",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table1: FID of different generation models and their classifiers' accuracy.",
            "table_header": "Model FID Acc. (%)",
            "table_id": "Table1",
            "row_id": "DDPM",
            "page": 5,
            "similarity": 0.37,
            "numeric_table_priority": 7.8,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
            "table_focus_columns": ["FID", "Acc"],
        },
        {
            "chunk": "CBDM (τ=1) | FID=5.86 | Acc=46.6",
            "raw_chunk_text": "CBDM (τ=1) 5.86 46.6",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table1: FID of different generation models and their classifiers' accuracy.",
            "table_header": "Model FID Acc. (%)",
            "table_id": "Table1",
            "row_id": "CBDM (τ=1)",
            "page": 5,
            "similarity": 0.41,
            "numeric_table_priority": 8.4,
            "numeric_table_anchor_hits": ["Table 1", "FID", "Acc"],
            "table_focus_columns": ["FID", "Acc"],
        },
    ]

    ordered = _prioritize_numeric_table_results(results, query)
    final = _ensure_numeric_table_evidence_slots(ordered, query, top_k=3)
    focused = _build_query_focused_table_row(final[0], hints)

    assert final[0].get("row_id") == "CBDM (τ=1)"
    assert focused["text"] == "CBDM (τ=1) | FID=5.86 | Acc=46.6"


def test_numeric_table_priority_prefers_highest_gain_row_for_table3_alias():
    query = "In Table 3, which generated sample type has the largest average gain per sample, and what are its ΔAcc/||D_gen|| and accuracy values?"
    hints = QueryRewriter().extract_numeric_table_hints(query)
    results = [
        {
            "chunk": "Narrative summary about sample quality and gain per sample.",
            "raw_chunk_text": "Narrative summary about sample quality and gain per sample.",
            "chunk_type": "text",
            "block_type": "text",
            "page": 5,
            "similarity": 0.92,
            "numeric_table_priority": 11.4,
            "numeric_table_anchor_hits": ["Table 3", "AID"],
        },
        {
            "chunk": "ID | Acc=44.2 | ΔAcc/||D_gen||=-42.75×10^-4",
            "raw_chunk_text": "ID 21,511 44.2 -42.75×10^-4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table3: Quantities and classifier enhancement.",
            "table_header": "ID AID OOD ||D_gen|| Acc ΔAcc/||D_gen||",
            "table_id": "Table3",
            "row_id": "ID",
            "page": 5,
            "similarity": 0.33,
            "numeric_table_priority": 7.6,
            "numeric_table_anchor_hits": ["Table 3", "Acc"],
            "table_focus_columns": ["||D_gen||", "Acc", "ΔAcc/||D_gen||"],
        },
        {
            "chunk": "AID | Acc=45.2 | ΔAcc/||D_gen||=5.78×10^-4",
            "raw_chunk_text": "AID 11,886 45.2 5.78×10^-4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table3: Quantities and classifier enhancement.",
            "table_header": "ID AID OOD ||D_gen|| Acc ΔAcc/||D_gen||",
            "table_id": "Table3",
            "row_id": "AID",
            "page": 5,
            "similarity": 0.35,
            "numeric_table_priority": 7.5,
            "numeric_table_anchor_hits": ["Table 3", "AID"],
            "table_focus_columns": ["||D_gen||", "Acc", "ΔAcc/||D_gen||"],
        },
        {
            "chunk": "OOD | Acc=36.2 | ΔAcc/||D_gen||=-3.61×10^-4",
            "raw_chunk_text": "OOD 5,756 36.2 -3.61×10^-4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table3: Quantities and classifier enhancement.",
            "table_header": "ID AID OOD ||D_gen|| Acc ΔAcc/||D_gen||",
            "table_id": "Table3",
            "row_id": "OOD",
            "page": 5,
            "similarity": 0.32,
            "numeric_table_priority": 7.4,
            "numeric_table_anchor_hits": ["Table 3", "Acc"],
            "table_focus_columns": ["||D_gen||", "Acc", "ΔAcc/||D_gen||"],
        },
    ]

    ordered = _prioritize_numeric_table_results(results, query)
    final = _ensure_numeric_table_evidence_slots(ordered, query, top_k=3)
    focused = _build_query_focused_table_row(final[0], hints)

    assert final[0].get("row_id") == "AID"
    assert focused["text"] == "AID | ||D_gen||=11,886 | Acc=45.2 | ΔAcc/||D_gen||=5.78×10^-4"


def test_numeric_table_cost_query_prefers_discussion_cost_anchor_over_result_tables():
    query = "DiffuLT 的推理开销（额外 FLOPs 或推理时间）相比基线增加了多少？"
    results = [
        {
            "chunk": (
                "Our method only modifies the training data, so inference adds no extra overhead. "
                "Training time is about 24 hours on CIFAR100-LT and 6 days on ImageNet-LT. "
                "This discussion clarifies the cost profile."
            ),
            "raw_chunk_text": (
                "Our method only modifies the training data, so inference adds no extra overhead. "
                "Training time is about 24 hours on CIFAR100-LT and 6 days on ImageNet-LT. "
                "This discussion clarifies the cost profile."
            ),
            "chunk_type": "text",
            "block_type": "text",
            "page": 12,
            "similarity": 0.46,
            "numeric_table_priority": 2.0,
            "numeric_table_anchor_hits": ["training time"],
        },
        {
            "chunk": "Table 13 baseline results with many numeric values.",
            "raw_chunk_text": "Table 13 baseline results with many numeric values.",
            "chunk_type": "table",
            "block_type": "table",
            "page": 12,
            "similarity": 0.96,
            "numeric_table_priority": 12.0,
            "numeric_table_anchor_hits": ["Table 13", "Baseline"],
        },
        {
            "chunk": "Table 8 | DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            "raw_chunk_text": "Table 8 | DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "table_id": "Table 8",
            "row_id": "DiffuLT",
            "page": 12,
            "similarity": 0.94,
            "numeric_table_priority": 11.4,
            "numeric_table_anchor_hits": ["Table 8", "DiffuLT", "All"],
            "table_focus_columns": ["All", "Many", "Med.", "Few"],
        },
    ]

    boosted = _apply_query_intent_boost(results, query)
    ordered = _prioritize_numeric_table_results(boosted, query)

    assert ordered[0].get("chunk_type") == "text"
    assert "24 hours" in ordered[0].get("chunk", "").lower()


def test_numeric_table_cost_query_demotes_generic_training_time_narrative_without_explicit_duration():
    query = "DiffuLT 的推理开销（额外 FLOPs 或推理时间）相比基线增加了多少？"
    results = [
        _make_numeric_candidate(
            (
                "Figure 4 summarizes the pipeline. "
                "Our method requires more training time, typically four times longer, "
                "but this paragraph does not quantify the actual duration."
            ),
            0.96,
            page=7,
            chunk_type="text",
            block_type="text",
        ),
        _make_numeric_candidate(
            (
                "B.2 Limitation. The primary limitation is the extensive training time required "
                "for the generative model. For instance, training a diffusion model on "
                "CIFAR100-LT takes 24 hours, while ImageNet-LT requires approximately six days."
            ),
            0.24,
            page=18,
            chunk_type="text",
            block_type="text",
        ),
    ]
    config = SimpleNamespace(
        enable_focus_mode=False,
        focus_mode_window_size=1,
        focus_mode_max_sentences=2,
        focus_mode_min_chars=80,
    )

    final = _finalize_without_rerank(results, query, top_k=1, config=config)

    assert final[0].get("page") == 18
    assert "24 hours" in (final[0].get("chunk") or "").lower()


def test_numeric_table_cost_query_augments_global_cost_anchor_page():
    query = "DiffuLT 的推理开销（额外 FLOPs 或推理时间）相比基线增加了多少？"
    results = [
        _make_numeric_candidate(
            "Table 8 | DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            0.94,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="DiffuLT",
            table_focus_columns=["All", "Many", "Med.", "Few"],
        )
    ]

    augmented = _augment_with_table_chunks(
        results,
        chunks=[],
        pages=[
            {"page": 9, "text": "Table 8 | DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4"},
            {
                "page": 18,
                "text": (
                    "B.2 Limitation. Our method adds no extra overhead at inference time. "
                    "Training time is about 24 hours on CIFAR100-LT and approximately six days on ImageNet-LT."
                ),
            },
        ],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
    )

    cost_chunks = [
        item for item in augmented
        if item.get("table_augmented_scope") == "page_global_anchor"
        and "24 hours" in (item.get("chunk") or "").lower()
    ]
    assert cost_chunks


def test_numeric_table_cost_query_reserves_ocr_appendix_anchor_when_augment_budget_is_tight():
    query = "DiffuLT 的推理开销（额外 FLOPs 或推理时间）相比基线增加了多少？"
    results = [
        _make_numeric_candidate(
            "Table 8 | DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            0.94,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="DiffuLT",
            table_focus_columns=["All", "Many", "Med.", "Few"],
        )
    ]

    augmented = _augment_with_table_chunks(
        results,
        chunks=[],
        pages=[
            {
                "page": 10,
                "text": (
                    "Discussion of DiffuLT runtime tradeoffs and future work. "
                    "This section describes the cost profile in narrative form."
                ),
            },
            {
                "page": 18,
                "text": (
                    "B.2 Limitation. The primary limitation is the extensive training time required "
                    "for the generative model. For instance, training a diffusion model on "
                    "CIFAR100-LT takes24hours, while ImageNet-LT requires approximatelysixdays. "
                    "Our method adds no extra overhead at inference time."
                ),
            },
        ],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=1,
    )

    appended = augmented[len(results):]
    assert len(appended) == 1
    compact = (appended[0].get("chunk") or "").lower().replace(" ", "")
    assert appended[0].get("page") == 18
    assert "24hours" in compact
    assert "sixdays" in compact




def test_numeric_table_cleanup_collapses_explicit_comparator_rows_into_single_bundle():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    results = [
        _make_numeric_candidate(
            "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            0.96,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="DiffuLT",
            table_focus_backbone="ResNet-50",
            table_focus_columns=["All", "Many", "Med.", "Few"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DiffuLT 56.4 63.3 55.6 39.4",
            table_row_raw_text="DiffuLT 56.4 63.3 55.6 39.4",
            raw_chunk_text="DiffuLT 56.4 63.3 55.6 39.4",
        ),
        _make_numeric_candidate(
            "cRT | ResNet-50 | All=47.3 | Many=58.8 | Med.=44.0 | Few=26.1",
            0.95,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="cRT",
            table_focus_backbone="ResNet-50",
            table_focus_columns=["All", "Many", "Med.", "Few"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="cRT 47.3 58.8 44.0 26.1",
            table_row_raw_text="cRT 47.3 58.8 44.0 26.1",
            raw_chunk_text="cRT 47.3 58.8 44.0 26.1",
        ),
        _make_numeric_candidate(
            "RIDE(3 experts) | ResNet-50 | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9",
            0.94,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="RIDE(3 experts)",
            table_focus_backbone="ResNet-50",
            table_focus_columns=["All", "Many", "Med.", "Few"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="RIDE(3 experts) 54.9 66.2 51.7 34.9",
            table_row_raw_text="RIDE(3 experts) 54.9 66.2 51.7 34.9",
            raw_chunk_text="RIDE(3 experts) 54.9 66.2 51.7 34.9",
        ),
        _make_numeric_candidate(
            "ADRW | ResNet-50 | All=54.1 | Many=62.9 | Med.=52.6 | Few=37.1",
            0.93,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="ADRW",
            table_focus_backbone="ResNet-50",
            table_focus_columns=["All", "Many", "Med.", "Few"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="ADRW 54.1 62.9 52.6 37.1",
            table_row_raw_text="ADRW 54.1 62.9 52.6 37.1",
            raw_chunk_text="ADRW 54.1 62.9 52.6 37.1",
        ),
    ]

    fallback_entries = [
        (item, _build_context_text_for_result(item, query))
        for item in results
    ]

    layered = _cleanup_numeric_table_context_entries(fallback_entries, query)

    assert len(layered) == 1
    assert layered[0]["context_role"] == "anchor"
    bundle_text = layered[0]["text"]
    assert bundle_text.count("Table 8: Results on ImageNet-LT.") == 1
    assert bundle_text.count("ResNet-10 ResNet-50 All All Many Med. Few") == 1
    assert "DiffuLT 56.4" in bundle_text
    assert "cRT 47.3" in bundle_text
    assert "RIDE(3 experts) 54.9" in bundle_text
    assert "ADRW 54.1" in bundle_text



def test_numeric_table_same_bundle_hard_gate_rejects_wrong_table_for_few_shot_winner_query():
    query = "实验结果表中 Few-shot 子集表现最好的方法是谁？在 CIFAR100-LT 的 ResNet-32 上数值是多少？"
    results = [
        _make_numeric_candidate(
            "CBDM (τ=1) | FID=5.86 | Acc=46.6",
            0.96,
            page=5,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 1: FID of different generation models and their classifiers' accuracy.",
            table_header="Model FID Acc.(%)",
            table_id="Table 1",
            row_id="CBDM (τ=1)",
            table_focus_columns=["FID", "Acc"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="CBDM (τ=1) 5.86 46.6",
            table_row_raw_text="CBDM (τ=1) 5.86 46.6",
            raw_chunk_text="CBDM (τ=1) 5.86 46.6",
        ),
        _make_numeric_candidate(
            "DiffuLT | ResNet-32 | Many=69.0 | Med.=51.6 | Few=29.7",
            0.74,
            page=8,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            table_header="Method | ResNet-32 | Many | Med. | Few",
            table_id="Table 7",
            row_id="DiffuLT",
            table_focus_backbone="ResNet-32",
            table_focus_columns=["Many", "Med.", "Few"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DiffuLT 69.0 51.6 29.7",
            table_row_raw_text="DiffuLT 69.0 51.6 29.7",
            raw_chunk_text="DiffuLT 69.0 51.6 29.7",
        ),
        _make_numeric_candidate(
            "RIDE | ResNet-32 | Many=67.4 | Med.=49.5 | Few=26.4",
            0.73,
            page=8,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            table_header="Method | ResNet-32 | Many | Med. | Few",
            table_id="Table 7",
            row_id="RIDE",
            table_focus_backbone="ResNet-32",
            table_focus_columns=["Many", "Med.", "Few"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="RIDE 67.4 49.5 26.4",
            table_row_raw_text="RIDE 67.4 49.5 26.4",
            raw_chunk_text="RIDE 67.4 49.5 26.4",
        ),
        _make_numeric_candidate(
            "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            0.48,
            page=8,
            chunk_type="caption",
            block_type="caption",
            table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            table_header="Method | ResNet-32 | Many | Med. | Few",
            table_id="Table 7",
            raw_chunk_text="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
        ),
    ]

    filtered = _apply_numeric_table_same_bundle_hard_gate(results, query)

    row_ids = [item.get("row_id") for item in filtered if item.get("chunk_type") == "table_row"]
    joined = " ".join(_build_context_text_for_result(item) for item in filtered)

    assert row_ids == ["DiffuLT", "RIDE"]
    assert "CBDM (τ=1)" not in joined
    assert "Table 1: FID of different generation models" not in joined
    assert "ResNet-32" in joined
    assert "29.7" in joined
    assert "26.4" in joined




def test_numeric_table_same_bundle_hard_gate_rejects_wrong_backbone_for_few_shot_winner_query():
    query = "CIFAR100-LT 上 ResNet-32 的 Few-shot 子集中，哪个方法最好？具体数值是多少？"
    results = [
        _make_numeric_candidate(
            "DiffuLT | ResNet-50 | Many=71.1 | Med.=53.8 | Few=31.9",
            0.95,
            page=8,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            table_header="Method | ResNet-50 | Many | Med. | Few",
            table_id="Table 7",
            row_id="DiffuLT",
            table_focus_backbone="ResNet-50",
            table_focus_columns=["Many", "Med.", "Few"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DiffuLT 71.1 53.8 31.9",
            table_row_raw_text="DiffuLT 71.1 53.8 31.9",
            raw_chunk_text="DiffuLT 71.1 53.8 31.9",
        ),
        _make_numeric_candidate(
            "DiffuLT | ResNet-32 | Many=69.0 | Med.=51.6 | Few=29.7",
            0.74,
            page=8,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            table_header="Method | ResNet-32 | Many | Med. | Few",
            table_id="Table 7",
            row_id="DiffuLT",
            table_focus_backbone="ResNet-32",
            table_focus_columns=["Many", "Med.", "Few"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DiffuLT 69.0 51.6 29.7",
            table_row_raw_text="DiffuLT 69.0 51.6 29.7",
            raw_chunk_text="DiffuLT 69.0 51.6 29.7",
        ),
        _make_numeric_candidate(
            "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            0.48,
            page=8,
            chunk_type="caption",
            block_type="caption",
            table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            table_header="Method | ResNet-32 | Many | Med. | Few",
            table_id="Table 7",
            raw_chunk_text="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
        ),
    ]

    filtered = _apply_numeric_table_same_bundle_hard_gate(results, query)

    row_ids = [item.get("row_id") for item in filtered if item.get("chunk_type") == "table_row"]
    joined = " ".join(_build_context_text_for_result(item) for item in filtered)

    assert row_ids == ["DiffuLT"]
    assert "ResNet-50" not in joined
    assert "ResNet-32" in joined
    assert "29.7" in joined





def test_unified_post_clean_keeps_appendix_cost_anchor_for_numeric_table_cost_query():
    query = "DiffuLT 的推理开销（额外 FLOPs 或推理时间）相比基线增加了多少？"
    results = [
        _make_numeric_candidate(
            "Table 8 | DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            0.96,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="DiffuLT",
            table_focus_backbone="ResNet-50",
            table_focus_columns=["All", "Many", "Med.", "Few"],
        ),
        _make_numeric_candidate(
            (
                "Appendix B limitation. Our method adds no extra overhead at inference time. "
                "Training time is about 24 hours on CIFAR100-LT and approximately six days on ImageNet-LT."
            ),
            0.18,
            page=18,
            chunk_type="text",
            block_type="text",
        ),
        _make_numeric_candidate(
            "Short title fragment",
            0.55,
            page=2,
            chunk_type="title",
            block_type="title",
        ),
    ]

    cleaned = _unified_post_clean(results, query, top_k=2)
    cleaned_text = " ".join((item.get("chunk") or "").lower() for item in cleaned)

    assert any("24 hours" in (item.get("chunk") or "").lower() for item in cleaned)
    assert "six days" in cleaned_text
    assert len(cleaned) == 2



def test_numeric_table_same_bundle_hard_gate_drops_wrong_table_rows_when_only_target_caption_survives():
    query = "实验结果表中 Few-shot 子集表现最好的方法是谁？在 CIFAR100-LT 的 ResNet-32 上数值是多少？"
    results = [
        _make_numeric_candidate(
            "CBDM (τ=1) | FID=5.86 | Acc=46.6",
            0.96,
            page=5,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 1: FID of different generation models and their classifiers' accuracy.",
            table_header="Model FID Acc.(%)",
            table_id="Table 1",
            row_id="CBDM (τ=1)",
            table_focus_columns=["FID", "Acc"],
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="CBDM (τ=1) 5.86 46.6",
            table_row_raw_text="CBDM (τ=1) 5.86 46.6",
            raw_chunk_text="CBDM (τ=1) 5.86 46.6",
        ),
        _make_numeric_candidate(
            "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            0.48,
            page=8,
            chunk_type="caption",
            block_type="caption",
            table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            table_header="Method | ResNet-32 | Many | Med. | Few",
            table_id="Table 7",
            raw_chunk_text="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
        ),
    ]

    filtered = _apply_numeric_table_same_bundle_hard_gate(results, query)

    assert [item.get("chunk_type") for item in filtered] == ["caption"]
    joined = " ".join(_build_context_text_for_result(item) for item in filtered)
    assert "Table 7" in joined
    assert "CBDM (τ=1)" not in joined
    assert "FID of different generation models" not in joined


def test_numeric_table_expansion_skips_generic_rows_without_exact_table_caption_match():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    chunk = (
        "final line of table1 shows gains over baseline. "
        "ID 2.75×10−4 40 AID 5.78×10−4 80 OOD −3.61×10−4 36.2"
    )
    results = [
        {
            "chunk": chunk,
            "raw_chunk_text": chunk,
            "chunk_type": "table",
            "block_type": "table",
            "page": 6,
            "similarity": 0.91,
            "similarity_percent": 91.0,
        }
    ]

    expanded = _expand_numeric_table_evidence_units(results, query)

    assert not any(item.get("chunk_type") == "table_row" for item in expanded[1:])
def test_search_document_chunks_preserves_low_rank_table_for_numeric_table_expansion(vector_store_dir):
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    synthetic_candidates = [
        {
            "chunk": f"实验叙述块 {idx}，总结 DiffuLT 在长尾分类上的整体收益。",
            "raw_chunk_text": f"实验叙述块 {idx}，总结 DiffuLT 在长尾分类上的整体收益。",
            "page": idx,
            "chunk_type": "text",
            "block_type": "text",
            "group_id": f"group-text-{idx}",
            "similarity": 0.95 - idx * 0.01,
            "similarity_percent": 95.0 - idx,
            "score": 0.95 - idx * 0.01,
            "snippet": f"实验叙述块 {idx}",
            "highlights": [],
            "reranked": False,
        }
        for idx in range(1, 9)
    ]
    synthetic_candidates.extend([
        {
            "chunk": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4"
            ),
            "raw_chunk_text": (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few "
                "cRT Kang et al. [2019] 41.8 47.3 58.8 44.0 26.1 "
                "RIDE (3 experts) Wang et al. [2020] 45.9 54.9 66.2 51.7 34.9 "
                "ADRW Wang et al. [2024b] - 54.1 62.9 52.6 37.1 "
                "DiffuLT 50.4 56.4 63.3 55.6 39.4"
            ),
            "page": 9,
            "chunk_type": "table",
            "block_type": "table",
            "group_id": "group-8",
            "chunk_heading": "Table 8",
            "section_path": "Experiments",
            "similarity": 0.52,
            "similarity_percent": 52.0,
            "score": 0.52,
            "snippet": "Table 8",
            "highlights": [],
            "reranked": False,
        },
        {
            "chunk": "Table 8: Results on ImageNet-LT.",
            "raw_chunk_text": "Table 8: Results on ImageNet-LT.",
            "page": 9,
            "chunk_type": "caption",
            "block_type": "caption",
            "group_id": "group-8",
            "chunk_heading": "Table 8",
            "section_path": "Experiments",
            "similarity": 0.5,
            "similarity_percent": 50.0,
            "score": 0.5,
            "snippet": "Table 8",
            "highlights": [],
            "reranked": False,
        },
    ])

    with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
         patch("services.embedding_service._query_vector_cache") as mock_cache, \
         patch("services.embedding_service._merge_with_group_search", side_effect=lambda **kwargs: kwargs["chunk_results"]), \
         patch("services.embedding_service._augment_with_table_chunks", side_effect=lambda *_args, **_kwargs: synthetic_candidates), \
         patch("services.embedding_service._apply_query_intent_boost", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._apply_numeric_table_boost", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._filter_reference_pollution", side_effect=lambda results, _query, evidence_need=None: results), \
         patch("services.chunk_expander.expand_context_chunks", side_effect=lambda results, *_args, **_kwargs: results):

        mock_cache.get.return_value = None
        results, _timings = search_document_chunks(
            doc_id="rerank-order-doc",
            query=query,
            vector_store_dir=vector_store_dir,
            pages=[{"page": 9, "text": "Table 8 page"}],
            top_k=2,
            use_hybrid=False,
            use_rerank=False,
        )

    assert len(results) == 2
    assert any(item.get("chunk_type") == "table_row" for item in results)
    assert results[0].get("chunk_type") == "table_row"


def test_search_document_chunks_numeric_table_uses_wider_candidate_pool(vector_store_dir):
    query = "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？"
    search_ks = []

    class FakeIndex:
        ntotal = 64
        metric_type = faiss.METRIC_INNER_PRODUCT

        def search(self, _vectors, k):
            search_ks.append(k)
            distances = np.linspace(0.99, 0.4, num=k, dtype="float32").reshape(1, k)
            indices = (np.arange(k, dtype="int64") % 4).reshape(1, k)
            return distances, indices

    fake_data = {
        "chunks": [
            "Figure 3 narrative block.",
            "Appendix note.",
            "Table 3 exact row block.",
            "Background paragraph.",
        ],
        "chunk_headings": ["Figure 3", "Appendix", "Table 3", "Background"],
        "chunk_pages": [5, 6, 5, 2],
        "chunk_types": ["text", "text", "table", "text"],
        "embedding_model": "local-minilm",
        "parent_chunks": [],
        "child_to_parent": {},
    }

    with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
         patch.object(search_document_chunks.__globals__["_index_cache"], "get_index", return_value=(FakeIndex(), fake_data)), \
         patch.object(search_document_chunks.__globals__["_query_vector_cache"], "get", return_value=None), \
         patch.object(search_document_chunks.__globals__["_query_vector_cache"], "put", return_value=None), \
         patch("services.embedding_service._merge_with_group_search", side_effect=lambda **kwargs: kwargs["chunk_results"]), \
         patch("services.embedding_service._augment_with_table_chunks", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.embedding_service._apply_query_intent_boost", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._apply_numeric_table_boost", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._filter_reference_pollution", side_effect=lambda results, _query, evidence_need=None: results), \
         patch("services.embedding_service._unified_post_clean", side_effect=lambda results, _query, _top_k: results), \
         patch("services.embedding_service._annotate_results_for_evidence_rerank", side_effect=lambda **kwargs: kwargs["results"]), \
         patch("services.embedding_service._expand_numeric_table_evidence_units", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.embedding_service._mark_numeric_table_support_chunks", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._dedupe_numeric_table_evidence_units", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._sanitize_by_chunk_type", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._finalize_with_optional_rerank", side_effect=lambda **kwargs: kwargs["results"][: kwargs["top_k"]]), \
         patch("services.chunk_expander.expand_context_chunks", side_effect=lambda results, *_args, **_kwargs: results):

        search_document_chunks(
            doc_id="rerank-order-doc",
            query=query,
            vector_store_dir=vector_store_dir,
            pages=[{"page": 5, "text": "Table 3 page"}],
            top_k=3,
            use_hybrid=False,
            use_rerank=False,
        )

    assert search_ks
    assert min(search_ks) >= 48


def test_search_document_chunks_non_rerank_same_bundle_gate(vector_store_dir):
    table8_caption = "Table 8: Results on ImageNet-LT."
    table8_header = "ResNet-10 ResNet-50 All All Many Med. Few"
    table1_caption = "Table 1: FID of different generation models and their classifiers' accuracy."
    table1_header = "Model FID Acc. (%)"
    table3_caption = "Table 3: Quantities and classifier enhancement."
    table3_header = "ID AID OOD Acc ΔAcc/||D_gen||"

    cases = [
        {
            "name": "q1_second_best_bundle",
            "query": "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？",
            "top_k": 5,
            "candidates": [
                _make_numeric_candidate(
                    "DiffuLT summary paragraph about bundle-level gains on the page.",
                    0.97,
                    page=9,
                    chunk_type="text",
                    block_type="text",
                    table_augmented_scope="page_content",
                    numeric_table_priority=13.2,
                    numeric_table_anchor_hits=["Table 8", "ImageNet-LT", "ResNet-50", "DiffuLT", "Many"],
                ),
                _make_numeric_candidate(
                    "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
                    0.49,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="DiffuLT",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
                    table_row_raw_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
                    raw_chunk_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
                    numeric_table_priority=12.8,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "DiffuLT", "Many", "Med.", "Few"],
                ),
                _make_numeric_candidate(
                    "cRT | ResNet-50 | All=47.3 | Many=58.8 | Med.=44.0 | Few=26.1",
                    0.47,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="cRT",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="cRT 41.8 47.3 58.8 44.0 26.1",
                    table_row_raw_text="cRT 41.8 47.3 58.8 44.0 26.1",
                    raw_chunk_text="cRT 41.8 47.3 58.8 44.0 26.1",
                    numeric_table_priority=12.6,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "cRT", "Many", "Med.", "Few"],
                ),
                _make_numeric_candidate(
                    "RIDE (3 experts) | ResNet-50 | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9",
                    0.46,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="RIDE (3 experts)",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
                    table_row_raw_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
                    raw_chunk_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
                    numeric_table_priority=12.5,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "RIDE (3 experts)", "Many", "Med.", "Few"],
                ),
                _make_numeric_candidate(
                    "ADRW | ResNet-50 | All=54.1 | Many=62.9 | Med.=52.6 | Few=37.1",
                    0.45,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="ADRW",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="ADRW 54.1 62.9 52.6 37.1",
                    table_row_raw_text="ADRW 54.1 62.9 52.6 37.1",
                    raw_chunk_text="ADRW 54.1 62.9 52.6 37.1",
                    numeric_table_priority=12.4,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "ADRW", "Many", "Med.", "Few"],
                ),
                _make_numeric_candidate(
                    table8_caption,
                    0.42,
                    page=9,
                    chunk_type="caption",
                    block_type="caption",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    raw_chunk_text=table8_caption,
                    numeric_table_priority=10.4,
                    numeric_table_anchor_hits=["Table 8", "ImageNet-LT"],
                ),
            ],
            "expected_rows": {"DiffuLT", "RIDE (3 experts)"},
            "forbidden": [
                "page_content",
                "summary paragraph",
                "DiffuLT(1)",
                "DiffuLT(2)",
                "DiffuLT(3)",
                "CE",
                "cRT",
                "ADRW",
                "DiffuLT + RIDE",
            ],
            "expect_support": False,
        },
        {
            "name": "q4_best_row",
            "query": "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？",
            "top_k": 3,
            "candidates": [
                _make_numeric_candidate(
                    "Narrative summary about improved FID and classifier accuracy.",
                    0.98,
                    page=5,
                    chunk_type="text",
                    block_type="text",
                    table_augmented_scope="page_content",
                    numeric_table_priority=13.2,
                    numeric_table_anchor_hits=["Table 1", "FID", "Acc"],
                ),
                _make_numeric_candidate(
                    "CBDM (τ=3) | FID=7.42 | Acc=44.8",
                    0.52,
                    page=5,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table1_caption,
                    table_header=table1_header,
                    table_id="Table 1",
                    row_id="CBDM (τ=3)",
                    table_focus_columns=["FID", "Acc"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="CBDM (τ=3) 7.42 44.8",
                    table_row_raw_text="CBDM (τ=3) 7.42 44.8",
                    raw_chunk_text="CBDM (τ=3) 7.42 44.8",
                    numeric_table_priority=12.1,
                    numeric_table_anchor_hits=["Table 1", "FID", "Acc"],
                ),
                _make_numeric_candidate(
                    "CBDM (τ=1) | FID=5.86 | Acc=46.6",
                    0.48,
                    page=5,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table1_caption,
                    table_header=table1_header,
                    table_id="Table 1",
                    row_id="CBDM (τ=1)",
                    table_focus_columns=["FID", "Acc"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="CBDM (τ=1) 5.86 46.6",
                    table_row_raw_text="CBDM (τ=1) 5.86 46.6",
                    raw_chunk_text="CBDM (τ=1) 5.86 46.6",
                    numeric_table_priority=11.8,
                    numeric_table_anchor_hits=["Table 1", "FID", "Acc"],
                ),
                _make_numeric_candidate(
                    "CBDM (τ=2) | FID=6.82 | Acc=46.0",
                    0.47,
                    page=5,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table1_caption,
                    table_header=table1_header,
                    table_id="Table 1",
                    row_id="CBDM (τ=2)",
                    table_focus_columns=["FID", "Acc"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="CBDM (τ=2) 6.82 46.0",
                    table_row_raw_text="CBDM (τ=2) 6.82 46.0",
                    raw_chunk_text="CBDM (τ=2) 6.82 46.0",
                    numeric_table_priority=11.7,
                    numeric_table_anchor_hits=["Table 1", "FID", "Acc"],
                ),
                _make_numeric_candidate(
                    "DDPM | FID=7.76 | Acc=43.8",
                    0.46,
                    page=5,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table1_caption,
                    table_header=table1_header,
                    table_id="Table 1",
                    row_id="DDPM",
                    table_focus_columns=["FID", "Acc"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="DDPM 7.76 43.8",
                    table_row_raw_text="DDPM 7.76 43.8",
                    raw_chunk_text="DDPM 7.76 43.8",
                    numeric_table_priority=11.6,
                    numeric_table_anchor_hits=["Table 1", "FID", "Acc"],
                ),
                _make_numeric_candidate(
                    table1_caption,
                    0.43,
                    page=5,
                    chunk_type="caption",
                    block_type="caption",
                    table_caption=table1_caption,
                    table_header=table1_header,
                    table_id="Table 1",
                    raw_chunk_text=table1_caption,
                    numeric_table_priority=10.3,
                    numeric_table_anchor_hits=["Table 1", "FID", "Acc"],
                ),
                _make_numeric_candidate(
                    "Table 1: FID of different generation models and their classifiers' accuracy.",
                    0.41,
                    page=5,
                    chunk_type="table",
                    block_type="table",
                    table_caption=table1_caption,
                    table_header=table1_header,
                    table_id="Table 1",
                    raw_chunk_text="Table 1: FID of different generation models and their classifiers' accuracy.",
                    numeric_table_priority=10.1,
                    numeric_table_anchor_hits=["Table 1", "FID", "Acc"],
                ),
            ],
            "expected_rows": {"CBDM (τ=1)"},
            "forbidden": ["CBDM (τ=3)", "CBDM (τ=2)", "DDPM", "Narrative summary"],
            "expect_support": True,
        },
        {
            "name": "q5_exact_cell",
            "query": "In Table 3, which generated sample type has the largest average gain per sample, and what are its ΔAcc/||D_gen|| and accuracy values?",
            "top_k": 3,
            "candidates": [
                _make_numeric_candidate(
                    "Figure 3 explanation mixed with appendix statistics.",
                    0.97,
                    page=5,
                    chunk_type="text",
                    block_type="text",
                    table_augmented_scope="page_content",
                    numeric_table_priority=11.4,
                    numeric_table_anchor_hits=["Table 3", "AID"],
                ),
                _make_numeric_candidate(
                    "CE | Acc=44.1 | ΔAcc/||D_gen||=5.10×10^-4",
                    0.52,
                    page=5,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table3_caption,
                    table_header=table3_header,
                    table_id="Table 3",
                    row_id="CE",
                    table_focus_columns=["ΔAcc/||D_gen||", "Acc"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="CE 44.1 5.10×10^-4",
                    table_row_raw_text="CE 44.1 5.10×10^-4",
                    raw_chunk_text="CE 44.1 5.10×10^-4",
                    numeric_table_priority=10.0,
                    numeric_table_anchor_hits=["Table 3", "AID"],
                ),
                _make_numeric_candidate(
                    "AID | Acc=45.2 | ΔAcc/||D_gen||=5.78×10^-4",
                    0.48,
                    page=5,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table3_caption,
                    table_header=table3_header,
                    table_id="Table 3",
                    row_id="AID",
                    table_focus_columns=["ΔAcc/||D_gen||", "Acc"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="AID 45.2 5.78×10^-4",
                    table_row_raw_text="AID 45.2 5.78×10^-4",
                    raw_chunk_text="AID 45.2 5.78×10^-4",
                    numeric_table_priority=11.2,
                    numeric_table_anchor_hits=["Table 3", "AID"],
                ),
                _make_numeric_candidate(
                    table3_caption,
                    0.44,
                    page=5,
                    chunk_type="caption",
                    block_type="caption",
                    table_caption=table3_caption,
                    table_header=table3_header,
                    table_id="Table 3",
                    raw_chunk_text=table3_caption,
                    numeric_table_priority=10.2,
                    numeric_table_anchor_hits=["Table 3", "AID"],
                ),
                _make_numeric_candidate(
                    "Table 3: Quantities and classifier enhancement.",
                    0.42,
                    page=5,
                    chunk_type="table",
                    block_type="table",
                    table_caption=table3_caption,
                    table_header=table3_header,
                    table_id="Table 3",
                    raw_chunk_text="Table 3: Quantities and classifier enhancement.",
                    numeric_table_priority=10.0,
                    numeric_table_anchor_hits=["Table 3", "AID"],
                ),
            ],
            "expected_rows": {"AID"},
            "forbidden": ["Figure 3 explanation", "CE | Acc", "CE 44.1", "5.10×10^-4"],
            "expect_support": True,
        },
        {
            "name": "q7_explicit_comparator_rows",
            "query": "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？",
            "top_k": 6,
            "candidates": [
                _make_numeric_candidate(
                    "DiffuLT(1) 51.5 56.3 63.8 84.7 86.9 90.7",
                    0.97,
                    page=9,
                    chunk_type="text",
                    block_type="text",
                    table_augmented_scope="page_content",
                    numeric_table_priority=12.8,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "DiffuLT", "cRT"],
                ),
                _make_numeric_candidate(
                    "DiffuLT(2) 51.4 56.2 63.7 84.6 86.8 90.6",
                    0.96,
                    page=9,
                    chunk_type="text",
                    block_type="text",
                    table_augmented_scope="page_content",
                    numeric_table_priority=12.7,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "DiffuLT", "cRT"],
                ),
                _make_numeric_candidate(
                    "DiffuLT(3) 51.3 56.1 63.6 84.5 86.7 90.5",
                    0.95,
                    page=9,
                    chunk_type="text",
                    block_type="text",
                    table_augmented_scope="page_content",
                    numeric_table_priority=12.6,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "DiffuLT", "cRT"],
                ),
                _make_numeric_candidate(
                    "DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9",
                    0.54,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="DiffuLT + RIDE (3 experts)",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9",
                    table_row_raw_text="DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9",
                    raw_chunk_text="DiffuLT + RIDE (3 experts) 51.1 56.9 64.1 55.8 39.9",
                    numeric_table_priority=11.2,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "DiffuLT", "RIDE (3 experts)"],
                ),
                _make_numeric_candidate(
                    "CE | ResNet-50 | All=54.5 | Many=60.1 | Med.=50.2 | Few=31.2",
                    0.53,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="CE",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="CE 54.5 60.1 50.2 31.2",
                    table_row_raw_text="CE 54.5 60.1 50.2 31.2",
                    raw_chunk_text="CE 54.5 60.1 50.2 31.2",
                    numeric_table_priority=11.1,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "CE", "All"],
                ),
                _make_numeric_candidate(
                    "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
                    0.49,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="DiffuLT",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
                    table_row_raw_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
                    raw_chunk_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
                    numeric_table_priority=12.5,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "DiffuLT", "All"],
                ),
                _make_numeric_candidate(
                    "cRT | ResNet-50 | All=47.3 | Many=58.8 | Med.=44.0 | Few=26.1",
                    0.48,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="cRT",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="cRT 41.8 47.3 58.8 44.0 26.1",
                    table_row_raw_text="cRT 41.8 47.3 58.8 44.0 26.1",
                    raw_chunk_text="cRT 41.8 47.3 58.8 44.0 26.1",
                    numeric_table_priority=12.4,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "cRT", "All"],
                ),
                _make_numeric_candidate(
                    "RIDE (3 experts) | ResNet-50 | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9",
                    0.47,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="RIDE (3 experts)",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
                    table_row_raw_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
                    raw_chunk_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
                    numeric_table_priority=12.3,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "RIDE (3 experts)", "All"],
                ),
                _make_numeric_candidate(
                    "ADRW | ResNet-50 | All=54.1 | Many=62.9 | Med.=52.6 | Few=37.1",
                    0.46,
                    page=9,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    row_id="ADRW",
                    table_focus_backbone="ResNet-50",
                    table_focus_columns=["All", "Many", "Med.", "Few"],
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="ADRW 54.1 62.9 52.6 37.1",
                    table_row_raw_text="ADRW 54.1 62.9 52.6 37.1",
                    raw_chunk_text="ADRW 54.1 62.9 52.6 37.1",
                    numeric_table_priority=12.2,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "ADRW", "All"],
                ),
                _make_numeric_candidate(
                    table8_caption,
                    0.43,
                    page=9,
                    chunk_type="caption",
                    block_type="caption",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    raw_chunk_text=table8_caption,
                    numeric_table_priority=10.4,
                    numeric_table_anchor_hits=["Table 8", "ImageNet-LT"],
                ),
                _make_numeric_candidate(
                    "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
                    0.41,
                    page=9,
                    chunk_type="table",
                    block_type="table",
                    table_caption=table8_caption,
                    table_header=table8_header,
                    table_id="Table 8",
                    raw_chunk_text="Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few",
                    numeric_table_priority=10.2,
                    numeric_table_anchor_hits=["Table 8", "ResNet-50", "All"],
                ),
            ],
            "expected_rows": {"DiffuLT", "cRT", "RIDE (3 experts)", "ADRW"},
            "forbidden": ["DiffuLT(1)", "DiffuLT(2)", "DiffuLT(3)", "DiffuLT + RIDE (3 experts)", "CE | ResNet-50", "wrong-slice"],
            "expect_support": True,
        },
    ]

    for case in cases:
        def fake_merge_with_group_search(**_kwargs):
            return [dict(item) for item in case["candidates"]]

        with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
             patch("services.embedding_service._query_vector_cache") as mock_cache, \
             patch("services.embedding_service._merge_with_group_search", side_effect=fake_merge_with_group_search), \
             patch("services.embedding_service._augment_with_table_chunks", side_effect=lambda results, *_args, **_kwargs: results), \
             patch("services.embedding_service._apply_query_intent_boost", side_effect=lambda results, _query: results), \
             patch("services.embedding_service._apply_numeric_table_boost", side_effect=lambda results, _query: results), \
             patch("services.embedding_service._filter_reference_pollution", side_effect=lambda results, _query, evidence_need=None: results), \
             patch("services.embedding_service._unified_post_clean", side_effect=lambda results, *_args, **_kwargs: results), \
             patch("services.chunk_expander.expand_context_chunks", side_effect=lambda results, *_args, **_kwargs: results), \
             patch.object(
                 search_document_chunks.__globals__["_rag_config_singleton"],
                 "enable_conditional_rerank",
                 False,
                 create=True,
             ), \
             patch.object(
                 search_document_chunks.__globals__["_rag_config_singleton"],
                 "enable_focus_mode",
                 False,
                 create=True,
             ):
            mock_cache.get.return_value = None
            results, _timings = search_document_chunks(
                doc_id="rerank-order-doc",
                query=case["query"],
                vector_store_dir=vector_store_dir,
                pages=[{"page": 1, "text": "页面文本"}],
                top_k=case["top_k"],
                use_hybrid=False,
                use_rerank=False,
            )

        row_ids = {item.get("row_id") for item in results if item.get("row_id")}
        chunk_types = {item.get("chunk_type") for item in results}
        joined_text = " ".join(
            " ".join(str(item.get(key, "")) for key in ("chunk", "raw_chunk_text"))
            for item in results
        )

        assert row_ids == case["expected_rows"], case["name"]
        assert "text" not in chunk_types, case["name"]
        assert case["forbidden"] and not any(fragment in joined_text for fragment in case["forbidden"]), case["name"]
        if case["expect_support"]:
            assert chunk_types & {"table", "caption"}, case["name"]


def test_focus_mode_skips_numeric_table_support_chunks():
    query = "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？"
    original_chunk = (
        "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few. "
        "DiffuLT 50.4 56.4 63.3 55.6 39.4. This sentence only exists to exceed the focus-mode threshold."
    )
    results = [
        {
            "chunk": original_chunk,
            "raw_chunk_text": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
            "chunk_type": "table",
            "block_type": "table",
        }
    ]

    compressed = _focus_mode_compress(results, query, window_size=1, max_sentences=2, min_chars=40)

    assert compressed[0]["chunk"] == original_chunk
    assert "focus_compression_ratio" not in compressed[0]


def test_focus_mode_skips_numeric_table_cost_anchor_chunks():
    query = "这篇论文的额外开销、推理时间和 FLOPs 分别是多少？"
    original_chunk = (
        "Figure 4 summarizes the training pipeline and the ablation setup. "
        "It also repeats some broad discussion that would normally dominate focus compression. "
        "Our method only modifies the training data, so inference adds no extra overhead. "
        "Training time is about 24 hours on CIFAR100-LT and 6 days on ImageNet-LT."
    )
    results = [
        {
            "chunk": original_chunk,
            "raw_chunk_text": original_chunk,
            "chunk_type": "text",
            "block_type": "text",
        }
    ]

    compressed = _focus_mode_compress(results, query, window_size=1, max_sentences=2, min_chars=40)

    assert compressed[0]["chunk"] == original_chunk
    assert "focus_compression_ratio" not in compressed[0]


def test_plain_table_row_extraction_supports_fid_and_acc_rows():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    hints = QueryRewriter().extract_numeric_table_hints(query)
    text = (
        "Model FID Acc. (%) Model pID pAID pOOD "
        "Baseline - 38.3 DDPM 39.1 21.2 39.7 DDPM 7.76 43.8 "
        "CBDM () 38.6 29.1 32.3 CBDM () 7.42 44.8 "
        "CBDM () 6.82 46.0 CBDM () 5.86 46.6"
    )

    rows = _extract_plain_table_rows(text, hints)
    focused_rows = {
        _build_query_focused_table_row(row, hints)["text"]
        for row in rows
        if _build_query_focused_table_row(row, hints)["text"]
    }

    assert "DDPM | FID=7.76 | Acc=43.8" in focused_rows
    assert "CBDM () | FID=5.86 | Acc=46.6" in focused_rows


def test_plain_table_row_extraction_supports_scientific_notation_metrics():
    query = "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？"
    hints = QueryRewriter().extract_numeric_table_hints(query)
    text = (
        "Table 3: Quantities, overall classifier enhancement, and average improvement per sample. "
        "Group ||D_gen|| Acc. (%) ΔAcc/||D_gen|| "
        "ID 21,511 44.2 −42.75 × 10−4 "
        "AID 11,886 45.2 5.78 × 10−4 "
        "OOD 5,756 36.2 −3.61 × 10−4 "
        "Table 4: p_h pAID Acc_t 40 33.2 29.7"
    )

    rows = _extract_plain_table_rows(text, hints)
    focused_rows = {
        row["row_id"]: _build_query_focused_table_row(row, hints)["text"]
        for row in rows
        if _build_query_focused_table_row(row, hints)["text"]
    }

    assert focused_rows["AID"] == "AID | ||D_gen||=11,886 | Acc=45.2 | ΔAcc/||D_gen||=5.78 × 10−4"


def test_table_augment_uses_structured_bundle_chunk_metadata_for_numeric_queries():
    query = "Table 7 中 Ours 的 All 值是多少？"
    results = [
        {
            "chunk": "实验摘要：该页对多个模型做了整体比较。",
            "page": 2,
            "score": 0.81,
            "similarity": 0.81,
            "similarity_percent": 81.0,
            "snippet": "实验摘要",
            "highlights": [],
            "reranked": False,
        }
    ]
    structured_chunk = (
        "[Structured Table Bundle]\n\nTable 7: Main results\n\n"
        "[Body]\n| Method | All |\n| --- | --- |\n| Ours | 55.5 |"
    )

    augmented = _augment_with_table_chunks(
        results,
        chunks=[structured_chunk],
        pages=[{"page": 2, "text": "实验摘要：该页对多个模型做了整体比较。"}],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
        chunk_pages=[2],
        chunk_metadata=[
            {
                "structured_table_bundle": True,
                "table_bundle_id": "id:42",
                "table_id": "Table 7",
                "table_caption": "Table 7: Main results",
                "table_header": "Method | All",
                "page_range": [2, 2],
            }
        ],
    )

    structured = [
        item for item in augmented
        if item.get("table_augmented_scope") == "structured_bundle"
    ]
    assert structured
    assert structured[0]["page"] == 2
    assert structured[0]["table_id"] == "Table 7"
    assert structured[0]["structured_table_bundle"] is True


@pytest.mark.parametrize(
    ("query", "page_text", "structured_chunk", "structured_metadata", "expected_row", "expected_value"),
    [
        (
            "表 1 中 CBDM(τ=1) 的 FID 和准确率分别是多少？",
            (
                "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n"
                "Model FID Acc.(%)\n"
                "DDPM 7.76 43.8\n"
                "CBDM (τ=3) 7.42 44.8\n"
                "CBDM (τ=2) 6.82 46.0\n"
                "CBDM (τ=1) 5.86 46.6\n"
            ),
            (
                "[Structured Table Bundle]\n\nTable 1: FID of different generation models and their corresponding classifiers' accuracy.\n\n"
                "[Body]\n| Model | FID | Acc |\n| --- | --- | --- |\n| DDPM | 7.76 | 43.8 |\n"
            ),
            {
                "structured_table_bundle": True,
                "table_bundle_id": "id:table-1",
                "table_id": "Table 1",
                "table_caption": "Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
                "table_header": "Model FID Acc.(%)",
                "table_body_markdown": "| Model | FID | Acc |\n| --- | --- | --- |\n| DDPM | 7.76 | 43.8 |",
                "evidence_units": [],
            },
            "CBDM (τ=1)",
            "5.86",
        ),
        (
            "In Table 3, which generated sample type has the largest average gain per sample, and what are its ΔAcc/||D_gen|| and accuracy values?",
            (
                "Figure 3 explanation mixed with appendix statistics.\n"
                "Table 3: Quantities and classifier enhancement.\n"
                "Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||\n"
                "Baseline - 38.3\n"
                "ID 21,511 44.2 2.75×10−4\n"
                "AID 11,886 45.2 5.78×10−4\n"
                "OOD 5,756 36.2 −3.61×10−4\n"
            ),
            (
                "[Structured Table Bundle]\n\nTable 3: Quantities and classifier enhancement.\n\n"
                "[Body]\n| Group | ||D_gen|| | Acc | ΔAcc/||D_gen|| |\n| --- | --- | --- | --- |\n| Baseline | - | 38.3 | - |\n| ID | 21,511 | 44.2 | 2.75×10^-4 |\n"
            ),
            {
                "structured_table_bundle": True,
                "table_bundle_id": "id:table-3",
                "table_id": "Table 3",
                "table_caption": "Table 3: Quantities and classifier enhancement.",
                "table_header": "Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||",
                "table_body_markdown": "| Group | ||D_gen|| | Acc | ΔAcc/||D_gen|| |\n| --- | --- | --- | --- |\n| Baseline | - | 38.3 | - |\n| ID | 21,511 | 44.2 | 2.75×10^-4 |",
                "evidence_units": [],
            },
            "AID",
            "5.78",
        ),
    ],
)
def test_numeric_table_augment_recovers_page_content_for_sparse_structured_bundle(
    query,
    page_text,
    structured_chunk,
    structured_metadata,
    expected_row,
    expected_value,
):
    results = [
        _make_numeric_candidate(
            "Narrative summary about the table on this page.",
            0.98,
            page=4,
            chunk_type="text",
            block_type="text",
            table_augmented_scope="page_content",
            numeric_table_anchor_hits=[structured_metadata["table_id"], expected_row],
        )
    ]

    augmented = _augment_with_table_chunks(
        results,
        chunks=[structured_chunk],
        pages=[{"page": 4, "text": page_text}],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
        chunk_pages=[4],
        chunk_metadata=[structured_metadata],
    )

    recovered_items = [
        item for item in augmented
        if item.get("table_augmented_scope") in {"page_content", "structured_bundle"}
    ]
    assert any(
        expected_row in (
            " ".join(
                str(item.get(field) or "")
                for field in ("chunk", "raw_chunk_text", "table_body_markdown")
            )
        )
        and expected_value in (
            " ".join(
                str(item.get(field) or "")
                for field in ("chunk", "raw_chunk_text", "table_body_markdown")
            )
        )
        for item in recovered_items
    )


@pytest.mark.parametrize(
    ("query", "page", "page_text", "structured_chunk", "structured_metadata", "expected_tokens"),
    [
        (
            "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？",
            9,
            (
                "Table 8: Results on ImageNet-LT. ResNet-10 ResNet-50 All All Many Med. Few\n"
                "cRT 41.8 47.3 58.8 44.0 26.1\n"
                "RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9\n"
                "ADRW 54.1 62.9 52.6 37.1\n"
                "DiffuLT 50.4 56.4 63.3 55.6 39.4\n"
            ),
            (
                "[Structured Table Bundle]\n\nTable 8: Results on ImageNet-LT.\n\n"
                "[Body]\n| Method | All | Many | Med. | Few |\n| --- | --- | --- | --- | --- |\n"
                "| DiffuLT | 56.4 | 63.3 | 55.6 | 39.4 |"
            ),
            {
                "structured_table_bundle": True,
                "table_bundle_id": "id:table-8",
                "table_id": "Table 8",
                "table_caption": "Table 8: Results on ImageNet-LT.",
                "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
                "page_range": [9, 9],
                "table_pages": [9],
                "evidence_units": [
                    {
                        "evidence_unit_type": "table_row",
                        "row_id": "DiffuLT",
                        "content": "DiffuLT 50.4 56.4 63.3 55.6 39.4",
                    }
                ],
            },
            ("cRT", "ADRW", "DiffuLT"),
        ),
        (
            "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？",
            4,
            (
                "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n"
                "Model FID Acc.(%)\n"
                "DDPM 7.76 43.8\n"
                "CBDM (τ=3) 7.42 44.8\n"
                "CBDM (τ=2) 6.82 46.0\n"
                "CBDM (τ=1) 5.86 46.6\n"
            ),
            (
                "[Structured Table Bundle]\n\nTable 1: FID of different generation models and their corresponding classifiers' accuracy.\n\n"
                "[Body]\n| Model | FID | Acc |\n| --- | --- | --- |\n| CBDM(τ=3) | 7.42 | 44.8 |"
            ),
            {
                "structured_table_bundle": True,
                "table_bundle_id": "id:table-1-best",
                "table_id": "Table 1",
                "table_caption": "Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
                "table_header": "Model FID Acc.(%)",
                "page_range": [4, 4],
                "table_pages": [4],
                "evidence_units": [
                    {
                        "evidence_unit_type": "table_row",
                        "row_id": "CBDM (τ=3)",
                        "content": "CBDM (τ=3) 7.42 44.8",
                    }
                ],
            },
            ("CBDM (τ=1)", "5.86", "46.6"),
        ),
        (
            "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？",
            4,
            (
                "Table 3: Quantities and classifier enhancement.\n"
                "Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||\n"
                "ID 21,511 44.2 2.75×10−4\n"
                "AID 11,886 45.2 5.78×10−4\n"
                "OOD 5,756 36.2 −3.61×10−4\n"
            ),
            (
                "[Structured Table Bundle]\n\nTable 3: Quantities and classifier enhancement.\n\n"
                "[Body]\n| Group | ||D_gen|| | Acc | ΔAcc/||D_gen|| |\n| --- | --- | --- | --- |\n| ID | 21,511 | 44.2 | 2.75×10^-4 |"
            ),
            {
                "structured_table_bundle": True,
                "table_bundle_id": "id:table-3-best",
                "table_id": "Table 3",
                "table_caption": "Table 3: Quantities and classifier enhancement.",
                "table_header": "Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||",
                "page_range": [4, 4],
                "table_pages": [4],
                "evidence_units": [
                    {
                        "evidence_unit_type": "table_row",
                        "row_id": "ID",
                        "content": "ID 21,511 44.2 2.75×10−4",
                    }
                ],
            },
            ("AID", "45.2", "5.78×10−4"),
        ),
    ],
)
def test_numeric_table_augment_keeps_page_content_fallback_when_structured_bundle_rows_are_sparse(
    query,
    page,
    page_text,
    structured_chunk,
    structured_metadata,
    expected_tokens,
):
    results = [
        _make_numeric_candidate(
            "Narrative summary about the table on this page.",
            0.98,
            page=page,
            chunk_type="text",
            block_type="text",
            table_augmented_scope="page_content",
            numeric_table_anchor_hits=[structured_metadata["table_id"]],
        )
    ]

    augmented = _augment_with_table_chunks(
        results,
        chunks=[structured_chunk],
        pages=[{"page": page, "text": page_text}],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
        chunk_pages=[page],
        chunk_metadata=[structured_metadata],
    )

    page_content_items = [
        item for item in augmented if item.get("table_augmented_scope") == "page_content"
    ]
    assert any(
        all(token in ((item.get("chunk") or "") + " " + (item.get("raw_chunk_text") or "")) for token in expected_tokens)
        for item in page_content_items
    )


def test_numeric_table_augment_upgrades_sparse_structured_bundle_with_recovered_rows():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    page_text = (
        "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n"
        "Model FID Acc.(%)\n"
        "DDPM 7.76 43.8\n"
        "CBDM (τ=3) 7.42 44.8\n"
        "CBDM (τ=2) 6.82 46.0\n"
        "CBDM (τ=1) 5.86 46.6\n"
    )
    structured_chunk = (
        "[Structured Table Bundle]\n\n"
        "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n\n"
        "[Body]\n| Model | FID | Acc |\n| --- | --- | --- |\n| DDPM | 7.76 | 43.8 |"
    )
    structured_metadata = {
        "structured_table_bundle": True,
        "table_bundle_id": "id:table-1-sparse",
        "table_id": "Table 1",
        "table_caption": "Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
        "table_header": "Model FID Acc.(%)",
        "table_body_markdown": "| Model | FID | Acc |\n| --- | --- | --- |\n| DDPM | 7.76 | 43.8 |",
        "evidence_units": [],
    }

    augmented = _augment_with_table_chunks(
        [_make_numeric_candidate("Narrative summary about Table 1.", 0.91, page=4)],
        chunks=[structured_chunk],
        pages=[{"page": 4, "text": page_text}],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
        chunk_pages=[4],
        chunk_metadata=[structured_metadata],
    )

    structured_items = [
        item for item in augmented if item.get("table_augmented_scope") == "structured_bundle"
    ]
    assert structured_items
    assert structured_items[0]["sparse_table_bundle"] is True
    assert "CBDM (τ=1)" in structured_items[0].get("table_body_markdown", "")

    expanded = _expand_numeric_table_evidence_units(structured_items, query)
    row_items = [
        item for item in expanded if (item.get("chunk_type") or item.get("block_type")) == "table_row"
    ]
    assert any(
        item.get("row_id") == "CBDM (τ=1)"
        and "5.86" in ((item.get("chunk") or "") + " " + (item.get("raw_chunk_text") or ""))
        and "46.6" in ((item.get("chunk") or "") + " " + (item.get("raw_chunk_text") or ""))
        for item in row_items
    )


@pytest.mark.parametrize(
    ("query", "page", "page_text", "structured_chunk", "structured_metadata", "expected_row_id", "expected_tokens"),
    [
        (
            "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？",
            4,
            (
                "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n"
                "Model FID Acc.(%)\n"
                "DDPM 7.76 43.8\n"
                "CBDM (τ=3) 7.42 44.8\n"
                "CBDM (τ=2) 6.82 46.0\n"
                "CBDM (τ=1) 5.86 46.6\n"
            ),
            (
                "[Structured Table Bundle]\n\n"
                "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n\n"
                "[Body]\nTable1: FIDof different generationmodels"
            ),
            {
                "structured_table_bundle": True,
                "table_bundle_id": "id:table-1-existing-sparse",
                "table_id": "Table 1",
                "table_caption": "Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
                "table_header": "Model FID Acc.(%)",
                "table_body_markdown": "Table1: FIDof different generationmodels",
                "evidence_units": [],
            },
            "CBDM (τ=1)",
            ("5.86", "46.6"),
        ),
        (
            "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？",
            5,
            (
                "Table 3: Quantities and classifier enhancement.\n"
                "Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||\n"
                "ID 21,511 44.2 2.75×10−4\n"
                "AID 11,886 45.2 5.78×10−4\n"
                "OOD 5,756 36.2 −3.61×10−4\n"
            ),
            (
                "[Structured Table Bundle]\n\n"
                "Table 3: Quantities and classifier enhancement.\n\n"
                "[Body]\nTable3:Quantities,overallclassifierenhancement,and"
            ),
            {
                "structured_table_bundle": True,
                "table_bundle_id": "id:table-3-existing-sparse",
                "table_id": "Table 3",
                "table_caption": "Table 3: Quantities and classifier enhancement.",
                "table_header": "Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||",
                "table_body_markdown": "Table3:Quantities,overallclassifierenhancement,and",
                "evidence_units": [],
            },
            "AID",
            ("45.2", "5.78×10−4"),
        ),
        (
            "实验结果表中哪个方法在 Few-shot 子集上取得最高准确率？具体数值是多少？",
            8,
            (
                "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.\n"
                "Method Many Med. Few\n"
                "CSA 64.3 49.7 18.2\n"
                "RIDE (3 experts) 68.1 49.2 23.9\n"
                "DiffuLT 69.0 51.6 29.7\n"
            ),
            (
                "[Structured Table Bundle]\n\n"
                "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.\n\n"
                "[Body]\nTable7: ResultsonCIFAR100-LTandCIFAR10-LTdatase..."
            ),
            {
                "structured_table_bundle": True,
                "table_bundle_id": "id:table-7-existing-sparse",
                "table_id": "Table 7",
                "table_caption": "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
                "table_header": "Method Many Med. Few",
                "page_range": [8, 8],
                "table_pages": [8],
                "table_body_markdown": "Table7: ResultsonCIFAR100-LTandCIFAR10-LTdatase...",
                "evidence_units": [],
            },
            "DiffuLT",
            ("29.7",),
        ),
    ],
)
def test_numeric_table_augment_upgrades_existing_sparse_structured_bundle_result(
    query,
    page,
    page_text,
    structured_chunk,
    structured_metadata,
    expected_row_id,
    expected_tokens,
):
    results = [
        _make_numeric_candidate(
            structured_chunk,
            0.93,
            page=page,
            chunk_type="table",
            block_type="table",
            **structured_metadata,
        )
    ]

    augmented = _augment_with_table_chunks(
        results,
        chunks=[structured_chunk],
        pages=[{"page": page, "text": page_text}],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
        chunk_pages=[page],
        chunk_metadata=[structured_metadata],
    )

    structured_item = next(
        item
        for item in augmented
        if item.get("table_id") == structured_metadata["table_id"]
        and item.get("structured_table_bundle")
    )
    assert structured_item.get("sparse_table_bundle") is True
    row_units = [
        unit
        for unit in (structured_item.get("evidence_units") or [])
        if isinstance(unit, dict)
        and (unit.get("evidence_unit_type") or "").strip().lower() == "table_row"
    ]

    assert row_units
    expected_unit = next(
        unit for unit in row_units if unit.get("row_id") == expected_row_id
    )
    combined_text = " ".join(
        str(expected_unit.get(field) or "")
        for field in ("content", "row_text", "row_numbers")
    )
    assert all(token in combined_text for token in expected_tokens)


def test_numeric_table_augment_upgrades_sparse_structured_bundle_preserves_fewshot_focus():
    query = "实验结果表中哪个方法在 Few-shot 子集上取得最高准确率？具体数值是多少？"
    page_text = (
        "Table7: ResultsonCIFAR100-LTandCIFAR10-LTdatasets. Theimbalanceratiorissetto100,50\n"
        "and10. Thehighest-performingresultsareinbold,withthesecond-bestinunderline. Additionally,\n"
        "wepresenttheresultsfordifferentgroups(many,medium,andfew)inCIFAR100-LTwithr =100.\n"
        "CIFAR100-LT CIFAR10-LT Statistics\n"
        "Method\n"
        "100 50 10 100 50 10 Many Med. Few\n"
        "FocalLoss 38.4 44.3 55.8 70.4 76.7 86.7 65.3 38.4 8.1\n"
        "CSA 46.6 51.9 62.6 82.5 86.0 90.8 64.3 49.7 18.2\n"
        "RIDE (3 experts) 48.0 - - - - - 68.1 49.2 23.9\n"
        "DiffuLT 51.5 56.3 63.8 84.7 86.9 90.7 69.0 51.6 29.7\n"
    )
    structured_chunk = (
        "[Structured Table Bundle]\n\n"
        "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.\n\n"
        "[Body]\n| Method | Many | Med. |\n| --- | --- | --- |\n| FocalLoss | 65.3 | 38.4 |"
    )
    structured_metadata = {
        "structured_table_bundle": True,
        "table_bundle_id": "id:table-7-few-sparse",
        "table_id": "Table 7",
        "table_caption": "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
        "table_header": "Method | 100 | 50 | 10 | 100 | 50 | 10 | Many | Med. | Few",
        "table_body_markdown": "Table7:ResultsonCIFAR100-LTandCIFAR10-LTdatasets.",
        "evidence_units": [
            {
                "evidence_unit_type": "table_row",
                "row_id": "FocalLoss",
                "row_text": "FocalLoss | Many=86.7 | Med.=65.3",
                "content": "FocalLoss | Many=86.7 | Med.=65.3",
                "row_numbers": "86.7 65.3",
                "table_caption": "Table7:ResultsonCIFAR100-LTandCIFAR10-LTdatasets.",
                "table_header": "Method | 100 | 50 | 10 | 100 | 50 | 10 | Many | Med. | Few",
            },
            {
                "evidence_unit_type": "table_row",
                "row_id": "DiffuLT",
                "row_text": "DiffuLT | Many=69.0 | Med.=51.6",
                "content": "DiffuLT | Many=69.0 | Med.=51.6",
                "row_numbers": "69.0 51.6",
                "table_caption": "Table7:ResultsonCIFAR100-LTandCIFAR10-LTdatasets.",
                "table_header": "Method | 100 | 50 | 10 | 100 | 50 | 10 | Many | Med. | Few",
            },
        ],
    }

    augmented = _augment_with_table_chunks(
        [_make_numeric_candidate("Narrative summary about Table 7.", 0.91, page=8)],
        chunks=[structured_chunk],
        pages=[{"page": 8, "text": page_text}],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
        chunk_pages=[8],
        chunk_metadata=[structured_metadata],
    )

    structured_item = next(
        item
        for item in augmented
        if item.get("table_augmented_scope") == "structured_bundle"
        and item.get("table_id") == "Table 7"
    )
    row_units = [
        unit
        for unit in (structured_item.get("evidence_units") or [])
        if isinstance(unit, dict)
        and (unit.get("evidence_unit_type") or "").strip().lower() == "table_row"
    ]

    assert structured_item["table_caption"] == "Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets."
    assert structured_item["table_header"] == "Method | 100 | 50 | 10 | 100 | 50 | 10 | Many | Med. | Few"
    assert structured_item["sparse_table_bundle"] is True
    assert "29.7" in structured_item.get("table_body_markdown", "")
    assert "69.0" not in structured_item.get("table_body_markdown", "")
    assert any(unit.get("row_id") == "DiffuLT" for unit in row_units)
    assert any(unit.get("row_id") == "FocalLoss" for unit in row_units)


def test_numeric_table_augment_does_not_upgrade_dense_structured_bundle():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    page_text = (
        "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n"
        "Model FID Acc.(%)\n"
        "DDPM 7.76 43.8\n"
        "CBDM (τ=3) 7.42 44.8\n"
        "CBDM (τ=2) 6.82 46.0\n"
        "CBDM (τ=1) 5.86 46.6\n"
    )
    structured_chunk = (
        "[Structured Table Bundle]\n\n"
        "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n\n"
        "[Body]\n| Model | FID | Acc |\n| --- | --- | --- |\n"
        "| DDPM | 7.76 | 43.8 |\n"
        "| CBDM(τ=3) | 7.42 | 44.8 |\n"
        "| CBDM(τ=2) | 6.82 | 46.0 |\n"
        "| CBDM(τ=1) | 5.86 | 46.6 |"
    )
    structured_metadata = {
        "structured_table_bundle": True,
        "table_bundle_id": "id:table-1-dense",
        "table_id": "Table 1",
        "table_caption": "Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
        "table_header": "Model FID Acc.(%)",
        "table_body_markdown": (
            "| Model | FID | Acc |\n| --- | --- | --- |\n"
            "| DDPM | 7.76 | 43.8 |\n"
            "| CBDM(τ=3) | 7.42 | 44.8 |\n"
            "| CBDM(τ=2) | 6.82 | 46.0 |\n"
            "| CBDM(τ=1) | 5.86 | 46.6 |"
        ),
        "evidence_units": [],
    }

    augmented = _augment_with_table_chunks(
        [_make_numeric_candidate("Narrative summary about Table 1.", 0.91, page=4)],
        chunks=[structured_chunk],
        pages=[{"page": 4, "text": page_text}],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
        chunk_pages=[4],
        chunk_metadata=[structured_metadata],
    )

    structured_items = [
        item for item in augmented if item.get("table_augmented_scope") == "structured_bundle"
    ]
    assert structured_items
    assert not structured_items[0].get("sparse_table_bundle")


def test_structured_bundle_page_provenance_overrides_stale_chunk_page():
    item = {
        "page": 1,
        "structured_table_bundle": True,
        "table_id": "Table 3",
    }
    metadata = {
        "structured_table_bundle": True,
        "table_id": "Table 3",
        "page_range": [5, 5],
        "table_pages": [5],
        "page_index": 4,
        "page_uid": "page:5",
    }

    _apply_page_provenance(item, metadata)

    assert item["page"] == 5
    assert item["page_index"] == 4
    assert item["page_uid"] == "page:5"


def test_numeric_table_augment_uses_structured_bundle_page_provenance_when_chunk_page_is_stale():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    results = [
        _make_numeric_candidate(
            "Narrative summary about Table 1 on the correct page.",
            0.92,
            page=4,
            chunk_type="text",
            block_type="text",
            table_id="Table 1",
        )
    ]
    structured_chunk = (
        "[Structured Table Bundle]\n\n"
        "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n\n"
        "[Body]\n| Model | FID | Acc |\n| --- | --- | --- |\n| CBDM(τ=1) | 5.86 | 46.6 |"
    )
    structured_metadata = {
        "structured_table_bundle": True,
        "table_bundle_id": "manual:table1",
        "table_id": "Table 1",
        "table_caption": "Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
        "table_header": "Model | FID | Acc",
        "page_range": [4, 4],
        "table_pages": [4],
        "page_index": 3,
        "page_uid": "page:4",
    }

    augmented = _augment_with_table_chunks(
        results,
        chunks=[structured_chunk],
        pages=[{"page": 4, "text": "Table 1 page content with CBDM(τ=1) 5.86 46.6"}],
        page_index={},
        query=query,
        evidence_need=["numeric_table"],
        max_augment=3,
        chunk_pages=[1],
        chunk_metadata=[structured_metadata],
    )

    structured = [
        item for item in augmented
        if item.get("table_augmented_scope") == "structured_bundle"
        and item.get("table_id") == "Table 1"
    ]

    assert structured
    assert structured[0]["page"] == 4
    assert structured[0]["page_index"] == 3
    assert structured[0]["page_uid"] == "page:4"


def test_search_document_chunks_recovers_runtime_structured_bundle_from_pages(vector_store_dir):
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    page_text = (
        "Table 1: FID of different generation models and their corresponding classifiers' accuracy.\n"
        "Model FID Acc.(%)\n"
        "Baseline - 38.3\n"
        "DDPM 7.76 43.8\n"
        "CBDM (τ=3) 7.42 44.8\n"
        "CBDM (τ=2) 6.82 46.0\n"
        "CBDM (τ=1) 5.86 46.6\n"
    )

    def fake_merge_with_group_search(**_kwargs):
        return [
            _make_numeric_candidate(
                "Narrative summary about improved FID and classifier accuracy.",
                0.98,
                page=4,
                chunk_type="text",
                block_type="text",
                table_augmented_scope="page_content",
                numeric_table_priority=13.2,
                numeric_table_anchor_hits=["Table 1", "FID", "Acc"],
            )
        ]

    with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
         patch("services.embedding_service._query_vector_cache") as mock_cache, \
         patch("services.embedding_service._merge_with_group_search", side_effect=fake_merge_with_group_search), \
         patch("services.embedding_service._apply_query_intent_boost", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._apply_numeric_table_boost", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._filter_reference_pollution", side_effect=lambda results, _query, evidence_need=None: results), \
         patch("services.embedding_service._unified_post_clean", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.chunk_expander.expand_context_chunks", side_effect=lambda results, *_args, **_kwargs: results), \
         patch.object(
             search_document_chunks.__globals__["_rag_config_singleton"],
             "enable_conditional_rerank",
             False,
             create=True,
         ), \
         patch.object(
             search_document_chunks.__globals__["_rag_config_singleton"],
             "enable_focus_mode",
             False,
             create=True,
         ):
        mock_cache.get.return_value = None
        results, _timings = search_document_chunks(
            doc_id="rerank-order-doc",
            query=query,
            vector_store_dir=vector_store_dir,
            pages=[{"page": 4, "text": page_text}],
            top_k=3,
            use_hybrid=False,
            use_rerank=False,
        )

    joined_text = " ".join(
        " ".join(str(item.get(key, "")) for key in ("chunk", "raw_chunk_text"))
        for item in results
    )
    normalized_text = re.sub(r"\s+", " ", joined_text)

    assert any(item.get("table_bundle_id") == "page-text:table 1" for item in results)
    assert "5.86" in normalized_text
    assert "46.6" in normalized_text


def test_search_document_chunks_recovers_runtime_table3_bundle_from_side_by_side_page_text(vector_store_dir):
    query = "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？"
    page_text = (
        "Figure2:Visualizationofgeneratedsamplesforclass90infeaturespaceusingt-SNE.\n"
        "Table3:Quantities,overallclassifierenhancement,and Table4: Diffusiontrainedwithvarying\n"
        "averageimprovementpersamplefordifferentgroups proportionsofheadclassdataandthe\n"
        "ofdatageneratedbydiffusionmodel. correspondingresultsfortailclasses.\n"
        "Group ||D_gen|| Acc.(%) ΔAcc/||D_gen|| p h p AID Acc t(%)\n"
        "- - 25.0\n"
        "Baseline - 38.3\n"
        "0 25.8 26.0\n"
        "ID 21,511 44.2 2.75×10−4\n"
        "40 33.2 29.7\n"
        "AID 11,886 45.2 5.78×10−4 80 35.7 32.5\n"
        "OOD 5,756 36.2 −3.61×10−4 100 39.1 32.8\n"
    )

    def fake_merge_with_group_search(**_kwargs):
        return [
            _make_numeric_candidate(
                "Narrative summary about average improvement per sample.",
                0.98,
                page=5,
                chunk_type="text",
                block_type="text",
                table_augmented_scope="page_content",
                numeric_table_priority=11.4,
                numeric_table_anchor_hits=["Table 3", "AID"],
            )
        ]

    with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
         patch("services.embedding_service._query_vector_cache") as mock_cache, \
         patch("services.embedding_service._merge_with_group_search", side_effect=fake_merge_with_group_search), \
         patch("services.embedding_service._apply_query_intent_boost", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._apply_numeric_table_boost", side_effect=lambda results, _query: results), \
         patch("services.embedding_service._filter_reference_pollution", side_effect=lambda results, _query, evidence_need=None: results), \
         patch("services.embedding_service._unified_post_clean", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.chunk_expander.expand_context_chunks", side_effect=lambda results, *_args, **_kwargs: results), \
         patch.object(
             search_document_chunks.__globals__["_rag_config_singleton"],
             "enable_conditional_rerank",
             False,
             create=True,
         ), \
         patch.object(
             search_document_chunks.__globals__["_rag_config_singleton"],
             "enable_focus_mode",
             False,
             create=True,
         ):
        mock_cache.get.return_value = None
        results, _timings = search_document_chunks(
            doc_id="rerank-order-doc",
            query=query,
            vector_store_dir=vector_store_dir,
            pages=[{"page": 5, "text": page_text}],
            top_k=3,
            use_hybrid=False,
            use_rerank=False,
        )

    joined_text = " ".join(
        " ".join(str(item.get(key, "")) for key in ("chunk", "raw_chunk_text"))
        for item in results
    )
    normalized_text = re.sub(r"\s+", " ", joined_text)

    assert any(item.get("table_bundle_id") == "page-text:table 3" for item in results)
    assert any(
        item.get("chunk_type") == "table_row"
        and item.get("row_id") == "AID"
        and "ΔAcc/||D_gen||=5.78×10−4" in (item.get("chunk") or "")
        for item in results
    )
    assert "AID" in normalized_text
    assert "45.2" in normalized_text
    assert "5.78×10−4" in normalized_text


def test_numeric_table_expansion_keeps_table1_winner_from_mixed_page_text():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    page_text = (
        "Table1: FIDof different generationmodels Table2: Percentageofdifferenttypesofgener-\n"
        "andtheircorrespondingclassifiers’accuracy. atedsamplesforeachmodel.\n"
        "Model FID Acc.(%) Model p p p\n"
        "ID AID OOD\n"
        "Baseline - 38.3\n"
        "DDPM 39.1 21.2 39.7\n"
        "DDPM 7.76 43.8\n"
        "CBDM(τ =3) 38.6 29.1 32.3\n"
        "CBDM(τ =3) 7.42 44.8\n"
        "CBDM(τ =2) 6.82 46.0 CBDM(τ =2) 40.2 33.5 26.3\n"
        "CBDM(τ =1) 5.86 46.6 CBDM(τ =1) 44.8 36.3 18.9\n"
    )
    results = [
        _make_numeric_candidate(
            page_text,
            0.42,
            page=4,
            chunk_type="table",
            block_type="table",
            table_id="Table 1",
        )
    ]

    expanded = _expand_numeric_table_evidence_units(results, query)
    ordered = _prioritize_numeric_table_results(expanded, query)
    ordered = _apply_numeric_table_same_bundle_hard_gate(ordered, query)
    final = _ensure_numeric_table_evidence_slots(ordered, query, top_k=3)
    final = _apply_numeric_table_same_bundle_hard_gate(final, query)

    assert any(
        item.get("chunk_type") == "table_row"
        and item.get("row_id") == "CBDM(τ =1)"
        and "5.86" in (item.get("chunk") or "")
        and "46.6" in (item.get("chunk") or "")
        for item in final
    )
    assert not any(
        item.get("chunk_type") == "table_row"
        and item.get("row_id") == "CBDM(τ =3)"
        and "7.42" in (item.get("chunk") or "")
        for item in final
    )


def test_numeric_table_expansion_keeps_table3_aid_row_from_mixed_page_text():
    query = "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？"
    page_text = (
        "Figure2:Visualizationofgeneratedsamplesforclass90infeaturespaceusingt-SNE.\n"
        "Table3:Quantities,overallclassifierenhancement,and Table4: Diffusiontrainedwithvarying\n"
        "averageimprovementpersamplefordifferentgroups proportionsofheadclassdataandthe\n"
        "ofdatageneratedbydiffusionmodel. correspondingresultsfortailclasses.\n"
        "Group ∥D gen∥ Acc.(%) ∆Acc/∥D gen∥ p h p AID Acc t(%)\n"
        "- - 25.0\n"
        "Baseline - 38.3\n"
        "0 25.8 26.0\n"
        "ID 21,511 44.2 2.75×10−4\n"
        "40 33.2 29.7\n"
        "AID 11,886 45.2 5.78×10−4 80 35.7 32.5\n"
        "OOD 5,756 36.2 −3.61×10−4 100 39.1 32.8\n"
    )
    results = [
        _make_numeric_candidate(
            page_text,
            0.42,
            page=5,
            chunk_type="table",
            block_type="table",
            table_id="Table 3",
        )
    ]

    expanded = _expand_numeric_table_evidence_units(results, query)
    ordered = _prioritize_numeric_table_results(expanded, query)
    ordered = _apply_numeric_table_same_bundle_hard_gate(ordered, query)
    final = _ensure_numeric_table_evidence_slots(ordered, query, top_k=3)
    final = _apply_numeric_table_same_bundle_hard_gate(final, query)

    ordered_row_ids = [item.get("row_id") for item in ordered if item.get("chunk_type") == "table_row"]
    assert ordered_row_ids[0] == "AID"

    row_items = [item for item in final if item.get("chunk_type") == "table_row"]
    assert row_items[0].get("row_id") == "AID"
    assert any(
        item.get("chunk_type") == "table_row"
        and item.get("row_id") == "AID"
        and "45.2" in (item.get("chunk") or "")
        and "5.78×10−4" in (item.get("chunk") or "")
        for item in row_items
    )
    assert not any(
        item.get("chunk_type") == "table_row"
        and item.get("row_id") in {"Baseline", "ID"}
        and "5.78×10−4" not in (item.get("chunk") or "")
        for item in row_items
    )


def test_numeric_table_finalize_without_rerank_prefers_exact_row_over_broad_support():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    results = [
        _make_numeric_candidate(
            "CBDM(τ=3) | FID=7.42 | Acc=44.8",
            0.96,
            page=4,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 1",
            row_id="CBDM(τ=3)",
            table_row_evidence=True,
            table_row_slice_kind="broad",
            table_row_boundary_text="CBDM (τ=3) 7.42 44.8",
            table_row_raw_text="CBDM (τ=3) 7.42 44.8",
            table_caption="Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
            table_header="Model FID Acc.(%)",
        ),
        _make_numeric_candidate(
            "CBDM(τ=1) | FID=5.86 | Acc=46.6",
            0.63,
            page=4,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 1",
            row_id="CBDM(τ=1)",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="CBDM (τ=1) 5.86 46.6",
            table_row_raw_text="CBDM (τ=1) 5.86 46.6",
            table_caption="Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
            table_header="Model FID Acc.(%)",
        ),
    ]
    config = SimpleNamespace(
        enable_focus_mode=False,
        focus_mode_window_size=1,
        focus_mode_max_sentences=2,
        focus_mode_min_chars=80,
    )

    final = _finalize_without_rerank(results, query, top_k=1, config=config)

    assert final[0]["row_id"] == "CBDM(τ=1)"
    assert "5.86" in _build_context_text_for_result(final[0])
    assert "44.8" not in _build_context_text_for_result(final[0])


def test_numeric_table_finalize_without_rerank_prefers_table7_exact_row_over_table1_ddpm_block():
    query = "实验结果表中哪个方法在 Few-shot 子集上取得最高准确率？具体数值是多少？"
    results = [
        _make_numeric_candidate(
            "DDPM | FID=7.76 | Acc=43.8",
            0.97,
            page=4,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 1",
            row_id="DDPM",
            table_caption="Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
            table_header="Model FID Acc.(%)",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DDPM 7.76 43.8",
            table_row_raw_text="DDPM 7.76 43.8",
            raw_chunk_text="DDPM 7.76 43.8",
            numeric_table_priority=13.4,
            numeric_table_anchor_hits=["Table 1", "FID", "Acc"],
        ),
        _make_numeric_candidate(
            "CSA | Many=64.3 | Med.=49.7 | Few=18.2",
            0.42,
            page=8,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 7",
            row_id="CSA",
            table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            table_header="Method | 100 | 50 | 10 | 100 | 50 | 10 | Many | Med. | Few",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="CSA 64.3 49.7 18.2",
            table_row_raw_text="CSA 64.3 49.7 18.2",
            raw_chunk_text="CSA 64.3 49.7 18.2",
            numeric_table_priority=11.3,
            numeric_table_anchor_hits=["Table 7", "Few"],
        ),
        _make_numeric_candidate(
            "DiffuLT | Many=69.0 | Med.=51.6 | Few=29.7",
            0.91,
            page=8,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 7",
            row_id="DiffuLT",
            table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
            table_header="Method | 100 | 50 | 10 | 100 | 50 | 10 | Many | Med. | Few",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DiffuLT 69.0 51.6 29.7",
            table_row_raw_text="DiffuLT 69.0 51.6 29.7",
            raw_chunk_text="DiffuLT 69.0 51.6 29.7",
            numeric_table_priority=12.6,
            numeric_table_anchor_hits=["Table 7", "DiffuLT", "Few"],
        ),
    ]
    config = SimpleNamespace(
        enable_focus_mode=False,
        focus_mode_window_size=1,
        focus_mode_max_sentences=2,
        focus_mode_min_chars=80,
    )

    final = _finalize_without_rerank(results, query, top_k=1, config=config)
    final_text = _build_context_text_for_result(final[0], query)

    assert final[0].get("table_id") == "Table 7"
    assert "DiffuLT" in final_text
    assert "29.7" in final_text
    assert "DDPM" not in final_text
    assert "7.76" not in final_text


@pytest.mark.parametrize(
    ("query", "results", "expected_fragments"),
    [
        (
            "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？",
            [
                _make_numeric_candidate(
                    "AID | Acc=45.2 | ΔAcc/||D_gen||=5.78×10^-4",
                    0.92,
                    page=5,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_id="Table 3",
                    row_id="AID",
                    table_caption="Table 3: Quantities and classifier enhancement.",
                    table_header="Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||",
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="AID 11,886 45.2 5.78×10^-4",
                    table_row_raw_text="AID 11,886 45.2 5.78×10^-4",
                    raw_chunk_text="AID 11,886 45.2 5.78×10^-4",
                    numeric_table_priority=11.4,
                    numeric_table_anchor_hits=["Table 3", "AID"],
                ),
                _make_numeric_candidate(
                    "CE | Acc=44.1 | ΔAcc/||D_gen||=5.10×10^-4",
                    0.71,
                    page=5,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_id="Table 3",
                    row_id="CE",
                    table_caption="Table 3: Quantities and classifier enhancement.",
                    table_header="Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||",
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="CE 10,000 44.1 5.10×10^-4",
                    table_row_raw_text="CE 10,000 44.1 5.10×10^-4",
                    raw_chunk_text="CE 10,000 44.1 5.10×10^-4",
                    numeric_table_priority=10.2,
                    numeric_table_anchor_hits=["Table 3", "AID"],
                ),
            ],
            ["AID", "45.2", "5.78×10^-4"],
        ),
        (
            "实验结果表中哪个方法在 Few-shot 子集上取得最高准确率？具体数值是多少？",
            [
                _make_numeric_candidate(
                    "DiffuLT | Many=69.0 | Med.=51.6 | Few=29.7",
                    0.91,
                    page=8,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_id="Table 7",
                    row_id="DiffuLT",
                    table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
                    table_header="Method | 100 | 50 | 10 | 100 | 50 | 10 | Many | Med. | Few",
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="DiffuLT 69.0 51.6 29.7",
                    table_row_raw_text="DiffuLT 69.0 51.6 29.7",
                    raw_chunk_text="DiffuLT 69.0 51.6 29.7",
                    numeric_table_priority=12.6,
                    numeric_table_anchor_hits=["Table 7", "DiffuLT", "Few"],
                ),
                _make_numeric_candidate(
                    "CSA | Many=64.3 | Med.=49.7 | Few=18.2",
                    0.42,
                    page=8,
                    chunk_type="table_row",
                    block_type="table_row",
                    table_id="Table 7",
                    row_id="CSA",
                    table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
                    table_header="Method | 100 | 50 | 10 | 100 | 50 | 10 | Many | Med. | Few",
                    table_row_evidence=True,
                    table_row_slice_kind="exact",
                    table_row_boundary_text="CSA 64.3 49.7 18.2",
                    table_row_raw_text="CSA 64.3 49.7 18.2",
                    raw_chunk_text="CSA 64.3 49.7 18.2",
                    numeric_table_priority=11.3,
                    numeric_table_anchor_hits=["Table 7", "Few"],
                ),
            ],
            ["DiffuLT", "29.7"],
        ),
    ],
)
def test_numeric_table_finalize_without_rerank_preserves_q3_q6_winner_rows(
    query,
    results,
    expected_fragments,
):
    config = SimpleNamespace(
        enable_focus_mode=False,
        focus_mode_window_size=1,
        focus_mode_max_sentences=2,
        focus_mode_min_chars=80,
    )

    final = _finalize_without_rerank(results, query, top_k=1, config=config)
    final_text = _build_context_text_for_result(final[0], query)

    for fragment in expected_fragments:
        assert fragment in final_text


def test_numeric_table_context_text_uses_exact_row_for_text_chunks():
    item = _make_numeric_candidate(
        "Figure 3 narrative summary that mentions AID but omits the exact row.",
        0.91,
        page=5,
        chunk_type="text",
        block_type="text",
        numeric_table_exact_context_caption="Table 3: Quantities and classifier enhancement.",
        numeric_table_exact_context_header="Type Acc ΔAcc/||D_gen||",
        numeric_table_exact_context_row_text="AID 11,886 45.2 5.78×10−4",
        evidence_units=[{"evidence_unit_type": "table_row", "content": "AID 11,886 45.2 5.78×10−4"}],
    )

    text = _build_context_text_for_result(item)

    assert "Figure 3 narrative" not in text
    assert "AID" in text
    assert "45.2" in text
    assert "5.78×10−4" in text


def test_numeric_table_context_text_projects_query_focused_exact_row_for_second_best_query():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    item = _make_numeric_candidate(
        "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
        0.49,
        page=9,
        chunk_type="table_row",
        block_type="table_row",
        table_id="Table 8",
        table_caption="Table 8: Results on ImageNet-LT.",
        table_header="ResNet-10 ResNet-50 All All Many Med. Few",
        row_id="DiffuLT",
        table_row_evidence=True,
        table_row_slice_kind="exact",
        table_row_boundary_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
        table_row_raw_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
    )

    text = _build_context_text_for_result(item, query)

    assert "Table 8: Results on ImageNet-LT." in text
    assert "ResNet-10 ResNet-50 All All Many Med. Few" not in text
    assert "DiffuLT" in text
    assert "Many=63.3" in text
    assert "Med.=55.6" in text
    assert "Few=39.4" in text


def test_numeric_table_context_text_projects_query_focused_exact_row_for_explicit_comparator_query():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    item = _make_numeric_candidate(
        "RIDE (3 experts) | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9",
        0.47,
        page=9,
        chunk_type="table_row",
        block_type="table_row",
        table_id="Table 8",
        table_caption="Table 8: Results on ImageNet-LT.",
        table_header="ResNet-10 ResNet-50 All All Many Med. Few",
        row_id="RIDE (3 experts)",
        table_row_evidence=True,
        table_row_slice_kind="exact",
        table_row_boundary_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
        table_row_raw_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
    )

    text = _build_context_text_for_result(item, query)

    assert "Table 8: Results on ImageNet-LT." in text
    assert "ResNet-10 ResNet-50 All All Many Med. Few" not in text
    assert "RIDE (3 experts)" in text
    assert "ResNet-50" in text
    assert "All=54.9" in text


def test_numeric_table_context_text_projects_exact_row_from_table_bundle_for_table1_winner_query():
    query = "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？"
    item = _make_numeric_candidate(
        "Table 1 mixed page narrative that still contains the right row inside evidence_units.",
        0.91,
        page=5,
        chunk_type="table",
        block_type="table",
        table_id="Table 1",
        table_caption="Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
        table_header="Model FID Acc.(%)",
        evidence_units=[
            {
                "evidence_unit_type": "table_row",
                "row_id": "DDPM",
                "row_text": "DDPM 7.76 43.8",
                "row_numbers": "7.76 43.8",
            },
            {
                "evidence_unit_type": "table_row",
                "row_id": "CBDM (τ=1)",
                "row_text": "CBDM (τ=1) 5.86 46.6 44.8 36.3 18.9",
                "row_numbers": "5.86 46.6 44.8 36.3 18.9",
            },
        ],
    )

    text = _build_context_text_for_result(item, query)

    assert "CBDM (τ=1)" in text
    assert "FID=5.86" in text
    assert "Acc=46.6" in text
    assert "36.3" not in text
    assert item["numeric_table_exact_context_row_text"] == "CBDM (τ=1) | FID=5.86 | Acc=46.6"


def test_numeric_table_context_text_projects_exact_row_from_table_bundle_for_table3_winner_query():
    query = "表 3 中哪类生成样本带来的平均每样本性能提升最大？对应的 ΔAcc/||D_gen|| 和分类准确率是多少？"
    item = _make_numeric_candidate(
        "Table 3 mixed page narrative that still contains the right row inside evidence_units.",
        0.92,
        page=5,
        chunk_type="table",
        block_type="table",
        table_id="Table 3",
        table_caption="Table 3: Quantities, overall classifier enhancement, and average improvement per sample.",
        table_header="Group ||D_gen|| Acc.(%) ΔAcc/||D_gen||",
        evidence_units=[
            {
                "evidence_unit_type": "table_row",
                "row_id": "CE",
                "row_text": "CE 10,000 44.1 5.10×10^-4",
                "row_numbers": "10,000 44.1 5.10×10^-4",
            },
            {
                "evidence_unit_type": "table_row",
                "row_id": "AID",
                "row_text": "AID 11,886 45.2 5.78×10−4 80 35.7 32.5",
                "row_numbers": "11,886 45.2 5.78×10−4 80 35.7 32.5",
            },
        ],
    )

    text = _build_context_text_for_result(item, query)

    assert "AID" in text
    assert "||D_gen||=11,886" in text
    assert "Acc=45.2" in text
    assert "ΔAcc/||D_gen||=5.78×10−4" in text
    assert "80" not in text


def test_numeric_table_context_text_projects_exact_row_from_sparse_table7_bundle_for_winner_query():
    query = "实验结果表中哪个方法在 Few-shot 子集上取得最高准确率？具体数值是多少？"
    item = _make_numeric_candidate(
        "[Structured Table Bundle] Table 7 ... [Body] Table7: ResultsonCIFAR100-LTandCIFAR10-LTdatase",
        0.9,
        page=8,
        chunk_type="table",
        block_type="table",
        structured_table_bundle=True,
        table_id="Table 7",
        table_caption="Table 7: Results on CIFAR100-LT and CIFAR10-LT datasets.",
        table_header="Method | 100 | 50 | 10 | 100 | 50 | 10 | Many | Med. | Few",
        evidence_units=[
            {
                "evidence_unit_type": "table_row",
                "row_id": "CE",
                "row_text": "CE 23.5 27.8 28.1",
                "row_numbers": "23.5 27.8 28.1",
            },
            {
                "evidence_unit_type": "table_row",
                "row_id": "DiffuLT",
                "row_text": "DiffuLT 25.1 31.4 29.7",
                "row_numbers": "25.1 31.4 29.7",
            },
        ],
    )

    text = _build_context_text_for_result(item, query)

    assert "DiffuLT" in text
    assert "Few=29.7" in text
    assert "ResultsonCIFAR100-LTandCIFAR10-LTdatase" not in text


def test_numeric_table_finalize_without_rerank_keeps_full_comparator_bundle():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    results = [
        _make_numeric_candidate(
            (
                "DiffuLT achieves strong ImageNet-LT performance with ResNet-50 and improves the All metric "
                "over prior work, while Table 8 summarizes the comparison against cRT, RIDE(3 experts), and ADRW."
            ),
            0.97,
            page=9,
            chunk_type="text",
            block_type="text",
            table_augmented_scope="page_content",
        ),
        _make_numeric_candidate(
            "DiffuLT | All=56.4 | cRT=47.3 | RIDE(3 experts)=54.9 | ADRW=54.1",
            0.41,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="DiffuLT",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
            table_row_raw_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
            numeric_table_anchor_hits=["Table 8", "ResNet-50", "DiffuLT", "All"],
        ),
        _make_numeric_candidate(
            "cRT | All=47.3",
            0.40,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="cRT",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="cRT 41.8 47.3 58.8 44.0 26.1",
            table_row_raw_text="cRT 41.8 47.3 58.8 44.0 26.1",
            numeric_table_anchor_hits=["Table 8", "ResNet-50", "cRT", "All"],
        ),
        _make_numeric_candidate(
            "RIDE (3 experts) | All=54.9",
            0.39,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="RIDE (3 experts)",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            table_row_raw_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            numeric_table_anchor_hits=["Table 8", "ResNet-50", "RIDE (3 experts)", "All"],
        ),
        _make_numeric_candidate(
            "ADRW | All=54.1",
            0.38,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="ADRW",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="ADRW 54.1 62.9 52.6 37.1",
            table_row_raw_text="ADRW 54.1 62.9 52.6 37.1",
            numeric_table_anchor_hits=["Table 8", "ResNet-50", "ADRW", "All"],
        ),
    ]
    config = SimpleNamespace(
        enable_focus_mode=False,
        focus_mode_window_size=1,
        focus_mode_max_sentences=2,
        focus_mode_min_chars=80,
    )

    final = _finalize_without_rerank(results, query, top_k=3, config=config)
    row_items = [item for item in final if item.get("chunk_type") == "table_row"]
    row_ids = [item.get("row_id") for item in row_items]

    assert {"DiffuLT", "cRT", "RIDE (3 experts)", "ADRW"} <= set(row_ids)
    assert len(row_items) >= 4


def test_numeric_table_finalize_without_rerank_limits_second_best_bundle_to_target_and_runner_up():
    query = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    results = [
        _make_numeric_candidate(
            "Table 8 summary paragraph mentioning DiffuLT and second-best comparison.",
            0.98,
            page=9,
            chunk_type="text",
            block_type="text",
            table_augmented_scope="page_content",
            table_id="Table 8",
            table_caption="Table 8: Results on ImageNet-LT.",
        ),
        _make_numeric_candidate(
            "DiffuLT | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            0.49,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="DiffuLT",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
            table_row_raw_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
            numeric_table_anchor_hits=["Table 8", "ImageNet-LT", "DiffuLT", "Many", "Med.", "Few"],
        ),
        _make_numeric_candidate(
            "RIDE (3 experts) | All=54.9 | Many=66.2 | Med.=51.7 | Few=34.9",
            0.48,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="RIDE (3 experts)",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            table_row_raw_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            numeric_table_anchor_hits=["Table 8", "ImageNet-LT", "RIDE (3 experts)", "Many", "Med.", "Few"],
        ),
        _make_numeric_candidate(
            "ADRW | All=54.1 | Many=62.9 | Med.=52.6 | Few=37.1",
            0.47,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="ADRW",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="ADRW 54.1 62.9 52.6 37.1",
            table_row_raw_text="ADRW 54.1 62.9 52.6 37.1",
            numeric_table_anchor_hits=["Table 8", "ImageNet-LT", "ADRW", "Many", "Med.", "Few"],
        ),
        _make_numeric_candidate(
            "cRT | All=47.3 | Many=58.8 | Med.=44.0 | Few=26.1",
            0.46,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="cRT",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="cRT 41.8 47.3 58.8 44.0 26.1",
            table_row_raw_text="cRT 41.8 47.3 58.8 44.0 26.1",
            numeric_table_anchor_hits=["Table 8", "ImageNet-LT", "cRT", "Many", "Med.", "Few"],
        ),
        _make_numeric_candidate(
            "CE | All=41.6 | Many=64.0 | Med.=33.8 | Few=5.8",
            0.45,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="CE",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="CE 34.8 41.6 64.0 33.8 5.8",
            table_row_raw_text="CE 34.8 41.6 64.0 33.8 5.8",
            numeric_table_anchor_hits=["Table 8", "ImageNet-LT", "CE", "Many", "Med.", "Few"],
        ),
    ]
    config = SimpleNamespace(
        enable_focus_mode=False,
        focus_mode_window_size=1,
        focus_mode_max_sentences=2,
        focus_mode_min_chars=80,
    )

    final = _finalize_without_rerank(results, query, top_k=6, config=config)
    row_ids = [item.get("row_id") for item in final if item.get("chunk_type") == "table_row"]

    assert row_ids == ["DiffuLT", "RIDE (3 experts)"]
    assert all(item.get("chunk_type") == "table_row" for item in final)


def test_numeric_table_finalize_without_rerank_dedupes_normalized_explicit_comparator_rows():
    query = "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？"
    results = [
        _make_numeric_candidate(
            "Table 8 summary paragraph mentioning DiffuLT and comparator gains.",
            0.98,
            page=9,
            chunk_type="text",
            block_type="text",
            table_augmented_scope="page_content",
        ),
        _make_numeric_candidate(
            "DiffuLT | All=56.4",
            0.44,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="DiffuLT",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
            table_row_raw_text="DiffuLT 50.4 56.4 63.3 55.6 39.4",
            numeric_table_anchor_hits=["Table 8", "ResNet-50", "DiffuLT", "All"],
        ),
        _make_numeric_candidate(
            "cRT | All=47.3",
            0.43,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="cRT",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="cRT 41.8 47.3 58.8 44.0 26.1",
            table_row_raw_text="cRT 41.8 47.3 58.8 44.0 26.1",
            numeric_table_anchor_hits=["Table 8", "ResNet-50", "cRT", "All"],
        ),
        _make_numeric_candidate(
            "RIDE (3 experts) | All=54.9",
            0.42,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="RIDE (3 experts)",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            table_row_raw_text="RIDE (3 experts) 45.9 54.9 66.2 51.7 34.9",
            numeric_table_anchor_hits=["Table 8", "ResNet-50", "RIDE (3 experts)", "All"],
        ),
        _make_numeric_candidate(
            "RIDE(3experts) | All=54.9",
            0.41,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="RIDE(3experts)",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="RIDE(3experts) 45.9 54.9 66.2 51.7 34.9",
            table_row_raw_text="RIDE(3experts) 45.9 54.9 66.2 51.7 34.9",
            numeric_table_anchor_hits=["Table 8", "ResNet-50", "RIDE(3experts)", "All"],
        ),
        _make_numeric_candidate(
            "ADRW | All=54.1",
            0.40,
            page=9,
            chunk_type="table_row",
            block_type="table_row",
            table_caption="Table 8: Results on ImageNet-LT.",
            table_header="ResNet-10 ResNet-50 All All Many Med. Few",
            table_id="Table 8",
            row_id="ADRW",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="ADRW 54.1 62.9 52.6 37.1",
            table_row_raw_text="ADRW 54.1 62.9 52.6 37.1",
            numeric_table_anchor_hits=["Table 8", "ResNet-50", "ADRW", "All"],
        ),
    ]
    config = SimpleNamespace(
        enable_focus_mode=False,
        focus_mode_window_size=1,
        focus_mode_max_sentences=2,
        focus_mode_min_chars=80,
    )

    final = _finalize_without_rerank(results, query, top_k=5, config=config)
    row_ids = [item.get("row_id") for item in final if item.get("chunk_type") == "table_row"]
    normalized_ride_rows = [
        row_id
        for row_id in row_ids
        if re.sub(r"\s+", "", str(row_id or "").lower()) == "ride(3experts)"
    ]

    assert {"DiffuLT", "cRT", "ADRW"} <= set(row_ids)
    assert len(normalized_ride_rows) == 1


def test_numeric_table_finalize_without_rerank_keeps_cost_anchor_text():
    query = "这篇论文的额外开销、推理时间和 FLOPs 分别是多少？"
    results = [
        _make_numeric_candidate(
            "no extra overhead 24 hours six days",
            0.96,
            page=5,
            chunk_type="text",
            block_type="text",
            table_augmented_scope="page_content",
        ),
        _make_numeric_candidate(
            "method note with unrelated table row 7.42 44.8",
            0.90,
            page=4,
            chunk_type="table_row",
            block_type="table_row",
            table_id="Table 1",
            row_id="CBDM(τ=3)",
            table_row_evidence=True,
            table_row_slice_kind="exact",
            table_row_boundary_text="CBDM (τ=3) 7.42 44.8",
            table_caption="Table 1: FID of different generation models and their corresponding classifiers' accuracy.",
            table_header="Model FID Acc.(%)",
        ),
    ]
    config = SimpleNamespace(
        enable_focus_mode=False,
        focus_mode_window_size=1,
        focus_mode_max_sentences=2,
        focus_mode_min_chars=80,
    )

    final = _finalize_without_rerank(results, query, top_k=2, config=config)
    final_texts = [_build_context_text_for_result(item) for item in final]

    assert any(
        "no extra overhead" in text or "24 hours" in text or "six days" in text
        for text in final_texts
    )
