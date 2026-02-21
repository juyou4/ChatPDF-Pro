"""timings 字段完整性属性测试

Feature: chatpdf-performance-optimization, Property 1: timings 字段完整性

使用 hypothesis 进行属性测试，验证 search_document_chunks() 返回的 timings 字典
在任意检索配置下都满足完整性约束。

**Validates: Requirements 1.1, 1.4, 1.5**
"""
import sys
import os
import pickle
import tempfile
from unittest.mock import patch, MagicMock

import faiss
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embedding_service import search_document_chunks


# ============================================================
# 测试 Fixture：创建最小化的 FAISS 索引和 chunks 数据
# ============================================================

# 向量维度（与 mock embedding 函数一致）
EMBED_DIM = 64
# 测试用 chunk 数量
NUM_CHUNKS = 5


@pytest.fixture(scope="module")
def vector_store_dir():
    """创建临时目录，包含 FAISS 索引和 chunks pickle 文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_id = "test_doc"

        # 创建 FAISS 内积索引
        index = faiss.IndexFlatIP(EMBED_DIM)
        vectors = np.random.randn(NUM_CHUNKS, EMBED_DIM).astype("float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)
        faiss.write_index(index, os.path.join(tmpdir, f"{doc_id}.index"))

        # 创建 chunks pickle 文件
        chunks = [f"这是测试文本块 {i}，包含一些用于检索的内容。" for i in range(NUM_CHUNKS)]
        data = {
            "chunks": chunks,
            "embedding_model": "local-minilm",
            "parent_chunks": [],
            "child_to_parent": {},
        }
        with open(os.path.join(tmpdir, f"{doc_id}.pkl"), "wb") as f:
            pickle.dump(data, f)

        yield tmpdir


# ============================================================
# Mock 辅助函数
# ============================================================

def _make_mock_embed_fn():
    """创建 mock embedding 函数，返回随机归一化向量"""
    def embed_fn(texts):
        vecs = np.random.randn(len(texts), EMBED_DIM).astype("float32")
        faiss.normalize_L2(vecs)
        return vecs
    return embed_fn


def _make_mock_rerank(query, candidates, *args, **kwargs):
    """mock rerank 函数，直接返回候选结果（标记 reranked=True）"""
    for item in candidates:
        item["reranked"] = True
    return candidates


# ============================================================
# Hypothesis 策略：检索配置参数
# ============================================================

# 核心布尔配置：use_hybrid 和 use_rerank 的任意组合
search_config_strategy = st.fixed_dictionaries({
    "use_hybrid": st.booleans(),
    "use_rerank": st.booleans(),
})


# ============================================================
# Property 1：timings 字段完整性
# **Validates: Requirements 1.1, 1.4, 1.5**
# ============================================================

class TestP1TimingsCompleteness:
    """Property 1: timings 字段完整性

    对于任意检索配置（use_hybrid、use_rerank 的任意组合），
    search_document_chunks() 返回的 timings 字典至少包含
    vector_search_ms 和 total_ms 两个字段，且所有值为非负浮点数。

    **Validates: Requirements 1.1, 1.4, 1.5**
    """

    @given(config=search_config_strategy)
    @settings(max_examples=100)
    def test_timings_always_contains_required_fields(self, config, vector_store_dir):
        """属性：timings 始终包含 vector_search_ms 和 total_ms"""
        use_hybrid = config["use_hybrid"]
        use_rerank = config["use_rerank"]

        # Mock 所有外部依赖，让函数能在测试环境中运行
        with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
             patch("services.embedding_service._apply_rerank", side_effect=_make_mock_rerank), \
             patch("services.embedding_service._merge_with_group_search", side_effect=lambda **kwargs: kwargs["chunk_results"]), \
             patch("services.embedding_service._query_vector_cache") as mock_cache:

            # 缓存始终 miss，强制走 embedding 路径
            mock_cache.get.return_value = None

            # 如果启用混合检索，mock BM25 相关依赖
            if use_hybrid:
                with patch("services.bm25_service.bm25_search", return_value=[]), \
                     patch("services.hybrid_search.hybrid_search_merge", side_effect=lambda vec, bm25, **kw: vec[:kw.get("top_k", 10)]):
                    results, timings = search_document_chunks(
                        doc_id="test_doc",
                        query="测试查询",
                        vector_store_dir=vector_store_dir,
                        pages=[{"page": 1, "text": "页面文本"}],
                        top_k=3,
                        use_hybrid=use_hybrid,
                        use_rerank=use_rerank,
                    )
            else:
                results, timings = search_document_chunks(
                    doc_id="test_doc",
                    query="测试查询",
                    vector_store_dir=vector_store_dir,
                    pages=[{"page": 1, "text": "页面文本"}],
                    top_k=3,
                    use_hybrid=use_hybrid,
                    use_rerank=use_rerank,
                )

            # 断言 1：timings 必须包含 vector_search_ms 和 total_ms
            assert "vector_search_ms" in timings, (
                f"timings 缺少 vector_search_ms 字段，当前 timings={timings}"
            )
            assert "total_ms" in timings, (
                f"timings 缺少 total_ms 字段，当前 timings={timings}"
            )

            # 断言 2：所有值为非负数值（int 或 float）
            for key, value in timings.items():
                assert isinstance(value, (int, float)), (
                    f"timings['{key}'] 应为数值类型，实际为 {type(value).__name__}: {value}"
                )
                assert value >= 0, (
                    f"timings['{key}'] 应为非负数，实际为 {value}"
                )

    @given(config=search_config_strategy)
    @settings(max_examples=100)
    def test_timings_conditional_fields(self, config, vector_store_dir):
        """属性：跳过的阶段不应出现在 timings 中（需求 1.5）

        - use_hybrid=False 时，bm25_search_ms 不应存在
        - use_rerank=False 时，rerank_ms 不应存在
        """
        use_hybrid = config["use_hybrid"]
        use_rerank = config["use_rerank"]

        with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
             patch("services.embedding_service._apply_rerank", side_effect=_make_mock_rerank), \
             patch("services.embedding_service._merge_with_group_search", side_effect=lambda **kwargs: kwargs["chunk_results"]), \
             patch("services.embedding_service._query_vector_cache") as mock_cache:

            mock_cache.get.return_value = None

            if use_hybrid:
                with patch("services.bm25_service.bm25_search", return_value=[]), \
                     patch("services.hybrid_search.hybrid_search_merge", side_effect=lambda vec, bm25, **kw: vec[:kw.get("top_k", 10)]):
                    results, timings = search_document_chunks(
                        doc_id="test_doc",
                        query="测试查询",
                        vector_store_dir=vector_store_dir,
                        pages=[{"page": 1, "text": "页面文本"}],
                        top_k=3,
                        use_hybrid=use_hybrid,
                        use_rerank=use_rerank,
                    )
            else:
                results, timings = search_document_chunks(
                    doc_id="test_doc",
                    query="测试查询",
                    vector_store_dir=vector_store_dir,
                    pages=[{"page": 1, "text": "页面文本"}],
                    top_k=3,
                    use_hybrid=use_hybrid,
                    use_rerank=use_rerank,
                )

            # 断言：use_hybrid=False 时不应有 bm25_search_ms
            if not use_hybrid:
                assert "bm25_search_ms" not in timings, (
                    f"use_hybrid=False 时 timings 不应包含 bm25_search_ms，"
                    f"当前 timings={timings}"
                )

            # 断言：use_rerank=False 时不应有 rerank_ms
            if not use_rerank:
                assert "rerank_ms" not in timings, (
                    f"use_rerank=False 时 timings 不应包含 rerank_ms，"
                    f"当前 timings={timings}"
                )

    @given(config=search_config_strategy)
    @settings(max_examples=100)
    def test_total_ms_not_less_than_vector_search_ms(self, config, vector_store_dir):
        """属性：total_ms 应大于等于 vector_search_ms（总耗时包含向量检索）"""
        use_hybrid = config["use_hybrid"]
        use_rerank = config["use_rerank"]

        with patch("services.embedding_service.get_embedding_function", return_value=_make_mock_embed_fn()), \
             patch("services.embedding_service._apply_rerank", side_effect=_make_mock_rerank), \
             patch("services.embedding_service._merge_with_group_search", side_effect=lambda **kwargs: kwargs["chunk_results"]), \
             patch("services.embedding_service._query_vector_cache") as mock_cache:

            mock_cache.get.return_value = None

            if use_hybrid:
                with patch("services.bm25_service.bm25_search", return_value=[]), \
                     patch("services.hybrid_search.hybrid_search_merge", side_effect=lambda vec, bm25, **kw: vec[:kw.get("top_k", 10)]):
                    results, timings = search_document_chunks(
                        doc_id="test_doc",
                        query="测试查询",
                        vector_store_dir=vector_store_dir,
                        pages=[{"page": 1, "text": "页面文本"}],
                        top_k=3,
                        use_hybrid=use_hybrid,
                        use_rerank=use_rerank,
                    )
            else:
                results, timings = search_document_chunks(
                    doc_id="test_doc",
                    query="测试查询",
                    vector_store_dir=vector_store_dir,
                    pages=[{"page": 1, "text": "页面文本"}],
                    top_k=3,
                    use_hybrid=use_hybrid,
                    use_rerank=use_rerank,
                )

            # 断言：total_ms >= vector_search_ms
            assert timings["total_ms"] >= timings["vector_search_ms"], (
                f"total_ms ({timings['total_ms']}) 应 >= vector_search_ms ({timings['vector_search_ms']})"
            )
