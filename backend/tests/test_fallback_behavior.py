"""
降级回退行为单元测试

测试 RAG 系统在以下三种降级场景下的行为：
1. 意群禁用降级：enable_semantic_groups=False 时回退到分块级别检索
2. 索引不存在降级：意群索引文件不存在时回退到分块级别检索
3. LLM 失败降级：LLM API 调用失败时使用文本截断作为降级方案

验证 RetrievalTrace 中的 fallback_type 正确标记：
- "groups_disabled"：意群功能被禁用
- "index_missing"：意群索引不存在
- "llm_failed"：LLM API 调用失败

Requirements: 6.3, 7.2, 7.3, 8.2
"""

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

from services.rag_config import RAGConfig
from services.retrieval_logger import RetrievalTrace, RetrievalLogger


# ============================================================
# 辅助函数
# ============================================================


def _make_fake_chunk_results(n: int = 3) -> list:
    """创建模拟的分块检索结果

    Args:
        n: 结果数量

    Returns:
        模拟的分块检索结果列表
    """
    results = []
    for i in range(n):
        results.append({
            "chunk": f"这是第{i+1}个测试分块的内容，包含一些有意义的文本信息。",
            "page": i + 1,
            "score": float(i),
            "similarity": 0.9 - i * 0.1,
            "similarity_percent": round((0.9 - i * 0.1) * 100, 2),
            "snippet": f"测试分块{i+1}的摘要片段",
            "highlights": [],
            "reranked": False,
        })
    return results


# ============================================================
# 测试 1：意群禁用降级（Requirements 7.2, 8.2）
# ============================================================


class TestGroupsDisabledFallback:
    """测试意群功能被禁用时的降级回退行为

    当 enable_semantic_groups=False 时：
    - get_relevant_context 应返回有效的上下文字符串（简单拼接）
    - retrieval_meta 中 fallback.type 应为 "groups_disabled"
    """

    def _get_disabled_config(self):
        """创建禁用意群功能的配置"""
        return RAGConfig(enable_semantic_groups=False)

    @patch("services.embedding_service.search_document_chunks")
    def test_意群禁用时返回有效上下文字符串(self, mock_search):
        """意群禁用时，get_relevant_context 应返回简单拼接的上下文字符串

        Validates: Requirements 7.2
        """
        from services.embedding_service import get_relevant_context

        # 模拟搜索结果
        fake_results = _make_fake_chunk_results(3)
        mock_search.return_value = fake_results

        # 使用 patch 替换 RAGConfig 类，使其返回禁用意群的配置
        with patch("services.rag_config.RAGConfig", return_value=self._get_disabled_config()):
            context, meta = get_relevant_context(
                doc_id="test-doc",
                query="测试查询",
                vector_store_dir="/tmp/test",
                pages=[],
            )

        # 验证返回了有效的上下文字符串
        assert isinstance(context, str)
        assert len(context) > 0
        # 验证上下文包含各分块内容
        for result in fake_results:
            assert result["chunk"] in context

    @patch("services.embedding_service.search_document_chunks")
    def test_意群禁用时fallback_type为groups_disabled(self, mock_search):
        """意群禁用时，retrieval_meta 中 fallback.type 应为 "groups_disabled"

        Validates: Requirements 7.2, 8.2
        """
        from services.embedding_service import get_relevant_context

        mock_search.return_value = _make_fake_chunk_results(2)

        with patch("services.rag_config.RAGConfig", return_value=self._get_disabled_config()):
            _, meta = get_relevant_context(
                doc_id="test-doc",
                query="测试查询",
                vector_store_dir="/tmp/test",
                pages=[],
            )

        # 验证 retrieval_meta 包含正确的降级信息
        assert "fallback" in meta
        assert meta["fallback"] is not None
        assert meta["fallback"]["type"] == "groups_disabled"

    @patch("services.embedding_service.search_document_chunks")
    def test_意群禁用时retrieval_meta字段完整(self, mock_search):
        """意群禁用时，retrieval_meta 应包含所有必要字段

        Validates: Requirements 8.2
        """
        from services.embedding_service import get_relevant_context

        mock_search.return_value = _make_fake_chunk_results(2)

        with patch("services.rag_config.RAGConfig", return_value=self._get_disabled_config()):
            _, meta = get_relevant_context(
                doc_id="test-doc",
                query="测试查询",
                vector_store_dir="/tmp/test",
                pages=[],
            )

        # 验证 retrieval_meta 包含所有必要字段
        assert "query_type" in meta
        assert "granularities" in meta
        assert "token_used" in meta
        assert "fallback" in meta
        assert "citations" in meta


# ============================================================
# 测试 2：索引不存在降级（Requirements 6.3, 8.2）
# ============================================================


class TestIndexMissingFallback:
    """测试意群索引不存在时的降级回退行为

    当意群功能启用但索引文件不存在时：
    - _merge_with_group_search 应返回原始分块结果
    - get_relevant_context 应优雅降级
    """

    def _get_enabled_config(self):
        """创建启用意群功能的配置"""
        return RAGConfig(enable_semantic_groups=True)

    def _get_disabled_config(self):
        """创建禁用意群功能的配置"""
        return RAGConfig(enable_semantic_groups=False)

    @patch("services.embedding_service._load_group_index", return_value=None)
    def test_索引不存在时merge返回原始分块结果(self, mock_load_index):
        """意群索引不存在时，_merge_with_group_search 应返回原始分块结果

        Validates: Requirements 6.3
        """
        from services.embedding_service import _merge_with_group_search
        import numpy as np

        fake_results = _make_fake_chunk_results(3)
        query_vector = np.zeros((1, 384), dtype="float32")

        with patch("services.rag_config.RAGConfig", return_value=self._get_enabled_config()):
            merged = _merge_with_group_search(
                doc_id="test-doc-no-index",
                chunk_results=fake_results,
                query_vector=query_vector,
                chunks=["chunk1", "chunk2", "chunk3"],
                pages=[],
                query="测试查询",
                top_k=10,
            )

        # 验证返回的是原始分块结果
        assert merged == fake_results
        assert len(merged) == 3

    @patch("services.embedding_service._build_context_with_groups", return_value=(None, {}))
    @patch("services.embedding_service.search_document_chunks")
    def test_索引不存在时get_relevant_context降级(
        self, mock_search, mock_build_ctx
    ):
        """意群索引不存在时，get_relevant_context 应降级到简单拼接

        Validates: Requirements 6.3, 8.2
        """
        from services.embedding_service import get_relevant_context

        mock_search.return_value = _make_fake_chunk_results(2)

        # 启用意群功能，但 _build_context_with_groups 返回 None（模拟索引不存在）
        with patch("services.rag_config.RAGConfig", return_value=self._get_enabled_config()):
            context, meta = get_relevant_context(
                doc_id="test-doc-no-index",
                query="测试查询",
                vector_store_dir="/tmp/test",
                pages=[],
            )

        # 验证返回了有效的上下文字符串
        assert isinstance(context, str)
        assert len(context) > 0

        # 验证 fallback_type 为 "index_missing"
        assert meta["fallback"] is not None
        assert meta["fallback"]["type"] == "index_missing"

    def test_意群禁用时merge直接返回原始结果(self):
        """意群功能禁用时，_merge_with_group_search 应直接返回原始分块结果

        Validates: Requirements 7.2
        """
        from services.embedding_service import _merge_with_group_search
        import numpy as np

        fake_results = _make_fake_chunk_results(3)
        query_vector = np.zeros((1, 384), dtype="float32")

        with patch("services.rag_config.RAGConfig", return_value=self._get_disabled_config()):
            merged = _merge_with_group_search(
                doc_id="test-doc",
                chunk_results=fake_results,
                query_vector=query_vector,
                chunks=["chunk1", "chunk2", "chunk3"],
                pages=[],
                query="测试查询",
                top_k=10,
            )

        # 验证返回的是原始分块结果（未经修改）
        assert merged == fake_results


# ============================================================
# 测试 3：LLM 失败降级（Requirements 7.3, 8.2）
# ============================================================


class TestLLMFailedFallback:
    """测试 LLM API 调用失败时的降级回退行为

    当 LLM API 调用失败时：
    - _build_semantic_group_index 应记录警告日志并继续
    - 系统应仍然使用分块级别检索正常工作
    """

    @patch("services.embedding_service._run_async", side_effect=Exception("LLM API 调用超时"))
    @patch("services.semantic_group_service.SemanticGroupService")
    def test_LLM失败时build_index记录警告并继续(
        self, mock_group_service_cls, mock_run_async, caplog
    ):
        """LLM API 失败时，_build_semantic_group_index 应记录警告日志并继续

        Validates: Requirements 7.3, 8.2
        """
        from services.embedding_service import _build_semantic_group_index
        import numpy as np

        enabled_config = RAGConfig(enable_semantic_groups=True)

        with patch("services.rag_config.RAGConfig", return_value=enabled_config):
            with caplog.at_level(logging.WARNING, logger="services.embedding_service"):
                # 调用不应抛出异常
                _build_semantic_group_index(
                    doc_id="test-doc-llm-fail",
                    chunks=["测试分块1", "测试分块2"],
                    pages=[{"page": 1, "content": "测试分块1"}],
                    embed_fn=lambda texts: np.zeros((len(texts), 384)),
                    api_key="test-key",
                )

        # 验证记录了警告日志
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) >= 1
        # 验证警告日志包含关键信息
        warning_text = " ".join(r.message for r in warning_records)
        assert "语义意群生成失败" in warning_text

    def test_意群功能禁用时build_index直接跳过(self, caplog):
        """意群功能禁用时，_build_semantic_group_index 应直接跳过

        Validates: Requirements 7.2
        """
        from services.embedding_service import _build_semantic_group_index
        import numpy as np

        disabled_config = RAGConfig(enable_semantic_groups=False)

        with patch("services.rag_config.RAGConfig", return_value=disabled_config):
            with caplog.at_level(logging.INFO, logger="services.embedding_service"):
                _build_semantic_group_index(
                    doc_id="test-doc-disabled",
                    chunks=["测试分块"],
                    pages=[],
                    embed_fn=lambda texts: np.zeros((len(texts), 384)),
                )

        # 验证记录了跳过信息
        log_text = caplog.text
        assert "禁用" in log_text or "跳过" in log_text

    def test_LLM失败时generate_summary降级为截断(self):
        """LLM API 失败时，_generate_summary 应降级为文本截断

        Validates: Requirements 7.3
        """
        from services.semantic_group_service import SemanticGroupService

        service = SemanticGroupService(api_key="fake-key")

        # Mock _call_llm 使其抛出异常
        with patch.object(
            service, "_call_llm", side_effect=Exception("API 连接失败")
        ):
            long_text = "这是一段很长的测试文本。" * 50  # 超过 80 字
            summary, status = asyncio.run(
                service._generate_summary(long_text, 80)
            )

        # 验证降级为截断
        assert status == "failed"
        assert len(summary) <= 80
        # 验证截断内容是原文的前缀
        assert long_text.startswith(summary)

    def test_LLM失败时digest也降级为截断(self):
        """LLM API 失败时，digest 生成也应降级为文本截断

        Validates: Requirements 7.3
        """
        from services.semantic_group_service import SemanticGroupService

        service = SemanticGroupService(api_key="fake-key")

        with patch.object(
            service, "_call_llm", side_effect=Exception("API 超时")
        ):
            long_text = "这是一段用于测试精要生成的文本内容。" * 200  # 超过 1000 字
            digest, status = asyncio.run(
                service._generate_summary(long_text, 1000)
            )

        # 验证降级为截断
        assert status == "failed"
        assert len(digest) <= 1000
        assert long_text.startswith(digest)

    def test_无API_key时直接降级为截断(self):
        """未配置 API key 时，应直接降级为文本截断

        Validates: Requirements 7.3
        """
        from services.semantic_group_service import SemanticGroupService

        # 不提供 API key
        service = SemanticGroupService(api_key="")

        long_text = "测试文本内容。" * 30
        summary, status = asyncio.run(
            service._generate_summary(long_text, 80)
        )

        assert status == "failed"
        assert len(summary) <= 80

    def test_LLM失败时extract_keywords返回空列表(self):
        """LLM API 失败时，_extract_keywords 应返回空列表

        Validates: Requirements 7.3
        """
        from services.semantic_group_service import SemanticGroupService

        service = SemanticGroupService(api_key="fake-key")

        with patch.object(
            service, "_call_llm", side_effect=Exception("API 错误")
        ):
            keywords = asyncio.run(
                service._extract_keywords("测试文本内容")
            )

        assert keywords == []


# ============================================================
# 测试 4：RetrievalTrace fallback_type 标记验证（Requirements 8.2）
# ============================================================


class TestRetrievalTraceFallbackType:
    """验证 RetrievalTrace 中 fallback_type 的正确标记

    确保三种降级类型在 RetrievalTrace 和 retrieval_meta 中正确表示。
    """

    def test_groups_disabled标记(self):
        """fallback_type="groups_disabled" 应正确传递到 retrieval_meta

        Validates: Requirements 8.2
        """
        trace = RetrievalTrace(
            query="测试",
            query_type="unknown",
            fallback_type="groups_disabled",
            fallback_detail="意群功能已禁用，回退到分块级别检索",
        )

        logger_inst = RetrievalLogger()
        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["fallback"]["type"] == "groups_disabled"
        assert "禁用" in meta["fallback"]["detail"]

    def test_index_missing标记(self):
        """fallback_type="index_missing" 应正确传递到 retrieval_meta

        Validates: Requirements 8.2
        """
        trace = RetrievalTrace(
            query="测试",
            query_type="unknown",
            fallback_type="index_missing",
            fallback_detail="意群向量索引不存在",
        )

        logger_inst = RetrievalLogger()
        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["fallback"]["type"] == "index_missing"
        assert "索引" in meta["fallback"]["detail"]

    def test_llm_failed标记(self):
        """fallback_type="llm_failed" 应正确传递到 retrieval_meta

        Validates: Requirements 8.2
        """
        trace = RetrievalTrace(
            query="测试",
            query_type="unknown",
            fallback_type="llm_failed",
            fallback_detail="LLM API 调用失败，使用文本截断",
        )

        logger_inst = RetrievalLogger()
        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["fallback"]["type"] == "llm_failed"
        assert "LLM" in meta["fallback"]["detail"]

    def test_无降级时fallback为None(self):
        """未发生降级时，fallback 应为 None

        Validates: Requirements 8.2
        """
        trace = RetrievalTrace(
            query="测试",
            query_type="analytical",
            fallback_type=None,
            fallback_detail=None,
        )

        logger_inst = RetrievalLogger()
        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["fallback"] is None

    def test_降级日志包含降级类型(self, caplog):
        """降级时日志应包含降级类型信息

        Validates: Requirements 8.2
        """
        trace = RetrievalTrace(
            query="测试",
            query_type="unknown",
            fallback_type="groups_disabled",
            fallback_detail="意群功能已禁用",
        )

        logger_inst = RetrievalLogger()
        with caplog.at_level(logging.WARNING, logger="services.retrieval_logger"):
            logger_inst.log_trace(trace)

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) >= 1
        warning_text = " ".join(r.message for r in warning_records)
        assert "groups_disabled" in warning_text
