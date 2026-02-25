"""引用相关性优化 - 单元测试

Feature: chatpdf-citation-relevance

覆盖融合逻辑的降级行为、边界场景等。
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 4.2**
"""
import sys
import os

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from routes.chat_routes import (
    _build_fused_context,
    _build_selected_text_citation,
    _build_selected_text_fallback_citations,
)


class TestBuildFusedContext:
    """_build_fused_context 单元测试"""

    def test_empty_retrieval_context_degradation(self):
        """降级行为：retrieval_context 为空时，融合上下文仅包含 selected_text
        **Validates: Requirements 1.3**
        """
        result = _build_fused_context("选中的文本内容", "", {"page_start": 3, "page_end": 3})

        assert "选中的文本内容" in result
        # 不应包含"相关文档片段"标记
        assert "相关文档片段" not in result

    def test_enable_vector_search_false_only_selected_text(self):
        """enable_vector_search=false 时仅用 selected_text
        模拟该场景：不传入 retrieval_context
        **Validates: Requirements 1.4**
        """
        result = _build_fused_context("用户框选的段落", "", None)

        assert "用户框选的段落" in result
        assert "相关文档片段" not in result

    def test_cross_page_range_label(self):
        """跨页 page_info 显示页码范围
        **Validates: Requirements 1.1, 1.2**
        """
        result = _build_fused_context(
            "跨页文本", "检索结果", {"page_start": 5, "page_end": 8}
        )

        # 应包含跨页页码标注
        assert "（页码: 5-8）" in result
        assert "跨页文本" in result
        assert "检索结果" in result

    def test_same_page_label(self):
        """同页 page_info 显示单页页码
        **Validates: Requirements 1.1**
        """
        result = _build_fused_context(
            "单页文本", "检索结果", {"page_start": 3, "page_end": 3}
        )

        # 同页时应显示单页页码格式
        assert "（页码: 3）" in result
        assert "3-3" not in result

    def test_none_page_info(self):
        """page_info 为 None 时不显示页码标注"""
        result = _build_fused_context("文本内容", "检索结果", None)

        assert "页码" not in result
        assert "文本内容" in result
        assert "检索结果" in result

    def test_empty_page_info(self):
        """page_info 为空字典时不显示页码标注（因为 page_start=0, page_end=0 相等）"""
        result = _build_fused_context("文本内容", "检索结果", {})

        assert "文本内容" in result
        assert "检索结果" in result

    def test_selected_text_before_retrieval_context(self):
        """selected_text 在 retrieval_context 之前
        **Validates: Requirements 1.2**
        """
        result = _build_fused_context(
            "框选内容AAA", "检索内容BBB", {"page_start": 1, "page_end": 1}
        )

        assert result.index("框选内容AAA") < result.index("检索内容BBB")


class TestBuildSelectedTextCitation:
    """_build_selected_text_citation 单元测试"""

    def test_none_page_info_defaults_to_page_1(self):
        """page_info 为 None 时默认页码为 1
        **Validates: Requirements 4.2**
        """
        citation = _build_selected_text_citation("一些文本", None)

        assert citation["page_range"] == [1, 1]
        assert citation["ref"] == 1
        assert citation["group_id"] == "selected-text"
        assert citation["highlight_text"] == "一些文本"

    def test_same_page_start_and_end(self):
        """page_start 和 page_end 相同
        **Validates: Requirements 4.2**
        """
        citation = _build_selected_text_citation(
            "单页文本", {"page_start": 5, "page_end": 5}
        )

        assert citation["page_range"] == [5, 5]

    def test_different_page_start_and_end(self):
        """page_start 和 page_end 不同（跨页）
        **Validates: Requirements 4.2**
        """
        citation = _build_selected_text_citation(
            "跨页文本", {"page_start": 3, "page_end": 7}
        )

        assert citation["page_range"] == [3, 7]

    def test_highlight_text_truncation(self):
        """selected_text 超过 200 字符时 highlight_text 被截断
        **Validates: Requirements 4.2**
        """
        long_text = "这是一段很长的文本内容" * 30  # 远超 200 字符
        citation = _build_selected_text_citation(
            long_text, {"page_start": 1, "page_end": 2}
        )

        assert len(citation["highlight_text"]) <= 200
        # highlight_text 应是 selected_text 前 200 字符的 strip 结果
        assert citation["highlight_text"] == long_text[:200].strip()

    def test_citation_structure_completeness(self):
        """citation 包含所有必需键
        **Validates: Requirements 4.2**
        """
        citation = _build_selected_text_citation(
            "测试文本", {"page_start": 10, "page_end": 12}
        )

        required_keys = {"ref", "group_id", "page_range", "highlight_text"}
        assert required_keys == set(citation.keys())

    def test_highlight_text_stripped(self):
        """highlight_text 应去除首尾空白"""
        citation = _build_selected_text_citation(
            "  带空格的文本  ", {"page_start": 1, "page_end": 1}
        )

        assert citation["highlight_text"] == "带空格的文本"

    def test_empty_page_info_dict(self):
        """空字典 page_info 使用默认值"""
        citation = _build_selected_text_citation("文本", {})

        # page_start 默认 1，page_end 默认等于 page_start
        assert citation["page_range"] == [1, 1]


class TestSelectedTextFallbackCitation:
    """selected_text 兜底引用策略测试"""

    def test_short_selected_text_should_not_generate_fallback_citation(self):
        """短 selected_text 不应生成兜底 citation（避免出现无关单一引用）"""
        citations = _build_selected_text_fallback_citations(
            "短标题",
            {"page_start": 1, "page_end": 1},
        )
        assert citations == []

    def test_long_selected_text_should_generate_fallback_citation(self):
        """较长 selected_text 可生成 1 条兜底 citation"""
        citations = _build_selected_text_fallback_citations(
            "这是一个足够长的框选文本片段，用于测试兜底引用生成逻辑是否生效。",
            {"page_start": 3, "page_end": 3},
        )
        assert len(citations) == 1
        assert citations[0]["ref"] == 1
        assert citations[0]["group_id"] == "selected-text"
        assert citations[0]["page_range"] == [3, 3]

    def test_build_fused_context_with_selected_ref(self):
        """selected_ref 传入时，框选文本标题应显式带引用编号"""
        fused = _build_fused_context(
            selected_text="框选内容",
            retrieval_context="",
            selected_page_info={"page_start": 2, "page_end": 2},
            selected_ref=1,
        )

        assert "[1]用户选中的文本（页码: 2）" in fused
        assert "框选内容" in fused
