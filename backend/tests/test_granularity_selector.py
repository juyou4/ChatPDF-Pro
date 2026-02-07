"""
粒度选择器单元测试

测试 GranularitySelector 的 select 和 select_mixed 方法，
验证查询类型映射规则和混合粒度位置分配规则的正确性。
"""

import sys
import os
import pytest

# 将 backend 目录添加到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.granularity_selector import GranularitySelector, GranularitySelection
from services.semantic_group_service import SemanticGroup


def _make_group(group_id: str, full_text: str = "测试文本") -> SemanticGroup:
    """创建测试用的 SemanticGroup 实例"""
    return SemanticGroup(
        group_id=group_id,
        chunk_indices=[0],
        char_count=len(full_text),
        summary="摘要",
        digest="精要内容",
        full_text=full_text,
        keywords=["关键词"],
        page_range=(1, 1),
        summary_status="ok",
        llm_meta=None,
    )


class TestGranularitySelect:
    """测试 select 方法：根据查询类型映射粒度和 max_groups"""

    def setup_method(self):
        self.selector = GranularitySelector()
        self.groups = [_make_group(f"group-{i}") for i in range(5)]

    def test_overview_query_returns_summary_granularity(self):
        """概览性查询应返回 summary 粒度，最多 10 个意群"""
        result = self.selector.select("请总结本文的主要内容", self.groups)
        assert isinstance(result, GranularitySelection)
        assert result.granularity == "summary"
        assert result.max_groups == 10
        assert result.query_type == "overview"

    def test_extraction_query_returns_full_granularity(self):
        """提取性查询应返回 full 粒度，最多 3 个意群"""
        result = self.selector.select("请列出具体的数据和公式", self.groups)
        assert result.granularity == "full"
        assert result.max_groups == 3
        assert result.query_type == "extraction"

    def test_analytical_query_returns_digest_granularity(self):
        """分析性查询应返回 digest 粒度，最多 5 个意群"""
        result = self.selector.select("为什么作者选择这种方法？", self.groups)
        assert result.granularity == "digest"
        assert result.max_groups == 5
        assert result.query_type == "analytical"

    def test_specific_query_returns_digest_granularity(self):
        """具体性查询应返回 digest 粒度，最多 5 个意群"""
        result = self.selector.select("transformer 架构是什么", self.groups)
        assert result.granularity == "digest"
        assert result.max_groups == 5
        assert result.query_type == "specific"

    def test_empty_query_returns_specific_type(self):
        """空查询应被分类为 specific 类型"""
        result = self.selector.select("", self.groups)
        assert result.query_type == "specific"
        assert result.granularity == "digest"
        assert result.max_groups == 5

    def test_result_contains_reasoning(self):
        """选择结果应包含非空的选择理由"""
        result = self.selector.select("概述全文", self.groups)
        assert result.reasoning
        assert len(result.reasoning) > 0

    def test_english_overview_query(self):
        """英文概览性查询也应正确分类"""
        result = self.selector.select("Give me a summary of this document", self.groups)
        assert result.query_type == "overview"
        assert result.granularity == "summary"

    def test_english_analytical_query(self):
        """英文分析性查询也应正确分类"""
        result = self.selector.select("Why does the author use this approach?", self.groups)
        assert result.query_type == "analytical"
        assert result.granularity == "digest"


class TestGranularitySelectMixed:
    """测试 select_mixed 方法：按排名位置分配混合粒度"""

    def setup_method(self):
        self.selector = GranularitySelector()

    def test_empty_list_returns_empty(self):
        """空意群列表应返回空结果"""
        result = self.selector.select_mixed("查询", [])
        assert result == []

    def test_single_group_gets_full_granularity(self):
        """单个意群应分配 full 粒度"""
        groups = [_make_group("group-0")]
        result = self.selector.select_mixed("查询", groups)
        assert len(result) == 1
        assert result[0]["granularity"] == "full"
        assert result[0]["group"].group_id == "group-0"
        assert result[0]["tokens"] == 0

    def test_two_groups_mixed_granularity(self):
        """两个意群：第1=full，第2=digest"""
        groups = [_make_group(f"group-{i}") for i in range(2)]
        result = self.selector.select_mixed("查询", groups)
        assert len(result) == 2
        assert result[0]["granularity"] == "full"
        assert result[1]["granularity"] == "digest"

    def test_three_groups_mixed_granularity(self):
        """三个意群：第1=full，第2-3=digest"""
        groups = [_make_group(f"group-{i}") for i in range(3)]
        result = self.selector.select_mixed("查询", groups)
        assert len(result) == 3
        assert result[0]["granularity"] == "full"
        assert result[1]["granularity"] == "digest"
        assert result[2]["granularity"] == "digest"

    def test_five_groups_mixed_granularity(self):
        """五个意群：第1=full，第2-3=digest，第4-5=summary"""
        groups = [_make_group(f"group-{i}") for i in range(5)]
        result = self.selector.select_mixed("查询", groups)
        assert len(result) == 5
        assert result[0]["granularity"] == "full"
        assert result[1]["granularity"] == "digest"
        assert result[2]["granularity"] == "digest"
        assert result[3]["granularity"] == "summary"
        assert result[4]["granularity"] == "summary"

    def test_tokens_initialized_to_zero(self):
        """所有意群的 tokens 字段应初始化为 0"""
        groups = [_make_group(f"group-{i}") for i in range(4)]
        result = self.selector.select_mixed("查询", groups)
        for item in result:
            assert item["tokens"] == 0

    def test_group_objects_preserved(self):
        """返回结果中的 group 对象应与输入一致"""
        groups = [_make_group(f"group-{i}") for i in range(3)]
        result = self.selector.select_mixed("查询", groups)
        for i, item in enumerate(result):
            assert item["group"] is groups[i]

    def test_many_groups_later_all_summary(self):
        """大量意群时，第4个及之后全部为 summary"""
        groups = [_make_group(f"group-{i}") for i in range(10)]
        result = self.selector.select_mixed("查询", groups)
        assert len(result) == 10
        # 第 4-10 名全部为 summary
        for i in range(3, 10):
            assert result[i]["granularity"] == "summary", (
                f"第 {i+1} 名应为 summary，实际为 {result[i]['granularity']}"
            )
