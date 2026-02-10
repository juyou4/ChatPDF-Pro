"""测试 _rank_groups_by_results 函数的 chunk_indices 精确映射优化

验证需求 7.1, 7.2, 7.3：
- 使用 chunk_indices 反向映射进行精确匹配
- 构建 chunk_index → group_id 的反向映射表
- 无法通过索引匹配时回退到子串匹配
"""
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import pytest

# 将 backend 目录添加到 sys.path，以便导入 services 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class MockSemanticGroup:
    """模拟 SemanticGroup 数据结构，用于测试"""
    group_id: str
    chunk_indices: List[int]
    char_count: int = 0
    summary: str = ""
    digest: str = ""
    full_text: str = ""
    keywords: List[str] = field(default_factory=list)
    page_range: Tuple[int, int] = (1, 1)
    summary_status: str = "ok"
    llm_meta: Optional[dict] = None


# 直接导入被测函数
from services.embedding_service import _rank_groups_by_results


class TestRankGroupsByResultsWithChunkIndices:
    """测试使用 chunk_indices 精确映射的意群排序"""

    def test_chunk_indices_exact_match(self):
        """验证通过 chunk_indices 精确匹配能正确映射 chunk 到意群"""
        chunks = ["chunk_0 文本内容", "chunk_1 文本内容", "chunk_2 文本内容", "chunk_3 文本内容"]

        groups = [
            MockSemanticGroup(
                group_id="group-0",
                chunk_indices=[0, 1],
                full_text="chunk_0 文本内容\nchunk_1 文本内容",
            ),
            MockSemanticGroup(
                group_id="group-1",
                chunk_indices=[2, 3],
                full_text="chunk_2 文本内容\nchunk_3 文本内容",
            ),
        ]

        results = [
            {"chunk": "chunk_2 文本内容", "similarity": 0.9},
            {"chunk": "chunk_0 文本内容", "similarity": 0.7},
        ]

        ranked, best_chunks = _rank_groups_by_results(groups, results, chunks=chunks)

        # group-1 排名第一（chunk_2 排名 0），group-0 排名第二（chunk_0 排名 1）
        assert len(ranked) == 2
        assert ranked[0].group_id == "group-1"
        assert ranked[1].group_id == "group-0"
        # 验证最佳 chunk 文本
        assert best_chunks["group-1"] == "chunk_2 文本内容"
        assert best_chunks["group-0"] == "chunk_0 文本内容"

    def test_fallback_to_substring_when_no_chunks(self):
        """验证 chunks=None 时回退到子串匹配"""
        groups = [
            MockSemanticGroup(
                group_id="group-0",
                chunk_indices=[0, 1],
                full_text="这是第一个意群的完整文本内容，包含多个分块",
            ),
            MockSemanticGroup(
                group_id="group-1",
                chunk_indices=[2, 3],
                full_text="这是第二个意群的完整文本内容，也包含多个分块",
            ),
        ]

        results = [
            {"chunk": "第二个意群的完整文本", "similarity": 0.8},
            {"chunk": "第一个意群的完整文本", "similarity": 0.6},
        ]

        # chunks=None，应回退到子串匹配
        ranked, best_chunks = _rank_groups_by_results(groups, results, chunks=None)

        assert len(ranked) == 2
        assert ranked[0].group_id == "group-1"
        assert ranked[1].group_id == "group-0"

    def test_fallback_to_substring_when_index_not_found(self):
        """验证 chunk 文本不在 chunks 列表中时回退到子串匹配"""
        chunks = ["chunk_0 原始文本", "chunk_1 原始文本"]

        groups = [
            MockSemanticGroup(
                group_id="group-0",
                chunk_indices=[0, 1],
                full_text="chunk_0 原始文本\nchunk_1 原始文本\n额外的修改文本",
            ),
        ]

        # 搜索结果中的 chunk 文本不在 chunks 列表中（可能经过了修改）
        # 但它是 group full_text 的子串
        results = [
            {"chunk": "额外的修改文本", "similarity": 0.7},
        ]

        ranked, best_chunks = _rank_groups_by_results(groups, results, chunks=chunks)

        # 应通过子串匹配找到 group-0
        assert len(ranked) == 1
        assert ranked[0].group_id == "group-0"

    def test_empty_groups_returns_empty(self):
        """验证空意群列表返回空结果"""
        results = [{"chunk": "some text", "similarity": 0.5}]
        ranked, best_chunks = _rank_groups_by_results([], results, chunks=["some text"])
        assert ranked == []
        assert best_chunks == {}

    def test_empty_results_returns_empty(self):
        """验证空搜索结果返回空结果"""
        groups = [
            MockSemanticGroup(group_id="group-0", chunk_indices=[0], full_text="text"),
        ]
        ranked, best_chunks = _rank_groups_by_results(groups, [], chunks=["text"])
        assert ranked == []
        assert best_chunks == {}

    def test_multiple_chunks_same_group(self):
        """验证同一意群的多个 chunk 出现在结果中时，保留最高排名"""
        chunks = ["chunk_a", "chunk_b", "chunk_c"]

        groups = [
            MockSemanticGroup(
                group_id="group-0",
                chunk_indices=[0, 1],
                full_text="chunk_a\nchunk_b",
            ),
            MockSemanticGroup(
                group_id="group-1",
                chunk_indices=[2],
                full_text="chunk_c",
            ),
        ]

        results = [
            {"chunk": "chunk_c", "similarity": 0.9},   # rank 0 -> group-1
            {"chunk": "chunk_b", "similarity": 0.8},   # rank 1 -> group-0
            {"chunk": "chunk_a", "similarity": 0.7},   # rank 2 -> group-0（已有更好排名）
        ]

        ranked, best_chunks = _rank_groups_by_results(groups, results, chunks=chunks)

        assert len(ranked) == 2
        assert ranked[0].group_id == "group-1"  # rank 0
        assert ranked[1].group_id == "group-0"  # rank 1（最佳排名）
        # group-1 最佳 chunk 是 chunk_c
        assert best_chunks["group-1"] == "chunk_c"
        # group-0 最佳 chunk 是 chunk_b（相似度 0.8 > 0.7）
        assert best_chunks["group-0"] == "chunk_b"

    def test_similarity_preserved_max(self):
        """验证同一意群的多个 chunk 保留最高相似度"""
        chunks = ["chunk_a", "chunk_b"]

        groups = [
            MockSemanticGroup(
                group_id="group-0",
                chunk_indices=[0, 1],
                full_text="chunk_a\nchunk_b",
            ),
        ]

        results = [
            {"chunk": "chunk_a", "similarity": 0.6},
            {"chunk": "chunk_b", "similarity": 0.9},
        ]

        ranked, best_chunks = _rank_groups_by_results(groups, results, chunks=chunks)

        # 应该有 1 个意群
        assert len(ranked) == 1
        assert ranked[0].group_id == "group-0"
        # 最佳 chunk 应该是 chunk_b（相似度 0.9 > 0.6）
        assert best_chunks["group-0"] == "chunk_b"

    def test_chunk_indices_priority_over_substring(self):
        """验证 chunk_indices 精确匹配优先于子串匹配

        当一个 chunk 文本同时是多个意群 full_text 的子串时，
        chunk_indices 映射能精确定位到正确的意群。
        """
        chunks = ["共享文本片段"]

        groups = [
            MockSemanticGroup(
                group_id="group-0",
                chunk_indices=[0],  # chunk 0 属于 group-0
                full_text="共享文本片段在这里出现",
            ),
            MockSemanticGroup(
                group_id="group-1",
                chunk_indices=[],  # group-1 不包含 chunk 0
                full_text="共享文本片段也在这里出现",
            ),
        ]

        results = [
            {"chunk": "共享文本片段", "similarity": 0.8},
        ]

        # 使用 chunk_indices 应精确匹配到 group-0
        ranked, best_chunks = _rank_groups_by_results(groups, results, chunks=chunks)

        assert len(ranked) == 1
        assert ranked[0].group_id == "group-0"

    def test_empty_chunk_text_skipped(self):
        """验证空 chunk 文本被跳过"""
        chunks = ["valid_chunk"]

        groups = [
            MockSemanticGroup(
                group_id="group-0",
                chunk_indices=[0],
                full_text="valid_chunk",
            ),
        ]

        results = [
            {"chunk": "", "similarity": 0.9},
            {"chunk": "valid_chunk", "similarity": 0.7},
        ]

        ranked, best_chunks = _rank_groups_by_results(groups, results, chunks=chunks)

        assert len(ranked) == 1
        assert ranked[0].group_id == "group-0"
        assert best_chunks["group-0"] == "valid_chunk"
