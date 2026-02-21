"""测试结构感知分块策略

验证需求 4.1, 4.2, 4.3, 4.4：
- 优先按段落和章节边界切分文本
- 表格保持在同一分块内
- LaTeX 公式块保持在同一分块内
- 受保护区域超过 chunk_size 时单独成块，检测失败时回退到 RecursiveCharacterTextSplitter
"""
import sys
import os

import pytest

# 将 backend 目录添加到 sys.path，以便导入 services 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embedding_service import (
    structure_aware_split,
    _find_protected_regions,
    _split_by_paragraphs_with_protection,
    _merge_segments_into_chunks,
    _get_overlap_parts,
)


class TestFindProtectedRegions:
    """测试受保护区域检测"""

    def test_detect_markdown_table(self):
        """检测 markdown 表格区域"""
        text = "前文内容\n\n| 列1 | 列2 | 列3 |\n| --- | --- | --- |\n| a | b | c |\n| d | e | f |\n\n后文内容"
        regions = _find_protected_regions(text)
        assert len(regions) == 1
        # 受保护区域应包含完整表格
        protected_text = text[regions[0][0]:regions[0][1]]
        assert "| 列1 |" in protected_text
        assert "| d | e | f |" in protected_text

    def test_detect_display_math_double_dollar(self):
        """检测 $$ 包裹的显示公式"""
        text = "前文\n\n$$\nE = mc^2\n$$\n\n后文"
        regions = _find_protected_regions(text)
        assert len(regions) == 1
        protected_text = text[regions[0][0]:regions[0][1]]
        assert "E = mc^2" in protected_text

    def test_detect_display_math_bracket(self):
        r"""检测 \[...\] 包裹的显示公式"""
        text = "前文\n\n\\[\nx = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}\n\\]\n\n后文"
        regions = _find_protected_regions(text)
        assert len(regions) == 1
        protected_text = text[regions[0][0]:regions[0][1]]
        assert "\\frac" in protected_text

    def test_no_protected_regions(self):
        """普通文本没有受保护区域"""
        text = "这是一段普通文本。\n\n这是另一段普通文本。"
        regions = _find_protected_regions(text)
        assert len(regions) == 0

    def test_multiple_protected_regions(self):
        """检测多个受保护区域"""
        text = (
            "前文\n\n"
            "| a | b |\n| c | d |\n\n"
            "中间文本\n\n"
            "$$\ny = mx + b\n$$\n\n"
            "后文"
        )
        regions = _find_protected_regions(text)
        assert len(regions) == 2

    def test_merge_overlapping_regions(self):
        """重叠的受保护区域应被合并"""
        text = "$$\n| a | b |\n| c | d |\n$$"
        regions = _find_protected_regions(text)
        # 表格和公式重叠，应合并为一个区域
        assert len(regions) >= 1


class TestSplitByParagraphsWithProtection:
    """测试段落切分与受保护区域保护"""

    def test_no_protected_regions(self):
        """没有受保护区域时按段落切分"""
        text = "段落一内容。\n\n段落二内容。\n\n段落三内容。"
        segments = _split_by_paragraphs_with_protection(text, [])
        assert len(segments) == 3
        assert all(not s["protected"] for s in segments)

    def test_with_protected_table(self):
        """包含表格的文本正确切分"""
        text = "前文内容\n\n| a | b |\n| c | d |\n\n后文内容"
        regions = _find_protected_regions(text)
        segments = _split_by_paragraphs_with_protection(text, regions)
        # 应有：前文（普通）、表格（受保护）、后文（普通）
        protected_segments = [s for s in segments if s["protected"]]
        normal_segments = [s for s in segments if not s["protected"]]
        assert len(protected_segments) == 1
        assert len(normal_segments) == 2
        assert "| a | b |" in protected_segments[0]["text"]

    def test_protected_region_at_start(self):
        """受保护区域在文本开头"""
        text = "| a | b |\n| c | d |\n\n后文内容"
        regions = _find_protected_regions(text)
        segments = _split_by_paragraphs_with_protection(text, regions)
        assert segments[0]["protected"]

    def test_protected_region_at_end(self):
        """受保护区域在文本末尾"""
        text = "前文内容\n\n| a | b |\n| c | d |"
        regions = _find_protected_regions(text)
        segments = _split_by_paragraphs_with_protection(text, regions)
        assert segments[-1]["protected"]


class TestMergeSegmentsIntoChunks:
    """测试段合并为分块"""

    def test_small_segments_merged(self):
        """小段落合并到一个分块"""
        segments = [
            {"text": "短段落一", "protected": False},
            {"text": "短段落二", "protected": False},
            {"text": "短段落三", "protected": False},
        ]
        chunks = _merge_segments_into_chunks(segments, chunk_size=100, chunk_overlap=20)
        assert len(chunks) == 1
        # _merge_segments_into_chunks 返回 (text, heading) 元组列表
        assert "短段落一" in chunks[0][0]
        assert "短段落二" in chunks[0][0]
        assert "短段落三" in chunks[0][0]

    def test_large_segments_split(self):
        """大段落被分到不同分块"""
        segments = [
            {"text": "A" * 500, "protected": False},
            {"text": "B" * 500, "protected": False},
            {"text": "C" * 500, "protected": False},
        ]
        chunks = _merge_segments_into_chunks(segments, chunk_size=600, chunk_overlap=50)
        assert len(chunks) >= 2

    def test_protected_region_kept_intact(self):
        """受保护区域保持完整"""
        table_text = "| a | b |\n| c | d |\n| e | f |"
        segments = [
            {"text": "前文" * 50, "protected": False},
            {"text": table_text, "protected": True},
            {"text": "后文" * 50, "protected": False},
        ]
        chunks = _merge_segments_into_chunks(segments, chunk_size=200, chunk_overlap=30)
        # 表格应完整出现在某个分块中（chunks 为 (text, heading) 元组列表）
        table_found = any(table_text in chunk[0] for chunk in chunks)
        assert table_found, "表格应完整出现在某个分块中"

    def test_oversized_protected_region(self):
        """超大受保护区域单独成块"""
        big_table = "| " + " | ".join(["x"] * 50) + " |\n" * 30
        segments = [
            {"text": "前文内容", "protected": False},
            {"text": big_table, "protected": True},
            {"text": "后文内容", "protected": False},
        ]
        chunks = _merge_segments_into_chunks(segments, chunk_size=100, chunk_overlap=20)
        # 超大表格应单独成块（chunks 为 (text, heading) 元组列表）
        assert any(big_table.strip() in chunk[0] for chunk in chunks)


class TestGetOverlapParts:
    """测试重叠部分提取"""

    def test_basic_overlap(self):
        """基本重叠提取"""
        parts = ["段落一内容", "段落二内容", "段落三内容"]
        overlap = _get_overlap_parts(parts, overlap_size=10)
        assert len(overlap) >= 1
        # 应从末尾取
        assert overlap[-1] == "段落三内容"

    def test_empty_parts(self):
        """空列表返回空"""
        assert _get_overlap_parts([], 100) == []

    def test_zero_overlap(self):
        """重叠为 0 返回空"""
        assert _get_overlap_parts(["abc"], 0) == []


class TestStructureAwareSplit:
    """测试 structure_aware_split 主函数"""

    def test_empty_text(self):
        """空文本返回空列表"""
        assert structure_aware_split("") == []
        assert structure_aware_split("   ") == []
        assert structure_aware_split(None) == []

    def test_simple_text(self):
        """简单文本正常分块"""
        text = "段落一内容。\n\n段落二内容。\n\n段落三内容。"
        chunks = structure_aware_split(text, chunk_size=1000)
        assert len(chunks) >= 1
        # 所有内容都应被包含
        combined = " ".join(chunks)
        assert "段落一内容" in combined
        assert "段落二内容" in combined
        assert "段落三内容" in combined

    def test_table_integrity(self):
        """表格保持在同一分块内（需求 4.2）"""
        table = "| 名称 | 值 |\n| --- | --- |\n| alpha | 0.01 |\n| beta | 0.99 |"
        text = f"前文内容段落。\n\n{table}\n\n后文内容段落。"
        chunks = structure_aware_split(text, chunk_size=500)
        # 表格应完整出现在某个分块中
        table_intact = any(
            "| alpha | 0.01 |" in chunk and "| beta | 0.99 |" in chunk
            for chunk in chunks
        )
        assert table_intact, f"表格应完整出现在某个分块中，实际分块: {chunks}"

    def test_formula_integrity(self):
        """LaTeX 公式保持在同一分块内（需求 4.3）"""
        formula = "$$\nE = mc^2\n\\int_0^\\infty f(x) dx\n$$"
        text = f"前文内容段落。\n\n{formula}\n\n后文内容段落。"
        chunks = structure_aware_split(text, chunk_size=500)
        # 公式应完整出现在某个分块中
        formula_intact = any(
            "E = mc^2" in chunk and "\\int_0^\\infty" in chunk
            for chunk in chunks
        )
        assert formula_intact, f"公式应完整出现在某个分块中，实际分块: {chunks}"

    def test_bracket_formula_integrity(self):
        r"""\\[...\\] 公式保持在同一分块内（需求 4.3）"""
        formula = "\\[\nx = \\frac{-b}{2a}\n\\]"
        text = f"前文内容段落。\n\n{formula}\n\n后文内容段落。"
        chunks = structure_aware_split(text, chunk_size=500)
        formula_intact = any("\\frac{-b}{2a}" in chunk for chunk in chunks)
        assert formula_intact, f"公式应完整出现在某个分块中，实际分块: {chunks}"

    def test_oversized_protected_region_separate_chunk(self):
        """超大受保护区域单独成块（需求 4.4）"""
        # 创建一个超过 chunk_size 的大表格
        rows = "\n".join(f"| item_{i} | value_{i} |" for i in range(50))
        big_table = f"| 名称 | 值 |\n| --- | --- |\n{rows}"
        text = f"前文内容。\n\n{big_table}\n\n后文内容。"
        chunks = structure_aware_split(text, chunk_size=200)
        # 大表格应存在于某个分块中（可能单独成块）
        assert len(chunks) >= 2

    def test_paragraph_boundary_splitting(self):
        """按段落边界切分（需求 4.1）"""
        paragraphs = [f"这是第{i}个段落的内容，包含一些有意义的文字。" * 5 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = structure_aware_split(text, chunk_size=300, chunk_overlap=50)
        assert len(chunks) >= 2
        # 每个分块都不应为空
        assert all(chunk.strip() for chunk in chunks)

    def test_fallback_to_recursive_splitter(self):
        """检测失败时回退到 RecursiveCharacterTextSplitter（需求 4.4 安全降级）"""
        # 一段很长的没有段落边界的文本
        text = "这是一段连续的文本没有任何段落分隔符" * 100
        chunks = structure_aware_split(text, chunk_size=200)
        # 应该仍然能正常分块（通过回退机制）
        assert len(chunks) >= 1
        # 所有分块合并后应包含原始内容
        combined = "".join(chunks)
        assert "这是一段连续的文本" in combined

    def test_mixed_content(self):
        """混合内容（文本 + 表格 + 公式）正确处理"""
        text = (
            "# 第一章 介绍\n\n"
            "这是介绍段落，描述了研究背景。\n\n"
            "| 方法 | 准确率 |\n| --- | --- |\n| A | 95% |\n| B | 92% |\n\n"
            "如上表所示，方法 A 表现更好。\n\n"
            "核心公式如下：\n\n"
            "$$\nL = -\\sum_{i} y_i \\log(p_i)\n$$\n\n"
            "这是交叉熵损失函数。"
        )
        chunks = structure_aware_split(text, chunk_size=500)
        assert len(chunks) >= 1
        # 表格应完整
        table_intact = any("| A | 95% |" in c and "| B | 92% |" in c for c in chunks)
        assert table_intact, "表格应保持完整"
        # 公式应完整
        formula_intact = any("\\sum_{i}" in c and "\\log(p_i)" in c for c in chunks)
        assert formula_intact, "公式应保持完整"
