"""测试 selected_text_locator 模块

验证需求 5.1, 5.2, 5.3：
- 通过文本匹配定位 selected_text 所在页码
- 跨页文本返回起始和结束页码范围
- 无法匹配时返回默认页码 1 并记录警告日志
"""
import sys
import os
import logging

import pytest

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.selected_text_locator import locate_selected_text


# ---- 测试数据 ----

SAMPLE_PAGES = [
    {"page": 1, "content": "第一章 引言。本文介绍了机器学习的基本概念和应用场景。"},
    {"page": 2, "content": "第二章 线性回归。线性回归是最基础的监督学习算法之一。"},
    {"page": 3, "content": "第三章 神经网络。深度学习通过多层神经网络实现特征提取。"},
    {"page": 4, "content": "第四章 总结。本文对机器学习的主要方法进行了综述。"},
]


class TestExactMatch:
    """测试精确子串匹配（需求 5.1）"""

    def test_match_on_first_page(self):
        """匹配第一页内容"""
        result = locate_selected_text("机器学习的基本概念", SAMPLE_PAGES)
        assert result == {"page_start": 1, "page_end": 1}

    def test_match_on_middle_page(self):
        """匹配中间页内容"""
        result = locate_selected_text("线性回归是最基础的监督学习算法", SAMPLE_PAGES)
        assert result == {"page_start": 2, "page_end": 2}

    def test_match_on_last_page(self):
        """匹配最后一页内容"""
        result = locate_selected_text("主要方法进行了综述", SAMPLE_PAGES)
        assert result == {"page_start": 4, "page_end": 4}

    def test_match_full_page_content(self):
        """匹配整页内容"""
        result = locate_selected_text(
            "第二章 线性回归。线性回归是最基础的监督学习算法之一。",
            SAMPLE_PAGES,
        )
        assert result == {"page_start": 2, "page_end": 2}


class TestCrossPageMatch:
    """测试跨页检测（需求 5.2）"""

    def test_text_spans_two_pages(self):
        """文本跨越两页时返回页码范围"""
        # 构造跨页场景：selected_text 同时出现在第 1 页和第 2 页
        pages = [
            {"page": 1, "content": "机器学习是人工智能的一个分支"},
            {"page": 2, "content": "机器学习包括监督学习和无监督学习"},
        ]
        result = locate_selected_text("机器学习", pages)
        assert result == {"page_start": 1, "page_end": 2}

    def test_text_spans_multiple_pages(self):
        """文本跨越多页时返回最小和最大页码"""
        pages = [
            {"page": 1, "content": "深度学习概述"},
            {"page": 2, "content": "其他内容"},
            {"page": 3, "content": "深度学习应用"},
            {"page": 5, "content": "深度学习总结"},
        ]
        result = locate_selected_text("深度学习", pages)
        assert result == {"page_start": 1, "page_end": 5}


class TestFuzzyMatch:
    """测试模糊匹配（前 80 字符）"""

    def test_long_text_fuzzy_match(self):
        """长文本通过前 80 字符模糊匹配"""
        # 页面 2 内容很长，selected_text 完整文本不在页面中，
        # 但 selected_text 的前 80 字符是页面 2 内容的子串
        page2_content = (
            "线性回归是监督学习中最基础的算法，它通过最小化损失函数来拟合数据，"
            "广泛应用于预测和分析领域。梯度下降法是求解线性回归参数的常用方法，"
            "通过迭代更新参数使损失函数逐步减小。"
        )
        pages = [
            {"page": 1, "content": "第一章 引言内容"},
            {"page": 2, "content": page2_content},
        ]
        # 前 80 字符完全在 page2_content 中，但后面追加了不存在的内容
        long_text = page2_content[:90] + "这部分内容不在任何页面中" * 5
        result = locate_selected_text(long_text, pages)
        assert result == {"page_start": 2, "page_end": 2}


class TestFallback:
    """测试回退行为（需求 5.3）"""

    def test_no_match_returns_default(self):
        """无法匹配时返回默认页码 1"""
        result = locate_selected_text("完全不存在的文本内容", SAMPLE_PAGES)
        assert result == {"page_start": 1, "page_end": 1}

    def test_no_match_logs_warning(self, caplog):
        """无法匹配时记录警告日志"""
        with caplog.at_level(logging.WARNING):
            locate_selected_text("完全不存在的文本", SAMPLE_PAGES)
        assert any("无法在任何页面中匹配到" in msg for msg in caplog.messages)

    def test_empty_selected_text(self):
        """空 selected_text 返回默认页码"""
        result = locate_selected_text("", SAMPLE_PAGES)
        assert result == {"page_start": 1, "page_end": 1}

    def test_whitespace_selected_text(self):
        """纯空白 selected_text 返回默认页码"""
        result = locate_selected_text("   ", SAMPLE_PAGES)
        assert result == {"page_start": 1, "page_end": 1}

    def test_empty_pages(self):
        """空页面列表返回默认页码"""
        result = locate_selected_text("一些文本", [])
        assert result == {"page_start": 1, "page_end": 1}

    def test_none_pages(self):
        """None 页面列表返回默认页码"""
        result = locate_selected_text("一些文本", None)
        assert result == {"page_start": 1, "page_end": 1}


class TestEdgeCases:
    """测试边界情况"""

    def test_page_with_empty_content(self):
        """页面 content 为空时不报错"""
        pages = [
            {"page": 1, "content": ""},
            {"page": 2, "content": "有内容的页面"},
        ]
        result = locate_selected_text("有内容的页面", pages)
        assert result == {"page_start": 2, "page_end": 2}

    def test_page_missing_content_key(self):
        """页面缺少 content 键时不报错"""
        pages = [
            {"page": 1},
            {"page": 2, "content": "正常内容"},
        ]
        result = locate_selected_text("正常内容", pages)
        assert result == {"page_start": 2, "page_end": 2}

    def test_page_missing_page_key(self):
        """页面缺少 page 键时使用默认值 1"""
        pages = [
            {"content": "测试内容"},
        ]
        result = locate_selected_text("测试内容", pages)
        assert result == {"page_start": 1, "page_end": 1}

    def test_single_page(self):
        """单页文档"""
        pages = [{"page": 1, "content": "唯一的页面内容"}]
        result = locate_selected_text("唯一的页面", pages)
        assert result == {"page_start": 1, "page_end": 1}
