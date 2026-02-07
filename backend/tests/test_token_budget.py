"""
Token 预算管理器单元测试

测试 TokenBudgetManager 的核心功能：
- 语言感知的 Token 估算
- 可用 Token 计算
- 预算内粒度降级逻辑
"""

import math

import pytest

from services.token_budget import TokenBudgetManager, _is_cjk_char
from services.semantic_group_service import SemanticGroup


# ---- 辅助函数 ----

def _make_group(
    group_id: str = "group-0",
    full_text: str = "这是完整文本",
    digest: str = "精要",
    summary: str = "摘要",
    chunk_indices: list = None,
    page_range: tuple = (1, 1),
) -> SemanticGroup:
    """创建测试用的 SemanticGroup 对象"""
    return SemanticGroup(
        group_id=group_id,
        chunk_indices=chunk_indices or [0],
        char_count=len(full_text),
        summary=summary,
        digest=digest,
        full_text=full_text,
        keywords=["测试"],
        page_range=page_range,
        summary_status="ok",
        llm_meta=None,
    )


# ---- _is_cjk_char 测试 ----

class TestIsCjkChar:
    """测试 CJK 字符判断函数"""

    def test_常见中文字符(self):
        """常见中文字符应被识别为 CJK"""
        assert _is_cjk_char("中") is True
        assert _is_cjk_char("文") is True
        assert _is_cjk_char("字") is True

    def test_英文字符(self):
        """英文字母不应被识别为 CJK"""
        assert _is_cjk_char("a") is False
        assert _is_cjk_char("Z") is False

    def test_数字和标点(self):
        """数字和标点不应被识别为 CJK"""
        assert _is_cjk_char("1") is False
        assert _is_cjk_char(".") is False
        assert _is_cjk_char(" ") is False


# ---- estimate_tokens 测试 ----

class TestEstimateTokens:
    """测试语言感知的 Token 估算"""

    def test_空字符串(self):
        """空字符串应返回 0"""
        manager = TokenBudgetManager()
        assert manager.estimate_tokens("") == 0

    def test_纯中文文本(self):
        """纯中文文本：1.5 字符/token"""
        manager = TokenBudgetManager()
        # 6 个中文字符 → 6 / 1.5 = 4 tokens
        assert manager.estimate_tokens("你好世界测试") == 4

    def test_纯英文文本(self):
        """纯英文文本：4 字符/token"""
        manager = TokenBudgetManager()
        # 8 个英文字符 → 8 / 4 = 2 tokens
        assert manager.estimate_tokens("hello wo") == 2

    def test_中英混合文本(self):
        """中英混合文本按字符类型加权"""
        manager = TokenBudgetManager()
        # "你好hello" → 2 个中文 + 5 个英文
        # 2/1.5 + 5/4 = 1.333 + 1.25 = 2.583 → ceil = 3
        assert manager.estimate_tokens("你好hello") == 3

    def test_向上取整(self):
        """Token 估算应使用 math.ceil 向上取整"""
        manager = TokenBudgetManager()
        # 1 个中文字符 → 1 / 1.5 = 0.667 → ceil = 1
        assert manager.estimate_tokens("中") == 1

    def test_纯中文比等长纯英文估算值更大(self):
        """纯中文文本的 Token 估算应大于等长纯英文文本"""
        manager = TokenBudgetManager()
        # 10 个中文字符 → 10 / 1.5 ≈ 6.67 → 7
        cn_tokens = manager.estimate_tokens("一二三四五六七八九十")
        # 10 个英文字符 → 10 / 4 = 2.5 → 3
        en_tokens = manager.estimate_tokens("abcdefghij")
        assert cn_tokens > en_tokens


# ---- available_tokens 测试 ----

class TestAvailableTokens:
    """测试可用 Token 计算"""

    def test_默认值(self):
        """默认配置：8000 - 1500 = 6500"""
        manager = TokenBudgetManager()
        assert manager.available_tokens == 6500

    def test_自定义值(self):
        """自定义配置应正确计算"""
        manager = TokenBudgetManager(max_tokens=10000, reserve_for_answer=2000)
        assert manager.available_tokens == 8000


# ---- fit_within_budget 测试 ----

class TestFitWithinBudget:
    """测试预算内粒度降级逻辑"""

    def test_空列表(self):
        """空意群列表应返回空列表"""
        manager = TokenBudgetManager()
        result = manager.fit_within_budget([])
        assert result == []

    def test_预算充足_保持原粒度(self):
        """预算充足时应保持原始粒度"""
        manager = TokenBudgetManager(max_tokens=10000, reserve_for_answer=0)
        group = _make_group(full_text="短文本")
        groups = [{"group": group, "granularity": "full", "tokens": 0}]

        result = manager.fit_within_budget(groups)

        assert len(result) == 1
        assert result[0]["granularity"] == "full"
        assert result[0]["tokens"] > 0

    def test_预算不足_降级到digest(self):
        """预算不足时应从 full 降级到 digest"""
        manager = TokenBudgetManager()
        # 创建一个 full_text 很长但 digest 较短的意群
        group = _make_group(
            full_text="这是一段非常长的中文文本" * 500,  # 很长的全文
            digest="精要内容",  # 短精要
            summary="摘要",
        )

        groups = [{"group": group, "granularity": "full", "tokens": 0}]
        # 设置一个较小的预算，full 放不下但 digest 可以
        result = manager.fit_within_budget(groups, max_tokens=100)

        assert len(result) == 1
        assert result[0]["granularity"] == "digest"

    def test_预算不足_降级到summary(self):
        """预算严重不足时应从 full 降级到 summary"""
        manager = TokenBudgetManager()
        group = _make_group(
            full_text="这是一段非常长的中文文本" * 500,
            digest="这也是一段比较长的精要内容" * 100,
            summary="摘要",
        )

        groups = [{"group": group, "granularity": "full", "tokens": 0}]
        # 设置一个很小的预算，只有 summary 能放下
        result = manager.fit_within_budget(groups, max_tokens=10)

        assert len(result) == 1
        assert result[0]["granularity"] == "summary"

    def test_预算耗尽_停止添加(self):
        """所有粒度都超预算时应停止添加"""
        manager = TokenBudgetManager()
        group = _make_group(
            full_text="这是一段非常长的中文文本" * 500,
            digest="这也是一段比较长的精要内容" * 100,
            summary="这个摘要也很长" * 50,
        )

        groups = [{"group": group, "granularity": "full", "tokens": 0}]
        # 预算为 1，任何粒度都放不下
        result = manager.fit_within_budget(groups, max_tokens=1)

        assert len(result) == 0

    def test_多个意群_部分添加(self):
        """多个意群时，预算耗尽后停止添加后续意群"""
        manager = TokenBudgetManager()
        group1 = _make_group(group_id="group-0", full_text="短文本", digest="精要", summary="摘")
        group2 = _make_group(group_id="group-1", full_text="短文本二", digest="精要二", summary="摘")
        group3 = _make_group(
            group_id="group-2",
            full_text="这是一段非常长的中文文本" * 500,
            digest="这也是一段比较长的精要内容" * 100,
            summary="这个摘要也很长" * 50,
        )

        groups = [
            {"group": group1, "granularity": "full", "tokens": 0},
            {"group": group2, "granularity": "full", "tokens": 0},
            {"group": group3, "granularity": "full", "tokens": 0},
        ]

        # 给一个较小的预算，前两个能放下，第三个放不下
        result = manager.fit_within_budget(groups, max_tokens=10)

        # 前两个短文本应该能放下
        assert len(result) >= 1
        assert len(result) <= 2
        # 第三个不应该出现
        result_ids = [r["group"].group_id for r in result]
        assert "group-2" not in result_ids

    def test_累计token不超预算(self):
        """结果中所有意群的累计 Token 数不应超过预算"""
        manager = TokenBudgetManager()
        groups_data = []
        for i in range(5):
            group = _make_group(
                group_id=f"group-{i}",
                full_text=f"这是第{i}个意群的完整文本内容，包含一些详细信息。",
                digest=f"第{i}个精要",
                summary=f"摘要{i}",
            )
            groups_data.append({"group": group, "granularity": "full", "tokens": 0})

        budget = 50
        result = manager.fit_within_budget(groups_data, max_tokens=budget)

        total_tokens = sum(r["tokens"] for r in result)
        assert total_tokens <= budget

    def test_digest起始粒度_不会升级(self):
        """初始粒度为 digest 时，不应升级到 full，只能降级到 summary"""
        manager = TokenBudgetManager()
        group = _make_group(
            full_text="这是一段非常长的中文文本" * 500,
            digest="精要内容比较短",
            summary="摘要",
        )

        groups = [{"group": group, "granularity": "digest", "tokens": 0}]
        result = manager.fit_within_budget(groups, max_tokens=1000)

        assert len(result) == 1
        # 粒度应该是 digest 或 summary，不应该是 full
        assert result[0]["granularity"] in ("digest", "summary")

    def test_tokens字段被正确填充(self):
        """返回结果中的 tokens 字段应被正确计算填充"""
        manager = TokenBudgetManager()
        group = _make_group(full_text="测试文本")
        groups = [{"group": group, "granularity": "full", "tokens": 0}]

        result = manager.fit_within_budget(groups, max_tokens=1000)

        assert len(result) == 1
        expected_tokens = manager.estimate_tokens("测试文本")
        assert result[0]["tokens"] == expected_tokens
