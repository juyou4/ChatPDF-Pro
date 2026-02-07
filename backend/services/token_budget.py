"""
Token 预算管理器模块

提供语言感知的 Token 估算和预算管理功能。
支持中英文混合文本的 Token 数估算，以及在 Token 预算内
对语义意群进行粒度降级调整。

Token 估算策略：
- 中文字符（CJK Unicode 范围）：1.5 字符/token
- 英文/ASCII 字符：4 字符/token
- 混合文本按字符类型加权计算
"""

import logging
import math
from typing import List, Optional

logger = logging.getLogger(__name__)

# 粒度降级顺序：full → digest → summary
GRANULARITY_LEVELS = ["full", "digest", "summary"]

# 粒度对应的文本属性名映射
GRANULARITY_TEXT_ATTR = {
    "full": "full_text",
    "digest": "digest",
    "summary": "summary",
}


def _is_cjk_char(char: str) -> bool:
    """判断字符是否为 CJK（中日韩）字符

    覆盖 Unicode CJK 统一表意文字的主要范围：
    - CJK 统一表意文字基本区 (U+4E00 - U+9FFF)
    - CJK 统一表意文字扩展 A (U+3400 - U+4DBF)
    - CJK 统一表意文字扩展 B (U+20000 - U+2A6DF)
    - CJK 兼容表意文字 (U+F900 - U+FAFF)
    - CJK 统一表意文字扩展 C-F (U+2A700 - U+2CEAF)
    - CJK 兼容表意文字补充 (U+2F800 - U+2FA1F)

    Args:
        char: 单个字符

    Returns:
        是否为 CJK 字符
    """
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)        # CJK 统一表意文字基本区
        or (0x3400 <= cp <= 0x4DBF)     # CJK 统一表意文字扩展 A
        or (0x20000 <= cp <= 0x2A6DF)   # CJK 统一表意文字扩展 B
        or (0xF900 <= cp <= 0xFAFF)     # CJK 兼容表意文字
        or (0x2A700 <= cp <= 0x2CEAF)   # CJK 统一表意文字扩展 C-F
        or (0x2F800 <= cp <= 0x2FA1F)   # CJK 兼容表意文字补充
    )


class TokenBudgetManager:
    """Token 预算管理器

    负责估算文本的 Token 数量，并在 Token 预算内对语义意群
    进行粒度降级调整，确保发送给 LLM 的上下文不超出限制。

    Attributes:
        max_tokens: 最大 Token 预算
        reserve_for_answer: 预留给回答和系统提示词的 Token 数
    """

    def __init__(self, max_tokens: int = 8000, reserve_for_answer: int = 1500):
        """初始化 Token 预算管理器

        Args:
            max_tokens: 最大 Token 预算，默认 8000
            reserve_for_answer: 预留给回答和系统提示词的 Token 数，默认 1500
        """
        self.max_tokens = max_tokens
        self.reserve_for_answer = reserve_for_answer

    @property
    def available_tokens(self) -> int:
        """可用于上下文的 Token 数 = max_tokens - reserve_for_answer

        Returns:
            可用 Token 数
        """
        return self.max_tokens - self.reserve_for_answer

    def estimate_tokens(self, text: str) -> int:
        """语言感知的 Token 估算

        根据字符类型分别计算 Token 数：
        - 中文字符（CJK Unicode 范围）：1.5 字符/token → 每个字符贡献 1/1.5 个 token
        - 英文/ASCII 字符：4 字符/token → 每个字符贡献 1/4 个 token
        - 混合文本按字符类型加权计算，最终向上取整

        Args:
            text: 待估算的文本

        Returns:
            估算的 Token 数（向上取整）
        """
        if not text:
            return 0

        cjk_count = 0
        other_count = 0

        for char in text:
            if _is_cjk_char(char):
                cjk_count += 1
            else:
                other_count += 1

        # 中文：1.5 字符/token → token 数 = 字符数 / 1.5
        # 英文：4 字符/token → token 数 = 字符数 / 4
        token_estimate = cjk_count / 1.5 + other_count / 4

        return math.ceil(token_estimate)

    def fit_within_budget(
        self,
        groups: List[dict],
        max_tokens: Optional[int] = None,
    ) -> List[dict]:
        """在 Token 预算内调整意群粒度

        遍历意群列表，对每个意群：
        1. 根据当前粒度获取对应文本并估算 Token 数
        2. 如果累计 Token 超预算，尝试降级粒度（full→digest→summary）
        3. 降级后仍超预算则停止添加更多意群

        同一意群不会同时出现多种粒度。

        Args:
            groups: 意群列表，每项格式为 {"group": SemanticGroup, "granularity": str, "tokens": int}
            max_tokens: 可用 Token 预算，默认使用 self.available_tokens

        Returns:
            调整后的意群列表 [{"group": SemanticGroup, "granularity": str, "tokens": int}]
        """
        if max_tokens is None:
            max_tokens = self.available_tokens

        result = []
        used_tokens = 0

        for item in groups:
            group = item["group"]
            granularity = item["granularity"]

            # 获取当前粒度在降级顺序中的起始位置
            try:
                start_idx = GRANULARITY_LEVELS.index(granularity)
            except ValueError:
                # 未知粒度，默认从 full 开始
                logger.warning(f"未知粒度 '{granularity}'，默认使用 'full'")
                start_idx = 0

            # 尝试当前粒度及其降级粒度
            added = False
            for level_idx in range(start_idx, len(GRANULARITY_LEVELS)):
                level = GRANULARITY_LEVELS[level_idx]
                text_attr = GRANULARITY_TEXT_ATTR[level]
                text = getattr(group, text_attr, "")

                tokens = self.estimate_tokens(text)

                if used_tokens + tokens <= max_tokens:
                    # 在预算内，添加该意群
                    result.append({
                        "group": group,
                        "granularity": level,
                        "tokens": tokens,
                    })
                    used_tokens += tokens
                    added = True
                    break
                # 超预算，尝试下一个更低的粒度

            if not added:
                # 所有粒度都超预算，停止添加更多意群
                logger.info(
                    f"Token 预算已用尽（已用 {used_tokens}/{max_tokens}），"
                    f"停止添加意群 {group.group_id}"
                )
                break

        return result
