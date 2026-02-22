"""
记忆分类标签模块

基于关键词规则的记忆自动分类标签器（零 LLM 调用）。
支持预定义标签类别：concept、fact、preference、method、conclusion、correction。
"""
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.memory_store import MemoryEntry

logger = logging.getLogger(__name__)

# 预定义标签及其关键词模式
# 每个标签对应一组关键词，内容中包含任一关键词即匹配该标签
TAG_PATTERNS: dict[str, list[str]] = {
    "concept": ["定义", "概念", "是指", "refers to", "definition", "meaning of"],
    "fact": ["数据", "统计", "百分比", "percent", "data", "statistics"],
    "preference": ["喜欢", "偏好", "prefer", "习惯", "倾向", "favorite"],
    "method": ["方法", "步骤", "流程", "algorithm", "approach", "procedure"],
    "conclusion": ["结论", "总结", "因此", "therefore", "conclude", "综上"],
    "correction": ["纠正", "错误", "不对", "correct", "wrong", "mistake"],
}

# fact 标签的额外正则模式：匹配数字、百分比、日期等
_FACT_REGEX = re.compile(
    r"\d+%"           # 百分比，如 50%
    r"|\d{4}[-/年]\d{1,2}"  # 日期，如 2024-01 或 2024年3
    r"|\d+\.\d+"      # 小数，如 3.14
)


class MemoryTagger:
    """基于关键词规则的记忆自动分类标签器（零 LLM 调用）"""

    # 合法的预定义标签集合
    VALID_TAGS = frozenset(TAG_PATTERNS.keys())

    def auto_tag(self, content: str) -> list[str]:
        """基于内容关键词自动推断标签

        Args:
            content: 记忆内容文本

        Returns:
            匹配到的标签列表（去重、排序）
        """
        if not content:
            return []

        content_lower = content.lower()
        tags: set[str] = set()

        # 关键词匹配
        for tag, keywords in TAG_PATTERNS.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    tags.add(tag)
                    break  # 一个标签只需匹配一个关键词即可

        # fact 标签的额外正则匹配（数字、百分比、日期）
        if "fact" not in tags and _FACT_REGEX.search(content):
            tags.add("fact")

        return sorted(tags)

    def add_tag(self, entry: "MemoryEntry", tag: str) -> "MemoryEntry":
        """手动添加标签

        Args:
            entry: 记忆条目
            tag: 要添加的标签

        Returns:
            修改后的记忆条目
        """
        if tag not in entry.tags:
            entry.tags.append(tag)
        return entry

    def remove_tag(self, entry: "MemoryEntry", tag: str) -> "MemoryEntry":
        """手动移除标签

        Args:
            entry: 记忆条目
            tag: 要移除的标签

        Returns:
            修改后的记忆条目
        """
        if tag in entry.tags:
            entry.tags.remove(tag)
        return entry
