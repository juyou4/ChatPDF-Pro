"""智能记忆上下文注入模块

根据记忆的 memory_tier 和 tags 将记忆分组注入到 system prompt 的不同区域。
支持 token 预算截断、记忆摘要和来源标注。
"""
import logging

logger = logging.getLogger(__name__)

# 记忆层级显示顺序（长期记忆优先）
TIER_ORDER = ["long_term", "short_term", "working", "archived"]

# 记忆层级 → 来源标注映射
TIER_LABELS = {
    "long_term": "[长期记忆]",
    "short_term": "[文档记忆]",
    "working": "[工作记忆]",
    "archived": "[归档记忆]",
}


def _estimate_tokens(text: str) -> int:
    """简单 token 估算：中文文本长度 // 2"""
    return max(len(text) // 2, 1) if text else 0


class ContextInjector:
    """智能记忆上下文注入器

    将记忆按 memory_tier 分组，按 importance 截断到 token 预算内，
    格式化为 Markdown 文本注入到 system prompt 中。
    """

    def __init__(self, token_budget: int = 800):
        """初始化注入器

        Args:
            token_budget: 默认 token 预算，默认 800
        """
        self.token_budget = token_budget

    def inject(self, system_prompt: str, memories: list[dict],
               token_budget: int = None) -> str:
        """将记忆按类型分组注入到 system prompt

        注入策略：
        1. 按 importance 降序截断到 token 预算内
        2. 超过 10 条时生成摘要
        3. 按 memory_tier 分组，长期记忆在前
        4. 每条记忆添加来源标注

        Args:
            system_prompt: 原始 system prompt
            memories: 记忆字典列表，每条包含 content, memory_tier, importance 等
            token_budget: 可选的 token 预算覆盖值

        Returns:
            注入记忆后的 system prompt
        """
        if not memories:
            return system_prompt

        budget = token_budget if token_budget is not None else self.token_budget

        # 按 importance 降序截断到 token 预算内
        truncated = self._truncate_by_budget(memories, budget)

        if not truncated:
            return system_prompt

        # 超过 10 条时生成摘要
        if len(truncated) > 10:
            summary = self._summarize_memories(truncated)
            return f"{system_prompt}\n\n---\n## 记忆摘要\n{summary}"

        # 按 memory_tier 分组
        grouped = self._group_memories(truncated)

        # 按层级顺序格式化各分组
        blocks = []
        for tier in TIER_ORDER:
            if tier in grouped and grouped[tier]:
                block = self._format_memory_block(tier, grouped[tier])
                blocks.append(block)

        if not blocks:
            return system_prompt

        memory_text = "\n\n".join(blocks)
        return f"{system_prompt}\n\n---\n{memory_text}"

    def _group_memories(self, memories: list[dict]) -> dict[str, list[dict]]:
        """按 memory_tier 分组

        Args:
            memories: 记忆字典列表

        Returns:
            tier -> 记忆列表的映射
        """
        groups: dict[str, list[dict]] = {}
        for mem in memories:
            tier = mem.get("memory_tier", "short_term")
            if tier not in groups:
                groups[tier] = []
            groups[tier].append(mem)
        return groups

    def _truncate_by_budget(self, memories: list[dict],
                            budget: int) -> list[dict]:
        """按 importance 降序截断到 token 预算内

        优先保留 importance 最高的记忆，逐条累加 token 直到超出预算。

        Args:
            memories: 记忆字典列表
            budget: token 预算

        Returns:
            截断后的记忆列表（按 importance 降序）
        """
        if budget <= 0:
            return []

        # 按 importance 降序排序
        sorted_mems = sorted(
            memories,
            key=lambda m: m.get("importance", 0.0),
            reverse=True,
        )

        result = []
        used_tokens = 0
        for mem in sorted_mems:
            content = mem.get("content", "")
            tokens = _estimate_tokens(content)
            if used_tokens + tokens > budget:
                break
            result.append(mem)
            used_tokens += tokens

        return result

    def _summarize_memories(self, memories: list[dict]) -> str:
        """当记忆过多时生成一句话摘要

        将所有记忆内容截取前 20 字拼接为摘要。

        Args:
            memories: 记忆字典列表

        Returns:
            摘要文本
        """
        snippets = []
        for mem in memories:
            content = mem.get("content", "").strip()
            if content:
                snippet = content[:20] + ("..." if len(content) > 20 else "")
                snippets.append(snippet)

        if not snippets:
            return "无可用记忆"

        return f"共 {len(memories)} 条记忆，涵盖：{'；'.join(snippets[:5])}" + (
            f" 等 {len(snippets)} 项" if len(snippets) > 5 else ""
        )

    def _format_memory_block(self, tier: str, memories: list[dict]) -> str:
        """格式化单个记忆分组为 Markdown 文本

        每条记忆添加来源标注（如 "[长期记忆]"），输出为 Markdown 列表。

        Args:
            tier: 记忆层级
            memories: 该层级的记忆列表

        Returns:
            格式化的 Markdown 文本
        """
        label = TIER_LABELS.get(tier, f"[{tier}]")
        title = f"## {label}"

        lines = [title]
        for mem in memories:
            content = mem.get("content", "").strip()
            if content:
                lines.append(f"- {label} {content}")

        return "\n".join(lines)
