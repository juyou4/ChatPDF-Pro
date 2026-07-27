"""智能记忆上下文注入模块

按照记忆层级配额选择真正值得注入的记忆，避免普通 QA 噪声挤占预算。
"""
import logging
import re

logger = logging.getLogger(__name__)

KIND_ORDER = ["working", "session_summary", "profile", "doc_fact", "consolidated", "graph", "episodic"]

KIND_LABELS = {
    "working": "[工作记忆]",
    "session_summary": "[会话摘要]",
    "profile": "[全局画像]",
    "doc_fact": "[文档事实]",
    "consolidated": "[压缩事实]",
    "graph": "[论文图谱]",
    "episodic": "[对话摘要]",
}

DEFAULT_KIND_BUDGETS = {
    "working": 160,
    "session_summary": 180,
    "profile": 120,
    "doc_fact": 220,
    "consolidated": 160,
    "graph": 100,
    "episodic": 40,
}


# 每条记忆额外计入的固定开销：块标签、换行、来源后缀与估算误差。
# 宁可高估一点少注入，也不要算少了在模型端被静默截断。
_PER_ITEM_TOKEN_OVERHEAD = 20

# CJK 与假名/谚文：这些字符按 1 token/字估算。
_CJK_RE = re.compile(
    r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]"
)


def estimate_text_tokens(text: str) -> int:
    """估算一段纯文本的 token 数（不含 per-item 开销）。

    CJK 按 1 token/字（对 cl100k 一类分词器是保守值），其余按 4 字符/token。
    统一的 len//2 会把中文低估近一半、把英文高估一倍，两边都不准。

    文档/联网这类大块证据用这个；单条记忆用 ``_estimate_tokens``，
    后者会额外计入块标签与分隔符的固定开销。
    """
    if not text:
        return 0
    cjk_count = len(_CJK_RE.findall(text))
    other_count = max(0, len(text) - cjk_count)
    return max(cjk_count + (other_count + 3) // 4, 1)


def _estimate_tokens(text: str) -> int:
    """估算一条记忆占用的 token 数（含固定的 per-item 开销）。"""
    if not text:
        return 0
    return estimate_text_tokens(text) + _PER_ITEM_TOKEN_OVERHEAD


class ContextInjector:
    """分层记忆上下文注入器。

    注入顺序固定为：
    working -> profile -> doc_fact -> consolidated -> graph -> episodic
    """

    def __init__(self, token_budget: int = 800, kind_budgets: dict[str, int] | None = None):
        self.token_budget = token_budget
        self.kind_budgets = {**DEFAULT_KIND_BUDGETS, **(kind_budgets or {})}

    def inject(
        self,
        system_prompt: str,
        memories: list[dict],
        token_budget: int = None,
        allowed_kinds: set[str] | None = None,
    ) -> str:
        if not memories:
            return system_prompt

        selected = self.prepare_memories(
            memories, token_budget=token_budget, allowed_kinds=allowed_kinds
        )
        if not selected:
            return system_prompt

        grouped = self._group_memories(selected)
        blocks = []
        for kind in KIND_ORDER:
            if kind in grouped and grouped[kind]:
                blocks.append(self._format_memory_block(kind, grouped[kind]))

        if not blocks:
            return system_prompt

        memory_text = "\n\n".join(blocks)
        return f"{system_prompt}\n\n---\n{memory_text}"

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """对外暴露的单条估算，供调用方核对实际注入量。"""
        return _estimate_tokens(text)

    def estimate_selection_tokens(self, memories: list[dict]) -> int:
        """统计一批已选记忆的估算 token 总量。"""
        return sum(_estimate_tokens(item.get("content", "")) for item in memories or [])

    def prepare_memories(
        self,
        memories: list[dict],
        token_budget: int = None,
        allowed_kinds: set[str] | None = None,
    ) -> list[dict]:
        """按层级预算和整体预算筛选真正要注入的记忆。

        Args:
            allowed_kinds: 允许注入的记忆层白名单。共享/导出场景下用它把
                个人画像与跨文档对话摘要挡在上下文之外——这是隐私边界，
                不是预算问题，所以在预算分配之前就过滤掉。
        """
        if not memories:
            return []

        total_budget = token_budget if token_budget is not None else self.token_budget
        if total_budget <= 0:
            return []

        if allowed_kinds is not None:
            memories = [
                mem for mem in memories
                if (mem.get("memory_kind") or self._infer_memory_kind(mem)) in allowed_kinds
            ]
            if not memories:
                return []

        grouped = self._group_memories(memories)
        selected: list[dict] = []
        used_tokens = 0

        for kind in KIND_ORDER:
            kind_memories = grouped.get(kind, [])
            if not kind_memories:
                continue

            remaining_total = total_budget - used_tokens
            if remaining_total <= 0:
                break

            kind_budget = min(self.kind_budgets.get(kind, remaining_total), remaining_total)
            picked = self._pick_by_budget(kind_memories, kind_budget)
            picked_tokens = sum(_estimate_tokens(item.get("content", "")) for item in picked)
            if picked_tokens <= 0:
                continue
            if picked_tokens > remaining_total:
                picked = self._pick_by_budget(picked, remaining_total)
                picked_tokens = sum(_estimate_tokens(item.get("content", "")) for item in picked)
            if not picked:
                continue

            selected.extend(picked)
            used_tokens += picked_tokens

        return selected

    def _group_memories(self, memories: list[dict]) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = {}
        for mem in memories:
            kind = mem.get("memory_kind") or self._infer_memory_kind(mem)
            groups.setdefault(kind, []).append(mem)
        for kind, items in groups.items():
            items.sort(key=self._score_key, reverse=True)
        return groups

    @staticmethod
    def _infer_memory_kind(mem: dict) -> str:
        memory_tier = mem.get("memory_tier")
        source_type = mem.get("source_type")
        if memory_tier == "working":
            return "working"
        if source_type == "compressed":
            return "consolidated"
        if source_type in {"manual", "liked", "keyword"} and mem.get("memory_scope") == "profile":
            return "profile"
        if source_type == "llm_distilled":
            return "doc_fact"
        return "episodic"

    @staticmethod
    def _score_key(mem: dict) -> tuple:
        return (
            mem.get("score", mem.get("rrf_score", 0.0)),
            mem.get("importance", 0.0),
            mem.get("hit_count", 0),
        )

    def _pick_by_budget(self, memories: list[dict], budget: int) -> list[dict]:
        if budget <= 0:
            return []
        picked: list[dict] = []
        used_tokens = 0
        for mem in memories:
            content = mem.get("content", "")
            tokens = _estimate_tokens(content)
            if used_tokens + tokens > budget:
                continue
            picked.append(mem)
            used_tokens += tokens
        return picked

    def _format_memory_block(self, kind: str, memories: list[dict]) -> str:
        label = KIND_LABELS.get(kind, f"[{kind}]")
        lines = [f"## {label}"]
        for mem in memories:
            content = mem.get("content", "").strip()
            if not content:
                continue
            title = mem.get("title", "").strip()
            source_ref = mem.get("source_ref") or {}
            prefix = f"{title}: " if title and title not in content[: len(title) + 2] else ""
            suffix = ""
            if source_ref.get("question"):
                suffix = f"（来源问题：{source_ref['question'][:40]}）"
            lines.append(f"- {label} {prefix}{content}{suffix}")
        return "\n".join(lines)
