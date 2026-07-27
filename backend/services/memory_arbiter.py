"""记忆写入裁决器

把"提炼出事实就直接追加"升级为：新事实先与同作用域的既有记忆比对，
由 LLM 逐条裁决 ADD / UPDATE / DELETE / NONE。

解决的问题：
- 同一事实反复出现堆积重复条目（原先只能等 20 条阈值的事后压缩兜底）
- 用户改口后旧记忆不会更新，会一直被召回（"我其实更关注方法而非实验"）

两个关键工程细节：
1. 真实 UUID 不给 LLM 看，映射成 "0"/"1"/... 的临时整数 ID 再还原，
   杜绝模型编造不存在的 ID（借鉴 mem0）。
2. DELETE 一律实现为**可逆失效**（invalid_at），绝不物理删除。
   学术场景里数据不能悄悄消失，而且 LLM 判"矛盾"本身可能出错。

任何一步失败都降级为"全部 ADD"，即改动前的行为。
"""
import json
import logging
from typing import Any, Optional

from services.memory_llm import (
    call_llm_sync,
    extract_json_object,
    strip_reasoning_and_fences,
)

logger = logging.getLogger(__name__)

ACTION_ADD = "ADD"
ACTION_UPDATE = "UPDATE"
ACTION_DELETE = "DELETE"
ACTION_NONE = "NONE"
_VALID_ACTIONS = {ACTION_ADD, ACTION_UPDATE, ACTION_DELETE, ACTION_NONE}

# 送进裁决 prompt 的既有记忆上限，避免 prompt 无界增长
DEFAULT_MAX_CANDIDATES = 20
# 每条新事实检索多少条相似旧记忆
DEFAULT_NEIGHBOURS_PER_FACT = 5

_SYSTEM_PROMPT = """你是论文阅读助手的记忆管理器。给你一份"已有记忆"和一份"新事实"，
请逐条决定每个新事实该如何并入记忆库。

四种操作：
- ADD：新事实是全新信息，已有记忆里没有。
- UPDATE：新事实与某条已有记忆讲的是同一件事，但信息量更大或更准确。
  需要给出该记忆的 id，以及合并后的完整表述。
- DELETE：新事实与某条已有记忆**直接矛盾**，旧的那条已经不成立。
  需要给出该记忆的 id。
- NONE：新事实与已有记忆语义等价，没有新增信息。

判断规则：
- 信息量更大才 UPDATE。"关注方法" 与 "主要关注方法部分，希望略过实验" → UPDATE。
- 语义等价一律 NONE。"准确率 82.4" 与 "Acc 为 82.4" → NONE。
- 只有真正互斥才 DELETE。"用户关注方法" 与 "用户关注实验" **不是**矛盾，
  两者可以同时成立，应该 ADD；只有 "表2最优方法是 A" 与 "表2最优方法是 B"
  这种同一槽位取值冲突才算矛盾。
- 特别注意：同名不同对象不算矛盾。不同论文可以都有叫 baseline 的方法，
  不同表格可以都有 Acc 列。拿不准时选 ADD，不要 DELETE。

输出严格的 JSON，不要任何解释或代码块标记：
{"decisions": [{"action": "ADD|UPDATE|DELETE|NONE", "id": <已有记忆的 id，ADD 时为 null>, "text": "<ADD/UPDATE 时的记忆内容，其余为空字符串>"}]}

decisions 的条数必须与新事实条数一致，顺序一一对应。"""


# 复用共享实现，但保留模块级别名：现有测试通过 patch
# ``services.memory_arbiter._call_llm_sync`` 注入假响应。
_strip_reasoning_and_fences = strip_reasoning_and_fences
_extract_json_object = extract_json_object


def _call_llm_sync(messages: list[dict], *, api_key: str, model: str, provider: str,
                   max_tokens: int = 800, timeout: float = 30.0) -> Optional[str]:
    return call_llm_sync(
        messages,
        api_key=api_key,
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        timeout=timeout,
    )


class MemoryDecision:
    """单条新事实的裁决结果。"""

    __slots__ = ("action", "text", "target_id", "old_content")

    def __init__(self, action: str, text: str = "", target_id: str = "", old_content: str = ""):
        self.action = action
        self.text = text
        self.target_id = target_id
        self.old_content = old_content

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"MemoryDecision({self.action}, target={self.target_id!r}, text={self.text[:30]!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryDecision):
            return NotImplemented
        return (
            self.action == other.action
            and self.text == other.text
            and self.target_id == other.target_id
        )


class MemoryArbiter:
    """新事实与既有记忆的合并裁决。"""

    def __init__(
        self,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        neighbours_per_fact: int = DEFAULT_NEIGHBOURS_PER_FACT,
    ):
        self.max_candidates = max_candidates
        self.neighbours_per_fact = neighbours_per_fact

    @staticmethod
    def _fallback_decisions(facts: list[str]) -> list[MemoryDecision]:
        """降级：全部当新事实追加，即启用裁决之前的行为。"""
        return [MemoryDecision(ACTION_ADD, text=fact) for fact in facts]

    def collect_candidates(
        self,
        facts: list[str],
        *,
        index,
        entry_map: dict[str, Any],
        doc_id: Optional[str],
    ) -> list[Any]:
        """为一批新事实收集同作用域的相似既有记忆。

        只保留当前文档记忆与全局画像记忆——别的文档的记忆不该参与本文档的裁决。
        """
        picked: dict[str, Any] = {}
        for fact in facts:
            if len(picked) >= self.max_candidates:
                break
            try:
                neighbours = index.search(fact, top_k=self.neighbours_per_fact) or []
            except Exception as exc:
                logger.debug(f"[MemoryArbiter] 候选检索失败: {exc}")
                continue
            for hit in neighbours:
                try:
                    entry_id = hit.get("entry_id", "")
                except AttributeError:
                    continue
                entry = entry_map.get(entry_id)
                if entry is None or entry_id in picked:
                    continue
                if not getattr(entry, "is_retrievable", True):
                    continue
                if entry.doc_id not in (None, doc_id):
                    continue
                picked[entry_id] = entry
                if len(picked) >= self.max_candidates:
                    break
        return list(picked.values())

    def arbitrate(
        self,
        facts: list[str],
        candidates: list[Any],
        *,
        api_key: str,
        model: str,
        provider: str,
    ) -> list[MemoryDecision]:
        """裁决一批新事实。失败时降级为全部 ADD。"""
        normalized_facts = [str(fact or "").strip() for fact in facts or []]
        normalized_facts = [fact for fact in normalized_facts if fact]
        if not normalized_facts:
            return []
        if not candidates or not (api_key and model and provider):
            return self._fallback_decisions(normalized_facts)

        # 真实 UUID 不进 prompt，换成临时整数 ID 防模型编造。
        temp_id_map: dict[str, Any] = {}
        existing_lines = []
        for position, entry in enumerate(candidates):
            temp_id = str(position)
            temp_id_map[temp_id] = entry
            existing_lines.append({"id": temp_id, "memory": entry.content})

        payload = {
            "已有记忆": existing_lines,
            "新事实": normalized_facts,
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        try:
            raw = _call_llm_sync(
                messages, api_key=api_key, model=model, provider=provider
            )
        except Exception as exc:
            logger.warning(f"[MemoryArbiter] 调用失败，降级为全部新增: {exc}")
            return self._fallback_decisions(normalized_facts)

        parsed = _extract_json_object(raw or "")
        if not parsed:
            logger.warning("[MemoryArbiter] 无法解析裁决结果，降级为全部新增")
            return self._fallback_decisions(normalized_facts)

        raw_decisions = parsed.get("decisions")
        if not isinstance(raw_decisions, list) or not raw_decisions:
            logger.warning("[MemoryArbiter] 裁决结果缺少 decisions，降级为全部新增")
            return self._fallback_decisions(normalized_facts)

        decisions: list[MemoryDecision] = []
        for position, fact in enumerate(normalized_facts):
            item = raw_decisions[position] if position < len(raw_decisions) else None
            decisions.append(self._normalize_decision(item, fact, temp_id_map))
        return decisions

    def _normalize_decision(
        self,
        item: Any,
        fact: str,
        temp_id_map: dict[str, Any],
    ) -> MemoryDecision:
        """把单条模型输出规整成可执行的裁决；任何异常形态都退回 ADD。"""
        if not isinstance(item, dict):
            return MemoryDecision(ACTION_ADD, text=fact)

        action = str(item.get("action", "")).strip().upper()
        if action not in _VALID_ACTIONS:
            return MemoryDecision(ACTION_ADD, text=fact)

        if action == ACTION_ADD:
            text = str(item.get("text") or "").strip() or fact
            return MemoryDecision(ACTION_ADD, text=text)

        if action == ACTION_NONE:
            return MemoryDecision(ACTION_NONE)

        # UPDATE / DELETE 必须能还原到真实条目，否则退回 ADD 而不是丢掉这条事实。
        raw_id = item.get("id")
        if raw_id is None:
            return MemoryDecision(ACTION_ADD, text=fact)
        entry = temp_id_map.get(str(raw_id).strip())
        if entry is None:
            logger.debug(f"[MemoryArbiter] 裁决引用了不存在的 id={raw_id!r}，退回新增")
            return MemoryDecision(ACTION_ADD, text=fact)

        if action == ACTION_UPDATE:
            text = str(item.get("text") or "").strip()
            if not text:
                return MemoryDecision(ACTION_ADD, text=fact)
            return MemoryDecision(
                ACTION_UPDATE,
                text=text,
                target_id=entry.id,
                old_content=entry.content,
            )

        # DELETE：旧条目失效，同时把新事实作为新条目补进来
        return MemoryDecision(
            ACTION_DELETE,
            text=fact,
            target_id=entry.id,
            old_content=entry.content,
        )
