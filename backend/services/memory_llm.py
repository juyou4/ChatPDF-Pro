"""记忆子系统共用的 LLM 调用与输出解析工具

记忆链路里的 LLM 调用（事实提炼、写入裁决、压缩）都跑在后台线程里，
需要同一套"同步壳 + 健壮解析 + 失败即降级"的处理，抽出来避免三份实现漂移。
"""
import asyncio
import json
import logging
import re
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_LLM_TIMEOUT = 30.0

# 一轮对话里记忆链路可能发起的后台 LLM 调用，按优先级从高到低：
#   distill(+retry) → arbitrate → compress(+retry) → session_summary → graph
# 最坏情况 7 次。没有总控时这笔开销既不可见也不可控，
# 所以统一走 MemoryLLMBudget：预算耗尽时**跳过低优先级的那些**，
# 而不是让先跑的把额度吃光。执行顺序已按优先级排列，因此简单计数即可。
MEMORY_LLM_PRIORITIES = (
    "distill",
    "arbitrate",
    "compress",
    "session_summary",
    "graph",
)


class MemoryLLMBudget:
    """单轮记忆写入的 LLM 调用预算。

    线程安全：同一轮的后台写入跑在单个守护线程里，但压缩/图谱可能被
    `_safe_execute` 包裹后在同一线程串行调用，这里仍加锁以防未来并行化。
    """

    def __init__(self, max_calls: int):
        self.max_calls = max(0, int(max_calls))
        self._spent = 0
        self._breakdown: dict[str, int] = {}
        self._skipped: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self._spent)

    def try_consume(self, label: str) -> bool:
        """申请一次调用额度；返回 False 表示预算已尽，调用方应跳过。"""
        with self._lock:
            if self._spent >= self.max_calls:
                self._skipped[label] = self._skipped.get(label, 0) + 1
                return False
            self._spent += 1
            self._breakdown[label] = self._breakdown.get(label, 0) + 1
            return True

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_calls": self.max_calls,
                "spent": self._spent,
                "breakdown": dict(self._breakdown),
                "skipped": dict(self._skipped),
            }

    def log_summary(self, doc_id: str | None = None) -> None:
        report = self.report()
        if report["skipped"]:
            logger.info(
                "[MemoryLLMBudget] doc_id=%s 用掉 %d/%d 次调用，因预算跳过: %s",
                doc_id, report["spent"], report["max_calls"], report["skipped"],
            )
        elif report["spent"]:
            logger.debug(
                "[MemoryLLMBudget] doc_id=%s 用掉 %d/%d 次调用: %s",
                doc_id, report["spent"], report["max_calls"], report["breakdown"],
            )


def consume_budget(budget: Optional[MemoryLLMBudget], label: str) -> bool:
    """budget 为 None 表示不设限（旧调用方与测试保持原行为）。"""
    if budget is None:
        return True
    return budget.try_consume(label)


def call_llm_sync(
    messages: list[dict],
    *,
    api_key: str,
    model: str,
    provider: str,
    max_tokens: int = 800,
    temperature: float = 0.1,
    timeout: float = DEFAULT_LLM_TIMEOUT,
) -> Optional[str]:
    """在同步上下文里调用异步 LLM 接口，返回文本内容。

    记忆写入跑在守护线程里，通常没有运行中的事件循环。这里必须把
    ``timeout`` 包到真正的协程上：只给 ``future.result`` 设超时会在
    ``asyncio.run`` 分支失效，慢连接会长期占住后台写入槽位。

    此函数是同步接口，不能从正在运行的事件循环线程直接调用。此前用
    ``run_coroutine_threadsafe(..., 当前循环)`` 后立刻 ``result()``，会让
    当前线程等待自己，直到超时。异步路由应使用 ``asyncio.to_thread``
    调用本函数（图谱重建入口已按此方式处理）。
    """
    from services.chat_service import call_ai_api

    def _make_coro():
        return call_ai_api(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        timeout_seconds = max(0.01, float(timeout))
    except (TypeError, ValueError):
        timeout_seconds = DEFAULT_LLM_TIMEOUT

    async def _call_with_timeout():
        return await asyncio.wait_for(_make_coro(), timeout=timeout_seconds)

    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop and loop.is_running():
        raise RuntimeError(
            "call_llm_sync 不能在运行中的事件循环线程直接调用；请使用 asyncio.to_thread"
        )

    response = asyncio.run(_call_with_timeout())

    return extract_response_text(response)


def extract_response_text(response: Any) -> Optional[str]:
    """从不同 provider 的返回结构里取出正文。"""
    if isinstance(response, str):
        return response or None
    if not isinstance(response, dict):
        return None
    if response.get("error"):
        logger.warning(f"[MemoryLLM] 调用返回错误: {response['error']}")
        return None
    content = response.get("content", "")
    if not content:
        content = (response.get("message") or {}).get("content", "")
    if not content:
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            content = (choices[0].get("message") or {}).get("content", "")
    return content or None


def strip_reasoning_and_fences(text: str) -> str:
    """剥掉推理模型的 <think> 段与 markdown 代码围栏。"""
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> Optional[dict]:
    """尽力从模型输出里抠出一个 JSON 对象。"""
    cleaned = strip_reasoning_and_fences(text)
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def parse_bullet_list(
    text: str,
    *,
    limit: Optional[int] = None,
    require_marker: bool = False,
) -> list[str]:
    """解析 '- xxx' 形式的列表输出，容忍编号与全角符号。

    Args:
        limit: 最多返回几条
        require_marker: 为 True 时只接受带列表标记的行。
            调用方若要靠"解析失败"来触发重试，**必须**打开它——
            否则模型返回的解释性散文会被当成一条合法结果，重试永远不会发生。
    """
    cleaned = strip_reasoning_and_fences(text)
    if not cleaned:
        return []
    items: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # 兼容 "- x" / "* x" / "1. x" / "1、x" / "・x"
        match = re.match(r"^(?:[-*・•]|\d+[.、)])\s*(.+)$", line)
        if match is None and require_marker:
            continue
        item = (match.group(1) if match else line).strip()
        if item:
            items.append(item)
    if limit is not None:
        return items[:limit]
    return items
