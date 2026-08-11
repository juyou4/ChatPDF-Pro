"""统一识别模型生成是否因输出上限而中断。

各 Provider 对同一状态使用不同字段和值：OpenAI 使用
``finish_reason=length``，Anthropic 使用 ``stop_reason=max_tokens``，
Gemini 使用 ``finishReason=MAX_TOKENS``，Ollama 使用
``done_reason=length``。下游只能依赖本模块的归一化结果，不能再把
"HTTP/SSE 已结束"等同于"内容已完整生成"。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping


class CompletionStatus(str, Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class IncompleteCompletionError(RuntimeError):
    """Raised when a caller is about to publish an incomplete model result."""

    def __init__(self, outcome: "CompletionOutcome", operation: str = "model output"):
        self.outcome = outcome
        self.operation = str(operation or "model output")
        reason = outcome.finish_reason or outcome.status.value
        super().__init__(f"{self.operation} is not publishable (finish_reason={reason})")


_TRUNCATION_REASONS = frozenset(
    {
        "length",
        "max_length",
        "max_token",
        "max_tokens",
        "max_output_token",
        "max_output_tokens",
        "output_limit",
        "output_token_limit",
        "token_limit",
        "context_length",
        "context_window_exceeded",
    }
)

_BLOCKED_REASONS = frozenset(
    {
        "blocked",
        "content_filter",
        "prohibited_content",
        "recitation",
        "refusal",
        "safety",
        "spi",
        "spii",
    }
)


def normalize_finish_reason(value: Any) -> str:
    """Return a stable snake-case finish reason without inventing a value."""

    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def is_truncation_finish_reason(value: Any) -> bool:
    reason = normalize_finish_reason(value)
    if reason in _TRUNCATION_REASONS:
        return True
    # 兼容少数 OpenAI-compatible 网关追加的命名空间或前后缀。
    return (
        "max_token" in reason
        or "token_limit" in reason
        or reason.endswith("_length_limit")
    )


def _first_text(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_finish_reason(response: Any) -> str:
    """Extract the raw provider finish reason from a normalized response."""

    if not isinstance(response, Mapping):
        return ""

    direct = _first_text(
        response,
        ("finish_reason", "stop_reason", "finishReason", "done_reason"),
    )
    if direct:
        return direct

    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        choice_reason = _first_text(
            choices[0],
            ("finish_reason", "stop_reason", "finishReason", "done_reason"),
        )
        if choice_reason:
            return choice_reason

    candidates = response.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
        candidate_reason = _first_text(
            candidates[0],
            ("finishReason", "finish_reason", "stop_reason"),
        )
        if candidate_reason:
            return candidate_reason

    return ""


@dataclass(frozen=True)
class CompletionOutcome:
    status: CompletionStatus
    finish_reason: str = ""
    normalized_reason: str = ""

    @property
    def truncated(self) -> bool:
        return self.status is CompletionStatus.TRUNCATED

    @property
    def publishable(self) -> bool:
        return self.status is CompletionStatus.COMPLETED

    def public(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "finish_reason": self.finish_reason,
            "normalized_reason": self.normalized_reason,
            "truncated": self.truncated,
        }


def resolve_completion_outcome(
    response: Any = None,
    *,
    finish_reason: Any = None,
    transport_complete: bool = True,
) -> CompletionOutcome:
    raw_reason = str(
        finish_reason if finish_reason is not None else extract_finish_reason(response)
    ).strip()
    normalized = normalize_finish_reason(raw_reason)

    if is_truncation_finish_reason(normalized):
        status = CompletionStatus.TRUNCATED
    elif normalized in _BLOCKED_REASONS:
        status = CompletionStatus.BLOCKED
    elif normalized or transport_complete:
        # 很多兼容网关不返回 finish_reason；在传输正常闭合时继续兼容。
        status = CompletionStatus.COMPLETED
    else:
        status = CompletionStatus.UNKNOWN

    return CompletionOutcome(
        status=status,
        finish_reason=raw_reason,
        normalized_reason=normalized,
    )


def require_publishable_completion(
    response: Any,
    *,
    operation: str = "model output",
) -> CompletionOutcome:
    """Reject truncated/blocked responses before cache or index publication."""

    outcome = resolve_completion_outcome(
        response,
        transport_complete=not bool(
            isinstance(response, Mapping) and response.get("error")
        ),
    )
    if not outcome.publishable:
        raise IncompleteCompletionError(outcome, operation)
    return outcome
