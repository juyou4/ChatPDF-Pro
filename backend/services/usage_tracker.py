"""Lightweight LLM usage normalization and in-memory accounting."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Optional
import uuid

from services.token_budget import TokenBudgetManager


_budget = TokenBudgetManager()
_RECENT_USAGE: deque[dict[str, Any]] = deque(maxlen=200)


# Prices are per 1M tokens. Keep this table intentionally conservative; unknown
# models still report tokens but do not invent a cost.
_PRICE_TABLE: dict[tuple[str, str], dict[str, Any]] = {
    ("openai", "gpt-4o-mini"): {"input": 0.15, "output": 0.60, "currency": "USD"},
    ("openai", "gpt-4o"): {"input": 5.00, "output": 15.00, "currency": "USD"},
    ("openai", "gpt-4.1-mini"): {"input": 0.40, "output": 1.60, "currency": "USD"},
    ("openai", "gpt-4.1-nano"): {"input": 0.10, "output": 0.40, "currency": "USD"},
    ("openai", "gpt-4.1"): {"input": 2.00, "output": 8.00, "currency": "USD"},
    ("deepseek", "deepseek-chat"): {"input": 0.27, "output": 1.10, "currency": "USD"},
    ("deepseek", "deepseek-reasoner"): {"input": 0.55, "output": 2.19, "currency": "USD"},
}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_usage_tokens(raw_usage: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
    if not isinstance(raw_usage, dict):
        return None, None, None

    prompt = _coerce_int(
        raw_usage.get("prompt_tokens")
        or raw_usage.get("input_tokens")
        or raw_usage.get("promptTokenCount")
        or raw_usage.get("inputTokenCount")
    )
    completion = _coerce_int(
        raw_usage.get("completion_tokens")
        or raw_usage.get("output_tokens")
        or raw_usage.get("candidatesTokenCount")
        or raw_usage.get("outputTokenCount")
    )
    total = _coerce_int(
        raw_usage.get("total_tokens")
        or raw_usage.get("totalTokenCount")
        or raw_usage.get("total_token_count")
    )
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return prompt, completion, total


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: list[str] = []
    image_count = 0
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item or ""))
            continue
        item_type = item.get("type")
        if item_type == "text":
            parts.append(str(item.get("text") or ""))
        elif item_type == "image_url":
            image_count += 1
        else:
            parts.append(str(item))

    if image_count:
        parts.append("\n".join("[image]" for _ in range(image_count)))
    return "\n".join(parts)


def estimate_prompt_tokens(messages: list[dict] | None) -> int:
    if not messages:
        return 0
    text_parts: list[str] = []
    image_count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or ""
        content = msg.get("content")
        if isinstance(content, list):
            image_count += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image_url")
        text_parts.append(f"{role}: {_text_from_content(content)}")
    # A deliberately moderate fallback for low-detail image inputs. Provider
    # usage wins whenever available.
    return _budget.estimate_tokens("\n".join(text_parts)) + image_count * 1024


def estimate_completion_tokens_from_response(response: dict | None) -> int:
    if not isinstance(response, dict):
        return 0
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return 0
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = ""
    if isinstance(message, dict):
        content = str(message.get("content") or "")
    return _budget.estimate_tokens(content)


def estimate_completion_tokens_from_text(text: str = "") -> int:
    return _budget.estimate_tokens(text or "")


def _estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[dict[str, Any]]:
    price = _PRICE_TABLE.get(((provider or "").lower(), model or ""))
    if not price or prompt_tokens is None or completion_tokens is None:
        return None
    amount = (prompt_tokens / 1_000_000) * float(price["input"]) + (
        completion_tokens / 1_000_000
    ) * float(price["output"])
    return {
        "amount": round(amount, 8),
        "currency": price["currency"],
        "estimated": True,
        "pricing_unit": "per_1m_tokens",
    }


def build_usage_meta(
    *,
    provider: str,
    model: str,
    purpose: str = "llm",
    messages: list[dict] | None = None,
    raw_usage: Any = None,
    response: dict | None = None,
    completion_text: str = "",
) -> dict[str, Any]:
    prompt_tokens, completion_tokens, total_tokens = _extract_usage_tokens(raw_usage)
    estimated = False

    if prompt_tokens is None:
        prompt_tokens = estimate_prompt_tokens(messages)
        estimated = True
    if completion_tokens is None:
        completion_tokens = (
            estimate_completion_tokens_from_text(completion_text)
            if completion_text
            else estimate_completion_tokens_from_response(response)
        )
        estimated = True
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    cost = _estimate_cost(provider, model, prompt_tokens, completion_tokens)
    return {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "purpose": purpose or "llm",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated": estimated,
        "raw_usage": raw_usage if isinstance(raw_usage, dict) else None,
        "cost": cost,
    }


def attach_usage_meta(
    response: dict,
    *,
    provider: str,
    model: str,
    purpose: str = "llm",
    messages: list[dict] | None = None,
) -> dict:
    if not isinstance(response, dict):
        return response
    meta = build_usage_meta(
        provider=provider,
        model=model,
        purpose=purpose,
        messages=messages,
        raw_usage=response.get("usage"),
        response=response,
    )
    response["_usage_meta"] = meta
    response.setdefault("usage", {
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": meta["completion_tokens"],
        "total_tokens": meta["total_tokens"],
        "estimated": meta["estimated"],
    })
    record_usage(meta)
    return response


def record_usage(meta: dict[str, Any]) -> None:
    _RECENT_USAGE.append(meta)


def get_recent_usage(limit: int = 50) -> list[dict[str, Any]]:
    return list(_RECENT_USAGE)[-max(1, min(int(limit or 50), 200)) :]

