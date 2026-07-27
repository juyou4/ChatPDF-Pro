"""Conservative JSON-object parsing for model-generated structured output."""

from __future__ import annotations

import json
import re
from typing import Any

from models.model_detector import infer_model_tags
from providers.provider_ids import ANTHROPIC, GEMINI, OLLAMA


class StructuredJSONError(ValueError):
    """Raised when model output cannot be recovered as a JSON object."""


_JSON_OBJECT_RESPONSE_FORMAT: dict[str, Any] = {"type": "json_object"}

# 用原生请求体、不接受 OpenAI 字段的通道。与 ``providers/factory.py`` 的路由一致：
# 未知 provider 会落到 OpenAI 兼容分支，所以这里也只排除这三家。
_NATIVE_BODY_PROVIDERS = ANTHROPIC | GEMINI | OLLAMA

# 只列出**确认**接受 ``thinking: {"type": "disabled"}`` 这个开关的模型族。
# 其余推理模型各家关闭思考的字段名不同（且有些根本关不掉），与其臆造一个参数让
# 请求直接 400，不如只锁定 JSON 输出格式，把预算问题交给调用方的 max_tokens。
_DISABLE_THINKING_MODEL_RE = re.compile(
    r"deepseek"          # DeepSeek V3.1+/V4 hybrid
    r"|glm-4\.[5-9]"     # 智谱 GLM-4.5 及以上
    r"|glm-5",
    re.IGNORECASE,
)


def structured_json_request_params(provider: str, model: str) -> dict[str, Any] | None:
    """Return the controls a reasoning model needs before it will emit JSON.

    A hybrid reasoning model can spend its entire completion budget on hidden
    reasoning and come back with ``content_chars == 0`` and
    ``finish_reason == "length"``; the JSON then fails to parse at line 1
    column 1 and every section in that batch is lost.

    This used to key off ``provider == "deepseek"``, which only recognised the
    vendor's own endpoint. The same model reached through an OpenAI-compatible
    gateway (silicon / aliyun / zhipu / ...) arrives with a different provider
    string and reproduced that failure verbatim. Detect the model's *capability*
    instead — ``provider`` is only a routing detail.
    """
    model_id = str(model or "").strip()
    if not model_id:
        return None
    if "reasoning" not in infer_model_tags(model_id):
        # 非推理模型没有隐藏思考预算问题，保持原样不加任何参数。
        return None
    if str(provider or "").strip().lower() in _NATIVE_BODY_PROVIDERS:
        # 能力看模型，字段形状看通道。这几家用各自的原生请求体，而 ``custom_params``
        # 是被直接 ``body.update()`` 进去的，塞入 OpenAI 的 ``response_format``
        # 只会让请求带上一个非法字段。
        return None
    params: dict[str, Any] = {"response_format": dict(_JSON_OBJECT_RESPONSE_FORMAT)}
    if _DISABLE_THINKING_MODEL_RE.search(model_id):
        params["thinking"] = {"type": "disabled"}
    return params


def _extract_json_object_text(content: Any) -> str:
    text = str(content or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def parse_json_object(
    content: Any,
    *,
    allow_partial: bool = True,
) -> dict[str, Any]:
    """Parse one model response with bounded, non-semantic repairs.

    A recovered JSON prefix is unsafe for document outlines: an apparently
    valid object can silently omit the final chapters. Callers that require
    complete coverage therefore pass ``allow_partial=False``.
    """
    text = _extract_json_object_text(content)
    candidates = [text]
    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", text)
    if without_trailing_commas != text:
        candidates.append(without_trailing_commas)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate, strict=False)
            if isinstance(parsed, dict):
                return parsed
            last_error = TypeError("structured output root is not an object")
        except (json.JSONDecodeError, TypeError) as exc:
            last_error = exc

    if allow_partial:
        try:
            from langchain_core.utils.json import parse_partial_json

            parsed = parse_partial_json(candidates[-1])
            if isinstance(parsed, dict):
                return parsed
            last_error = TypeError("structured output root is not an object")
        except Exception as exc:
            last_error = exc

    raise StructuredJSONError("模型返回的结构化结果格式不完整") from last_error
