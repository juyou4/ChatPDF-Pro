"""Conservative JSON-object parsing for model-generated structured output."""

from __future__ import annotations

import json
import re
from typing import Any


class StructuredJSONError(ValueError):
    """Raised when model output cannot be recovered as a JSON object."""


def structured_json_request_params(provider: str, model: str) -> dict[str, Any] | None:
    """Return provider-specific controls for schema-constrained generation.

    Hybrid DeepSeek models can spend their entire completion budget on hidden
    reasoning, leaving no JSON body to parse.  This policy belongs beside the
    structured-output contract so outlines and other JSON consumers cannot
    drift into different provider behaviour.
    """
    provider_lower = str(provider or "").strip().lower()
    model_lower = str(model or "").strip().lower()
    if provider_lower == "deepseek" and "deepseek" in model_lower:
        return {
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
    return None


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
