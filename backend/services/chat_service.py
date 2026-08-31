from typing import Dict, List, Optional
import asyncio
import json as _json
import logging
from urllib.parse import urlsplit
import httpx

logger = logging.getLogger(__name__)

from providers.factory import ProviderFactory
from providers.provider_ids import OPENAI_LIKE, ANTHROPIC, GEMINI, OPENAI_NATIVE, MINIMAX, MOONSHOT, DOUBAO
from models.provider_registry import PROVIDER_CONFIG
from models.dynamic_store import load_dynamic_providers
from models.api_key_selector import select_api_key
from utils.middleware import (
    BaseMiddleware,
    apply_middlewares_before,
    apply_middlewares_after,
    RetryMiddleware,
    FallbackMiddleware,
)
from services.usage_tracker import attach_usage_meta
from services.completion_outcome import resolve_completion_outcome
from services.provider_auth import build_api_key_headers, resolve_api_key_auth
from services.reasoning_effort_service import (
    apply_reasoning_to_payload,
    ensure_reasoning_output_budget,
    merge_request_body,
    prepare_reasoning_history_messages,
    sanitize_reasoning_sampling_parameters,
)
from services.think_tag_stream import (
    StreamingThinkSplitter,
    apply_stream_think_split,
    split_think_tags,
)


def _sanitize_api_key(api_key: Optional[str]) -> str:
    """清理 API Key，兼容空值与多 Key 轮换池。"""
    return select_api_key(api_key) or (api_key.strip() if api_key else "")


_REASONING_MODEL_PATTERNS = (
    "seed", "-r1", "-r2", "o1", "o3", "o4", "thinking", "reasoning",
    "-think", "deepseek-v4",
)


def _is_reasoning_model(model: str) -> bool:
    """判断模型是否为推理/思考模型（不支持 logprobs、logit_bias 等参数）。"""
    m = (model or "").lower()
    return any(p in m for p in _REASONING_MODEL_PATTERNS)


def _dynamic_provider_config(provider: str) -> dict:
    try:
        config = load_dynamic_providers().get(str(provider or "").strip())
    except Exception:
        config = None
    return config if isinstance(config, dict) else {}


def _is_openai_compatible_provider(provider: str, endpoint: str = "") -> bool:
    """识别动态 OpenAI-compatible Provider，不能只依赖固定 ID 白名单。"""
    pid = str(provider or "").strip().lower()
    if pid in OPENAI_LIKE:
        return True
    if pid in ANTHROPIC or pid in GEMINI or pid in {"ollama"}:
        return False
    config = _dynamic_provider_config(provider)
    protocol = str(config.get("type") or "").strip().lower()
    # 未写入动态存储的历史自定义 Provider 也沿用 ProviderFactory 的
    # OpenAI-compatible fallback，避免升级后突然退回“假流式”。
    return bool(endpoint) and (not config or protocol in {"openai", "custom"})


def _is_anthropic_provider(provider: str) -> bool:
    """识别内置和动态 Anthropic Messages Provider。"""

    pid = str(provider or "").strip().lower()
    if pid in ANTHROPIC:
        return True
    config = _dynamic_provider_config(provider)
    return str(config.get("type") or "").strip().lower() == "anthropic"


def _dynamic_provider_supports_streaming(provider: str) -> bool:
    config = _dynamic_provider_config(provider)
    if not config:
        return True
    return bool(config.get("supports_streaming", True))


def _dynamic_provider_supports_reasoning(provider: str) -> bool:
    config = _dynamic_provider_config(provider)
    return bool(config.get("supports_reasoning", False))


def _dynamic_provider_auth(provider: str) -> tuple[str | None, str | None]:
    """Return non-secret auth metadata for a persisted dynamic provider."""

    config = _dynamic_provider_config(provider)
    if not config:
        return None, None
    if "api_key_header" not in config and "api_key_prefix" not in config:
        return None, None
    header, prefix = resolve_api_key_auth(
        provider_type=config.get("type"),
        api_key_header=config.get("api_key_header"),
        api_key_prefix=config.get("api_key_prefix"),
    )
    return header, prefix


def _create_provider_client(provider: str, endpoint: str):
    """Create a client while keeping legacy providers on their old path."""

    config = _dynamic_provider_config(provider)
    provider_type = str(config.get("type") or "").strip().lower() or None
    auth_header, auth_prefix = _dynamic_provider_auth(provider)
    if provider_type == "anthropic":
        return ProviderFactory.create(
            provider,
            endpoint,
            provider_type=provider_type,
            api_key_header=auth_header,
            api_key_prefix=auth_prefix,
        )
    if auth_header is None and auth_prefix is None:
        return ProviderFactory.create(provider, endpoint)
    return ProviderFactory.create(
        provider,
        endpoint,
        provider_type=provider_type,
        api_key_header=auth_header,
        api_key_prefix=auth_prefix,
    )


def _join_provider_endpoint(base: str, path: str) -> str:
    """Join a dynamic provider base URL and relative chat path safely."""
    base_clean = str(base or "").rstrip("/")
    path_clean = "/" + str(path or "").lstrip("/")
    if not base_clean:
        return path_clean
    if not path:
        return base_clean
    try:
        parsed = urlsplit(base_clean)
    except ValueError:
        parsed = None
    if parsed and parsed.scheme and parsed.netloc:
        base_path = (parsed.path or "").rstrip("/")
        if base_path and (base_path == path_clean or base_path.endswith(path_clean)):
            return base_clean
        if base_path and path_clean.startswith(base_path + "/"):
            return parsed._replace(path=path_clean).geturl().rstrip("/")
    return f"{base_clean}{path_clean}"


def _provider_default_endpoint(provider: str) -> str:
    """Resolve static and persisted dynamic Provider chat endpoints."""
    config = _dynamic_provider_config(provider)
    if config:
        endpoint = str(config.get("endpoint") or "").strip()
        chat_path = str(config.get("chat_endpoint") or "").strip()
        if endpoint and chat_path:
            return _join_provider_endpoint(endpoint, chat_path)
        if endpoint:
            return endpoint
    return PROVIDER_CONFIG.get(provider, {}).get("endpoint", "")


def _extract_api_error_message(body: str, status_code: int) -> str:
    """从 API 错误响应体中提取用户友好的中文错误信息。
    兼容 OpenAI 兼容格式：{"error": {"code": "...", "message": "..."}}。
    """
    try:
        parsed = _json.loads(body) if body else {}
        error_obj = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(error_obj, dict):
            msg = error_obj.get("message") or ""
            code = error_obj.get("code") or ""
            label = f"（{code}）" if code else ""
            if status_code == 429:
                suffix = f"：{msg}" if msg else "，请稍后再试"
                return f"请求过于频繁{label}{suffix}"
            if status_code in (401, 403):
                return f"认证失败（HTTP {status_code}）：{msg or 'API Key 无效或格式错误'}"
            return f"API 错误（HTTP {status_code}{label}）：{msg}" if (msg or code) else f"API 返回错误（HTTP {status_code}）"
        elif isinstance(error_obj, str):
            return f"API 错误（HTTP {status_code}）：{error_obj}"
    except Exception:
        pass
    # 无法解析，给出通用提示
    if status_code == 429:
        return "请求过于频繁（HTTP 429），请稍后再试"
    if status_code in (401, 403):
        return f"认证失败（HTTP {status_code}），请检查 API Key"
    return f"API 返回错误（HTTP {status_code}）"


_REASONING_PART_TYPES = frozenset({
    "thinking", "reasoning", "reasoning_text", "thought", "reasoning_content",
})
_TEXT_PART_TYPES = frozenset({"text", "output_text", "answer", "output_text_delta"})
_RESPONSES_REASONING_EVENT_TYPES = frozenset({
    "response.reasoning_text.delta",
    "response.reasoning.delta",
})
_RESPONSES_TEXT_EVENT_TYPES = frozenset({
    "response.output_text.delta",
    "response.content_part.delta",
})


def _coerce_stream_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text") or value.get("content") or value.get("thinking") or ""
        return text if isinstance(text, str) else ""
    if isinstance(value, list):
        return "".join(_coerce_stream_text(item) for item in value)
    return ""


def _split_delta_content_parts(content) -> tuple[str, str]:
    """把 delta.content 的字符串 / 多模态 parts 拆成 (正文, 思考)."""
    if content is None:
        return "", ""
    if isinstance(content, str):
        return content, ""
    if isinstance(content, dict):
        part_type = str(content.get("type") or "").strip().lower()
        text = _coerce_stream_text(
            content.get("text")
            or content.get("content")
            or content.get("thinking")
            or content.get("reasoning")
        )
        if part_type in _REASONING_PART_TYPES:
            return "", text
        if part_type in _TEXT_PART_TYPES or not part_type:
            return text, ""
        return text, ""
    if isinstance(content, list):
        texts: list[str] = []
        reasons: list[str] = []
        for item in content:
            text, reason = _split_delta_content_parts(item)
            if text:
                texts.append(text)
            if reason:
                reasons.append(reason)
        return "".join(texts), "".join(reasons)
    return "", ""


def extract_reasoning_content(chunk: dict | list | str | None) -> str:
    """Normalize reasoning content across providers (DeepSeek-R1 / o1)."""
    if chunk is None:
        return ""

    # DeepSeek/OpenAI responses often nest reasoning_content under message/delta
    if isinstance(chunk, dict):
        candidate = chunk.get("reasoning_content")
        if candidate is None:
            for alt_key in ("reasoning", "thinking", "reasoning_text", "thinking_text"):
                candidate = chunk.get(alt_key)
                if candidate is not None:
                    break
            if candidate is None:
                return ""
    else:
        candidate = chunk

    coerced = _coerce_stream_text(candidate)
    if coerced:
        return coerced
    if isinstance(candidate, str):
        return candidate
    return ""


def normalize_openai_stream_delta(delta: dict | None) -> tuple[str, str]:
    """从 OpenAI delta 取出 (可见正文, 思考增量)。"""
    if not isinstance(delta, dict):
        return "", ""
    reasoning = extract_reasoning_content(delta)
    content, part_reasoning = _split_delta_content_parts(delta.get("content"))
    if part_reasoning:
        reasoning = f"{reasoning}{part_reasoning}"
    return content, reasoning


def extract_responses_api_delta(chunk: dict | None) -> tuple[str, str] | None:
    """解析 DeepSeek V4 Responses API 的 reasoning/output 增量。无匹配时返回 None。"""
    if not isinstance(chunk, dict):
        return None
    event_type = str(chunk.get("type") or "").strip()
    if event_type in _RESPONSES_REASONING_EVENT_TYPES:
        return "", _coerce_stream_text(chunk.get("delta") or chunk.get("text"))
    if event_type in _RESPONSES_TEXT_EVENT_TYPES:
        part_type = str((chunk.get("part") or {}).get("type") or "").strip().lower()
        text = _coerce_stream_text(chunk.get("delta") or chunk.get("text"))
        if part_type in _REASONING_PART_TYPES:
            return "", text
        return text, ""
    return None


def _flush_think_splitter_chunk(
    splitter: StreamingThinkSplitter,
    *,
    provider: str,
    model: str,
) -> dict | None:
    flushed = splitter.flush()
    if not flushed.content and not flushed.reasoning:
        return None
    return {
        "content": flushed.content,
        "reasoning_content": flushed.reasoning,
        "done": False,
        "used_provider": provider,
        "used_model": model,
        "fallback_used": False,
    }


def _stream_terminal_payload(
    *,
    provider: str,
    model: str,
    content_chars: int,
    reasoning_chars: int = 0,
    finish_reason: str = "",
    invalid_event_count: int = 0,
    fallback_used: bool = False,
    degraded: bool = False,
    qa_score: float | None = None,
    reasoning_resolution: dict | None = None,
) -> dict:
    """Build one provider-neutral terminal event.

    A transport completion only means the provider stopped sending events.  It
    is not a successful chat turn unless a non-whitespace answer was emitted.
    Keeping that distinction here prevents individual provider adapters from
    accidentally treating reasoning-only streams as successful answers.
    """
    outcome = resolve_completion_outcome(finish_reason=finish_reason)
    observed_reasoning_resolution = None
    if isinstance(reasoning_resolution, dict):
        observed_reasoning_resolution = dict(reasoning_resolution)
        observed_reasoning_resolution["output_observed"] = reasoning_chars > 0
        observed_reasoning_resolution["reasoning_chars"] = max(0, int(reasoning_chars or 0))
    if content_chars <= 0:
        reason_suffix = f"（finish_reason={finish_reason}）" if finish_reason else ""
        payload = {
            "error": (
                f"模型达到输出上限但尚未返回正文{reason_suffix}"
                if outcome.truncated
                else f"模型未返回正文{reason_suffix}"
            ),
            "error_code": (
                "llm_output_truncated_before_answer"
                if outcome.truncated
                else "llm_stream_empty_answer"
            ),
            "finish_reason": finish_reason,
            "completion_status": outcome.status.value,
            "truncated": outcome.truncated,
            "reasoning_chars": reasoning_chars,
            "invalid_event_count": invalid_event_count,
            "done": True,
            "used_provider": provider,
            "used_model": model,
            "fallback_used": fallback_used,
            "degraded": degraded,
        }
        if observed_reasoning_resolution is not None:
            payload["reasoning_resolution"] = observed_reasoning_resolution
        return payload

    payload = {
        "content": "",
        "done": True,
        "used_provider": provider,
        "used_model": model,
        "fallback_used": fallback_used,
        "degraded": degraded,
        "completion_status": outcome.status.value,
        "truncated": outcome.truncated,
    }
    if finish_reason:
        payload["finish_reason"] = finish_reason
    if qa_score is not None:
        payload["qa_score"] = qa_score
    if observed_reasoning_resolution is not None:
        payload["reasoning_resolution"] = observed_reasoning_resolution
    return payload


def _anthropic_stream_delta_parts(chunk: dict) -> tuple[str, str, str]:
    """Return text, thinking and finish_reason from Anthropic SSE payloads."""
    if not isinstance(chunk, dict):
        return "", "", ""

    delta = chunk.get("delta")
    if not isinstance(delta, dict):
        delta = {}
    content_block = chunk.get("content_block")
    if not isinstance(content_block, dict):
        content_block = {}

    content = delta.get("text") or content_block.get("text") or ""
    reasoning = (
        delta.get("thinking")
        or delta.get("reasoning")
        or content_block.get("thinking")
        or ""
    )
    finish_reason = delta.get("stop_reason") or chunk.get("stop_reason") or ""
    return (
        str(content) if isinstance(content, str) else "",
        str(reasoning) if isinstance(reasoning, str) else "",
        str(finish_reason) if isinstance(finish_reason, str) else "",
    )


async def call_ai_api(
    messages: List[dict],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    middlewares: List[BaseMiddleware] | None = None,
    stream: bool = False,
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    custom_params: Optional[Dict] = None,
    reasoning_effort: Optional[str] = None,
    tools: Optional[List[dict]] = None,
    purpose: str = "llm",
):
    """统一的AI API调用接口，使用 ProviderFactory 分发，可挂载中间件"""
    # 清理 API Key：去除首尾空白（处理复制粘贴带来的换行/空格），支持多 Key 轮换池
    sanitized_key = _sanitize_api_key(api_key)
    payload = {
        "messages": messages,
        "api_key": sanitized_key,
        "model": model,
        "provider": provider,
        # 如果未显式传入 endpoint，使用 ProviderRegistry 中的默认值（支持集成/单一服务商）
        "endpoint": endpoint or _provider_default_endpoint(provider)
    }

    payload = await apply_middlewares_before(payload, middlewares or [])
    # 读取 FallbackMiddleware 标记
    fb_target = payload.pop("_fallback_target", None)
    if fb_target:
        payload["_fallback_target"] = fb_target

    retry_cfg = payload.pop("_retry_cfg", None) or {"retries": 0, "delay": 0.0}
    retries = retry_cfg.get("retries", 0)
    delay = retry_cfg.get("delay", 0.0)
    timeout = payload.get("_timeout")

    client = _create_provider_client(
        payload["provider"],
        payload.get("endpoint", endpoint),
    )

    attempt = 0
    fallback_used = False
    fallback_payload = payload.copy()
    provider_messages = list(payload.get("messages") or [])
    while True:
        try:
            # Provider 适配器只接收统一的 reasoning_effort 参数和自定义请求体。
            # 先在这里完成一次能力解析，确保非流式路径与 SSE 路径使用同一套
            # 厂商映射；fallback 切换 provider/model 后会在下一轮重新解析。
            effective_custom_params = dict(custom_params or {})
            reasoning_body = dict(effective_custom_params)
            reasoning_resolution = apply_reasoning_to_payload(
                reasoning_body,
                payload["provider"],
                payload["model"],
                enable_thinking=enable_thinking,
                requested_effort=reasoning_effort,
            )
            native_reasoning_effort = reasoning_body.pop("reasoning_effort", None)
            effective_custom_params = reasoning_body
            effective_temperature, effective_top_p = sanitize_reasoning_sampling_parameters(
                payload["provider"],
                payload["model"],
                reasoning_resolution,
                temperature=effective_custom_params.get("temperature", temperature),
                top_p=effective_custom_params.get("top_p", top_p),
            )
            if effective_temperature is None:
                effective_custom_params.pop("temperature", None)
            if effective_top_p is None:
                effective_custom_params.pop("top_p", None)
            effective_max_tokens = ensure_reasoning_output_budget(
                max_tokens,
                reasoning_resolution,
            )
            provider_messages = prepare_reasoning_history_messages(
                payload.get("messages"),
                payload["provider"],
                payload["model"],
            )
            response = await client.chat(
                provider_messages,
                payload["api_key"],
                payload["model"],
                timeout=timeout,
                stream=stream,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                top_p=effective_top_p,
                custom_params=effective_custom_params,
                reasoning_effort=native_reasoning_effort,
                tools=tools,
            )
            # 如果上游返回错误结构，同样走重试逻辑
            if isinstance(response, dict) and response.get("error"):
                raise RuntimeError(response.get("error"))
            break
        except Exception as e:
            attempt += 1
            if attempt > retries:
                response = {"error": str(e)}
                # 尝试从 response/fallback 中读取备用信息
                fb = payload.get("_fallback_target")
                if fb and not fallback_used:
                    fallback_used = True
                    payload["provider"] = fb.get("provider") or payload["provider"]
                    payload["endpoint"] = _provider_default_endpoint(payload["provider"]) or endpoint
                    payload["model"] = fb.get("model") or payload["model"]
                    client = _create_provider_client(
                        payload["provider"],
                        payload.get("endpoint", endpoint),
                    )
                    attempt = 0
                    continue
                break
            if delay > 0:
                await asyncio.sleep(delay)

    # 标记使用的最终 provider/model，便于前端判断计费/来源
    if isinstance(response, dict):
        response["_used_provider"] = payload.get("provider")
        response["_used_model"] = payload.get("model")
        response["_fallback_used"] = fallback_used
        response["_completion_outcome"] = resolve_completion_outcome(
            response,
            transport_complete=not bool(response.get("error")),
        ).public()
        attach_usage_meta(
            response,
            provider=payload.get("provider"),
            model=payload.get("model"),
            purpose=purpose,
            messages=provider_messages,
        )
        if isinstance(response, dict):
            response["_reasoning_resolution"] = reasoning_resolution.public()

    response = await apply_middlewares_after(response, middlewares or [])
    return response


async def call_ai_api_stream(
    messages: List[dict],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    middlewares: List[BaseMiddleware] | None = None,
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    custom_params: Optional[Dict] = None,
    reasoning_effort: Optional[str] = None,
    purpose: str = "chat_stream",
):
    """流式调用（OpenAI 兼容走真正流式，其他回退为单次响应拆分）"""
    payload = {
        "messages": messages,
        "api_key": api_key,
        "model": model,
        "provider": provider,
        "endpoint": endpoint or _provider_default_endpoint(provider)
    }

    payload = await apply_middlewares_before(payload, middlewares or [])
    timeout = payload.get("_timeout")
    endpoint = payload.get("endpoint") or endpoint
    provider = payload.get("provider") or provider
    model = payload.get("model") or model
    messages = prepare_reasoning_history_messages(
        payload.get("messages"),
        provider,
        model,
    )

    # OpenAI 兼容流式
    if _is_openai_compatible_provider(provider, endpoint) and endpoint and _dynamic_provider_supports_streaming(provider):
        # 清理 API Key：去除首尾空白（处理复制粘贴带来的换行/空格），支持多 Key 轮换池
        sanitized_key = _sanitize_api_key(api_key)
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }
        auth_header, auth_prefix = _dynamic_provider_auth(provider)
        headers.update(
            build_api_key_headers(
                sanitized_key,
                api_key_header=auth_header,
                api_key_prefix=auth_prefix,
            )
        )
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if provider.lower() in OPENAI_NATIVE:
            body["stream_options"] = {"include_usage": True}
        # 仅在参数非 None 时添加对应字段
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        # 合并自定义参数
        if custom_params:
            merge_request_body(body, custom_params)
        reasoning_resolution = apply_reasoning_to_payload(
            body,
            provider,
            model,
            enable_thinking=enable_thinking,
            requested_effort=reasoning_effort,
        )
        effective_temperature, effective_top_p = sanitize_reasoning_sampling_parameters(
            provider,
            model,
            reasoning_resolution,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
        )
        if effective_temperature is None:
            body.pop("temperature", None)
        else:
            body["temperature"] = effective_temperature
        if effective_top_p is None:
            body.pop("top_p", None)
        else:
            body["top_p"] = effective_top_p
        think_splitter = StreamingThinkSplitter()
        if reasoning_resolution.enabled:
            # DeepSeek 的 reasoning 与正文共享 completion 预算。标准回答默认的
            # 1000/2024 tokens 可能被思考过程耗尽，最终只返回 reasoning 而没有正文。
            if provider.lower() == "deepseek" or "deepseek" in str(model or "").lower():
                try:
                    configured_max_tokens = int(body.get("max_tokens") or 0)
                except (TypeError, ValueError):
                    configured_max_tokens = 0
                body["max_tokens"] = max(configured_max_tokens, 8192)
            elif reasoning_resolution.mode in {"thinking_toggle", "qwen_budget"} and "max_tokens" not in body:
                body["max_tokens"] = 8192
            # 思考模式下不支持 temperature，移除避免报错
            body.pop("temperature", None)

        # 推理模型（自动思考）同样不支持 temperature
        if not reasoning_resolution.enabled and _is_reasoning_model(model):
            body.pop("temperature", None)

        # 置信度评分：请求 logprobs（仅非思考模式且非推理模型，避免干扰）
        if (
            not enable_thinking
            and not _is_reasoning_model(model)
            and provider.lower() in OPENAI_LIKE
        ):
            body["logprobs"] = True
            body.setdefault("top_logprobs", 1)

        # ── 诊断日志 ──
        logger.debug(f"[Stream] ▶ provider={provider}, model={model}, endpoint={endpoint}, enable_thinking={enable_thinking}, body keys={list(body.keys())}")
        _chunk_count = 0
        _content_chars = 0
        _reasoning_chars = 0
        _logprobs_sum = 0.0
        _logprobs_count = 0
        _finish_reason = ""
        _invalid_event_count = 0

        try:
            async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
                async with client.stream("POST", endpoint, headers=headers, json=body) as resp:
                    logger.debug(f"[Stream] HTTP {resp.status_code}")
                    if resp.status_code != 200:
                        err_text = await resp.aread()
                        err_body = err_text.decode("utf-8", errors="ignore")
                        logger.warning(f"[Stream] Error body: {err_body[:500]}")
                        yield {"error": _extract_api_error_message(err_body, resp.status_code), "done": True}
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        # 前 3 行原始 SSE 打印，帮助诊断格式问题
                        if _chunk_count < 3:
                            logger.debug(f"[Stream] raw[{_chunk_count}]: {line[:200]}")
                        # 兼容 "data: " 和 "data:" 两种 SSE 前缀（某些代理/服务商省略空格）
                        if line.startswith("data: "):
                            data = line[6:].strip()
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                        else:
                            data = line.strip()
                        if data == "[DONE]":
                            flushed_chunk = _flush_think_splitter_chunk(
                                think_splitter, provider=provider, model=model
                            )
                            if flushed_chunk:
                                _chunk_count += 1
                                _content_chars += len(str(flushed_chunk.get("content") or "").strip())
                                _reasoning_chars += len(str(flushed_chunk.get("reasoning_content") or ""))
                                yield flushed_chunk
                            logger.debug(f"[Stream] done chunks={_chunk_count}, content_chars={_content_chars}, reasoning_chars={_reasoning_chars}")
                            qa_score = None
                            if _logprobs_count > 0:
                                import math
                                qa_score = round(math.exp(_logprobs_sum / _logprobs_count), 4)
                            yield _stream_terminal_payload(
                                provider=provider,
                                model=model,
                                content_chars=_content_chars,
                                reasoning_chars=_reasoning_chars,
                                finish_reason=_finish_reason,
                                invalid_event_count=_invalid_event_count,
                                qa_score=qa_score,
                                reasoning_resolution=reasoning_resolution.public(),
                            )
                            return
                        try:
                            chunk = _json.loads(data)
                        except Exception:
                            _invalid_event_count += 1
                            continue
                        # Detect API-level errors embedded inside HTTP-200 SSE bodies
                        # (e.g. Doubao / volcengine returns {"error": {...}} with status 200)
                        api_error = chunk.get("error")
                        if api_error:
                            if isinstance(api_error, dict):
                                err_msg = api_error.get("message") or api_error.get("msg") or str(api_error)
                            else:
                                err_msg = str(api_error)
                            logger.warning(f"[Stream] API error in SSE: {err_msg}")
                            yield {"error": err_msg, "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
                            return
                        # 防止 choices 为空列表时 [0] 抛 IndexError
                        if chunk.get("usage"):
                            yield {
                                "content": "",
                                "reasoning_content": "",
                                "done": False,
                                "usage": chunk.get("usage"),
                                "used_provider": provider,
                                "used_model": model,
                                "fallback_used": False,
                            }

                        responses_delta = extract_responses_api_delta(chunk)
                        if responses_delta is not None:
                            content, reasoning_content = responses_delta
                            if isinstance(content, str):
                                content, reasoning_content = apply_stream_think_split(
                                    think_splitter,
                                    content,
                                    reasoning_content,
                                )
                            if content or reasoning_content:
                                _chunk_count += 1
                                _content_chars += len(str(content).strip())
                                _reasoning_chars += len(reasoning_content)
                                if reasoning_content and _reasoning_chars <= 500:
                                    import time as _time
                                    logger.info(
                                        f"[Stream] REASONING chunk#{_chunk_count} t={_time.time():.3f} "
                                        f"len={len(reasoning_content)} first40={reasoning_content[:40]!r}"
                                    )
                                yield {
                                    "content": content,
                                    "reasoning_content": reasoning_content,
                                    "done": False,
                                    "used_provider": provider,
                                    "used_model": model,
                                    "fallback_used": False,
                                }
                            continue

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        if choice.get("finish_reason"):
                            _finish_reason = str(choice.get("finish_reason"))
                        delta = choice.get("delta") or choice.get("message") or {}
                        content, reasoning_content = normalize_openai_stream_delta(delta)
                        # MiniMax 的思考内容在 reasoning_details 字段中
                        if not reasoning_content:
                            reasoning_details = delta.get("reasoning_details") or choice.get("reasoning_details")
                            if reasoning_details:
                                reasoning_content = extract_reasoning_content(reasoning_details)
                        if not reasoning_content:
                            reasoning_content = (
                                extract_reasoning_content(choice)
                                or extract_reasoning_content(chunk)
                            )
                        if isinstance(content, str):
                            content, reasoning_content = apply_stream_think_split(
                                think_splitter,
                                content,
                                reasoning_content,
                            )
                        # 收集 logprobs 用于置信度评分
                        chunk_logprobs = choice.get("logprobs")
                        if chunk_logprobs and isinstance(chunk_logprobs, dict):
                            for token_info in (chunk_logprobs.get("content") or []):
                                lp = token_info.get("logprob")
                                if lp is not None and isinstance(lp, (int, float)):
                                    _logprobs_sum += lp
                                    _logprobs_count += 1
                        # 只要有内容或推理内容，就 yield。
                        if content or reasoning_content:
                            _chunk_count += 1
                            _content_chars += len(content.strip())
                            _reasoning_chars += len(reasoning_content)
                            if reasoning_content and _reasoning_chars <= 500:
                                import time as _time
                                logger.info(f"[Stream] REASONING chunk#{_chunk_count} t={_time.time():.3f} len={len(reasoning_content)} first40={reasoning_content[:40]!r}")
                            yield {
                                "content": content,
                                "reasoning_content": reasoning_content,
                                "done": False,
                                "used_provider": provider,
                                "used_model": model,
                                "fallback_used": False
                            }
                        elif _chunk_count == 0:
                            # 发送一个空的心跳包，防止前端因长时间拿不到第一个 chunk 而判定超时/无响应
                            yield {
                                "content": "",
                                "done": False,
                                "used_provider": provider,
                                "used_model": model,
                                "fallback_used": False
                            }
                    flushed_chunk = _flush_think_splitter_chunk(
                        think_splitter, provider=provider, model=model
                    )
                    if flushed_chunk:
                        _chunk_count += 1
                        _content_chars += len(str(flushed_chunk.get("content") or "").strip())
                        _reasoning_chars += len(str(flushed_chunk.get("reasoning_content") or ""))
                        yield flushed_chunk
                    logger.debug(f"[Stream] end-of-stream (no [DONE]) chunks={_chunk_count}, content_chars={_content_chars}, reasoning_chars={_reasoning_chars}")
                    qa_score = None
                    if _logprobs_count > 0:
                        import math
                        qa_score = round(math.exp(_logprobs_sum / _logprobs_count), 4)
                    yield _stream_terminal_payload(
                        provider=provider,
                        model=model,
                        content_chars=_content_chars,
                        reasoning_chars=_reasoning_chars,
                        finish_reason=_finish_reason,
                        invalid_event_count=_invalid_event_count,
                        qa_score=qa_score,
                        reasoning_resolution=reasoning_resolution.public(),
                    )
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, ConnectionError, OSError) as e:
            logger.warning(f"[Stream] connection interrupted: {type(e).__name__}: {e}")
            yield {"error": f"LLM API connection interrupted: {type(e).__name__}", "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
        except Exception as e:
            logger.warning(f"[Stream] unknown error: {type(e).__name__}: {e}")
            yield {"error": f"LLM API call failed: {type(e).__name__}", "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
        return

    # Anthropic 流式
    if _is_anthropic_provider(provider):
        sanitized_key = _sanitize_api_key(api_key)
        auth_header, auth_prefix = _dynamic_provider_auth(provider)
        headers = build_api_key_headers(
            sanitized_key,
            provider_type="anthropic",
            api_key_header=auth_header,
            api_key_prefix=auth_prefix,
            extra_headers={
                "anthropic-version": "2023-06-01",
            },
        )
        body = {
            "model": model,
            "messages": [m for m in messages if m.get("role") != "system"],
            "system": next((m["content"] for m in messages if m.get("role") == "system"), ""),
            "stream": True
        }
        # 仅在参数非 None 时添加对应字段
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        # 合并自定义参数
        if custom_params:
            merge_request_body(body, custom_params)
        reasoning_resolution = apply_reasoning_to_payload(
            body,
            provider,
            model,
            provider_type="anthropic",
            enable_thinking=enable_thinking,
            requested_effort=reasoning_effort,
        )
        effective_temperature, effective_top_p = sanitize_reasoning_sampling_parameters(
            provider,
            model,
            reasoning_resolution,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
        )
        if effective_temperature is None:
            body.pop("temperature", None)
        else:
            body["temperature"] = effective_temperature
        if effective_top_p is None:
            body.pop("top_p", None)
        else:
            body["top_p"] = effective_top_p
        # Anthropic requires the total output cap to exceed the thinking
        # budget. A normal 1k chat cap would otherwise fail before an answer.
        if reasoning_resolution.enabled:
            # Anthropic requires the total output cap to exceed the thinking
            # budget.  A normal 1k chat cap would otherwise fail before any
            # visible answer is generated.
            try:
                configured_max_tokens = int(body.get("max_tokens") or 0)
            except (TypeError, ValueError):
                configured_max_tokens = 0
            thinking_budget = reasoning_resolution.budget_tokens or 8192
            body["max_tokens"] = max(configured_max_tokens, thinking_budget + 1024)
        else:
            body.setdefault("max_tokens", 8192)
        _content_chars = 0
        _reasoning_chars = 0
        _finish_reason = ""
        _invalid_event_count = 0
        try:
            async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
                async with client.stream("POST", endpoint or "https://api.anthropic.com/v1/messages", headers=headers, json=body) as resp:
                    if resp.status_code != 200:
                        err_text = await resp.aread()
                        err_body = err_text.decode("utf-8", errors="ignore")
                        yield {
                            "error": _extract_api_error_message(err_body, resp.status_code),
                            "done": True,
                            "used_provider": provider,
                            "used_model": model,
                            "fallback_used": False,
                        }
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data = line[6:].strip() if line.startswith("data: ") else line.strip()
                        if data == "[DONE]":
                            yield _stream_terminal_payload(
                                provider=provider,
                                model=model,
                                content_chars=_content_chars,
                                reasoning_chars=_reasoning_chars,
                                finish_reason=_finish_reason,
                                invalid_event_count=_invalid_event_count,
                                reasoning_resolution=reasoning_resolution.public(),
                            )
                            return
                        try:
                            chunk = _json.loads(data)
                        except Exception:
                            _invalid_event_count += 1
                            continue
                        api_error = chunk.get("error") if isinstance(chunk, dict) else None
                        if api_error:
                            error_message = (
                                api_error.get("message")
                                if isinstance(api_error, dict)
                                else str(api_error)
                            )
                            yield {
                                "error": error_message or "Anthropic stream error",
                                "done": True,
                                "used_provider": provider,
                                "used_model": model,
                                "fallback_used": False,
                            }
                            return
                        content, reasoning_content, finish_reason = _anthropic_stream_delta_parts(chunk)
                        if finish_reason:
                            _finish_reason = finish_reason
                        if not content and not reasoning_content:
                            continue
                        _content_chars += len(content.strip())
                        _reasoning_chars += len(reasoning_content)
                        yield {
                            "content": content,
                            "reasoning_content": reasoning_content,
                            "done": False,
                            "used_provider": provider,
                            "used_model": model,
                            "fallback_used": False,
                        }
                    yield _stream_terminal_payload(
                        provider=provider,
                        model=model,
                        content_chars=_content_chars,
                        reasoning_chars=_reasoning_chars,
                        finish_reason=_finish_reason,
                        invalid_event_count=_invalid_event_count,
                        reasoning_resolution=reasoning_resolution.public(),
                    )
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, ConnectionError, OSError) as exc:
            logger.warning("[AnthropicStream] connection interrupted: %s", exc)
            yield {
                "error": f"LLM API connection interrupted: {type(exc).__name__}",
                "done": True,
                "used_provider": provider,
                "used_model": model,
                "fallback_used": False,
            }
        except Exception as exc:
            logger.warning("[AnthropicStream] failed: %s", exc)
            yield {
                "error": f"LLM API call failed: {type(exc).__name__}",
                "done": True,
                "used_provider": provider,
                "used_model": model,
                "fallback_used": False,
            }
        return

    # Gemini 流式（简单版，若失败则回退）
    if provider.lower() in GEMINI:
        sanitized_key = _sanitize_api_key(api_key)
        # Gemini 流式 endpoint：:streamGenerateContent
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={sanitized_key}"
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            parts = []
            if isinstance(msg["content"], str):
                parts.append({"text": msg["content"]})
            elif isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "text":
                        parts.append({"text": item["text"]})
            contents.append({"role": "user" if msg["role"] == "user" else "model", "parts": parts})

        payload = {
            "contents": contents,
            "stream": True,
        }
        # 仅在参数非 None 时添加 generationConfig 对应字段
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if top_p is not None:
            generation_config["topP"] = top_p
        if generation_config:
            payload["generationConfig"] = generation_config
        # 合并自定义参数
        if custom_params:
            merge_request_body(payload, custom_params)
        reasoning_resolution = apply_reasoning_to_payload(
            payload,
            provider,
            model,
            provider_type="gemini",
            enable_thinking=enable_thinking,
            requested_effort=reasoning_effort,
        )
        if reasoning_resolution.enabled and reasoning_resolution.budget_tokens:
            generation = payload.get("generationConfig")
            if isinstance(generation, dict):
                generation["maxOutputTokens"] = ensure_reasoning_output_budget(
                    generation.get("maxOutputTokens"),
                    reasoning_resolution,
                )

        _content_chars = 0
        _reasoning_chars = 0
        _finish_reason = ""
        _invalid_event_count = 0
        try:
            async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
                async with client.stream("POST", endpoint, json=payload) as resp:
                    if resp.status_code != 200:
                        err_text = await resp.aread()
                        err_body = err_text.decode("utf-8", errors="ignore")
                        yield {
                            "error": _extract_api_error_message(err_body, resp.status_code),
                            "done": True,
                            "used_provider": provider,
                            "used_model": model,
                            "fallback_used": False,
                        }
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data = line[6:].strip() if line.startswith("data: ") else line.strip()
                        if data == "[DONE]":
                            yield _stream_terminal_payload(
                                provider=provider,
                                model=model,
                                content_chars=_content_chars,
                                reasoning_chars=_reasoning_chars,
                                finish_reason=_finish_reason,
                                invalid_event_count=_invalid_event_count,
                                reasoning_resolution=reasoning_resolution.public(),
                            )
                            return
                        try:
                            chunk = _json.loads(data)
                        except Exception:
                            _invalid_event_count += 1
                            continue
                        api_error = chunk.get("error") if isinstance(chunk, dict) else None
                        if api_error:
                            error_message = (
                                api_error.get("message")
                                if isinstance(api_error, dict)
                                else str(api_error)
                            )
                            yield {
                                "error": error_message or "Gemini stream error",
                                "done": True,
                                "used_provider": provider,
                                "used_model": model,
                                "fallback_used": False,
                            }
                            return
                        candidates = chunk.get("candidates", []) if isinstance(chunk, dict) else []
                        for candidate in candidates:
                            if not isinstance(candidate, dict):
                                continue
                            if candidate.get("finishReason"):
                                _finish_reason = str(candidate.get("finishReason"))
                            parts = (candidate.get("content") or {}).get("parts", [])
                            for part in parts if isinstance(parts, list) else []:
                                if not isinstance(part, dict):
                                    continue
                                text = str(part.get("text") or "")
                                if not text:
                                    continue
                                if part.get("thought") is True:
                                    _reasoning_chars += len(text)
                                    yield {
                                        "content": "",
                                        "reasoning_content": text,
                                        "done": False,
                                        "used_provider": provider,
                                        "used_model": model,
                                        "fallback_used": False,
                                    }
                                else:
                                    _content_chars += len(text.strip())
                                    yield {
                                        "content": text,
                                        "done": False,
                                        "used_provider": provider,
                                        "used_model": model,
                                        "fallback_used": False,
                                    }
                    yield _stream_terminal_payload(
                        provider=provider,
                        model=model,
                        content_chars=_content_chars,
                        reasoning_chars=_reasoning_chars,
                        finish_reason=_finish_reason,
                        invalid_event_count=_invalid_event_count,
                        reasoning_resolution=reasoning_resolution.public(),
                    )
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, ConnectionError, OSError) as exc:
            logger.warning("[GeminiStream] connection interrupted: %s", exc)
            yield {
                "error": f"LLM API connection interrupted: {type(exc).__name__}",
                "done": True,
                "used_provider": provider,
                "used_model": model,
                "fallback_used": False,
            }
        except Exception as exc:
            logger.warning("[GeminiStream] failed: %s", exc)
            yield {
                "error": f"LLM API call failed: {type(exc).__name__}",
                "done": True,
                "used_provider": provider,
                "used_model": model,
                "fallback_used": False,
            }
        return

    # 其他 provider 回退为一次性响应
    try:
        resp = await call_ai_api(messages, api_key, model, provider, endpoint=endpoint, middlewares=middlewares,
                                enable_thinking=enable_thinking, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                custom_params=custom_params, reasoning_effort=reasoning_effort,
                                purpose=purpose)
        if isinstance(resp, dict) and resp.get("error"):
            raise RuntimeError(str(resp.get("error")))
        message = resp.get("choices", [{}])[0].get("message", {}) or {}
        answer = str(message.get("content") or "")
        reasoning_text = extract_reasoning_content(message)
        tagged_reasoning, answer = split_think_tags(answer)
        if tagged_reasoning:
            reasoning_text = f"{reasoning_text}{tagged_reasoning}"
        used_provider = resp.get("_used_provider", provider)
        used_model = resp.get("_used_model", model)
        fallback_used = resp.get("_fallback_used", False)
        degraded = bool(resp.get("degraded") or resp.get("answer_status") == "degraded")
        if reasoning_text:
            yield {
                "content": "",
                "reasoning_content": reasoning_text,
                "done": False,
                "used_provider": used_provider,
                "used_model": used_model,
                "fallback_used": fallback_used,
                "degraded": degraded,
            }
        for idx, word in enumerate(answer.split(" ")):
            chunk = word if idx == 0 else f" {word}"
            if chunk:
                yield {
                    "content": chunk,
                    "done": False,
                    "used_provider": used_provider,
                    "used_model": used_model,
                    "fallback_used": fallback_used,
                    "degraded": degraded,
                }
        yield _stream_terminal_payload(
            provider=used_provider,
            model=used_model,
            content_chars=len(answer.strip()),
            reasoning_chars=len(reasoning_text),
            fallback_used=fallback_used,
            degraded=degraded,
            reasoning_resolution=resp.get("_reasoning_resolution"),
        )
    except Exception as e:
        yield {"error": str(e), "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
