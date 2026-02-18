from typing import Dict, List, Optional
import asyncio
import json as _json
import httpx

from providers.factory import ProviderFactory
from providers.provider_ids import OPENAI_LIKE, ANTHROPIC, GEMINI, OPENAI_NATIVE, MINIMAX, MOONSHOT
from models.provider_registry import PROVIDER_CONFIG
from models.api_key_selector import select_api_key
from utils.middleware import (
    BaseMiddleware,
    apply_middlewares_before,
    apply_middlewares_after,
    RetryMiddleware,
    FallbackMiddleware,
)


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


def extract_reasoning_content(chunk: dict | list | str | None) -> str:
    """Normalize reasoning content across providers (DeepSeek-R1 / o1)."""
    if chunk is None:
        return ""

    # DeepSeek/OpenAI responses often nest reasoning_content under message/delta
    if isinstance(chunk, dict):
        candidate = chunk.get("reasoning_content")
        if candidate is None:
            return ""
    else:
        candidate = chunk

    if isinstance(candidate, str):
        return candidate

    if isinstance(candidate, dict):
        text = candidate.get("text") or candidate.get("content") or ""
        return text if isinstance(text, str) else ""

    if isinstance(candidate, list):
        parts = []
        for item in candidate:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return ""


async def call_ai_api(
    messages: List[dict],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    middlewares: List[BaseMiddleware] | None = None,
    stream: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    custom_params: Optional[Dict] = None,
    reasoning_effort: Optional[str] = None,
):
    """统一的AI API调用接口，使用 ProviderFactory 分发，可挂载中间件"""
    # 清理 API Key：去除首尾空白（处理复制粘贴带来的换行/空格），支持多 Key 轮换池
    sanitized_key = select_api_key(api_key) or (api_key.strip() if api_key else "")
    payload = {
        "messages": messages,
        "api_key": sanitized_key,
        "model": model,
        "provider": provider,
        # 如果未显式传入 endpoint，使用 ProviderRegistry 中的默认值（支持集成/单一服务商）
        "endpoint": endpoint or PROVIDER_CONFIG.get(provider, {}).get("endpoint", "")
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

    client = ProviderFactory.create(payload["provider"], payload.get("endpoint", endpoint))

    attempt = 0
    fallback_used = False
    fallback_payload = payload.copy()
    while True:
        try:
            response = await client.chat(
                payload["messages"],
                payload["api_key"],
                payload["model"],
                timeout=timeout,
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                custom_params=custom_params,
                reasoning_effort=reasoning_effort,
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
                    payload["endpoint"] = PROVIDER_CONFIG.get(payload["provider"], {}).get("endpoint", endpoint)
                    payload["model"] = fb.get("model") or payload["model"]
                    client = ProviderFactory.create(payload["provider"], payload.get("endpoint", endpoint))
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
):
    """流式调用（OpenAI 兼容走真正流式，其他回退为单次响应拆分）"""
    payload = {
        "messages": messages,
        "api_key": api_key,
        "model": model,
        "provider": provider,
        "endpoint": endpoint or PROVIDER_CONFIG.get(provider, {}).get("endpoint", "")
    }

    payload = await apply_middlewares_before(payload, middlewares or [])
    timeout = payload.get("_timeout")
    endpoint = payload.get("endpoint") or endpoint
    provider = payload.get("provider") or provider
    model = payload.get("model") or model

    # OpenAI 兼容流式
    if provider.lower() in OPENAI_LIKE and endpoint:
        # 清理 API Key：去除首尾空白（处理复制粘贴带来的换行/空格），支持多 Key 轮换池
        sanitized_key = select_api_key(api_key) or api_key.strip()
        headers = {
            "Authorization": f"Bearer {sanitized_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        # 仅在参数非 None 时添加对应字段
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        # 透传 reasoning_effort 参数
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        # 合并自定义参数
        if custom_params:
            body.update(custom_params)
        # 深度思考模式：根据 provider 使用不同参数
        if enable_thinking:
            if provider.lower() in OPENAI_NATIVE:
                # OpenAI 原生 API（GPT-5/o3/o4 系列）：使用 reasoning_effort 参数
                # 如果前端已传入 reasoning_effort 则优先使用，否则默认 high
                if "reasoning_effort" not in body:
                    body["reasoning_effort"] = "high"
            elif provider.lower() in MINIMAX:
                # MiniMax：使用 reasoning_split 分离思考内容
                body["reasoning_split"] = True
            elif provider.lower() not in MOONSHOT:
                # DeepSeek / 智谱 / 通用 OpenAI 兼容：使用 thinking 参数
                # Moonshot/Kimi 的思考模型自动输出，无需额外参数
                body["thinking"] = {"type": "enabled"}
            # 思考模式下不支持 temperature，移除避免报错
            body.pop("temperature", None)

        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            async with client.stream("POST", endpoint, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    err_body = err_text.decode("utf-8", errors="ignore")
                    yield {"error": _extract_api_error_message(err_body, resp.status_code), "done": True}
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = line[6:].strip() if line.startswith("data: ") else line.strip()
                    if data == "[DONE]":
                        yield {"content": "", "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
                        return
                    try:
                        chunk = _json.loads(data)
                    except Exception:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {}) or chunk.get("choices", [{}])[0].get("message", {})
                    content = delta.get("content") or ""
                    reasoning_content = extract_reasoning_content(delta)
                    # MiniMax 的思考内容在 reasoning_details 字段中
                    if not reasoning_content:
                        reasoning_details = delta.get("reasoning_details") or chunk.get("choices", [{}])[0].get("reasoning_details")
                        if reasoning_details:
                            reasoning_content = extract_reasoning_content(reasoning_details)
                    if content or reasoning_content:
                        yield {
                            "content": content,
                            "reasoning_content": reasoning_content,
                            "done": False,
                            "used_provider": provider,
                            "used_model": model,
                            "fallback_used": False
                        }
                yield {"content": "", "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
        return

    # Anthropic 流式
    if provider.lower() in ANTHROPIC:
        sanitized_key = select_api_key(api_key) or api_key.strip()
        headers = {
            "x-api-key": sanitized_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
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
            body.update(custom_params)
        # 深度思考模式：Anthropic extended thinking
        if enable_thinking:
            body["thinking"] = {"type": "enabled", "budget_tokens": 8192}
        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            async with client.stream("POST", "https://api.anthropic.com/v1/messages", headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    err_body = err_text.decode("utf-8", errors="ignore")
                    yield {"error": _extract_api_error_message(err_body, resp.status_code), "done": True}
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = line[6:].strip() if line.startswith("data: ") else line.strip()
                    if data == "[DONE]":
                        yield {"content": "", "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
                        return
                    try:
                        chunk = httpx.Response(200, content=data).json()
                    except Exception:
                        continue
                    # Anthropic streaming fields: delta -> text
                    delta_list = chunk.get("delta") or []
                    for delta in delta_list:
                        content = delta.get("text", "")
                        if content:
                            yield {"content": content, "done": False, "used_provider": provider, "used_model": model, "fallback_used": False}
                yield {"content": "", "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
        return

    # Gemini 流式（简单版，若失败则回退）
    if provider.lower() in GEMINI:
        sanitized_key = select_api_key(api_key) or api_key.strip()
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
            payload.update(custom_params)
        # 深度思考模式：Gemini thinkingConfig
        if enable_thinking:
            if "generationConfig" not in payload:
                payload["generationConfig"] = {}
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 8192}

        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            async with client.stream("POST", endpoint, json=payload) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    err_body = err_text.decode("utf-8", errors="ignore")
                    yield {"error": _extract_api_error_message(err_body, resp.status_code), "done": True}
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = line[6:].strip() if line.startswith("data: ") else line.strip()
                    if data == "[DONE]":
                        yield {"content": "", "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
                        return
                    try:
                        import json as _json
                        chunk = _json.loads(data)
                    except Exception:
                        continue
                    # Gemini streaming uses candidates[].content.parts[].text
                    candidates = chunk.get("candidates", [])
                    for cand in candidates:
                        parts = cand.get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text") or ""
                            if text:
                                yield {"content": text, "done": False, "used_provider": provider, "used_model": model, "fallback_used": False}
                yield {"content": "", "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
        return

    # 其他 provider 回退为一次性响应
    try:
        resp = await call_ai_api(messages, api_key, model, provider, endpoint=endpoint, middlewares=middlewares,
                                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                custom_params=custom_params, reasoning_effort=reasoning_effort)
        message = resp.get("choices", [{}])[0].get("message", {}) or {}
        answer = message.get("content", "")
        reasoning_text = extract_reasoning_content(message)
        for idx, word in enumerate(answer.split(" ")):
            chunk = word if idx == 0 else f" {word}"
            yield {"content": chunk, "done": False, "used_provider": resp.get("_used_provider", provider), "used_model": resp.get("_used_model", model), "fallback_used": resp.get("_fallback_used", False)}
        yield {
            "content": "",
            "reasoning_content": reasoning_text,
            "done": True,
            "used_provider": resp.get("_used_provider", provider),
            "used_model": resp.get("_used_model", model),
            "fallback_used": resp.get("_fallback_used", False)
        }
    except Exception as e:
        yield {"error": str(e), "done": True, "used_provider": provider, "used_model": model, "fallback_used": False}
