from typing import List
import asyncio
import httpx

from providers.factory import ProviderFactory
from providers.provider_ids import OPENAI_LIKE, ANTHROPIC, GEMINI, OPENAI_NATIVE, MINIMAX, MOONSHOT
from models.provider_registry import PROVIDER_CONFIG
from utils.middleware import (
    BaseMiddleware,
    apply_middlewares_before,
    apply_middlewares_after,
    RetryMiddleware,
    FallbackMiddleware,
)


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
):
    """统一的AI API调用接口，使用 ProviderFactory 分发，可挂载中间件"""
    payload = {
        "messages": messages,
        "api_key": api_key,
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
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 4000
        }
        # 深度思考模式：根据 provider 使用不同参数
        if enable_thinking:
            if provider.lower() in OPENAI_NATIVE:
                # OpenAI 原生 API（GPT-5/o3/o4 系列）：使用 reasoning_effort 参数
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
                    yield {"error": err_text.decode("utf-8", errors="ignore"), "done": True}
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
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": [m for m in messages if m.get("role") != "system"],
            "system": next((m["content"] for m in messages if m.get("role") == "system"), ""),
            "max_tokens": 4000,
            "stream": True
        }
        # 深度思考模式：Anthropic extended thinking
        if enable_thinking:
            body["thinking"] = {"type": "enabled", "budget_tokens": 8192}
        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            async with client.stream("POST", "https://api.anthropic.com/v1/messages", headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    yield {"error": err_text.decode("utf-8", errors="ignore"), "done": True}
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
        # Gemini 流式 endpoint：:streamGenerateContent
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
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
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4000},
            "stream": True,
        }
        # 深度思考模式：Gemini thinkingConfig
        if enable_thinking:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 8192}

        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            async with client.stream("POST", endpoint, json=payload) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    yield {"error": err_text.decode("utf-8", errors="ignore"), "done": True}
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
        resp = await call_ai_api(messages, api_key, model, provider, endpoint=endpoint, middlewares=middlewares)
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
