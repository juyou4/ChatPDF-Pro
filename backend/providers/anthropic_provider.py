import httpx
from fastapi import HTTPException
from typing import Dict, List, Optional

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic Claude Provider"""

    async def chat(
        self,
        messages: List[dict],
        api_key: str,
        model: str,
        timeout: Optional[float] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        custom_params: Optional[Dict] = None,
        reasoning_effort: Optional[str] = None,
    ) -> dict:
        system_message = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        # 构建请求体，仅在参数非 None 时添加对应字段
        body = {
            "model": model,
            "messages": user_messages,
            "stream": stream,
        }
        if system_message:
            body["system"] = system_message
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        # Anthropic 不支持 reasoning_effort，忽略该参数
        # 合并自定义参数
        if custom_params:
            body.update(custom_params)

        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json=body,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Anthropic API错误: {response.text}"
                )

            result = response.json()
            return {
                "choices": [{
                    "message": {
                        "content": result["content"][0]["text"]
                    }
                }]
            }
