import httpx
from fastapi import HTTPException
from typing import Dict, List, Optional

from .base import BaseProvider


class GrokProvider(BaseProvider):
    """xAI Grok Provider (OpenAI兼容格式)"""

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
        # 构建请求体，仅在参数非 None 时添加对应字段
        body = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        # 合并自定义参数
        if custom_params:
            body.update(custom_params)

        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            response = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=body,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Grok API错误: {response.text}"
                )

            return response.json()
