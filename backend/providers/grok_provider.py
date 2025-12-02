import httpx
from fastapi import HTTPException
from typing import List, Optional

from .base import BaseProvider


class GrokProvider(BaseProvider):
    """xAI Grok Provider (OpenAI兼容格式)"""

    async def chat(self, messages: List[dict], api_key: str, model: str, timeout: Optional[float] = None, stream: bool = False) -> dict:
        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            response = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "stream": stream
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Grok API错误: {response.text}"
                )

            return response.json()
