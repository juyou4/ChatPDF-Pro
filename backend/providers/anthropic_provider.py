import httpx
from fastapi import HTTPException
from typing import List, Optional

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic Claude Provider"""

    async def chat(self, messages: List[dict], api_key: str, model: str, timeout: Optional[float] = None, stream: bool = False) -> dict:
        system_message = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "max_tokens": 4000,
                    "system": system_message,
                    "messages": user_messages,
                    "stream": stream
                }
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
