import httpx
from fastapi import HTTPException
from typing import List, Optional

from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Google Gemini Provider (支持图片)"""

    async def chat(self, messages: List[dict], api_key: str, model: str, timeout: Optional[float] = None, stream: bool = False) -> dict:
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
                    elif item["type"] == "image_url":
                        image_data = item["image_url"]["url"].split(",")[1]
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_data
                            }
                        })

            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": parts
            })

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            response = await client.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": contents,
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 4000
                    },
                    "stream": stream
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Gemini API错误: {response.text}"
                )

            result = response.json()
            return {
                "choices": [{
                    "message": {
                        "content": result["candidates"][0]["content"]["parts"][0]["text"]
                    }
                }]
            }
