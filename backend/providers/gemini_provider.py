import httpx
from fastapi import HTTPException
from typing import Dict, List, Optional

from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Google Gemini Provider (支持图片)"""

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

        # 构建 generationConfig，仅在参数非 None 时添加对应字段
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if top_p is not None:
            generation_config["topP"] = top_p

        body = {
            "contents": contents,
            "stream": stream,
        }
        if generation_config:
            body["generationConfig"] = generation_config
        # Gemini 不支持 reasoning_effort，忽略该参数
        # 合并自定义参数
        if custom_params:
            body.update(custom_params)

        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            response = await client.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=body,
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
