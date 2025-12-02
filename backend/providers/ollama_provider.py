import httpx
from fastapi import HTTPException
from typing import List, Optional

from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """本地 Ollama Provider"""

    async def chat(self, messages: List[dict], api_key: str, model: str, timeout: Optional[float] = None, stream: bool = False) -> dict:  # api_key unused, kept for interface
        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream
            }

            try:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json=payload
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Ollama API错误: {response.text}"
                    )

                result = response.json()
                return {
                    "choices": [{
                        "message": {
                            "content": result["message"]["content"]
                        }
                    }]
                }
            except httpx.ConnectError:
                raise HTTPException(
                    status_code=503,
                    detail="无法连接到本地Ollama服务，请确保Ollama已启动 (localhost:11434)"
                )
