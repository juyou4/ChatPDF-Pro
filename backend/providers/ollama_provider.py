import httpx
from fastapi import HTTPException
from typing import Dict, List, Optional

from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """本地 Ollama Provider"""

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
    ) -> dict:  # api_key 未使用，保留以兼容接口
        # 构建请求体，仅在参数非 None 时添加对应字段
        options = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        # Ollama 不支持 reasoning_effort，忽略该参数
        # 合并自定义参数
        if custom_params:
            payload.update(custom_params)

        try:
            async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json=payload,
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
