import json
import logging
import httpx
from fastapi import HTTPException
from typing import Dict, List, Optional

from .base import BaseProvider

logger = logging.getLogger(__name__)


def _parse_ollama_version(version_str: str) -> tuple:
    """解析 Ollama 版本号为可比较的元组，如 '0.1.50' -> (0, 1, 50)"""
    try:
        parts = version_str.strip().split(".")
        return tuple(int(p) for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


async def _get_ollama_version(timeout: float = 5.0) -> str:
    """探测本地 Ollama 版本号，失败时返回空字符串"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get("http://localhost:11434/api/version")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("version", "")
    except Exception:
        pass
    return ""


class OllamaProvider(BaseProvider):
    """本地 Ollama Provider"""

    # 支持 tools 的最低版本
    _MIN_TOOLS_VERSION = (0, 1, 50)

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
        tools: Optional[List[dict]] = None,
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

        # 探测 Ollama 版本，仅在版本 >= 0.1.50 时传入 tools
        if tools:
            version_str = await _get_ollama_version(timeout=5.0)
            version_tuple = _parse_ollama_version(version_str)
            if version_tuple >= self._MIN_TOOLS_VERSION:
                payload["tools"] = tools
            else:
                logger.warning(
                    f"[OllamaProvider] Ollama 版本 {version_str or '未知'} "
                    f"不支持 tools（需要 >= 0.1.50），跳过 tools 传入"
                )

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
                return self._normalize_response(result)
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="无法连接到本地Ollama服务，请确保Ollama已启动 (localhost:11434)"
            )

    def _normalize_response(self, result: dict) -> dict:
        """将 Ollama 响应规范化为 OpenAI 风格 choices[0].message 结构"""
        msg = result.get("message", {})
        text_content = msg.get("content", "")
        tool_calls_raw = msg.get("tool_calls", [])

        tool_calls = []
        for tc in tool_calls_raw:
            fn = tc.get("function", {})
            tool_calls.append({
                "id": f"call_{fn.get('name', '')}",
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": json.dumps(fn.get("arguments", {}), ensure_ascii=False),
                },
            })

        message = {
            "role": "assistant",
            "content": text_content,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {"choices": [{"message": message}]}
