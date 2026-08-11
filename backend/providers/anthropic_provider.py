import json
import httpx
from fastapi import HTTPException
from typing import Dict, List, Optional

from .base import BaseProvider
from services.provider_auth import build_api_key_headers


def _convert_tools_to_anthropic(tools: List[dict]) -> List[dict]:
    """将 OpenAI 格式的 tools 转换为 Anthropic Claude 格式

    OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    Anthropic: {"name": ..., "description": ..., "input_schema": ...}
    """
    converted = []
    for tool in tools:
        fn = tool.get("function", {})
        converted.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


class AnthropicProvider(BaseProvider):
    """Anthropic Claude Provider"""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key_header: Optional[str] = None,
        api_key_prefix: Optional[str] = None,
    ):
        # 官方 Provider 使用固定默认地址；动态 Provider 可以传入自定义
        # Anthropic Messages 网关，但仍复用同一套原生请求格式。
        self.endpoint = endpoint or "https://api.anthropic.com/v1/messages"
        self.api_key_header = api_key_header
        self.api_key_prefix = api_key_prefix

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
        # 传入 tools 时转换为 Anthropic 格式
        if tools:
            body["tools"] = _convert_tools_to_anthropic(tools)
        # 合并自定义参数
        if custom_params:
            body.update(custom_params)

        headers = build_api_key_headers(
            api_key,
            provider_type="anthropic",
            api_key_header=self.api_key_header,
            api_key_prefix=self.api_key_prefix,
            extra_headers={"anthropic-version": "2023-06-01"},
        )
        async with httpx.AsyncClient(timeout=timeout or 120.0) as client:
            response = await client.post(
                self.endpoint,
                headers=headers,
                json=body,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Anthropic API错误: {response.text}"
                )

            try:
                result = response.json()
            except ValueError as exc:
                preview = response.text[:500].replace("\n", "\\n")
                raise RuntimeError(
                    f"Anthropic API返回了无效JSON: {exc}; body_preview={preview}"
                ) from exc
            return self._normalize_response(result)

    def _normalize_response(self, result: dict) -> dict:
        """将 Anthropic 响应规范化为 OpenAI 风格 choices[0].message 结构"""
        content_blocks = result.get("content", [])
        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                # 将 Anthropic tool_use 块转换为 OpenAI tool_calls 格式
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                })

        message = {
            "role": "assistant",
            "content": text_content,
        }
        # 仅当存在 tool_calls 时才添加该字段
        if tool_calls:
            message["tool_calls"] = tool_calls

        choice = {
            "message": message,
            "finish_reason": str(result.get("stop_reason") or ""),
        }
        normalized = {"choices": [choice]}
        if result.get("usage"):
            normalized["usage"] = result.get("usage")
        return normalized
