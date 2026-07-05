import json
import copy
import httpx
from fastapi import HTTPException
from typing import Dict, List, Optional

from .base import BaseProvider


def _strip_additional_properties(schema: dict) -> dict:
    """递归剥离 schema 中的 additionalProperties 字段（Gemini 不支持）"""
    if not isinstance(schema, dict):
        return schema
    cleaned = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_additional_properties(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _strip_additional_properties(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _convert_tools_to_gemini(tools: List[dict]) -> List[dict]:
    """将 OpenAI 格式的 tools 转换为 Gemini function_declarations 格式

    OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    Gemini: {"function_declarations": [{"name": ..., "description": ..., "parameters": ...}]}
    """
    declarations = []
    for tool in tools:
        fn = tool.get("function", {})
        # 深拷贝 parameters 并剥离 additionalProperties
        params = copy.deepcopy(fn.get("parameters", {"type": "object", "properties": {}}))
        params = _strip_additional_properties(params)
        declarations.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": params,
        })
    return [{"function_declarations": declarations}]


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
        tools: Optional[List[dict]] = None,
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
        # 传入 tools 时转换为 Gemini function_declarations 格式
        if tools:
            body["tools"] = _convert_tools_to_gemini(tools)
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

            try:
                result = response.json()
            except ValueError as exc:
                preview = response.text[:500].replace("\n", "\\n")
                raise RuntimeError(
                    f"Gemini API返回了无效JSON: {exc}; body_preview={preview}"
                ) from exc
            return self._normalize_response(result)

    def _normalize_response(self, result: dict) -> dict:
        """将 Gemini 响应规范化为 OpenAI 风格 choices[0].message 结构"""
        candidates = result.get("candidates", [])
        if not candidates:
            normalized = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
            if result.get("usageMetadata"):
                normalized["usage"] = result.get("usageMetadata")
            return normalized

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        text_content = ""
        tool_calls = []

        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                # 将 Gemini functionCall 转换为 OpenAI tool_calls 格式
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{fc.get('name', '')}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                    },
                })

        message = {
            "role": "assistant",
            "content": text_content,
        }
        # 仅当存在 tool_calls 时才添加该字段
        if tool_calls:
            message["tool_calls"] = tool_calls

        normalized = {"choices": [{"message": message}]}
        if result.get("usageMetadata"):
            normalized["usage"] = result.get("usageMetadata")
        return normalized
