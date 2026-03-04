"""对话自动命名服务

参考 kotaemon 的 SuggestConvNamePipeline，根据对话历史生成简短中文会话名。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

CONV_NAME_PROMPT = (
    "根据以上对话内容，为这个对话建议一个简洁的中文名称（不超过10个字）。\n"
    "直接输出名称，不要加引号或其他解释。"
)


async def suggest_conversation_name(
    chat_history: list[dict],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
) -> Optional[str]:
    """根据对话历史生成会话名

    仅在首轮对话后调用（chat_history 含至少 1 轮问答）。

    Args:
        chat_history: [{"role": "user"|"assistant", "content": "..."}]
        api_key: LLM API 密钥
        model: LLM 模型
        provider: LLM 提供商
        endpoint: API 端点

    Returns:
        ≤10 字中文会话名，失败返回 None
    """
    if not chat_history or not api_key:
        return None

    try:
        from services.chat_service import call_ai_api

        messages = []
        for msg in chat_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": CONV_NAME_PROMPT})

        response = await call_ai_api(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=30,
            temperature=0.3,
        )

        content = ""
        if isinstance(response, dict):
            if response.get("error"):
                return None
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

        name = content.strip().strip('"\'""''')

        # 清除可能的 <think> 标签（推理模型）
        if "</think>" in name:
            name = name.split("</think>")[-1].strip()

        if name and len(name) <= 30:
            logger.info(f"[ConvName] 生成会话名: {name}")
            return name

        return None

    except Exception as e:
        logger.warning(f"[ConvName] 会话命名失败: {e}")
        return None
