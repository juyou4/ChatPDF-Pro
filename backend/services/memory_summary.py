"""滚动会话摘要

working 层只保留最近若干轮原文，再往前就只剩零散的 distilled 事实——
"本次会话在聊什么"这条叙事线是断的，长会话里用户说"刚才那个公式"容易接不上。

这里维护每个文档一条滚动摘要：每积累若干条消息，用
「旧摘要 + 新对话 → 合并重写」的方式更新一次，成本是 O(1) 而不是 O(会话长度)。

摘要以 MemoryEntry 形式存储（memory_kind="session_summary"），
从而复用既有的事件日志、审计、索引与解析代际校验——
如果塞进 session 字典的自定义字段，事件回放时会被静默丢弃。
"""
import logging
from typing import Optional

from services.memory_llm import call_llm_sync, strip_reasoning_and_fences

logger = logging.getLogger(__name__)

SESSION_SUMMARY_SOURCE_TYPE = "session_summary"
SESSION_SUMMARY_KIND = "session_summary"

# 摘要正文长度上限（字符），防止滚动过程中无限膨胀
DEFAULT_SUMMARY_MAX_CHARS = 600

_SUMMARY_PROMPT = """你在维护一段论文阅读对话的"会话摘要"，供后续回答快速接上下文。

给你上一版摘要和新发生的对话，请输出一份**更新后的完整摘要**（不是增量）。

保留这些内容：
- 用户当前在追问的主线问题
- 已经确认的关键结论与数值（数字、单位、专有名词保持原样）
- 用户明确提过的要求与偏好
- 尚未解决的问题

丢弃这些内容：
- 寒暄、致谢、重复确认
- 已被后续对话推翻的旧结论

要求：
- 用中文连贯叙述，不要分点、不要标题
- 不超过 {max_chars} 字
- 只输出摘要正文本身，不要任何前后缀"""


def _format_rounds(messages: list[dict], max_chars_per_message: int = 300) -> str:
    lines = []
    for msg in messages or []:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        speaker = "用户" if role == "user" else "助手"
        lines.append(f"{speaker}：{content[:max_chars_per_message]}")
    return "\n".join(lines)


def build_rolling_summary(
    previous_summary: str,
    new_messages: list[dict],
    *,
    api_key: str,
    model: str,
    provider: str,
    max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
) -> Optional[str]:
    """把旧摘要与新对话合并为一版新摘要；失败返回 None 让调用方保留旧摘要。"""
    conversation = _format_rounds(new_messages)
    if not conversation:
        return None

    previous = str(previous_summary or "").strip()
    user_content = (
        f"上一版摘要：\n{previous or '（暂无，这是第一版）'}\n\n新发生的对话：\n{conversation}"
    )
    messages = [
        {"role": "system", "content": _SUMMARY_PROMPT.format(max_chars=max_chars)},
        {"role": "user", "content": user_content},
    ]

    try:
        response = call_llm_sync(
            messages,
            api_key=api_key,
            model=model,
            provider=provider,
            max_tokens=max(300, max_chars),
        )
    except Exception as exc:
        logger.warning(f"[SessionSummary] 调用失败，保留旧摘要: {exc}")
        return None

    text = strip_reasoning_and_fences(response or "").strip()
    if not text:
        return None
    # 模型偶尔会无视长度要求，这里硬截断兜底
    if len(text) > max_chars * 2:
        text = text[: max_chars * 2].rstrip() + "..."
    return text
