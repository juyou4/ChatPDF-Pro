"""动态追问建议服务

每轮回答后，基于最近对话历史用 LLM 生成 3-5 个后续追问建议。
参考 kotaemon 的 SuggestFollowupQuesPipeline 设计。
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

FOLLOWUP_PROMPT = """基于以下对话历史，生成 3 到 5 个用户可能想继续追问的后续问题。

要求：
- 问题应与对话内容相关，帮助用户深入理解文档
- 使用与用户相同的语言（中文）
- 问题应简洁、具体、有探索价值
- 直接输出 JSON 格式，不要加任何前缀或解释

输出格式示例：
{{"questions": ["问题1", "问题2", "问题3"]}}

对话历史：
{history}
"""


def _extract_json_payload(content: str) -> object:
    """从 LLM 文本中提取 JSON，兼容 markdown 代码块和前后解释文本。"""
    text = str(content or "").strip()
    if not text:
        return {}

    if "```" in text:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

    candidates = [text]
    for start_token, end_token in (("{", "}"), ("[", "]")):
        start = text.find(start_token)
        end = text.rfind(end_token)
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    return {}


def _normalize_questions_payload(parsed: object) -> list[str]:
    if isinstance(parsed, list):
        questions = parsed
    elif isinstance(parsed, dict):
        questions = (
            parsed.get("questions")
            or parsed.get("followup_questions")
            or parsed.get("followups")
            or parsed.get("items")
            or []
        )
    else:
        questions = []
    if not isinstance(questions, list):
        return []
    normalized = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
    return normalized[:5]


async def generate_followup_questions(
    chat_history: list[dict],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    max_rounds: int = 3,
) -> list[str]:
    """根据对话历史生成追问建议

    Args:
        chat_history: 对话历史 [{"role": "user"|"assistant", "content": "..."}]
        api_key: LLM API 密钥
        model: LLM 模型名称
        provider: LLM 提供商
        endpoint: LLM API 端点
        max_rounds: 使用最近几轮对话（每轮 = 一问一答）

    Returns:
        追问问题列表，失败时返回空列表
    """
    if not chat_history or not api_key:
        return []

    try:
        from services.chat_service import call_ai_api
        from services.completion_outcome import require_publishable_completion

        # 取最近 max_rounds 轮对话
        recent = chat_history[-(max_rounds * 2):]
        history_lines = []
        for msg in recent:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")[:300]
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        prompt = FOLLOWUP_PROMPT.format(history=history_text)

        response = await call_ai_api(
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=200,
            temperature=0.7,
        )
        require_publishable_completion(response, operation="follow-up questions")

        # 解析响应
        content = ""
        if isinstance(response, dict):
            if response.get("error"):
                logger.warning(f"[FollowupService] LLM 调用失败: {response['error']}")
                return []
            content = response.get("content", "")
            if not content and "choices" in response:
                choices = response["choices"]
                if choices and isinstance(choices, list):
                    content = choices[0].get("message", {}).get("content", "")
        else:
            content = str(response) if response else ""

        content = content.strip()
        if not content:
            return []

        parsed = _extract_json_payload(content)
        questions = _normalize_questions_payload(parsed)
        if questions:
            logger.info(f"[FollowupService] 生成 {len(questions)} 个追问建议")
            return questions

        return []

    except json.JSONDecodeError:
        logger.warning(f"[FollowupService] JSON 解析失败: {content[:200]}")
        return []
    except Exception as e:
        logger.warning(f"[FollowupService] 生成追问建议失败: {e}")
        return []
