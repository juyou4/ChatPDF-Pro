"""复杂问题分解服务

参考 kotaemon 的 DecomposeQuestionPipeline，使用 LLM 将复杂问题
拆分为最多 3 个子问题，每个子问题独立检索后合并结果。
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """你是一个问题分析专家。请判断以下用户问题是否需要拆分为子问题来回答。

如果问题涉及多个概念的比较、多个方面的分析、或者包含"和""与""比较""区别""异同""优缺点"等关键词，
请将其拆分为最多 3 个具体的子问题，每个子问题只关注一个概念或事实。

如果问题足够简单，不需要拆分，请返回空列表。

直接输出 JSON 格式，不要加任何前缀或解释。

输出格式：
{"needs_decompose": true, "sub_questions": ["子问题1", "子问题2", "子问题3"]}
或
{"needs_decompose": false, "sub_questions": []}

用户问题：{question}
"""

# 触发分解的关键词模式
_DECOMPOSE_TRIGGERS = re.compile(
    r"(比较|区别|异同|不同|优缺点|对比|差异|相同|各自|分别|以及|和.*的关系|与.*的区别)",
    re.IGNORECASE,
)


def should_decompose(question: str) -> bool:
    """快速判断是否值得尝试问题分解

    Args:
        question: 用户原始问题

    Returns:
        True 如果问题可能需要分解
    """
    if len(question) < 10:
        return False
    return bool(_DECOMPOSE_TRIGGERS.search(question))


async def decompose_question(
    question: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
) -> list[str]:
    """使用 LLM 将复杂问题分解为子问题

    Args:
        question: 用户原始问题
        api_key: LLM API 密钥
        model: LLM 模型
        provider: LLM 提供商
        endpoint: API 端点

    Returns:
        子问题列表，不需要分解时返回空列表
    """
    if not should_decompose(question):
        return []

    if not api_key:
        return []

    try:
        from services.chat_service import call_ai_api

        prompt = DECOMPOSE_PROMPT.format(question=question)

        response = await call_ai_api(
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=200,
            temperature=0.3,
        )

        content = ""
        if isinstance(response, dict):
            if response.get("error"):
                return []
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

        content = content.strip()
        if not content:
            return []

        # 处理可能的 markdown 代码块
        if "```" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]

        parsed = json.loads(content)
        if not parsed.get("needs_decompose"):
            return []

        subs = parsed.get("sub_questions", [])
        if isinstance(subs, list) and all(isinstance(q, str) for q in subs):
            logger.info(f"[Decompose] 问题分解为 {len(subs)} 个子问题: {subs}")
            return subs[:3]

        return []

    except json.JSONDecodeError:
        logger.debug(f"[Decompose] JSON 解析失败: {content[:200]}")
        return []
    except Exception as e:
        logger.warning(f"[Decompose] 问题分解失败: {e}")
        return []
