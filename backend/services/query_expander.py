"""
查询增强服务 — HyDE 假设文档嵌入 + 多查询扩展

HyDE (Hypothetical Document Embeddings):
  用 LLM 生成一段假设性答案，用答案的 embedding 替代原始查询做检索，
  弥合"问题"与"陈述"之间的语义鸿沟。

Multi-Query Expansion:
  用 LLM 将原始查询改写为多个不同角度的子查询，分别检索后合并，
  提高召回率。
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# HyDE — Hypothetical Document Embeddings
# ============================================================

_HYDE_PROMPT = """请根据以下问题，写一段 150-250 字的中文段落，作为该问题的假设性答案。
要求：
- 直接以陈述句的形式回答，不要出现"根据文档"等措辞
- 内容应尽可能具体、学术化，像论文/教材中的一段文字
- 不要加前缀、编号或标题，直接输出段落

问题：{query}

假设性答案："""


async def generate_hyde_passage(
    query: str,
    api_key: str,
    model: str = "",
    provider: str = "",
    endpoint: str = "",
) -> Optional[str]:
    """用 LLM 生成 HyDE 假设文档段落

    Args:
        query: 用户查询
        api_key: LLM API 密钥
        model: LLM 模型名称
        provider: LLM 提供商
        endpoint: LLM API 端点

    Returns:
        假设文档段落文本，失败返回 None
    """
    if not query or not query.strip():
        return None

    try:
        from services.chat_service import call_ai_api
        from models.provider_registry import PROVIDER_CONFIG

        if not model:
            model = "gpt-4o-mini"
        if not provider:
            provider = "openai"
        if not endpoint:
            endpoint = PROVIDER_CONFIG.get(provider, {}).get("endpoint", "")

        messages = [
            {"role": "user", "content": _HYDE_PROMPT.format(query=query)}
        ]

        response = await call_ai_api(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=300,
            temperature=0.5,
        )

        if isinstance(response, dict):
            if response.get("error"):
                logger.warning(f"[HyDE] LLM 调用失败: {response['error']}")
                return None
            content = response.get("content", "")
            if not content and "choices" in response:
                choices = response["choices"]
                if choices and isinstance(choices, list):
                    content = choices[0].get("message", {}).get("content", "")
        else:
            content = str(response) if response else ""

        if content and len(content.strip()) > 20:
            logger.info(f"[HyDE] 生成假设文档段落: {len(content)} 字符")
            return content.strip()

        logger.warning("[HyDE] LLM 返回内容过短，跳过")
        return None

    except Exception as e:
        logger.warning(f"[HyDE] 假设文档生成失败，降级为原始查询: {e}")
        return None


# ============================================================
# Multi-Query Expansion — 多查询扩展
# ============================================================

_EXPANSION_PROMPT = """请将以下用户问题改写为 {n} 个不同角度的检索查询。
每个查询应从不同侧面表达同一信息需求，使用不同的关键词和表述方式。
每行输出一个查询，不要加编号或前缀。

原始问题：{query}

改写查询："""


async def expand_query(
    query: str,
    api_key: str,
    n: int = 3,
    model: str = "",
    provider: str = "",
    endpoint: str = "",
) -> List[str]:
    """用 LLM 将查询扩展为多个改写版本

    Args:
        query: 原始查询
        api_key: LLM API 密钥
        n: 扩展查询数量
        model: LLM 模型名称
        provider: LLM 提供商
        endpoint: LLM API 端点

    Returns:
        改写查询列表（不含原始查询），失败返回空列表
    """
    if not query or not query.strip():
        return []

    try:
        from services.chat_service import call_ai_api
        from models.provider_registry import PROVIDER_CONFIG

        if not model:
            model = "gpt-4o-mini"
        if not provider:
            provider = "openai"
        if not endpoint:
            endpoint = PROVIDER_CONFIG.get(provider, {}).get("endpoint", "")

        messages = [
            {"role": "user", "content": _EXPANSION_PROMPT.format(query=query, n=n)}
        ]

        response = await call_ai_api(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=200,
            temperature=0.7,
        )

        if isinstance(response, dict):
            if response.get("error"):
                logger.warning(f"[QueryExpansion] LLM 调用失败: {response['error']}")
                return []
            content = response.get("content", "")
            if not content and "choices" in response:
                choices = response["choices"]
                if choices and isinstance(choices, list):
                    content = choices[0].get("message", {}).get("content", "")
        else:
            content = str(response) if response else ""

        if not content or not content.strip():
            return []

        # 解析：每行一个查询，去除编号前缀
        lines = content.strip().split("\n")
        expanded = []
        for line in lines:
            line = line.strip()
            # 去除编号前缀（"1. " "- " "1) "）
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-•]\s*", "", line)
            line = line.strip()
            if line and line != query and len(line) > 3:
                expanded.append(line)

        expanded = expanded[:n]
        if expanded:
            logger.info(
                f"[QueryExpansion] 查询扩展: '{query}' → {len(expanded)} 个改写"
            )
        return expanded

    except Exception as e:
        logger.warning(f"[QueryExpansion] 查询扩展失败: {e}")
        return []
