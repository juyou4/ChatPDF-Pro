"""
查询简化服务 (P3.3b - 借鉴 RAGPaper.query_rewriter.simplify_query)

功能：移除查询中的冗余前缀和填充词，提取核心检索意图。
设计：
- 第一阶段：本地正则去除常见停用前缀（"请""能否""帮我"等）
- 第二阶段：可选 LLM 优化（默认走轻量本地版本，避免额外延迟）
- gate：原查询长度 > 50 字符时才触发

预期：提升 BM25/向量检索的关键词信噪比，减少召回噪音。
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# 常见停用前缀和冗余词（按出现顺序去除）
_STOP_PREFIX_PATTERNS = [
    r"^请[问您帮我]*\s*",
    r"^能否[请帮我]*\s*",
    r"^麻烦[请您]*\s*",
    r"^帮我\s*",
    r"^我[想要需希望]*\s*",
    r"^可以[请你您]*\s*",
    r"^你[能可以]+\s*",
    r"^请[你您]\s*",
    r"^如果[可以方便]*\s*",
]

# 冗余尾部词（删除时保留语义完整）
_STOP_SUFFIX_PATTERNS = [
    r"[，,]?\s*谢谢[啊呀！!]*\s*$",
    r"[，,]?\s*请[告诉]+我\s*$",
    r"[，,]?\s*好[吗么]\s*[？?]*\s*$",
]

# 冗余填充短语（替换为空格）
_FILLER_PATTERNS = [
    r"[，,]?\s*简单[来地]?[说讲]\s*[，,]?",
    r"[，,]?\s*具体[来地]?[说讲]\s*[，,]?",
    r"[，,]?\s*比较[详细全面]+地?\s*[，,]?",
]


def simplify_query_local(query: str) -> str:
    """本地正则版查询简化（无 LLM 调用，零延迟）

    Args:
        query: 原始查询

    Returns:
        简化后的查询
    """
    if not query or not query.strip():
        return query

    simplified = query.strip()

    for pattern in _STOP_PREFIX_PATTERNS:
        simplified = re.sub(pattern, "", simplified, count=1)

    for pattern in _STOP_SUFFIX_PATTERNS:
        simplified = re.sub(pattern, "", simplified, count=1)

    for pattern in _FILLER_PATTERNS:
        simplified = re.sub(pattern, " ", simplified)

    simplified = re.sub(r"\s+", " ", simplified).strip()

    return simplified or query


_LLM_SIMPLIFY_PROMPT = """请将以下用户查询简化为最核心的检索意图，**只保留关键信息词**。

要求：
- 移除"请""能否""帮我""我想"等填充词
- 移除"具体说""简单讲"等无信息冗余
- 保留所有专有名词、数字、方法名、术语
- 输出一句简洁的查询，不超过原查询长度的 70%
- 直接输出简化后的查询，不要解释

原始查询：{query}

简化后："""


async def simplify_query_llm(
    query: str,
    api_key: str,
    model: str = "",
    provider: str = "",
    endpoint: str = "",
) -> Optional[str]:
    """LLM 版查询简化（更智能但增加延迟）

    Args:
        query: 原始查询
        api_key: LLM API 密钥
        model: LLM 模型名称
        provider: LLM 提供商
        endpoint: LLM API 端点

    Returns:
        简化后的查询，失败返回 None
    """
    if not query or not query.strip():
        return None

    try:
        from services.chat_service import call_ai_api
        from services.completion_outcome import require_publishable_completion
        from services.intent_constraints import IntentConstraintSet
        from models.provider_registry import PROVIDER_CONFIG

        if not model:
            model = "gpt-4o-mini"
        if not provider:
            provider = "openai"
        if not endpoint:
            endpoint = PROVIDER_CONFIG.get(provider, {}).get("endpoint", "")

        messages = [
            {"role": "user", "content": _LLM_SIMPLIFY_PROMPT.format(query=query)}
        ]

        response = await call_ai_api(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=120,
            temperature=0.0,
        )
        require_publishable_completion(response, operation="query simplification")

        content = ""
        if isinstance(response, dict):
            if response.get("error"):
                logger.warning(f"[QuerySimplify] LLM 调用失败: {response['error']}")
                return None
            content = response.get("content", "")
            if not content and "choices" in response:
                choices = response["choices"]
                if choices and isinstance(choices, list):
                    content = choices[0].get("message", {}).get("content", "")

        simplified = (content or "").strip().splitlines()[0] if content else ""
        simplified = re.sub(r"^[\d\.\-\)、]+\s*", "", simplified).strip()
        simplified = simplified.strip("\"'`")

        if not simplified or len(simplified) < 3:
            return None
        if len(simplified) > len(query):
            return None
        if not IntentConstraintSet.from_text(query).validate_rewrite(simplified).allowed:
            return None

        logger.info(f"[QuerySimplify] LLM 简化: '{query[:40]}' → '{simplified[:40]}'")
        return simplified

    except Exception as e:
        logger.warning(f"[QuerySimplify] LLM 简化失败: {e}")
        return None


async def simplify_query(
    query: str,
    *,
    min_chars: int = 50,
    use_llm: bool = False,
    api_key: Optional[str] = None,
    model: str = "",
    provider: str = "",
    endpoint: str = "",
) -> str:
    """统一入口：根据 gate 决定是否调用 LLM 简化

    Args:
        query: 原始查询
        min_chars: 触发简化的最小字符数（小于此值跳过）
        use_llm: 是否使用 LLM 简化（False 则只用本地正则）
        api_key/model/provider/endpoint: LLM 配置（use_llm=True 时必需）

    Returns:
        简化后的查询，失败/不触发时返回原始查询
    """
    if not query:
        return query

    if len(query) < min_chars:
        return query

    if use_llm and api_key:
        llm_result = await simplify_query_llm(
            query, api_key, model=model, provider=provider, endpoint=endpoint
        )
        if llm_result:
            return llm_result

    local_result = simplify_query_local(query)
    if local_result and local_result != query:
        logger.info(f"[QuerySimplify] 本地简化: '{query[:40]}' → '{local_result[:40]}'")
    return local_result or query
