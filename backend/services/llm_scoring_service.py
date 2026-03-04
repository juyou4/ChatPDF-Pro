"""LLM 精细相关性评分服务（TruLens 式）

参考 kotaemon 的 LLMTrulensScoring：
- 使用 LLM 对每个检索 chunk 进行 0-10 分相关性评分
- 并发评分所有 chunk
- 分数归一化到 0-1 区间
- 可选功能（成本高），适合精确场景
"""

import re
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

SCORING_SYSTEM_PROMPT = """你是一个专业的文档检索相关性评估专家。请根据以下标准对检索片段与用户问题的相关性进行 0-10 分评分：

评分标准：
- 10分：完全相关，直接回答了问题
- 8-9分：高度相关，包含回答问题所需的关键信息
- 6-7分：中等相关，包含部分有用信息
- 4-5分：低相关，仅有间接关联
- 2-3分：几乎无关，只有极少主题重叠
- 0-1分：完全无关

请只输出一个整数分数（0-10），不要输出任何解释。"""

SCORING_USER_PROMPT = """QUESTION: {question}

CONTEXT: {context}

RELEVANCE:"""

MAX_CONTEXT_LEN = 2000


def _extract_score(text: str) -> Optional[float]:
    """从 LLM 输出中提取整数分数

    处理各种输出格式：
    - "8"
    - "8/10"
    - "8 out of 10"
    - "评分：8"
    """
    matches = re.findall(r"\b(\d{1,2})\b", text.strip())
    if not matches:
        return None
    # 取所有匹配的最小值（保守评分），归一化到 0-1
    scores = [int(m) for m in matches if 0 <= int(m) <= 10]
    if not scores:
        return None
    return min(scores) / 10.0


async def score_single_chunk(
    question: str,
    chunk_text: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
) -> Optional[float]:
    """对单个 chunk 进行 LLM 相关性评分

    Args:
        question: 用户问题
        chunk_text: 检索到的文本块
        api_key: LLM API 密钥
        model: LLM 模型
        provider: LLM 提供商
        endpoint: API 端点

    Returns:
        0-1 归一化分数，失败返回 None
    """
    try:
        from services.chat_service import call_ai_api

        # 截断过长的 context
        truncated = chunk_text[:MAX_CONTEXT_LEN]

        response = await call_ai_api(
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": SCORING_USER_PROMPT.format(
                    question=question,
                    context=truncated,
                )},
            ],
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=10,
            temperature=0.0,
        )

        content = ""
        if isinstance(response, dict):
            if response.get("error"):
                return None
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

        return _extract_score(content)

    except Exception as e:
        logger.debug(f"[LLMScoring] 单 chunk 评分失败: {e}")
        return None


async def score_chunks(
    question: str,
    chunks: list[dict],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    max_concurrent: int = 5,
) -> list[dict]:
    """并发评分多个 chunk

    Args:
        question: 用户问题
        chunks: 检索结果列表 [{"text": "...", ...}, ...]
        api_key: LLM API 密钥
        model: LLM 模型
        provider: LLM 提供商
        endpoint: API 端点
        max_concurrent: 最大并发数

    Returns:
        增强后的 chunks 列表，每项新增 "llm_relevance_score" 字段
    """
    if not chunks or not api_key:
        return chunks

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _score_with_limit(chunk):
        async with semaphore:
            text = chunk.get("text", "") or chunk.get("chunk", "")
            if not text:
                return None
            return await score_single_chunk(
                question, text, api_key, model, provider, endpoint
            )

    tasks = [_score_with_limit(c) for c in chunks]
    scores = await asyncio.gather(*tasks, return_exceptions=True)

    for i, score in enumerate(scores):
        if isinstance(score, (int, float)) and score is not None:
            chunks[i]["llm_relevance_score"] = round(score, 2)
        else:
            chunks[i]["llm_relevance_score"] = None

    scored_count = sum(1 for s in scores if isinstance(s, (int, float)) and s is not None)
    logger.info(f"[LLMScoring] 评分完成: {scored_count}/{len(chunks)} 个 chunk")

    return chunks
