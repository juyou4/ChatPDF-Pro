"""
上下文压缩服务 — Contextual Compression

对检索到的 chunk 使用 LLM 提取仅与查询相关的句子，
减少发送给最终 LLM 的噪声，提升回答精度。

使用并发处理加速多 chunk 压缩。
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

_COMPRESS_PROMPT = """请从以下文档片段中提取与用户问题直接相关的内容。
要求：
- 只保留与问题相关的句子和段落
- 保持原文措辞不变，不要改写或总结
- 如果整段都相关，保留全文
- 如果整段都不相关，返回"[无关内容]"
- 直接输出提取的内容，不要加前缀或解释

用户问题：{query}

文档片段：
{chunk_text}

相关内容："""


async def compress_chunk(
    chunk_text: str,
    query: str,
    api_key: str,
    model: str = "",
    provider: str = "",
    endpoint: str = "",
) -> str:
    """用 LLM 压缩单个 chunk，仅保留与查询相关的内容

    Args:
        chunk_text: 原始 chunk 文本
        query: 用户查询
        api_key: LLM API 密钥
        model: LLM 模型
        provider: LLM 提供商
        endpoint: LLM API 端点

    Returns:
        压缩后的文本，失败时返回原文
    """
    if not chunk_text or not query:
        return chunk_text

    # 短 chunk 不压缩
    if len(chunk_text) < 200:
        return chunk_text

    try:
        from services.chat_service import call_ai_api
        from models.provider_registry import PROVIDER_CONFIG

        if not model:
            model = "gpt-4o-mini"
        if not provider:
            provider = "openai"
        if not endpoint:
            endpoint = PROVIDER_CONFIG.get(provider, {}).get("endpoint", "")

        response = await call_ai_api(
            messages=[
                {"role": "user", "content": _COMPRESS_PROMPT.format(
                    query=query, chunk_text=chunk_text[:3000]
                )}
            ],
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=500,
            temperature=0.1,
        )

        if isinstance(response, dict):
            if response.get("error"):
                logger.warning(f"[ContextCompress] LLM 调用失败: {response['error']}")
                return chunk_text
            content = response.get("content", "")
            if not content and "choices" in response:
                choices = response["choices"]
                if choices and isinstance(choices, list):
                    content = choices[0].get("message", {}).get("content", "")
        else:
            content = str(response) if response else ""

        content = content.strip()

        # 如果 LLM 判断无关，返回空
        if content == "[无关内容]" or not content:
            logger.info(f"[ContextCompress] chunk 被判定为无关，长度 {len(chunk_text)}")
            return ""

        # 压缩成功
        compression_ratio = len(content) / max(len(chunk_text), 1)
        if compression_ratio < 0.95:
            logger.info(
                f"[ContextCompress] 压缩: {len(chunk_text)} → {len(content)} "
                f"(比例 {compression_ratio:.1%})"
            )
        return content

    except Exception as e:
        logger.warning(f"[ContextCompress] 压缩失败，使用原文: {e}")
        return chunk_text


async def compress_results(
    results: List[dict],
    query: str,
    api_key: str,
    max_concurrent: int = 5,
    model: str = "",
    provider: str = "",
    endpoint: str = "",
) -> List[dict]:
    """并发压缩多个检索结果

    Args:
        results: search_document_chunks 返回的结果列表
        query: 用户查询
        api_key: LLM API 密钥
        max_concurrent: 最大并发数
        model: LLM 模型
        provider: LLM 提供商
        endpoint: LLM API 端点

    Returns:
        压缩后的结果列表（过滤掉被判定为无关的 chunk）
    """
    if not results or not api_key:
        return results

    try:
        from utils.concurrency import run_with_concurrency

        async def _compress_one(item: dict) -> Optional[dict]:
            chunk_text = item.get("chunk", "")
            if not chunk_text:
                return item

            compressed = await compress_chunk(
                chunk_text, query, api_key,
                model=model, provider=provider, endpoint=endpoint,
            )
            if not compressed:
                return None  # 被判定为无关

            new_item = item.copy()
            new_item["chunk"] = compressed
            new_item["compressed"] = True
            return new_item

        tasks = [lambda item=item: _compress_one(item) for item in results]
        compressed_results = await run_with_concurrency(tasks, max_concurrent=max_concurrent)

        # 过滤掉 None（无关 chunk）
        filtered = [r for r in compressed_results if r is not None]

        logger.info(
            f"[ContextCompress] 批量压缩完成: "
            f"{len(results)} → {len(filtered)} 个结果"
        )
        return filtered

    except Exception as e:
        logger.warning(f"[ContextCompress] 批量压缩失败，使用原始结果: {e}")
        return results
