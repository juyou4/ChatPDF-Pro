from typing import List, Optional
import asyncio
import logging

from services.embedding_service import (
    build_vector_index,
    search_document_chunks,
    get_relevant_context,
    normalize_embedding_model_id
)
from models.model_id_resolver import resolve_model_id, get_available_model_ids
from fastapi import HTTPException
from utils.middleware import BaseMiddleware, apply_middlewares_before, apply_middlewares_after
from services.embedding_service import get_embedding_function  # re-export if needed

logger = logging.getLogger(__name__)


def validate_embedding_model(embedding_model: str) -> str:
    """验证并归一化 embedding 模型 ID

    使用 Model_ID_Resolver 统一解析前端传入的模型 ID，
    支持 composite key（provider:modelId）和 plain key 两种格式。

    Args:
        embedding_model: 前端传入的模型 ID

    Returns:
        Model_Registry 中的键名

    Raises:
        HTTPException: 当模型 ID 无法解析时，抛出 HTTP 400 错误，
                       错误信息包含无效的模型 ID 和可用模型列表
    """
    # 使用 resolve_model_id() 统一解析模型 ID
    registry_key, config = resolve_model_id(embedding_model)
    if registry_key is not None:
        return registry_key

    # 解析失败，抛出 HTTP 400 错误，包含无效的模型 ID 和可用模型列表
    available_models = get_available_model_ids()
    raise HTTPException(
        status_code=400,
        detail=(
            f"Embedding模型 '{embedding_model}' 未配置或不受支持，"
            f"可用模型列表: {available_models}"
        )
    )


def create_index(doc_id: str, full_text: str, vector_store_dir: str, embedding_model: str, api_key: Optional[str], api_host: Optional[str], pages: Optional[list] = None):
    """Wrapper to build vector index with validation"""
    embedding_model = validate_embedding_model(embedding_model)
    build_vector_index(doc_id, full_text, vector_store_dir, embedding_model, api_key, api_host, pages=pages)


async def vector_search(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: Optional[str],
    top_k: int,
    candidate_k: int,
    use_rerank: bool,
    reranker_model: Optional[str],
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None,
    middlewares: Optional[List[BaseMiddleware]] = None
):
    """向量搜索包装函数，支持中间件钩子

    将同步的 search_document_chunks 放到线程池中执行，
    避免阻塞 FastAPI 异步事件循环（尤其是 rerank 操作）。
    设置 60 秒超时，防止无限等待。
    """
    payload = {
        "doc_id": doc_id,
        "query": query,
        "vector_store_dir": vector_store_dir,
        "pages": pages,
        "api_key": api_key,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "use_rerank": use_rerank,
        "reranker_model": reranker_model,
        "rerank_provider": rerank_provider,
        "rerank_api_key": rerank_api_key,
        "rerank_endpoint": rerank_endpoint
    }

    payload = await apply_middlewares_before(payload, middlewares or [])

    try:
        # 将同步搜索函数放到线程池中执行，避免阻塞事件循环
        results = await asyncio.wait_for(
            asyncio.to_thread(
                search_document_chunks,
                doc_id,
                query,
                vector_store_dir=vector_store_dir,
                pages=pages,
                api_key=api_key,
                top_k=top_k,
                candidate_k=candidate_k,
                use_rerank=use_rerank,
                reranker_model=reranker_model,
                rerank_provider=rerank_provider,
                rerank_api_key=rerank_api_key,
                rerank_endpoint=rerank_endpoint
            ),
            timeout=60.0  # 60 秒超时
        )
        wrapped = {"results": results}
    except asyncio.TimeoutError:
        logger.error(f"[vector_search] 搜索超时 (60s): doc_id={doc_id}, rerank={use_rerank}")
        wrapped = {"results": [], "error": "搜索超时，请稍后重试或关闭重排序功能"}
    except Exception as e:
        logger.error(f"[vector_search] 搜索失败: {e}", exc_info=True)
        wrapped = {"results": [], "error": str(e)}

    wrapped = await apply_middlewares_after(wrapped, middlewares or [])
    return wrapped.get("results", wrapped)


async def vector_context(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: Optional[str],
    top_k: int,
    candidate_k: int,
    use_rerank: bool,
    reranker_model: Optional[str],
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None,
    middlewares: Optional[List[BaseMiddleware]] = None,
    model_context_window: int = 0,
) -> dict:
    """获取相关上下文的包装函数，支持中间件钩子

    返回包含 context 和 retrieval_meta 的字典。
    get_relevant_context 现在返回 (context_string, retrieval_meta) 元组。

    Returns:
        包含 "context" 和 "retrieval_meta" 键的字典
    """
    payload = {
        "doc_id": doc_id,
        "query": query,
        "vector_store_dir": vector_store_dir,
        "pages": pages,
        "api_key": api_key,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "use_rerank": use_rerank,
        "reranker_model": reranker_model,
        "rerank_provider": rerank_provider,
        "rerank_api_key": rerank_api_key,
        "rerank_endpoint": rerank_endpoint
    }

    payload = await apply_middlewares_before(payload, middlewares or [])

    try:
        # 将同步函数放到线程池中执行，避免阻塞事件循环
        ctx, retrieval_meta = await asyncio.wait_for(
            asyncio.to_thread(
                get_relevant_context,
                doc_id,
                query,
                vector_store_dir=vector_store_dir,
                pages=pages,
                api_key=api_key,
                top_k=top_k,
                candidate_k=candidate_k,
                use_rerank=use_rerank,
                reranker_model=reranker_model,
                rerank_provider=rerank_provider,
                rerank_api_key=rerank_api_key,
                rerank_endpoint=rerank_endpoint,
                model_context_window=model_context_window
            ),
            timeout=60.0  # 60 秒超时
        )
        wrapped = {"context": ctx, "retrieval_meta": retrieval_meta}
    except asyncio.TimeoutError:
        logger.error(f"[vector_context] 上下文检索超时 (60s): doc_id={doc_id}, rerank={use_rerank}")
        wrapped = {"context": "", "retrieval_meta": {}, "error": "检索超时，请稍后重试或关闭重排序功能"}
    except Exception as e:
        logger.error(f"[vector_context] 上下文检索失败: {e}", exc_info=True)
        wrapped = {"context": "", "retrieval_meta": {}, "error": str(e)}

    wrapped = await apply_middlewares_after(wrapped, middlewares or [])
    return {
        "context": wrapped.get("context", ""),
        "retrieval_meta": wrapped.get("retrieval_meta", {}),
    }
