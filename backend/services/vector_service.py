from typing import Any, Dict, List, Optional, Union
import asyncio
import logging

from services.embedding_service import (
    build_vector_index as _build_vector_index_impl,
    search_document_chunks as _search_document_chunks_impl,
    get_relevant_context as _get_relevant_context_impl,
)
from models.model_id_resolver import resolve_model_id, get_available_model_ids
from fastapi import HTTPException
from utils.middleware import BaseMiddleware, apply_middlewares_before, apply_middlewares_after
from services.embedding_service import get_embedding_function  # re-export if needed

logger = logging.getLogger(__name__)


def _resolve_embedding_request_args(
    embedding_model: str = "",
    embedding_provider: str = "",
    embedding_api_host: str = "",
    provider: str = "",
    api_host: str = "",
) -> tuple[str, str, str]:
    # ``provider`` / ``api_host`` belong to legacy generic callers. Embedding
    # identity must come from the dedicated fields so chat transport settings
    # can never be reinterpreted as vector credentials.
    return (
        str(embedding_model or "").strip(),
        str(embedding_provider or "").strip(),
        str(embedding_api_host or "").strip(),
    )


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

    # composite key（如 silicon:NewModel）允许直通，
    # 下游 get_embedding_function 的 fallback 逻辑能根据 provider 推断 base_url
    if ":" in embedding_model:
        logger.warning(
            f"模型 '{embedding_model}' 未在注册表中找到，"
            f"将使用 composite key fallback 继续"
        )
        return embedding_model

    # plain key 无法 fallback，抛出 HTTP 400 错误
    available_models = get_available_model_ids()
    raise HTTPException(
        status_code=400,
        detail=(
            f"Embedding模型 '{embedding_model}' 未配置或不受支持，"
            f"可用模型列表: {available_models}"
        )
    )


def create_index(
    doc_id: str,
    full_text: str,
    vector_store_dir: str,
    embedding_model: str,
    api_key: Optional[str],
    api_host: Optional[str],
    pages: Optional[list] = None,
    evidence_chunks: Optional[list] = None,
    structured_table_bundles: Optional[list] = None,
    summary_api_key: Optional[str] = None,
    summary_model: str = "gpt-4o-mini",
    summary_provider: str = "openai",
    summary_api_host: str = "",
    index_source: str = "pdf_native",
    index_meta: Optional[dict] = None,
    build_semantic_groups: bool = True,
    embedding_provider: str = "",
    embedding_api_host: str = "",
    provider: str = "",
):
    """Wrapper to build vector index with validation"""
    embedding_model, embedding_provider, embedding_api_host = _resolve_embedding_request_args(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
        provider=provider,
        api_host=api_host or "",
    )
    return build_vector_index(
        doc_id,
        full_text,
        vector_store_dir,
        embedding_model,
        api_key,
        embedding_api_host,
        pages=pages,
        evidence_chunks=evidence_chunks,
        structured_table_bundles=structured_table_bundles,
        summary_api_key=summary_api_key,
        summary_model=summary_model,
        summary_provider=summary_provider,
        summary_api_host=summary_api_host,
        index_source=index_source,
        index_meta=index_meta,
        build_semantic_groups=build_semantic_groups,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
        provider=provider,
    )


def build_vector_index(
    doc_id: str,
    text: str,
    vector_store_dir: str,
    embedding_model_id: str = "local-minilm",
    api_key: str = None,
    api_host: str = None,
    pages: Optional[list] = None,
    evidence_chunks: Optional[list] = None,
    structured_table_bundles: Optional[list] = None,
    summary_api_key: Optional[str] = None,
    summary_model: str = "gpt-4o-mini",
    summary_provider: str = "openai",
    summary_api_host: str = "",
    index_source: str = "pdf_native",
    index_meta: Optional[dict] = None,
    build_semantic_groups: bool = True,
    embedding_provider: str = "",
    embedding_api_host: str = "",
    provider: str = "",
):
    """兼容旧调用并向核心实现透传 embedding 身份参数。"""
    embedding_model_id, embedding_provider, embedding_api_host = _resolve_embedding_request_args(
        embedding_model=embedding_model_id,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
        provider=provider,
        api_host=api_host or "",
    )
    normalized_model = validate_embedding_model(embedding_model_id)
    return _build_vector_index_impl(
        doc_id,
        text,
        vector_store_dir,
        normalized_model,
        api_key,
        embedding_api_host,
        pages=pages,
        evidence_chunks=evidence_chunks,
        structured_table_bundles=structured_table_bundles,
        summary_api_key=summary_api_key,
        summary_model=summary_model,
        summary_provider=summary_provider,
        summary_api_host=summary_api_host,
        index_source=index_source,
        index_meta=index_meta,
        build_semantic_groups=build_semantic_groups,
        embedding_provider=embedding_provider or None,
    )


def search_document_chunks(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: Optional[str] = None,
    top_k: int = 10,
    candidate_k: int = 20,
    use_rerank: bool = False,
    reranker_model: Optional[str] = None,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None,
    use_hybrid: bool = True,
    selected_text: Optional[str] = None,
    progress_callback=None,
    enable_query_expansion_override: Optional[bool] = None,
    query_expansion_api_key: Optional[str] = None,
    query_expansion_model: str = "",
    query_expansion_provider: str = "",
    query_expansion_endpoint: str = "",
    visual_evidence: Optional[List[dict]] = None,
    intent_decision: Optional[dict] = None,
    query_is_canonical: bool = False,
    embedding_model: str = "",
    embedding_provider: str = "",
    embedding_api_host: str = "",
    provider: str = "",
    api_host: str = "",
):
    """兼容旧测试替身，并把 embedding 身份参数透传给核心实现。"""
    embedding_model, embedding_provider, embedding_api_host = _resolve_embedding_request_args(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
        provider=provider,
        api_host=api_host,
    )
    return _search_document_chunks_impl(
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
        use_hybrid=use_hybrid,
        selected_text=selected_text,
        progress_callback=progress_callback,
        enable_query_expansion_override=enable_query_expansion_override,
        query_expansion_api_key=query_expansion_api_key,
        query_expansion_model=query_expansion_model,
        query_expansion_provider=query_expansion_provider,
        query_expansion_endpoint=query_expansion_endpoint,
        visual_evidence=visual_evidence,
        intent_decision=intent_decision,
        query_is_canonical=query_is_canonical,
        embedding_model=embedding_model or None,
        embedding_provider=embedding_provider or None,
        embedding_api_host=embedding_api_host or None,
    )


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
    visual_evidence: Optional[List[dict]] = None,
    middlewares: Optional[List[BaseMiddleware]] = None,
    progress_callback=None,
    return_status: bool = False,
    intent_decision: Optional[dict] = None,
    query_is_canonical: bool = False,
    embedding_model: str = "",
    embedding_provider: str = "",
    embedding_api_host: str = "",
    provider: str = "",
    api_host: str = "",
) -> Union[List[dict], Dict[str, Any]]:
    """向量搜索包装函数，支持中间件钩子

    将同步的 search_document_chunks 放到线程池中执行，
    避免阻塞 FastAPI 异步事件循环（尤其是 rerank 操作）。
    设置 60 秒超时，防止无限等待。

    return_status 默认为 False，保留历史调用只接收结果列表的契约。
    搜索 API 应传入 True，从而区分正常零命中和检索失败。
    """
    embedding_model, embedding_provider, embedding_api_host = _resolve_embedding_request_args(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
        provider=provider,
        api_host=api_host,
    )
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
        "rerank_endpoint": rerank_endpoint,
        "visual_evidence": visual_evidence,
        "intent_decision": intent_decision,
        "query_is_canonical": bool(query_is_canonical),
        "embedding_model": embedding_model,
        "embedding_provider": embedding_provider,
        "embedding_api_host": embedding_api_host,
        "provider": embedding_provider,
        "api_host": embedding_api_host,
    }

    payload = await apply_middlewares_before(payload, middlewares or [])

    retrieval_degrade = {"reason": None}

    def _capture_retrieval_progress(event: dict) -> None:
        if isinstance(event, dict) and event.get("phase") == "vector_search_fallback":
            retrieval_degrade["reason"] = "vector_search_fallback"
        if progress_callback:
            progress_callback(event)

    try:
        # 将同步搜索函数放到线程池中执行，避免阻塞事件循环
        results, timings = await asyncio.wait_for(
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
                rerank_endpoint=rerank_endpoint,
                visual_evidence=visual_evidence,
                progress_callback=_capture_retrieval_progress,
                intent_decision=intent_decision,
                query_is_canonical=query_is_canonical,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                embedding_api_host=embedding_api_host,
            ),
            timeout=60.0  # 60 秒超时
        )
        wrapped = {
            "results": results,
            "timings": timings,
            "error": (
                "向量召回失败，已降级为关键词检索"
                if retrieval_degrade["reason"]
                else None
            ),
            "error_code": retrieval_degrade["reason"],
            "degraded": bool(retrieval_degrade["reason"]),
            "fallback_reason": retrieval_degrade["reason"],
            "fallback_used": bool(retrieval_degrade["reason"]),
        }
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error(f"[vector_search] 搜索超时 (60s): doc_id={doc_id}, rerank={use_rerank}")
        wrapped = {
            "results": [],
            "timings": {},
            "error": "搜索超时，请稍后重试或关闭重排序功能",
            "error_code": "vector_search_timeout",
            "degraded": True,
            "fallback_reason": "vector_search_timeout",
            "fallback_used": False,
        }
    except Exception as e:
        logger.error(f"[vector_search] 搜索失败: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            error_message = str(e.detail or "向量检索失败")
            error_code = "vector_index_unavailable" if e.status_code == 404 else "vector_search_failed"
        else:
            # 原始异常可能包含本机路径或供应商细节，仅写日志，不直接暴露给客户端。
            error_message = "向量检索失败，请稍后重试或重新构建文档索引"
            error_code = "vector_search_failed"
        wrapped = {
            "results": [],
            "timings": {},
            "error": error_message,
            "error_code": error_code,
            "degraded": True,
            "fallback_reason": error_code,
            "fallback_used": False,
        }

    wrapped = await apply_middlewares_after(wrapped, middlewares or [])
    if not isinstance(wrapped, dict):
        wrapped = {
            "results": [],
            "timings": {},
            "error": "检索中间件返回了无效响应",
            "error_code": "invalid_search_response",
            "degraded": True,
            "fallback_reason": "invalid_search_response",
            "fallback_used": False,
        }

    normalized_results = wrapped.get("results")
    if not isinstance(normalized_results, list):
        normalized_results = []
        wrapped["error"] = wrapped.get("error") or "检索服务返回了无效结果"
        wrapped["error_code"] = wrapped.get("error_code") or "invalid_search_response"
        wrapped["degraded"] = True
        wrapped["fallback_reason"] = (
            wrapped.get("fallback_reason") or wrapped["error_code"]
        )
    wrapped["results"] = normalized_results
    wrapped["degraded"] = bool(wrapped.get("degraded") or wrapped.get("error"))
    if wrapped["degraded"] and not wrapped.get("fallback_reason"):
        wrapped["fallback_reason"] = wrapped.get("error_code") or "vector_search_failed"
    wrapped.setdefault("fallback_used", False)
    wrapped.setdefault("timings", {})

    return wrapped if return_status else normalized_results


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
    visual_evidence: Optional[List[dict]] = None,
    middlewares: Optional[List[BaseMiddleware]] = None,
    model_context_window: int = 0,
    selected_text: Optional[str] = None,  # 框选文本，用于融合检索
    answer_max_tokens: int = 0,  # 期望的输出 Token 数，传入 RAG 预算感知
    progress_callback=None,
    query_expansion_api_key: Optional[str] = None,
    query_expansion_model: str = "",
    query_expansion_provider: str = "",
    query_expansion_endpoint: str = "",
    intent_decision: Optional[dict] = None,
    query_is_canonical: bool = False,
    embedding_model: str = "",
    embedding_provider: str = "",
    embedding_api_host: str = "",
    provider: str = "",
    api_host: str = "",
    retrieval_identity: Optional[dict] = None,
) -> dict:
    """获取相关上下文的包装函数，支持中间件钩子

    返回包含 context 和 retrieval_meta 的字典。
    get_relevant_context 现在返回 (context_string, retrieval_meta) 元组。

    Returns:
        包含 "context" 和 "retrieval_meta" 键的字典
    """
    embedding_model, embedding_provider, embedding_api_host = _resolve_embedding_request_args(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
        provider=provider,
        api_host=api_host,
    )
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
        "rerank_endpoint": rerank_endpoint,
        "visual_evidence": visual_evidence,
        "intent_decision": intent_decision,
        "query_is_canonical": bool(query_is_canonical),
        "embedding_model": embedding_model,
        "embedding_provider": embedding_provider,
        "embedding_api_host": embedding_api_host,
        "provider": embedding_provider,
        "api_host": embedding_api_host,
        "retrieval_identity": retrieval_identity,
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
                visual_evidence=visual_evidence,
                model_context_window=model_context_window,
                selected_text=selected_text,  # 透传框选文本
                answer_max_tokens=answer_max_tokens,
                progress_callback=progress_callback,
                query_expansion_api_key=query_expansion_api_key,
                query_expansion_model=query_expansion_model,
                query_expansion_provider=query_expansion_provider,
                query_expansion_endpoint=query_expansion_endpoint,
                intent_decision=intent_decision,
                query_is_canonical=query_is_canonical,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                embedding_api_host=embedding_api_host,
                retrieval_identity=retrieval_identity,
            ),
            timeout=60.0  # 60 秒超时
        )
        wrapped = {"context": ctx, "retrieval_meta": retrieval_meta}
    except HTTPException:
        raise
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
        "error": wrapped.get("error"),
    }


def get_relevant_context(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: Optional[str] = None,
    top_k: int = 10,
    use_rerank: bool = False,
    reranker_model: Optional[str] = None,
    candidate_k: int = 20,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None,
    selected_text: Optional[str] = None,
    model_context_window: int = 0,
    answer_max_tokens: int = 0,
    progress_callback=None,
    query_expansion_api_key: Optional[str] = None,
    query_expansion_model: str = "",
    query_expansion_provider: str = "",
    query_expansion_endpoint: str = "",
    visual_evidence: Optional[List[dict]] = None,
    intent_decision: Optional[dict] = None,
    query_is_canonical: bool = False,
    embedding_model: str = "",
    embedding_provider: str = "",
    embedding_api_host: str = "",
    provider: str = "",
    api_host: str = "",
    retrieval_identity: Optional[dict] = None,
):
    """兼容旧调用，并把 embedding 身份参数透传给核心实现。"""
    embedding_model, embedding_provider, embedding_api_host = _resolve_embedding_request_args(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
        provider=provider,
        api_host=api_host,
    )
    return _get_relevant_context_impl(
        doc_id,
        query,
        vector_store_dir=vector_store_dir,
        pages=pages,
        api_key=api_key,
        top_k=top_k,
        use_rerank=use_rerank,
        reranker_model=reranker_model,
        candidate_k=candidate_k,
        rerank_provider=rerank_provider,
        rerank_api_key=rerank_api_key,
        rerank_endpoint=rerank_endpoint,
        selected_text=selected_text,
        model_context_window=model_context_window,
        answer_max_tokens=answer_max_tokens,
        progress_callback=progress_callback,
        query_expansion_api_key=query_expansion_api_key,
        query_expansion_model=query_expansion_model,
        query_expansion_provider=query_expansion_provider,
        query_expansion_endpoint=query_expansion_endpoint,
        visual_evidence=visual_evidence,
        intent_decision=intent_decision,
        query_is_canonical=query_is_canonical,
        embedding_model=embedding_model or None,
        embedding_provider=embedding_provider or None,
        embedding_api_host=embedding_api_host or None,
        retrieval_identity=retrieval_identity,
    )
