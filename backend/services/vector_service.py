from typing import List, Optional

from services.embedding_service import (
    build_vector_index,
    search_document_chunks,
    get_relevant_context,
    normalize_embedding_model_id
)
from models.model_registry import EMBEDDING_MODELS
from fastapi import HTTPException
from utils.middleware import BaseMiddleware, apply_middlewares_before, apply_middlewares_after
from services.embedding_service import get_embedding_function  # re-export if needed


def validate_embedding_model(embedding_model: str) -> str:
    """Validate and normalize embedding model id"""
    normalized = normalize_embedding_model_id(embedding_model)
    if normalized:
        return normalized
    if ":" in embedding_model:
        _, model_part = embedding_model.split(":", 1)
        if model_part in EMBEDDING_MODELS:
            return model_part
    raise HTTPException(status_code=400, detail=f"Embedding模型 '{embedding_model}' 未配置或不受支持")


def create_index(doc_id: str, full_text: str, vector_store_dir: str, embedding_model: str, api_key: Optional[str], api_host: Optional[str]):
    """Wrapper to build vector index with validation"""
    embedding_model = validate_embedding_model(embedding_model)
    build_vector_index(doc_id, full_text, vector_store_dir, embedding_model, api_key, api_host)


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
    """Wrapper for vector search with optional middleware hooks"""
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
        results = search_document_chunks(
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
        )
        wrapped = {"results": results}
    except Exception as e:
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
    middlewares: Optional[List[BaseMiddleware]] = None
) -> str:
    """Wrapper to get relevant context with middleware hooks"""
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
        ctx = get_relevant_context(
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
        )
        wrapped = {"context": ctx}
    except Exception as e:
        wrapped = {"context": "", "error": str(e)}

    wrapped = await apply_middlewares_after(wrapped, middlewares or [])
    return wrapped.get("context", ctx if 'ctx' in locals() else "")
