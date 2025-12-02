from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.vector_service import vector_search
from utils.middleware import (
    LoggingMiddleware,
    RetryMiddleware,
    ErrorCaptureMiddleware,
    TimeoutMiddleware,
    FallbackMiddleware,
)
from config import settings

router = APIRouter()


class SearchRequest(BaseModel):
    doc_id: str
    query: str
    api_key: Optional[str] = None
    top_k: int = 5
    candidate_k: int = 20
    use_rerank: bool = False
    reranker_model: Optional[str] = None
    rerank_provider: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    doc_store_key: Optional[str] = None  # injected

    def validate_rerank(self):
        provider = (self.rerank_provider or "").lower()
        if self.use_rerank and provider in {"cohere", "jina"} and not self.rerank_api_key:
            raise HTTPException(status_code=400, detail=f"使用 {provider} rerank 需要提供 rerank_api_key")


def build_search_middlewares():
    middlewares = []
    if settings.enable_search_logging:
        middlewares.append(LoggingMiddleware())
    middlewares.append(RetryMiddleware(retries=settings.search_retry_retries, delay=settings.search_retry_delay))
    middlewares.append(ErrorCaptureMiddleware(log_path=settings.error_log_path))
    middlewares.append(TimeoutMiddleware(timeout=settings.search_timeout))
    if settings.search_fallback_provider or settings.search_fallback_model:
        middlewares.append(FallbackMiddleware(settings.search_fallback_provider, settings.search_fallback_model))
    if settings.enable_search_degrade:
        middlewares.append(FallbackMiddleware(settings.search_fallback_provider, settings.search_fallback_model))
    return middlewares


@router.post("/api/search")
async def search_in_pdf(request: SearchRequest):
    request.validate_rerank()
    try:
        # pages/doc store 将由 app 注入，避免全局重复
        if not hasattr(router, "documents_store"):
            raise HTTPException(status_code=500, detail="文档存储未初始化")

        store = router.documents_store if not request.doc_store_key else router.documents_store.get(request.doc_store_key, {})
        if request.doc_id not in store:
            raise HTTPException(status_code=404, detail="文档未找到")

        doc = store[request.doc_id]
        pages = doc.get("data", {}).get("pages", [])

        middlewares = build_search_middlewares()

        results = await vector_search(
            request.doc_id,
            request.query,
            vector_store_dir=router.vector_store_dir,
            pages=pages,
            api_key=request.api_key,
            top_k=request.top_k,
            candidate_k=max(request.candidate_k, request.top_k),
            use_rerank=request.use_rerank,
            reranker_model=request.reranker_model,
            rerank_provider=request.rerank_provider,
            rerank_api_key=request.rerank_api_key,
            rerank_endpoint=request.rerank_endpoint,
            middlewares=middlewares
        )

        return {
            "results": results,
            "rerank_enabled": request.use_rerank and len(results) > 0,
            "candidate_k": max(request.candidate_k, request.top_k),
            "used_provider": request.rerank_provider or "local",
            "used_model": request.reranker_model or ("BAAI/bge-reranker-base" if request.use_rerank else None),
            "fallback_used": False
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
