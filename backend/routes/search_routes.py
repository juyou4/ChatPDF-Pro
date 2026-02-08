from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.vector_service import vector_search
from services.query_analyzer import get_retrieval_strategy
from services.advanced_search import AdvancedSearchService
from utils.middleware import (
    LoggingMiddleware,
    RetryMiddleware,
    ErrorCaptureMiddleware,
    TimeoutMiddleware,
    FallbackMiddleware,
)
from config import settings

router = APIRouter()

# 高级搜索服务实例
_advanced_search_service = AdvancedSearchService()


class RegexSearchRequest(BaseModel):
    """正则表达式搜索请求模型"""
    doc_id: str
    pattern: str
    limit: int = 20
    context_chars: int = 200


class BooleanSearchRequest(BaseModel):
    """布尔逻辑搜索请求模型"""
    doc_id: str
    query: str
    limit: int = 20
    context_chars: int = 200


class SearchRequest(BaseModel):
    doc_id: str
    query: str
    api_key: Optional[str] = None
    top_k: int = 10  # 增加到10，获取更多上下文
    candidate_k: int = 20
    use_rerank: bool = False
    reranker_model: Optional[str] = None
    rerank_provider: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    doc_store_key: Optional[str] = None  # injected

    def validate_rerank(self):
        provider = (self.rerank_provider or "").lower()
        # 所有非本地的 rerank provider 都需要 api_key
        cloud_providers = {"cohere", "jina", "silicon", "aliyun", "openai", "moonshot", "deepseek", "zhipu", "minimax"}
        if self.use_rerank and provider in cloud_providers and not self.rerank_api_key:
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

        # 智能分析查询类型，动态调整top_k
        strategy = get_retrieval_strategy(request.query)
        dynamic_top_k = strategy['top_k']
        
        print(f"[Search] 查询类型: {strategy['query_type']}, 动态top_k: {dynamic_top_k}, 原因: {strategy['reasoning']}")

        middlewares = build_search_middlewares()

        results = await vector_search(
            request.doc_id,
            request.query,
            vector_store_dir=router.vector_store_dir,
            pages=pages,
            api_key=request.api_key,
            top_k=dynamic_top_k,  # 使用动态计算的top_k
            candidate_k=max(request.candidate_k, dynamic_top_k),
            use_rerank=request.use_rerank,
            reranker_model=request.reranker_model,
            rerank_provider=request.rerank_provider,
            rerank_api_key=request.rerank_api_key,
            rerank_endpoint=request.rerank_endpoint,
            middlewares=middlewares
        )

        return {
            "results": results,
            "query_type": strategy['query_type'],
            "dynamic_top_k": dynamic_top_k,
            "rerank_enabled": request.use_rerank and len(results) > 0,
            "candidate_k": max(request.candidate_k, dynamic_top_k),
            "used_provider": request.rerank_provider or "local",
            "used_model": request.reranker_model or ("BAAI/bge-reranker-base" if request.use_rerank else None),
            "fallback_used": False
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.post("/api/search/regex")
async def regex_search(request: RegexSearchRequest):
    """正则表达式搜索端点

    在指定文档的全文中执行正则表达式匹配搜索。
    正则语法无效时返回 HTTP 400 错误。
    """
    try:
        # 检查文档存储是否已初始化
        if not hasattr(router, "documents_store"):
            raise HTTPException(status_code=500, detail="文档存储未初始化")

        # 查找文档
        if request.doc_id not in router.documents_store:
            raise HTTPException(status_code=404, detail="文档未找到")

        doc = router.documents_store[request.doc_id]
        full_text = doc.get("data", {}).get("full_text", "")

        if not full_text:
            return {"results": [], "total": 0}

        # 调用高级搜索服务执行正则搜索
        results = _advanced_search_service.regex_search(
            pattern=request.pattern,
            text=full_text,
            limit=request.limit,
            context_chars=request.context_chars,
        )

        return {"results": results, "total": len(results)}

    except ValueError as e:
        # 正则表达式语法无效，返回 400
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"正则搜索失败: {str(e)}")


@router.post("/api/search/boolean")
async def boolean_search(request: BooleanSearchRequest):
    """布尔逻辑搜索端点

    在指定文档的全文中执行布尔逻辑搜索（支持 AND/OR/NOT）。
    结果按相关性分数降序排列。
    """
    try:
        # 检查文档存储是否已初始化
        if not hasattr(router, "documents_store"):
            raise HTTPException(status_code=500, detail="文档存储未初始化")

        # 查找文档
        if request.doc_id not in router.documents_store:
            raise HTTPException(status_code=404, detail="文档未找到")

        doc = router.documents_store[request.doc_id]
        full_text = doc.get("data", {}).get("full_text", "")

        if not full_text:
            return {"results": [], "total": 0}

        # 调用高级搜索服务执行布尔搜索
        results = _advanced_search_service.boolean_search(
            query=request.query,
            text=full_text,
            limit=request.limit,
            context_chars=request.context_chars,
        )

        return {"results": results, "total": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"布尔搜索失败: {str(e)}")
