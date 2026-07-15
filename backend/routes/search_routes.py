from datetime import datetime
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.vector_service import vector_search
from services.query_analyzer import get_retrieval_strategy
from services.advanced_search import AdvancedSearchService
from services.grep_service import grep_search
from services.semantic_group_service import SemanticGroupService
from services.embedding_service import _get_semantic_groups_dir
from services.document_parse_state import is_parse_prepared, read_parse_manifest
from services.semantic_group_store import active_manifest_path
from utils.middleware import (
    LoggingMiddleware,
    RetryMiddleware,
    ErrorCaptureMiddleware,
    TimeoutMiddleware,
    FallbackMiddleware,
)
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# 高级搜索服务实例
_advanced_search_service = AdvancedSearchService()


def _require_document_parse_ready(doc_id: str, doc: dict) -> dict:
    """Do not expose provisional parser output through search endpoints."""
    manifest = read_parse_manifest(doc or {}, doc_id=doc_id)
    if is_parse_prepared(manifest):
        return manifest

    route = str(manifest.get("requested_route") or manifest.get("route") or "auto")
    stage = str(manifest.get("stage") or "")
    if route == "mineru" and stage == "awaiting_rag_index":
        detail = "MinerU 已完成版面解析，正在等待问答索引发布"
    elif route == "mineru":
        detail = "当前文档正在按 MinerU 全程解析，完成前不能搜索文档内容"
    else:
        detail = "当前文档解析尚未完成，请稍后重试"
    raise HTTPException(status_code=409, detail=detail)


def _is_legacy_parse_manifest(manifest: dict) -> bool:
    return bool((manifest.get("metadata") or {}).get("legacy_inferred"))


def _vector_index_matches_parse_manifest(
    doc_id: str,
    vector_store_dir: str,
    manifest: dict,
) -> bool:
    """Reject an old vector generation after a same-document reparse."""
    if _is_legacy_parse_manifest(manifest):
        return True
    index_path = Path(vector_store_dir or "") / f"{doc_id}.pkl"
    # Fresh local uploads can fall back to their current pages before a vector
    # artifact exists. Only an existing artifact can belong to an old parse.
    if not index_path.exists():
        return True
    try:
        with open(index_path, "rb") as handle:
            data = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return False
    if not isinstance(data, dict):
        return False
    index_meta = data.get("index_meta") if isinstance(data.get("index_meta"), dict) else {}
    return (
        str(index_meta.get("parse_generation") or "") == str(manifest.get("generation") or "")
        and str(index_meta.get("document_source_hash") or "") == str(manifest.get("source_hash") or "")
    )


def _semantic_groups_match_parse_manifest(doc_id: str, groups_dir: str, manifest: dict) -> bool:
    """Only expose semantic groups published for the current parse generation."""
    if _is_legacy_parse_manifest(manifest):
        return True
    try:
        active_manifest = json.loads(active_manifest_path(groups_dir, doc_id).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        str(active_manifest.get("transaction_id") or "") == str(manifest.get("generation") or "")
        and str(active_manifest.get("source_hash") or "") == str(manifest.get("source_hash") or "")
    )


class GrepSearchRequest(BaseModel):
    """精确文本搜索（grep）请求模型"""
    doc_id: str
    query: str
    limit: int = 20
    context_chars: int = 2000
    case_insensitive: bool = True


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
        parse_manifest = _require_document_parse_ready(request.doc_id, doc)
        if not _vector_index_matches_parse_manifest(
            request.doc_id,
            str(getattr(router, "vector_store_dir", "") or ""),
            parse_manifest,
        ):
            raise HTTPException(status_code=409, detail="当前文档的问答索引尚未发布，请稍后重试")
        pages = doc.get("data", {}).get("pages", [])

        # 智能分析查询类型，动态调整top_k
        strategy = get_retrieval_strategy(request.query)
        dynamic_top_k = strategy['top_k']

        logger.debug(
            "[Search] query_type=%s dynamic_top_k=%s reason=%s",
            strategy["query_type"],
            dynamic_top_k,
            strategy["reasoning"],
        )

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


@router.post("/api/search/grep")
async def grep_search_endpoint(request: GrepSearchRequest):
    """精确文本搜索（grep）端点

    支持 | 分隔多关键词 OR 逻辑，返回匹配位置和上下文片段。
    """
    try:
        if not hasattr(router, "documents_store"):
            raise HTTPException(status_code=500, detail="文档存储未初始化")

        if request.doc_id not in router.documents_store:
            raise HTTPException(status_code=404, detail="文档未找到")

        doc = router.documents_store[request.doc_id]
        _require_document_parse_ready(request.doc_id, doc)
        full_text = doc.get("data", {}).get("full_text", "")

        if not full_text:
            return {"results": [], "total": 0}

        results = grep_search(
            query=request.query,
            text=full_text,
            limit=request.limit,
            context_chars=request.context_chars,
            case_insensitive=request.case_insensitive,
        )

        return {"results": results, "total": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grep搜索失败: {str(e)}")


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
        _require_document_parse_ready(request.doc_id, doc)
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
        _require_document_parse_ready(request.doc_id, doc)
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


class DocumentMapRequest(BaseModel):
    """文档地图请求模型"""
    doc_id: str
    limit: int = 50


@router.post("/api/document/map")
async def document_map(request: DocumentMapRequest):
    """文档结构概览（意群地图）端点

    返回文档所有意群的 ID、字数、关键词、摘要和页码范围，
    用于快速了解文档整体结构。
    """
    try:
        if not hasattr(router, "documents_store"):
            raise HTTPException(status_code=500, detail="文档存储未初始化")
        if request.doc_id not in router.documents_store:
            raise HTTPException(status_code=404, detail="文档未找到")
        parse_manifest = _require_document_parse_ready(request.doc_id, router.documents_store[request.doc_id])

        # ``active`` manifest lives at the semantic-group root, while the
        # published JSON itself may live in its active generation directory.
        groups_root = _get_semantic_groups_dir()
        if not _semantic_groups_match_parse_manifest(request.doc_id, groups_root, parse_manifest):
            return {
                "map": [],
                "total": 0,
                "message": "该文档尚未生成当前解析代际的语义意群",
            }
        groups_dir = _get_semantic_groups_dir(request.doc_id)
        group_svc = SemanticGroupService()
        groups = group_svc.load_groups(request.doc_id, groups_dir)

        if groups is None:
            return {
                "map": [],
                "total": 0,
                "message": "该文档尚未生成语义意群，请先启用意群功能",
            }

        map_entries = []
        for g in groups[:request.limit]:
            map_entries.append({
                "group_id": g.group_id,
                "char_count": g.char_count,
                "keywords": g.keywords,
                "summary": g.summary[:200] if g.summary else "",
                "page_range": list(g.page_range),
            })

        return {
            "map": map_entries,
            "total": len(map_entries),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档地图失败: {str(e)}")
