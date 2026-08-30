from datetime import datetime
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.vector_service import vector_search
from services.chat_intent_service import prepare_chat_intent
from services.advanced_search import AdvancedSearchService
from services.grep_service import grep_search
from services.semantic_group_service import SemanticGroupService
from services.embedding_service import (
    RAG_INDEX_VERSION,
    _extract_vector_semantic_identity,
    _get_semantic_groups_dir,
    _semantic_generation_identity_complete,
)
from services.block_index_service import load_block_index
from services.document_parse_state import (
    is_parse_prepared,
    parse_identity_matches,
    read_parse_manifest,
)
from services.visual_supplement_service import committed_visual_evidence_for_document
from services.semantic_group_store import active_manifest_path
from services.rerank_service import validate_rerank_configuration
from utils.middleware import (
    LoggingMiddleware,
    RetryMiddleware,
    ErrorCaptureMiddleware,
    TimeoutMiddleware,
    FallbackMiddleware,
)
from config import settings
from runtime_mode import runtime

logger = logging.getLogger(__name__)

router = APIRouter()
_VECTOR_STORE_DIR = Path(runtime.data_dir) / "vector_stores"

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


def _vector_index_matches_parse_manifest(
    doc_id: str,
    vector_store_dir: str,
    manifest: dict,
) -> bool:
    """Reject stale parse generations and obsolete index schemas."""
    index_path = Path(vector_store_dir or "") / f"{doc_id}.pkl"
    # Fresh local uploads can fall back to their current pages before a vector
    # artifact exists. Only an existing artifact can belong to an old parse.
    if not index_path.exists():
        return True
    try:
        with open(index_path, "rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:
        logger.warning("[%s] 无法读取向量索引身份: %s", doc_id, exc)
        return False
    if not isinstance(data, dict):
        return False
    try:
        index_version = int(data.get("index_version") or 0)
    except (TypeError, ValueError):
        index_version = 0
    if index_version != RAG_INDEX_VERSION:
        return False
    index_meta = data.get("index_meta") if isinstance(data.get("index_meta"), dict) else {}
    # Search used to compare only (parse_generation, document_source_hash), so a
    # parser repair that rebuilt the block tree inside one generation left these
    # chunks admissible while the reading UI had moved on. Share the admission
    # rule with the RAG index gate, chat retrieval and GraphRAG.
    try:
        block_index = load_block_index(Path(runtime.data_dir), doc_id)
    except Exception as exc:
        logger.warning("[%s] 块索引不可读，拒绝该向量索引: %s", doc_id, exc)
        return False
    if not parse_identity_matches(index_meta, manifest, block_index=block_index):
        return False
    return _semantic_generation_identity_complete(_extract_vector_semantic_identity(data))


def _semantic_groups_match_parse_manifest(doc_id: str, groups_dir: str, manifest: dict) -> bool:
    """Only expose semantic groups published for the current parse generation."""
    expected_generation = str(manifest.get("generation") or "").strip()
    expected_source_hash = str(manifest.get("source_hash") or "").strip()
    if not expected_generation or not expected_source_hash:
        return False
    try:
        active_manifest = json.loads(active_manifest_path(groups_dir, doc_id).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        str(active_manifest.get("transaction_id") or "").strip() == expected_generation
        and str(active_manifest.get("source_hash") or "").strip() == expected_source_hash
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
    embedding_api_key: Optional[str] = None
    top_k: int = 10  # 增加到10，获取更多上下文
    candidate_k: int = 20
    use_rerank: bool = False
    reranker_model: Optional[str] = None
    rerank_provider: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    parse_generation: Optional[str] = None
    document_source_hash: Optional[str] = None
    doc_store_key: Optional[str] = None  # injected
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_api_host: Optional[str] = None

    def validate_rerank(self):
        try:
            validate_rerank_configuration(
                use_rerank=self.use_rerank,
                model_name=self.reranker_model,
                provider=self.rerank_provider,
                api_key=self.rerank_api_key,
                endpoint=self.rerank_endpoint,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_identity(manifest: dict) -> tuple[str, str]:
    """Return the immutable identity of one published parse snapshot."""
    return (
        str((manifest or {}).get("generation") or "").strip(),
        str((manifest or {}).get("source_hash") or "").strip(),
    )


def _resolve_document_store(doc_store_key: Optional[str] = None) -> dict:
    if not hasattr(router, "documents_store"):
        raise HTTPException(status_code=500, detail="文档存储未初始化")
    root_store = router.documents_store
    if not doc_store_key:
        return root_store
    nested_store = root_store.get(doc_store_key, {}) if isinstance(root_store, dict) else {}
    return nested_store if isinstance(nested_store, dict) else {}


def _require_requested_parse_identity(
    request: SearchRequest,
    active_identity: tuple[str, str],
) -> None:
    """Fence requests issued from an already stale document view."""
    requested_generation = str(request.parse_generation or "").strip()
    requested_source_hash = str(request.document_source_hash or "").strip()
    if bool(requested_generation) != bool(requested_source_hash):
        raise HTTPException(
            status_code=400,
            detail="parse_generation 与 document_source_hash 必须成对提供",
        )
    if requested_generation and (requested_generation, requested_source_hash) != active_identity:
        raise HTTPException(status_code=409, detail="文档解析代际已变化，请基于当前文档重新搜索")


def _require_search_snapshot_current(
    request: SearchRequest,
    expected_identity: tuple[str, str],
) -> dict:
    """Reject results produced while the document switched parse generations."""
    current_store = _resolve_document_store(request.doc_store_key)
    current_doc = current_store.get(request.doc_id)
    if not isinstance(current_doc, dict):
        raise HTTPException(status_code=409, detail="搜索期间文档已被移除或替换，请重新搜索")

    current_manifest = _require_document_parse_ready(request.doc_id, current_doc)
    if _parse_identity(current_manifest) != expected_identity:
        raise HTTPException(status_code=409, detail="搜索期间文档解析代际已变化，请重新搜索")
    if not _vector_index_matches_parse_manifest(
        request.doc_id,
        str(getattr(router, "vector_store_dir", "") or ""),
        current_manifest,
    ):
        raise HTTPException(status_code=409, detail="搜索期间问答索引已更新，请重新搜索")
    return current_manifest


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
        store = _resolve_document_store(request.doc_store_key)
        if request.doc_id not in store:
            raise HTTPException(status_code=404, detail="文档未找到")

        doc = store[request.doc_id]
        parse_manifest = _require_document_parse_ready(request.doc_id, doc)
        parse_identity = _parse_identity(parse_manifest)
        if not all(parse_identity):
            raise HTTPException(status_code=409, detail="当前文档缺少完整解析身份，请重新上传或重新解析")
        _require_requested_parse_identity(request, parse_identity)
        if not _vector_index_matches_parse_manifest(
            request.doc_id,
            str(getattr(router, "vector_store_dir", "") or ""),
            parse_manifest,
        ):
            raise HTTPException(status_code=409, detail="当前文档的问答索引尚未发布，请稍后重试")
        pages = doc.get("data", {}).get("pages", [])

        # Independent search has the same immutable question contract as chat.
        # Its explicit syntax endpoints (grep/regex/boolean) deliberately stay
        # outside this semantic route.
        intent = prepare_chat_intent(
            original_question=request.query,
            intent_question=request.query,
            interaction_mode="default",
        )
        intent_decision = intent.to_dict()
        retrieval_query = intent.intent_question
        dynamic_top_k = intent.top_k

        logger.debug(
            "[Search] query_type=%s dynamic_top_k=%s reason=%s",
            intent.query_type,
            dynamic_top_k,
            intent.reasoning,
        )

        middlewares = build_search_middlewares()

        search_status = await vector_search(
            request.doc_id,
            retrieval_query,
            vector_store_dir=router.vector_store_dir,
            pages=pages,
            api_key=request.embedding_api_key,
            top_k=dynamic_top_k,  # 使用动态计算的top_k
            candidate_k=max(request.candidate_k, dynamic_top_k),
            use_rerank=request.use_rerank,
            reranker_model=request.reranker_model,
            rerank_provider=request.rerank_provider,
            rerank_api_key=request.rerank_api_key,
            rerank_endpoint=request.rerank_endpoint,
            visual_evidence=committed_visual_evidence_for_document(doc),
            middlewares=middlewares,
            return_status=True,
            intent_decision=intent_decision,
            query_is_canonical=True,
            embedding_model=request.embedding_model or "",
            embedding_provider=request.embedding_provider or "",
            embedding_api_host=request.embedding_api_host or "",
        )

        # 兼容仍返回历史列表形态的中间件和测试替身。
        if isinstance(search_status, dict):
            results = search_status.get("results")
            if not isinstance(results, list):
                results = []
                search_status = {
                    **search_status,
                    "error": search_status.get("error") or "检索服务返回了无效结果",
                    "error_code": search_status.get("error_code") or "invalid_search_response",
                    "degraded": True,
                    "fallback_reason": search_status.get("fallback_reason") or "invalid_search_response",
                }
        elif isinstance(search_status, list):
            results = search_status
            search_status = {
                "results": results,
                "error": None,
                "error_code": None,
                "degraded": False,
                "fallback_reason": None,
                "fallback_used": False,
                "timings": {},
            }
        else:
            results = []
            search_status = {
                "results": [],
                "error": "检索服务返回了无效响应",
                "error_code": "invalid_search_response",
                "degraded": True,
                "fallback_reason": "invalid_search_response",
                "fallback_used": False,
                "timings": {},
            }

        current_manifest = _require_search_snapshot_current(request, parse_identity)
        current_generation, current_source_hash = _parse_identity(current_manifest)
        rerank_degraded = any(
            isinstance(item, dict) and item.get("_rerank_degraded")
            for item in results
        )
        degraded = bool(
            search_status.get("degraded")
            or search_status.get("error")
            or rerank_degraded
        )
        response_error = search_status.get("error")
        response_error_code = search_status.get("error_code")
        response_fallback_reason = search_status.get("fallback_reason")
        if rerank_degraded:
            response_error = response_error or "重排模型暂不可用，已保留混合检索原排序"
            response_error_code = response_error_code or "rerank_unavailable"
            response_fallback_reason = response_fallback_reason or "rerank_unavailable"
        public_results = []
        for item in results:
            if not isinstance(item, dict):
                continue
            safe_item = dict(item)
            # Internal fallback diagnostics can contain provider/library
            # details; expose the stable top-level status instead.
            for key in ("_rerank_degraded", "_rerank_error_code", "_rerank_error"):
                safe_item.pop(key, None)
            public_results.append(safe_item)
        return {
            "results": public_results,
            "query_type": intent.query_type,
            "dynamic_top_k": dynamic_top_k,
            "rerank_enabled": request.use_rerank and len(results) > 0 and not rerank_degraded,
            "candidate_k": max(request.candidate_k, dynamic_top_k),
            # `used_*` describes what actually ran. An omitted/failed rerank
            # must not be reported as the implicit local model.
            "used_provider": (
                request.rerank_provider or "local"
                if request.use_rerank and not rerank_degraded
                else None
            ),
            "used_model": (
                request.reranker_model or "BAAI/bge-reranker-base"
                if request.use_rerank and not rerank_degraded
                else None
            ),
            "rerank_requested": bool(request.use_rerank),
            "fallback_used": bool(search_status.get("fallback_used") or rerank_degraded),
            "fallback_reason": response_fallback_reason,
            "degraded": degraded,
            "retrieval_degraded": degraded,
            "retrieval_status": "degraded" if degraded else "ok",
            "error": response_error,
            "error_code": response_error_code,
            "rerank_degraded": rerank_degraded,
            "retrieval_timings": search_status.get("timings") or {},
            "parse_generation": current_generation,
            "document_source_hash": current_source_hash,
            "parse_identity": {
                "generation": current_generation,
                "source_hash": current_source_hash,
            },
            "original_question": intent.original_question,
            "intent_question": intent.intent_question,
            "retrieval_query": retrieval_query,
            "intent_id": intent.intent_id,
            "intent_version": intent.version,
            "intent": intent_decision,
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

        # The semantic map is a published parse-generation artifact.  It can
        # remain useful while the optional chunk vector index is rebuilding,
        # but it must never fall back to a prior parser generation.
        semantic_root = _get_semantic_groups_dir()
        if not _semantic_groups_match_parse_manifest(
            request.doc_id,
            semantic_root,
            parse_manifest,
        ):
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
