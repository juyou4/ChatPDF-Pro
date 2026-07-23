"""Agentic RAG 执行服务。

该模块只负责运行多轮检索、收集诊断、生成可注入 prompt 的上下文与引用候选。
路由层负责请求校验、SSE 包装和最终模型调用。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from config import settings
from services.decompose_service import decompose_question, should_decompose
from services.retrieval_agent import RetrievalAgent

logger = logging.getLogger(__name__)


def _resolve_web_search_mode(request) -> str:
    explicit = str(getattr(request, "web_search_mode", "") or "").strip().lower()
    if explicit in {"off", "auto", "force"}:
        return explicit
    return "auto" if bool(getattr(request, "enable_web_search", False)) else "off"


def _intent_field(intent_decision: Any, field: str, default: Any = None) -> Any:
    """Read a frozen IntentDecision, serialized form, or turn context safely."""
    missing = object()
    decision = intent_decision
    nested = None
    if isinstance(decision, dict):
        value = decision.get(field, missing)
        nested = decision.get("intent")
    elif decision is not None:
        value = getattr(decision, field, missing)
        nested = getattr(decision, "intent", None)
    else:
        value = missing
    if value is missing and nested is not None:
        value = (
            nested.get(field, missing)
            if isinstance(nested, dict)
            else getattr(nested, field, missing)
        )
    return default if value is missing or value is None else value


def _frozen_intent_question(
    intent_decision: Any,
    intent_question: str,
    fallback: str,
) -> str:
    return str(
        _intent_field(intent_decision, "intent_question", "")
        or intent_question
        or fallback
        or ""
    ).strip()


def _callable_accepts_keyword(target: Any, keyword: str) -> bool:
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


@dataclass(frozen=True)
class AgentRetrievalDependencies:
    """Agent 检索服务依赖包。

    这些 callable 由路由层提供，避免 service 反向导入 route，同时让依赖边界显式化。
    """

    get_cheap_model_params: Callable[[Any], tuple[str, str, str]]
    build_agent_doc_context: Callable[..., Any]
    merge_retrieval_meta: Callable[[dict | None, dict | None], dict]
    annotate_agent_gate: Callable[..., dict]
    resolve_citation_candidate_limit: Callable[..., int]
    build_numbered_context_and_citations: Callable[..., tuple[str, list[dict]]]
    generate_page_level_citations: Callable[..., list[dict]]
    build_agent_detail_citations: Callable[..., list[dict]]
    build_visual_evidence_analyzer: Callable[..., Any] | None = None
    perform_web_search: Callable[..., Any] | None = None
    primary_key_for_target: Callable[[Any, str, str], str] | None = None


def _build_context_from_citation_candidates(citations: list[dict], fallback_context: str = "") -> str:
    """用已有引用候选构造带编号的上下文，保证 prompt 中存在可引用锚点。"""
    parts: list[str] = []
    for idx, citation in enumerate(citations or [], start=1):
        if not isinstance(citation, dict):
            continue
        try:
            ref = int(citation.get("ref") or idx)
        except (TypeError, ValueError):
            ref = idx
        text = re.sub(
            r"\s+",
            " ",
            str(
                citation.get("context_segment_text")
                or citation.get("source_text")
                or citation.get("display_text")
                or citation.get("highlight_text")
                or ""
            ),
        ).strip()
        if not text:
            continue
        page_range = citation.get("page_range") or []
        page_text = ""
        if isinstance(page_range, (list, tuple)) and page_range:
            page_text = f" 页码:{page_range[0]}-{page_range[-1]}" if len(page_range) > 1 else f" 页码:{page_range[0]}"
        parts.append(f"[{ref}]{page_text}\n{text}")
    if parts:
        return "\n\n".join(parts)
    return fallback_context


def _build_context_segments_from_agent_citations(citations: list[dict]) -> list[dict]:
    """从最终 agent 引用候选生成响应侧 context_segments。"""
    segments: list[dict] = []
    for citation in citations or []:
        if not isinstance(citation, dict):
            continue
        try:
            ref = int(citation.get("ref") or 0)
        except (TypeError, ValueError):
            continue
        if ref <= 0:
            continue
        text = re.sub(
            r"\s+",
            " ",
            str(
                citation.get("context_segment_text")
                or citation.get("source_text")
                or citation.get("display_text")
                or citation.get("highlight_text")
                or ""
            ),
        ).strip()
        if not text:
            continue
        segments.append({
            "ref": ref,
            "text": text,
            "page_range": citation.get("page_range") or [],
            "group_id": citation.get("group_id", ""),
            "context_id": citation.get("context_id", ""),
            "evidence_id": citation.get("evidence_id", ""),
            "asset_id": citation.get("asset_id", ""),
            "analyzed_asset_id": citation.get("analyzed_asset_id", ""),
            "block_id": citation.get("block_id", ""),
            "chunk_id": citation.get("chunk_id", ""),
            "child_chunk_id": citation.get("child_chunk_id", ""),
            "parent_id": citation.get("parent_id", ""),
            "table_id": citation.get("table_id", ""),
            "table_bundle_id": citation.get("table_bundle_id", ""),
            "evidence_unit_id": citation.get("evidence_unit_id", ""),
            "retrieval_type": citation.get("retrieval_type", ""),
            "visual_evidence_id": citation.get("visual_evidence_id", ""),
            "visual_enhancement": citation.get("visual_enhancement"),
            "visual_source": citation.get("visual_source", ""),
            "visual_supplement_revision": citation.get("visual_supplement_revision", ""),
            "figure_id": citation.get("figure_id", ""),
            "figure_bbox": citation.get("figure_bbox") or citation.get("bbox") or [],
            "visual_model": citation.get("visual_model") or {},
            "runtime_visual_analysis": citation.get("runtime_visual_analysis"),
            "purpose": citation.get("purpose", ""),
            "prompt_version": citation.get("prompt_version", ""),
            "parse_generation": citation.get("parse_generation", ""),
            "confidence": citation.get("confidence"),
        })
    return segments


def _citation_provenance_dedupe_key(citation: dict) -> tuple[str, str, str]:
    """优先用稳定 provenance 字段去重，粗粒度来源需结合文本指纹。"""
    if not isinstance(citation, dict):
        return ("empty", "", "")
    text = re.sub(
        r"\s+",
        " ",
        str(
            citation.get("context_segment_text")
            or citation.get("source_text")
            or citation.get("display_text")
            or citation.get("highlight_text")
            or ""
        ),
    ).strip().casefold()
    for field in (
        "evidence_id",
        "chunk_id",
        "child_chunk_id",
    ):
        value = re.sub(r"\s+", " ", str(citation.get(field) or "")).strip().casefold()
        if value:
            return ("id", field, value)
    text_fingerprint = _citation_text_fingerprint(text)
    for field in ("table_bundle_id", "table_id", "context_id", "parent_id", "group_id"):
        value = re.sub(r"\s+", " ", str(citation.get(field) or "")).strip().casefold()
        if value:
            return ("scoped_text", field, f"{value}:{text_fingerprint}")
    return ("text", "", text_fingerprint)


def _citation_text_fingerprint(text: str) -> str:
    """生成短文本指纹，避免只看前缀导致同段不同证据被误合并。"""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    if not normalized:
        return ""
    prefix = normalized[:120]
    suffix = normalized[-120:] if len(normalized) > 120 else ""
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).hexdigest()
    return f"{len(normalized)}:{prefix}:{suffix}:{digest}"


def _preview_for_log(text: str | None, limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


_FATAL_VECTOR_ERROR_MESSAGES = {
    "vector_embedding_identity_conflict": "当前 Embedding 配置与文档索引不一致，请切换原配置或重建索引",
    "vector_index_schema_conflict": "当前文档问答索引格式已升级，请按当前解析结果重建",
    "vector_index_identity_conflict": "当前文档问答索引与请求身份不一致，请重建索引后重试",
    "vector_search_http_401": "当前 Embedding 凭证无效或已过期，请检查后重试",
    "vector_search_http_403": "当前 Embedding 服务拒绝访问，请检查权限配置后重试",
    "vector_index_unavailable": "当前文档向量索引不可用，请重新上传 PDF 或等待索引构建完成",
}


def _is_fatal_vector_error_code(error_code: str) -> bool:
    normalized = _normalize_agent_error_code(error_code)
    return bool(
        normalized
        and (
            normalized in _FATAL_VECTOR_ERROR_MESSAGES
            or normalized.startswith("vector_search_http_")
        )
    )


def _normalize_agent_error_code(value: Any) -> str:
    text = str(value or "").strip().lower()
    if re.fullmatch(r"[a-z0-9_]{3,80}", text):
        return text
    return ""


def _coerce_http_status_code(value: Any) -> int | None:
    try:
        status_code = int(value)
    except (TypeError, ValueError):
        return None
    return status_code if 100 <= status_code <= 599 else None


def _safe_issue_message(value: Any, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _iter_agent_tool_issue_candidates(agent_result: dict):
    if not isinstance(agent_result, dict):
        return

    diagnostics = agent_result.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key in ("tool_errors", "tool_timings", "candidate_pool_trace"):
            items = diagnostics.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        yield item

    retrieval_diagnostics = agent_result.get("retrieval_diagnostics")
    if isinstance(retrieval_diagnostics, dict):
        retrieval_diag = retrieval_diagnostics.get("retrieval")
        if isinstance(retrieval_diag, dict):
            tool_errors = retrieval_diag.get("tool_errors")
            if isinstance(tool_errors, list):
                for item in tool_errors:
                    if isinstance(item, dict):
                        yield item
            candidate_pool = retrieval_diag.get("candidate_pool")
            if isinstance(candidate_pool, dict):
                by_tool = candidate_pool.get("by_tool")
                if isinstance(by_tool, list):
                    for item in by_tool:
                        if isinstance(item, dict):
                            yield item

    search_history = agent_result.get("search_history")
    if isinstance(search_history, list):
        for item in search_history:
            if isinstance(item, dict):
                yield item


def _match_fatal_vector_issue(
    item: dict,
    *,
    default_tool: str = "",
) -> dict | None:
    if not isinstance(item, dict):
        return None

    owner = str(item.get("channel") or item.get("tool") or default_tool or "").strip().lower()
    channel_errors = item.get("channel_errors")
    if isinstance(channel_errors, list):
        for child in channel_errors:
            if not isinstance(child, dict):
                continue
            matched = _match_fatal_vector_issue(child, default_tool=owner)
            if matched is not None:
                return matched

    error_code = _normalize_agent_error_code(
        item.get("error_code") or item.get("errorCode")
    )
    status_code = _coerce_http_status_code(
        item.get("status_code") or item.get("statusCode")
    )
    fatal = bool(item.get("fatal"))
    if (
        not fatal
        or status_code is None
        or not _is_fatal_vector_error_code(error_code)
    ):
        return None

    if owner and owner not in {"vector", "vector_search", "search_document"}:
        return None

    return {
        "tool": owner or default_tool or "search_document",
        "error_code": error_code,
        "status_code": status_code,
        "error": _safe_issue_message(item.get("error")),
    }


def _find_fatal_vector_identity_issue(agent_result: dict) -> dict | None:
    if not isinstance(agent_result, dict):
        return None
    for item in _iter_agent_tool_issue_candidates(agent_result):
        matched = _match_fatal_vector_issue(item)
        if matched is not None:
            return matched
    return None


def _fatal_vector_identity_detail(issue: dict | None) -> str:
    issue = issue if isinstance(issue, dict) else {}
    error_code = _normalize_agent_error_code(issue.get("error_code"))
    fallback = _safe_issue_message(issue.get("error"))
    if error_code in _FATAL_VECTOR_ERROR_MESSAGES:
        return _FATAL_VECTOR_ERROR_MESSAGES[error_code]
    if error_code.startswith("vector_search_http_"):
        return "向量检索请求失败，请稍后重试"
    return fallback or "向量检索请求失败，请稍后重试"


async def _emit_progress(emit_progress: Callable[[dict], Any] | None, event: dict) -> None:
    if emit_progress is None:
        return
    result = emit_progress(event)
    if asyncio.iscoroutine(result):
        await result


async def _iterate_with_total_timeout(async_iterable, timeout_seconds: float):
    iterator = async_iterable.__aiter__()
    deadline = time.perf_counter() + max(1.0, float(timeout_seconds or 1.0))
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            close = getattr(iterator, "aclose", None)
            if close:
                await close()
            raise asyncio.TimeoutError()
        try:
            yield await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            close = getattr(iterator, "aclose", None)
            if close:
                await close()
            raise


async def run_agent_retrieval_for_context(
    *,
    request,
    doc: dict,
    search_query: str,
    query_type: str,
    agent_gate: dict,
    intent_decision: Any | None = None,
    intent_question: str = "",
    retrieval_meta: dict | None = None,
    emit_progress: Callable[[dict], Any] | None = None,
    trace: Callable[..., None] | None = None,
    vector_store_dir: str = "",
    deps: AgentRetrievalDependencies | None = None,
) -> tuple[str, dict]:
    """执行 Agentic RAG 并返回上下文与检索元数据。

    外部依赖通过 AgentRetrievalDependencies 注入，避免 service 反向导入 route。
    """
    if deps is None:
        raise ValueError("Agent retrieval service missing dependencies")

    def _trace(stage: str, **fields) -> None:
        if trace is not None:
            trace(stage, **fields)

    retrieval_meta = dict(retrieval_meta or {})
    use_agent = True
    citation_query = _frozen_intent_question(
        intent_decision,
        intent_question,
        str(search_query or getattr(request, "question", "") or ""),
    )
    frozen_query_type = str(_intent_field(intent_decision, "query_type", "") or "").strip()
    effective_query_type = frozen_query_type or str(query_type or "").strip()

    await _emit_progress(emit_progress, {
        "type": "retrieval_progress",
        "phase": "agent_mode",
        "message": "正在启动多轮检索代理...",
    })

    agent_model, agent_provider, agent_endpoint = deps.get_cheap_model_params(request)
    # Planner/decomposition requests are an auxiliary provider boundary. The
    # route resolver permits the chat credential only for the same provider
    # and endpoint origin.
    credential_resolver = deps.primary_key_for_target
    agent_api_key = (
        credential_resolver(request, agent_provider, agent_endpoint)
        if callable(credential_resolver)
        else ""
    )

    # The route-frozen intent question is the root task. ``search_query`` may
    # remain useful as retrieval provenance, but must not make the planner
    # reclassify a rewritten template as a different user intent.
    decomposition_question = citation_query or str(search_query or request.question or "").strip()
    sub_questions: list = []
    if should_decompose(decomposition_question):
        try:
            sub_questions = await asyncio.wait_for(
                decompose_question(
                    question=decomposition_question,
                    api_key=agent_api_key,
                    model=agent_model,
                    provider=agent_provider,
                    endpoint=agent_endpoint or "",
                ),
                timeout=2.5,
            )
            sub_questions = (sub_questions or [])[:3]
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(f"[AgentRetrieval] decompose 失败，跳过分解: {exc}")
            sub_questions = []

    web_search_mode = _resolve_web_search_mode(request)
    web_search_executor = None
    if web_search_mode != "off" and deps.perform_web_search is not None:
        # Freeze the network query before the planner sees any untrusted PDF
        # evidence. Later planner rounds may decide whether to use the one-shot
        # tool, but document text can never become an outbound query.
        frozen_web_query = citation_query or str(search_query or request.question or "").strip()

        async def _agent_web_search():
            return await deps.perform_web_search(
                request,
                query_override=frozen_web_query,
                doc_title=doc.get("filename", ""),
                selected_text=request.selected_text or "",
                doc_id=request.doc_id,
                vector_store_dir=vector_store_dir,
            )

        web_search_executor = _agent_web_search

    context_kwargs = {
        # Embedding calls are an independent provider boundary. Never fall
        # back to the primary chat credential here: the selected embedding
        # endpoint may belong to a different provider.
        "api_key": request.embedding_api_key or "",
        "use_rerank": bool(request.use_rerank),
        "reranker_model": request.reranker_model or "",
        "rerank_provider": request.rerank_provider or "",
        "rerank_api_key": request.rerank_api_key or "",
        "rerank_endpoint": request.rerank_endpoint or "",
        "web_search_executor": web_search_executor,
        "embedding_model": getattr(request, "embedding_model", None) or "",
        "embedding_provider": getattr(request, "embedding_provider", None) or "",
        "embedding_api_host": getattr(request, "embedding_api_host", None) or "",
    }
    if _callable_accepts_keyword(deps.build_agent_doc_context, "intent_decision"):
        context_kwargs["intent_decision"] = intent_decision
    agent_doc_ctx = deps.build_agent_doc_context(
        request.doc_id,
        doc,
        vector_store_dir,
        **context_kwargs,
    )
    # Keep custom/test builders compatible while making the route-owned
    # decision available before any planner tool can run.
    if intent_decision is not None:
        set_intent_decision = getattr(agent_doc_ctx, "set_intent_decision", None)
        if callable(set_intent_decision):
            set_intent_decision(intent_decision)
        else:
            setattr(agent_doc_ctx, "intent_decision", intent_decision)
    visual_analyzer = None
    if deps.build_visual_evidence_analyzer is not None:
        try:
            visual_analyzer = deps.build_visual_evidence_analyzer(
                request=request,
                doc=doc,
                modal_asset_index=getattr(agent_doc_ctx, "modal_asset_index", {}) or {},
            )
        except Exception as exc:
            logger.warning("[AgentRetrieval] 构建请求级视觉分析器失败，降级为文本检索: %s", exc)
    configure_visual_analyzer = getattr(agent_doc_ctx, "configure_visual_analyzer", None)
    if callable(configure_visual_analyzer):
        configure_visual_analyzer(
            visual_analyzer,
            active_question=citation_query,
        )
    visual_analysis_available = bool(
        callable(getattr(agent_doc_ctx, "visual_analysis_available", None))
        and agent_doc_ctx.visual_analysis_available()
    )
    agent = RetrievalAgent(
        api_key=agent_api_key,
        model=agent_model,
        provider=agent_provider,
        endpoint=agent_endpoint,
        max_rounds=max(1, min(int(getattr(settings, "agent_max_rounds", 5) or 5), 10)),
        temperature=float(getattr(settings, "agent_planner_temperature", 0.3) or 0.3),
        planner_retries=max(0, int(getattr(settings, "agent_planner_retries", 1) or 1)),
        max_context_tokens=max(500, int(getattr(settings, "agent_context_max_tokens", 12000) or 12000)),
        max_iterations=max(1, min(int(getattr(settings, "agent_max_iterations", 3) or 3), 10)),
        max_tool_calls=max(1, int(getattr(settings, "agent_max_tool_calls", 12) or 12)),
        context_compress_threshold=max(0, int(getattr(settings, "agent_context_compress_threshold", 16000) or 16000)),
        max_tool_concurrency=max(1, min(int(getattr(settings, "agent_tool_max_concurrency", 5) or 5), 5)),
        sub_questions=sub_questions or None,
        use_rerank=bool(request.use_rerank),
        reranker_model=request.reranker_model or "",
        rerank_provider=request.rerank_provider or "",
        rerank_api_key=request.rerank_api_key or "",
        rerank_endpoint=request.rerank_endpoint or "",
        web_search_mode=web_search_mode,
        intent_decision=intent_decision,
    )

    agent_result: dict = {}
    agent_timeout = max(5.0, float(getattr(settings, "agent_total_timeout", 75.0) or 75.0))
    if visual_analysis_available:
        # Two selected figures run concurrently, but still need room for two
        # planner rounds and ordinary retrieval around the visual call.
        agent_timeout = max(agent_timeout, 105.0)
    try:
        agent_events = agent.run(
            question=citation_query or search_query or request.question or "",
            doc_ctx=agent_doc_ctx,
            doc_name=doc.get("filename", ""),
        )
        async for agent_event in _iterate_with_total_timeout(agent_events, agent_timeout):
            event_type = agent_event.get("type")
            if event_type == "retrieval_progress":
                await _emit_progress(emit_progress, agent_event)
                phase = agent_event.get("phase", "")
                message = agent_event.get("message", "")
                if phase in {"agent_start", "round_start", "planning", "planner_error", "executing", "tool_result", "complete"}:
                    _trace(f"agent_{phase}", message=_preview_for_log(message, 120))
            elif event_type == "retrieval_complete":
                agent_result = agent_event
    except asyncio.TimeoutError:
        timeout_error = f"agent_total_timeout(>{agent_timeout:.0f}s)"
        logger.warning(f"[Agent] 多轮检索超时，降级为全文编号上下文: {timeout_error}")
        _trace("retrieval_agent_timeout", timeout_s=agent_timeout)
        await _emit_progress(emit_progress, {
            "type": "retrieval_progress",
            "phase": "loop_guard",
            "message": f"Agent 检索超过 {agent_timeout:.0f}s，使用降级上下文。",
            "error": timeout_error,
        })
        try:
            partial_retrieval_diag = agent.snapshot_partial_diagnostics(
                fallback_reason="agent_total_timeout"
            )
        except Exception as exc:
            logger.warning(f"[Agent] snapshot_partial_diagnostics 失败: {exc}")
            partial_retrieval_diag = {
                "retrieval": {"fallback_reason": "agent_total_timeout"},
                "context_assembly": {"fallback_reason": "agent_total_timeout"},
            }
        partial_agent_diag = dict(getattr(agent, "diagnostics", {}) or {})
        partial_agent_diag.update({
            "fallback_reason": "agent_total_timeout",
            "last_error": timeout_error,
            "errors": [
                *(partial_agent_diag.get("errors") or []),
                {"type": "timeout", "message": timeout_error},
            ],
            "timeout_s": agent_timeout,
        })
        agent_result = {
            "type": "retrieval_complete",
            "context": "",
            "detail": [],
            "search_history": list(
                (agent._partial_state.get("search_history") if hasattr(agent, "_partial_state") else None) or []
            ),
            "task_status": {},
            "diagnostics": partial_agent_diag,
            "retrieval_diagnostics": partial_retrieval_diag,
            "web_search_sources": list((agent._partial_state.get("web_search_sources") if hasattr(agent, "_partial_state") else None) or []),
            "web_search_context": "\n\n".join((agent._partial_state.get("web_search_context_parts") if hasattr(agent, "_partial_state") else None) or []),
        }
    except Exception as exc:
        logger.warning(f"[Agent] 多轮检索失败，降级为全文编号上下文: {exc}")
        _trace("retrieval_agent_exception", error=str(exc))
        agent_result = {}

    agent_context = agent_result.get("context", "") if isinstance(agent_result, dict) else ""
    agent_detail = agent_result.get("detail", []) if isinstance(agent_result, dict) else []
    if isinstance(agent_result, dict):
        if agent_result.get("search_history"):
            retrieval_meta["agent_search_history"] = agent_result.get("search_history")
        web_sources = agent_result.get("web_search_sources")
        if isinstance(web_sources, list):
            retrieval_meta["web_search_sources"] = [dict(item) for item in web_sources if isinstance(item, dict)]
        web_context = str(agent_result.get("web_search_context") or "").strip()
        if web_context:
            retrieval_meta["web_search_context"] = web_context
        if agent_result.get("task_status"):
            retrieval_meta["task_status"] = agent_result.get("task_status")
        agent_retrieval_diagnostics = agent_result.get("retrieval_diagnostics")
        agent_diagnostics = agent_result.get("diagnostics")
        merged_agent_diagnostics = {}
        if isinstance(agent_retrieval_diagnostics, dict):
            merged_agent_diagnostics.update(agent_retrieval_diagnostics)
        if isinstance(agent_diagnostics, dict):
            merged_agent_diagnostics["agent"] = agent_diagnostics
        if merged_agent_diagnostics:
            retrieval_meta = deps.merge_retrieval_meta(
                retrieval_meta,
                {"diagnostics": merged_agent_diagnostics},
            )
        if isinstance(agent_diagnostics, dict):
            evidence_state = agent_diagnostics.get("evidence_state")
            if isinstance(evidence_state, dict):
                retrieval_meta["agent_evidence_state"] = dict(evidence_state)
            if agent_diagnostics.get("last_error"):
                retrieval_meta["agent_error"] = agent_diagnostics.get("last_error")
            if agent_diagnostics.get("fallback_reason"):
                retrieval_meta["agent_fallback_reason"] = agent_diagnostics.get("fallback_reason")

    fatal_vector_issue = _find_fatal_vector_identity_issue(agent_result)
    if fatal_vector_issue is not None:
        detail = _fatal_vector_identity_detail(fatal_vector_issue)
        status_code = _coerce_http_status_code(fatal_vector_issue.get("status_code")) or 409
        _trace(
            "retrieval_agent_fatal_vector_conflict",
            error_code=fatal_vector_issue.get("error_code", ""),
            status_code=status_code,
        )
        raise HTTPException(status_code=status_code, detail=detail)

    retrieval_meta["agent_detail"] = agent_detail
    retrieval_meta["agent_mode"] = True
    retrieval_meta["agent_gate"] = deps.annotate_agent_gate(
        retrieval_meta.get("agent_gate", agent_gate),
        use_agent=use_agent,
        agent_mode=True,
        search_query_passthrough=True,
    )
    retrieval_meta["query_type"] = effective_query_type
    if citation_query:
        retrieval_meta.setdefault("intent_question", citation_query)
    frozen_intent_id = str(_intent_field(intent_decision, "intent_id", "") or "").strip()
    frozen_intent_version = str(_intent_field(intent_decision, "version", "") or "").strip()
    if frozen_intent_id:
        retrieval_meta.setdefault("intent_id", frozen_intent_id)
    if frozen_intent_version:
        retrieval_meta.setdefault("intent_version", frozen_intent_version)

    pages = doc.get("data", {}).get("pages", [])
    if agent_context:
        agent_citation_limit = deps.resolve_citation_candidate_limit(agent_mode=True)
        detail_cits = deps.build_agent_detail_citations(
            agent_detail,
            query=citation_query,
            sub_questions=sub_questions or None,
            start_ref=1,
            max_citations=agent_citation_limit,
        )
        fallback_limit = max(0, agent_citation_limit - len(detail_cits or []))
        if fallback_limit > 0:
            numbered_ctx, fb_cits = deps.build_numbered_context_and_citations(
                pages,
                agent_context,
                query=citation_query,
                max_citations=fallback_limit,
            )
        else:
            numbered_ctx = _build_context_from_citation_candidates(detail_cits, agent_context)
            fb_cits = []
        if detail_cits:
            detail_keys = {
                _citation_provenance_dedupe_key(citation)
                for citation in (detail_cits or [])
                if isinstance(citation, dict)
            }
            rebased_fallback: list[dict] = []
            for citation in (fb_cits or []):
                key = _citation_provenance_dedupe_key(citation)
                if key in detail_keys:
                    continue
                rebased = dict(citation)
                rebased["ref"] = len(detail_cits) + len(rebased_fallback) + 1
                rebased_fallback.append(rebased)
                if len(detail_cits) + len(rebased_fallback) >= agent_citation_limit:
                    break
            fb_cits = [*detail_cits, *rebased_fallback]
            numbered_ctx = _build_context_from_citation_candidates(fb_cits, numbered_ctx or agent_context)
            retrieval_meta["agent_detail_citation_count"] = len(detail_cits)
        context = numbered_ctx or agent_context
        agent_citations = fb_cits or deps.generate_page_level_citations(
            pages,
            agent_context,
            query=citation_query,
            max_citations=agent_citation_limit,
        )
        retrieval_meta["citations"] = agent_citations
        context_segments = _build_context_segments_from_agent_citations(agent_citations)
        if context_segments:
            retrieval_meta["_context_segments"] = context_segments
        retrieval_meta["agent_citation_candidate_limit"] = agent_citation_limit
        retrieval_meta["agent_context_chars"] = len(agent_context)
        _trace(
            "retrieval_agent_done",
            context_chars=len(context),
            citations=len(agent_citations or []),
            detail=len(agent_detail or []),
        )
        return context, retrieval_meta

    # Agent 可能已经 fetch/backfill 到高质量 detail 证据，但最终 context 因压缩、
    # planner 结束或异常为空。此时优先用 detail 证据恢复编号上下文，避免退回全文前缀。
    if agent_detail:
        agent_citation_limit = deps.resolve_citation_candidate_limit(agent_mode=True)
        detail_cits = deps.build_agent_detail_citations(
            agent_detail,
            query=citation_query,
            sub_questions=sub_questions or None,
            start_ref=1,
            max_citations=agent_citation_limit,
        )
        if detail_cits:
            context = _build_context_from_citation_candidates(detail_cits, "")
            if context:
                retrieval_meta["citations"] = detail_cits
                context_segments = _build_context_segments_from_agent_citations(detail_cits)
                if context_segments:
                    retrieval_meta["_context_segments"] = context_segments
                retrieval_meta["agent_fallback"] = True
                retrieval_meta["agent_fallback_reason"] = (
                    retrieval_meta.get("agent_fallback_reason")
                    or "empty_agent_context_detail_recovered"
                )
                retrieval_meta["agent_detail_citation_count"] = len(detail_cits)
                retrieval_meta["agent_citation_candidate_limit"] = agent_citation_limit
                _trace(
                    "retrieval_agent_detail_recovered",
                    context_chars=len(context),
                    citations=len(detail_cits),
                    detail=len(agent_detail or []),
                )
                return context, retrieval_meta

    fallback_text = (doc.get("data", {}) or {}).get("full_text", "")[:30000]
    numbered_ctx, fb_cits = deps.build_numbered_context_and_citations(
        pages,
        fallback_text,
        query=citation_query,
    )
    context = numbered_ctx or fallback_text
    retrieval_meta["citations"] = fb_cits
    retrieval_meta["agent_fallback"] = True
    retrieval_meta["agent_fallback_reason"] = retrieval_meta.get("agent_fallback_reason") or "empty_agent_context"
    _trace(
        "retrieval_agent_fallback",
        context_chars=len(context),
        citations=len(fb_cits or []),
    )
    return context, retrieval_meta
