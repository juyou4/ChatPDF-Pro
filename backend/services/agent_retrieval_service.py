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
from services.clarification_service import read_decomposition_signals
from services.decompose_service import decompose_question, should_decompose
from services.document_context_sampling import sample_document_text
from services.citation_authorization import (
    citation_authorization_summary,
    filter_authorized_citations,
    filter_authorized_context_segments,
)
from services.retrieval_agent import RetrievalAgent
# 联网三态策略的唯一实现在 services/web_search_policy.py。此处只做转发，
# 保证 route 层与 Agent 层永远看到同一个 mode。
from services.web_search_policy import (
    resolve_effective_web_search_mode as _resolve_effective_web_search_mode,
    resolve_web_search_mode as _resolve_web_search_mode,
)

logger = logging.getLogger(__name__)


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
    build_page_covered_context: Callable[..., str] | None = None
    build_visual_evidence_analyzer: Callable[..., Any] | None = None
    perform_web_search: Callable[..., Any] | None = None
    primary_key_for_target: Callable[[Any, str, str], str] | None = None


def _agent_citation_authorization(agent_result: Any, agent_doc_ctx: Any) -> dict:
    """Resolve the request-local ledger, including partial Agent failures."""
    if isinstance(agent_result, dict):
        value = agent_result.get("citation_authorization")
        if isinstance(value, dict):
            return dict(value)
    snapshot = getattr(agent_doc_ctx, "citation_authorization_snapshot", None)
    if callable(snapshot):
        value = snapshot()
        if isinstance(value, dict):
            return dict(value)
    return {}


def _merge_agent_citation_candidates(
    detail_citations: list[dict],
    fallback_citations: list[dict],
) -> list[dict]:
    """Keep the existing provenance dedupe while deferring ref rebasing to one place."""
    detail_keys = {
        _citation_provenance_dedupe_key(citation)
        for citation in detail_citations
        if isinstance(citation, dict)
    }
    merged = [dict(citation) for citation in detail_citations if isinstance(citation, dict)]
    for citation in fallback_citations:
        if not isinstance(citation, dict):
            continue
        if _citation_provenance_dedupe_key(citation) in detail_keys:
            continue
        merged.append(dict(citation))
    return merged


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


def _build_degraded_agent_result(
    agent: Any,
    *,
    degraded_to: str,
    error_kind: str = "",
    last_error: str = "",
    error_entry: dict | None = None,
    extra_diagnostics: dict | None = None,
) -> dict:
    """构造降级时的 ``retrieval_complete`` 结果，保留 agent 已积累的 partial 状态。

    超时降级、异常降级与"真空"（agent 未产出 retrieval_complete）共用同一 shape，
    调用方只通过 ``status`` / ``degraded_to`` / ``diagnostics`` 区分，
    避免下游拿到两种类型的结果对象。
    """
    partial_state = getattr(agent, "_partial_state", None)
    if not isinstance(partial_state, dict):
        partial_state = {}
    try:
        partial_retrieval_diag = agent.snapshot_partial_diagnostics(
            fallback_reason=degraded_to
        )
    except Exception as exc:
        logger.warning(f"[Agent] snapshot_partial_diagnostics 失败: {exc}")
        partial_retrieval_diag = {
            "retrieval": {"fallback_reason": degraded_to},
            "context_assembly": {"fallback_reason": degraded_to},
        }
    partial_agent_diag = dict(getattr(agent, "diagnostics", {}) or {})
    partial_agent_diag.update({
        "fallback_reason": degraded_to,
        "last_error": last_error,
        "errors": [
            *(partial_agent_diag.get("errors") or []),
            *([error_entry] if isinstance(error_entry, dict) else []),
        ],
    })
    if error_kind:
        partial_agent_diag["error_kind"] = error_kind
    if isinstance(extra_diagnostics, dict):
        partial_agent_diag.update(extra_diagnostics)
    return {
        "type": "retrieval_complete",
        "context": "",
        "detail": [],
        "search_history": list(partial_state.get("search_history") or []),
        "task_status": {},
        "diagnostics": partial_agent_diag,
        "retrieval_diagnostics": partial_retrieval_diag,
        "web_search_sources": list(partial_state.get("web_search_sources") or []),
        "web_search_context": "\n\n".join(partial_state.get("web_search_context_parts") or []),
        "web_search_reads": list(partial_state.get("web_search_reads") or []),
        "status": "degraded",
        "error_kind": error_kind,
        "degraded_to": degraded_to,
    }


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
    decomposition_signals: dict | None = None,
    web_search_audit: dict | None = None,
    web_search_execution_mode: str | None = None,
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
    # 阶段 3.1（降级版）：分解信号优先读路由层那次澄清调用的产物。
    # ``decided`` 为真表示 LLM 确实对**这一句**给出了判定（含"判定为不用拆"），
    # 此时不再发那次独立的 decompose_question——这就是本阶段净减少的一次往返。
    # 信号缺失 / 身份不匹配 / LLM 失败，一律回落既有规则 + 独立调用（fail-open）。
    hint_subs, hint_decided, hint_source = read_decomposition_signals(
        decomposition_signals,
        decomposition_question,
    )
    sub_questions: list = list(hint_subs)
    decompose_source = f"llm_signals:{hint_source}" if hint_decided else "rule"
    if not hint_decided:
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
                decompose_source = "rule_llm_call"
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning(f"[AgentRetrieval] decompose 失败，跳过分解: {exc}")
                sub_questions = []
                decompose_source = "rule_llm_failed"
        else:
            sub_questions = []
            decompose_source = "rule_no_split"
    retrieval_meta["decompose"] = {
        "source": decompose_source,
        "signal_source": hint_source,
        "sub_questions": len(sub_questions or []),
    }

    # force/off 是路由层冻结的硬边界；auto 则保留给 Planner 自主决定是否
    # 调用 web_search。无论哪种模式，执行器、查询改写和来源预算仍由系统控制。
    resolved_execution_mode = str(web_search_execution_mode or "").strip().lower()
    if resolved_execution_mode in {"off", "force"}:
        web_search_mode = resolved_execution_mode
    else:
        frozen_web_mode = str(
            _intent_field(intent_decision, "web_policy", "") or ""
        ).strip().lower()
        web_search_mode = (
            frozen_web_mode
            if frozen_web_mode in {"off", "auto", "force"}
            else _resolve_effective_web_search_mode(request, request.question)
        )
    web_search_executor = None
    frozen_web_query = citation_query or str(search_query or request.question or "").strip()
    if web_search_mode != "off" and deps.perform_web_search is not None:
        # Freeze the network query before the planner sees any untrusted PDF
        # evidence. Later planner rounds may decide whether to use the one-shot
        # tool, but document text can never become an outbound query.
        async def _agent_web_search(effective_query: str = ""):
            kwargs = {
                # retrieval_tools has already constrained this value to the
                # frozen route query plus public document anchors.
                "query_override": effective_query or frozen_web_query,
                "doc_title": doc.get("filename", ""),
                "selected_text": request.selected_text or "",
                "doc_id": request.doc_id,
                "vector_store_dir": vector_store_dir,
            }
            # Keep test and third-party dependency shims source-compatible while
            # allowing the route-owned audit record to observe the real call.
            if (
                isinstance(web_search_audit, dict)
                and _callable_accepts_keyword(deps.perform_web_search, "audit")
            ):
                kwargs["audit"] = web_search_audit
            return await deps.perform_web_search(request, **kwargs)

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
    set_web_search_query = getattr(agent_doc_ctx, "set_web_search_request_query", None)
    if callable(set_web_search_query):
        set_web_search_query(frozen_web_query if web_search_mode != "off" else "")
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
        agent_result = _build_degraded_agent_result(
            agent,
            degraded_to="agent_total_timeout",
            error_kind="TimeoutError",
            last_error=timeout_error,
            error_entry={"type": "timeout", "message": timeout_error},
            extra_diagnostics={"timeout_s": agent_timeout},
        )
    except Exception as exc:
        # 非超时异常同样保留 partial 诊断（search_history / candidate_pool /
        # evidence_state），不再把 agent_result 清空导致全部诊断丢失。
        error_kind = type(exc).__name__
        agent_error = f"agent_exception({error_kind})"
        logger.exception(f"[Agent] 多轮检索失败，降级为全文编号上下文: {exc}")
        _trace("retrieval_agent_exception", error=str(exc), error_kind=error_kind)
        await _emit_progress(emit_progress, {
            "type": "retrieval_progress",
            "phase": "loop_guard",
            "message": "Agent 检索异常中断，使用降级上下文。",
            "error": agent_error,
        })
        agent_result = _build_degraded_agent_result(
            agent,
            degraded_to="agent_exception",
            error_kind=error_kind,
            last_error=agent_error,
            error_entry={"type": "exception", "message": _safe_issue_message(exc)},
        )

    if not agent_result:
        # agent 正常结束却没有产出 retrieval_complete：真空与降级返回同一 shape。
        agent_result = _build_degraded_agent_result(
            agent,
            degraded_to="agent_no_result",
            last_error="agent_no_retrieval_complete_event",
        )

    citation_authorization = _agent_citation_authorization(agent_result, agent_doc_ctx)
    citation_authorization_summary_value = citation_authorization_summary(citation_authorization)
    retrieval_meta["_citation_authorization"] = citation_authorization
    retrieval_meta["agent_citation_authorization"] = citation_authorization_summary_value

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
        web_reads = agent_result.get("web_search_reads")
        if isinstance(web_reads, list):
            retrieval_meta["web_search_reads"] = [
                dict(item) for item in web_reads if isinstance(item, dict)
            ]
            if retrieval_meta["web_search_reads"]:
                await _emit_progress(emit_progress, {
                    "type": "web_search_read",
                    "reads": retrieval_meta["web_search_reads"],
                })
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

    if isinstance(web_search_audit, dict):
        web_sources = retrieval_meta.get("web_search_sources") or []
        if web_sources and not web_search_audit.get("executed"):
            web_search_audit.update({
                "executed": True,
                "status": "completed",
                "result_count": len(web_sources),
                "reason": "",
            })
        elif web_search_audit.get("status") == "pending":
            web_search_audit.update({
                "status": "skipped",
                "reason": (
                    "agent_web_tool_not_called"
                    if web_search_mode == "force"
                    else "agent_policy_not_selected"
                ),
            })
        retrieval_meta["web_search_audit"] = dict(web_search_audit)

    # 让降级对调用方可见，而不是只埋在 diagnostics 里。
    if isinstance(agent_result, dict):
        if agent_result.get("status"):
            retrieval_meta["agent_status"] = str(agent_result.get("status") or "")
        if agent_result.get("error_kind"):
            retrieval_meta["agent_error_kind"] = str(agent_result.get("error_kind") or "")
        if agent_result.get("degraded_to"):
            retrieval_meta["degraded_to"] = str(agent_result.get("degraded_to") or "")

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
        raw_detail_cits = deps.build_agent_detail_citations(
            agent_detail,
            query=citation_query,
            sub_questions=sub_questions or None,
            start_ref=1,
            max_citations=agent_citation_limit,
        )
        detail_cits, detail_authorization_diag = filter_authorized_citations(
            raw_detail_cits,
            citation_authorization,
            rebase_refs=False,
        )
        fallback_limit = max(0, agent_citation_limit - len(detail_cits or []))
        if fallback_limit > 0:
            _numbered_ctx, raw_fb_cits = deps.build_numbered_context_and_citations(
                pages,
                agent_context,
                query=citation_query,
                max_citations=fallback_limit,
            )
        else:
            raw_fb_cits = []
        fallback_cits, fallback_authorization_diag = filter_authorized_citations(
            raw_fb_cits,
            citation_authorization,
            rebase_refs=False,
        )
        candidate_citations = _merge_agent_citation_candidates(detail_cits, fallback_cits)
        candidate_citations = candidate_citations[:agent_citation_limit]
        agent_citations, combined_authorization_diag = filter_authorized_citations(
            candidate_citations,
            citation_authorization,
            rebase_refs=True,
        )
        # When the ledger is active, generated page-level fallbacks are never
        # allowed to silently cite sampled/expanded text.  The prompt can still
        # use the bounded Agent context, but its citations remain evidence-only.
        if not citation_authorization_summary_value["enforced"] and not agent_citations:
            agent_citations = deps.generate_page_level_citations(
                pages,
                agent_context,
                query=citation_query,
                max_citations=agent_citation_limit,
            )
        context = _build_context_from_citation_candidates(agent_citations, agent_context)
        retrieval_meta["agent_detail_citation_count"] = len(detail_cits)
        retrieval_meta["agent_citation_authorization"].update({
            "detail_filtered_count": detail_authorization_diag["filtered_count"],
            "fallback_filtered_count": fallback_authorization_diag["filtered_count"],
            "final_filtered_count": combined_authorization_diag["filtered_count"],
        })
        retrieval_meta["citations"] = agent_citations
        context_segments = _build_context_segments_from_agent_citations(agent_citations)
        context_segments, _context_authorization_diag = filter_authorized_context_segments(
            context_segments,
            citation_authorization,
        )
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
        raw_detail_cits = deps.build_agent_detail_citations(
            agent_detail,
            query=citation_query,
            sub_questions=sub_questions or None,
            start_ref=1,
            max_citations=agent_citation_limit,
        )
        detail_cits, detail_authorization_diag = filter_authorized_citations(
            raw_detail_cits,
            citation_authorization,
            rebase_refs=True,
        )
        retrieval_meta["agent_citation_authorization"].update({
            "detail_filtered_count": detail_authorization_diag["filtered_count"],
        })
        if detail_cits:
            context = _build_context_from_citation_candidates(detail_cits, "")
            if context:
                retrieval_meta["citations"] = detail_cits
                context_segments = _build_context_segments_from_agent_citations(detail_cits)
                context_segments, _context_authorization_diag = filter_authorized_context_segments(
                    context_segments,
                    citation_authorization,
                )
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

    full_text = str((doc.get("data", {}) or {}).get("full_text", "") or "")
    if deps.build_page_covered_context is not None:
        # Do not turn an agent timeout into a first-30k-character answer. The
        # route's page-covered builder preserves evidence from the middle and
        # tail while keeping the same bounded fallback budget.
        fallback_text = deps.build_page_covered_context(
            pages,
            full_text,
            max_total_chars=30_000,
        )
    else:
        # External callers can omit the route helper. Keep their fallback
        # bounded and page-agnostic, but never reduce it to the document prefix.
        fallback_text = sample_document_text(full_text, max_chars=30_000)
    numbered_ctx, fb_cits = deps.build_numbered_context_and_citations(
        pages,
        fallback_text,
        query=citation_query,
    )
    context = numbered_ctx or fallback_text
    authorized_fallback_cits, fallback_authorization_diag = filter_authorized_citations(
        fb_cits,
        citation_authorization,
        rebase_refs=True,
    )
    if citation_authorization_summary_value["enforced"]:
        # The fallback text is a resilience aid, not a tool result.  Do not
        # manufacture page citations for it when this request read no evidence.
        retrieval_meta["citations"] = authorized_fallback_cits
        if authorized_fallback_cits:
            context = _build_context_from_citation_candidates(authorized_fallback_cits, context)
            context_segments = _build_context_segments_from_agent_citations(authorized_fallback_cits)
            if context_segments:
                retrieval_meta["_context_segments"] = context_segments
    else:
        retrieval_meta["citations"] = fb_cits
    retrieval_meta["agent_citation_authorization"].update({
        "fallback_filtered_count": fallback_authorization_diag["filtered_count"],
    })
    retrieval_meta["agent_fallback"] = True
    retrieval_meta["agent_fallback_reason"] = retrieval_meta.get("agent_fallback_reason") or "empty_agent_context"
    # 这条上下文来自全文取样而非检索命中，必须显式标注，避免静默降级。
    retrieval_meta["degraded_to"] = "fulltext_sampling"
    _trace(
        "retrieval_agent_fallback",
        context_chars=len(context),
        citations=len(fb_cits or []),
        degraded_to="fulltext_sampling",
    )
    return context, retrieval_meta
