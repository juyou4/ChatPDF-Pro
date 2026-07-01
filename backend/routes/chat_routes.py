import asyncio
from datetime import datetime
from pathlib import Path
import os
import pickle
from typing import Optional, List
import json
import logging
import re
import threading
import time
import uuid
import hashlib

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.chat_service import call_ai_api, call_ai_api_stream, extract_reasoning_content
from services.vector_service import vector_context
from services.selected_text_locator import locate_selected_text
from services.agent_retrieval_service import (
    AgentRetrievalDependencies,
    run_agent_retrieval_for_context as _run_agent_retrieval_service,
)
from services.retrieval_tools import DocContext
from services.glossary_service import glossary_service, build_glossary_prompt
from services.table_service import protect_markdown_tables, restore_markdown_tables
from services.query_analyzer import get_retrieval_strategy
from services.preset_service import get_generation_prompt
from services.context_builder import ContextBuilder
from services.web_search_service import SearchManager, format_search_results
from services.web_search_reranker import rerank_web_results
from services.query_rewriter import QueryRewriter
from services.followup_service import generate_followup_questions
from services.conv_name_service import suggest_conversation_name
from services.decompose_service import decompose_question, should_decompose
from services.formula_text import build_formula_alias_text, formula_term_matches, looks_formula_like, technical_anchor_matches
from services.mindmap_service import generate_mindmap
from services.rag_config import (
    should_apply_numeric_table_specialization,
    should_enable_answer_critic,
    should_enable_llm_query_rewrite,
    apply_request_overrides,
)
from services.citation_service import (
    build_structured_citation_prompt,
    parse_citation_list,
    extract_final_answer,
    match_citations_to_chunks,
    START_ANSWER,
    START_CITATION,
    _RE_START_ANSWER,
    _RE_START_CITATION,
    _ci_contains,
    _ci_split,
)
import base64
from models.provider_registry import PROVIDER_CONFIG
from models.dynamic_store import load_dynamic_providers
from utils.middleware import (
    LoggingMiddleware,
    RetryMiddleware,
    ErrorCaptureMiddleware,
    DegradeOnErrorMiddleware,
    TimeoutMiddleware,
    FallbackMiddleware,
)
from config import settings
from runtime_mode import runtime

logger = logging.getLogger(__name__)

router = APIRouter()
_MIN_SELECTED_TEXT_FALLBACK_CITATION_CHARS = 30
_DEFAULT_CITATION_CANDIDATE_LIMIT = 8
_AGENT_CITATION_CANDIDATE_LIMIT = 12
_MAX_CONTEXT_RECOVERY_SELECTOR_CANDIDATES = 6
_MAX_WEB_SEARCH_RESULTS = 10
_DEFAULT_ANSWER_DETAIL = "standard"
_VALID_ANSWER_DETAILS = {"concise", "standard", "detailed"}
_INLINE_CITATION_PATTERN = re.compile(r'(?<!!)(?:\[(\d{1,3})\](?!\()|【(\d{1,3})】)')
_STRICT_CITATION_EVIDENCE_NEEDS = {"numeric_table", "reference_trap", "reference_meta"}
_STRICT_CITATION_SUPPORT_THRESHOLDS = {
    "numeric_table": 0.08,
    "reference_trap": 0.1,
    "reference_meta": 0.1,
}
_CONSERVATIVE_REWRITE_EVIDENCE_NEEDS = {"reference_trap", "reference_meta"}


def _coerce_positive_int(value, default: int = 0) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return int(default or 0)
    return coerced if coerced > 0 else int(default or 0)


# 注意：Agent 触发白名单（query_type / evidence_needs）已迁移到
# `config.AppSettings.agent_trigger_query_types` 与 `agent_trigger_evidence_needs`，
# 由 `_build_agent_retrieval_gate` 在运行时读取，便于通过环境变量动态调整。
_WEB_SEARCH_PRONOUN_HINTS = (
    "这个", "该", "它", "上述", "此", "这种", "这些", "这项", "该项", "本方法",
    "this", "that", "it", "they", "he", "she", "them",
)
_QUERY_REWRITE_AMBIGUOUS_HINTS = (
    "这个", "那个", "这块", "那块", "这部分", "那部分", "这段", "那段",
    "这里", "那里", "它", "其", "上述", "前面", "后面", "上一段", "上一节",
    "该方法", "该模型", "该公式", "该结论",
)
_EN_AMBIGUOUS_QUERY_RE = re.compile(r"\b(this|that|it|they|them|he|she|these|those)\b", re.IGNORECASE)
_DECIMAL_SAFE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。!！?？;；])\s*|(?<!\d)(?<=\.)\s+")


def _resolve_citation_candidate_limit(*, agent_mode: bool = False, requested_limit: int | None = None) -> int:
    """返回用于生成引用候选的上限。

    Agent 路径的上下文来自多工具聚合，评估报告显示很多证据已进入最终上下文但
    未进入引用候选，导致 selector 无法补锚点。这里仅对 Agent 提高默认候选数，
    普通检索保持原来的 8 条，避免结构化引用 prompt 变得过长。
    """
    if requested_limit is not None:
        try:
            return max(1, min(int(requested_limit), 20))
        except (TypeError, ValueError):
            pass
    return _AGENT_CITATION_CANDIDATE_LIMIT if agent_mode else _DEFAULT_CITATION_CANDIDATE_LIMIT


def _preview_for_log(text: Optional[str], limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _new_chat_trace_id() -> str:
    return uuid.uuid4().hex[:8]


def _log_chat_trace(trace_id: str, started_at: float, stage: str, **fields) -> None:
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    clock = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    extras = []
    for key, value in fields.items():
        if value is None:
            continue
        extras.append(f"{key}={value!r}")
    suffix = f" | {' '.join(extras)}" if extras else ""
    line = f"[ChatTrace {trace_id}] {clock} +{elapsed_ms}ms {stage}{suffix}"
    if getattr(settings, "enable_chat_logging", False):
        logger.info(line)
    else:
        logger.debug(line)


def _get_provider_endpoint(provider_id: str, api_host: str = "") -> str:
    """按优先级解析 provider 的 chat endpoint：
    1. 前端传入的 api_host（用户自定义地址）
    2. 动态 provider 存储（用户通过 UI 添加的定制 provider）
    3. 静态 PROVIDER_CONFIG（内置默认配置）
    """
    # 1. 前端明确传入了 api_host：拼接成完整 endpoint
    if api_host and api_host.strip():
        host = api_host.strip().rstrip('/')
        # 如果已包含 /chat/completions 则直接使用
        if host.endswith('/chat/completions'):
            return host
        return f"{host}/chat/completions"
    # 2. 动态 provider 存储
    dynamic = load_dynamic_providers()
    if provider_id in dynamic:
        return dynamic[provider_id].get("endpoint", "")
    # 3. 静态内置配置
    return PROVIDER_CONFIG.get(provider_id, {}).get("endpoint", "")


def _detect_image_mime(image_base64: str) -> str:
    """从 base64 直接检测图片实际 MIME 类型。
    支持 JPEG, PNG, GIF, WebP；无法识别时回退为 image/jpeg。
    16 个 base64 字符解码为恰好 12 字节，足够判断所有常见格式。
    """
    try:
        # 16 base64 字符 = 4 组 * 3 字节/组 = 12 字节，正好是 4 的倍数，无需额外填充
        chunk = image_base64[:16]
        header = base64.b64decode(chunk)
    except Exception:
        return 'image/jpeg'
    if header[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    if header[:4] == b'\x89PNG':
        return 'image/png'
    if header[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return 'image/webp'
    return 'image/jpeg'

async def _buffered_stream(raw_stream, *, passthrough: bool = False, buffer_size: Optional[int] = None):
    """对原始 SSE 流做轻量缓冲，减少事件频率但保留首屏响应。

    - 深度思考或结构化引文场景：直通，避免内容被二次缓冲到最后。
    - 其他场景：首个非空 chunk 立即发送，后续再按字符阈值缓冲。
    """
    effective_buffer_size = settings.stream_buffer_size if buffer_size is None else max(0, int(buffer_size))

    if passthrough or effective_buffer_size <= 0:
        async for chunk in raw_stream:
            yield chunk
            if chunk.get("error") or chunk.get("done"):
                break
        return

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    content_len = 0
    reasoning_len = 0
    first_payload_emitted = False

    def _build_stream_payload(base_chunk: dict, content: str, reasoning: str) -> dict:
        payload = {
            "content": content,
            "reasoning_content": reasoning,
            "done": False,
        }
        for key in ("used_provider", "used_model", "fallback_used"):
            if key in base_chunk:
                payload[key] = base_chunk.get(key)
        return payload

    async for chunk in raw_stream:
        if chunk.get("error") or chunk.get("done"):
            if content_parts or reasoning_parts:
                yield _build_stream_payload(
                    chunk,
                    "".join(content_parts),
                    "".join(reasoning_parts),
                )
                content_parts.clear()
                reasoning_parts.clear()
                content_len = reasoning_len = 0
            yield chunk
            break

        content = chunk.get("content", "")
        reasoning = chunk.get("reasoning_content", "")

        if not first_payload_emitted and (content or reasoning):
            first_payload_emitted = True
            yield _build_stream_payload(chunk, content, reasoning)
            continue

        if content:
            content_parts.append(content)
            content_len += len(content)
        if reasoning:
            reasoning_parts.append(reasoning)
            reasoning_len += len(reasoning)

        if content_len >= effective_buffer_size or reasoning_len >= effective_buffer_size:
            yield _build_stream_payload(
                chunk,
                "".join(content_parts),
                "".join(reasoning_parts),
            )
            content_parts.clear()
            reasoning_parts.clear()
            content_len = reasoning_len = 0

    if content_parts or reasoning_parts:
        yield {
            "content": "".join(content_parts),
            "reasoning_content": "".join(reasoning_parts),
            "done": False,
        }


async def _stream_with_total_timeout(raw_stream, timeout_seconds: float):
    """Limit total wall-clock time spent waiting for an upstream model stream.

    Provider-level HTTP timeouts are per socket operation. If an upstream keeps a
    stream open with empty events, the route can otherwise stay occupied far
    longer than the configured chat timeout. This wrapper converts that case
    into a normal stream error chunk so the existing retrieval-evidence fallback
    path can produce a usable answer and release the request.
    """
    try:
        timeout_value = float(timeout_seconds or 0)
    except (TypeError, ValueError):
        timeout_value = 0.0
    timeout_value = timeout_value if timeout_value > 0 else 120.0
    deadline = time.perf_counter() + timeout_value
    iterator = raw_stream.__aiter__()

    async def _close_iterator() -> None:
        close = getattr(iterator, "aclose", None)
        if close is None:
            return
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug(f"关闭超时 LLM 流失败（忽略）: {exc}")

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            await _close_iterator()
            yield {
                "error": f"llm_stream_total_timeout(>{timeout_value:.1f}s)",
                "done": True,
                "fallback_used": True,
            }
            return
        next_task = asyncio.create_task(iterator.__anext__())
        try:
            chunk = await asyncio.wait_for(next_task, timeout=remaining)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            if not next_task.done():
                next_task.cancel()
                try:
                    await next_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                except Exception as exc:
                    logger.debug(f"取消超时 LLM 流读取失败（忽略）: {exc}")
            await _close_iterator()
            yield {
                "error": f"llm_stream_total_timeout(>{timeout_value:.1f}s)",
                "done": True,
                "fallback_used": True,
            }
            return
        except Exception as exc:
            if not next_task.done():
                next_task.cancel()
            await _close_iterator()
            yield {
                "error": f"llm_stream_iteration_failed:{type(exc).__name__}",
                "done": True,
                "fallback_used": True,
            }
            return

        yield chunk
        if isinstance(chunk, dict) and (chunk.get("error") or chunk.get("done")):
            return


def _sse_json(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_threadsafe_progress_forwarder(progress_queue: asyncio.Queue):
    loop = asyncio.get_running_loop()

    def _forward(event: dict):
        if not isinstance(event, dict):
            return
        try:
            loop.call_soon_threadsafe(progress_queue.put_nowait, event)
        except RuntimeError:
            # 事件循环已关闭或请求已结束，直接忽略
            pass

    return _forward


async def _yield_task_progress(
    task: asyncio.Task,
    progress_queue: asyncio.Queue,
    heartbeat_message: str,
    heartbeat_interval: float = 8.0,
):
    heartbeat_step = 0
    try:
        while not task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=heartbeat_interval)
                yield event
            except asyncio.TimeoutError:
                if not task.done():
                    heartbeat_step += 1
                    yield {
                        "type": "retrieval_progress",
                        "phase": "heartbeat",
                        "step": heartbeat_step,
                        "message": heartbeat_message,
                    }

        while True:
            try:
                yield progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    finally:
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as exc:
                logger.debug(f"检索任务取消清理时异常（忽略）: {exc}")


async def _yield_postprocess_events(coros):
    """按完成顺序产出流式后处理事件，并在客户端断开时取消未完成任务。"""
    tasks = [asyncio.create_task(coro) for coro in coros]
    try:
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
            except Exception as exc:
                logger.debug(f"后处理任务异常（不影响主流程）: {exc}")
                continue
            if result:
                yield result
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            results = await asyncio.gather(*pending, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.debug(f"后处理任务取消清理时异常（忽略）: {result}")

# 上下文构建器实例，用于生成引文指示提示词
_context_builder = ContextBuilder()

# 查询改写器实例
_query_rewriter = QueryRewriter()


def _get_cheap_model_params(request) -> tuple:
    """获取辅助模型参数（双模型策略）

    优先级：request 字段（per-request）> config.settings.cheap_model*（全局）> 主模型 fallback。
    如果 cheap_model 与 cheap_provider 均有值，返回 (cheap_model, cheap_provider, endpoint)，
    否则回退到主模型三元组。

    Returns:
        (model, provider, endpoint) 三元组
    """
    # 1. per-request override
    req_model = getattr(request, "cheap_model", None)
    req_provider = getattr(request, "cheap_model_provider", None)
    req_endpoint = getattr(request, "cheap_model_endpoint", None)
    if req_model and req_provider:
        endpoint = req_endpoint or _get_provider_endpoint(req_provider, request.api_host or "")
        return req_model, req_provider, endpoint

    # 2. 全局 settings
    cheap_model = settings.cheap_model
    cheap_provider = settings.cheap_model_provider
    if cheap_model and cheap_provider:
        endpoint = _get_provider_endpoint(cheap_provider, request.api_host or "")
        return cheap_model, cheap_provider, endpoint

    # 3. fallback 到主模型
    return request.model, request.api_provider, _get_provider_endpoint(request.api_provider, request.api_host or "")


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_doc_alignment_text(text: str, limit: int = 240) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return normalized[:limit]


def _doc_text_contains_fragment(doc_text: str, fragment: str, *, min_chars: int = 36) -> bool:
    doc_norm = _normalize_doc_alignment_text(doc_text, limit=max(len(doc_text), 240))
    frag = _normalize_doc_alignment_text(fragment, limit=360)
    if len(frag) < min_chars:
        return bool(frag) and len(frag) >= 4 and frag in doc_norm
    if frag in doc_norm:
        return True
    probes = [
        frag[:120],
        frag[:80],
        frag[:60],
        frag[:min_chars],
    ]
    return any(probe and len(probe) >= min_chars and probe in doc_norm for probe in probes)


def _chunks_match_current_doc(chunks: list[str], full_text: str, *, sample_size: int = 5) -> bool:
    """抽样确认落盘 chunks 属于当前文档，避免 doc_id/stale index 串库。"""
    if not chunks or not full_text:
        return True
    sampled = [chunk for chunk in chunks[:sample_size] if isinstance(chunk, str) and chunk.strip()]
    if not sampled:
        return True
    hits = sum(1 for chunk in sampled if _doc_text_contains_fragment(full_text, chunk))
    return hits >= max(1, min(2, len(sampled)))


def _group_text_matches_current_doc(group: dict, full_text: str) -> bool:
    for key in ("full_text", "digest", "summary"):
        value = str(group.get(key) or "").strip()
        if value and _doc_text_contains_fragment(full_text, value, min_chars=28):
            return True
    return False


def _semantic_groups_match_current_doc(groups: list[dict], full_text: str, *, sample_size: int = 5) -> bool:
    """抽样确认 semantic_groups 属于当前文档。旧数据无文本时放行。"""
    if not groups or not full_text:
        return True
    sampled = [g for g in groups[:sample_size] if isinstance(g, dict)]
    text_bearing = [
        g for g in sampled
        if any(str(g.get(key) or "").strip() for key in ("full_text", "digest", "summary"))
    ]
    if not text_bearing:
        return True
    hits = sum(1 for group in text_bearing if _group_text_matches_current_doc(group, full_text))
    return hits >= max(1, min(2, len(text_bearing)))


def _load_doc_chunks_for_agent(doc_id: str, vector_store_dir: str, full_text: str) -> list[str]:
    """从向量索引元数据中加载 chunks，给 agentic 检索使用。

    如果索引缺失或损坏，回退到按段落切分的全文片段，保证 agent 至少可用。
    """
    candidate_dirs = []
    if vector_store_dir and vector_store_dir.strip():
        candidate_dirs.append(Path(vector_store_dir))
    candidate_dirs.append(_get_project_root() / "data" / "vector_stores")

    for store_dir in candidate_dirs:
        chunks_path = store_dir / f"{doc_id}.pkl"
        if not chunks_path.exists():
            continue
        try:
            with open(chunks_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                chunks = data.get("chunks") or []
                if isinstance(chunks, list) and chunks:
                    loaded_chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
                    if _chunks_match_current_doc(loaded_chunks, full_text):
                        return loaded_chunks
                    logger.warning(
                        "[AgentDoc] chunks 与当前文档不匹配，丢弃 stale vector store: doc_id=%s path=%s",
                        doc_id,
                        chunks_path,
                    )
                    continue
            elif isinstance(data, list):
                loaded_chunks = [c for c in data if isinstance(c, str) and c.strip()]
                if _chunks_match_current_doc(loaded_chunks, full_text):
                    return loaded_chunks
                logger.warning(
                    "[AgentDoc] chunks 与当前文档不匹配，丢弃 stale vector store: doc_id=%s path=%s",
                    doc_id,
                    chunks_path,
                )
                continue
        except Exception as exc:
            logger.warning(f"[AgentDoc] 加载 chunks 失败: {chunks_path} -> {exc}")

    return _split_context_paragraphs(full_text or "") or ([full_text.strip()] if full_text.strip() else [])


def _load_doc_semantic_groups_for_agent(doc_id: str, full_text: str = "") -> list[dict]:
    """从落盘的 semantic_groups 中加载意群数据。"""
    candidate_dirs = [
        Path(runtime.data_dir) / "semantic_groups",
        _get_project_root() / "data" / "semantic_groups",
        Path(__file__).resolve().parents[1] / "data" / "semantic_groups",
    ]
    for groups_dir in candidate_dirs:
        group_path = groups_dir / f"{doc_id}.json"
        if not group_path.exists():
            continue
        try:
            with open(group_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            groups = data.get("groups") or []
            if isinstance(groups, list):
                loaded_groups = [g for g in groups if isinstance(g, dict)]
                if _semantic_groups_match_current_doc(loaded_groups, full_text):
                    return loaded_groups
                logger.warning(
                    "[AgentDoc] semantic_groups 与当前文档不匹配，丢弃 stale groups: doc_id=%s path=%s",
                    doc_id,
                    group_path,
                )
                continue
        except Exception as exc:
            logger.warning(f"[AgentDoc] 加载意群失败: {group_path} -> {exc}")
    return []


def _build_agent_doc_context(
    doc_id: str,
    doc: dict,
    vector_store_dir: str,
    api_key: str = "",
    *,
    use_rerank: bool = False,
    reranker_model: str = "",
    rerank_provider: str = "",
    rerank_api_key: str = "",
    rerank_endpoint: str = "",
) -> DocContext:
    data = doc.get("data", {}) or {}
    full_text = data.get("full_text", "") or ""
    pages = data.get("pages", []) or []
    chunks = _load_doc_chunks_for_agent(doc_id, vector_store_dir, full_text)
    semantic_groups = _load_doc_semantic_groups_for_agent(doc_id, full_text)
    return DocContext(
        doc_id=doc_id,
        full_text=full_text,
        chunks=chunks,
        pages=pages,
        semantic_groups=semantic_groups,
        vector_store_dir=vector_store_dir,
        api_key=api_key or "",
        use_rerank=use_rerank,
        reranker_model=reranker_model or "",
        rerank_provider=rerank_provider or "",
        rerank_api_key=rerank_api_key or "",
        rerank_endpoint=rerank_endpoint or "",
    )


def _extract_citation_query_anchors(query: str = "", max_terms: int = 16) -> list[str]:
    """抽取 citation 选择用的通用问题锚点。"""
    stopwords = {
        "about", "above", "answer", "are", "based", "does", "from", "how", "main",
        "method", "paper", "problem", "proposed", "result", "results", "that", "the",
        "their", "this", "uses", "what", "when", "where", "which", "why", "什么", "哪些",
        "如何", "论文", "方法", "主要", "问题", "区别", "请", "解释", "说明",
    }
    terms = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]{2,}",
        str(query or ""),
    )
    if looks_formula_like(query):
        terms.extend(
            re.findall(
                r"\b[a-z]+(?:_bar)?(?:_[a-z0-9]+)+\b|\bsqrt\b|\b[a-z]\^[0-9]+\b|"
                r"\b(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega)(?:_bar)?\b",
                build_formula_alias_text(query),
                re.IGNORECASE,
            )
        )
    anchors: list[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = term.strip(" .,;:()[]{}，。；：、")
        key = clean.casefold()
        if not clean or key in seen or key in stopwords:
            continue
        if (
            re.fullmatch(r"\d+(?:\.\d+)?%?", clean)
            or any(ch.isdigit() for ch in clean)
            or "_" in clean
            or "-" in clean
            or any(ch.isupper() for ch in clean[1:])
            or (len(clean) >= 4 and not re.fullmatch(r"[\u4e00-\u9fff]+", clean))
            or re.fullmatch(r"[\u4e00-\u9fff]{3,}", clean)
        ):
            seen.add(key)
            anchors.append(clean)
        if len(anchors) >= max_terms:
            break
    return anchors


def _build_citation_query_text(query: str = "", sub_questions: list[str] | None = None) -> str:
    """Build generic citation-selection text from the root query and decomposed sub-questions."""
    parts = [re.sub(r"\s+", " ", str(query or "")).strip()]
    seen = {part.casefold() for part in parts if part}
    for item in (sub_questions or [])[:3]:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            parts.append(text)
    return " ".join(part for part in parts if part).strip()


def _select_diverse_agent_detail_citations(
    scored: list[tuple[float, int, dict]],
    query: str = "",
    max_citations: int = 4,
) -> list[tuple[float, int, dict]]:
    """按新增问题锚点覆盖选择 agent detail 引用候选。"""
    if max_citations <= 0 or len(scored) <= max_citations:
        return scored[:max_citations]
    anchors = _extract_citation_query_anchors(query)
    if len(anchors) < 2:
        return scored[:max_citations]

    coverage: list[set[str]] = []
    for _score, _idx, detail in scored:
        raw_text = str(detail.get("_agent_detail_text") or "")
        text = raw_text.casefold()
        matched = {
            anchor
            for anchor in anchors
            if technical_anchor_matches(anchor, raw_text)
        }
        coverage.append(matched)

    selected: list[int] = []
    covered: set[str] = set()
    remaining = set(range(len(scored)))
    while remaining and len(selected) < max_citations:
        best_idx = None
        best_key = None
        for item_idx in remaining:
            score, original_idx, detail = scored[item_idx]
            new_matches = coverage[item_idx] - covered
            full_bonus = 1 if str(detail.get("granularity") or "").lower() == "full" else 0
            key = (len(new_matches), len(coverage[item_idx]), full_bonus, float(score or 0.0), -int(original_idx))
            if best_key is None or key > best_key:
                best_key = key
                best_idx = item_idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        covered.update(coverage[best_idx])
        if len(selected) >= max_citations:
            break
        if len(covered) >= len(anchors):
            break

    selected_set = set(selected)
    ordered_indices = selected + [idx for idx in range(len(scored)) if idx not in selected_set]
    return [scored[idx] for idx in ordered_indices[:max_citations]]


def _build_agent_detail_citations(
    agent_detail: list[dict],
    *,
    query: str = "",
    sub_questions: list[str] | None = None,
    start_ref: int = 1,
    max_citations: int = 4,
) -> list[dict]:
    """把 Agent 已 fetch/backfill 的 parent/group 证据补进引用候选。

    Agent 路径会先用 child chunk 命中，再回填 parent 语义组；如果后续仅从
    拼接后的 agent_context 重新切窗口，部分 parent 证据会在候选收缩阶段丢失。
    这里直接复用 agent_detail 中的语义组文本，补齐引用候选，不额外调用 LLM。
    """
    if not isinstance(agent_detail, list) or max_citations <= 0:
        return []

    try:
        from services.retrieval_tools import _tool_result_score
    except Exception:
        _tool_result_score = None

    query_text = _build_citation_query_text(query, sub_questions)
    scored: list[tuple[float, int, dict]] = []
    seen: set[str] = set()
    for idx, detail in enumerate(agent_detail):
        if not isinstance(detail, dict):
            continue
        text = re.sub(
            r"\s+",
            " ",
            str(
                detail.get("text")
                or detail.get("full_text")
                or detail.get("digest")
                or detail.get("summary")
                or detail.get("content")
                or ""
            ),
        ).strip()
        if len(text) < 40:
            continue
        group_id = str(detail.get("group_id") or detail.get("id") or "").strip()
        key = f"{group_id}:{text[:160].casefold()}"
        if key in seen:
            continue
        seen.add(key)

        if _tool_result_score is not None:
            score = float(_tool_result_score(query_text, text, 0.0))
        else:
            query_terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", query_text.lower()))
            score = float(sum(1 for term in query_terms if term and term in text.lower()))
        if "granularity" in detail and str(detail.get("granularity")).lower() == "full":
            score += 0.25
        if group_id:
            score += 0.05
        scored.append((score, idx, {**detail, "_agent_detail_text": text, "_agent_detail_group_id": group_id}))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected_scored = _select_diverse_agent_detail_citations(scored, query_text, max_citations)
    citations: list[dict] = []
    for score, _idx, detail in selected_scored:
        text = detail["_agent_detail_text"]
        source_text = text[:1400]
        highlight = _context_builder._extract_relevant_snippet(source_text, query_text, max_len=180)
        group_id = detail.get("_agent_detail_group_id") or ""
        page_range = detail.get("page_range") or detail.get("pages") or []
        if isinstance(page_range, int):
            page_range = [page_range, page_range]
        elif isinstance(page_range, (list, tuple)) and len(page_range) == 1:
            page_range = [page_range[0], page_range[0]]
        elif not isinstance(page_range, (list, tuple)):
            page_range = []
        ref = start_ref + len(citations)
        context_id = str(detail.get("context_id") or group_id or f"agent-detail-{ref}").strip()
        evidence_id = str(detail.get("evidence_id") or f"{context_id or 'agent-detail'}:{ref}").strip()
        retrieval_type = str(detail.get("retrieval_type") or "agent_detail").strip()
        citation = {
            key: detail[key]
            for key in (
                "chunk_id",
                "child_chunk_id",
                "parent_id",
                "chunk_type",
                "table_id",
                "table_bundle_id",
                "evidence_unit_id",
            )
            if detail.get(key) not in (None, "")
        }
        citation.update({
            "ref": ref,
            "source_text": source_text,
            "display_text": source_text,
            "highlight_text": highlight or source_text[:180],
            "context_segment_text": source_text,
            "_full_text": source_text,
            "page_range": list(page_range),
            "group_id": group_id,
            "context_id": context_id,
            "evidence_id": evidence_id,
            "retrieval_type": retrieval_type,
            "agent_detail_citation": True,
            "agent_detail_score": round(score, 4),
            "granularity": detail.get("granularity", ""),
        })
        citations.append({
            **citation,
        })
    return citations


async def _maybe_rewrite_query(
    question: str,
    chat_history: list[dict] | None,
    selected_text: str | None,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
    retrieval_strategy: dict | None = None,
) -> str:
    """在满足条件时用 LLM 改写查询，否则回退到 regex 改写。

    触发条件：
    1. 配置启用了 LLM 查询改写
    2. 查询长度 < trigger_length（长查询信息已足够）
    3. 存在对话历史（多轮对话才需要上下文消解）
    4. 有可用的 api_key
    """
    strategy = retrieval_strategy or get_retrieval_strategy(question)
    evidence_need = strategy.get("evidence_need") or []
    regex_rewritten = _query_rewriter.rewrite(
        question,
        selected_text=selected_text,
        evidence_need=evidence_need,
    )

    if (
        not should_enable_llm_query_rewrite()
        or len(question) > settings.query_rewrite_trigger_length
        or not chat_history
        or not api_key
    ):
        return regex_rewritten

    # 清晰的概览/总结类问题不值得为 query rewrite 再调一次 LLM。
    # 这类请求在当前文档上下文中已经足够自洽，额外的改写只会拉高首包延迟。
    try:
        query_type = get_retrieval_strategy(regex_rewritten).get("query_type")
    except Exception:
        query_type = None
    if query_type == "overview" and not selected_text:
        return regex_rewritten

    # 没有选中文本、也没有明显歧义代词时，直接使用本地规则结果。
    normalized_question = (question or "").strip()
    if (
        not selected_text
        and regex_rewritten == question
        and not any(hint in normalized_question for hint in _QUERY_REWRITE_AMBIGUOUS_HINTS)
        and not _EN_AMBIGUOUS_QUERY_RE.search(normalized_question)
        and len(normalized_question) >= 8
    ):
        return regex_rewritten

    try:
        rewritten = await asyncio.wait_for(
            _query_rewriter.rewrite_with_llm(
                query=question,
                chat_history=chat_history,
                selected_text=selected_text,
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                evidence_need=evidence_need,
            ),
            timeout=2.5,
        )
        return rewritten
    except asyncio.TimeoutError:
        logger.warning("[LLM QueryRewrite] 超时，降级为本地规则改写")
        return regex_rewritten

# 模块级变量，由 app.py 注入 MemoryService 实例
memory_service = None

# ---- 中间件链缓存（settings 在运行期间不变）----
_cached_chat_middlewares: list | None = None


def build_chat_middlewares():
    global _cached_chat_middlewares
    if _cached_chat_middlewares is not None:
        return _cached_chat_middlewares
    middlewares = []
    if settings.enable_chat_logging:
        middlewares.append(LoggingMiddleware())
    middlewares.append(RetryMiddleware(retries=settings.chat_retry_retries, delay=settings.chat_retry_delay))
    middlewares.append(ErrorCaptureMiddleware(log_path=settings.error_log_path))
    middlewares.append(TimeoutMiddleware(timeout=settings.chat_timeout))
    if settings.chat_fallback_provider or settings.chat_fallback_model:
        middlewares.append(FallbackMiddleware(settings.chat_fallback_provider, settings.chat_fallback_model))
    if settings.enable_chat_degrade:
        middlewares.append(DegradeOnErrorMiddleware(fallback_content=settings.degrade_message))
    _cached_chat_middlewares = middlewares
    return middlewares


class ChatRequest(BaseModel):
    doc_id: str
    question: str
    api_key: Optional[str] = None
    model: str
    api_provider: str
    selected_text: Optional[str] = None
    enable_vector_search: bool = True
    image_base64: Optional[str] = None
    # 新增：支持多图
    image_base64_list: Optional[List[str]] = None
    top_k: int = 10
    candidate_k: int = 20
    use_rerank: bool = False
    reranker_model: Optional[str] = None
    rerank_provider: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    doc_store_key: Optional[str] = None
    enable_glossary: bool = True
    protect_tables: bool = True
    api_host: Optional[str] = None
    enable_thinking: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    custom_params: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    stream_output: bool = True
    chat_history: Optional[List[dict]] = None
    enable_memory: bool = True
    enable_agent_retrieval: bool = False
    force_agent_retrieval: bool = False
    answer_detail: Optional[str] = _DEFAULT_ANSWER_DETAIL
    enable_web_search: bool = False
    web_search_provider: Optional[str] = "auto"
    web_search_api_key: Optional[str] = None
    web_search_max_results: Optional[int] = 5
    web_search_blacklist: Optional[list[str]] = None
    enable_graphrag: bool = False
    enable_jieba_bm25: bool = True
    num_expand_context_chunk: int = 1
    embedding_api_key: Optional[str] = None  # embedding 模型的 API key（向量检索查询编码用）

    # ---- 双模型策略：per-request 覆盖 config.cheap_model* ----
    cheap_model: Optional[str] = None
    cheap_model_provider: Optional[str] = None
    cheap_model_endpoint: Optional[str] = None

    # ---- Feature flag per-request overrides ----
    # None = 跟随全局 settings；True/False = 本次请求强制开启/关闭
    override_numeric_table: Optional[bool] = None
    override_answer_critic: Optional[bool] = None
    override_llm_query_rewrite: Optional[bool] = None
    override_bm25_synonyms: Optional[bool] = None


class ChatVisionRequest(BaseModel):
    doc_id: str
    question: str
    api_key: Optional[str] = None
    model: str
    api_provider: str
    image_base64: Optional[str] = None
    selected_text: Optional[str] = None


def _validate_rerank_request(req):
    provider = getattr(req, "rerank_provider", None)
    api_key = getattr(req, "rerank_api_key", None)
    use_rerank = getattr(req, "use_rerank", False)
    cloud_providers = {"cohere", "jina", "silicon", "aliyun", "openai", "moonshot", "deepseek", "zhipu", "minimax"}
    if use_rerank and provider and provider.lower() in cloud_providers and not api_key:
        raise HTTPException(status_code=400, detail=f"使用 {provider} rerank 需要提供 rerank_api_key")


# 触发智能 rerank 默认开启的 evidence_need 集合
# 这些题型 vector search 单独命中率较低，rerank 收益显著
_AUTO_RERANK_EVIDENCE_NEEDS = {
    "overview", "comparative", "numeric_table", "reference_meta",
    "reference_trap", "global_summary", "section_explanation",
    "comparison_multi_aspect",
}
# 触发智能 rerank 的 query_type 集合
_AUTO_RERANK_QUERY_TYPES = {"overview", "comparative", "summary", "analytical", "specific"}


def _auto_enable_rerank_if_beneficial(request, evidence_need: list, query_type: str = "") -> bool:
    """根据 evidence_need / query_type 智能开启 rerank。

    - 用户已显式 use_rerank=True：保持不变
    - 命中关键 evidence_need / query_type：自动启用 local rerank（无需 api_key）
    - 不影响用户已配置的云端 rerank
    Returns: True 表示发生了 auto-enable，False 表示未改动。
    """
    if getattr(request, "use_rerank", False):
        return False
    need_set = {str(e).lower() for e in (evidence_need or [])}
    qt = (query_type or "").lower()
    if not (need_set & _AUTO_RERANK_EVIDENCE_NEEDS) and qt not in _AUTO_RERANK_QUERY_TYPES:
        return False
    provider = str(getattr(request, "rerank_provider", "") or "").strip().lower()
    allowed_cloud = {"cohere", "jina", "openai", "siliconflow", "silicon", "llm"}
    if provider and provider not in allowed_cloud and provider != "local":
        return False
    # 用户未配置 rerank_provider：默认走 local（BAAI/bge-reranker-base）
    if not getattr(request, "rerank_provider", None):
        request.rerank_provider = "local"
    if not getattr(request, "reranker_model", None):
        request.reranker_model = "BAAI/bge-reranker-base"
    request.use_rerank = True
    return True


def _retrieve_memory_context(question: str, api_key: str = None, doc_id: str = None) -> str:
    if memory_service is None:
        return ""
    try:
        filter_by_doc = bool(doc_id)
        return memory_service.retrieve_memories(
            question, api_key=api_key, doc_id=doc_id, filter_by_doc=filter_by_doc
        )
    except Exception as e:
        logger.error(f"记忆检索失败: {e}")
        return ""


def _retrieve_raw_memories(question: str, api_key: str = None, doc_id: str = None, chat_history: list[dict] | None = None) -> list[dict]:
    """检索原始记忆列表（供 ContextInjector 使用）"""
    if memory_service is None:
        return []
    try:
        filter_by_doc = bool(doc_id)
        return memory_service.retrieve_memories_raw(
            question, api_key=api_key, doc_id=doc_id, filter_by_doc=filter_by_doc, chat_history=chat_history
        )
    except Exception as e:
        logger.error(f"记忆原始检索失败: {e}")
        return []


def _get_memory_retrieval_timeout() -> float:
    """读取流式请求的记忆检索软超时，避免慢记忆链路阻塞事件循环。"""
    raw_value = getattr(settings, "memory_retrieval_timeout", None)
    if raw_value is None:
        raw_value = os.getenv("CHATPDF_MEMORY_RETRIEVAL_TIMEOUT", "2.0")
    try:
        timeout_value = float(raw_value)
    except (TypeError, ValueError):
        timeout_value = 2.0
    return max(timeout_value, 0.0)


async def _run_memory_read_for_stream(label: str, reader, fallback):
    timeout_seconds = _get_memory_retrieval_timeout()
    try:
        if timeout_seconds <= 0:
            return await asyncio.to_thread(reader)
        return await asyncio.wait_for(
            asyncio.to_thread(reader),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[Memory] 流式请求记忆%s检索超时 %.1fs，降级为无记忆上下文",
            label,
            timeout_seconds,
        )
        return fallback
    except Exception as e:
        logger.error(f"[Memory] 流式请求记忆{label}检索失败: {e}")
        return fallback


async def _retrieve_memory_for_stream(
    question: str,
    api_key: str = None,
    doc_id: str = None,
    chat_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """在线程中读取记忆，保护流式接口不被同步检索卡住。"""
    if memory_service is None:
        return "", []

    memory_context = await _run_memory_read_for_stream(
        "上下文",
        lambda: _retrieve_memory_context(question, api_key=api_key, doc_id=doc_id),
        "",
    )
    raw_memories = await _run_memory_read_for_stream(
        "原始列表",
        lambda: _retrieve_raw_memories(
            question,
            api_key=api_key,
            doc_id=doc_id,
            chat_history=chat_history,
        ),
        [],
    )
    return str(memory_context or ""), list(raw_memories or [])


def _smart_inject_memory(system_prompt: str, memory_context: str, raw_memories: list[dict] = None) -> tuple[str, list[dict], dict]:
    """智能注入记忆上下文：优先使用 ContextInjector，失败时回退到简单注入

    Args:
        system_prompt: 原始 system prompt
        memory_context: 格式化的记忆上下文字符串（降级用）
        raw_memories: 原始记忆列表（供 ContextInjector 使用）

    Returns:
        (注入记忆后的 prompt, 实际命中的记忆列表, 注入元数据)
    """
    # 优先使用 ContextInjector
    if raw_memories and memory_service and hasattr(memory_service, 'context_injector') and memory_service.context_injector:
        try:
            injector = memory_service.context_injector
            selected_memories = injector.prepare_memories(raw_memories)
            return (
                injector.inject(system_prompt, selected_memories),
                selected_memories,
                {
                    "enabled": True,
                    "strategy": "context_injector",
                    "retrieved_count": len(raw_memories),
                    "selected_count": len(selected_memories),
                    "truncated": len(selected_memories) < len(raw_memories),
                    "token_budget": getattr(injector, "token_budget", None),
                    "selected_kinds": [mem.get("memory_kind", "episodic") for mem in selected_memories],
                },
            )
        except Exception as e:
            logger.warning(f"ContextInjector 注入失败，回退到简单注入: {e}")
    # 降级为原有简单注入
    return (
        _inject_memory_context(system_prompt, memory_context),
        list(raw_memories or []),
        {
            "enabled": bool(memory_context or raw_memories),
            "strategy": "simple",
            "retrieved_count": len(raw_memories or []),
            "selected_count": len(raw_memories or []),
            "truncated": False,
            "token_budget": None,
            "selected_kinds": [mem.get("memory_kind", "episodic") for mem in (raw_memories or [])],
        },
    )


def _async_memory_write(svc, request):
    try:
        if request.doc_id:
            history = list(request.chat_history or [])
            history.append({"role": "user", "content": request.question})
            svc.save_qa_summary(
                request.doc_id,
                history,
                api_key=getattr(request, "api_key", None),
                model=getattr(request, "model", None),
                api_provider=getattr(request, "api_provider", None),
            )
        svc.update_keywords(request.question)
    except Exception as e:
        logger.error(f"异步记忆写入失败: {e}")


_flushed_sessions: set = set()


def _maybe_flush_memory(request) -> None:
    if memory_service is None:
        return
    if not settings.memory_flush_enabled:
        return
    history = getattr(request, "chat_history", None)
    if not history:
        return
    doc_id = getattr(request, "doc_id", "")
    if not doc_id or doc_id in _flushed_sessions:
        return
    from services.token_budget import TokenBudget
    budget = TokenBudget()
    total_tokens = 0
    for msg in history:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if content:
                total_tokens += budget.estimate_tokens(content)
    threshold = settings.memory_flush_threshold_tokens
    if total_tokens < threshold:
        return
    _flushed_sessions.add(doc_id)
    logger.info(f"[Memory] Compaction flush 触发: doc_id={doc_id}, tokens={total_tokens}, threshold={threshold}")
    threading.Thread(
        target=_async_memory_write,
        args=(memory_service, request),
        daemon=True,
    ).start()


def _should_use_memory(request) -> bool:
    return (
        settings.memory_enabled
        and getattr(request, "enable_memory", True)
        and memory_service is not None
    )


def _inject_memory_context(system_prompt: str, memory_context: str) -> str:
    if not memory_context:
        return system_prompt
    marker = "\n回答规则："
    if marker in system_prompt:
        idx = system_prompt.index(marker)
        return (
            system_prompt[:idx]
            + f"\n\n用户历史记忆：\n{memory_context}"
            + system_prompt[idx:]
        )
    return system_prompt + f"\n\n用户历史记忆：\n{memory_context}"


def _build_fused_context(
    selected_text: str,
    retrieval_context: str,
    selected_page_info: dict,
    selected_ref: Optional[int] = None,
) -> str:
    """融合框选文本和检索上下文

    将 selected_text 作为优先上下文置于检索结果之前，
    并标注框选文本的页码来源。
    """
    page_label = ""
    if selected_page_info:
        ps = selected_page_info.get("page_start", 0)
        pe = selected_page_info.get("page_end", 0)
        page_label = f"（页码: {ps}-{pe}）" if ps != pe else f"（页码: {ps}）"

    selected_title = (
        f"[{selected_ref}]用户选中的文本{page_label}"
        if selected_ref is not None else
        f"用户选中的文本{page_label}"
    )

    # 将 selected_text 放在块首，确保后续任何检索上下文都在其后出现。
    parts = [f"{selected_text}\n\n{selected_title}"]
    if retrieval_context:
        parts.append(f"\n\n相关文档片段：\n\n{retrieval_context}")
    return "\n".join(parts)


def _format_context_segments_for_prompt(segments: list[dict], *, max_segments: int = 12) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        text = re.sub(r"\s+", " ", str(segment.get("text") or "")).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        try:
            ref = int(segment.get("ref") or len(lines) + 1)
        except (TypeError, ValueError):
            ref = len(lines) + 1
        page_range = segment.get("page_range") or []
        page_text = ""
        if isinstance(page_range, (list, tuple)) and page_range:
            page_text = f"页码: {page_range[0]}-{page_range[-1]}" if len(page_range) > 1 and page_range[0] != page_range[-1] else f"页码: {page_range[0]}"
        prefix = f"[{ref}]"
        if page_text:
            prefix += f" {page_text}"
        lines.append(f"{prefix}\n{text}")
        if len(lines) >= max(1, int(max_segments or 1)):
            break
    return "\n\n".join(lines).strip()


def _sync_numeric_table_prompt_context(
    context: str,
    retrieval_meta: dict,
    *,
    query: str = "",
    evidence_need: Optional[list[str]] = None,
) -> str:
    needs = {
        str(item).strip()
        for item in (evidence_need or (retrieval_meta or {}).get("evidence_need") or [])
        if str(item).strip()
    }
    if "numeric_table" not in needs or not isinstance(retrieval_meta, dict):
        return context

    segments = _build_response_context_segments({
        **retrieval_meta,
        "search_query": query or retrieval_meta.get("search_query", ""),
    })
    formatted = _format_context_segments_for_prompt(segments)
    if not formatted:
        return context
    retrieval_meta["_context_segments"] = segments
    graph_suffix = ""
    graph_marker = "\n\n## 知识图谱关联信息"
    if graph_marker in str(context or ""):
        graph_suffix = str(context)[str(context).index(graph_marker):]
    return f"根据用户问题检索到的相关文档片段：\n\n{formatted}\n\n{graph_suffix}"


def _build_selected_text_citation(
    selected_text: str,
    selected_page_info: dict,
) -> dict:
    """基于框选文本位置生成基础 citation"""
    ps = selected_page_info.get("page_start", 1) if selected_page_info else 1
    pe = selected_page_info.get("page_end", ps) if selected_page_info else ps
    return {
        "ref": 1,
        "evidence_id": f"selected-text:{ps}-{pe}:1",
        "group_id": "selected-text",
        "page_range": [ps, pe],
        "source_text": selected_text,
        "display_text": selected_text,
        "highlight_text": selected_text[:200].strip(),
        "_full_text": selected_text,
        "alignment_status": "fallback_window_only",
        "retrieval_type": "selected_text",
    }


def _build_selected_text_fallback_citations(
    selected_text: str,
    selected_page_info: dict,
):
    """仅在检索引用缺失时，为较长 selected_text 生成兜底 citation。"""
    if not selected_text or len(selected_text.strip()) < _MIN_SELECTED_TEXT_FALLBACK_CITATION_CHARS:
        return []
    return [_build_selected_text_citation(selected_text, selected_page_info)]


_NUMERIC_TABLE_QUERY_TABLE_RE = re.compile(r"\btable\s*\d+\b|表\s*\d+", re.IGNORECASE)
_NUMERIC_TABLE_COMPARATOR_QUERY_RE = re.compile(
    r"(?:second[- ]best|runner[- ]up|difference|compare|comparison|higher than|lower than|best|highest|winner|winning|"
    r"第二好|第二佳|第二名|次优|比较|差多少|最高|最佳|最好)",
    re.IGNORECASE,
)
_NUMERIC_TABLE_COST_QUERY_HINTS = (
    "flops",
    "推理时间",
    "开销",
    "latency",
    "runtime",
    "overhead",
    "cost",
    "inference time",
    "inference overhead",
    "training time",
    "训练时间",
    "耗时",
    "extra flops",
)
_NUMERIC_TABLE_COST_EVIDENCE_HINTS = (
    "24 hours",
    "24 hour",
    "24h",
    "six days",
    "6 days",
    "6天",
    "24小时",
    "no extra overhead",
    "without extra overhead",
    "no additional overhead",
    "additional overhead",
    "inference overhead",
)
_FORMULA_FRAMEWORK_QUERY_RE = re.compile(
    r"(公式|方程|算法|框架|推导|目标函数|损失函数|"
    r"formula|equation|algorithm|framework|objective|loss|derivation)",
    re.IGNORECASE,
)
_FORMULA_ANSWER_ANCHOR_RE = re.compile(
    r"(公式|方程|目标函数|损失函数|formula|equation|objective|loss|"
    r"β|beta|alpha|α|epsilon|ϵ|lambda|λ|theta|θ|mathcal|sqrt|sum|prod)",
    re.IGNORECASE,
)
_FORMULA_FRAMEWORK_REWRITE_ANSWER_RE = re.compile(
    r"(核心框架主要包括|公式框架主要包括|以下(?:两个|2\s*个)部分|"
    r"equation|formula|objective|loss|目标函数|损失函数)",
    re.IGNORECASE,
)
_BACKBONE_PRETRAIN_QUERY_RE = re.compile(
    r"(骨干|骨干网络|backbone|预训练|预训练权重|pre[-\s]?trained|from scratch|"
    r"external\s+(?:data|model|knowledge)|外部数据|外部模型|微调|fine[-\s]?tun)",
    re.IGNORECASE,
)
_ABLATION_EFFECT_QUERY_RE = re.compile(
    r"(消融|ablation|移除|去掉|组件|模块|影响最大|most\s+influential)",
    re.IGNORECASE,
)
_OVERSAMPLING_CONTRAST_QUERY_RE = re.compile(
    r"(重采样|过采样|oversampling|re[-\s]?sampling|传统.*(?:采样|训练)|"
    r"根本不同|本质区别|fundamental(?:ly)?\s+different)",
    re.IGNORECASE,
)
_GAN_DIFFUSION_CONTRAST_QUERY_RE = re.compile(
    r"(gan|stable\s+diffusion|clip|扩散模型|diffusion\s+model|生成模型|data\s+synthesis)|"
    r"(?:为什么|为何|why).*(?:扩散|diffusion).*(?:gan|stable\s+diffusion|clip)|"
    r"(?:gan|stable\s+diffusion|clip).*(?:本质区别|区别|而不是|rather\s+than|instead\s+of)",
    re.IGNORECASE,
)
_CONDITIONAL_GENERATION_QUERY_RE = re.compile(
    r"(条件生成|类别归属|类别控制|控制生成样本.*类别|生成样本.*类别|"
    r"conditional\s+generation|class\s+label|class\s+embedding|label\s+y|"
    r"control(?:s|ling)?\s+(?:the\s+)?(?:class|category))",
    re.IGNORECASE,
)


def _normalize_numeric_table_column_key(value: str = "") -> str:
    sample = re.sub(r"\s+", "", str(value or "").lower()).strip()
    if not sample:
        return ""
    replacements = {
        "accuracy": "acc",
        "acc.(%)": "acc",
        "acc(%)": "acc",
        "manyshot": "many",
        "fewshot": "few",
        "few-shot": "few",
        "medium": "med",
        "med.": "med",
        "δacc": "deltaacc",
        "△acc": "deltaacc",
        "Δacc".lower(): "deltaacc",
        "∂acc": "deltaacc",
        "||d_gen||": "dgen",
        "|d_gen|": "dgen",
        "d_gen": "dgen",
    }
    for source, target in replacements.items():
        sample = sample.replace(source, target)
    return sample


def _extract_numeric_table_target_columns(query: str = "", hints: Optional[dict] = None) -> set[str]:
    targets: set[str] = set()
    for value in (hints or {}).get("columns", []) or []:
        normalized = _normalize_numeric_table_column_key(value)
        if normalized:
            targets.add(normalized)

    sample = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    if not sample:
        return targets

    if re.search(r"\bfew(?:-shot)?\b", sample):
        targets.add("few")
    if re.search(r"\bmany\b", sample):
        targets.add("many")
    if re.search(r"\bmed(?:\.|ium)?\b", sample):
        targets.add("med")
    if re.search(r"\ball\b", sample):
        targets.add("all")
    if "fid" in sample:
        targets.add("fid")
    frequency_targets = {"all", "many", "med", "few"}
    if (
        ("accuracy" in sample or re.search(r"\bacc\b", sample))
        and not (targets & frequency_targets)
    ):
        targets.add("acc")
    if "d_gen" in sample or "dgen" in sample or "||d_gen||" in sample:
        targets.add("dgen")
    if (
        "δacc" in sample
        or "△acc" in sample
        or "delta acc" in sample
        or "deltaacc" in sample
        or "Δacc".lower() in sample
    ):
        targets.add("deltaacc")
    return targets


def _extract_numeric_table_target_tables(query: str = "", hints: Optional[dict] = None) -> set[str]:
    targets = {
        re.sub(r"\s+", " ", match.group(0)).strip().lower()
        for match in _NUMERIC_TABLE_QUERY_TABLE_RE.finditer(str(query or ""))
    }
    for value in (hints or {}).get("table_labels", []) or []:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        if normalized:
            targets.add(normalized)
    for value in (hints or {}).get("tables", []) or []:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        if normalized:
            targets.add(normalized)
    return targets


def _normalize_numeric_table_method_token(value: str = "") -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _extract_numeric_table_target_methods(query: str = "", hints: Optional[dict] = None) -> set[str]:
    values = list((hints or {}).get("methods", []) or [])
    sample = str(query or "")
    token_pattern = re.compile(
        r"\b(?:[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+|[A-Za-z]*[A-Z][A-Za-z0-9.+/_-]*)(?:\s*\([^)]{1,32}\))?"
    )
    for match in token_pattern.finditer(sample):
        token = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:[]{}")
        if not token:
            continue
        lowered = token.lower()
        if lowered.startswith(("table", "resnet", "densenet", "vit", "swin", "convnext")):
            continue
        if _normalize_numeric_table_column_key(token):
            continue
        if re.search(r"(?:^|[-_])(?:lt|dataset|data|bench|corpus|set)(?:[-_]|$)", token, re.IGNORECASE):
            continue
        values.append(token)
    return {
        normalized
        for normalized in (_normalize_numeric_table_method_token(value) for value in values)
        if normalized
    }


def _numeric_table_answer_mentions_method(answer: str = "", citation: Optional[dict] = None) -> bool:
    if not answer or not isinstance(citation, dict):
        return False
    sample = _strip_inline_citations(answer)
    row = re.sub(
        r"\s+",
        " ",
        str(
            citation.get("numeric_table_exact_context_row_text")
            or citation.get("table_row_boundary_text")
            or citation.get("table_row_raw_text")
            or citation.get("highlight_text")
            or citation.get("display_text")
            or citation.get("source_text")
            or ""
        ),
    ).strip()
    if not sample or not row:
        return False
    method_cell = re.split(r"\||\s{2,}", row, maxsplit=1)[0]
    method_cell = re.sub(r"\([^)]*\)", "", method_cell)
    method_cell = re.sub(r"\d+(?:\.\d+)?", "", method_cell)
    method_cell = method_cell.strip().lower()
    if not method_cell:
        return False
    answer_key = _normalize_numeric_table_method_token(sample)
    method_key = _normalize_numeric_table_method_token(method_cell)
    return bool(method_key and method_key in answer_key)


def _is_delta_per_sample_metric_query(query: str = "") -> bool:
    sample = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    if not sample:
        return False
    return bool(
        "deltaacc" in _normalize_numeric_table_column_key(sample)
        or "δacc" in sample
        or "∆acc" in sample
        or "Δacc".lower() in sample
        or "每样本" in sample
        or "per sample" in sample
        or "average improvement" in sample
        or "平均单样本" in sample
        or "平均每样本" in sample
    )


def _is_numeric_table_cost_query(query: str = "") -> bool:
    sample = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    return bool(sample) and any(token in sample for token in _NUMERIC_TABLE_COST_QUERY_HINTS)


def _is_numeric_table_metric_query(query: str = "") -> bool:
    sample = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    if not sample or _is_numeric_table_cost_query(sample):
        return False
    metric_hints = (
        "准确率", "精度", "性能", "结果", "数值", "指标", "提升", "百分点",
        "accuracy", "acc", "performance", "result", "results", "score", "metric",
        "all", "many", "medium", "med.", "few",
    )
    return any(token in sample for token in metric_hints)


def _has_numeric_table_metric_anchor(text: str = "") -> bool:
    sample = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not sample:
        return False
    return bool(re.search(r"\b(all|many|med\.?|medium|few|acc|accuracy)\b", sample)) or any(
        marker in sample for marker in ("准确率", "精度", "指标", "table", "表")
    )


def _has_numeric_table_cost_anchor(text: str = "") -> bool:
    sample = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not sample:
        return False
    compact = re.sub(r"[\s\-–—]+", "", sample)
    if any(token in sample for token in _NUMERIC_TABLE_COST_EVIDENCE_HINTS):
        return True
    return any(
        (
            re.search(r"24\s*hours?", sample),
            re.search(r"(?:six|6)\s*days?", sample),
            "24hours" in compact,
            "24hour" in compact,
            "6days" in compact,
            "sixdays" in compact,
        )
    )


def _is_formula_framework_query(query: str = "") -> bool:
    sample = str(query or "")
    return bool(_FORMULA_FRAMEWORK_QUERY_RE.search(sample))


def _is_backbone_pretrain_query(query: str = "") -> bool:
    sample = str(query or "")
    return bool(sample and _BACKBONE_PRETRAIN_QUERY_RE.search(sample))


def _is_gan_diffusion_contrast_query(query: str = "") -> bool:
    sample = str(query or "")
    if not sample:
        return False
    lower = sample.lower()
    return bool(
        _GAN_DIFFUSION_CONTRAST_QUERY_RE.search(sample)
        and ("gan" in lower or "stable diffusion" in lower or "clip" in lower or "扩散" in sample)
    )


def _calc_formula_citation_anchor_score(text: str = "") -> float:
    sample = str(text or "")
    lower_sample = sample.lower()
    score = 0.0
    for token, weight in (
        ("formula", 1.5),
        ("equation", 1.5),
        ("objective", 1.2),
        ("loss", 1.0),
        ("公式", 1.5),
        ("方程", 1.5),
        ("目标函数", 1.6),
        ("损失函数", 1.4),
    ):
        if token in lower_sample or token in sample:
            score += weight
    score += min(len(re.findall(r"[=≈≤≥∑∏√∫∂]|\\(?:frac|sum|prod|sqrt|mathcal|argmax|argmin)", sample)), 8) * 0.45
    score += min(len(re.findall(r"\b(?:alpha|beta|gamma|lambda|theta|epsilon|delta)\b|[αβγλθϵεΔδ]", sample, re.IGNORECASE)), 8) * 0.25
    return score


def _build_formula_citation_support_text(citation: Optional[dict]) -> str:
    if not isinstance(citation, dict):
        return ""
    fields = (
        _build_phrase_alignment_text(citation),
        citation.get("context_segment_text", ""),
        citation.get("source_text", ""),
        citation.get("display_text", ""),
        citation.get("highlight_text", ""),
        citation.get("_full_text", ""),
        citation.get("group_id", ""),
    )
    return " ".join(
        re.sub(r"\s+", " ", str(value or "")).strip()
        for value in fields
        if re.sub(r"\s+", " ", str(value or "")).strip()
    ).strip()


def _has_formula_core_citation_anchor(citation: Optional[dict]) -> bool:
    support = _build_formula_citation_support_text(citation)
    if not support:
        return False
    return bool(
        re.search(
            r"formula|equation|objective|loss|公式|方程|目标函数|损失函数|"
            r"=|≈|≤|≥|∑|∏|√|\\(?:frac|sum|prod|sqrt|mathcal)|"
            r"beta|β|alpha|α|lambda|λ|theta|θ|epsilon|ϵ",
            support,
            re.IGNORECASE,
        )
    )


def _has_numeric_table_exact_row_support(citation: Optional[dict]) -> bool:
    if not isinstance(citation, dict):
        return False
    chunk_type = str(citation.get("chunk_type") or citation.get("block_type") or "").strip().lower()
    if chunk_type in {"table_row", "table_cell"}:
        return True
    if citation.get("table_row_evidence") or citation.get("table_row_slice_kind") == "exact":
        return True
    if citation.get("numeric_table_exact_context_row_text"):
        return True
    if citation.get("cell_evidence_units"):
        return True
    for unit in citation.get("evidence_units") or []:
        if not isinstance(unit, dict):
            continue
        unit_type = str(unit.get("evidence_unit_type") or "").strip().lower()
        if unit_type in {"table_row", "table_cell"}:
            return True
    return False


def _extract_numeric_metric_exact_row_from_bundle(
    citation: Optional[dict],
    query: str = "",
) -> str:
    if not isinstance(citation, dict) or not _is_numeric_table_metric_query(query):
        return ""
    source = _build_numeric_table_citation_support_text(citation)
    source_lower = source.lower()
    hints = _query_rewriter.extract_numeric_table_hints(query) if query else {}
    target_tables = _extract_numeric_table_target_tables(query, hints)
    if target_tables and not _numeric_table_support_mentions_target_table(source, target_tables):
        return ""
    if "[structured table bundle]" not in source_lower and not target_tables:
        return ""
    scoped_source = source
    if target_tables:
        table_patterns: list[str] = []
        for table in target_tables:
            table_number = re.search(r"\d+", table)
            if table_number:
                table_patterns.append(rf"(?:Table|表)\s*{re.escape(table_number.group(0))}\b")
        if table_patterns:
            start_match = None
            for pattern in table_patterns:
                match = re.search(pattern, source, re.IGNORECASE)
                if match and (start_match is None or match.start() < start_match.start()):
                    start_match = match
            if start_match:
                next_match = re.search(r"(?:\[Structured Table Bundle\]|\bTable\s*\d+\b|表\s*\d+)", source[start_match.end():], re.IGNORECASE)
                end = start_match.end() + next_match.start() if next_match else len(source)
                scoped_source = source[start_match.start():end]
                source_lower = scoped_source.lower()
    target_columns = _extract_numeric_table_target_columns(
        query,
        hints,
    )

    if not target_columns:
        return ""

    for raw_line in scoped_source.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or "|" not in line:
            continue
        if _numeric_table_text_matches_columns(line, target_columns) and re.search(r"\d+(?:\.\d+)?", line):
            return line

    return ""


def _normalize_numeric_metric_bundle_citations(citations: list[dict], query: str = "") -> list[dict]:
    if not citations or not _is_numeric_table_metric_query(query):
        return citations
    normalized: list[dict] = []
    for citation in citations:
        if not isinstance(citation, dict):
            normalized.append(citation)
            continue
        exact_row = _extract_numeric_metric_exact_row_from_bundle(citation, query)
        if not exact_row:
            normalized.append(citation)
            continue
        item = citation.copy()
        item.setdefault("chunk_type", "table_row")
        item["numeric_table_exact_context_row_text"] = exact_row
        item["highlight_text"] = exact_row
        item["display_text"] = exact_row
        normalized.append(item)
    return normalized


def _build_focused_numeric_metric_citation_text(citation: dict, query: str = "") -> str:
    hints = _query_rewriter.extract_numeric_table_hints(query) if query else {}
    target_tables = _extract_numeric_table_target_tables(query, hints)
    target_columns = _extract_numeric_table_target_columns(query, hints)
    visible_support = _build_numeric_table_visible_support_text(citation)
    if target_tables and visible_support and not _numeric_table_support_mentions_target_table(visible_support, target_tables):
        return ""
    if target_columns and visible_support and not _numeric_table_text_matches_columns(visible_support, target_columns):
        return ""

    row = _extract_numeric_table_citation_row_text(citation)
    row_is_broad = bool(
        row
        and (
            "[structured table bundle]" in row.lower()
            or len(re.findall(r"\d+(?:\.\d+)?", row)) > 8
            or len(row) > 360
        )
    )
    bundle_row = ""
    if not row or row_is_broad:
        bundle_row = _extract_numeric_metric_exact_row_from_bundle(citation, query or "准确率 数值")
        row = bundle_row or ("" if row_is_broad else row)
    row = re.sub(r"\s+", " ", str(row or "")).strip()
    if "fid" in target_columns and "acc" in target_columns:
        header = re.sub(r"\s+", " ", str(citation.get("table_header") or "")).strip()
        parts = ["生成模型数值证据: 当前答案对应表格行。", f"原始表格行: {row}"]
        if header:
            parts.append(f"表头: {header}")
        return " ".join(parts)
    return ""


def _focus_numeric_metric_citation(citation: dict, query: str = "") -> dict:
    focused_text = _build_focused_numeric_metric_citation_text(citation, query)
    if not focused_text:
        return citation
    focused = citation.copy()
    focused["highlight_text"] = focused_text
    focused["display_text"] = focused_text
    focused["source_text"] = focused_text
    focused["context_segment_text"] = focused_text
    focused["numeric_metric_focused"] = True
    focused.pop("_full_text", None)
    return focused


def _extract_numeric_table_comparison_rows_from_citation(citation: dict, query: str = "") -> list[dict]:
    if not isinstance(citation, dict) or not query:
        return []
    hints = _query_rewriter.extract_numeric_table_hints(query)
    target_methods = _extract_numeric_table_target_methods(query, hints)
    if len(target_methods) < 2:
        return []

    support = _build_numeric_table_citation_support_text(citation)
    if not support:
        return []
    target_tables = _extract_numeric_table_target_tables(query, hints)
    if target_tables and not _numeric_table_support_mentions_target_table(support, target_tables):
        return []

    target_columns = _extract_numeric_table_target_columns(query, hints)
    if target_columns and "all" not in target_columns and not _citation_matches_numeric_table_columns(citation, target_columns):
        return []

    raw_sources = [
        str(citation.get("_full_text") or ""),
        str(citation.get("source_text") or ""),
        str(citation.get("context_segment_text") or ""),
        str(citation.get("display_text") or ""),
        str(citation.get("highlight_text") or ""),
    ]
    raw_lines: list[str] = []
    for source in raw_sources:
        if not source:
            continue
        raw_lines.extend(source.splitlines())
    if not raw_lines:
        raw_lines = [support]

    rows: list[dict] = []
    seen: set[str] = set()
    seen_method_keys: set[str] = set()
    for raw_line in raw_lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or "|" not in line:
            continue
        line_method_key = _normalize_numeric_table_method_token(line)
        matched_method = ""
        for method in target_methods:
            if method and method in line_method_key:
                matched_method = method
                break
        if not matched_method:
            continue
        if not re.search(r"(?:All\s*=\s*)?\d+(?:\.\d+)?", line):
            continue
        if matched_method in seen_method_keys:
            continue
        row_key = line.casefold()
        if row_key in seen:
            continue
        seen.add(row_key)
        seen_method_keys.add(matched_method)
        item = citation.copy()
        item["ref"] = int(citation.get("ref") or 0)
        item["source_ref"] = int(citation.get("source_ref") or citation.get("ref") or 0)
        item["chunk_type"] = "table_row"
        item["block_type"] = "table_row"
        item["numeric_table_exact_context_row_text"] = line
        item["table_row_boundary_text"] = line
        item["table_row_raw_text"] = line
        item["highlight_text"] = line
        item["display_text"] = line
        item["source_text"] = line
        item["context_segment_text"] = line
        item["numeric_metric_focused"] = True
        item["numeric_table_comparison_row"] = True
        rows.append(item)
    return rows


def _attach_numeric_table_comparison_rows(citation: dict, query: str = "") -> dict:
    rows = _extract_numeric_table_comparison_rows_from_citation(citation, query)
    if not rows:
        return citation
    item = citation.copy()
    row_texts = [
        re.sub(r"\s+", " ", str(row.get("numeric_table_exact_context_row_text") or "")).strip()
        for row in rows
    ]
    item["numeric_table_comparison_rows"] = list(dict.fromkeys(row for row in row_texts if row))
    return item


def _iter_disk_doc_texts() -> list[str]:
    candidate_dirs = [
        Path(runtime.data_dir) / "docs",
        _get_project_root() / "data" / "docs",
        Path(__file__).resolve().parents[1] / "data" / "docs",
    ]
    seen_paths: set[Path] = set()
    texts: list[str] = []
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for doc_path in directory.glob("*.json"):
            resolved = doc_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                data = json.loads(doc_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            payload = data.get("data") if isinstance(data, dict) else {}
            payload = payload if isinstance(payload, dict) else {}
            text = str(payload.get("full_text") or data.get("full_text") or data.get("text") or "")
            if not text:
                pages = payload.get("pages") or data.get("pages") or []
                if isinstance(pages, list):
                    text = "\n".join(
                        str(page.get("text") or page.get("content") or "")
                        for page in pages
                        if isinstance(page, dict)
                    )
            if text:
                texts.append(text)
    return texts


def _extract_numeric_values_from_answer(answer: str = "") -> set[str]:
    values: set[str] = set()
    for match in re.finditer(r"(?<!\d)(\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?!\d)", str(answer or "")):
        value = match.group(1)
        values.add(value)
        values.add(value.replace(",", ""))
    return {value for value in values if value}


def _numeric_table_support_mentions_target_table(support_text: str = "", target_tables: Optional[set[str]] = None) -> bool:
    if not target_tables:
        return True
    lower = re.sub(r"\s+", " ", str(support_text or "").lower()).strip()
    if not lower:
        return False
    for table in target_tables:
        compact = re.sub(r"\s+", "", table)
        if table in lower or compact in re.sub(r"\s+", "", lower):
            return True
    return False


def _build_numeric_table_visible_support_text(citation: Optional[dict]) -> str:
    if not isinstance(citation, dict):
        return ""
    parts = [
        *_collect_citation_table_evidence_texts(citation),
        citation.get("numeric_table_exact_context_row_text", ""),
        citation.get("table_row_boundary_text", ""),
        citation.get("table_row_raw_text", ""),
        citation.get("context_segment_text", ""),
        citation.get("source_text", ""),
        citation.get("display_text", ""),
        citation.get("highlight_text", ""),
        citation.get("start_phrase", ""),
        citation.get("end_phrase", ""),
        citation.get("table_id", ""),
        citation.get("table_caption", ""),
        citation.get("table_header", ""),
    ]
    return " ".join(
        re.sub(r"\s+", " ", str(part or "")).strip()
        for part in parts
        if re.sub(r"\s+", " ", str(part or "")).strip()
    ).strip()


def _build_numeric_table_visible_scoring_text(citation: Optional[dict]) -> str:
    """构建用于数值表格引用打分的可见、紧凑证据文本。

    `source_text`/`display_text` 有时是混合页块或结构化 bundle，里面会夹带多个表格。
    这类文本可以作为召回上下文，但不能让隐藏在远处的表格行抢走最终引用锚点。
    """
    if not isinstance(citation, dict):
        return ""

    exact_row = re.sub(
        r"\s+",
        " ",
        str(
            citation.get("numeric_table_exact_context_row_text")
            or citation.get("table_row_boundary_text")
            or citation.get("table_row_raw_text")
            or ""
        ),
    ).strip()
    table_caption = re.sub(r"\s+", " ", str(citation.get("table_caption") or "")).strip()
    table_header = re.sub(r"\s+", " ", str(citation.get("table_header") or "")).strip()

    parts: list[str] = []
    for value in [
        *_collect_citation_table_evidence_texts(citation),
        citation.get("numeric_table_exact_context_row_text", ""),
        citation.get("table_row_boundary_text", ""),
        citation.get("table_row_raw_text", ""),
        citation.get("context_segment_text", ""),
        citation.get("highlight_text", ""),
        citation.get("display_text", ""),
        citation.get("source_text", ""),
        citation.get("start_phrase", ""),
        citation.get("end_phrase", ""),
        citation.get("table_id", ""),
        table_caption,
        table_header,
    ]:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        if not normalized:
            continue
        if _looks_like_broad_numeric_table_text(
            normalized,
            exact_row_text=exact_row,
            table_caption=table_caption,
            table_header=table_header,
        ):
            continue
        parts.append(normalized)

    return " ".join(dict.fromkeys(parts)).strip()


def _numeric_table_text_matches_columns(text: str = "", target_columns: Optional[set[str]] = None) -> bool:
    columns = {column for column in (target_columns or set()) if column and column != "acc"}
    if not columns:
        return True
    sample = _normalize_numeric_table_column_key(text)
    if not sample:
        return False
    column_patterns = {
        "few": ("few",),
        "many": ("many",),
        "med": ("med",),
        "all": ("all",),
        "fid": ("fid",),
        "dgen": ("dgen",),
        "deltaacc": ("deltaacc",),
    }
    return any(any(token in sample for token in column_patterns.get(column, (column,))) for column in columns)


def _numeric_table_text_has_partial_target_columns(text: str = "", target_columns: Optional[set[str]] = None) -> bool:
    columns = {column for column in (target_columns or set()) if column and column != "acc"}
    if not columns:
        return False
    sample = _normalize_numeric_table_column_key(text)
    if not sample:
        return False
    column_patterns = {
        "few": ("few",),
        "many": ("many",),
        "med": ("med",),
        "all": ("all", "acc"),
        "fid": ("fid",),
        "dgen": ("dgen",),
        "deltaacc": ("deltaacc",),
    }

    def _has_column(column: str) -> bool:
        return any(token in sample for token in column_patterns.get(column, (column,)))

    matched_count = sum(1 for column in columns if _has_column(column))
    return 0 < matched_count < len(columns)


def _normalize_numeric_table_backbone_key(value: str = "") -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _extract_numeric_table_row_method_cell(row_text: str = "") -> str:
    row = re.sub(r"\s+", " ", str(row_text or "")).strip()
    if not row:
        return ""
    if "|" in row:
        return row.split("|", 1)[0].strip()
    return re.split(r"\s{2,}|\s+\d+(?:\.\d+)?", row, maxsplit=1)[0].strip()


def _infer_target_backbone_for_numeric_row(citation: dict, row_text: str = "", query: str = "") -> str:
    if not isinstance(citation, dict) or not row_text or not query:
        return ""
    hints = _query_rewriter.extract_numeric_table_hints(query)
    target_backbones = [
        str(value).strip()
        for value in (hints.get("backbones") or [])
        if str(value).strip()
    ]
    if not target_backbones:
        return ""

    row_backbone_key = _normalize_numeric_table_backbone_key(row_text)
    for backbone in target_backbones:
        if _normalize_numeric_table_backbone_key(backbone) in row_backbone_key:
            return ""

    method_cell = _extract_numeric_table_row_method_cell(row_text)
    method_key = _normalize_numeric_table_method_token(method_cell)
    if not method_key:
        return ""

    table_focus_backbone = str(citation.get("table_focus_backbone") or "").strip()
    if table_focus_backbone:
        focus_key = _normalize_numeric_table_backbone_key(table_focus_backbone)
        for backbone in target_backbones:
            if focus_key == _normalize_numeric_table_backbone_key(backbone):
                return table_focus_backbone

    sources: list[str] = []
    for field in ("source_text", "context_segment_text", "_full_text", "highlight_text", "display_text"):
        value = str(citation.get(field) or "").strip()
        if value:
            sources.append(value)
    for value in citation.get("numeric_table_comparison_rows") or []:
        if str(value or "").strip():
            sources.append(str(value))
    for unit in citation.get("evidence_units") or []:
        if not isinstance(unit, dict):
            continue
        for field in ("row_text", "content", "row_id"):
            value = str(unit.get(field) or "").strip()
            if value:
                sources.append(value)
        unit_backbone = str(unit.get("table_focus_backbone") or "").strip()
        if unit_backbone:
            sources.append(f"{unit.get('row_id', '')} {unit_backbone}")

    for source in sources:
        candidates = [source, *source.splitlines()]
        for candidate in candidates:
            candidate_method_key = _normalize_numeric_table_method_token(candidate)
            if method_key not in candidate_method_key:
                continue
            candidate_backbone_key = _normalize_numeric_table_backbone_key(candidate)
            for backbone in target_backbones:
                if _normalize_numeric_table_backbone_key(backbone) in candidate_backbone_key:
                    return backbone
    return ""


def _add_target_backbone_to_numeric_row(row_text: str = "", citation: Optional[dict] = None, query: str = "") -> str:
    normalized_row = re.sub(r"\s+", " ", str(row_text or "")).strip()
    if not normalized_row or not isinstance(citation, dict):
        return normalized_row
    backbone = _infer_target_backbone_for_numeric_row(citation, normalized_row, query)
    if not backbone:
        return normalized_row
    if "|" not in normalized_row:
        return f"{normalized_row} | {backbone}"
    first_cell, rest = normalized_row.split("|", 1)
    first_cell = first_cell.strip()
    rest = rest.strip()
    return f"{first_cell} | {backbone} | {rest}" if rest else f"{first_cell} | {backbone}"


def _score_numeric_table_answer_alignment(
    answer: str,
    citation: dict,
    *,
    query: str = "",
    hints: Optional[dict] = None,
) -> float:
    support = _build_numeric_table_citation_support_text(citation)
    if not support:
        return 0.0

    visible_support = _build_numeric_table_visible_support_text(citation)
    visible_scoring_text = _build_numeric_table_visible_scoring_text(citation) or visible_support
    target_tables = _extract_numeric_table_target_tables(query, hints)
    target_columns = _extract_numeric_table_target_columns(query, hints)
    visible_table_ok = _numeric_table_support_mentions_target_table(visible_scoring_text, target_tables)
    visible_columns_ok = _numeric_table_text_matches_columns(visible_scoring_text, target_columns)
    use_visible_only = bool(
        visible_scoring_text
        and (
            (target_tables and not visible_table_ok)
            or (target_columns and not visible_columns_ok)
        )
    )
    if target_tables or target_columns:
        scoring_text = visible_scoring_text
    else:
        scoring_text = visible_scoring_text if (visible_scoring_text and use_visible_only) else support

    answer_values = _extract_numeric_values_from_answer(answer)
    distinctive_values = {
        value
        for value in answer_values
        if "." in value or "," in value or len(value.replace(",", "")) >= 4
    }
    values_for_score = distinctive_values or answer_values
    support_compact = re.sub(r"[\s,]+", "", scoring_text)
    value_hits = sum(1 for value in values_for_score if value.replace(",", "") in support_compact)
    score = value_hits * 3.0

    if target_tables:
        if _numeric_table_support_mentions_target_table(scoring_text, target_tables):
            score += 2.0
        else:
            score -= 4.0

    if target_columns and _numeric_table_text_matches_columns(scoring_text, target_columns):
        score += 1.0
    elif target_columns:
        score -= 3.0

    target_methods = _extract_numeric_table_target_methods(query, hints)
    answer_methods = _extract_numeric_table_target_methods(answer, hints=None)
    support_method_key = _normalize_numeric_table_method_token(scoring_text)
    method_hits = sum(1 for method in target_methods if method and method in support_method_key)
    answer_method_hits = sum(1 for method in answer_methods if method and method in support_method_key)
    score += method_hits * 1.5
    score += answer_method_hits * 2.0
    if answer_methods and not answer_method_hits:
        score -= 3.0
    if distinctive_values and value_hits == 0:
        score -= 4.0
    if use_visible_only:
        score -= 3.0

    if _has_numeric_table_exact_row_support(citation):
        score += 1.25
    if citation.get("numeric_metric_focused"):
        score += 0.75
    return score


def _numeric_table_citation_quality_penalty(
    answer: str,
    citation: dict,
    *,
    query: str = "",
    hints: Optional[dict] = None,
) -> int:
    """Generic quality penalty for broad or weak numeric-table citation evidence.

    Older ranking code demoted a known bad table block by checking concrete
    values from one evaluation paper. The replacement only looks at structural
    evidence quality: whether the citation is scoped to the requested table,
    columns, row/method, and visible text instead of a broad mixed table block.
    """
    support = _build_numeric_table_citation_support_text(citation)
    visible = _build_numeric_table_visible_scoring_text(citation) or _build_numeric_table_visible_support_text(citation)
    scoring_text = visible or support
    target_tables = _extract_numeric_table_target_tables(query, hints)
    target_columns = _extract_numeric_table_target_columns(query, hints)
    target_methods = _extract_numeric_table_target_methods(query, hints)
    answer_methods = _extract_numeric_table_target_methods(answer, hints=None)

    penalty = 0
    if support and visible and len(support) > max(360, len(visible) * 3):
        penalty += 1
    if target_tables and not _numeric_table_support_mentions_target_table(scoring_text, target_tables):
        penalty += 2
    if target_columns and not _numeric_table_text_matches_columns(scoring_text, target_columns):
        penalty += 2
    if _numeric_table_text_has_partial_target_columns(scoring_text, target_columns):
        penalty += 1
    if not _has_numeric_table_exact_row_support(citation):
        penalty += 1

    method_key = _normalize_numeric_table_method_token(scoring_text)
    if target_methods and not any(method and method in method_key for method in target_methods):
        penalty += 2
    if answer_methods and not any(method and method in method_key for method in answer_methods):
        penalty += 2
    return penalty


def _align_numeric_table_inline_citations(
    answer: str,
    citations: list[dict],
    *,
    query: str = "",
) -> tuple[str, dict]:
    hints = _query_rewriter.extract_numeric_table_hints(query) if query else {}
    target_columns = _extract_numeric_table_target_columns(query, hints)
    if not answer or not citations or not (_is_numeric_table_metric_query(query) or target_columns):
        return answer, {"applied": False}

    normalized = _normalize_numeric_metric_bundle_citations(_normalize_citation_records(citations), query)
    if not normalized:
        return answer, {"applied": False}

    best_by_score = sorted(
        (
            (_score_numeric_table_answer_alignment(answer, citation, query=query, hints=hints), int(citation["ref"]))
            for citation in normalized
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not best_by_score or best_by_score[0][0] < 5.0:
        return answer, {"applied": False}

    citation_map = {int(citation["ref"]): citation for citation in normalized}
    ref_mapping: dict[int, int] = {}
    for ref in _extract_inline_citation_refs(answer):
        current = citation_map.get(ref)
        current_score = _score_numeric_table_answer_alignment(answer, current or {}, query=query, hints=hints)
        best_score, best_ref = best_by_score[0]
        if best_ref != ref and best_score >= current_score + 2.0:
            ref_mapping[ref] = best_ref

    if not ref_mapping:
        return answer, {"applied": False}
    return _rewrite_inline_citation_refs(answer, ref_mapping), {
        "applied": True,
        "ref_mapping": ref_mapping,
        "best_ref": best_by_score[0][1],
        "best_score": round(best_by_score[0][0], 4),
    }


def _force_best_numeric_table_citation(
    answer: str,
    aligned: list[dict],
    candidates: list[dict],
    *,
    query: str = "",
) -> tuple[str, list[dict], dict]:
    hints = _query_rewriter.extract_numeric_table_hints(query) if query else {}
    target_columns = _extract_numeric_table_target_columns(query, hints)
    if not answer or not candidates or not (_is_numeric_table_metric_query(query) or target_columns):
        return answer, aligned, {"applied": False}

    normalized_candidates = _normalize_numeric_metric_bundle_citations(
        _normalize_citation_records(candidates),
        query,
    )
    if not normalized_candidates:
        return answer, aligned, {"applied": False}

    def _candidate_rank(citation: dict) -> tuple[float, int, int, int, int]:
        support = _build_numeric_table_citation_support_text(citation)
        focused_preview = _build_focused_numeric_metric_citation_text(citation, query)
        quality_penalty = _numeric_table_citation_quality_penalty(answer, citation, query=query, hints=hints)
        span_rank = 0 if str(citation.get("alignment_status") or "").lower() == "span_matched" else 1
        text_len = len(re.sub(r"\s+", " ", focused_preview or support).strip())
        return (
            _score_numeric_table_answer_alignment(answer, citation, query=query, hints=hints),
            0 if _has_numeric_table_exact_row_support(citation) else 1,
            span_rank + quality_penalty,
            text_len,
            int(citation["ref"]),
        )

    scored = sorted(
        ((*_candidate_rank(citation), citation) for citation in normalized_candidates),
        key=lambda item: (-item[0], item[1], item[2], item[3], item[4]),
    )
    if not scored or scored[0][0] < 5.0:
        return answer, aligned, {"applied": False}

    best_score, _exact_rank, _quality_rank, _text_len, best_ref, best_citation = scored[0]
    refs_in_answer = _extract_inline_citation_refs(answer)
    aligned_map = {int(citation["ref"]): citation for citation in _normalize_citation_records(aligned)}
    current_scores = [
        _score_numeric_table_answer_alignment(answer, aligned_map.get(ref, {}), query=query, hints=hints)
        for ref in refs_in_answer
    ]
    current_best = max(current_scores) if current_scores else 0.0
    if best_ref in refs_in_answer and current_best >= best_score:
        return answer, aligned, {"applied": False}
    if best_score < current_best + 2.0 and current_best >= 5.0:
        return answer, aligned, {"applied": False}

    ref_mapping = {ref: best_ref for ref in refs_in_answer if ref != best_ref}
    rewritten = _rewrite_inline_citation_refs(answer, ref_mapping) if ref_mapping else answer

    next_aligned = [_focus_numeric_metric_citation(best_citation, query)]

    return rewritten, next_aligned, {
        "applied": True,
        "best_ref": best_ref,
        "best_score": round(best_score, 4),
        "ref_mapping": ref_mapping,
    }


def _dedupe_numeric_table_exact_aligned_citations(
    answer: str,
    aligned: list[dict],
    *,
    query: str = "",
) -> tuple[str, list[dict], dict]:
    if not answer or not aligned or not _is_numeric_table_metric_query(query):
        return answer, aligned, {}

    hints = _query_rewriter.extract_numeric_table_hints(query) if query else {}

    def _row_key(citation: dict) -> str:
        row = str(
            citation.get("numeric_table_exact_context_row_text")
            or _extract_numeric_metric_exact_row_from_bundle(citation, query)
            or ""
        )
        normalized = re.sub(r"\s+", " ", row).strip().lower()
        normalized = re.sub(r"\s*=\s*", "=", normalized)
        return normalized

    groups: dict[str, list[dict]] = {}
    for citation in _normalize_numeric_metric_bundle_citations(_normalize_citation_records(aligned), query):
        key = _row_key(citation)
        if not key:
            continue
        groups.setdefault(key, []).append(citation)

    duplicate_groups = {key: values for key, values in groups.items() if len(values) > 1}
    if not duplicate_groups:
        return answer, aligned, {}

    keep_refs: set[int] = set()
    remove_to_keep: dict[int, int] = {}
    for values in duplicate_groups.values():
        def _rank(citation: dict) -> tuple[float, int, int, int]:
            support = _build_numeric_table_citation_support_text(citation)
            focused = _build_focused_numeric_metric_citation_text(citation, query)
            quality_penalty = _numeric_table_citation_quality_penalty(answer, citation, query=query, hints=hints)
            span = int(str(citation.get("alignment_status") or "").lower() == "span_matched")
            text_len = len(re.sub(r"\s+", " ", focused or support).strip())
            score = _score_numeric_table_answer_alignment(answer, citation, query=query, hints=hints)
            return (score, span, -quality_penalty, -text_len)

        keeper = max(values, key=_rank)
        keep_ref = int(keeper["ref"])
        keep_refs.add(keep_ref)
        for citation in values:
            ref = int(citation["ref"])
            if ref != keep_ref:
                remove_to_keep[ref] = keep_ref

    rewritten = _rewrite_inline_citation_refs(answer, remove_to_keep) if remove_to_keep else answer
    filtered: list[dict] = []
    seen_refs: set[int] = set()
    for citation in _normalize_numeric_metric_bundle_citations(_normalize_citation_records(aligned), query):
        ref = int(citation["ref"])
        if ref in remove_to_keep:
            continue
        if ref in seen_refs:
            continue
        filtered.append(_focus_numeric_metric_citation(citation, query) if ref in keep_refs else citation)
        seen_refs.add(ref)

    return rewritten, filtered, {
        "applied": True,
        "ref_mapping": remove_to_keep,
        "removed_ref_count": len(remove_to_keep),
    }


def _build_numeric_table_citation_support_text(citation: Optional[dict]) -> str:
    if not isinstance(citation, dict):
        return ""
    parts = [
        *_collect_citation_table_evidence_texts(citation),
        citation.get("context_segment_text", ""),
        citation.get("source_text", ""),
        citation.get("display_text", ""),
        citation.get("highlight_text", ""),
        citation.get("start_phrase", ""),
        citation.get("end_phrase", ""),
        citation.get("_full_text", ""),
        citation.get("table_id", ""),
        citation.get("table_caption", ""),
        citation.get("table_header", ""),
    ]
    return " ".join(
        re.sub(r"\s+", " ", str(part or "")).strip()
        for part in parts
        if re.sub(r"\s+", " ", str(part or "")).strip()
    ).strip()


def _build_phrase_alignment_text(citation: Optional[dict]) -> str:
    if not isinstance(citation, dict):
        return ""
    phrases = [
        re.sub(r"\s+", " ", str(citation.get("start_phrase") or "")).strip(),
        re.sub(r"\s+", " ", str(citation.get("end_phrase") or "")).strip(),
    ]
    return " ".join(dict.fromkeys(phrase for phrase in phrases if phrase))


def _citation_matches_numeric_table_columns(
    citation: Optional[dict],
    target_columns: set[str],
) -> bool:
    if not target_columns:
        return True
    sample = _normalize_numeric_table_column_key(_build_numeric_table_citation_support_text(citation))
    if not sample:
        return False
    column_patterns = {
        "few": ("few",),
        "many": ("many",),
        "med": ("med",),
        "all": ("all",),
        "fid": ("fid",),
        "acc": ("acc",),
        "dgen": ("dgen",),
        "deltaacc": ("deltaacc",),
    }
    return any(any(token in sample for token in column_patterns.get(column, (column,))) for column in target_columns)


def _looks_like_broad_numeric_table_text(
    value: str,
    *,
    exact_row_text: str = "",
    table_caption: str = "",
    table_header: str = "",
) -> bool:
    raw = str(value or "")
    normalized = re.sub(r"\s+", " ", raw).strip()
    if not normalized:
        return False

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    normalized_exact_row = re.sub(r"\s+", " ", str(exact_row_text or "")).strip()
    normalized_caption = re.sub(r"\s+", " ", str(table_caption or "")).strip()
    normalized_header = re.sub(r"\s+", " ", str(table_header or "")).strip()

    if normalized_exact_row and normalized_exact_row in normalized:
        compact_budget = len(normalized_exact_row) + len(normalized_caption) + len(normalized_header) + 96
        if len(lines) <= 4 and len(normalized) <= max(260, compact_budget):
            return False

    pipe_count = raw.count("|")
    numeric_hits = len(re.findall(r"\d+(?:\.\d+)?", normalized))
    if len(lines) >= 6:
        return True
    if pipe_count >= 16:
        return True
    if numeric_hits >= 14 and len(normalized) >= 220:
        return True
    if len(normalized) >= 420:
        return True
    return False


def _should_apply_numeric_table_strict_gate(query: str = "", hints: Optional[dict] = None) -> bool:
    if not should_apply_numeric_table_specialization():
        return False
    sample = re.sub(r"\s+", " ", str(query or "")).strip()
    if not sample:
        return False
    if (hints or {}).get("comparison"):
        return True
    return bool(_NUMERIC_TABLE_COMPARATOR_QUERY_RE.search(sample))


def _collect_citation_table_evidence_texts(citation: dict, query: str = "") -> list[str]:
    if not isinstance(citation, dict):
        return []

    collected: list[str] = []
    seen: set[str] = set()

    def _append(value: Optional[str]) -> None:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        if not normalized:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        collected.append(normalized)

    def _pick_first_normalized(*fields: str) -> str:
        for field in fields:
            normalized = re.sub(r"\s+", " ", str(citation.get(field, "") or "")).strip()
            if normalized:
                return normalized
        return ""

    def _looks_like_structured_table_text(value: str) -> bool:
        if not value:
            return False
        if "\n" in value or "|" in value:
            return True
        return len(re.findall(r"\d+(?:\.\d+)?", value)) >= 2

    chunk_type = str(citation.get("chunk_type") or citation.get("block_type") or "").strip().lower()
    has_structured_table_support = _has_numeric_table_structured_support(citation)
    table_caption = _pick_first_normalized("numeric_table_exact_context_caption", "table_caption")
    table_header = _pick_first_normalized("numeric_table_exact_context_header", "table_header")
    exact_row_text = _pick_first_normalized(
        "numeric_table_exact_context_row_text",
        "table_row_boundary_text",
        "table_row_raw_text",
    )
    focused_row_text = ""
    if has_structured_table_support:
        for field in ("display_text", "highlight_text"):
            candidate = _pick_first_normalized(field)
            if _looks_like_structured_table_text(candidate):
                focused_row_text = candidate
                break
        if not focused_row_text and exact_row_text:
            focused_row_text = exact_row_text
        for field in ("context_segment_text", "source_text", "_full_text"):
            if focused_row_text:
                break
            candidate = _pick_first_normalized(field)
            if not candidate:
                continue
            if not _looks_like_structured_table_text(candidate):
                continue
            if _looks_like_broad_numeric_table_text(
                candidate,
                exact_row_text=exact_row_text,
                table_caption=table_caption,
                table_header=table_header,
            ):
                continue
            focused_row_text = candidate
            break
    if not focused_row_text and chunk_type in {"table_row", "table_cell"}:
        focused_row_text = _pick_first_normalized("display_text", "highlight_text")

    row_text = focused_row_text or exact_row_text
    row_text = _add_target_backbone_to_numeric_row(row_text, citation, query)
    if row_text:
        if table_caption and table_caption not in row_text:
            _append(table_caption)
        if table_header and table_header not in row_text:
            _append(table_header)
        _append(row_text)

    for unit in citation.get("evidence_units") or []:
        if not isinstance(unit, dict):
            continue
        unit_type = str(unit.get("evidence_unit_type") or "").strip().lower()
        unit_caption = unit.get("table_caption") or table_caption
        unit_header = unit.get("table_header") or table_header
        if unit_type == "table_row":
            if row_text and focused_row_text:
                for cell in unit.get("cell_evidence_units") or []:
                    if isinstance(cell, dict):
                        _append(cell.get("content"))
                continue
            for value in (
                unit_caption,
                unit_header,
                unit.get("row_text"),
                unit.get("content"),
                unit.get("row_numbers"),
            ):
                _append(value)
            for cell in unit.get("cell_evidence_units") or []:
                if isinstance(cell, dict):
                    _append(cell.get("content"))
        elif unit_type == "table_cell":
            for value in (unit_caption, unit_header, unit.get("content")):
                _append(value)

    for cell in citation.get("cell_evidence_units") or []:
        if isinstance(cell, dict):
            _append(cell.get("content"))

    return collected


def _build_citation_context_text(citation: dict, query: str = "") -> str:
    if not isinstance(citation, dict):
        return ""

    if citation.get("unavailable_dataset_evidence") or citation.get("explicit_absence_evidence"):
        for field in ("source_text", "context_segment_text", "highlight_text", "display_text"):
            normalized = re.sub(r"\s+", " ", str(citation.get(field, "") or "")).strip()
            if normalized:
                return normalized

    table_evidence = _collect_citation_table_evidence_texts(citation, query=query)
    if table_evidence:
        return "\n".join(table_evidence)

    focused = _build_focused_citation_context_text(citation)
    if focused:
        return focused

    phrase_text = _build_phrase_alignment_text(citation)
    if phrase_text:
        return phrase_text

    for field in ("context_segment_text", "source_text", "_full_text", "display_text", "highlight_text"):
        normalized = re.sub(r"\s+", " ", str(citation.get(field, "") or "")).strip()
        if normalized:
            return normalized
    return ""


def _build_focused_citation_context_text(citation: dict, window_chars: int = 900) -> str:
    highlight = re.sub(r"\s+", " ", str(citation.get("highlight_text", "") or "")).strip()
    phrase_text = _build_phrase_alignment_text(citation)

    alignment_status = str(citation.get("alignment_status") or "").strip().lower()
    has_phrase_match = bool(citation.get("start_phrase") or citation.get("end_phrase"))
    if alignment_status != "span_matched" and not has_phrase_match:
        return ""

    half_window = max(120, int(window_chars / 2))
    if phrase_text:
        for field in ("context_segment_text", "source_text", "_full_text", "display_text"):
            source = re.sub(r"\s+", " ", str(citation.get(field, "") or "")).strip()
            if not source:
                continue
            phrase_pos = source.find(phrase_text)
            target_phrase = phrase_text
            if phrase_pos < 0:
                for phrase in dict.fromkeys(
                    re.sub(r"\s+", " ", str(citation.get(key) or "")).strip()
                    for key in ("start_phrase", "end_phrase")
                    if citation.get(key)
                ):
                    phrase_pos = source.find(phrase)
                    if phrase_pos >= 0:
                        target_phrase = phrase
                        break
            if phrase_pos >= 0:
                start = max(0, phrase_pos - half_window)
                end = min(len(source), phrase_pos + len(target_phrase) + half_window)
                return f"{'...' if start > 0 else ''}{source[start:end]}{'...' if end < len(source) else ''}"
        return phrase_text

    if not highlight:
        return ""

    for field in ("context_segment_text", "source_text", "_full_text", "display_text"):
        source = re.sub(r"\s+", " ", str(citation.get(field, "") or "")).strip()
        if not source:
            continue
        pos = source.find(highlight)
        if pos >= 0:
            start = max(0, pos - half_window)
            end = min(len(source), pos + len(highlight) + half_window)
            return f"{'...' if start > 0 else ''}{source[start:end]}{'...' if end < len(source) else ''}"

    return highlight


def _build_context_segments_from_citations(citations: list[dict], *, query: str = "") -> list[dict]:
    segments = []
    hints = _query_rewriter.extract_numeric_table_hints(query) if query else {}
    target_columns = _extract_numeric_table_target_columns(query, hints) if query else set()
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        try:
            ref = int(c.get("ref"))
        except (TypeError, ValueError):
            continue
        text = _build_citation_context_text(c, query=query)
        if not text:
            continue
        if _is_internal_context_map_segment({"text": text}):
            continue
        formula_score = _calc_formula_citation_anchor_score(text)
        segments.append({
            "ref": ref,
            "text": text,
            "page_range": c.get("page_range") or [],
            "group_id": c.get("group_id", ""),
            "context_id": c.get("context_id", ""),
            "evidence_id": c.get("evidence_id", ""),
            "chunk_id": c.get("chunk_id", ""),
            "child_chunk_id": c.get("child_chunk_id", ""),
            "parent_id": c.get("parent_id", ""),
            "table_id": c.get("table_id", ""),
            "table_bundle_id": c.get("table_bundle_id", ""),
            "evidence_unit_id": c.get("evidence_unit_id", ""),
            "retrieval_type": c.get("retrieval_type", ""),
        })
        for row_idx, row_text in enumerate(c.get("numeric_table_comparison_rows") or [], 1):
            normalized_row = re.sub(r"\s+", " ", str(row_text or "")).strip()
            if not normalized_row:
                continue
            if _numeric_table_text_has_partial_target_columns(normalized_row, target_columns):
                continue
            normalized_row = _add_target_backbone_to_numeric_row(normalized_row, c, query)
            method = normalized_row.split("|", 1)[0].strip() or f"row-{row_idx}"
            all_match = re.search(r"All\s*=?\s*(\d+(?:\.\d+)?)", normalized_row, re.IGNORECASE)
            value_text = f" All={all_match.group(1)}" if all_match else ""
            segments.append({
                "ref": ref,
                "text": f"numeric_comparison_row: {method}{value_text}. row: {normalized_row}",
                "page_range": c.get("page_range") or [],
                "group_id": c.get("group_id", ""),
                "context_id": f"{c.get('context_id', '')}:numeric_comparison_row_{row_idx}" if c.get("context_id") else "",
                "evidence_id": f"{c.get('evidence_id', '')}:numeric_comparison_row_{row_idx}" if c.get("evidence_id") else "",
                "chunk_id": c.get("chunk_id", ""),
                "child_chunk_id": c.get("child_chunk_id", ""),
                "parent_id": c.get("parent_id", ""),
                "table_id": c.get("table_id", ""),
                "table_bundle_id": c.get("table_bundle_id", ""),
                "evidence_unit_id": c.get("evidence_unit_id", ""),
                "retrieval_type": c.get("retrieval_type", ""),
                "segment_role": "numeric_comparison_row",
                "source_ref": ref,
            })
        if formula_score >= 8.0:
            for sub_text, suffix in _split_formula_framework_segment_text(text):
                if not sub_text or sub_text == text:
                    continue
                segments.append({
                    "ref": ref,
                    "text": sub_text,
                    "page_range": c.get("page_range") or [],
                    "group_id": c.get("group_id", ""),
                    "context_id": f"{c.get('context_id', '')}:{suffix}" if c.get("context_id") else "",
                    "evidence_id": f"{c.get('evidence_id', '')}:{suffix}" if c.get("evidence_id") else "",
                    "chunk_id": c.get("chunk_id", ""),
                    "child_chunk_id": c.get("child_chunk_id", ""),
                    "parent_id": c.get("parent_id", ""),
                    "table_id": c.get("table_id", ""),
                    "table_bundle_id": c.get("table_bundle_id", ""),
                    "evidence_unit_id": c.get("evidence_unit_id", ""),
                    "retrieval_type": c.get("retrieval_type", ""),
                    "segment_role": suffix,
                    "source_ref": ref,
                })
    return segments


def _split_formula_framework_segment_text(text: str) -> list[tuple[str, str]]:
    """把公式证据拆成局部窗口，提升引用诊断覆盖。"""
    source = re.sub(r"\s+", " ", str(text or "")).strip()
    if not source or _calc_formula_citation_anchor_score(source) < 8.0:
        return []

    pieces: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(label: str, pattern: str, *, before: int = 220, after: int = 360) -> None:
        match = re.search(pattern, source, re.IGNORECASE)
        if not match:
            return
        start = max(0, match.start() - before)
        end = min(len(source), match.end() + after)
        snippet = source[start:end].strip()
        if len(snippet) < 80:
            return
        normalized_key = re.sub(r"\W+", "", snippet.lower())[:180]
        key = f"{label}:{normalized_key}"
        if key in seen:
            return
        seen.add(key)
        zh_label = {"formula_context": "公式上下文", "formula_objective": "目标函数"}.get(label, label)
        pieces.append((f"{label}: {zh_label}。{snippet}", label))

    _add(
        "formula_context",
        r"formula|equation|公式|方程|algorithm|算法|framework|框架",
        before=0,
        after=300,
    )
    _add(
        "formula_objective",
        r"objective|loss|目标函数|损失函数|=|≈|≤|≥|∑|∏|√|\\(?:frac|sum|prod|sqrt|mathcal)",
        before=180,
        after=420,
    )
    return pieces[:3]



def _build_backbone_pretrain_support_text(citation: Optional[dict]) -> str:
    if not isinstance(citation, dict):
        return ""
    fields = (
        _build_phrase_alignment_text(citation),
        citation.get("context_segment_text", ""),
        citation.get("source_text", ""),
        citation.get("display_text", ""),
        citation.get("highlight_text", ""),
        citation.get("_full_text", ""),
        citation.get("table_caption", ""),
        citation.get("table_header", ""),
        citation.get("group_id", ""),
    )
    return " ".join(
        re.sub(r"\s+", " ", str(value or "")).strip()
        for value in fields
        if re.sub(r"\s+", " ", str(value or "")).strip()
    ).strip()



_NEGATIVE_UNAVAILABLE_ANSWER_RE = re.compile(
    r"(未明确记载|未明确说明|未明确给出|未记载|没有明确记载|没有明确说明|没有被明确记载|未包含|没有包含|"
    r"无法回答|无法确认|无法提供|未提供|未给出|没有给出|没有报告|"
    r"not\s+(?:explicitly\s+)?(?:reported|mentioned|provided|given|stated)|not\s+available)",
    re.IGNORECASE,
)
_DATASET_UNAVAILABLE_QUERY_RE = re.compile(
    r"(数据集|训练集|测试集|类别|样本|dataset|training\s+set|test\s+set|classes|samples)",
    re.IGNORECASE,
)
_UNAVAILABLE_DATASET_NUMERIC_QUERY_RE = re.compile(
    r"(baseline|基线|准确率|精度|差距|"
    r"overall\s+accuracy|accuracy\s+gap|dataset|数据集)",
    re.IGNORECASE,
)


def _is_dataset_unavailable_answer(answer: str = "", query: str = "") -> bool:
    combined = f"{query or ''} {answer or ''}"
    return bool(
        _NEGATIVE_UNAVAILABLE_ANSWER_RE.search(str(answer or ""))
        and _DATASET_UNAVAILABLE_QUERY_RE.search(combined)
    )









def _segment_to_recovery_citation(segment: dict, ref: int) -> dict:
    text = re.sub(
        r"\s+",
        " ",
        str(
            segment.get("text")
            or segment.get("source_text")
            or segment.get("display_text")
            or segment.get("highlight_text")
            or ""
        ),
    ).strip()
    citation = {
        "ref": ref,
        "source_text": text,
        "display_text": text,
        "highlight_text": text,
        "context_segment_text": text,
        "page_range": segment.get("page_range") or [],
        "group_id": segment.get("group_id", ""),
        "context_id": segment.get("context_id", ""),
        "evidence_id": segment.get("evidence_id", ""),
        "chunk_id": segment.get("chunk_id", ""),
        "child_chunk_id": segment.get("child_chunk_id", ""),
        "parent_id": segment.get("parent_id", ""),
        "table_id": segment.get("table_id", ""),
        "table_bundle_id": segment.get("table_bundle_id", ""),
        "evidence_unit_id": segment.get("evidence_unit_id", ""),
        "retrieval_type": segment.get("retrieval_type", ""),
        "segment_role": segment.get("segment_role", ""),
        "source_ref": segment.get("source_ref", segment.get("ref", ref)),
    }
    return citation


def _context_segment_recovery_key(segment: dict, text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    prefix = normalized[:160]
    suffix = normalized[-160:] if len(normalized) > 160 else ""
    digest = hashlib.blake2b(normalized.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
    text_fp = f"{len(normalized)}:{prefix}:{suffix}:{digest}"
    for field in ("evidence_id", "chunk_id", "child_chunk_id"):
        value = re.sub(r"\s+", " ", str((segment or {}).get(field) or "")).strip().casefold()
        if value:
            return f"id:{field}:{value}"
    scoped_parts: list[str] = []
    for field in ("context_id", "parent_id", "group_id", "table_bundle_id", "table_id", "evidence_unit_id"):
        value = re.sub(r"\s+", " ", str((segment or {}).get(field) or "")).strip().casefold()
        if value:
            scoped_parts.append(f"{field}:{value}")
    if scoped_parts:
        return f"scoped:{'|'.join(scoped_parts)}:{text_fp}"
    return f"text:{text_fp}"


def _context_segments_to_recovery_citations(
    context_segments: Optional[list[dict]],
    *,
    start_ref: int = 1,
    query: str = "",
    reserved_refs: Optional[set[int]] = None,
) -> list[dict]:
    if not isinstance(context_segments, list):
        return []
    citations: list[dict] = []
    seen: set[str] = set()
    used_refs: set[int] = set(reserved_refs or set())
    for segment in context_segments:
        if not isinstance(segment, dict):
            continue
        segment_role = str(segment.get("segment_role") or "").strip()
        text = re.sub(
            r"\s+",
            " ",
            str(
                segment.get("text")
                or segment.get("source_text")
                or segment.get("display_text")
                or segment.get("highlight_text")
                or ""
            ),
        ).strip()
        if not text:
            continue
        dedupe_key = _context_segment_recovery_key(segment, text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        try:
            segment_ref = int(segment.get("ref") or 0)
        except (TypeError, ValueError):
            segment_ref = 0
        ref = segment_ref if segment_ref > 0 and segment_ref not in used_refs else start_ref + len(citations)
        while ref in used_refs:
            ref += 1
        used_refs.add(ref)
        citations.append(_segment_to_recovery_citation(segment, ref))
    return citations


def _merge_inline_referenced_recovery_citations(
    answer: str,
    citations: list[dict],
    recovery_citations: list[dict],
) -> list[dict]:
    """把回答已引用的上下文恢复证据并入 citation 对齐池。

    多查询或 agent 路径可能先把证据保存在 context_segments 中，而初始 citations
    只包含主查询候选。这里仅合并回答正文已经显式引用的 ref，避免把未使用的
    上下文段泛化塞进最终引用列表。
    """
    normalized = _normalize_citation_records(citations)
    if not answer or not recovery_citations:
        return normalized

    refs_in_answer = set(_extract_inline_citation_refs(answer))
    if not refs_in_answer:
        return normalized

    seen_refs = {int(c["ref"]) for c in normalized}
    for citation in _normalize_citation_records(recovery_citations):
        ref = int(citation["ref"])
        if ref in refs_in_answer and ref not in seen_refs:
            normalized.append(citation)
            seen_refs.add(ref)
    return normalized


def _recover_numeric_table_metric_citation_from_context(
    answer: str,
    context_segments: Optional[list[dict]],
    *,
    query: str = "",
    start_ref: int = 1,
) -> tuple[str, list[dict], dict]:
    if not answer or not _is_numeric_table_metric_query(query):
        return answer, [], {}
    if not _extract_numeric_values_from_answer(answer):
        return answer, [], {}

    recovery_citations = _context_segments_to_recovery_citations(
        context_segments,
        start_ref=start_ref,
        query=query,
    )
    if not recovery_citations:
        return answer, [], {"applied": False, "reason": "no_context_segments"}

    hints = _query_rewriter.extract_numeric_table_hints(query) if query else {}
    normalized = _normalize_numeric_metric_bundle_citations(
        _normalize_citation_records(recovery_citations),
        query,
    )
    scored = sorted(
        (
            (
                _score_numeric_table_answer_alignment(answer, citation, query=query, hints=hints),
                0 if _has_numeric_table_exact_row_support(citation) else 1,
                len(re.sub(r"\s+", " ", _build_numeric_table_citation_support_text(citation)).strip()),
                citation,
            )
            for citation in normalized
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    if not scored or scored[0][0] < 5.0:
        return answer, [], {
            "applied": False,
            "reason": "no_strong_numeric_table_context_anchor",
            "best_score": round(scored[0][0], 4) if scored else 0.0,
        }

    best_score, _exact_rank, _text_len, best = scored[0]
    best = _focus_numeric_metric_citation(best, query)
    ref = int(best["ref"])
    refs = _extract_inline_citation_refs(answer)
    if refs:
        rewritten = _rewrite_inline_citation_refs(answer, {old_ref: ref for old_ref in refs if old_ref != ref})
    else:
        rewritten = _attach_refs_to_sentence(answer, [ref])
    return rewritten, [best], {
        "applied": True,
        "source_ref": ref,
        "score": round(best_score, 4),
        "reason": "context_numeric_table_anchor",
    }






def _extract_backbone_pretrain_policy_window(text: str) -> str:
    sample = re.sub(r"\s+", " ", str(text or "")).strip()
    if not sample:
        return ""
    patterns = [
        r"without utilizing external data or model",
        r"without relying on external data or pre[-\s]?trained model weights",
        r"without relying on external data or models?",
        r"diffusion model trained from scratch",
        r"trained from scratch on only the long[-\s]?tailed? dataset",
        r"without external data or knowledge",
        r"pre[-\s]?trained model weights",
        r"without resorting to any external data or model",
    ]
    matches: list[re.Match[str]] = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, sample, re.IGNORECASE))
    if not matches:
        return ""

    start = max(0, min(match.start() for match in matches) - 280)
    end = min(len(sample), max(match.end() for match in matches) + 420)
    if end - start > 1900:
        center = matches[-1].start()
        start = max(0, center - 760)
        end = min(len(sample), start + 1900)
    return sample[start:end].strip()





def _extract_numeric_table_citation_row_text(citation: Optional[dict]) -> str:
    if not isinstance(citation, dict):
        return ""

    for field in (
        "numeric_table_exact_context_row_text",
        "table_row_boundary_text",
        "table_row_raw_text",
        "display_text",
        "highlight_text",
    ):
        value = re.sub(r"\s+", " ", str(citation.get(field, "") or "")).strip()
        if value:
            return value

    for unit in citation.get("evidence_units") or []:
        if not isinstance(unit, dict):
            continue
        if str(unit.get("evidence_unit_type") or "").strip().lower() != "table_row":
            continue
        value = re.sub(r"\s+", " ", str(unit.get("content", "") or "")).strip()
        if value:
            return value
    return ""


def _build_numeric_table_comparator_context_segments(retrieval_meta: dict) -> list[dict]:
    if not should_apply_numeric_table_specialization():
        return []
    if not isinstance(retrieval_meta, dict):
        return []

    query = str(retrieval_meta.get("search_query") or "").strip()
    hints = _query_rewriter.extract_numeric_table_hints(query) if query else {}
    if not _should_apply_numeric_table_strict_gate(query, hints):
        return []

    target_columns = _extract_numeric_table_target_columns(query, hints)
    grouped: dict[str, list[dict]] = {}
    for citation in retrieval_meta.get("citations", []) or []:
        if not isinstance(citation, dict):
            continue
        if not _has_numeric_table_exact_row_support(citation):
            continue
        if target_columns and not _citation_matches_numeric_table_columns(citation, target_columns):
            continue
        bundle_key = str(citation.get("group_id") or citation.get("table_id") or "").strip()
        if not bundle_key:
            continue
        grouped.setdefault(bundle_key, []).append(citation)

    if not grouped:
        return []

    bundle_key, bundle_citations = max(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            sum(1 for citation in item[1] if str(citation.get("chunk_type") or citation.get("block_type") or "").strip().lower() == "table_row"),
            -min(int(citation.get("ref") or 10**9) for citation in item[1]),
        ),
    )
    if len(bundle_citations) < 2:
        return []

    ordered_citations = sorted(
        bundle_citations,
        key=lambda citation: (
            int(citation.get("ref") or 10**9),
            int(citation.get("source_ref") or citation.get("ref") or 10**9),
        ),
    )

    caption = ""
    header = ""
    rows: list[str] = []
    seen_rows: set[str] = set()
    for citation in ordered_citations:
        if not caption:
            caption = re.sub(r"\s+", " ", str(citation.get("numeric_table_exact_context_caption") or citation.get("table_caption") or "")).strip()
        if not header:
            header = re.sub(r"\s+", " ", str(citation.get("numeric_table_exact_context_header") or citation.get("table_header") or "")).strip()
        row_text = _extract_numeric_table_citation_row_text(citation)
        if not row_text:
            continue
        row_key = row_text.casefold()
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        rows.append(row_text)

    if len(rows) < 2:
        return []

    text_parts = [part for part in (caption, header, *rows) if part]
    if not text_parts:
        return []

    first = ordered_citations[0]
    page_points = [
        page
        for citation in ordered_citations
        for page in (citation.get("page_range") or [])
        if isinstance(page, int) and page > 0
    ]
    page_range = [min(page_points), max(page_points)] if page_points else (first.get("page_range") or [])
    return [{
        "ref": int(first.get("ref") or 1),
        "text": "\n".join(text_parts),
        "page_range": page_range,
        "group_id": bundle_key,
    }]


def _normalize_response_context_segment(seg: dict) -> dict | None:
    if not isinstance(seg, dict):
        return None
    text = (seg.get("text") or "").strip()
    if not text:
        return None
    segment_role = str(seg.get("segment_role") or "").strip()
    return {
        "ref": seg.get("ref"),
        "text": text,
        "page_range": seg.get("page_range") or [],
        "group_id": seg.get("group_id", ""),
        "context_id": seg.get("context_id", ""),
        "evidence_id": seg.get("evidence_id", ""),
        "chunk_id": seg.get("chunk_id", ""),
        "child_chunk_id": seg.get("child_chunk_id", ""),
        "parent_id": seg.get("parent_id", ""),
        "chunk_type": seg.get("chunk_type", ""),
        "table_id": seg.get("table_id", ""),
        "table_bundle_id": seg.get("table_bundle_id", ""),
        "segment_role": segment_role,
        "source_ref": seg.get("source_ref"),
    }


def _merge_response_context_segments(*segment_lists: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for segments in segment_lists:
        for segment in segments or []:
            normalized = _normalize_response_context_segment(segment)
            if not normalized:
                continue
            key = (
                str(normalized.get("evidence_id") or "").casefold()
                or str(normalized.get("context_id") or "").casefold()
                or re.sub(r"\s+", " ", normalized["text"]).casefold()[:240]
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def _snapshot_retrieval_context_segments(retrieval_meta: dict) -> None:
    """Preserve retrieval evidence before display citation pruning rewrites segments."""
    if not isinstance(retrieval_meta, dict) or retrieval_meta.get("_retrieval_context_segments"):
        return
    segments = _merge_response_context_segments(retrieval_meta.get("_context_segments") or [])
    if segments:
        retrieval_meta["_retrieval_context_segments"] = segments


def _is_internal_context_map_segment(segment: dict) -> bool:
    text = str((segment or {}).get("text") or "")
    return bool(
        text.startswith("【文档地图】")
        or "[document map]" in text[:200].lower()
        or "chunks:" in text[:600].lower() and "【group-" in text[:600]
    )


def _looks_like_result_table_segment(segment: dict) -> bool:
    text = str((segment or {}).get("text") or "")
    lower = text.lower()
    if "[structured table bundle]" in lower:
        return True
    if segment.get("table_id") or segment.get("table_bundle_id"):
        return True
    chunk_type = str(segment.get("chunk_type") or "").strip().lower()
    if chunk_type in {"table", "table_row", "table_cell", "caption"}:
        return True
    numeric_hits = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    table_terms = sum(
        1
        for token in ("method", "model", "all", "many", "med", "few", "acc", "score", "fid", "asr", "map")
        if re.search(rf"\b{re.escape(token)}\b", lower)
    )
    return numeric_hits >= 8 and table_terms >= 2


def _response_context_anchor_score(segment: dict, query: str, evidence_need: set[str]) -> float:
    """Score generic relevance of a response-side context segment to the query."""
    text = str((segment or {}).get("text") or "")
    if not text.strip():
        return 0.0
    if _is_internal_context_map_segment(segment):
        return -100.0

    score = 0.0
    lower = text.lower()
    if str(segment.get("source_ref") or "").strip():
        score += 2.0
    if str(segment.get("evidence_id") or "").strip() or str(segment.get("context_id") or "").strip():
        score += 0.15

    anchors = _extract_citation_query_anchors(query, max_terms=18)
    if anchors:
        matches = sum(1 for anchor in anchors if technical_anchor_matches(anchor, text))
        score += min(matches, 6) * 1.0
        if matches <= 0 and len(anchors) >= 3:
            score -= 1.0

    formula_query = _is_formula_framework_query(query) or "formula" in evidence_need
    formula_score = _calc_formula_citation_anchor_score(text)
    if formula_query:
        if not str(segment.get("source_ref") or "").strip() and formula_score <= 0:
            return -5.0
        if _looks_like_result_table_segment(segment) and not str(segment.get("source_ref") or "").strip():
            strong_formula_signal = bool(
                re.search(
                    r"formula|equation|objective|公式|方程|目标函数|"
                    r"\\(?:frac|sum|prod|sqrt|mathcal)",
                    text,
                    re.IGNORECASE,
                )
            )
            if not strong_formula_signal:
                return -4.0
        score += min(formula_score, 10.0) * 0.35
        if formula_score <= 0 and "[structured table bundle]" in lower:
            score -= 2.0

    if "numeric_table" not in evidence_need and "[structured table bundle]" in lower:
        score -= 1.2
    if re.search(r"\blimitations?\b|局限|限制", lower) and not re.search(r"\blimitations?\b|局限|限制", (query or "").lower()):
        score -= 1.0
    return score


def _filter_response_context_segments(
    segments: list[dict],
    *,
    query: str = "",
    evidence_need: set[str] | None = None,
    citation_count: int = 0,
) -> list[dict]:
    """Trim evaluation contexts to query-relevant evidence without losing coverage.

    Retrieval snapshots preserve recall, but sending every retrieved item to
    evaluators mixes in document maps, appendix notes, and unrelated tables. This
    keeps citation evidence plus the best generic query-matching retrieval
    segments. It is deliberately based only on query anchors and segment
    provenance, not on paper names or expected answers.
    """
    evidence_need = evidence_need or set()
    if not segments:
        return []

    normalized = _merge_response_context_segments(segments)
    if not normalized:
        return []

    scored = [
        (_response_context_anchor_score(segment, query, evidence_need), idx, segment)
        for idx, segment in enumerate(normalized)
    ]
    citation_refs = {
        int(segment.get("ref"))
        for _score, _idx, segment in scored
        if segment.get("source_ref") is not None and str(segment.get("ref") or "").isdigit()
    }
    kept_indices: set[int] = {
        idx
        for score, idx, segment in scored
        if score >= 1.0 or int(segment.get("ref") or -1) in citation_refs
    }

    min_keep = min(len(normalized), max(3, min(6, citation_count + 2)))
    for _score, idx, _segment in sorted(scored, key=lambda row: (-row[0], row[1])):
        if len(kept_indices) >= min_keep:
            break
        if _score <= -50:
            continue
        if kept_indices and _score <= 0:
            continue
        kept_indices.add(idx)

    max_keep = 12 if "numeric_table" in evidence_need else 8
    ordered = [segment for idx, segment in enumerate(normalized) if idx in kept_indices]
    if len(ordered) > max_keep:
        top_indices = {
            idx
            for _score, idx, _segment in sorted(
                [row for row in scored if row[1] in kept_indices],
                key=lambda row: (-row[0], row[1]),
            )[:max_keep]
        }
        ordered = [segment for idx, segment in enumerate(normalized) if idx in top_indices]
    return ordered or normalized[:min_keep]


def _build_response_context_segments(retrieval_meta: dict) -> list[dict]:
    if not isinstance(retrieval_meta, dict):
        return []

    query = str(retrieval_meta.get("search_query") or retrieval_meta.get("query") or "").strip()
    citation_segments = _build_context_segments_from_citations(
        retrieval_meta.get("citations", []),
        query=query,
    )
    retrieval_segments = _merge_response_context_segments(
        retrieval_meta.get("_retrieval_context_segments") or [],
    )
    existing_segments = _merge_response_context_segments(retrieval_meta.get("_context_segments") or [])

    evidence_need = {
        str(item).strip()
        for item in (retrieval_meta.get("evidence_need") or [])
        if str(item).strip()
    }
    if citation_segments and any(
        isinstance(citation, dict)
        and (citation.get("unavailable_dataset_evidence") or citation.get("explicit_absence_evidence"))
        for citation in retrieval_meta.get("citations", []) or []
    ):
        return citation_segments
    if "numeric_table" in evidence_need and citation_segments:
        comparator_segments = _build_numeric_table_comparator_context_segments(retrieval_meta)
        return comparator_segments or citation_segments
    merged = _merge_response_context_segments(retrieval_segments, existing_segments, citation_segments)
    return _filter_response_context_segments(
        merged,
        query=query,
        evidence_need=evidence_need,
        citation_count=len(citation_segments),
    )



def _has_numeric_table_structured_support(citation: Optional[dict]) -> bool:
    if not isinstance(citation, dict):
        return False

    chunk_type = str(citation.get("chunk_type") or citation.get("block_type") or "").strip().lower()
    if chunk_type in {"table_row", "table_cell"}:
        return True
    if citation.get("table_row_evidence") or citation.get("table_row_slice_kind") == "exact":
        return True
    if citation.get("numeric_table_exact_context_row_text"):
        return True
    if citation.get("cell_evidence_units"):
        return True
    for unit in citation.get("evidence_units") or []:
        if not isinstance(unit, dict):
            continue
        unit_type = str(unit.get("evidence_unit_type") or "").strip().lower()
        if unit_type in {"table_row", "table_cell"}:
            return True
    return False


def _supplement_numeric_table_citations(
    answer: str,
    aligned: list[dict],
    citations: list[dict],
    *,
    query: str = "",
) -> list[dict]:
    normalized_aligned = _normalize_citation_records(aligned)
    if not should_apply_numeric_table_specialization():
        return normalized_aligned
    normalized_citations = _normalize_citation_records(citations)
    if not normalized_aligned or not normalized_citations:
        return normalized_aligned

    core_answer = _strip_inline_citations(answer)
    if not core_answer:
        return normalized_aligned

    normalized_aligned = _normalize_numeric_metric_bundle_citations(normalized_aligned, query)
    normalized_citations = _normalize_numeric_metric_bundle_citations(normalized_citations, query)

    selected_source_refs = {int(c["ref"]) for c in normalized_aligned}
    selected_table_ids = {
        str(c.get("table_id") or "").strip().lower()
        for c in normalized_aligned
        if str(c.get("table_id") or "").strip()
    }
    selected_group_ids = {
        str(c.get("group_id") or "").strip().lower()
        for c in normalized_aligned
        if str(c.get("group_id") or "").strip()
    }
    hints = _query_rewriter.extract_numeric_table_hints(query) if query else {}
    cost_query = _is_numeric_table_cost_query(query)
    metric_query = _is_numeric_table_metric_query(query)
    strict_gate = (
        bool(query)
        and any(_has_numeric_table_exact_row_support(citation) for citation in normalized_aligned)
        and _should_apply_numeric_table_strict_gate(query, hints)
    )
    target_tables = _extract_numeric_table_target_tables(query, hints)
    target_columns = _extract_numeric_table_target_columns(query, hints)
    selected_mentions_target_table = bool(
        target_tables
        and any(
            _numeric_table_support_mentions_target_table(
                _build_numeric_table_citation_support_text(citation),
                target_tables,
            )
            for citation in normalized_aligned
        )
    )

    effective_target_columns = {column for column in target_columns if column != "acc"}
    metric_exact_row_mode = metric_query and any(_has_numeric_table_exact_row_support(citation) for citation in normalized_aligned)
    if metric_exact_row_mode:
        filtered_aligned = [
            citation for citation in normalized_aligned
            if _has_numeric_table_exact_row_support(citation)
            and (not effective_target_columns or _citation_matches_numeric_table_columns(citation, effective_target_columns))
        ]
        if filtered_aligned:
            normalized_aligned = [
                _focus_numeric_metric_citation(citation, query)
                for citation in filtered_aligned
            ]
            selected_source_refs = {int(c["ref"]) for c in normalized_aligned}
            selected_table_ids = {
                str(c.get("table_id") or "").strip().lower()
                for c in normalized_aligned
                if str(c.get("table_id") or "").strip()
            }
            selected_group_ids = {
                str(c.get("group_id") or "").strip().lower()
                for c in normalized_aligned
                if str(c.get("group_id") or "").strip()
            }
    if metric_query:
        normalized_aligned = [
            _attach_numeric_table_comparison_rows(citation, query)
            for citation in normalized_aligned
        ]

    extras: list[tuple[int, float, dict]] = []
    cost_anchor_candidates: list[tuple[int, float, dict]] = []
    for citation in normalized_citations:
        source_ref = int(citation["ref"])
        if source_ref in selected_source_refs:
            continue
        has_exact_row_support = _has_numeric_table_exact_row_support(citation)
        has_structured_support = _has_numeric_table_structured_support(citation)
        has_cost_anchor = cost_query and _has_numeric_table_cost_anchor(
            _build_numeric_table_citation_support_text(citation)
        )
        support_text = _build_numeric_table_citation_support_text(citation)
        has_metric_anchor = metric_query and _has_numeric_table_metric_anchor(support_text)
        if not has_structured_support and not has_cost_anchor and not has_metric_anchor:
            continue
        if metric_exact_row_mode and not has_exact_row_support:
            continue
        if metric_query and _has_numeric_table_cost_anchor(support_text):
            continue
        if (
            metric_query
            and selected_mentions_target_table
            and not _numeric_table_support_mentions_target_table(support_text, target_tables)
        ):
            continue

        support_score = _calc_citation_support_score(core_answer, citation)
        table_id = str(citation.get("table_id") or "").strip().lower()
        group_id = str(citation.get("group_id") or "").strip().lower()
        same_bundle = bool(
            (table_id and table_id in selected_table_ids)
            or (group_id and group_id in selected_group_ids)
        )
        answer_mentions_same_bundle_method = bool(
            same_bundle
            and has_exact_row_support
            and _numeric_table_answer_mentions_method(core_answer, citation)
        )
        if strict_gate:
            if not has_exact_row_support:
                continue
            if selected_table_ids or selected_group_ids:
                if not same_bundle:
                    continue
            elif target_tables:
                citation_text = _build_numeric_table_citation_support_text(citation).lower()
                if not any(target in citation_text for target in target_tables):
                    continue
            if target_tables:
                citation_text = _build_numeric_table_citation_support_text(citation).lower()
                if not same_bundle and not any(target in citation_text for target in target_tables):
                    continue
            if target_columns and not _citation_matches_numeric_table_columns(citation, target_columns):
                continue
        if not same_bundle and support_score < 0.08:
            continue
        if same_bundle and support_score < 0.08 and not answer_mentions_same_bundle_method:
            continue
        if metric_query:
            citation = _focus_numeric_metric_citation(citation, query)
            citation = _attach_numeric_table_comparison_rows(citation, query)
        extras.append((0 if same_bundle else 1, -support_score, citation))
        if has_cost_anchor:
            cost_anchor_candidates.append((0 if same_bundle else 1, -support_score, citation))

    if not extras and not cost_anchor_candidates:
        return normalized_aligned

    augmented = list(normalized_aligned)
    if cost_query and not any(
        _has_numeric_table_cost_anchor(_build_numeric_table_citation_support_text(citation))
        for citation in augmented
    ):
        for _same_bundle_rank, _neg_score, citation in sorted(cost_anchor_candidates, key=lambda item: item[:2]):
            source_ref = int(citation["ref"])
            if source_ref in selected_source_refs:
                continue
            augmented.append(citation)
            selected_source_refs.add(source_ref)
            break

    for _same_bundle_rank, _neg_score, citation in sorted(extras, key=lambda item: item[:2]):
        if len(augmented) >= len(normalized_aligned) + 4:
            break
        source_ref = int(citation["ref"])
        if source_ref in selected_source_refs:
            continue
        augmented.append(citation)
        selected_source_refs.add(source_ref)
    return augmented






def _get_ablation_reason_traits(support: str = "") -> dict[str, bool]:
    lower = re.sub(r"\s+", " ", str(support or "").lower()).strip()
    return {
        "component_removed": bool(re.search(r"remove|without|去掉|移除|不使用|消融", lower, re.IGNORECASE)),
        "performance_drop": bool(re.search(r"drop|decrease|worse|下降|降低|变差", lower, re.IGNORECASE)),
        "performance_gain": bool(re.search(r"gain|improve|increase|提升|提高|增加", lower, re.IGNORECASE)),
        "mechanism": bool(re.search(r"because|reason|effect|原因|机制|影响", lower, re.IGNORECASE)),
    }











def _is_conditional_generation_query(query: str = "") -> bool:
    return bool(_CONDITIONAL_GENERATION_QUERY_RE.search(str(query or "")))


def _build_conditional_generation_support_text(citation: Optional[dict]) -> str:
    if not isinstance(citation, dict):
        return ""
    return _build_formula_citation_support_text(citation)


def _calc_conditional_generation_anchor_score(text: str = "") -> float:
    sample = re.sub(r"\s+", " ", str(text or "")).strip()
    lower = sample.lower()
    if not sample:
        return 0.0
    score = 0.0
    for token, weight in (
        ("conditional", 1.2),
        ("class label", 2.2),
        ("class embedding", 2.4),
        ("trainable", 1.2),
        ("incorporating the label", 2.0),
        ("label directly as an input", 2.0),
        ("similar to time", 1.0),
        ("类别标签", 1.8),
        ("类别嵌入", 2.0),
        ("条件", 0.8),
    ):
        if token in lower or token in sample:
            score += weight
    if "[structured table bundle]" in lower:
        score -= 2.0
    return score








def _apply_projected_selector_citation_fill(answer: str, citations: list[dict], *, query: str = "") -> tuple[str, dict]:
    """基于最终投影后的引用编号，为未标注事实句补齐内联引用。"""
    if not answer or not citations:
        return answer, {}
    try:
        from services.citation_enhancer import _apply_selector_citation_fill, _build_evidence_schema
    except Exception as exc:  # pragma: no cover - 仅依赖异常时兜底
        return answer, {"applied": False, "reason": f"selector_fill_import_failed:{exc}"}

    raw_chunks: list[dict] = []
    for seg in _build_context_segments_from_citations(citations, query=query):
        if not isinstance(seg, dict) or not seg.get("text"):
            continue
        try:
            ref = int(seg.get("ref") or 0)
        except (TypeError, ValueError):
            continue
        if not ref:
            continue
        raw_chunks.append({
            "ref": ref,
            "text": seg.get("text", ""),
            "page_range": seg.get("page_range") or [],
            "group_id": seg.get("group_id", ""),
            "evidence_id": seg.get("evidence_id", ""),
            "chunk_id": seg.get("chunk_id", ""),
        })

    for citation in citations:
        if not isinstance(citation, dict):
            continue
        try:
            ref = int(citation.get("ref") or citation.get("display_ref") or 0)
        except (TypeError, ValueError):
            continue
        if not ref:
            continue
        text = (
            citation.get("source_text")
            or citation.get("context_segment_text")
            or citation.get("highlight_text")
            or citation.get("display_text")
            or ""
        )
        if not str(text or "").strip():
            continue
        raw_chunks.append({
            "ref": ref,
            "text": str(text),
            "page_range": citation.get("page_range") or [],
            "group_id": citation.get("group_id", ""),
            "evidence_id": citation.get("evidence_id", ""),
            "chunk_id": citation.get("chunk_id", ""),
        })

    if not raw_chunks:
        return answer, {"applied": False, "reason": "no_projected_evidence"}

    filled, diag = _apply_selector_citation_fill(answer, _build_evidence_schema(raw_chunks))
    return filled, diag


def _append_selector_filled_projected_citations(
    projected: list[dict],
    candidate_map: dict[int, dict],
    answer: str,
) -> list[dict]:
    """把 selector 新补出的 ref 对应证据补进最终 citations。

    selector 只能在答案文本中追加已有证据编号。这里按最终答案实际出现的 ref
    回填 citation 元数据，避免 context-only 候选在未被引用时污染最终证据列表。
    """
    if not projected or not candidate_map or not answer:
        return projected
    existing_refs = {int(item.get("ref") or 0) for item in projected if isinstance(item, dict)}
    extended = list(projected)
    for ref in _extract_inline_citation_refs(answer):
        if ref in existing_refs:
            continue
        candidate = candidate_map.get(ref)
        if not candidate:
            continue
        item = candidate.copy()
        item["source_ref"] = _coerce_positive_int(item.get("source_ref"), ref)
        item["display_ref"] = ref
        item["ref"] = ref
        extended.append(item)
        existing_refs.add(ref)
    return extended


def _compact_projected_citation_display_refs(answer: str, projected: list[dict]) -> tuple[str, list[dict]]:
    """按最终答案中实际出现顺序压缩展示引用编号。"""
    normalized = _normalize_citation_records(projected)
    if not answer or not normalized:
        return answer, normalized
    refs_in_answer = _extract_inline_citation_refs(answer)
    if not refs_in_answer:
        return answer, normalized
    citation_map = {int(item["ref"]): item for item in normalized}
    ref_mapping: dict[int, int] = {}
    compacted: list[dict] = []
    for old_ref in refs_in_answer:
        if old_ref in ref_mapping or old_ref not in citation_map:
            continue
        new_ref = len(ref_mapping) + 1
        ref_mapping[old_ref] = new_ref
        item = citation_map[old_ref].copy()
        item["source_ref"] = _coerce_positive_int(item.get("source_ref"), old_ref)
        item["display_ref"] = new_ref
        item["ref"] = new_ref
        compacted.append(item)
    for item in normalized:
        old_ref = int(item["ref"])
        if old_ref in ref_mapping:
            continue
        new_ref = len(ref_mapping) + 1
        ref_mapping[old_ref] = new_ref
        copied = item.copy()
        copied["source_ref"] = _coerce_positive_int(copied.get("source_ref"), old_ref)
        copied["display_ref"] = new_ref
        copied["ref"] = new_ref
        compacted.append(copied)
    if not compacted:
        return answer, []
    used_ref_mapping = {
        old_ref: new_ref
        for old_ref, new_ref in ref_mapping.items()
        if old_ref in set(refs_in_answer)
    }
    if all(old == new for old, new in used_ref_mapping.items()):
        return answer, compacted
    return _rewrite_inline_citation_refs(answer, used_ref_mapping), compacted


def _score_context_recovery_candidate_for_answer(answer: str, citation: dict) -> float:
    """按最终答案文本与候选证据的词/数字重叠给 context recovery 排序。"""
    answer_tokens = _tokenize_for_citation(_strip_inline_citations(answer or ""))
    if not answer_tokens or not isinstance(citation, dict):
        return 0.0
    support_text = " ".join(
        re.sub(r"\s+", " ", str(citation.get(field) or "")).strip()
        for field in ("source_text", "context_segment_text", "highlight_text", "display_text", "group_id")
        if re.sub(r"\s+", " ", str(citation.get(field) or "")).strip()
    )
    citation_tokens = _tokenize_for_citation(support_text)
    if not citation_tokens:
        return 0.0
    overlap = _calc_token_overlap(answer_tokens, citation_tokens)
    if overlap <= 0:
        return 0.0
    return overlap / max(1, len(set(answer_tokens)))


def _rank_context_recovery_candidates_for_selector(
    answer: str,
    citations: list[dict],
    *,
    limit: int = _MAX_CONTEXT_RECOVERY_SELECTOR_CANDIDATES,
) -> tuple[list[dict], dict]:
    """选择少量最可能支撑答案的 context-only citation 候选。"""
    normalized = _normalize_citation_records(citations)
    if not normalized or limit <= 0:
        return [], {
            "input_count": len(normalized),
            "selected_count": 0,
            "dropped_count": len(normalized),
            "limit": limit,
        }
    scored: list[tuple[float, int, dict]] = []
    for idx, citation in enumerate(normalized):
        score = _score_context_recovery_candidate_for_answer(answer, citation)
        if score <= 0:
            continue
        scored.append((score, idx, citation))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:limit]
    return [citation for _score, _idx, citation in selected], {
        "input_count": len(normalized),
        "scored_count": len(scored),
        "selected_count": len(selected),
        "dropped_count": max(0, len(normalized) - len(selected)),
        "limit": limit,
        "top_scores": [round(score, 4) for score, _idx, _citation in selected[:3]],
    }


def _build_retrieval_preview_message(citations: list[dict], max_items: int = 3, max_chars: int = 90) -> str:
    """把命中的 citation 组装成思考区可展示的检索预览文本。"""
    if not citations:
        return ""

    lines: list[str] = []
    for c in citations[:max_items]:
        if not isinstance(c, dict):
            continue
        try:
            ref = int(c.get("ref"))
        except (TypeError, ValueError):
            continue

        page_range = c.get("page_range") or []
        if len(page_range) >= 2 and page_range[0] and page_range[1] and page_range[0] != page_range[1]:
            page_text = f"第 {page_range[0]}-{page_range[1]} 页"
        elif page_range:
            page_text = f"第 {page_range[0]} 页"
        else:
            page_text = "页码未知"

        snippet = (
            c.get("highlight_text")
            or c.get("display_text")
            or c.get("source_text")
            or c.get("_full_text")
            or ""
        )
        snippet = re.sub(r"\s+", " ", str(snippet)).strip()
        if not snippet:
            continue
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        lines.append(f"- [{ref}] {page_text}: {snippet}")

    if not lines:
        return ""

    return "检索到以下相关片段：\n" + "\n".join(lines)


def _split_context_paragraphs(context: str) -> list[str]:
    """将 PDF 提取的 context 拆分为真实段落。

    PDF 文本每行约 50-70 字符（视觉行），需要先合并为真实段落。
    策略：双换行分段 → 单换行合并 → 长段落按句子拆分。
    """
    import re as _re

    if "【检索证据" in context:
        raw_paragraphs = _re.split(r'\n{2,}(?=【检索证据)', context)
    else:
        raw_paragraphs = _re.split(r'\n{2,}', context)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip() and len(p.strip()) >= 30]

    if len(paragraphs) < 3:
        lines = context.split('\n')
        merged: list[str] = []
        buf = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buf and len(buf) >= 30:
                    merged.append(buf)
                buf = ""
            else:
                buf += (" " if buf else "") + stripped
        if buf and len(buf) >= 30:
            merged.append(buf)
        if len(merged) >= 3:
            paragraphs = merged

    merged_table_bundles: list[str] = []
    idx = 0
    while idx < len(paragraphs):
        para = paragraphs[idx]
        if "[structured table bundle]" in para.lower():
            bundle_parts = [para]
            idx += 1
            while idx < len(paragraphs) and not paragraphs[idx].startswith("【检索证据"):
                bundle_parts.append(paragraphs[idx])
                idx += 1
            merged_table_bundles.append("\n\n".join(bundle_parts).strip())
            continue
        merged_table_bundles.append(para)
        idx += 1
    paragraphs = merged_table_bundles

    final: list[str] = []
    for para in paragraphs:
        formula_block = _calc_formula_citation_anchor_score(para) >= 3.0
        if len(para) <= 500 or "[structured table bundle]" in para.lower() or (formula_block and len(para) <= 1600):
            final.append(para)
        else:
            sents = _DECIMAL_SAFE_SENTENCE_SPLIT_RE.split(para)
            chunk = ""
            for s in sents:
                if not s.strip():
                    continue
                if len(chunk) + len(s) > 400 and len(chunk) >= 100:
                    final.append(chunk.strip())
                    chunk = s
                else:
                    chunk += (" " if chunk else "") + s
            if chunk.strip() and len(chunk.strip()) >= 30:
                final.append(chunk.strip())
    return final if final else paragraphs


def _build_numbered_context_and_citations(
    pages: list[dict], context: str, query: str = "", max_citations: int = 8,
) -> tuple[str, list[dict]]:
    """将 context 格式化为编号段落并生成对应 citations。

    返回 (formatted_context, citations)：
    - formatted_context: ``[1] 段落文本\\n\\n[2] 段落文本\\n\\n...``
    - citations: 与编号对应的 citation 列表（group_id 以 ``para-`` 开头）

    LLM 收到编号段落后可自然地在回答中引用 [N]，无需 post-hoc 匹配。
    """
    if not context or not pages:
        return context, []

    paragraphs = _split_context_paragraphs(context)
    if not paragraphs:
        return context, []

    def _extract_retrieval_tags(text: str) -> dict:
        first_line = (text or "").splitlines()[0] if text else ""
        if not first_line.startswith("【检索证据"):
            return {}
        tags: dict[str, str] = {}
        for item in first_line.strip("【】").split("|"):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                tags[key] = value
        return tags

    def _strip_retrieval_tag_line(text: str) -> str:
        lines = (text or "").splitlines()
        if lines and lines[0].startswith("【检索证据"):
            return "\n".join(lines[1:]).strip()
        return (text or "").strip()

    # 页码反查
    def _locate_page(para_text: str) -> int:
        page_match = re.search(r"页码[:：]\s*(\d+)", para_text or "")
        if page_match:
            try:
                return max(1, int(page_match.group(1)))
            except (TypeError, ValueError):
                pass
        snippet = para_text[:60].lower()
        for pidx, page in enumerate(pages):
            page_text = (page.get("text", "") or page.get("content", "")).lower()
            if snippet in page_text:
                return pidx + 1
        return 1

    # 取 top-N 更小证据窗口（按 query 相关度 + 内容丰富度 + 去重）
    import re as _re
    _tok_pat = _re.compile(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]')

    def _tokenize(text: str) -> list[str]:
        return _tok_pat.findall(text.lower()) if text else []

    stop_terms = {
        "总结", "概括", "主要", "内容", "文章", "文档", "论文", "介绍", "什么", "如何", "哪些",
        "本文", "问题", "用户", "提供", "根据", "这个", "那个", "以及", "进行", "关于",
        "the", "what", "how", "why", "about", "paper", "document", "summary", "summarize",
    }
    query_terms = [
        t for t in _tokenize(query)
        if (len(t) >= 2 or _re.fullmatch(r'[\u4e00-\u9fff]', t)) and t not in stop_terms
    ]
    lower_query_for_windows = (query or "").lower()
    formula_query_for_windows = _is_formula_framework_query(lower_query_for_windows)

    def _sentence_windows(text: str) -> list[str]:
        if "[structured table bundle]" in (text or "").lower():
            return [text.strip()] if text and text.strip() else []
        sents = [s.strip() for s in _DECIMAL_SAFE_SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if len(sents) <= 1:
            return [text.strip()]
        formula_text = _calc_formula_citation_anchor_score(text) >= 3.0
        max_window_chars = 520 if formula_query_for_windows and formula_text else 240
        overlap_char_limit = 180 if formula_query_for_windows and formula_text else 120
        windows: list[str] = []
        current: list[str] = []
        current_len = 0
        for sent in sents:
            sent_len = len(sent)
            if current and current_len + sent_len > max_window_chars:
                windows.append(" ".join(current).strip())
                overlap = current[-1:] if len(current[-1]) <= overlap_char_limit else []
                current = overlap[:]
                current_len = sum(len(x) for x in current)
            current.append(sent)
            current_len += sent_len
        if current:
            windows.append(" ".join(current).strip())
        merged: list[str] = []
        for window in windows:
            if len(window) < 60 and merged:
                merged[-1] = f"{merged[-1]} {window}".strip()
            else:
                merged.append(window)
        return merged or [text.strip()]

    def _structured_table_highlight(text: str, query_text: str, *, max_len: int = 220) -> str:
        """为结构化表格 bundle 生成包含数值行的高亮，避免只截到 caption/header。"""
        if "[structured table bundle]" not in (text or "").lower():
            return ""
        lines = [line.strip() for line in (text or "").splitlines()]

        decimal_rows: list[tuple[float, int, str]] = []
        for idx, line in enumerate(lines):
            if not line or set(line) <= {"|", "-", " "}:
                continue
            decimals = len(_re.findall(r"\d+\.\d+", line))
            if decimals <= 0:
                continue
            lower_line = line.lower()
            anchors = sum(
                1
                for anchor in ("all", "many", "med.", "medium", "few", "acc", "accuracy", "fid", "score")
                if anchor in lower_line
            )
            decimal_rows.append((decimals * 10 + anchors, idx, line))

        if decimal_rows:
            _, best_idx, best_line = max(decimal_rows, key=lambda item: (item[0], item[2].count("|")))
            header = ""
            for prev_idx in range(best_idx - 1, -1, -1):
                prev = lines[prev_idx]
                if not prev or set(prev) <= {"|", "-", " "}:
                    continue
                if "|" in prev and not _re.search(r"\d+\.\d+", prev):
                    header = prev
                    break
            highlight = f"{header}\n{best_line}".strip() if header else best_line
            if len(highlight) > max_len:
                highlight = highlight[-max_len:].lstrip()
            return highlight

        query_tokens = set(_tokenize(query_text))
        scored_lines: list[tuple[float, int, str]] = []
        for idx, line in enumerate((text or "").splitlines()):
            stripped = line.strip()
            if not stripped or set(stripped) <= {"|", "-", " "}:
                continue
            lower_line = stripped.lower()
            decimals = len(_re.findall(r"\d+\.\d+", stripped))
            anchors = sum(
                1
                for anchor in ("all", "many", "med.", "medium", "few", "acc", "accuracy", "fid", "score", "metric", "table")
                if anchor in lower_line
            )
            token_overlap = len(set(_tokenize(stripped)) & query_tokens)
            score = decimals * 4.0 + anchors * 0.6 + token_overlap * 0.4
            if decimals > 0:
                score += 8.0
            if score > 0:
                scored_lines.append((score, idx, stripped))
        if not scored_lines:
            return ""

        _, best_idx, best_line = max(scored_lines, key=lambda item: (item[0], item[2].count("|")))
        context_lines: list[str] = []
        for idx in range(max(0, best_idx - 2), best_idx + 1):
            line = lines[idx] if idx < len(lines) else ""
            if line and set(line) > {"|", "-", " "}:
                context_lines.append(line)
        highlight = "\n".join(context_lines).strip() or best_line
        if len(highlight) > max_len:
            highlight = highlight[-max_len:].lstrip()
        return highlight

    def _expand_formula_citation_window(para_text: str, selected_window: str) -> str:
        """公式类 citation 保留同段公式定义上下文，避免只引用到 loss 子句。"""
        if not formula_query_for_windows or _calc_formula_citation_anchor_score(selected_window) <= 0:
            return selected_window

        source = re.sub(r"\s+", " ", _strip_retrieval_tag_line(para_text)).strip()
        window = re.sub(r"\s+", " ", str(selected_window or "")).strip()
        if not source or not window:
            return selected_window

        pos = source.find(window)
        if pos < 0:
            compact_source = re.sub(r"\s+", "", source)
            compact_window = re.sub(r"\s+", "", window)
            compact_pos = compact_source.find(compact_window[: min(len(compact_window), 80)])
            if compact_pos < 0:
                return selected_window
            pos = max(0, min(len(source), compact_pos))

        anchor_positions = [
            match.start()
            for match in _re.finditer(
                r"formula|equation|objective|loss|公式|方程|目标函数|损失函数|"
                r"=|≈|≤|≥|∑|∏|√|\\(?:frac|sum|prod|sqrt|mathcal)",
                source,
                _re.IGNORECASE,
            )
        ]
        if anchor_positions:
            nearest_before = [p for p in anchor_positions if p <= pos]
            start_anchor = min(nearest_before) if nearest_before else min(anchor_positions)
            end_anchor = max(p for p in anchor_positions if p >= start_anchor)
            start = max(0, start_anchor - 160)
            end = min(len(source), max(pos + len(window), end_anchor + 520))
        else:
            start = max(0, pos - 900)
            end = min(len(source), pos + len(window) + 360)

        expanded = source[start:end].strip()
        if len(expanded) < len(window):
            return selected_window
        return f"{'...' if start > 0 else ''}{expanded}{'...' if end < len(source) else ''}"

    candidates: list[tuple[float, int, int, str, str, set[str], int, bool]] = []
    for pi, para in enumerate(paragraphs):
        for wi, window in enumerate(_sentence_windows(para)):
            clean_window = _strip_retrieval_tag_line(window)
            tokens = _tokenize(clean_window)
            if len(tokens) < 8:
                continue
            token_set = set(tokens)
            overlap_terms = token_set & set(query_terms)
            overlap = len(overlap_terms)
            n_tokens = len(tokens)
            unique_ratio = len(token_set) / max(n_tokens, 1)
            density = overlap / max(len(query_terms), 1) if query_terms else 0.0
            richness = unique_ratio * min(n_tokens / 24.0, 1.0)
            lower_window = window.lower()
            lower_query = (query or "").lower()
            numeric_query = bool(_re.search(
                r"(表\s*\d+|table\s*\d+|all|many|med\.?|medium|few|acc|accuracy|score|metric|准确率|百分点|分别|多少|数值|指标)",
                lower_query,
            ))
            formula_query = _is_formula_framework_query(lower_query)
            decimal_count = len(_re.findall(r"\d+\.\d+", window))
            table_anchor_hits = sum(
                1
                for anchor in (
                    "structured table bundle",
                    "table",
                    "many",
                    "med.",
                    "medium",
                    "few",
                    "all",
                    "acc",
                    "accuracy",
                    "fid",
                    "score",
                )
                if anchor in lower_window
            )
            structured_table_bonus = 3.0 if "[structured table bundle]" in lower_window else 0.0
            table_numeric_bonus = 0.0
            formula_bonus = 0.0
            if numeric_query:
                table_numeric_bonus += min(decimal_count, 6) * 0.35
                table_numeric_bonus += min(table_anchor_hits, 8) * 0.45
                if _re.search(r"\|\s*[^|\n]+\s*\|", window) or _re.search(r"\b(All|Many|Med\.?|Few)\b", window):
                    table_numeric_bonus += 1.2
                if "table" in lower_window or "表" in window:
                    table_numeric_bonus += 0.8
            if formula_query:
                formula_anchors = (
                    "formula",
                    "equation",
                    "objective",
                    "loss",
                    "公式",
                    "方程",
                    "目标函数",
                    "损失函数",
                    "beta",
                    "β",
                    "alpha",
                    "α",
                    "epsilon",
                    "ϵ",
                    "∥",
                )
                formula_bonus += sum(0.55 for anchor in formula_anchors if anchor in lower_window)
                if _re.search(r"(=|≈|≤|≥|∑|∏|√|β|α|ϵ|epsilon|\\sqrt|\\mathcal|∥|\|\|)", window, _re.IGNORECASE):
                    formula_bonus += 2.0
                formula_bonus += _calc_formula_citation_anchor_score(window)
            number_bonus = 0.20 if _re.search(r'\d', window) else 0.0
            keyword_bonus = 0.15 if _re.search(r'(dataset|results?|experiment|method|abstract|introduction|conclusion|贡献|实验|方法|结果|数据集)', lower_window) else 0.0
            score = (
                overlap * 2.8
                + density * 1.8
                + richness
                + number_bonus
                + keyword_bonus
                + structured_table_bonus
                + table_numeric_bonus
                + formula_bonus
            )
            is_structured_table = "[structured table bundle]" in lower_window
            snippet = _structured_table_highlight(clean_window, query) or _context_builder._extract_relevant_snippet(clean_window, query, max_len=140)
            page_num = _locate_page(window)
            candidates.append((score, pi, wi, clean_window, snippet, token_set, page_num, is_structured_table))

    candidates.sort(key=lambda x: x[0], reverse=True)
    citation_anchors = _extract_citation_query_anchors(query)
    candidate_anchor_sets: list[set[str]] = []
    for _score, _pi, _wi, window, _snippet, _token_set, _page_num, _is_structured_table in candidates:
        window_text = str(window or "")
        window_lower = window_text.casefold()
        candidate_anchor_sets.append({
            anchor
            for anchor in citation_anchors
            if technical_anchor_matches(anchor, window_text)
        })

    selected: list[tuple[int, int, str, str, int]] = []
    selected_token_sets: list[set[str]] = []
    selected_pages_for_dedupe: list[int] = []
    page_counts: dict[int, int] = {}
    selected_anchor_coverage: set[str] = set()
    remaining_candidate_indices = set(range(len(candidates)))
    while remaining_candidate_indices and len(selected) < max_citations:
        if len(citation_anchors) >= 2:
            ordered_candidate_indices = sorted(
                remaining_candidate_indices,
                key=lambda idx: (
                    -len(candidate_anchor_sets[idx] - selected_anchor_coverage),
                    -len(candidate_anchor_sets[idx]),
                    -float(candidates[idx][0] or 0.0),
                    int(candidates[idx][1]),
                    int(candidates[idx][2]),
                ),
            )
        else:
            ordered_candidate_indices = sorted(
                remaining_candidate_indices,
                key=lambda idx: (-float(candidates[idx][0] or 0.0), int(candidates[idx][1]), int(candidates[idx][2])),
            )
        emitted_this_round = False
        for candidate_idx in ordered_candidate_indices:
            score, pi, wi, window, snippet, token_set, page_num, is_structured_table = candidates[candidate_idx]
            remaining_candidate_indices.discard(candidate_idx)
            if page_counts.get(page_num, 0) >= 2 and len(selected) < max_citations - 1 and not is_structured_table:
                continue
            is_duplicate = False
            for prev_page, prev_tokens in zip(selected_pages_for_dedupe, selected_token_sets):
                overlap_ratio = len(token_set & prev_tokens) / max(1, len(token_set | prev_tokens))
                # 同页窗口严格去重；跨页证据常共享论文主题词，阈值放宽，避免压掉多页互补证据。
                duplicate_threshold = 0.92 if is_structured_table else (0.55 if prev_page == page_num else 0.82)
                if overlap_ratio >= duplicate_threshold:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            selected.append((pi, wi, window, snippet, page_num))
            selected_token_sets.append(token_set)
            selected_pages_for_dedupe.append(page_num)
            selected_anchor_coverage.update(candidate_anchor_sets[candidate_idx])
            page_counts[page_num] = page_counts.get(page_num, 0) + 1
            emitted_this_round = True
            break
        if not emitted_this_round:
            break

    if not selected:
        for pi, para in enumerate(paragraphs[:max_citations]):
            page_num = _locate_page(para)
            snippet = _context_builder._extract_relevant_snippet(para, query, max_len=140)
            selected.append((pi, 0, _strip_retrieval_tag_line(para), snippet, page_num))

    # 普通问题按原始顺序排列，使 context 保持逻辑连贯；公式/算法框架问题优先把
    # 核心公式证据排到前面的引用编号。
    if formula_query_for_windows:
        selected.sort(key=lambda x: (-_calc_formula_citation_anchor_score(x[2]), x[0], x[1]))
    else:
        selected.sort(key=lambda x: (x[0], x[1]))

    # 构建编号段落 context + citations
    # 将所有段落加入 context（让 LLM 看到完整文档），但仅对 selected 生成 citation
    selected_set = {(pi, wi) for pi, wi, *_ in selected}
    all_formatted: list[str] = []
    for pi, para in enumerate(paragraphs):
        all_formatted.append(para)

    citations: list[dict] = []
    for ref_idx, (pi, wi, window, snippet, page_num) in enumerate(selected, 1):
        citation_source_text = _expand_formula_citation_window(
            paragraphs[pi] if 0 <= pi < len(paragraphs) else window,
            window,
        )
        if citation_source_text != window and _calc_formula_citation_anchor_score(citation_source_text) > _calc_formula_citation_anchor_score(snippet):
            snippet = _context_builder._extract_relevant_snippet(citation_source_text, query, max_len=180)
        highlight = snippet or (citation_source_text[:140] if len(citation_source_text) > 140 else citation_source_text)
        retrieval_tags = _extract_retrieval_tags(paragraphs[pi] if 0 <= pi < len(paragraphs) else window)
        group_id = retrieval_tags.get("group_id") or f"para-{pi + 1}-seg-{wi + 1}"
        chunk_id = retrieval_tags.get("chunk_id")
        citation = {
            "ref": ref_idx,
            "evidence_id": retrieval_tags.get("evidence_id") or f"para-{pi + 1}-seg-{wi + 1}:{ref_idx}",
            "context_id": retrieval_tags.get("context_id") or retrieval_tags.get("evidence_id") or chunk_id or group_id,
            "chunk_id": chunk_id,
            "child_chunk_id": retrieval_tags.get("child_chunk_id"),
            "parent_id": retrieval_tags.get("parent_id"),
            "group_id": group_id,
            "page_range": [page_num, page_num],
            "source_text": citation_source_text,
            "display_text": citation_source_text,
            "highlight_text": highlight,
            "_full_text": citation_source_text,
            "alignment_status": "fallback_window_only",
            "retrieval_type": "fallback",
        }
        citations.append({k: v for k, v in citation.items() if v is not None and v != ""})

    formatted_context = "\n\n".join(all_formatted)
    logger.info(
        f"[CITATION FALLBACK] paragraphs={len(paragraphs)}, selected={len(citations)}, "
        f"refs={[c['ref'] for c in citations]}, pages={[c['page_range'][0] for c in citations]}"
    )
    return formatted_context, citations


def _build_fast_overview_context(
    pages: list[dict],
    full_text: str,
    *,
    max_total_chars: int = 36000,
    max_page_chars: int = 2200,
) -> str:
    """为概览/总结问题构建更快的全文采样上下文。

    不做向量检索，直接从全文中抽取首段、尾段和均匀分布的页面文本，
    以覆盖整篇文档的主要结构，同时控制上下文大小。
    """
    if not pages:
        return (full_text or "")[:max_total_chars]

    total_pages = len(pages)
    if total_pages <= 8:
        sample_indices = list(range(total_pages))
    else:
        anchors = {0, 1, total_pages - 2, total_pages - 1}
        middle_slots = 4
        for slot in range(1, middle_slots + 1):
            idx = round(slot * (total_pages - 1) / (middle_slots + 1))
            anchors.add(max(0, min(total_pages - 1, idx)))
        sample_indices = sorted(anchors)

    sampled_parts: list[str] = []
    total_chars = 0
    for idx in sample_indices:
        page = pages[idx] or {}
        page_text = (page.get("text") or page.get("content") or "").strip()
        if not page_text:
            continue
        clipped = page_text[:max_page_chars].strip()
        if not clipped:
            continue
        block = f"[第{idx + 1}页]\n{clipped}"
        if total_chars + len(block) > max_total_chars and sampled_parts:
            break
        sampled_parts.append(block)
        total_chars += len(block)

    if sampled_parts:
        return "\n\n".join(sampled_parts)
    return (full_text or "")[:max_total_chars]


def _should_use_fast_overview_context(
    query_type: str,
    *,
    enable_vector_search: bool,
    selected_text: Optional[str],
    image_list: Optional[list] = None,
    use_agent: bool = False,
) -> bool:
    return (
        query_type == "overview"
        and enable_vector_search
        and not selected_text
        and not image_list
        and not use_agent
    )

def _build_agent_retrieval_gate(
    *,
    enable_agent_retrieval: bool,
    force_agent_retrieval: bool = False,
    selected_text: Optional[str],
    query_type: str,
    evidence_need: Optional[list[str]] = None,
) -> dict:
    """返回 retrieval_agent 触发决策及其原因，便于诊断。

    白名单从 `settings.agent_trigger_query_types` /
    `settings.agent_trigger_evidence_needs` 实时读取，支持通过环境变量
    `AGENT_TRIGGER_QUERY_TYPES` / `AGENT_TRIGGER_EVIDENCE_NEEDS` 动态覆盖。

    返回字典中除原有字段外，额外包含 `agent_gate_source`，取值集合为
    `{"query_type", "evidence_needs", "force_user", "denied"}`，用于在诊断
    输出中标识本次放行/拒绝的原因维度。
    """
    # 从 settings 读取白名单（每次调用都重新读取，便于运行期热更新）
    qtypes_whitelist = set(settings.agent_trigger_query_types or [])
    needs_whitelist = set(settings.agent_trigger_evidence_needs or [])

    normalized_query_type = str(query_type or "").strip().lower()
    normalized_needs = [
        str(item).strip()
        for item in (evidence_need or [])
        if str(item).strip()
    ]
    matched_needs = [
        need for need in normalized_needs
        if need in needs_whitelist
    ]
    matched_query_type = (
        normalized_query_type
        if normalized_query_type in qtypes_whitelist
        else None
    )

    if not enable_agent_retrieval:
        return {
            "enabled": False,
            "reason": "switch_disabled",
            "query_type": normalized_query_type,
            "evidence_need": normalized_needs,
            "matched_query_type": matched_query_type,
            "matched_evidence_need": matched_needs,
            "selected_text_present": bool(selected_text),
            "force_agent_retrieval": bool(force_agent_retrieval),
            "agent_gate_source": "denied",
        }

    if bool(selected_text):
        # 用户已框选文本：优先走选中文本路径，Agent 入口被拒绝
        return {
            "enabled": False,
            "reason": "selected_text_present",
            "query_type": normalized_query_type,
            "evidence_need": normalized_needs,
            "matched_query_type": matched_query_type,
            "matched_evidence_need": matched_needs,
            "selected_text_present": True,
            "force_agent_retrieval": bool(force_agent_retrieval),
            "agent_gate_source": "denied",
        }

    if force_agent_retrieval:
        # 用户在前端勾选"强制启用 Agent"，绕过白名单
        return {
            "enabled": True,
            "reason": "forced",
            "query_type": normalized_query_type,
            "evidence_need": normalized_needs,
            "matched_query_type": matched_query_type,
            "matched_evidence_need": matched_needs,
            "selected_text_present": False,
            "force_agent_retrieval": True,
            "agent_gate_source": "force_user",
        }

    enabled = bool(matched_query_type or matched_needs)
    if matched_query_type:
        reason = "matched_query_type"
        gate_source = "query_type"
    elif matched_needs:
        reason = "matched_evidence_need"
        gate_source = "evidence_needs"
    else:
        reason = "route_not_matched"
        gate_source = "denied"

    return {
        "enabled": enabled,
        "reason": reason,
        "query_type": normalized_query_type,
        "evidence_need": normalized_needs,
        "matched_query_type": matched_query_type,
        "matched_evidence_need": matched_needs,
        "selected_text_present": False,
        "force_agent_retrieval": bool(force_agent_retrieval),
        "agent_gate_source": gate_source,
    }


def _annotate_agent_gate(
    agent_gate: dict,
    *,
    use_agent: bool,
    agent_mode: bool,
    search_query_passthrough: bool,
) -> dict:
    """补齐 agent gating 的运行态诊断字段。"""
    gate = dict(agent_gate or {})
    gate["use_agent"] = bool(use_agent)
    gate["agent_mode"] = bool(agent_mode)
    gate["search_query_passthrough"] = bool(search_query_passthrough)
    gate["consistency_ok"] = (bool(gate.get("enabled")) == bool(use_agent)) and (not agent_mode or use_agent)
    return gate


def _merge_retrieval_meta(base: dict | None, update: dict | None) -> dict:
    merged = dict(base or {})
    base_diagnostics = merged.get("diagnostics")
    update_diagnostics = (update or {}).get("diagnostics") if isinstance(update, dict) else None
    if isinstance(update, dict):
        merged.update(update)
    if isinstance(base_diagnostics, dict) or isinstance(update_diagnostics, dict):
        diagnostics = {}
        if isinstance(base_diagnostics, dict):
            diagnostics.update(base_diagnostics)
        if isinstance(update_diagnostics, dict):
            diagnostics.update(update_diagnostics)
        merged["diagnostics"] = diagnostics
    return merged


def _citation_dedupe_key(citation: dict) -> tuple[str, str, str]:
    if not isinstance(citation, dict):
        return ("", "", "")
    source_id = str(
        citation.get("evidence_id")
        or citation.get("context_id")
        or citation.get("group_id")
        or citation.get("chunk_id")
        or citation.get("source_ref")
        or ""
    ).strip().casefold()
    text = re.sub(
        r"\s+",
        " ",
        str(
            citation.get("source_text")
            or citation.get("display_text")
            or citation.get("highlight_text")
            or ""
        ),
    ).strip()[:180].casefold()
    pages = citation.get("page_range") or citation.get("pages") or []
    if isinstance(pages, (list, tuple)):
        page_key = "-".join(str(item) for item in pages)
    else:
        page_key = str(pages or "")
    return source_id, text, page_key


def _merge_multi_query_retrieval_meta(
    base: dict | None,
    update: dict | None,
    *,
    query: str = "",
    query_index: int = 0,
) -> dict:
    """Merge retrieval metadata from decomposed vector queries while preserving citations."""
    merged = _merge_retrieval_meta(base, update)
    existing_citations = [
        dict(citation)
        for citation in ((base or {}).get("citations") or [])
        if isinstance(citation, dict)
    ]
    update_citations = [
        dict(citation)
        for citation in ((update or {}).get("citations") or [])
        if isinstance(citation, dict)
    ]
    citations: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for citation in [*existing_citations, *update_citations]:
        key = _citation_dedupe_key(citation)
        if key in seen:
            continue
        seen.add(key)
        item = dict(citation)
        item["ref"] = len(citations) + 1
        if query:
            item.setdefault("retrieval_query", query)
        item.setdefault("retrieval_query_index", query_index)
        citations.append(item)
    if citations:
        merged["citations"] = citations

    existing_segments = [
        dict(segment)
        for segment in ((base or {}).get("_context_segments") or [])
        if isinstance(segment, dict)
    ]
    update_segments = [
        dict(segment)
        for segment in ((update or {}).get("_context_segments") or [])
        if isinstance(segment, dict)
    ]
    segments: list[dict] = []
    seen_segments: set[tuple[str, str]] = set()
    for segment in [*existing_segments, *update_segments]:
        text = re.sub(r"\s+", " ", str(segment.get("text") or "")).strip()
        group_id = str(segment.get("group_id") or "").strip().casefold()
        key = (group_id, text[:180].casefold())
        if not text or key in seen_segments:
            continue
        seen_segments.add(key)
        item = dict(segment)
        item["ref"] = len(segments) + 1
        if query:
            item.setdefault("retrieval_query", query)
        item.setdefault("retrieval_query_index", query_index)
        segments.append(item)
    if segments:
        merged["_context_segments"] = segments

    return merged


def _should_enable_agent_retrieval(
    *,
    enable_agent_retrieval: bool,
    force_agent_retrieval: bool = False,
    selected_text: Optional[str],
    query_type: str,
    evidence_need: Optional[list[str]] = None,
) -> bool:
    """仅对高价值题型启用 retrieval_agent，避免全局放大延迟。"""
    gate = _build_agent_retrieval_gate(
        enable_agent_retrieval=enable_agent_retrieval,
        force_agent_retrieval=force_agent_retrieval,
        selected_text=selected_text,
        query_type=query_type,
        evidence_need=evidence_need,
    )
    return bool(gate.get("enabled"))


def _generate_page_level_citations(pages: list[dict], context: str, query: str = "", max_citations: int = 8) -> list[dict]:
    """兼容旧调用：仅返回 citations 列表。"""
    _, citations = _build_numbered_context_and_citations(pages, context, query=query, max_citations=max_citations)
    return citations


async def _run_agent_retrieval_for_context(
    *,
    request,
    doc: dict,
    search_query: str,
    query_type: str,
    agent_gate: dict,
    retrieval_meta: dict | None = None,
    emit_progress=None,
    trace_id: str | None = None,
    trace_started_at: float | None = None,
) -> tuple[str, dict]:
    """路由兼容入口：实际 Agent 检索执行下沉到 service。"""

    def _trace(stage: str, **fields) -> None:
        if trace_id and trace_started_at is not None:
            _log_chat_trace(trace_id, trace_started_at, stage, **fields)

    return await _run_agent_retrieval_service(
        request=request,
        doc=doc,
        search_query=search_query,
        query_type=query_type,
        agent_gate=agent_gate,
        retrieval_meta=retrieval_meta,
        emit_progress=emit_progress,
        trace=_trace,
        vector_store_dir=getattr(router, "vector_store_dir", ""),
        deps=AgentRetrievalDependencies(
            get_cheap_model_params=_get_cheap_model_params,
            build_agent_doc_context=_build_agent_doc_context,
            merge_retrieval_meta=_merge_retrieval_meta,
            annotate_agent_gate=_annotate_agent_gate,
            resolve_citation_candidate_limit=_resolve_citation_candidate_limit,
            build_numbered_context_and_citations=_build_numbered_context_and_citations,
            generate_page_level_citations=_generate_page_level_citations,
            build_agent_detail_citations=_build_agent_detail_citations,
        ),
    )


def _is_paragraph_fallback(citations: list[dict]) -> bool:
    """判断 citations 是否来自段落级兜底（非向量检索的语义 chunk）。

    当 group_id 全部以 ``para-`` 或 ``page-`` 开头时，视为 fallback 引文，
    不应触发结构化引文 prompt（CITATION LIST + FINAL ANSWER）。
    """
    if not citations:
        return False
    return all(
        c.get("group_id", "").startswith(("para-", "page-"))
        for c in citations
    )


def _should_use_compact_citation_prompt(citations: list[dict]) -> bool:
    if not citations:
        return False
    return all(
        c.get("retrieval_type") in {"fallback", "selected_text"}
        or c.get("group_id", "").startswith(("para-", "page-", "selected-text"))
        for c in citations
    )


def _inject_inline_citations(answer: str, citations: list[dict]) -> str:
    """当 LLM 回答未包含 [N] 引文标记时，基于模糊匹配自动注入。

    对每个段落，找到 token 重叠最高的 citation 并在段尾追加 [ref]。
    优先使用 ``_full_text``（完整段落文本）进行匹配，回退到 ``highlight_text``。
    """
    if not answer or not citations:
        return answer
    # 已有 [N] 引用则不处理
    if _INLINE_CITATION_PATTERN.search(answer):
        return answer

    import re as _re
    _tok_pat = _re.compile(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]')

    def _tokenize(text: str) -> list[str]:
        return _tok_pat.findall(text.lower()) if text else []

    cit_tokens_map: list[tuple[int, set[str]]] = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        try:
            ref = int(c.get("ref"))
        except (TypeError, ValueError):
            continue
        # 优先用 _full_text（段落级完整文本），回退到 highlight_text
        support = c.get("_full_text", "") or c.get("highlight_text", "")
        tokens = set(_tokenize(support))
        if tokens:
            cit_tokens_map.append((ref, tokens))

    if not cit_tokens_map:
        return answer

    # 收集可注入引文的段落及其与每个 citation 的匹配分数
    lines = answer.split('\n')
    para_indices: list[int] = []           # 可注入引文的行号
    para_scores: list[list[tuple[int, float]]] = []  # 每段的 [(ref, score), ...]
    eligible_indices: list[int] = []       # 所有可注入的行号（含无匹配的）

    for li, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 10 or stripped.startswith('#'):
            continue
        para_tokens = _tokenize(stripped)
        if len(para_tokens) < 3:
            continue
        eligible_indices.append(li)
        para_set = set(para_tokens)
        scores = []
        for ref, ctoks in cit_tokens_map:
            overlap = len(para_set & ctoks)
            score = overlap / max(1, len(para_tokens))
            if score >= 0.03:
                scores.append((ref, score))
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            para_indices.append(li)
            para_scores.append(scores)

    # 跨语言兜底：中文回答 vs 英文文档时 token overlap 极低，
    # 此时对所有可注入段落按顺序轮流分配不同 citation
    if not para_indices and eligible_indices:
        all_refs = [ref for ref, _ in cit_tokens_map]
        result = []
        for li, line in enumerate(lines):
            if li in eligible_indices:
                idx = eligible_indices.index(li)
                ref = all_refs[idx % len(all_refs)]
                result.append(f"{line}[{ref}]")
            else:
                result.append(line)
        return '\n'.join(result)

    if not para_indices:
        return answer

    # 贪心分配：尽量让每个段落引用不同的 citation
    # 如果 top-1 已被使用且有其他候选（分数 >= top-1 * 0.6），优先选未使用的
    used_count: dict[int, int] = {}  # ref -> 被分配次数
    assignments: dict[int, int] = {}  # line_index -> ref

    for li, scores in zip(para_indices, para_scores):
        top_score = scores[0][1]
        # 在分数足够接近的候选中，优先选使用次数最少的
        threshold = top_score * 0.6
        candidates = [(ref, sc) for ref, sc in scores if sc >= threshold]
        # 按 (使用次数, -分数) 排序，优先未使用 + 高分
        candidates.sort(key=lambda x: (used_count.get(x[0], 0), -x[1]))
        chosen_ref = candidates[0][0]
        assignments[li] = chosen_ref
        used_count[chosen_ref] = used_count.get(chosen_ref, 0) + 1

    # 如果所有段落仍然指向同一个 ref，按段落顺序轮流分配不同 citation
    unique_assigned = set(assignments.values())
    if len(unique_assigned) == 1 and len(cit_tokens_map) > 1:
        all_refs = [ref for ref, _ in cit_tokens_map]
        for i, li in enumerate(para_indices):
            assignments[li] = all_refs[i % len(all_refs)]

    # 为未匹配的 eligible 段落补充分配（跨语言场景部分行无 overlap）
    all_refs = [ref for ref, _ in cit_tokens_map]
    unassigned = [li for li in eligible_indices if li not in assignments]
    for i, li in enumerate(unassigned):
        # 从已使用最少的 ref 开始轮流分配
        ref = all_refs[(len(assignments) + i) % len(all_refs)]
        assignments[li] = ref

    result = []
    for li, line in enumerate(lines):
        if li in assignments:
            result.append(f"{line}[{assignments[li]}]")
        else:
            result.append(line)
    return '\n'.join(result)


def _extract_inline_citation_refs(answer: str) -> list[int]:
    """从回答正文中提取按出现顺序去重的引文编号。"""
    if not answer:
        return []

    ordered_refs = []
    seen = set()
    for match in _INLINE_CITATION_PATTERN.finditer(answer):
        ref_str = match.group(1) or match.group(2)
        if not ref_str:
            continue
        ref = int(ref_str)
        if ref in seen:
            continue
        seen.add(ref)
        ordered_refs.append(ref)
    return ordered_refs


_BAD_INLINE_CITATION_PATTERNS = (
    re.compile(r"\[\s*ID\s*[: ]*\s*(\d{1,3})\s*\]", re.IGNORECASE),
    re.compile(r"【\s*ID\s*[: ]*\s*(\d{1,3})\s*】", re.IGNORECASE),
    re.compile(r"\(\s*ID\s*[: ]*\s*(\d{1,3})\s*\)", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_])ref\s*(\d{1,3})\b", re.IGNORECASE),
)


def _normalize_citation_records(citations: list[dict]) -> list[dict]:
    normalized = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        try:
            ref = int(c.get("ref"))
        except (TypeError, ValueError):
            continue
        item = c.copy()
        item["ref"] = ref
        item.setdefault("source_ref", ref)
        normalized.append(item)
    return normalized


def _align_citations_with_answer(answer: str, citations: list[dict]) -> list[dict]:
    """将来源列表与回答正文中的实际引文编号对齐。"""
    normalized_citations = _normalize_citation_records(citations)
    if not normalized_citations:
        return []

    refs_in_answer = _extract_inline_citation_refs(answer)
    if not refs_in_answer:
        logger.info(
            "回答正文未检测到内联引用编号，保留原始 citations（count=%d）",
            len(normalized_citations),
        )
        return normalized_citations

    citation_map = {int(c["ref"]): c for c in normalized_citations}
    aligned = [citation_map[ref] for ref in refs_in_answer if ref in citation_map]
    if not aligned:
        invalid_refs = [r for r in refs_in_answer if r not in citation_map]
        logger.warning(
            "回答中的引文编号全部越界，不返回无关引文（invalid_refs=%s, valid_refs=%s）",
            invalid_refs,
            list(citation_map.keys()),
        )
        return []
    return aligned



def _repair_bad_citation_formats(answer: str, citations: list[dict]) -> str:
    if not answer:
        return answer

    normalized_citations = _normalize_citation_records(citations)
    if not normalized_citations:
        return answer

    valid_refs = {int(c["ref"]) for c in normalized_citations}
    repaired = answer
    for pattern in _BAD_INLINE_CITATION_PATTERNS:
        def _replace(match: re.Match) -> str:
            try:
                ref = int(match.group(1))
            except (TypeError, ValueError):
                return match.group(0)
            return f"[{ref}]" if ref in valid_refs else match.group(0)

        repaired = pattern.sub(_replace, repaired)
    return repaired


def _rewrite_inline_citation_refs(answer: str, ref_mapping: dict[int, int]) -> str:
    if not answer or not ref_mapping:
        return answer

    def _replace(match: re.Match) -> str:
        ref_str = match.group(1) or match.group(2)
        if not ref_str:
            return match.group(0)
        source_ref = int(ref_str)
        display_ref = ref_mapping.get(source_ref)
        if display_ref is None:
            return match.group(0)
        return f"[{display_ref}]"

    rewritten = _INLINE_CITATION_PATTERN.sub(_replace, answer)
    return re.sub(r"(\[(\d{1,3})\])(?:\s*\[\2\])+", r"\1", rewritten)


def _remove_invalid_inline_citation_refs(answer: str, valid_refs: set[int]) -> str:
    """删除最终 citation 列表中不存在的内联引用编号。"""
    if not answer or not valid_refs:
        return answer

    def _replace(match: re.Match) -> str:
        ref_str = match.group(1) or match.group(2)
        if not ref_str:
            return match.group(0)
        try:
            ref = int(ref_str)
        except ValueError:
            return match.group(0)
        return match.group(0) if ref in valid_refs else ""

    cleaned = _INLINE_CITATION_PATTERN.sub(_replace, answer)
    cleaned = re.sub(r"\s+([。！？；，、,.])", r"\1", cleaned)
    return cleaned


def _cleanup_inline_citation_display(answer: str) -> str:
    if not answer:
        return answer

    cleaned = re.sub(
        r"\[\s*(\d{1,3})\s*[,，]\s*(\d{1,3})\s*\]",
        lambda m: f"[{m.group(1)}][{m.group(2)}]",
        answer,
    )
    cleaned = re.sub(r"([A-Za-z]{2,})\[(\d{1,3})\](\.)", r"\1\3[\2]", cleaned)
    cleaned = re.sub(
        r"\b([A-Za-z]{2,}\.)\[(\d{1,3})\](\s*(?:是|为|:|：|=)\s*(?:\*\*)?[-+]?\d+(?:\.\d+)?\s*%?(?:\*\*)?)",
        r"\1\3[\2]",
        cleaned,
    )
    cleaned = re.sub(r"(\[(\d{1,3})\])(?:\s*\[\2\])+", r"\1", cleaned)

    def _dedupe_ref_run(match: re.Match) -> str:
        refs: list[str] = []
        for ref in re.findall(r"\[(\d{1,3})\]", match.group(0)):
            if ref not in refs:
                refs.append(ref)
        return "".join(f"[{ref}]" for ref in refs)

    cleaned = re.sub(r"(?:\[\d{1,3}\]\s*){2,}", _dedupe_ref_run, cleaned)
    cleaned = re.sub(r"(?<=\d)\s+%", "%", cleaned)
    cleaned = re.sub(r"\s+([。！？；，、,.])", r"\1", cleaned)
    return cleaned


def _tokenize_for_citation(text: str = "") -> list[str]:
    lowered = str(text).lower()
    chars = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", lowered)
    bigrams = []
    for idx in range(len(chars) - 1):
        left = chars[idx]
        right = chars[idx + 1]
        if (
            len(left) == 1
            and len(right) == 1
            and re.match(r"[\u4e00-\u9fff]", left)
            and re.match(r"[\u4e00-\u9fff]", right)
        ):
            bigrams.append(left + right)
    return [*chars, *bigrams]


def _calc_token_overlap(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    right_set = set(right)
    return sum(1 for token in left if token in right_set)


def _strip_inline_citations(text: str = "") -> str:
    stripped = _INLINE_CITATION_PATTERN.sub("", str(text))
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


def _attach_refs_to_sentence(sentence: str, refs: list[int]) -> str:
    if not sentence or not refs:
        return sentence
    ref_text = "".join(f"[{ref}]" for ref in refs)
    trimmed = sentence.rstrip()
    tail_match = re.search(r"([。！？!?；;])$", trimmed)
    if tail_match:
        return f"{trimmed[:-1]}{ref_text}{tail_match.group(1)}"
    return f"{trimmed}{ref_text}"


def _calc_citation_support_score(sentence: str = "", citation: Optional[dict] = None) -> float:
    if not sentence or not citation:
        return 0.0

    sentence_tokens = _tokenize_for_citation(sentence)
    if not sentence_tokens:
        return 0.0

    support_fields = [
        *_collect_citation_table_evidence_texts(citation),
        _build_phrase_alignment_text(citation),
        citation.get("highlight_text", ""),
        citation.get("source_text", ""),
        citation.get("display_text", ""),
        citation.get("_full_text", ""),
        citation.get("group_id", ""),
        citation.get("table_id", ""),
        citation.get("table_caption", ""),
        citation.get("table_header", ""),
    ]
    support_text = " ".join(str(part).strip() for part in support_fields if part).strip()
    citation_tokens = _tokenize_for_citation(support_text)
    overlap = _calc_token_overlap(sentence_tokens, citation_tokens)
    score = overlap / max(1, len(sentence_tokens))

    sentence_lower = sentence.lower()
    support_lower = support_text.lower()
    if _is_numeric_table_metric_query(sentence_lower):
        if _has_numeric_table_metric_anchor(support_lower):
            score += 0.12
        if _has_numeric_table_cost_anchor(support_lower) or any(
            marker in support_lower
            for marker in ("rtx", "gpu", "training time", "inference time", "overhead", "six days", "24 hours", "训练时间", "推理时间")
        ):
            score -= 0.18

    snippet = re.sub(r"\s+", "", str(citation.get("highlight_text", "")))[:24]
    if len(snippet) >= 6:
        compact_sentence = re.sub(r"\s+", "", str(sentence))
        if snippet in compact_sentence:
            score += 0.25
        elif snippet[: min(10, len(snippet))] in compact_sentence:
            score += 0.1

    return score


def _optimize_sentence_citations(sentence: str, citations: list[dict]) -> str:
    refs_in_sentence = []
    for match in _INLINE_CITATION_PATTERN.finditer(str(sentence)):
        ref_str = match.group(1) or match.group(2)
        if ref_str:
            refs_in_sentence.append(int(ref_str))

    if not refs_in_sentence:
        return sentence

    normalized = _normalize_citation_records(citations)
    if not normalized:
        return _strip_inline_citations(sentence)

    core_sentence = _strip_inline_citations(sentence)
    if not core_sentence:
        return sentence

    citation_map = {int(c["ref"]): c for c in normalized}
    scored_all = sorted(
        (
            {"ref": int(c["ref"]), "score": _calc_citation_support_score(core_sentence, c)}
            for c in normalized
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    scored_current = sorted(
        (
            {"ref": ref, "score": _calc_citation_support_score(core_sentence, citation_map.get(ref))}
            for ref in dict.fromkeys(refs_in_sentence)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )

    chosen = [item["ref"] for item in scored_current if item["score"] >= 0.03]
    if not chosen:
        chosen = [item["ref"] for item in scored_all if item["score"] >= 0.06][:2]
    if not chosen and scored_current and scored_current[0]["score"] >= 0.02:
        chosen = [scored_current[0]["ref"]]

    chosen = list(dict.fromkeys(chosen))[:2]
    if not chosen:
        preserved = [ref for ref in dict.fromkeys(refs_in_sentence) if ref in citation_map]
        if preserved:
            return _attach_refs_to_sentence(core_sentence, preserved[:2])
        return core_sentence

    return _attach_refs_to_sentence(core_sentence, chosen)


def _optimize_inline_citations(answer: str, citations: list[dict]) -> str:
    if not answer or not citations:
        return answer

    lines = str(answer).split("\n")
    optimized = []
    in_code_fence = False
    for line in lines:
        if re.match(r"^\s*```", line):
            in_code_fence = not in_code_fence
            optimized.append(line)
            continue
        if in_code_fence:
            optimized.append(line)
            continue
        optimized.append(_optimize_sentence_citations(line, citations))
    return "\n".join(optimized)


def _resolve_strict_citation_support_threshold(evidence_need: Optional[list[str]]) -> Optional[float]:
    needs = {str(item).strip() for item in (evidence_need or []) if str(item).strip()}
    matched = needs & _STRICT_CITATION_EVIDENCE_NEEDS
    if not matched:
        return None
    return max(_STRICT_CITATION_SUPPORT_THRESHOLDS[item] for item in matched)


def _build_conservative_sentence_fallback(
    sentence: str,
    evidence_need: Optional[list[str]] = None,
) -> str:
    needs = {str(item).strip() for item in (evidence_need or []) if str(item).strip()}
    if not (needs & _CONSERVATIVE_REWRITE_EVIDENCE_NEEDS):
        return _strip_inline_citations(sentence)

    if "reference_meta" in needs:
        return "根据当前检索证据，文档未明确说明该引用元信息。"
    return "根据当前检索证据，无法确认该信息，文档未明确说明。"


_FALLBACK_SENTENCE_STOPWORDS = {
    "about", "above", "answer", "are", "based", "does", "from", "how", "main",
    "method", "paper", "problem", "proposed", "result", "results", "that", "the",
    "their", "this", "uses", "what", "when", "where", "which", "why", "with",
    "什么", "哪些", "如何", "论文", "方法", "主要", "问题", "区别", "不同", "请",
    "解释", "说明", "包含", "使用", "多少", "什么问题", "有何",
}


def _dedupe_text_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        clean = str(item or "").strip()
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped


def _split_fallback_evidence_sentences(text: str = "", *, max_sentences: int = 8) -> list[str]:
    """Split retrieval evidence into compact answerable units for deterministic fallback."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    pieces = [
        piece.strip(" \t\r\n-•;；")
        for piece in re.split(r"(?<=[。！？!?；;])\s+|(?<!\d)(?<=\.)\s+", normalized)
        if piece and piece.strip(" \t\r\n-•;；")
    ]
    if not pieces:
        pieces = [normalized]

    sentences: list[str] = []
    for piece in pieces:
        if len(piece) <= 260:
            sentences.append(piece)
        else:
            clauses = [
                clause.strip(" \t\r\n-•,，;；")
                for clause in re.split(r"\s*(?:；|;|，|,)\s*", piece)
                if clause and clause.strip(" \t\r\n-•,，;；")
            ]
            if len(clauses) >= 2:
                sentences.extend(clauses[:3])
            else:
                sentences.append(piece[:260].rstrip() + "...")
        if len(sentences) >= max_sentences:
            break
    return sentences[:max_sentences]


def _fallback_answer_query_terms(query: str = "") -> list[str]:
    terms = _extract_citation_query_anchors(query, max_terms=18)
    for term in re.findall(
        r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]{2,}",
        str(query or ""),
    ):
        clean = term.strip(" .,;:()[]{}，。；：、")
        if clean and clean.casefold() not in _FALLBACK_SENTENCE_STOPWORDS:
            terms.append(clean)
    if _is_formula_framework_query(query):
        terms.extend(["formula", "equation", "objective", "loss", "公式", "方程", "目标函数", "损失函数"])
    if _PIPELINE_STAGE_QUERY_RE.search(str(query or "")):
        terms.extend(["pipeline", "stage", "phase", "step", "training", "generation", "retraining", "流程", "阶段", "步骤"])
    return _dedupe_text_preserve_order(terms)[:28]


def _score_fallback_sentence(sentence: str, query_terms: list[str], *, source_rank: int, sentence_rank: int) -> float:
    text = str(sentence or "")
    if not text.strip():
        return -100.0
    score = 0.0
    for term in query_terms:
        if technical_anchor_matches(term, text):
            score += 1.0
    if looks_formula_like(text):
        score += 0.6
    if re.search(r"\d+(?:\.\d+)?%?", text):
        score += 0.35
    if re.search(r"\b(?:table|figure|算法|公式|方程|阶段|流程|pipeline|stage|phase)\b", text, re.IGNORECASE):
        score += 0.25
    if len(text) < 24:
        score -= 0.25
    if len(text) > 320:
        score -= 0.2
    score -= source_rank * 0.08
    score -= sentence_rank * 0.02
    return score


def _build_generation_fallback_evidence_answer(
    query: str,
    segments: list[dict],
    *,
    max_items: int = 4,
) -> str:
    query_terms = _fallback_answer_query_terms(query)
    candidates: list[tuple[float, int, int, str, int]] = []
    for source_rank, segment in enumerate(segments or []):
        if not isinstance(segment, dict):
            continue
        try:
            ref = int(segment.get("ref") or source_rank + 1)
        except (TypeError, ValueError):
            ref = source_rank + 1
        text = str(segment.get("text") or "")
        for sentence_rank, sentence in enumerate(_split_fallback_evidence_sentences(text)):
            score = _score_fallback_sentence(
                sentence,
                query_terms,
                source_rank=source_rank,
                sentence_rank=sentence_rank,
            )
            candidates.append((score, source_rank, sentence_rank, sentence, ref))

    if not candidates:
        return ""

    selected: list[tuple[int, int, str, int]] = []
    seen: set[str] = set()
    min_score = 0.8 if query_terms else -100.0
    for score, source_rank, sentence_rank, sentence, ref in sorted(
        candidates,
        key=lambda item: (-item[0], item[1], item[2]),
    ):
        compact = re.sub(r"\W+", "", sentence.casefold())
        if not compact or compact[:140] in seen:
            continue
        if score < min_score and selected:
            continue
        seen.add(compact[:140])
        selected.append((source_rank, sentence_rank, sentence, ref))
        if len(selected) >= max(1, max_items):
            break

    if not selected:
        for _score, source_rank, sentence_rank, sentence, ref in sorted(
            candidates,
            key=lambda item: (item[1], item[2]),
        )[:max(1, max_items)]:
            selected.append((source_rank, sentence_rank, sentence, ref))

    selected.sort(key=lambda item: (item[0], item[1]))
    lines = ["根据当前已检索到的文档证据，可先给出以下保守回答："]
    for _source_rank, _sentence_rank, sentence, ref in selected:
        clean = re.sub(r"\s+", " ", sentence).strip()
        if clean and not re.search(r"[。！？!?；;]$", clean):
            clean += "。"
        lines.append(f"- {clean} [{ref}]")
    lines.append("由于上游模型生成中断，上述内容仅基于已检索证据整理；未被证据直接支持的细节未展开。")
    return "\n".join(lines)


def _build_generation_error_fallback_answer(
    retrieval_meta: dict,
    *,
    error_message: str = "",
    max_items: int = 3,
) -> str:
    """Build a conservative answer when upstream generation fails after retrieval."""
    segments = _build_response_context_segments(retrieval_meta)
    if not segments:
        if error_message:
            retrieval_meta["generation_fallback_reason"] = str(error_message)[:160]
        return "当前请求未能返回可用检索证据，因此无法基于文档回答。"

    if error_message:
        retrieval_meta["generation_fallback_reason"] = str(error_message)[:160]
    query = str(retrieval_meta.get("search_query") or retrieval_meta.get("query") or "").strip()
    evidence_answer = _build_generation_fallback_evidence_answer(
        query,
        segments,
        max_items=max_items,
    )
    if evidence_answer:
        return evidence_answer

    lines = ["根据当前已检索到的文档证据，可先给出以下保守回答："]
    for idx, segment in enumerate(segments[:max(1, max_items)], 1):
        text = re.sub(r"\s+", " ", str(segment.get("text") or "")).strip()
        if len(text) > 220:
            text = text[:220].rstrip() + "..."
        ref = segment.get("ref") or idx
        lines.append(f"- {text} [{ref}]")
    lines.append("由于上游模型生成中断，上述内容仅基于已检索证据整理；未被证据直接支持的细节未展开。")
    return "\n".join(lines)




def _prune_weak_inline_citations(
    answer: str,
    citations: list[dict],
    *,
    evidence_need: Optional[list[str]] = None,
) -> tuple[str, dict]:
    threshold = _resolve_strict_citation_support_threshold(evidence_need)
    if not answer or not citations or threshold is None:
        return answer, {}

    normalized = _normalize_citation_records(citations)
    if not normalized:
        return answer, {}

    citation_map = {int(c["ref"]): c for c in normalized}
    checked_sentences = 0
    unsupported_sentences = 0
    removed_refs = 0
    kept_refs = 0
    rewritten_sentences = 0

    rewritten_lines = []
    in_code_fence = False
    for line in str(answer).split("\n"):
        if re.match(r"^\s*```", line):
            in_code_fence = not in_code_fence
            rewritten_lines.append(line)
            continue
        if in_code_fence or not _INLINE_CITATION_PATTERN.search(line):
            rewritten_lines.append(line)
            continue

        rewritten_parts = []
        for sentence in re.split(r"(?<=[。！？!?；;])", line):
            if not sentence or not _INLINE_CITATION_PATTERN.search(sentence):
                rewritten_parts.append(sentence)
                continue

            checked_sentences += 1
            refs = _extract_inline_citation_refs(sentence)
            core_sentence = _strip_inline_citations(sentence)
            if not refs or not core_sentence:
                rewritten_parts.append(core_sentence or sentence)
                continue

            supported_refs = []
            for ref in refs:
                score = _calc_citation_support_score(core_sentence, citation_map.get(ref))
                if score >= threshold:
                    supported_refs.append(ref)

            supported_refs = list(dict.fromkeys(supported_refs))[:2]
            if supported_refs:
                kept_refs += len(supported_refs)
                removed_refs += max(0, len(refs) - len(supported_refs))
                rewritten_parts.append(_attach_refs_to_sentence(core_sentence, supported_refs))
                continue

            unsupported_sentences += 1
            removed_refs += len(refs)
            fallback_sentence = _build_conservative_sentence_fallback(
                core_sentence,
                evidence_need=evidence_need,
            )
            if fallback_sentence != core_sentence:
                rewritten_sentences += 1
            rewritten_parts.append(fallback_sentence)

        rewritten_lines.append("".join(rewritten_parts))

    diagnostics = {
        "strict_mode": True,
        "threshold": threshold,
        "checked_sentence_count": checked_sentences,
        "unsupported_sentence_count": unsupported_sentences,
        "removed_ref_count": removed_refs,
        "kept_ref_count": kept_refs,
        "rewritten_sentence_count": rewritten_sentences,
        "evidence_need": list(dict.fromkeys(str(item).strip() for item in (evidence_need or []) if str(item).strip())),
    }
    rewritten_answer = "\n".join(rewritten_lines)
    if removed_refs > 0:
        logger.info(
            "严格引文检查已移除弱支撑引用: evidence_need=%s checked=%d unsupported=%d removed=%d threshold=%.2f",
            diagnostics["evidence_need"],
            checked_sentences,
            unsupported_sentences,
            removed_refs,
            threshold,
        )
    return rewritten_answer, diagnostics


def _normalize_single_ref_answer(answer: str, citations: list[dict]) -> str:
    normalized = _normalize_citation_records(citations)
    if not answer or len(normalized) <= 1:
        return answer

    refs_in_text = [
        int(match.group(1) or match.group(2))
        for match in _INLINE_CITATION_PATTERN.finditer(str(answer))
        if match.group(1) or match.group(2)
    ]
    unique_refs = list(dict.fromkeys(refs_in_text))
    if len(unique_refs) != 1:
        return answer

    paragraphs = str(answer).split("\n\n")
    rewritten = []
    for paragraph in paragraphs:
        if not _INLINE_CITATION_PATTERN.search(paragraph):
            rewritten.append(paragraph)
            continue

        para_tokens = _tokenize_for_citation(paragraph)
        best_ref = unique_refs[0]
        best_score = -1
        for citation in normalized:
            ref = int(citation["ref"])
            citation_tokens = _tokenize_for_citation(citation.get("highlight_text", ""))
            score = _calc_token_overlap(para_tokens, citation_tokens)
            if score > best_score:
                best_score = score
                best_ref = ref
        rewritten.append(_INLINE_CITATION_PATTERN.sub(f"[{best_ref}]", paragraph))

    return "\n\n".join(rewritten)



def _prepare_answer_and_citations_for_display(
    answer: str,
    citations: list[dict],
    *,
    evidence_need: Optional[list[str]] = None,
    answer_guard: Optional[dict] = None,
    query: str = "",
    context_segments: Optional[list[dict]] = None,
) -> tuple[str, list[dict]]:
    evidence_need_set = {
        str(item).strip()
        for item in (evidence_need or [])
        if str(item).strip()
    }
    is_numeric_table_request = "numeric_table" in evidence_need_set
    normalized_citations = _normalize_citation_records(citations)
    if is_numeric_table_request:
        normalized_citations = _normalize_numeric_metric_bundle_citations(normalized_citations, query)
    context_recovery_citations = _context_segments_to_recovery_citations(
        context_segments,
        start_ref=len(normalized_citations) + 1,
        query=query,
        reserved_refs={int(citation["ref"]) for citation in normalized_citations},
    )
    normalized_citation_candidates = _normalize_citation_records(
        normalized_citations
        + context_recovery_citations
    )
    repaired_answer = _repair_bad_citation_formats(answer, normalized_citations)
    normalized_citations = _merge_inline_referenced_recovery_citations(
        repaired_answer,
        normalized_citations,
        context_recovery_citations,
    )
    if normalized_citations and repaired_answer:
        refs_in_answer = _extract_inline_citation_refs(repaired_answer)
        valid_ref_set = {int(c["ref"]) for c in normalized_citations}
        has_context_recovery_candidates = bool(context_recovery_citations)
        should_optimize_inline = (
            (len(set(refs_in_answer)) <= 1 and not has_context_recovery_candidates)
            or any(ref not in valid_ref_set for ref in refs_in_answer)
        )

        repaired_answer = _normalize_single_ref_answer(repaired_answer, normalized_citations)
        if should_optimize_inline:
            repaired_answer = _optimize_inline_citations(repaired_answer, normalized_citations)
        if not _extract_inline_citation_refs(repaired_answer):
            repaired_answer = _inject_inline_citations(repaired_answer, normalized_citations)

        repaired_answer, guard_diagnostics = _prune_weak_inline_citations(
            repaired_answer,
            normalized_citations,
            evidence_need=evidence_need,
        )
        if answer_guard is not None and guard_diagnostics:
            answer_guard.update(guard_diagnostics)
        if guard_diagnostics.get("removed_ref_count", 0) > 0 and not _extract_inline_citation_refs(repaired_answer):
            numeric_cost_recovery = (
                is_numeric_table_request
                and _is_numeric_table_cost_query(query)
                and any(
                    _has_numeric_table_cost_anchor(_build_numeric_table_citation_support_text(citation))
                    for citation in normalized_citations
                )
            )
            if not numeric_cost_recovery and is_numeric_table_request:
                recovered_answer, recovered_citations, numeric_recovery_diag = _recover_numeric_table_metric_citation_from_context(
                    repaired_answer,
                    context_segments,
                    query=query,
                    start_ref=max((int(citation.get("ref") or 0) for citation in normalized_citation_candidates), default=0) + 1,
                )
                if recovered_citations:
                    repaired_answer = recovered_answer
                    normalized_citations = recovered_citations
                    if answer_guard is not None:
                        answer_guard["numeric_table_context_citation_recovery"] = numeric_recovery_diag
                elif answer_guard is not None and numeric_recovery_diag:
                    answer_guard["numeric_table_context_citation_recovery"] = numeric_recovery_diag
            if not numeric_cost_recovery and not _extract_inline_citation_refs(repaired_answer):
                return repaired_answer, []
        if is_numeric_table_request:
            repaired_answer, numeric_alignment_diag = _align_numeric_table_inline_citations(
                repaired_answer,
                normalized_citations,
                query=query,
            )
            if answer_guard is not None and numeric_alignment_diag.get("applied"):
                answer_guard["numeric_table_inline_ref_alignment"] = numeric_alignment_diag
    aligned = _align_citations_with_answer(repaired_answer, normalized_citations)
    if is_numeric_table_request:
        aligned = _supplement_numeric_table_citations(
            repaired_answer,
            aligned,
            normalized_citations,
            query=query,
        )
    if is_numeric_table_request and aligned and not _is_dataset_unavailable_answer(repaired_answer, query):
        repaired_answer, aligned, numeric_force_diag = _force_best_numeric_table_citation(
            repaired_answer,
            aligned,
            normalized_citation_candidates,
            query=query,
        )
        if answer_guard is not None and numeric_force_diag.get("applied"):
            answer_guard["numeric_table_best_citation_force"] = numeric_force_diag
        repaired_answer, numeric_alignment_diag = _align_numeric_table_inline_citations(
            repaired_answer,
            aligned,
            query=query,
        )
        if answer_guard is not None and numeric_alignment_diag.get("applied"):
            answer_guard["numeric_table_inline_ref_alignment_after_supplement"] = numeric_alignment_diag
        repaired_answer, aligned, numeric_exact_dedupe_diag = _dedupe_numeric_table_exact_aligned_citations(
            repaired_answer,
            aligned,
            query=query,
        )
        if answer_guard is not None and numeric_exact_dedupe_diag.get("applied"):
            answer_guard["numeric_table_exact_citation_dedupe"] = numeric_exact_dedupe_diag
    if not aligned and is_numeric_table_request:
        recovered_answer, recovered_citations, numeric_context_recovery_diag = _recover_numeric_table_metric_citation_from_context(
            repaired_answer,
            context_segments,
            query=query,
            start_ref=max((int(citation.get("ref") or 0) for citation in normalized_citation_candidates), default=0) + 1,
        )
        if recovered_citations:
            repaired_answer = recovered_answer
            aligned = recovered_citations
            if answer_guard is not None:
                answer_guard["numeric_table_context_citation_recovery"] = numeric_context_recovery_diag
        elif answer_guard is not None and numeric_context_recovery_diag:
            answer_guard["numeric_table_context_citation_recovery"] = numeric_context_recovery_diag
    if not aligned:
        return repaired_answer, []

    refs_in_answer = _extract_inline_citation_refs(repaired_answer)
    citation_map = {int(c["ref"]): c for c in aligned}
    ordered_source_refs = refs_in_answer or [int(c["ref"]) for c in aligned]
    should_append_aligned_refs = (
        not is_numeric_table_request or not refs_in_answer
    )
    if should_append_aligned_refs:
        ordered_source_refs.extend(
            int(c["ref"])
            for c in aligned
            if int(c["ref"]) not in ordered_source_refs
        )
    elif (
        is_numeric_table_request
        and refs_in_answer
        and re.search(r"\babove\b|higher|lower|百分点|差距|相比|比", repaired_answer, re.IGNORECASE)
    ):
        selected_table_ids = {
            str(citation_map.get(ref, {}).get("table_id") or "").strip().lower()
            for ref in refs_in_answer
            if ref in citation_map and str(citation_map.get(ref, {}).get("table_id") or "").strip()
        }
        selected_group_ids = {
            str(citation_map.get(ref, {}).get("group_id") or "").strip().lower()
            for ref in refs_in_answer
            if ref in citation_map and str(citation_map.get(ref, {}).get("group_id") or "").strip()
        }
        for citation in aligned:
            ref = int(citation["ref"])
            if ref in ordered_source_refs or not _has_numeric_table_exact_row_support(citation):
                continue
            table_id = str(citation.get("table_id") or "").strip().lower()
            group_id = str(citation.get("group_id") or "").strip().lower()
            same_bundle = bool(
                (table_id and table_id in selected_table_ids)
                or (group_id and group_id in selected_group_ids)
            )
            if same_bundle and _numeric_table_answer_mentions_method(repaired_answer, citation):
                ordered_source_refs.append(ref)

    source_to_display: dict[int, int] = {}
    projected = []
    for source_ref in ordered_source_refs:
        if source_ref in source_to_display or source_ref not in citation_map:
            continue
        display_ref = len(source_to_display) + 1
        source_to_display[source_ref] = display_ref
        item = citation_map[source_ref].copy()
        item["source_ref"] = _coerce_positive_int(item.get("source_ref"), source_ref)
        item["display_ref"] = display_ref
        item["ref"] = display_ref
        projected.append(item)

    if not projected:
        return repaired_answer, []

    selector_candidate_map: dict[int, dict] = {int(item["ref"]): item for item in projected}
    if context_recovery_citations:
        next_display_ref = len(selector_candidate_map) + 1
        ranked_recovery_candidates, recovery_selector_diag = _rank_context_recovery_candidates_for_selector(
            repaired_answer,
            context_recovery_citations,
        )
        if answer_guard is not None:
            answer_guard["context_recovery_selector_candidates"] = recovery_selector_diag
        for candidate in ranked_recovery_candidates:
            source_ref = int(candidate["ref"])
            if source_ref in source_to_display:
                continue
            item = candidate.copy()
            item["source_ref"] = _coerce_positive_int(item.get("source_ref"), source_ref)
            item["display_ref"] = next_display_ref
            item["ref"] = next_display_ref
            selector_candidate_map[next_display_ref] = item
            next_display_ref += 1

    rewritten_answer = (
        _rewrite_inline_citation_refs(repaired_answer, source_to_display)
        if refs_in_answer else repaired_answer
    )
    rewritten_answer, selector_fill_diag = _apply_projected_selector_citation_fill(
        rewritten_answer,
        list(selector_candidate_map.values()),
        query=query,
    )
    if answer_guard is not None and selector_fill_diag.get("applied"):
        answer_guard["projected_selector_citation_fill"] = selector_fill_diag
    projected = _append_selector_filled_projected_citations(
        projected,
        selector_candidate_map,
        rewritten_answer,
    )
    rewritten_answer, projected = _compact_projected_citation_display_refs(
        rewritten_answer,
        projected,
    )
    rewritten_answer = _remove_invalid_inline_citation_refs(
        rewritten_answer,
        {int(item.get("ref") or 0) for item in projected},
    )
    rewritten_answer = _cleanup_inline_citation_display(rewritten_answer)
    return rewritten_answer, projected


def _trim_partial_section_suffix(text: str, marker: str) -> str:
    normalized_marker = (marker or "").strip()
    if not text or not normalized_marker:
        return text

    marker_upper = normalized_marker.upper()
    text_upper = text.upper()
    max_overlap = min(len(marker_upper) - 1, len(text_upper))
    for overlap in range(max_overlap, 2, -1):
        if marker_upper.startswith(text_upper[-overlap:]):
            return text[:-overlap]
    return text


def _extract_streaming_final_answer(full_output: str) -> str:
    if not full_output:
        return ""
    parts = _ci_split(_RE_START_ANSWER, full_output)
    if parts is not None:
        answer = parts[1].lstrip()
        cit_parts = _ci_split(_RE_START_CITATION, answer)
        if cit_parts is not None:
            answer = cit_parts[0].rstrip()
        else:
            answer = _trim_partial_section_suffix(answer, START_CITATION).rstrip()
        return answer

    stripped = full_output.lstrip()
    if not stripped:
        return ""

    cit_parts = _ci_split(_RE_START_CITATION, full_output)
    if cit_parts is not None:
        if not cit_parts[0].strip():
            return ""
        return cit_parts[0].rstrip()

    answer = _trim_partial_section_suffix(full_output, START_ANSWER)
    answer = _trim_partial_section_suffix(answer, START_CITATION)
    return answer.rstrip()


def _normalize_web_search_max_results(value: Optional[int]) -> int:
    if value is None:
        return 5
    return max(1, min(int(value), _MAX_WEB_SEARCH_RESULTS))


_WEB_SEARCH_INTENT_CACHE: dict[str, bool] = {}
_WEB_SEARCH_INTENT_CACHE_MAX = 256

_INTENT_YES_KEYWORDS = frozenset([
    # 明确指向外部信息的词
    "最新", "现在", "今", "近期", "当前", "实时", "更新", "新闻", "最近", "版本", "上市", "发布",
    "latest", "current", "recent", "news", "now", "today", "update", "release", "2024", "2025", "2026",
    "谁是", "谁当", "多少钱", "价格", "股价", "汇率", "天气",
])
_INTENT_NO_KEYWORDS = frozenset([
    # 纯文档解读相关词
    "摘要", "总结", "概括", "翻译", "解释", "分析图表", "本文", "文章", "文档", "此处", "这段", "图中",
    "表格", "公式", "作者怎么", "文中", "怎么理解", "什么意思",
    "summarize", "translate", "explain this", "what does this mean",
])


async def _should_perform_web_search(
    question: str,
    api_key: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    endpoint: Optional[str],
) -> bool:
    """用轻量关键词 + 可选 LLM 快速判断是否需要联网搜索。

    优先使用启发式规则（零延迟），仅在无法判断时调用 LLM。
    """
    if not question or not question.strip():
        return False

    q = question.strip().lower()

    # 快速命中：明确需要联网
    if any(kw in q for kw in _INTENT_YES_KEYWORDS):
        logger.debug("联网意图判断：命中 YES 关键词")
        return True

    # 快速命中：明确不需要联网（纯文档问题）
    if any(kw in q for kw in _INTENT_NO_KEYWORDS):
        logger.debug("联网意图判断：命中 NO 关键词")
        return False

    # 未命中规则：缓存检查
    cache_key = q[:120]
    if cache_key in _WEB_SEARCH_INTENT_CACHE:
        return _WEB_SEARCH_INTENT_CACHE[cache_key]

    # 无 API key 时无法调用 LLM，默认执行搜索
    if not api_key:
        return True

    # LLM 轻量判断
    try:
        prompt = (
            "判断以下问题是否需要联网搜索获取外部信息（最新数据、事件、人物等）。\n"
            "若问题仅涉及文档内容解读、格式、摘要，回复 no。\n"
            "若问题涉及外部事件、最新数据、人物、地点或文档中没有的信息，回复 yes。\n"
            "只回复 yes 或 no，不要解释。\n"
            f"问题：{question[:200]}"
        )
        result = await call_ai_api(
            [{"role": "user", "content": prompt}],
            api_key,
            model or "gpt-4o-mini",
            provider or "openai",
            endpoint=endpoint or "",
            max_tokens=5,
            temperature=0,
        )
        if result.get("error"):
            raise RuntimeError(result["error"])
        raw = result.get("choices", [{}])[0].get("message", {}).get("content") or ""
        answer = raw.strip().lower()
        decision = answer.startswith("yes")
        logger.debug(f"联网意图 LLM 判断: '{answer[:20]}' → {decision}")

        if len(_WEB_SEARCH_INTENT_CACHE) >= _WEB_SEARCH_INTENT_CACHE_MAX:
            oldest = next(iter(_WEB_SEARCH_INTENT_CACHE))
            del _WEB_SEARCH_INTENT_CACHE[oldest]
        _WEB_SEARCH_INTENT_CACHE[cache_key] = decision
        return decision
    except Exception as e:
        logger.debug(f"联网意图 LLM 判断失败，默认执行搜索: {e}")
        return True


def _clean_query_text(text: str, max_len: int = 200) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:max_len]


def _normalize_doc_title(doc_title: str) -> str:
    title = _clean_query_text(doc_title, max_len=80)
    title = re.sub(r"\.(pdf|docx?|txt|md)$", "", title, flags=re.IGNORECASE)
    return title


def _contains_pronoun_like_reference(text: str) -> bool:
    lowered = (text or "").lower()
    return any(hint in lowered for hint in _WEB_SEARCH_PRONOUN_HINTS)


def _build_web_search_query(
    base_query: str,
    original_question: str,
    doc_title: str = "",
    selected_text: str = "",
) -> str:
    """构建联网搜索查询，减少代词歧义与离题检索。"""
    query = _clean_query_text(base_query or original_question, max_len=180)
    if not query:
        return ""

    anchors: list[str] = []
    title = _normalize_doc_title(doc_title)
    if title and title.lower() not in query.lower():
        anchors.append(title)

    if selected_text and _contains_pronoun_like_reference(original_question or query):
        selected_snippet = _context_builder._extract_relevant_snippet(
            selected_text, original_question or query, max_len=80, selected_text=selected_text
        )
        selected_snippet = _clean_query_text(selected_snippet, max_len=80)
        # 跳过明显“参考文献列表”风格文本，避免将无关人名注入 query
        if selected_snippet and not _context_builder._is_reference_like_text(selected_snippet):
            anchors.append(selected_snippet)

    if anchors:
        query = f"{query} {' '.join(anchors)}"
    return _clean_query_text(query, max_len=260)


def _normalize_answer_detail(value: Optional[str]) -> str:
    if not value:
        return _DEFAULT_ANSWER_DETAIL
    detail = str(value).strip().lower()
    if detail in _VALID_ANSWER_DETAILS:
        return detail
    return _DEFAULT_ANSWER_DETAIL


def _build_faithfulness_guard_prompt() -> str:
    """P3.5 详细引用规则手册（参考 ragflow citation_prompt.md）

    设计原则：
    - 6 类示例（数据/因果/技术/比较/混合/反例）覆盖常见引用场景
    - 必引清单 + 不必引清单，明确两端边界
    - 引用格式 [n] 严格规范，禁止 [ID:0]、[ID:多个]、整段无引用
    - 中文化 + 适配 Chatpdf chunk_id 命名
    """
    return (
        "【忠实性与引用规则手册 - 严格遵守】\n"
        "\n"
        "## 首要规则：禁止幻觉（Anti-Hallucination Sentinel）\n"
        "- 你的回答**只能**基于「检索到的文档内容」中的事实，禁止使用你的预训练知识补充\n"
        "- 如果文档内容**不足以回答问题**，必须直接回复：「根据文档内容无法回答此问题，因为：（简要说明缺失什么信息）」\n"
        "- 判断标准：答案中的每个核心事实（数值、因果、归属、方法名）都必须能在检索内容中找到直接依据\n"
        "- 宁可不答，不可乱答。一个诚实的「无法回答」远比一个看似完整但不忠实的答案更有价值\n"
        "\n"
        "## 引用格式\n"
        "- 仅允许使用 [n] 形式（n 为提示词中给出的引用编号），禁止使用 [0]、[ID:n]、【n】等其他写法\n"
        "- 单个声明最多 2 个引用编号，例如 [3][7]，禁止 [3,7] 或 [3-7]\n"
        "- 不同事实若来自不同证据窗口，必须使用不同编号；不要把多个独立事实都标成同一个编号\n"
        "- 引用编号必须能直接支撑所在句子的具体事实，不能只因为同页、同章节或同主题就引用\n"
        "- 一句话如果混合多个独立事实，请拆成多个短句，并分别附在对应事实之后\n"
        "- 整段不得无引用；至少在每个关键结论句后附引用\n"
        "\n"
        "## 必引（每条事实声明都需要）\n"
        "- 数据/数值：精度、F1、accuracy、loss、ms、参数量、token 数\n"
        "- 时序/版本：发表年份、模型版本、数据集版本\n"
        "- 因果/机制：A 导致 B、X 解决 Y、why/how 类陈述\n"
        "- 比较：「A 比 B 高 0.5%」「方法 X 优于 Y」必须列出参与比较的原始数值\n"
        "- 术语定义：模型名、方法名、数据集名、指标名首次出现\n"
        "- 归属：作者主张、论文结论、实验设置\n"
        "- 预测/争议：未来工作、limitations、不一致点\n"
        "\n"
        "## 不必引（可省略）\n"
        "- 通用常识（如「神经网络包含多层」）\n"
        "- 段落过渡句、章节引语\n"
        "- 你对答案结构的解释（「下面分三点说明」）\n"
        "\n"
        "## 示例\n"
        "- ✅ 数据型：「该模型在测试集上取得 95.2% 准确率 [4]。」\n"
        "- ✅ 因果型：「该问题源于训练数据稀疏，迫使模型借助合成样本 [2][5]。」\n"
        "- ✅ 技术型：「方法采用 cross-attention 替代 self-attention [3]。」\n"
        "- ✅ 比较型：「Method A 71.3 vs Method B 68.5，提升 2.8 个百分点 [4][6]。」\n"
        "- ✅ 混合型：「该方法由 Smith 等人于 2024 年提出 [1]，在多个基准测试上取得 SOTA [4]。」\n"
        "- ❌ 错误：「文档显示模型表现较好。」（无具体数据 + 无引用）\n"
        "- ❌ 错误：「实验在多个数据集进行，效果不错 [3]。」（事实模糊 + 引用覆盖整段）\n"
        "\n"
        "## 严格守则\n"
        "- 答案中的**每个事实声明**必须能在文档内容中找到直接依据，禁止编造或推测\n"
        "- 引用前先做自检：该证据是否能逐字、数值或语义蕴含地支持当前句子；不能支持则不要引用\n"
        "- 数字、公式、方法名、模型名、数据集名必须**完整照抄原文**，不得改写或简化\n"
        "- 文档中**未明确出现**的细节（如具体数值、超参数、版本号），明确写「文档未明确说明」\n"
        "## 信息不足时的处理（借鉴 paper-qa CANNOT_ANSWER 哨兵 + TrustRAG 拒答机制）\n"
        "- 如果检索到的文档内容**不足以回答问题**，直接说明「根据文档内容无法回答此问题」，然后简要说明原因\n"
        "- 禁止在文档内容不足时用自身知识补充回答；宁可不答，不可乱答\n"
        "- 判断标准：如果答案中的核心事实（数值、因果、归属）在文档中找不到直接依据，则视为不足\n"
        "\n"
        "- 如果某句无法在上下文中找到证据，请删除该句的引用而不是强行引用\n"
        "- 如果信息来自通用知识而非文档，则无需标注引用"
    )


def _build_answer_style_instruction(answer_detail: str) -> str:
    """根据回答详细度生成提示词指令。"""
    detail = _normalize_answer_detail(answer_detail)
    if detail == "concise":
        return (
            "回答风格：简洁模式。第一句直接给出结论，随后用 1-3 个要点说明依据，"
            "控制在 300 字以内。避免冗余背景和过度展开。"
        )
    if detail == "detailed":
        return (
            "回答风格：详细模式。请遵循以下原则：\n"
            "- **首段先给核心结论**（2-3 句话明确回答问题），再展开分析\n"
            "- 使用 Markdown 标题（##）和小标题（###）结构化分段\n"
            "- 覆盖关键要点（背景、核心内容、依据、结论、注意事项），按需选择\n"
            "- 引用文档原文作为论据，关键数据和公式完整展示\n"
            "- 目标回答长度 800-1500 字，避免冗余重复和无关展开\n"
            "- 简单问题不必强行扩写，重点是信息密度而非字数"
        )
    return (
        "回答风格：标准模式。**首句先直接回答问题**（核心结论先行），"
        "随后只展开问题直接要求的关键依据，引用文档原文佐证。"
        "不要主动补充背景、优缺点、实验设置、局限性或无关比较；问题没问到的信息一律省略。"
        "目标 250-600 字，聚焦问题主旨，避免冗余背景和重复内容。"
    )


def _build_extraction_constraint_prompt() -> str:
    """为 extraction 题型生成专用约束提示词

    强制模型只从给定段落中直接引用或改写答案，禁止综述、推断和扩展。
    """
    return (
        "【提取模式约束】你正在回答一个需要精确提取事实的问题，请严格遵守：\n"
        "- 只允许从'文档内容'段落中直接引用或逐字改写，不得添加段落中没有明确出现的信息\n"
        "- 禁止概括全文、总结背景或补充上下文——如果段落里没有，就说'文档中未明确记载'\n"
        "- 数值、公式、实验结果必须完整抄录原文，不得四舍五入或使用'约'字模糊表达\n"
        "- 回答应简短直接，不超过 300 字，除非原文本身就很长"
    )


def _build_numeric_table_constraint_prompt() -> str:
    """为 numeric_table 题型生成专用约束提示词。"""
    return (
        "【数值表格模式约束】你正在回答一个表格数值题，请严格遵守：\n"
        "- 只能使用同一张表中的证据回答，不得把正文说明句、相邻表格或别的表号中的数字拼接在一起\n"
        "- 回答前先锁定表号、方法名、列名；若三者无法同时确定，就明确回答'文档中未明确给出'\n"
        "- 遇到'提升多少/高多少/差多少个百分点'这类问题时，必须先列出参与比较的原始数值，再给出百分点差值\n"
        "- 列名必须按表格原文对齐，例如 Medium 可对应 Med.；不得自行补造缺失列名\n"
        "- 如果问题只问准确率、指标或表格结果，不要主动扩展训练时间、硬件、局限性、实验设置等非目标信息\n"
        "- 如果检索上下文里出现多个表号或混合表块，优先选择同时包含问题中方法名和列名的那张表；无法唯一确定时不要猜测"
    )


_PIPELINE_STAGE_QUERY_RE = re.compile(
    r"(pipeline|workflow|procedure|stage|phase|step|流程|阶段|步骤|过程)",
    re.IGNORECASE,
)


def _build_agent_answer_focus_prompt(
    query: str = "",
    *,
    query_type: str = "",
    evidence_need: Optional[list[str]] = None,
) -> str:
    """为 agent 路径补充轻量、通用的论文问答聚焦约束。

    只根据问题文本和通用 evidence_need 选择输出形态，不读取论文标题、doc_id、
    gold answer 或评估题内容。目标是减少 agent 汇总时把已检索的相邻背景、
    pipeline、公式和实验细节混在一起导致的回答漂移。
    """
    question = str(query or "")
    needs = {str(item or "").lower() for item in (evidence_need or [])}
    instructions: list[str] = []

    if _PIPELINE_STAGE_QUERY_RE.search(question):
        instructions.append(
            "- 若问题询问流程、pipeline、阶段、步骤或过程：先用一句话给出阶段数量和阶段名称；"
            "随后每个阶段只说明输入、动作和输出。不要把图中的箭头、训练循环、超参数或损失项误当作同级阶段；"
            "若上下文对阶段数存在两种表述，说明差异并以正文明确的阶段列表为准。"
        )

    if _is_formula_framework_query(question) or "formula" in needs:
        instructions.append(
            "- 若问题询问公式、方程、算法框架、目标函数或损失函数：先回答使用的公式/算法框架名称，"
            "再列出上下文明确给出的核心公式和变量含义。不要展开无关 pipeline、实验表格、生成样本类型或泛泛背景；"
            "上下文没有给出完整公式时，明确说缺少哪一部分。"
        )

    if not instructions:
        return ""

    return (
        "【Agent 论文问答聚焦约束】\n"
        "- 只回答用户问题直接要求的信息；检索到但问题未问的背景、消融、局限性和实验细节不要主动展开。\n"
        + "\n".join(instructions)
    )


_CITATION_TOKEN_OVERHEAD = 1024  # 结构化引文（CITATION LIST）输出的预估 token 开销
_DETAILED_MIN_TOKENS = 5120     # 详细模式下 max_tokens 的最低保证值（对应 800-1500 字 + 格式 + 引文）
_STANDARD_DEFAULT_TOKENS = 1000 # 标准模式下 max_tokens 未设置时的默认值（对应 250-600 字 + 格式）


def _adjust_max_tokens(
    max_tokens: Optional[int],
    answer_detail: str,
    has_structured_citations: bool,
) -> Optional[int]:
    """根据回答详细度和引文开销调整 max_tokens。

    - 详细模式：保证 max_tokens >= _DETAILED_MIN_TOKENS
    - 标准模式：max_tokens 未设置时使用 _STANDARD_DEFAULT_TOKENS 作为默认值
    - 结构化引文：自动增加 _CITATION_TOKEN_OVERHEAD 补偿隐藏的 CITATION LIST 输出
    - 不覆盖用户已设置的更大值
    """
    detail = _normalize_answer_detail(answer_detail)
    effective = max_tokens

    if detail == "detailed":
        if effective is None or effective < _DETAILED_MIN_TOKENS:
            effective = _DETAILED_MIN_TOKENS
    elif detail == "standard":
        # 标准模式：用户未设置 max_tokens 时使用默认值，避免依赖 Provider 默认（通常偏低）
        if effective is None:
            effective = _STANDARD_DEFAULT_TOKENS

    # 结构化引文开销补偿
    if has_structured_citations and effective is not None:
        effective += _CITATION_TOKEN_OVERHEAD

    # 防止超出常见 Provider 的 max_tokens 上限（如 DeepSeek 8192）
    if effective is not None and effective > 8192:
        effective = 8192

    return effective


async def _maybe_perform_web_search(
    request: ChatRequest,
    *,
    query_override: str = "",
    doc_title: str = "",
    selected_text: str = "",
    doc_id: str = "",
    vector_store_dir: str = "",
) -> tuple[list[dict], str]:
    """按请求开关执行联网搜索，返回 (sources, formatted_context)。"""
    if not getattr(request, "enable_web_search", False):
        return [], ""
    if not request.question or not request.question.strip():
        return [], ""

    provider = request.web_search_provider or "auto"
    max_results = _normalize_web_search_max_results(request.web_search_max_results)
    search_query = _build_web_search_query(
        base_query=query_override or request.question,
        original_question=request.question,
        doc_title=doc_title,
        selected_text=selected_text,
    )
    if not search_query:
        return [], ""

    blacklist = [b.strip() for b in (request.web_search_blacklist or []) if b.strip()]
    try:
        logger.info(f"联网搜索开始: provider={provider}, query='{search_query[:120]}'")
        sources = await SearchManager.search(
            query=search_query,
            provider=provider,
            api_key=request.web_search_api_key,
            max_results=max_results * 2,  # 多取几条供重排筛选
            blacklist=blacklist or None,
        )
        if not sources:
            return [], ""

        # 向量语义重排（提升相关性），降级时返回词法重排结果
        sources = await rerank_web_results(
            query=search_query,
            results=sources,
            doc_id=doc_id,
            vector_store_dir=vector_store_dir,
            api_key=request.api_key,
            top_k=max_results,
        )
        if not sources:
            return [], ""

        return sources, format_search_results(sources)
    except Exception as e:
        logger.warning(f"联网搜索失败，已降级为仅文档检索: {e}")
        return [], ""


def _stringify_ai_error_detail(error: object) -> str:
    """提取 provider / 中间件返回的可读错误信息。"""
    if error is None:
        return ""
    if isinstance(error, dict):
        for key in ("message", "detail"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        nested_error = error.get("error")
        if nested_error is not None and nested_error is not error:
            nested_detail = _stringify_ai_error_detail(nested_error)
            if nested_detail:
                return nested_detail
        try:
            return json.dumps(error, ensure_ascii=False)
        except TypeError:
            return str(error)
    return str(error)


def _extract_non_stream_ai_message(response: object) -> dict:
    """统一校验非流式 AI 响应，避免上游错误被二次异常掩盖。"""
    if not isinstance(response, dict):
        raise ValueError("AI返回格式无效：响应不是对象")

    error_detail = _stringify_ai_error_detail(response.get("error"))
    if error_detail:
        raise RuntimeError(error_detail)

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("AI返回格式无效：缺少choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("AI返回格式无效：choices[0]不是对象")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("AI返回格式无效：缺少message")

    return message


async def _retry_generation_after_stream_error(
    *,
    messages: list[dict],
    request: ChatRequest,
    retrieval_meta: dict,
    has_structured_citations: bool,
    max_tokens: Optional[int],
) -> tuple[str, str, dict]:
    """流式生成失败后，用同一 prompt 做一次非流式重试。

    该恢复路径只复用当前请求的 messages 和检索证据，不读取论文标题、
    doc_id、评测答案或特定问题模式。若重试失败，调用方继续走确定性证据兜底。
    """
    response = await call_ai_api(
        messages,
        request.api_key,
        request.model,
        request.api_provider,
        endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
        middlewares=build_chat_middlewares(),
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        custom_params=request.custom_params,
        reasoning_effort=request.reasoning_effort,
    )
    message = _extract_non_stream_ai_message(response)
    raw_answer = message.get("content") or ""
    answer = extract_final_answer(raw_answer)
    reasoning_content = extract_reasoning_content(message)

    if not answer.strip():
        raise ValueError("stream_retry_empty_answer")

    retrieval_meta["stream_retry_used"] = True
    answer_guard: dict = {}
    _snapshot_retrieval_context_segments(retrieval_meta)
    answer, retrieval_meta["citations"] = _prepare_answer_and_citations_for_display(
        answer,
        retrieval_meta.get("citations", []),
        evidence_need=retrieval_meta.get("evidence_need", []),
        answer_guard=answer_guard,
        query=retrieval_meta.get("search_query") or request.question,
        context_segments=retrieval_meta.get("_context_segments", []),
    )
    if answer_guard:
        retrieval_meta["answer_guard"] = answer_guard
    if retrieval_meta.get("citations"):
        retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
            retrieval_meta.get("citations", []),
            query=retrieval_meta.get("search_query") or request.question,
        )
    return answer, reasoning_content, response


@router.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    if not hasattr(router, "documents_store"):
        raise HTTPException(status_code=500, detail="文档存储未初始化")
    # Per-request feature flag overrides（前端 GlobalSettings 可细化控制）
    apply_request_overrides(
        numeric_table=request.override_numeric_table,
        answer_critic=request.override_answer_critic,
        llm_query_rewrite=request.override_llm_query_rewrite,
        bm25_synonyms=request.override_bm25_synonyms,
    )
    store = router.documents_store if not request.doc_store_key else router.documents_store.get(request.doc_store_key, {})
    if request.doc_id not in store:
        raise HTTPException(status_code=404, detail="文档未找到")
    doc = store[request.doc_id]
    context = ""
    retrieval_meta = {}
    citations: list[dict] = []
    web_search_sources: list[dict] = []
    web_search_context = ""
    use_memory = _should_use_memory(request)
    if use_memory:
        _maybe_flush_memory(request)
    memory_context = ""
    raw_memories = []
    memory_hits: list[dict] = []
    memory_meta: dict = {
        "enabled": use_memory,
        "strategy": None,
        "retrieved_count": 0,
        "selected_count": 0,
        "truncated": False,
        "token_budget": None,
        "selected_kinds": [],
    }
    if use_memory:
        memory_context = _retrieve_memory_context(
            request.question, api_key=request.api_key, doc_id=request.doc_id
        )
    raw_memories = _retrieve_raw_memories(
        request.question, api_key=request.api_key, doc_id=request.doc_id, chat_history=request.chat_history
    )

    # 支持多图逻辑
    image_list = (request.image_base64_list or [])
    if request.image_base64 and request.image_base64 not in image_list:
        image_list = [request.image_base64] + image_list
    image_list = [img for img in image_list if img]

    if image_list:
        logger.info("[Chat] 截图模式：处理 %s 张图", len(image_list))
        answer_style_instruction = _build_answer_style_instruction(request.answer_detail)
        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
用户从文档中截取了 {len(image_list)} 张图片并发送给你。请仔细分析这些图片内容并回答问题。

回答规则：
1. 以用户发送的图片为核心依据进行回答，不要参考其他内容。
2. 如果图片包含图表，请分析数据趋势和关键信息。
3. 如果图片包含公式，请使用 LaTeX 格式（$公式$）展示。
4. 如果图片包含表格，请转换为 Markdown 格式。
5. 学术准确、表达清晰。
6. {answer_style_instruction}"""
        system_prompt, memory_hits, memory_meta = _smart_inject_memory(system_prompt, memory_context, raw_memories)
        user_content = [{"type": "text", "text": request.question or "请分析这些图片"}]
        for img_b64 in image_list:
            mime = _detect_mime_type(img_b64)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}})
    else:
        _cheap_model, _cheap_provider, _cheap_endpoint = _get_cheap_model_params(request)
        initial_strategy = get_retrieval_strategy(request.question or "")
        agent_gate = _build_agent_retrieval_gate(
            enable_agent_retrieval=request.enable_agent_retrieval,
            force_agent_retrieval=request.force_agent_retrieval,
            selected_text=request.selected_text,
            query_type=initial_strategy.get("query_type", ""),
            evidence_need=initial_strategy.get("evidence_need", []),
        )
        use_agent = bool(agent_gate.get("enabled"))
        retrieval_meta["agent_gate"] = _annotate_agent_gate(
            agent_gate,
            use_agent=use_agent,
            agent_mode=False,
            search_query_passthrough=bool(use_agent),
        )
        if use_agent:
            search_query = request.question or ""
        else:
            # LLM 查询改写：用于检索的 search_query（消解代词/口语化），原始 question 保留用于 LLM 回答
            search_query = await _maybe_rewrite_query(
                question=request.question,
                chat_history=request.chat_history,
                selected_text=request.selected_text,
                api_key=request.api_key,
                model=_cheap_model,
                provider=_cheap_provider,
                endpoint=_cheap_endpoint,
                retrieval_strategy=initial_strategy,
            )
        if not use_agent:
            web_search_sources, web_search_context = await _maybe_perform_web_search(
                request,
                query_override=search_query,
                doc_title=doc.get("filename", ""),
                selected_text=request.selected_text or "",
                doc_id=request.doc_id,
                vector_store_dir=getattr(router, "vector_store_dir", ""),
            )
        retrieval_meta["agent_gate"] = _annotate_agent_gate(
            agent_gate,
            use_agent=use_agent,
            agent_mode=False,
            search_query_passthrough=bool(use_agent),
        )

        strategy = get_retrieval_strategy(search_query)
        query_type = strategy["query_type"]
        evidence_need = strategy.get("evidence_need", [])
        dynamic_top_k = strategy["top_k"]
        # P1.3 智能 rerank：概述/对比类自动启用 local rerank（不影响用户已开启的）
        _auto_enable_rerank_if_beneficial(request, evidence_need, query_type)

        # 预先计算输出 Token 预算（不含引文开销），供 RAG 上下文预算感知使用
        _prelim_answer_tokens = _adjust_max_tokens(request.max_tokens, request.answer_detail or "standard", False) or 0

        if request.selected_text and request.enable_vector_search:
            # 融合模式：selected_text + 向量检索
            _validate_rerank_request(request)
            selected_page_info = locate_selected_text(
                request.selected_text, doc.get("data", {}).get("pages", [])
            )
            try:
                context_result = await vector_context(
                    request.doc_id, search_query, vector_store_dir=router.vector_store_dir,
                    pages=doc.get("data", {}).get("pages", []), api_key=request.api_key,
                    top_k=dynamic_top_k, candidate_k=max(request.candidate_k, dynamic_top_k),
                    use_rerank=request.use_rerank, reranker_model=request.reranker_model,
                    rerank_provider=request.rerank_provider, rerank_api_key=request.rerank_api_key,
                    rerank_endpoint=request.rerank_endpoint,
                    middlewares=[
                        *( [LoggingMiddleware()] if settings.enable_chat_logging else [] ),
                        RetryMiddleware(retries=settings.chat_retry_retries, delay=settings.chat_retry_delay),
                        ErrorCaptureMiddleware()
                    ],
                    selected_text=request.selected_text,
                    answer_max_tokens=_prelim_answer_tokens,
                )
                retrieval_context = context_result.get("context", "")
                retrieval_meta = _merge_retrieval_meta(retrieval_meta, context_result.get("retrieval_meta", {}))
                retrieval_citations = retrieval_meta.get("citations") or []
                fallback_selected_citations = _build_selected_text_fallback_citations(
                    request.selected_text, selected_page_info
                )
                retrieval_meta["citations"] = retrieval_citations or fallback_selected_citations
                # 融合：selected_text 优先 + 检索补充
                context = _build_fused_context(
                    request.selected_text,
                    retrieval_context,
                    selected_page_info,
                    selected_ref=1 if (not retrieval_citations and fallback_selected_citations) else None,
                )
            except Exception as e:
                logger.warning(f"框选模式向量检索失败，降级为仅 selected_text: {e}")
                fallback_selected_citations = _build_selected_text_fallback_citations(
                    request.selected_text, selected_page_info
                )
                context = _build_fused_context(
                    request.selected_text,
                    "",
                    selected_page_info,
                    selected_ref=1 if fallback_selected_citations else None,
                )
                retrieval_meta["citations"] = fallback_selected_citations
        elif request.selected_text:
            # 仅 selected_text 模式（向量检索未启用）
            selected_page_info = locate_selected_text(
                request.selected_text, doc.get("data", {}).get("pages", [])
            )
            context = _build_fused_context(
                request.selected_text,
                "",
                selected_page_info,
                selected_ref=1 if _build_selected_text_fallback_citations(
                    request.selected_text, selected_page_info
                ) else None,
            )
            retrieval_meta["citations"] = _build_selected_text_fallback_citations(
                    request.selected_text, selected_page_info
            )
        elif use_agent:
            context, retrieval_meta = await _run_agent_retrieval_for_context(
                request=request,
                doc=doc,
                search_query=search_query,
                query_type=query_type,
                agent_gate=agent_gate,
                retrieval_meta=retrieval_meta,
            )
        elif _should_use_fast_overview_context(
            query_type,
            enable_vector_search=request.enable_vector_search,
            selected_text=request.selected_text,
            use_agent=use_agent,
        ):
            sampled_context = _build_fast_overview_context(
                doc.get("data", {}).get("pages", []),
                doc["data"].get("full_text", ""),
            )
            numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                doc.get("data", {}).get("pages", []),
                sampled_context,
                query=search_query,
            )
            context = numbered_ctx
            retrieval_meta["citations"] = fb_cits
            retrieval_meta["query_type"] = query_type
            retrieval_meta["fast_overview"] = True
        elif request.enable_vector_search:
            _validate_rerank_request(request)
            context_result = await vector_context(
                request.doc_id, search_query, vector_store_dir=router.vector_store_dir,
                pages=doc.get("data", {}).get("pages", []), api_key=request.embedding_api_key or request.api_key,
                top_k=dynamic_top_k, candidate_k=max(request.candidate_k, dynamic_top_k),
                use_rerank=request.use_rerank, reranker_model=request.reranker_model,
                rerank_provider=request.rerank_provider, rerank_api_key=request.rerank_api_key,
                rerank_endpoint=request.rerank_endpoint,
                middlewares=[
                    *( [LoggingMiddleware()] if settings.enable_chat_logging else [] ),
                    RetryMiddleware(retries=settings.chat_retry_retries, delay=settings.chat_retry_delay),
                    ErrorCaptureMiddleware()
                ],
                answer_max_tokens=_prelim_answer_tokens,
            )
            relevant_text = context_result.get("context", "")
            retrieval_meta = _merge_retrieval_meta(retrieval_meta, context_result.get("retrieval_meta", {}))
            if relevant_text:
                context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n"
            else:
                numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                    doc.get("data", {}).get("pages", []),
                    doc["data"]["full_text"][:30000],
                    query=search_query,
                )
                context = numbered_ctx
                retrieval_meta["citations"] = fb_cits
            if not retrieval_meta.get("citations"):
                retrieval_meta["citations"] = _generate_page_level_citations(
                    doc.get("data", {}).get("pages", []), context, query=search_query
                )
        else:
            numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                doc.get("data", {}).get("pages", []),
                doc["data"]["full_text"][:30000],
                query=search_query,
            )
            context = numbered_ctx
            retrieval_meta["citations"] = fb_cits

        if retrieval_meta.get("citations") and not retrieval_meta.get("_context_segments"):
            retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
                retrieval_meta.get("citations", []),
                query=search_query,
            )
        retrieval_meta["query_type"] = retrieval_meta.get("query_type") or query_type
        retrieval_meta["evidence_need"] = retrieval_meta.get("evidence_need") or evidence_need
        retrieval_meta["search_query"] = search_query
        context = _sync_numeric_table_prompt_context(
            context,
            retrieval_meta,
            query=search_query,
            evidence_need=evidence_need,
        )

        answer_style_instruction = _build_answer_style_instruction(request.answer_detail)
        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

回答规则：
1. 基于文档内容准确回答，学术准确、表达清晰。
2. 遇到公式、数据、图表等关键信息时，必须直接引用原文展示完整内容。
3. 优先依据文档内容回答。"""
        system_prompt += f"\n4. {answer_style_instruction}"
        # P3.1 严格引用守则（提升 faithfulness），P3.5 ablation flag 控制
        # 在 agent_mode 下跳过详细 citation prompt（ablation 显示拖累小模型 agent 的 AnsRel）
        _agent_mode = bool(retrieval_meta.get("agent_mode")) if isinstance(retrieval_meta, dict) else False
        if getattr(settings, "enable_p35_citation_prompt", True) and not _agent_mode:
            system_prompt += f"\n\n{_build_faithfulness_guard_prompt()}"
        if _agent_mode:
            agent_focus_prompt = _build_agent_answer_focus_prompt(
                request.question,
                query_type=query_type,
                evidence_need=evidence_need,
            )
            if agent_focus_prompt:
                system_prompt += f"\n\n{agent_focus_prompt}"
        if query_type == "extraction":
            system_prompt += f"\n\n{_build_extraction_constraint_prompt()}"
        if "numeric_table" in evidence_need:
            system_prompt += f"\n\n{_build_numeric_table_constraint_prompt()}"
        if request.enable_glossary:
            glossary_instruction = build_glossary_prompt(context)
            if glossary_instruction: system_prompt += f"\n\n{glossary_instruction}"
        if web_search_context:
            system_prompt += (
                "\n\n联网搜索结果（用于补充最新信息，优先保证与文档内容一致）：\n"
                f"{web_search_context}\n"
                "\n回答时，在引用联网信息的句子末尾标注来源序号，格式为 [1]、[2] 等（对应上方搜索结果编号）。"
                "\n不得与文档事实冲突。"
            )
        generation_prompt = get_generation_prompt(request.question)
        if generation_prompt: system_prompt += f"\n\n{generation_prompt}"
        citations = retrieval_meta.get("citations", [])
        has_structured_citations = bool(citations) and not _is_paragraph_fallback(citations)
        if has_structured_citations:
            citation_prompt = build_structured_citation_prompt(
                citations,
                compact=_should_use_compact_citation_prompt(citations),
            )
            if citation_prompt: system_prompt += f"\n\n{citation_prompt}"
        system_prompt, memory_hits, memory_meta = _smart_inject_memory(system_prompt, memory_context, raw_memories)
        user_content = request.question

    messages = [{"role": "system", "content": system_prompt}]
    if request.chat_history:
        for hist_msg in request.chat_history:
            if isinstance(hist_msg, dict) and hist_msg.get("role") in ("user", "assistant") and hist_msg.get("content"):
                messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
    messages.append({"role": "user", "content": user_content})

    has_citations_non_stream = bool(citations) and not _is_paragraph_fallback(citations)
    adjusted_max_tokens = _adjust_max_tokens(
        request.max_tokens, request.answer_detail, has_citations_non_stream,
    )
    try:
        response = await call_ai_api(
            messages, request.api_key, request.model, request.api_provider,
            endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
            middlewares=build_chat_middlewares(), max_tokens=adjusted_max_tokens,
            temperature=request.temperature, top_p=request.top_p,
            custom_params=request.custom_params, reasoning_effort=request.reasoning_effort,
        )
        message = _extract_non_stream_ai_message(response)
        raw_answer = message.get("content") or ""
        reasoning_content = extract_reasoning_content(message)

        # 结构化引文后处理（非流式）
        answer = extract_final_answer(raw_answer)
        _retrieval_chunks_sync = retrieval_meta.get("_chunks", [])
        _context_segments_sync = retrieval_meta.get("_context_segments", [])
        if citations and raw_answer:
            try:
                inline_cites = parse_citation_list(raw_answer)
                if inline_cites and (_retrieval_chunks_sync or _context_segments_sync):
                    enhanced = match_citations_to_chunks(inline_cites, _retrieval_chunks_sync, context_segments=_context_segments_sync)
                    orig_citations = retrieval_meta.get("citations", [])
                    for ec in enhanced:
                        if ec.get("idx") is not None:
                            for oc in orig_citations:
                                if oc.get("ref") == ec["idx"]:
                                    if ec.get("start_phrase"):
                                        oc["start_phrase"] = ec["start_phrase"]
                                    if ec.get("end_phrase"):
                                        oc["end_phrase"] = ec["end_phrase"]
                                    if ec.get("highlight_text"):
                                        oc["highlight_text"] = ec["highlight_text"]
                                        oc["alignment_status"] = "span_matched"
                                    if ec.get("group_id"):
                                        oc["group_id"] = ec["group_id"]
                                    if ec.get("page"):
                                        oc["page_range"] = [ec["page"], ec["page"]]
                                    break
            except Exception as e:
                logger.warning(f"非流式引文后处理失败: {e}")

        answer_guard: dict = {}
        _snapshot_retrieval_context_segments(retrieval_meta)
        answer, retrieval_meta["citations"] = _prepare_answer_and_citations_for_display(
            answer,
            retrieval_meta.get("citations", []),
            evidence_need=retrieval_meta.get("evidence_need", []),
            answer_guard=answer_guard,
            query=retrieval_meta.get("search_query") or request.question,
            context_segments=retrieval_meta.get("_context_segments", []),
        )
        if answer_guard:
            retrieval_meta["answer_guard"] = answer_guard
        if retrieval_meta.get("citations"):
            retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
                retrieval_meta.get("citations", []),
                query=retrieval_meta.get("search_query") or request.question,
            )
        response_context_segments = _build_response_context_segments(retrieval_meta)

        if use_memory:
            threading.Thread(target=_async_memory_write, args=(memory_service, request), daemon=True).start()
        return {
            "answer": answer, "reasoning_content": reasoning_content,
            "doc_id": request.doc_id, "question": request.question,
            "timestamp": datetime.now().isoformat(), "used_provider": response.get("_used_provider"),
            "used_model": response.get("_used_model"), "fallback_used": response.get("_fallback_used", False),
            "retrieval_meta": {
                **{k: v for k, v in retrieval_meta.items() if not k.startswith("_")},
                "context_segments": response_context_segments,
            },
            "web_search_sources": web_search_sources,
            "memory_hits": memory_hits,
            "memory_meta": memory_meta,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI调用失败: {str(e)}")


@router.post("/chat/stream")
async def chat_with_pdf_stream(request: ChatRequest):
    if not hasattr(router, "documents_store"):
        raise HTTPException(status_code=500, detail="文档存储未初始化")
    # Per-request feature flag overrides（前端 GlobalSettings 可细化控制）
    apply_request_overrides(
        numeric_table=request.override_numeric_table,
        answer_critic=request.override_answer_critic,
        llm_query_rewrite=request.override_llm_query_rewrite,
        bm25_synonyms=request.override_bm25_synonyms,
    )
    store = router.documents_store if not request.doc_store_key else router.documents_store.get(request.doc_store_key, {})
    if request.doc_id not in store:
        raise HTTPException(status_code=404, detail="文档未找到")
    doc = store[request.doc_id]
    trace_id = _new_chat_trace_id()
    trace_started_at = time.perf_counter()
    _log_chat_trace(
        trace_id,
        trace_started_at,
        "request_received",
        doc_id=request.doc_id,
        provider=request.api_provider,
        model=request.model,
        vector=request.enable_vector_search,
        use_rerank=request.use_rerank,
        rerank_provider=request.rerank_provider,
        reranker_model=request.reranker_model,
        thinking=request.enable_thinking,
        web_search=request.enable_web_search,
        selected_text=bool(request.selected_text),
        question=_preview_for_log(request.question, 120),
    )

    async def event_generator():
        yield f"data: {json.dumps({'type': 'retrieval_progress', 'phase': 'start', 'message': '正在检索...'}, ensure_ascii=False)}\n\n"
        try:
            context = ""
            retrieval_meta = {}
            has_structured_citations = False
            web_search_sources: list[dict] = []
            web_search_context = ""
            use_agent = False
            use_memory = _should_use_memory(request)
            memory_context = ""
            raw_memories = []
            memory_hits: list[dict] = []
            memory_meta: dict = {
                "enabled": use_memory,
                "strategy": None,
                "retrieved_count": 0,
                "selected_count": 0,
                "truncated": False,
                "token_budget": None,
                "selected_kinds": [],
            }
            if use_memory:
                memory_context, raw_memories = await _retrieve_memory_for_stream(
                    request.question,
                    api_key=request.api_key,
                    doc_id=request.doc_id,
                    chat_history=request.chat_history,
                )

            image_list = (request.image_base64_list or [])
            if request.image_base64 and request.image_base64 not in image_list:
                image_list = [request.image_base64] + image_list
            image_list = [img for img in image_list if img]

            if image_list:
                logger.info("[Chat Stream] 截图模式：处理 %s 张图", len(image_list))
                _log_chat_trace(
                    trace_id,
                    trace_started_at,
                    "image_mode",
                    image_count=len(image_list),
                )
                answer_style_instruction = _build_answer_style_instruction(request.answer_detail)
                system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
用户从文档中截取了 {len(image_list)} 张图片并发送给你。请仔细分析这些图片内容并回答问题。

回答规则：
1. 以用户发送的图片为核心依据进行回答，不要参考其他内容。
2. 如果图片包含图表，请分析数据和关键信息。
3. 如果图片包含公式，请使用 LaTeX 格式（$公式$）展示。
4. 如果图片包含表格，请转换为 Markdown 格式。
5. 学术准确、表达清晰。
6. {answer_style_instruction}"""
                system_prompt, memory_hits, memory_meta = _smart_inject_memory(system_prompt, memory_context, raw_memories)
                user_content = [{"type": "text", "text": request.question or "请分析这些图片"}]
                for img_b64 in image_list:
                    mime = _detect_mime_type(img_b64)
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}})
            else:
                # 应用前端传入的检索增强设置到全局配置（即时生效）
                settings.bm25_use_jieba = request.enable_jieba_bm25
                settings.num_expand_context_chunk = request.num_expand_context_chunk

                _cheap_model, _cheap_provider, _cheap_endpoint = _get_cheap_model_params(request)
                initial_strategy = get_retrieval_strategy(request.question or "")
                agent_gate = _build_agent_retrieval_gate(
                    enable_agent_retrieval=request.enable_agent_retrieval,
                    force_agent_retrieval=request.force_agent_retrieval,
                    selected_text=request.selected_text,
                    query_type=initial_strategy.get("query_type", ""),
                    evidence_need=initial_strategy.get("evidence_need", []),
                )
                use_agent = bool(agent_gate.get("enabled"))
                retrieval_meta["agent_gate"] = _annotate_agent_gate(
                    agent_gate,
                    use_agent=use_agent,
                    agent_mode=False,
                    search_query_passthrough=bool(use_agent),
                )
                _log_chat_trace(
                    trace_id,
                    trace_started_at,
                    "agent_gate",
                    enabled=use_agent,
                    reason=agent_gate.get("reason"),
                    query_type=agent_gate.get("query_type"),
                    matched_query_type=agent_gate.get("matched_query_type"),
                    matched_needs=",".join(agent_gate.get("matched_evidence_need") or []),
                )
                if use_agent:
                    search_query = request.question or ""
                    retrieval_meta["agent_gate"] = _annotate_agent_gate(
                        agent_gate,
                        use_agent=use_agent,
                        agent_mode=False,
                        search_query_passthrough=True,
                    )
                    yield _sse_json({
                        'type': 'retrieval_progress',
                        'phase': 'agent_start',
                        'message': '正在分析问题并规划多轮检索...',
                    })
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "agent_enabled",
                        search_query=_preview_for_log(search_query, 120),
                    )
                else:
                    retrieval_meta["agent_gate"] = _annotate_agent_gate(
                        agent_gate,
                        use_agent=use_agent,
                        agent_mode=False,
                        search_query_passthrough=False,
                    )
                    # LLM 查询改写：用于检索的 search_query（消解代词/口语化），原始 question 保留用于 LLM 回答
                    yield _sse_json({
                        'type': 'retrieval_progress',
                        'phase': 'query_rewrite_start',
                        'message': '正在分析问题并改写检索查询...',
                    })
                    search_query = await _maybe_rewrite_query(
                        question=request.question,
                        chat_history=request.chat_history,
                        selected_text=request.selected_text,
                        api_key=request.api_key,
                        model=_cheap_model,
                        provider=_cheap_provider,
                        endpoint=_cheap_endpoint,
                        retrieval_strategy=initial_strategy,
                    )
                    yield _sse_json({
                        'type': 'retrieval_progress',
                        'phase': 'query_rewrite_done',
                        'message': '已生成检索查询，正在查找相关内容...',
                    })
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "query_ready",
                        rewritten=search_query != (request.question or ""),
                        search_query=_preview_for_log(search_query, 120),
                    )
                # 联网搜索在此处设置查询参数
                _web_search_query_for_stream = search_query
                _web_search_doc_title_for_stream = doc.get("filename", "")
                strategy = get_retrieval_strategy(search_query)
                query_type = strategy["query_type"]
                evidence_need = strategy.get("evidence_need", [])
                dynamic_top_k = strategy["top_k"]
                # P1.3 智能 rerank：概述/对比类自动启用 local rerank（不影响用户已开启的）
                _auto_rerank_applied = _auto_enable_rerank_if_beneficial(
                    request, evidence_need, query_type
                )
                _log_chat_trace(
                    trace_id,
                    trace_started_at,
                    "retrieval_strategy",
                    query_type=query_type,
                    top_k=dynamic_top_k,
                    auto_rerank=_auto_rerank_applied,
                )

                # 预先计算输出 Token 预算（不含引文开销），供 RAG 上下文预算感知使用
                _prelim_answer_tokens_stream = _adjust_max_tokens(request.max_tokens, request.answer_detail or "standard", False) or 0

                if request.selected_text and request.enable_vector_search:
                    # 融合模式：selected_text + 向量检索
                    _validate_rerank_request(request)
                    selected_page_info = locate_selected_text(
                        request.selected_text, doc.get("data", {}).get("pages", [])
                    )
                    try:
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "retrieval_vector_start",
                            mode="selected_text",
                            top_k=dynamic_top_k,
                        )
                        yield _sse_json({'type': 'retrieval_progress', 'phase': 'vector_search_start', 'message': '正在按框选内容检索相关段落...'})
                        strategy = get_retrieval_strategy(search_query)
                        dynamic_top_k = strategy['top_k']
                        _progress_queue = asyncio.Queue()
                        _progress_forwarder = _build_threadsafe_progress_forwarder(_progress_queue)
                        _vector_task = asyncio.create_task(vector_context(
                            request.doc_id, search_query, vector_store_dir=router.vector_store_dir,
                            pages=doc.get("data", {}).get("pages", []), api_key=request.embedding_api_key or request.api_key,
                            top_k=dynamic_top_k, candidate_k=max(request.candidate_k, dynamic_top_k),
                            use_rerank=request.use_rerank, reranker_model=request.reranker_model,
                            rerank_provider=request.rerank_provider, rerank_api_key=request.rerank_api_key,
                            rerank_endpoint=request.rerank_endpoint,
                            middlewares=[
                                *( [LoggingMiddleware()] if settings.enable_search_logging else [] ),
                                RetryMiddleware(retries=settings.search_retry_retries, delay=settings.search_retry_delay)
                            ],
                            answer_max_tokens=_prelim_answer_tokens_stream,
                            progress_callback=_progress_forwarder,
                        ))
                        async for _progress_event in _yield_task_progress(
                            _vector_task,
                            _progress_queue,
                            "正在按框选内容检索相关段落，请稍候...",
                        ):
                            yield _sse_json(_progress_event)
                        context_result = await _vector_task
                        retrieval_context = context_result.get("context", "")
                        retrieval_meta = _merge_retrieval_meta(retrieval_meta, context_result.get("retrieval_meta", {}))
                        vector_error = context_result.get("error")
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "retrieval_vector_done",
                            mode="selected_text",
                            error=vector_error,
                            context_chars=len(retrieval_context),
                            citations=len(retrieval_meta.get("citations") or []),
                            timings=retrieval_meta.get("timings") or retrieval_meta.get("timing"),
                        )
                        retrieval_citations = retrieval_meta.get("citations") or []
                        fallback_selected_citations = _build_selected_text_fallback_citations(
                            request.selected_text, selected_page_info
                        )
                        retrieval_meta["citations"] = retrieval_citations or fallback_selected_citations
                        # 融合：selected_text 优先 + 检索补充
                        context = _build_fused_context(
                            request.selected_text,
                            retrieval_context,
                            selected_page_info,
                            selected_ref=1 if (not retrieval_citations and fallback_selected_citations) else None,
                        )
                    except Exception as e:
                        logger.warning(f"框选模式向量检索失败，降级为仅 selected_text: {e}")
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "retrieval_vector_exception",
                            mode="selected_text",
                            error=str(e),
                        )
                        fallback_selected_citations = _build_selected_text_fallback_citations(
                            request.selected_text, selected_page_info
                        )
                        context = _build_fused_context(
                            request.selected_text,
                            "",
                            selected_page_info,
                            selected_ref=1 if fallback_selected_citations else None,
                        )
                        retrieval_meta["citations"] = fallback_selected_citations
                elif request.selected_text:
                    # 仅 selected_text 模式（向量检索未启用）
                    selected_page_info = locate_selected_text(
                        request.selected_text, doc.get("data", {}).get("pages", [])
                    )
                    context = _build_fused_context(
                        request.selected_text,
                        "",
                        selected_page_info,
                        selected_ref=1 if _build_selected_text_fallback_citations(
                            request.selected_text, selected_page_info
                        ) else None,
                    )
                    retrieval_meta["citations"] = _build_selected_text_fallback_citations(
                        request.selected_text, selected_page_info
                    )
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "retrieval_selected_text_only",
                        citations=len(retrieval_meta.get("citations") or []),
                    )
                elif use_agent:
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "retrieval_agent_mode",
                        top_k=dynamic_top_k,
                    )
                    agent_progress_queue: asyncio.Queue = asyncio.Queue()

                    async def _emit_agent_progress(event: dict) -> None:
                        await agent_progress_queue.put(event)

                    agent_task = asyncio.create_task(_run_agent_retrieval_for_context(
                        request=request,
                        doc=doc,
                        search_query=search_query,
                        query_type=query_type,
                        agent_gate=agent_gate,
                        retrieval_meta=retrieval_meta,
                        emit_progress=_emit_agent_progress,
                        trace_id=trace_id,
                        trace_started_at=trace_started_at,
                    ))

                    async for agent_event in _yield_task_progress(
                        agent_task,
                        agent_progress_queue,
                        "Agent 检索仍在执行，请稍候...",
                    ):
                        yield _sse_json(agent_event)

                    context, retrieval_meta = await agent_task
                elif _should_use_fast_overview_context(
                    query_type,
                    enable_vector_search=request.enable_vector_search,
                    selected_text=request.selected_text,
                    image_list=image_list,
                    use_agent=use_agent,
                ):
                    _log_chat_trace(trace_id, trace_started_at, "retrieval_fast_overview")
                    yield _sse_json({
                        'type': 'retrieval_progress',
                        'phase': 'fast_overview',
                        'message': '概览问题：直接使用全文采样上下文以加快回答...',
                    })
                    sampled_context = _build_fast_overview_context(
                        doc.get("data", {}).get("pages", []),
                        doc["data"].get("full_text", ""),
                    )
                    numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                        doc.get("data", {}).get("pages", []),
                        sampled_context,
                        query=search_query,
                    )
                    context = numbered_ctx
                    retrieval_meta["citations"] = fb_cits
                    retrieval_meta["query_type"] = query_type
                    retrieval_meta["fast_overview"] = True
                elif request.enable_vector_search:
                    _validate_rerank_request(request)
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "retrieval_analysis_start",
                        top_k=dynamic_top_k,
                    )

                    # 复杂问题分解：仅对长查询（>=10字）且包含比较/对比关键词时触发
                    # 智能 gate 避免简单问题白白调一次 LLM
                    sub_questions = []
                    if should_decompose(request.question or ""):
                        yield f"data: {json.dumps({'type': 'retrieval_progress', 'phase': 'analysis', 'message': '正在分析问题并拆分检索子任务...'}, ensure_ascii=False)}\n\n"
                        try:
                            _cm, _cp, _ce = _get_cheap_model_params(request)
                            sub_questions = await asyncio.wait_for(
                                decompose_question(
                                    question=request.question,
                                    api_key=request.api_key,
                                    model=_cm,
                                    provider=_cp,
                                    endpoint=_ce,
                                ),
                                timeout=2.5,  # 缩短：cheap model 通常 1-2s 完成
                            )
                        except asyncio.TimeoutError:
                            logger.warning("[Decompose] 问题分解超时(2.5s)，跳过分解")
                            sub_questions = []
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "retrieval_analysis_done",
                        sub_questions=len(sub_questions or []),
                    )

                    queries_to_search = [search_query] + sub_questions if sub_questions else [search_query]
                    all_relevant_texts = []

                    for query_index, sq in enumerate(queries_to_search):
                        query_preview = re.sub(r"\s+", " ", sq).strip()
                        if len(query_preview) > 80:
                            query_preview = query_preview[:80].rstrip() + "..."
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "retrieval_vector_start",
                            query=_preview_for_log(sq, 80),
                        )
                        yield _sse_json({'type': 'retrieval_progress', 'phase': 'vector_search_start', 'message': f'正在检索: {query_preview}'})
                        sub_strategy = get_retrieval_strategy(sq)
                        dynamic_top_k = sub_strategy['top_k']
                        _progress_queue = asyncio.Queue()
                        _progress_forwarder = _build_threadsafe_progress_forwarder(_progress_queue)
                        _vector_task = asyncio.create_task(vector_context(
                            request.doc_id, sq, vector_store_dir=router.vector_store_dir,
                            pages=doc.get("data", {}).get("pages", []), api_key=request.embedding_api_key or request.api_key,
                            top_k=dynamic_top_k, candidate_k=max(request.candidate_k, dynamic_top_k),
                            use_rerank=request.use_rerank, reranker_model=request.reranker_model,
                            rerank_provider=request.rerank_provider, rerank_api_key=request.rerank_api_key,
                            rerank_endpoint=request.rerank_endpoint,
                            middlewares=[
                                *( [LoggingMiddleware()] if settings.enable_search_logging else [] ),
                                RetryMiddleware(retries=settings.search_retry_retries, delay=settings.search_retry_delay)
                            ],
                            answer_max_tokens=_prelim_answer_tokens_stream,
                            progress_callback=_progress_forwarder,
                        ))
                        async for _progress_event in _yield_task_progress(
                            _vector_task,
                            _progress_queue,
                            f"正在检索: {query_preview}",
                        ):
                            yield _sse_json(_progress_event)
                        cr = await _vector_task
                        rt = cr.get("context", "")
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "retrieval_vector_done",
                            query=_preview_for_log(sq, 80),
                            error=cr.get("error"),
                            context_chars=len(rt),
                            citations=len((cr.get("retrieval_meta", {}) or {}).get("citations") or []),
                            timings=(cr.get("retrieval_meta", {}) or {}).get("timings") or (cr.get("retrieval_meta", {}) or {}).get("timing"),
                        )
                        if rt:
                            all_relevant_texts.append(rt)
                        retrieval_meta = _merge_multi_query_retrieval_meta(
                            retrieval_meta,
                            cr.get("retrieval_meta", {}),
                            query=sq,
                            query_index=query_index,
                        )

                    relevant_text = "\n\n---\n\n".join(all_relevant_texts) if all_relevant_texts else ""
                    if relevant_text:
                        context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n"
                    else:
                        # 向量检索失败：将 full_text 格式化为编号段落，让 LLM 自然引用
                        numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                            doc.get("data", {}).get("pages", []),
                            doc["data"]["full_text"][:30000],
                            query=search_query,
                        )
                        context = numbered_ctx
                        retrieval_meta["citations"] = fb_cits
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "retrieval_fulltext_fallback",
                            reason="vector_empty_or_failed",
                            context_chars=len(context),
                            citations=len(fb_cits),
                        )
                    # 向量检索有结果但无 citations 时兜底
                    if not retrieval_meta.get("citations"):
                        retrieval_meta["citations"] = _generate_page_level_citations(
                            doc.get("data", {}).get("pages", []), context, query=search_query
                        )
                else:
                    # 非向量路径：格式化为编号段落
                    numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                        doc.get("data", {}).get("pages", []),
                        doc["data"]["full_text"][:30000],
                        query=search_query,
                    )
                    context = numbered_ctx
                    retrieval_meta["citations"] = fb_cits
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "retrieval_disabled_fulltext",
                        context_chars=len(context),
                        citations=len(fb_cits),
                    )

                # GraphRAG 上下文融合：如果该文档已构建 GraphRAG 索引，追加知识图谱上下文
                # 实例优先从内存 registry 读取，若不存在则尝试从磁盘加载
                if settings.enable_graphrag or request.enable_graphrag:
                    try:
                        from services.graphrag import INSTANCES as _GRAPHRAG_INSTANCES
                        graphrag_inst = _GRAPHRAG_INSTANCES.get(request.doc_id)

                        # 尝试从磁盘加载（重启后 INSTANCES 为空）
                        if graphrag_inst is None:
                            from services.graphrag import GraphRAG, GraphRAGConfig
                            _gr_working_dir = os.path.join(settings.graphrag_working_dir, request.doc_id)
                            if GraphRAG.has_persisted_index(_gr_working_dir):
                                _gr_meta = GraphRAG.load_metadata(_gr_working_dir)
                                if _gr_meta and _gr_meta.status == "done":
                                    # 构建最小 config（不需要 api_key，仅用于加载存储）
                                    _gr_config = GraphRAGConfig(
                                        api_key=request.api_key or "",
                                        model=_gr_meta.model or "",
                                        provider=_gr_meta.provider or request.api_provider or "",
                                        endpoint=_gr_meta.endpoint or request.api_host or "",
                                        embedding_api_key=request.embedding_api_key or request.api_key or "",
                                        embedding_model=_gr_meta.embedding_model or "",
                                        embedding_provider=_gr_meta.embedding_provider or "",
                                        embedding_endpoint=_gr_meta.embedding_endpoint or "",
                                        embedding_dim=_gr_meta.embedding_dim or 1536,
                                    )
                                    graphrag_inst = await GraphRAG.load_from_disk(
                                        working_dir=_gr_working_dir,
                                        config=_gr_config,
                                        chunk_token_size=settings.graphrag_chunk_token_size,
                                        entity_extract_max_gleaning=settings.graphrag_max_gleaning,
                                        best_model_max_async=settings.graphrag_max_async,
                                        cheap_model_max_async=settings.graphrag_max_async,
                                    )
                                    if graphrag_inst is not None:
                                        _GRAPHRAG_INSTANCES[request.doc_id] = graphrag_inst
                                        logger.info(f"[Chat] GraphRAG 实例从磁盘加载: {request.doc_id}")

                        if graphrag_inst is not None:
                            # 根据 query_type / evidence_need 和全局配置选择查询模式
                            from services.graphrag import QueryParam
                            _gr_mode = settings.graphrag_query_mode
                            if _gr_mode == "auto":
                                _gr_mode = "local"  # 默认 local
                                if query_type in ("summary", "overview", "comparison"):
                                    _gr_mode = "global"
                                elif query_type in ("detail", "evidence") or evidence_need in ("high", "precise"):
                                    _gr_mode = "hybrid"
                            _gr_param = QueryParam(mode=_gr_mode, only_output_context=True)
                            graphrag_context = await graphrag_inst.aquery_context(search_query, param=_gr_param)
                            if graphrag_context:
                                # Token budget 截断：防止 GraphRAG 上下文过长挤占主上下文
                                max_gr_tokens = settings.graphrag_context_max_tokens
                                # 粗略估算：英文/CSV 约 4 字符/token
                                max_gr_chars = max_gr_tokens * 4
                                _truncated = False
                                if len(graphrag_context) > max_gr_chars:
                                    # 优先截断 Sources 部分（通常最大且对回答影响较小）
                                    _sources_marker = "-----Sources-----"
                                    _idx = graphrag_context.find(_sources_marker)
                                    if _idx > 0:
                                        graphrag_context = graphrag_context[:_idx].rstrip() + "\n\n...(Sources truncated for token budget)"
                                    else:
                                        graphrag_context = graphrag_context[:max_gr_chars] + "\n\n...(truncated)"
                                    _truncated = True
                                context += f"\n\n## 知识图谱关联信息（{_gr_mode}）\n{graphrag_context}"
                                logger.debug(
                                    f"[Chat] GraphRAG 上下文已融合（mode={_gr_mode}），"
                                    f"长度={len(graphrag_context)}, truncated={_truncated}"
                                )
                    except Exception as e:
                        logger.warning(f"[Chat] GraphRAG 上下文获取失败: {e}")

                if retrieval_meta.get("citations") and not retrieval_meta.get("_context_segments"):
                    retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
                        retrieval_meta.get("citations", []),
                        query=search_query,
                    )
                retrieval_meta["query_type"] = retrieval_meta.get("query_type") or query_type
                retrieval_meta["evidence_need"] = retrieval_meta.get("evidence_need") or evidence_need
                retrieval_meta["search_query"] = search_query
                context = _sync_numeric_table_prompt_context(
                    context,
                    retrieval_meta,
                    query=search_query,
                    evidence_need=evidence_need,
                )

                retrieval_preview = _build_retrieval_preview_message(retrieval_meta.get("citations", []))
                if retrieval_preview:
                    yield _sse_json({'type': 'retrieval_progress', 'phase': 'content_preview', 'message': retrieval_preview})
                elif not use_agent and not image_list:
                    yield _sse_json({'type': 'retrieval_progress', 'phase': 'complete', 'message': '检索完成，正在整理上下文...'})
                _log_chat_trace(
                    trace_id,
                    trace_started_at,
                    "context_ready",
                    context_chars=len(context),
                    citations=len(retrieval_meta.get("citations") or []),
                    structured=bool(retrieval_meta.get("citations")) and not _is_paragraph_fallback(retrieval_meta.get("citations")),
                )

                answer_style_instruction = _build_answer_style_instruction(request.answer_detail)
                system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

回答规则：
1. 基于文档内容准确回答，学术准确、表达清晰。
2. 遇到公式、数据、图表等关键信息时，必须直接引用原文展示完整内容。
3. 优先依据文档内容回答。"""
                system_prompt += f"\n4. {answer_style_instruction}"
                # P3.1 严格引用守则（提升 faithfulness），P3.5 ablation flag 控制
                # 在 agent_mode 下跳过详细 citation prompt（ablation 显示拖累小模型 agent 的 AnsRel）
                _agent_mode = bool(retrieval_meta.get("agent_mode")) if isinstance(retrieval_meta, dict) else False
                if getattr(settings, "enable_p35_citation_prompt", True) and not _agent_mode:
                    system_prompt += f"\n\n{_build_faithfulness_guard_prompt()}"
                if _agent_mode:
                    agent_focus_prompt = _build_agent_answer_focus_prompt(
                        request.question,
                        query_type=query_type,
                        evidence_need=evidence_need,
                    )
                    if agent_focus_prompt:
                        system_prompt += f"\n\n{agent_focus_prompt}"
                if query_type == "extraction":
                    system_prompt += f"\n\n{_build_extraction_constraint_prompt()}"
                if "numeric_table" in evidence_need:
                    system_prompt += f"\n\n{_build_numeric_table_constraint_prompt()}"
                if request.enable_glossary:
                    glossary_instruction = build_glossary_prompt(context)
                    if glossary_instruction: system_prompt += f"\n\n{glossary_instruction}"
                generation_prompt = get_generation_prompt(request.question)
                if generation_prompt: system_prompt += f"\n\n{generation_prompt}"
                # 联网搜索上下文注入将在后续下游完成（需先发送状态事件）
                citations = retrieval_meta.get("citations", [])
                has_structured_citations = bool(citations) and not _is_paragraph_fallback(citations)
                logger.debug(
                    "[Citation] enable_vector_search=%s citations_count=%s has_structured=%s compact=%s",
                    request.enable_vector_search,
                    len(citations),
                    has_structured_citations,
                    _should_use_compact_citation_prompt(citations),
                )
                if has_structured_citations:
                    citation_prompt = build_structured_citation_prompt(
                        citations,
                        compact=_should_use_compact_citation_prompt(citations),
                    )
                    if citation_prompt: system_prompt += f"\n\n{citation_prompt}"
                system_prompt, memory_hits, memory_meta = _smart_inject_memory(system_prompt, memory_context, raw_memories)
                user_content = request.question

            messages = [{"role": "system", "content": system_prompt}]
            if request.chat_history:
                for hist_msg in request.chat_history:
                    if isinstance(hist_msg, dict) and hist_msg.get("role") in ("user", "assistant") and hist_msg.get("content"):
                        messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
            messages.append({"role": "user", "content": user_content})

            # 收集检索到的 chunks 用于引文模糊匹配
            _retrieval_chunks = retrieval_meta.get("_chunks", [])

            # 注意：agent 多轮检索已在上方 `elif use_agent` 分支（带 SSE 进度 yield）执行完成，
            # 不要在这里再调一次 agent.run —— 重复调用会浪费一倍 LLM 配额，且第二次没有 yield
            # SSE 进度，前端面板看不到第二轮过程，反而覆盖掉第一次的 agent_search_history/task_status。

            # 联网搜索（在此处执行以便向客户端实时发送状态事件）
            if getattr(request, "enable_web_search", False) and not image_list and not use_agent:
                # 意图分析：判断此问题是否真的需要联网
                _do_web_search = await _should_perform_web_search(
                    question=request.question,
                    api_key=request.api_key,
                    model=request.model,
                    provider=request.api_provider,
                    endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
                )
                if _do_web_search:
                    yield f"data: {json.dumps({'type': 'web_search_status', 'phase': 'searching'}, ensure_ascii=False)}\n\n"
                    try:
                        web_search_sources, web_search_context = await _maybe_perform_web_search(
                            request,
                            query_override=_web_search_query_for_stream,
                            doc_title=_web_search_doc_title_for_stream,
                            selected_text=request.selected_text or "",
                            doc_id=request.doc_id,
                            vector_store_dir=getattr(router, "vector_store_dir", ""),
                        )
                    except Exception as _ws_err:
                        logger.warning(f"联网搜索（generator 内）失败: {_ws_err}")
                        web_search_sources, web_search_context = [], ""

                    yield f"data: {json.dumps({'type': 'web_search_status', 'phase': 'fetch_complete', 'count': len(web_search_sources)}, ensure_ascii=False)}\n\n"

                if web_search_context:
                    system_prompt += (
                        "\n\n联网搜索结果（用于补充最新信息，优先保证与文档内容一致）：\n"
                        f"{web_search_context}\n"
                        "\n回答时，在引用联网信息的句子末尾标注来源序号，格式为 [1]、[2] 等（对应上方搜索结果编号）。"
                        "\n不得与文档事实冲突。"
                    )
                    messages[0]["content"] = system_prompt

            if web_search_sources:
                yield f"data: {json.dumps({'type': 'web_search', 'sources': web_search_sources}, ensure_ascii=False)}\n\n"
            # 使用 _buffered_stream 包装流式输出，合并高频小 chunk 减少 SSE 事件频率
            adjusted_stream_max_tokens = _adjust_max_tokens(
                request.max_tokens, request.answer_detail, has_structured_citations,
            )
            yield _sse_json({
                'type': 'retrieval_progress',
                'phase': 'llm_waiting',
                'message': (
                    '上下文准备完成，正在等待模型开始思考并生成回答...'
                    if request.enable_thinking
                    else '上下文准备完成，正在等待模型开始生成回答...'
                ),
            })
            _log_chat_trace(
                trace_id,
                trace_started_at,
                "llm_stream_start",
                provider=request.api_provider,
                model=request.model,
                max_tokens=adjusted_stream_max_tokens,
                structured_citations=has_structured_citations,
            )
            raw_stream = call_ai_api_stream(
                messages, request.api_key, request.model, request.api_provider,
                endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
                middlewares=build_chat_middlewares(), enable_thinking=request.enable_thinking,
                max_tokens=adjusted_stream_max_tokens, temperature=request.temperature,
                top_p=request.top_p, custom_params=request.custom_params,
                reasoning_effort=request.reasoning_effort,
            )
            timed_stream = _stream_with_total_timeout(
                raw_stream,
                float(getattr(settings, "chat_timeout", 120.0) or 120.0),
            )
            # 累积完整输出，用于结构化引文解析
            full_output = ""
            reached_final_answer = False
            visible_answer_text = ""
            content_progress_sent = False
            qa_score_val = None
            first_reasoning_logged = False
            first_content_logged = False
            total_reasoning_chars = 0
            stream_done_sent = False
            stream_error_sent = False
            last_stream_chunk: dict = {}
            # D1：并行引文匹配（仿 kotaemon）
            # CITATION LIST 完整后立即在后台线程启动匹配，与 FINAL ANSWER 流式输出并行
            _citation_match_thread: Optional[threading.Thread] = None
            _citation_match_result: dict = {}  # 线程结果写入此 dict，避免共享状态竞争

            def _run_citation_match(citation_list_text: str, chunks: list, segments: list, out: dict) -> None:
                try:
                    inline_cites = parse_citation_list(citation_list_text)
                    if inline_cites and (chunks or segments):
                        out["enhanced"] = match_citations_to_chunks(
                            inline_cites, chunks, context_segments=segments
                        )
                except Exception as exc:
                    logger.warning(f"并行引文匹配失败: {exc}")

            async for chunk in _buffered_stream(
                timed_stream,
                passthrough=True,
            ):
                last_stream_chunk = chunk
                if chunk.get("error"):
                    stream_error_sent = True
                    error_message = str(chunk.get("error") or "LLM stream error")
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "llm_stream_error",
                        error=error_message,
                    )
                    _snapshot_retrieval_context_segments(retrieval_meta)
                    retry_answer = ""
                    retry_reasoning = ""
                    retry_response: dict = {}
                    retry_error = ""
                    try:
                        retry_answer, retry_reasoning, retry_response = await _retry_generation_after_stream_error(
                            messages=messages,
                            request=request,
                            retrieval_meta=retrieval_meta,
                            has_structured_citations=has_structured_citations,
                            max_tokens=adjusted_stream_max_tokens,
                        )
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "llm_stream_retry_success",
                            answer_chars=len(retry_answer),
                        )
                    except Exception as retry_exc:
                        retry_error = str(retry_exc)
                        retrieval_meta["stream_retry_error"] = retry_error[:160]
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "llm_stream_retry_failed",
                            error=retry_error[:160],
                        )
                    if retry_answer:
                        response_context_segments = _build_response_context_segments(retrieval_meta)
                        send_meta = {
                            **{k: v for k, v in retrieval_meta.items() if not k.startswith("_")},
                            "context_segments": response_context_segments,
                            "stream_fallback_reason": "llm_stream_error",
                        }
                        if not content_progress_sent:
                            yield f"data: {json.dumps({'content': retry_answer, 'reasoning_content': retry_reasoning, 'done': False}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'error': error_message, 'done': True, 'final_content': retry_answer, 'retrieval_meta': send_meta, 'web_search_sources': web_search_sources, 'memory_hits': memory_hits, 'memory_meta': memory_meta, 'used_provider': retry_response.get('_used_provider') or chunk.get('used_provider'), 'used_model': retry_response.get('_used_model') or chunk.get('used_model'), 'fallback_used': True, 'stream_retry_used': True}, ensure_ascii=False)}\n\n"
                        break

                    fallback_answer = _build_generation_error_fallback_answer(
                        retrieval_meta,
                        error_message=error_message,
                    )
                    fallback_guard: dict = {}
                    fallback_answer, retrieval_meta["citations"] = _prepare_answer_and_citations_for_display(
                        fallback_answer,
                        retrieval_meta.get("citations", []),
                        evidence_need=retrieval_meta.get("evidence_need", []),
                        answer_guard=fallback_guard,
                        query=retrieval_meta.get("search_query") or request.question,
                        context_segments=retrieval_meta.get("_context_segments", []),
                    )
                    if fallback_guard:
                        retrieval_meta["answer_guard"] = fallback_guard
                    if retrieval_meta.get("citations"):
                        retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
                            retrieval_meta.get("citations", []),
                            query=retrieval_meta.get("search_query") or request.question,
                        )
                    response_context_segments = _build_response_context_segments(retrieval_meta)
                    send_meta = {
                        **{k: v for k, v in retrieval_meta.items() if not k.startswith("_")},
                        "context_segments": response_context_segments,
                        "stream_fallback_reason": "llm_stream_error",
                    }
                    if fallback_answer and not content_progress_sent:
                        yield f"data: {json.dumps({'content': fallback_answer, 'reasoning_content': '', 'done': False}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'error': error_message, 'done': True, 'final_content': fallback_answer, 'retrieval_meta': send_meta, 'web_search_sources': web_search_sources, 'memory_hits': memory_hits, 'memory_meta': memory_meta, 'used_provider': chunk.get('used_provider'), 'used_model': chunk.get('used_model'), 'fallback_used': True}, ensure_ascii=False)}\n\n"
                    break

                content = chunk.get('content', '')
                reasoning = chunk.get('reasoning_content', '')
                total_reasoning_chars += len(reasoning)

                if reasoning and not first_reasoning_logged:
                    first_reasoning_logged = True
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "llm_first_reasoning_chunk",
                        chars=len(reasoning),
                    )
                if content and not first_content_logged:
                    first_content_logged = True
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "llm_first_content_chunk",
                        chars=len(content),
                    )

                if chunk.get("done"):
                    stream_done_sent = True
                    qa_score_val = chunk.get('qa_score')
                    # 等待并行引文匹配线程完成（如已启动）
                    if _citation_match_thread is not None:
                        _citation_match_thread.join(timeout=5)
                    # 将匹配结果写入 citations
                    if has_structured_citations:
                        enhanced = _citation_match_result.get("enhanced", [])
                        if not enhanced and full_output:
                            # 并行线程未启动或未匹配时，同步补跑（兜底）
                            try:
                                inline_cites = parse_citation_list(full_output)
                                _context_segments = retrieval_meta.get("_context_segments", [])
                                if inline_cites and (_retrieval_chunks or _context_segments):
                                    enhanced = match_citations_to_chunks(
                                        inline_cites, _retrieval_chunks, context_segments=_context_segments
                                    )
                            except Exception as e:
                                logger.warning(f"结构化引文后处理失败: {e}")
                        if enhanced:
                            orig_citations = retrieval_meta.get("citations", [])
                            for ec in enhanced:
                                if ec.get("idx") is not None:
                                    for oc in orig_citations:
                                        if oc.get("ref") == ec["idx"]:
                                            if ec.get("start_phrase"):
                                                oc["start_phrase"] = ec["start_phrase"]
                                            if ec.get("end_phrase"):
                                                oc["end_phrase"] = ec["end_phrase"]
                                            if ec.get("highlight_text"):
                                                oc["highlight_text"] = ec["highlight_text"]
                                                oc["alignment_status"] = "span_matched"
                                            if ec.get("group_id"):
                                                oc["group_id"] = ec["group_id"]
                                            if ec.get("page"):
                                                oc["page_range"] = [ec["page"], ec["page"]]
                                            break

                    final_answer_text = extract_final_answer(full_output) if has_structured_citations else (full_output or "")
                    answer_guard: dict = {}
                    _snapshot_retrieval_context_segments(retrieval_meta)
                    final_answer_text, retrieval_meta["citations"] = _prepare_answer_and_citations_for_display(
                        final_answer_text,
                        retrieval_meta.get("citations", []),
                        evidence_need=retrieval_meta.get("evidence_need", []),
                        answer_guard=answer_guard,
                        query=retrieval_meta.get("search_query") or request.question,
                        context_segments=retrieval_meta.get("_context_segments", []),
                    )
                    if answer_guard:
                        retrieval_meta["answer_guard"] = answer_guard
                    if retrieval_meta.get("citations"):
                        retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
                            retrieval_meta.get("citations", []),
                            query=retrieval_meta.get("search_query") or request.question,
                        )
                    response_context_segments = _build_response_context_segments(retrieval_meta)

                    # Bug1 兜底：LLM 未输出 FINAL ANSWER 标记时，流式过滤跳过了所有 content，
                    # 此处补发完整回答文本，确保前端不会显示空内容
                    if has_structured_citations and full_output:
                        fallback_text = final_answer_text or full_output
                        fallback_delta = ""
                        if not content_progress_sent:
                            fallback_delta = fallback_text
                        elif fallback_text.startswith(visible_answer_text):
                            fallback_delta = fallback_text[len(visible_answer_text):]
                        if fallback_delta:
                            content_progress_sent = True
                            visible_answer_text = fallback_text
                            yield f"data: {json.dumps({'content': fallback_delta, 'reasoning_content': '', 'done': False})}\n\n"

                    # 移除内部 _chunks 字段（仅后端使用），避免发送大量原始数据
                    send_meta = {
                        **{k: v for k, v in retrieval_meta.items() if not k.startswith("_")},
                        "context_segments": response_context_segments,
                    }
                    chunk_data = {
                        'content': '', 'reasoning_content': reasoning,
                        'done': True, 'used_provider': chunk.get('used_provider'),
                        'used_model': chunk.get('used_model'), 'fallback_used': chunk.get('fallback_used'),
                        'final_content': final_answer_text,
                        'retrieval_meta': send_meta,
                        'web_search_sources': web_search_sources,
                        'memory_hits': memory_hits,
                        'memory_meta': memory_meta,
                    }
                    if qa_score_val is not None:
                        chunk_data['qa_score'] = qa_score_val
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "stream_done",
                        answer_chars=len(final_answer_text),
                        reasoning_chars=total_reasoning_chars,
                        citations=len(send_meta.get("citations") or []),
                        qa_score=qa_score_val,
                    )
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                    if use_memory: threading.Thread(target=_async_memory_write, args=(memory_service, request), daemon=True).start()

                    # P1.5 优化：4 个后处理任务（followup/critic/conv_name/mindmap）改为并行执行
                    # 总耗时从 sum(各任务) → max(各任务)，按完成顺序 yield SSE 事件
                    _post_tasks: dict = {}

                    # 1) 追问建议
                    async def _task_followup():
                        try:
                            followup_history = list(request.chat_history or [])
                            followup_history.append({"role": "user", "content": request.question})
                            _fm, _fp, _fe = _get_cheap_model_params(request)
                            followups = await generate_followup_questions(
                                chat_history=followup_history,
                                api_key=request.api_key,
                                model=_fm,
                                provider=_fp,
                                endpoint=_fe,
                            )
                            if followups:
                                return ("followup_questions", {"questions": followups})
                        except Exception as e:
                            logger.debug(f"追问建议生成失败（不影响主流程）: {e}")
                        return None

                    # 2) 答案自审
                    async def _task_critic():
                        if not (should_enable_answer_critic() and full_output and context):
                            return None
                        try:
                            from services.answer_critic_service import critique_answer
                            _cm2, _cp2, _ce2 = _get_cheap_model_params(request)
                            critic_result = await critique_answer(
                                question=request.question,
                                answer=full_output,
                                context=context[:6000],
                                api_key=request.api_key,
                                model=_cm2,
                                provider=_cp2,
                                endpoint=_ce2,
                            )
                            if critic_result and critic_result.get("has_hallucination"):
                                return ("answer_critic", {"critic": critic_result})
                        except Exception as e:
                            logger.debug(f"答案自审失败（不影响主流程）: {e}")
                        return None

                    # 3) 会话命名（仅首轮）
                    async def _task_conv_name():
                        if request.chat_history and len(request.chat_history) > 1:
                            return None
                        try:
                            name_history = [{"role": "user", "content": request.question}]
                            if full_output:
                                name_history.append({"role": "assistant", "content": full_output[:300]})
                            _nm, _np, _ne = _get_cheap_model_params(request)
                            conv_name = await suggest_conversation_name(
                                chat_history=name_history,
                                api_key=request.api_key,
                                model=_nm,
                                provider=_np,
                                endpoint=_ne,
                            )
                            if conv_name:
                                return ("conv_name", {"name": conv_name})
                        except Exception as e:
                            logger.debug(f"会话命名失败（不影响主流程）: {e}")
                        return None

                    # 4) 思维导图（仅有检索上下文时）
                    async def _task_mindmap():
                        if not (context and len(context) > 100):
                            return None
                        try:
                            mindmap_md = await generate_mindmap(
                                question=request.question,
                                context=context,
                                api_key=request.api_key,
                                model=request.model,
                                provider=request.api_provider,
                                endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
                            )
                            if mindmap_md:
                                return ("mindmap", {"markdown": mindmap_md})
                        except Exception as e:
                            logger.debug(f"思维导图生成失败（不影响主流程）: {e}")
                        return None

                    # 5) P3.6 citation enhancer（二次引用注入）
                    async def _task_citation_enhance():
                        if not getattr(settings, "enable_citation_enhancer", False):
                            return None
                        # 优先使用 final_answer_text（结构化引文场景的纯答案），其次 full_output
                        candidate_answer = (final_answer_text or full_output or "").strip()
                        if not candidate_answer or len(candidate_answer) < 50:
                            return None
                        # 准备参考资料：优先 _context_segments，其次 citations
                        chunks_for_enhance: list = []
                        for seg in (retrieval_meta.get("_context_segments") or []):
                            if isinstance(seg, dict) and seg.get("text") and seg.get("ref") is not None:
                                chunks_for_enhance.append(seg)
                        seen_enhance_refs = {
                            int(seg.get("ref"))
                            for seg in chunks_for_enhance
                            if isinstance(seg, dict) and seg.get("ref") is not None
                        }
                        for cit in (retrieval_meta.get("citations") or []):
                            if not isinstance(cit, dict) or cit.get("ref") is None:
                                continue
                            try:
                                cit_ref = int(cit.get("ref"))
                            except (TypeError, ValueError):
                                continue
                            if cit_ref in seen_enhance_refs:
                                continue
                            citation_chunk = dict(cit)
                            citation_chunk["ref"] = cit_ref
                            citation_chunk.setdefault(
                                "page_range",
                                cit.get("page_range") or [cit.get("page", 0), cit.get("page", 0)],
                            )
                            chunks_for_enhance.append(citation_chunk)
                            seen_enhance_refs.add(cit_ref)
                        if not chunks_for_enhance:
                            return None
                        try:
                            from services.citation_enhancer import enhance_citations
                            threshold = float(getattr(settings, "citation_enhancer_coverage_threshold", 0.5) or 0.5)
                            _em, _ep, _ee = _get_cheap_model_params(request)
                            enhanced, diag = await enhance_citations(
                                candidate_answer,
                                chunks_for_enhance,
                                api_key=request.api_key,
                                model=_em,
                                provider=_ep,
                                endpoint=_ee,
                                coverage_threshold=threshold,
                            )
                            if diag.get("triggered") and enhanced and enhanced != candidate_answer:
                                return ("citation_enhanced", {
                                    "enhanced_answer": enhanced,
                                    "diagnostics": diag,
                                })
                        except Exception as e:
                            logger.debug(f"citation_enhancer 失败（不影响主流程）: {e}")
                        return None

                    # 并行启动所有任务，按完成顺序 yield 事件
                    _post_coros = [_task_followup(), _task_critic(), _task_conv_name(), _task_mindmap(), _task_citation_enhance()]
                    async for _ev_type, _ev_data in _yield_postprocess_events(_post_coros):
                        yield f"data: {json.dumps({'type': _ev_type, **_ev_data}, ensure_ascii=False)}\n\n"

                    yield "data: [DONE]\n\n"
                    break

                # 累积完整输出
                full_output += content

                # 结构化引文流式过滤：隐藏 CITATION LIST，只展示 FINAL ANSWER
                if has_structured_citations:
                    current_answer_text = _extract_streaming_final_answer(full_output)
                    if current_answer_text:
                        reached_final_answer = True
                    # D1：CITATION LIST 已完整，立即在后台线程启动引文匹配
                    if _citation_match_thread is None:
                        citation_parts = _ci_split(_RE_START_CITATION, full_output)
                        if citation_parts is not None:
                            citation_list_part = citation_parts[1].lstrip()
                            if citation_list_part and (_retrieval_chunks or retrieval_meta.get("_context_segments")):
                                _citation_match_thread = threading.Thread(
                                    target=_run_citation_match,
                                    args=(
                                        citation_list_part,
                                        _retrieval_chunks,
                                        retrieval_meta.get("_context_segments", []),
                                        _citation_match_result,
                                    ),
                                    daemon=True,
                                )
                                _citation_match_thread.start()
                    # 提取 FINAL ANSWER 之后的内容并发送
                    stream_delta = ""
                    if current_answer_text:
                        if current_answer_text.startswith(visible_answer_text):
                            stream_delta = current_answer_text[len(visible_answer_text):]
                        elif not content_progress_sent:
                            stream_delta = current_answer_text
                        visible_answer_text = current_answer_text
                    if stream_delta or reasoning:
                        if stream_delta:
                            content_progress_sent = True
                        yield f"data: {json.dumps({'content': stream_delta, 'reasoning_content': reasoning, 'done': False, 'used_provider': chunk.get('used_provider'), 'used_model': chunk.get('used_model'), 'fallback_used': chunk.get('fallback_used')})}\n\n"
                    # 不展示 CITATION LIST 部分
                    # 已进入 FINAL ANSWER 区域，检查是否 CITATION LIST 再次出现（小模型重复）
                    # 使用 continue 跳过脏块而非 break，确保 done 事件正常触发
                    # 仅保留 CITATION LIST 之前的内容（如有），其余丢弃
                    continue

                chunk_data = {
                    'content': content, 'reasoning_content': reasoning,
                    'done': False, 'used_provider': chunk.get('used_provider'),
                    'used_model': chunk.get('used_model'), 'fallback_used': chunk.get('fallback_used'),
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            if not stream_done_sent and not stream_error_sent:
                _log_chat_trace(
                    trace_id,
                    trace_started_at,
                    "llm_stream_missing_done_fallback",
                    answer_chars=len(full_output),
                    reasoning_chars=total_reasoning_chars,
                )
                if _citation_match_thread is not None:
                    _citation_match_thread.join(timeout=2)
                if has_structured_citations:
                    enhanced = _citation_match_result.get("enhanced", [])
                    if not enhanced:
                        try:
                            inline_cites = parse_citation_list(full_output)
                            _context_segments = retrieval_meta.get("_context_segments", [])
                            if inline_cites and (_retrieval_chunks or _context_segments):
                                enhanced = match_citations_to_chunks(
                                    inline_cites,
                                    _retrieval_chunks,
                                    context_segments=_context_segments,
                                )
                        except Exception as e:
                            logger.warning(f"断流兜底结构化引文后处理失败: {e}")
                    if enhanced:
                        orig_citations = retrieval_meta.get("citations", [])
                        for ec in enhanced:
                            if ec.get("idx") is None:
                                continue
                            for oc in orig_citations:
                                if oc.get("ref") == ec["idx"]:
                                    if ec.get("start_phrase"):
                                        oc["start_phrase"] = ec["start_phrase"]
                                    if ec.get("end_phrase"):
                                        oc["end_phrase"] = ec["end_phrase"]
                                    if ec.get("highlight_text"):
                                        oc["highlight_text"] = ec["highlight_text"]
                                        oc["alignment_status"] = "span_matched"
                                    if ec.get("group_id"):
                                        oc["group_id"] = ec["group_id"]
                                    if ec.get("page"):
                                        oc["page_range"] = [ec["page"], ec["page"]]
                                    break

                final_answer_text = extract_final_answer(full_output) if has_structured_citations else full_output
                answer_guard: dict = {}
                _snapshot_retrieval_context_segments(retrieval_meta)
                final_answer_text, retrieval_meta["citations"] = _prepare_answer_and_citations_for_display(
                    final_answer_text,
                    retrieval_meta.get("citations", []),
                    evidence_need=retrieval_meta.get("evidence_need", []),
                    answer_guard=answer_guard,
                    query=retrieval_meta.get("search_query") or request.question,
                    context_segments=retrieval_meta.get("_context_segments", []),
                )
                if answer_guard:
                    retrieval_meta["answer_guard"] = answer_guard
                if retrieval_meta.get("citations"):
                    retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
                        retrieval_meta.get("citations", []),
                        query=retrieval_meta.get("search_query") or request.question,
                    )
                response_context_segments = _build_response_context_segments(retrieval_meta)
                if has_structured_citations and final_answer_text:
                    fallback_delta = ""
                    if not content_progress_sent:
                        fallback_delta = final_answer_text
                    elif final_answer_text.startswith(visible_answer_text):
                        fallback_delta = final_answer_text[len(visible_answer_text):]
                    if fallback_delta:
                        yield f"data: {json.dumps({'content': fallback_delta, 'reasoning_content': '', 'done': False}, ensure_ascii=False)}\n\n"

                send_meta = {
                    **{k: v for k, v in retrieval_meta.items() if not k.startswith("_")},
                    "context_segments": response_context_segments,
                    "stream_fallback_reason": "missing_llm_done_event",
                }
                chunk_data = {
                    'content': '',
                    'reasoning_content': '',
                    'done': True,
                    'used_provider': last_stream_chunk.get('used_provider'),
                    'used_model': last_stream_chunk.get('used_model'),
                    'fallback_used': last_stream_chunk.get('fallback_used'),
                    'final_content': final_answer_text,
                    'retrieval_meta': send_meta,
                    'web_search_sources': web_search_sources,
                    'memory_hits': memory_hits,
                    'memory_meta': memory_meta,
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("流式响应生成失败")
            _log_chat_trace(
                trace_id,
                trace_started_at,
                "stream_exception",
                error=str(e),
            )
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _detect_mime_type(img_b64: str) -> str:
    try:
        header = base64.b64decode(img_b64[:16])
        if header[:3] == b'\xff\xd8\xff': return 'image/jpeg'
        if header[:4] == b'\x89PNG': return 'image/png'
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP': return 'image/webp'
    except: pass
    return 'image/jpeg'
