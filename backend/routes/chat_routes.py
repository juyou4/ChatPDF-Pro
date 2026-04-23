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

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.chat_service import call_ai_api, call_ai_api_stream, extract_reasoning_content
from services.vector_service import vector_context
from services.selected_text_locator import locate_selected_text
from services.retrieval_agent import RetrievalAgent
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
from services.decompose_service import decompose_question
from services.mindmap_service import generate_mindmap
from services.rag_config import should_apply_numeric_table_specialization
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

logger = logging.getLogger(__name__)

router = APIRouter()
_MIN_SELECTED_TEXT_FALLBACK_CITATION_CHARS = 30
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
_AGENT_RETRIEVAL_QUERY_TYPES = {"overview"}
_AGENT_RETRIEVAL_EVIDENCE_NEEDS = {
    "section_explanation",
    "comparison_multi_aspect",
    "reference_meta",
}
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
    print(line, flush=True)
    logger.warning(line)


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


# 上下文构建器实例，用于生成引文指示提示词
_context_builder = ContextBuilder()

# 查询改写器实例
_query_rewriter = QueryRewriter()


def _get_cheap_model_params(request) -> tuple:
    """获取辅助模型参数（双模型策略）

    如果配置了 cheap_model，返回 (cheap_model, cheap_provider, cheap_endpoint)，
    否则返回请求中的主模型参数。

    Returns:
        (model, provider, endpoint) 三元组
    """
    cheap_model = settings.cheap_model
    cheap_provider = settings.cheap_model_provider
    if cheap_model and cheap_provider:
        endpoint = _get_provider_endpoint(cheap_provider, request.api_host or "")
        return cheap_model, cheap_provider, endpoint
    return request.model, request.api_provider, _get_provider_endpoint(request.api_provider, request.api_host or "")


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
                    return [c for c in chunks if isinstance(c, str) and c.strip()]
            elif isinstance(data, list):
                return [c for c in data if isinstance(c, str) and c.strip()]
        except Exception as exc:
            logger.warning(f"[AgentDoc] 加载 chunks 失败: {chunks_path} -> {exc}")

    return _split_context_paragraphs(full_text or "") or ([full_text.strip()] if full_text.strip() else [])


def _load_doc_semantic_groups_for_agent(doc_id: str) -> list[dict]:
    """从落盘的 semantic_groups 中加载意群数据。"""
    candidate_dirs = [
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
                return [g for g in groups if isinstance(g, dict)]
        except Exception as exc:
            logger.warning(f"[AgentDoc] 加载意群失败: {group_path} -> {exc}")
    return []


def _build_agent_doc_context(doc_id: str, doc: dict, vector_store_dir: str, api_key: str = "") -> DocContext:
    data = doc.get("data", {}) or {}
    full_text = data.get("full_text", "") or ""
    pages = data.get("pages", []) or []
    chunks = _load_doc_chunks_for_agent(doc_id, vector_store_dir, full_text)
    semantic_groups = _load_doc_semantic_groups_for_agent(doc_id)
    return DocContext(
        doc_id=doc_id,
        full_text=full_text,
        chunks=chunks,
        pages=pages,
        semantic_groups=semantic_groups,
        vector_store_dir=vector_store_dir,
        api_key=api_key or "",
    )


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
        not settings.enable_llm_query_rewrite
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


_NUMERIC_TABLE_QUERY_TABLE_RE = re.compile(r"\btable\s*\d+\b", re.IGNORECASE)
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
    if "accuracy" in sample or re.search(r"\bacc\b", sample):
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
    for value in (hints or {}).get("tables", []) or []:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        if normalized:
            targets.add(normalized)
    return targets


def _is_numeric_table_cost_query(query: str = "") -> bool:
    sample = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    return bool(sample) and any(token in sample for token in _NUMERIC_TABLE_COST_QUERY_HINTS)


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


def _build_numeric_table_citation_support_text(citation: Optional[dict]) -> str:
    if not isinstance(citation, dict):
        return ""
    parts = [
        *_collect_citation_table_evidence_texts(citation),
        citation.get("context_segment_text", ""),
        citation.get("source_text", ""),
        citation.get("display_text", ""),
        citation.get("highlight_text", ""),
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


def _collect_citation_table_evidence_texts(citation: dict) -> list[str]:
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
        for field in ("context_segment_text", "source_text", "_full_text"):
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
        if not focused_row_text:
            for field in ("display_text", "highlight_text"):
                candidate = _pick_first_normalized(field)
                if _looks_like_structured_table_text(candidate):
                    focused_row_text = candidate
                    break
    if not focused_row_text and chunk_type in {"table_row", "table_cell"}:
        focused_row_text = _pick_first_normalized("display_text", "highlight_text")

    row_text = focused_row_text or exact_row_text
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


def _build_citation_context_text(citation: dict) -> str:
    if not isinstance(citation, dict):
        return ""

    table_evidence = _collect_citation_table_evidence_texts(citation)
    if table_evidence:
        return "\n".join(table_evidence)

    for field in ("context_segment_text", "source_text", "_full_text", "display_text", "highlight_text"):
        normalized = re.sub(r"\s+", " ", str(citation.get(field, "") or "")).strip()
        if normalized:
            return normalized
    return ""


def _build_context_segments_from_citations(citations: list[dict]) -> list[dict]:
    segments = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        try:
            ref = int(c.get("ref"))
        except (TypeError, ValueError):
            continue
        text = _build_citation_context_text(c)
        if not text:
            continue
        segments.append({
            "ref": ref,
            "text": text,
            "page_range": c.get("page_range") or [],
            "group_id": c.get("group_id", ""),
        })
    return segments


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


def _build_response_context_segments(retrieval_meta: dict) -> list[dict]:
    if not isinstance(retrieval_meta, dict):
        return []

    citation_segments = _build_context_segments_from_citations(retrieval_meta.get("citations", []))
    existing_segments = []
    for seg in retrieval_meta.get("_context_segments") or []:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        existing_segments.append(
            {
                "ref": seg.get("ref"),
                "text": text,
                "page_range": seg.get("page_range") or [],
                "group_id": seg.get("group_id", ""),
            }
        )

    evidence_need = {
        str(item).strip()
        for item in (retrieval_meta.get("evidence_need") or [])
        if str(item).strip()
    }
    if "numeric_table" in evidence_need and citation_segments:
        comparator_segments = _build_numeric_table_comparator_context_segments(retrieval_meta)
        return comparator_segments or citation_segments
    return existing_segments or citation_segments



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
    strict_gate = (
        bool(query)
        and any(_has_numeric_table_exact_row_support(citation) for citation in normalized_aligned)
        and _should_apply_numeric_table_strict_gate(query, hints)
    )
    target_tables = _extract_numeric_table_target_tables(query, hints)
    target_columns = _extract_numeric_table_target_columns(query, hints)

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
        if not has_structured_support and not has_cost_anchor:
            continue

        support_score = _calc_citation_support_score(core_answer, citation)
        table_id = str(citation.get("table_id") or "").strip().lower()
        group_id = str(citation.get("group_id") or "").strip().lower()
        same_bundle = bool(
            (table_id and table_id in selected_table_ids)
            or (group_id and group_id in selected_group_ids)
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

    final: list[str] = []
    for para in paragraphs:
        if len(para) <= 500:
            final.append(para)
        else:
            sents = _re.split(r'(?<=[.。!！?？;；])\s*', para)
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

    # 页码反查
    def _locate_page(para_text: str) -> int:
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

    def _sentence_windows(text: str) -> list[str]:
        sents = [s.strip() for s in _re.split(r'(?<=[.。!！?？;；])\s*', text) if s.strip()]
        if len(sents) <= 1:
            return [text.strip()]
        windows: list[str] = []
        current: list[str] = []
        current_len = 0
        for sent in sents:
            sent_len = len(sent)
            if current and current_len + sent_len > 240:
                windows.append(" ".join(current).strip())
                overlap = current[-1:] if len(current[-1]) <= 120 else []
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

    candidates: list[tuple[float, int, int, str, str, set[str], int]] = []
    for pi, para in enumerate(paragraphs):
        for wi, window in enumerate(_sentence_windows(para)):
            tokens = _tokenize(window)
            if len(tokens) < 8:
                continue
            token_set = set(tokens)
            overlap_terms = token_set & set(query_terms)
            overlap = len(overlap_terms)
            n_tokens = len(tokens)
            unique_ratio = len(token_set) / max(n_tokens, 1)
            density = overlap / max(len(query_terms), 1) if query_terms else 0.0
            richness = unique_ratio * min(n_tokens / 24.0, 1.0)
            number_bonus = 0.20 if _re.search(r'\d', window) else 0.0
            keyword_bonus = 0.15 if _re.search(r'(dataset|results?|experiment|method|abstract|introduction|conclusion|贡献|实验|方法|结果|数据集)', window.lower()) else 0.0
            score = overlap * 2.8 + density * 1.8 + richness + number_bonus + keyword_bonus
            snippet = _context_builder._extract_relevant_snippet(window, query, max_len=140)
            page_num = _locate_page(window)
            candidates.append((score, pi, wi, window, snippet, token_set, page_num))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected: list[tuple[int, int, str, str, int]] = []
    selected_token_sets: list[set[str]] = []
    page_counts: dict[int, int] = {}
    for score, pi, wi, window, snippet, token_set, page_num in candidates:
        if page_counts.get(page_num, 0) >= 2 and len(selected) < max_citations - 1:
            continue
        if any(len(token_set & prev) / max(1, len(token_set | prev)) >= 0.55 for prev in selected_token_sets):
            continue
        selected.append((pi, wi, window, snippet, page_num))
        selected_token_sets.append(token_set)
        page_counts[page_num] = page_counts.get(page_num, 0) + 1
        if len(selected) >= max_citations:
            break

    if not selected:
        for pi, para in enumerate(paragraphs[:max_citations]):
            page_num = _locate_page(para)
            snippet = _context_builder._extract_relevant_snippet(para, query, max_len=140)
            selected.append((pi, 0, para, snippet, page_num))

    # 按原始顺序排列，使 context 保持逻辑连贯
    selected.sort(key=lambda x: (x[0], x[1]))

    # 构建编号段落 context + citations
    # 将所有段落加入 context（让 LLM 看到完整文档），但仅对 selected 生成 citation
    selected_set = {(pi, wi) for pi, wi, *_ in selected}
    all_formatted: list[str] = []
    for pi, para in enumerate(paragraphs):
        all_formatted.append(para)

    citations: list[dict] = []
    for ref_idx, (pi, wi, window, snippet, page_num) in enumerate(selected, 1):
        highlight = snippet or (window[:140] if len(window) > 140 else window)
        citations.append({
            "ref": ref_idx,
            "evidence_id": f"para-{pi + 1}-seg-{wi + 1}:{ref_idx}",
            "group_id": f"para-{pi + 1}-seg-{wi + 1}",
            "page_range": [page_num, page_num],
            "source_text": window,
            "display_text": window,
            "highlight_text": highlight,
            "_full_text": window,
            "alignment_status": "fallback_window_only",
            "retrieval_type": "fallback",
        })

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
    selected_text: Optional[str],
    query_type: str,
    evidence_need: Optional[list[str]] = None,
) -> dict:
    """返回 retrieval_agent 触发决策及其原因，便于诊断。"""
    normalized_query_type = str(query_type or "").strip().lower()
    normalized_needs = [
        str(item).strip()
        for item in (evidence_need or [])
        if str(item).strip()
    ]
    matched_needs = [
        need for need in normalized_needs
        if need in _AGENT_RETRIEVAL_EVIDENCE_NEEDS
    ]
    matched_query_type = (
        normalized_query_type
        if normalized_query_type in _AGENT_RETRIEVAL_QUERY_TYPES
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
        }

    if bool(selected_text):
        return {
            "enabled": False,
            "reason": "selected_text_present",
            "query_type": normalized_query_type,
            "evidence_need": normalized_needs,
            "matched_query_type": matched_query_type,
            "matched_evidence_need": matched_needs,
            "selected_text_present": True,
        }

    enabled = bool(matched_query_type or matched_needs)
    if matched_query_type:
        reason = "matched_query_type"
    elif matched_needs:
        reason = "matched_evidence_need"
    else:
        reason = "route_not_matched"

    return {
        "enabled": enabled,
        "reason": reason,
        "query_type": normalized_query_type,
        "evidence_need": normalized_needs,
        "matched_query_type": matched_query_type,
        "matched_evidence_need": matched_needs,
        "selected_text_present": False,
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


def _should_enable_agent_retrieval(
    *,
    enable_agent_retrieval: bool,
    selected_text: Optional[str],
    query_type: str,
    evidence_need: Optional[list[str]] = None,
) -> bool:
    """仅对高价值题型启用 retrieval_agent，避免全局放大延迟。"""
    gate = _build_agent_retrieval_gate(
        enable_agent_retrieval=enable_agent_retrieval,
        selected_text=selected_text,
        query_type=query_type,
        evidence_need=evidence_need,
    )
    return bool(gate.get("enabled"))


def _generate_page_level_citations(pages: list[dict], context: str, query: str = "", max_citations: int = 8) -> list[dict]:
    """兼容旧调用：仅返回 citations 列表。"""
    _, citations = _build_numbered_context_and_citations(pages, context, query=query, max_citations=max_citations)
    return citations


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

    return _INLINE_CITATION_PATTERN.sub(_replace, answer)


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
) -> tuple[str, list[dict]]:
    normalized_citations = _normalize_citation_records(citations)
    repaired_answer = _repair_bad_citation_formats(answer, normalized_citations)
    if normalized_citations and repaired_answer:
        refs_in_answer = _extract_inline_citation_refs(repaired_answer)
        valid_ref_set = {int(c["ref"]) for c in normalized_citations}
        should_optimize_inline = (
            len(set(refs_in_answer)) <= 1
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
                "numeric_table" in {
                    str(item).strip()
                    for item in (evidence_need or [])
                    if str(item).strip()
                }
                and _is_numeric_table_cost_query(query)
                and any(
                    _has_numeric_table_cost_anchor(_build_numeric_table_citation_support_text(citation))
                    for citation in normalized_citations
                )
            )
            if not numeric_cost_recovery:
                return repaired_answer, []

    aligned = _align_citations_with_answer(repaired_answer, normalized_citations)
    if "numeric_table" in {
        str(item).strip()
        for item in (evidence_need or [])
        if str(item).strip()
    }:
        aligned = _supplement_numeric_table_citations(
            repaired_answer,
            aligned,
            normalized_citations,
            query=query,
        )
    if not aligned:
        return repaired_answer, []

    refs_in_answer = _extract_inline_citation_refs(repaired_answer)
    citation_map = {int(c["ref"]): c for c in aligned}
    ordered_source_refs = refs_in_answer or [int(c["ref"]) for c in aligned]
    if "numeric_table" in {
        str(item).strip()
        for item in (evidence_need or [])
        if str(item).strip()
    }:
        ordered_source_refs.extend(
            int(c["ref"])
            for c in aligned
            if int(c["ref"]) not in ordered_source_refs
        )

    source_to_display: dict[int, int] = {}
    projected = []
    for source_ref in ordered_source_refs:
        if source_ref in source_to_display or source_ref not in citation_map:
            continue
        display_ref = len(source_to_display) + 1
        source_to_display[source_ref] = display_ref
        item = citation_map[source_ref].copy()
        item["source_ref"] = source_ref
        item["display_ref"] = display_ref
        item["ref"] = display_ref
        projected.append(item)

    if not projected:
        return repaired_answer, []

    rewritten_answer = (
        _rewrite_inline_citation_refs(repaired_answer, source_to_display)
        if refs_in_answer else repaired_answer
    )
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


def _build_answer_style_instruction(answer_detail: str) -> str:
    """根据回答详细度生成提示词指令。"""
    detail = _normalize_answer_detail(answer_detail)
    if detail == "concise":
        return "回答风格：简洁模式。优先给出结论，控制篇幅，避免冗长展开。"
    if detail == "detailed":
        return (
            "回答风格：详细模式。请严格遵循以下要求：\n"
            "- 使用 Markdown 标题（##）和小标题（###）对回答进行结构化分段\n"
            "- 从多个角度展开分析：背景介绍→核心内容→依据与推理→结论→局限性或注意事项\n"
            "- 至少覆盖 3-5 个要点，每个要点用完整段落充分展开，严禁一句话带过\n"
            "- 直接引用文档原文作为论据佐证，不要仅概括，需给出具体内容\n"
            "- 涉及数据、公式、表格时必须完整展示，不可省略或以\u201c如表所示\u201d代替\n"
            "- 目标回答长度不少于 600 字，复杂问题应达到 1000 字以上\n"
            "- 绝对不得因\u201c篇幅限制\u201d或\u201c简洁起见\u201d而截断或省略任何重要信息\n"
            "- 回答结尾附上简要总结段落"
        )
    return (
        "回答风格：标准模式。结构清晰，使用分点或分段组织回答，"
        "覆盖所有关键点并适度展开说明，引用文档原文佐证重要论述。"
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
        "- 如果检索上下文里出现多个表号或混合表块，优先选择同时包含问题中方法名和列名的那张表；无法唯一确定时不要猜测"
    )


_CITATION_TOKEN_OVERHEAD = 1024  # 结构化引文（CITATION LIST）输出的预估 token 开销
_DETAILED_MIN_TOKENS = 8192     # 详细模式下 max_tokens 的最低保证值
_STANDARD_DEFAULT_TOKENS = 4096 # 标准模式下 max_tokens 未设置时的默认值


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


@router.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    if not hasattr(router, "documents_store"):
        raise HTTPException(status_code=500, detail="文档存储未初始化")
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
        print(f"[Chat] 📸 截图模式：处理 {len(image_list)} 张图")
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
                retrieval_meta = context_result.get("retrieval_meta", {})
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
        elif _should_use_fast_overview_context(
            query_type,
            enable_vector_search=request.enable_vector_search,
            selected_text=request.selected_text,
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
            retrieval_meta = context_result.get("retrieval_meta", {})
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
                retrieval_meta.get("citations", [])
            )
        retrieval_meta["query_type"] = retrieval_meta.get("query_type") or query_type
        retrieval_meta["evidence_need"] = retrieval_meta.get("evidence_need") or evidence_need
        retrieval_meta["search_query"] = search_query

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
        answer, retrieval_meta["citations"] = _prepare_answer_and_citations_for_display(
            answer,
            retrieval_meta.get("citations", []),
            evidence_need=retrieval_meta.get("evidence_need", []),
            answer_guard=answer_guard,
            query=retrieval_meta.get("search_query") or request.question,
        )
        if answer_guard:
            retrieval_meta["answer_guard"] = answer_guard
        if retrieval_meta.get("citations"):
            retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
                retrieval_meta.get("citations", [])
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
                memory_context = _retrieve_memory_context(
                    request.question, api_key=request.api_key, doc_id=request.doc_id
                )
                raw_memories = _retrieve_raw_memories(
                    request.question, api_key=request.api_key, doc_id=request.doc_id, chat_history=request.chat_history
                )

            image_list = (request.image_base64_list or [])
            if request.image_base64 and request.image_base64 not in image_list:
                image_list = [request.image_base64] + image_list
            image_list = [img for img in image_list if img]

            if image_list:
                print(f"[Chat Stream] 📸 截图模式：处理 {len(image_list)} 张图")
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
                _log_chat_trace(
                    trace_id,
                    trace_started_at,
                    "retrieval_strategy",
                    query_type=query_type,
                    top_k=dynamic_top_k,
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
                        retrieval_meta = context_result.get("retrieval_meta", {})
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
                    yield _sse_json({
                        'type': 'retrieval_progress',
                        'phase': 'agent_mode',
                        'message': '正在启动多轮检索代理...',
                    })
                    agent_api_key = request.api_key or ""
                    agent_model, agent_provider, agent_endpoint = _get_cheap_model_params(request)
                    agent_doc_ctx = _build_agent_doc_context(
                        request.doc_id,
                        doc,
                        getattr(router, "vector_store_dir", ""),
                        api_key=request.embedding_api_key or request.api_key or "",
                    )
                    agent = RetrievalAgent(
                        api_key=agent_api_key,
                        model=agent_model,
                        provider=agent_provider,
                        endpoint=agent_endpoint,
                        max_rounds=max(1, min(int(getattr(settings, "agent_max_rounds", 5) or 5), 10)),
                        temperature=float(getattr(settings, "agent_planner_temperature", 0.3) or 0.3),
                    )
                    agent_result: dict = {}
                    try:
                        async for agent_event in agent.run(
                            question=search_query or request.question or "",
                            doc_ctx=agent_doc_ctx,
                            doc_name=doc.get("filename", ""),
                        ):
                            event_type = agent_event.get("type")
                            if event_type == "retrieval_progress":
                                yield _sse_json(agent_event)
                                phase = agent_event.get("phase", "")
                                message = agent_event.get("message", "")
                                if phase in {"start", "round_start", "planning", "executing", "tool_result", "complete"}:
                                    _log_chat_trace(
                                        trace_id,
                                        trace_started_at,
                                        f"agent_{phase}",
                                        message=_preview_for_log(message, 120),
                                    )
                            elif event_type == "retrieval_complete":
                                agent_result = agent_event
                    except Exception as e:
                        logger.warning(f"[Agent] 多轮检索失败，降级为全文编号上下文: {e}")
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "retrieval_agent_exception",
                            error=str(e),
                        )
                        agent_result = {}

                    agent_context = agent_result.get("context", "") if isinstance(agent_result, dict) else ""
                    agent_detail = agent_result.get("detail", []) if isinstance(agent_result, dict) else []
                    if isinstance(agent_result, dict):
                        if agent_result.get("search_history"):
                            retrieval_meta["agent_search_history"] = agent_result.get("search_history")
                        if agent_result.get("task_status"):
                            retrieval_meta["task_status"] = agent_result.get("task_status")
                    retrieval_meta["agent_detail"] = agent_detail
                    retrieval_meta["agent_mode"] = True
                    retrieval_meta["agent_gate"] = _annotate_agent_gate(
                        retrieval_meta.get("agent_gate", agent_gate),
                        use_agent=use_agent,
                        agent_mode=True,
                        search_query_passthrough=True,
                    )
                    retrieval_meta["query_type"] = query_type

                    if agent_context:
                        numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                            doc.get("data", {}).get("pages", []),
                            agent_context,
                            query=search_query or request.question or "",
                        )
                        context = numbered_ctx or agent_context
                        agent_citations = fb_cits or _generate_page_level_citations(
                            doc.get("data", {}).get("pages", []),
                            agent_context,
                            query=search_query or request.question or "",
                        )
                        retrieval_meta["citations"] = agent_citations
                        retrieval_meta["agent_context_chars"] = len(agent_context)
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "retrieval_agent_done",
                            context_chars=len(context),
                            citations=len(agent_citations or []),
                            detail=len(agent_detail or []),
                        )
                    else:
                        fallback_text = (doc.get("data", {}) or {}).get("full_text", "")[:30000]
                        numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                            doc.get("data", {}).get("pages", []),
                            fallback_text,
                            query=search_query or request.question or "",
                        )
                        context = numbered_ctx or fallback_text
                        retrieval_meta["citations"] = fb_cits
                        retrieval_meta["agent_fallback"] = True
                        _log_chat_trace(
                            trace_id,
                            trace_started_at,
                            "retrieval_agent_fallback",
                            context_chars=len(context),
                            citations=len(fb_cits or []),
                        )
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

                    # 复杂问题分解：对包含"比较""区别"等关键词的查询，拆分为子问题分别检索
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
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[Decompose] 问题分解超时(5s)，跳过分解")
                        sub_questions = []
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "retrieval_analysis_done",
                        sub_questions=len(sub_questions or []),
                    )

                    queries_to_search = [search_query] + sub_questions if sub_questions else [search_query]
                    all_relevant_texts = []

                    for sq in queries_to_search:
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
                        # 使用第一个（主查询）的 retrieval_meta
                        if sq == search_query:
                            retrieval_meta = cr.get("retrieval_meta", {})

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
                if (settings.enable_graphrag or request.enable_graphrag) and hasattr(router, "_graphrag_instances") and request.doc_id in router._graphrag_instances:
                    try:
                        graphrag_inst = router._graphrag_instances[request.doc_id]
                        graphrag_context = await graphrag_inst.aquery_context(search_query)
                        if graphrag_context:
                            context += f"\n\n## 知识图谱关联信息\n{graphrag_context}"
                            logger.debug(f"[Chat] GraphRAG 上下文已融合，长度={len(graphrag_context)}")
                    except Exception as e:
                        logger.warning(f"[Chat] GraphRAG 上下文获取失败: {e}")

                if retrieval_meta.get("citations") and not retrieval_meta.get("_context_segments"):
                    retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
                        retrieval_meta.get("citations", [])
                    )
                retrieval_meta["query_type"] = retrieval_meta.get("query_type") or query_type
                retrieval_meta["evidence_need"] = retrieval_meta.get("evidence_need") or evidence_need
                retrieval_meta["search_query"] = search_query

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
                logger.info(f"[CITATION DEBUG] enable_vector_search={request.enable_vector_search}, citations_count={len(citations)}, has_structured={has_structured_citations}, compact={_should_use_compact_citation_prompt(citations)}")
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

            if use_agent:
                _log_chat_trace(
                    trace_id,
                    trace_started_at,
                    "retrieval_agent_mode",
                    top_k=dynamic_top_k,
                )
                agent_api_key = request.api_key or ""
                agent_model, agent_provider, agent_endpoint = _get_cheap_model_params(request)
                agent_doc_ctx = _build_agent_doc_context(
                    request.doc_id,
                    doc,
                    getattr(router, "vector_store_dir", ""),
                    api_key=request.embedding_api_key or request.api_key or "",
                )
                agent = RetrievalAgent(
                    api_key=agent_api_key,
                    model=agent_model,
                    provider=agent_provider,
                    endpoint=agent_endpoint,
                    max_rounds=max(1, min(int(getattr(settings, "agent_max_rounds", 5) or 5), 10)),
                    temperature=float(getattr(settings, "agent_planner_temperature", 0.3) or 0.3),
                )
                agent_result: dict = {}
                try:
                    async for agent_event in agent.run(
                        question=search_query or request.question or "",
                        doc_ctx=agent_doc_ctx,
                        doc_name=doc.get("filename", ""),
                    ):
                        event_type = agent_event.get("type")
                        if event_type == "retrieval_complete":
                            agent_result = agent_event
                        elif event_type == "retrieval_progress":
                            phase = agent_event.get("phase", "")
                            if phase in {"start", "round_start", "planning", "executing", "tool_result", "complete"}:
                                _log_chat_trace(
                                    trace_id,
                                    trace_started_at,
                                    f"agent_{phase}",
                                    message=_preview_for_log(agent_event.get("message", ""), 120),
                                )
                except Exception as e:
                    logger.warning(f"[Agent] 多轮检索失败，降级为全文编号上下文: {e}")
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "retrieval_agent_exception",
                        error=str(e),
                    )
                    agent_result = {}

                agent_context = agent_result.get("context", "") if isinstance(agent_result, dict) else ""
                agent_detail = agent_result.get("detail", []) if isinstance(agent_result, dict) else []
                if isinstance(agent_result, dict):
                    if agent_result.get("search_history"):
                        retrieval_meta["agent_search_history"] = agent_result.get("search_history")
                    if agent_result.get("task_status"):
                        retrieval_meta["task_status"] = agent_result.get("task_status")
                retrieval_meta["agent_detail"] = agent_detail
                retrieval_meta["agent_mode"] = True
                retrieval_meta["query_type"] = query_type

                if agent_context:
                    numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                        doc.get("data", {}).get("pages", []),
                        agent_context,
                        query=search_query or request.question or "",
                    )
                    context = numbered_ctx or agent_context
                    agent_citations = fb_cits or _generate_page_level_citations(
                        doc.get("data", {}).get("pages", []),
                        agent_context,
                        query=search_query or request.question or "",
                    )
                    retrieval_meta["citations"] = agent_citations
                    retrieval_meta["agent_context_chars"] = len(agent_context)
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "retrieval_agent_done",
                        context_chars=len(context),
                        citations=len(agent_citations or []),
                        detail=len(agent_detail or []),
                    )
                else:
                    fallback_text = (doc.get("data", {}) or {}).get("full_text", "")[:30000]
                    numbered_ctx, fb_cits = _build_numbered_context_and_citations(
                        doc.get("data", {}).get("pages", []),
                        fallback_text,
                        query=search_query or request.question or "",
                    )
                    context = numbered_ctx or fallback_text
                    retrieval_meta["citations"] = fb_cits
                    retrieval_meta["agent_fallback"] = True
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "retrieval_agent_fallback",
                        context_chars=len(context),
                        citations=len(fb_cits or []),
                    )

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
            # 累积完整输出，用于结构化引文解析
            full_output = ""
            reached_final_answer = False
            visible_answer_text = ""
            content_progress_sent = False
            qa_score_val = None
            first_reasoning_logged = False
            first_content_logged = False
            total_reasoning_chars = 0
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
                raw_stream,
                passthrough=True,
            ):
                if chunk.get("error"):
                    _log_chat_trace(
                        trace_id,
                        trace_started_at,
                        "llm_stream_error",
                        error=chunk.get("error"),
                    )
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
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
                    final_answer_text, retrieval_meta["citations"] = _prepare_answer_and_citations_for_display(
                        final_answer_text,
                        retrieval_meta.get("citations", []),
                        evidence_need=retrieval_meta.get("evidence_need", []),
                        answer_guard=answer_guard,
                        query=retrieval_meta.get("search_query") or request.question,
                    )
                    if answer_guard:
                        retrieval_meta["answer_guard"] = answer_guard
                    if retrieval_meta.get("citations"):
                        retrieval_meta["_context_segments"] = _build_context_segments_from_citations(
                            retrieval_meta.get("citations", [])
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
                    # 异步生成追问建议
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
                            yield f"data: {json.dumps({'type': 'followup_questions', 'questions': followups}, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        logger.debug(f"追问建议生成失败（不影响主流程）: {e}")
                    # 答案自审（检测幻觉）
                    if settings.enable_answer_critic and full_output and context:
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
                                yield f"data: {json.dumps({'type': 'answer_critic', 'critic': critic_result}, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            logger.debug(f"答案自审失败（不影响主流程）: {e}")
                    # 首轮对话自动命名
                    if not request.chat_history or len(request.chat_history) <= 1:
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
                                yield f"data: {json.dumps({'type': 'conv_name', 'name': conv_name}, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            logger.debug(f"会话命名失败（不影响主流程）: {e}")
                    # 思维导图生成（仅有检索上下文时）
                    if context and len(context) > 100:
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
                                yield f"data: {json.dumps({'type': 'mindmap', 'markdown': mindmap_md}, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            logger.debug(f"思维导图生成失败（不影响主流程）: {e}")
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
        except Exception as e:
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
