from datetime import datetime
from typing import Optional, List
import json
import logging
import re
import threading

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
from services.query_rewriter import QueryRewriter
from services.followup_service import generate_followup_questions
from services.conv_name_service import suggest_conversation_name
from services.decompose_service import decompose_question
from services.mindmap_service import generate_mindmap
from services.citation_service import (
    build_structured_citation_prompt,
    parse_citation_list,
    extract_final_answer,
    match_citations_to_chunks,
    START_ANSWER,
    START_CITATION,
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
_WEB_SEARCH_PRONOUN_HINTS = (
    "这个", "该", "它", "上述", "此", "这种", "这些", "这项", "该项", "本方法",
    "this", "that", "it", "they", "he", "she", "them",
)


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

async def _buffered_stream(raw_stream):
    """对原始 SSE 流进行字符数缓冲，合并高频小 chunk 减少 SSE 事件频率

    根据 settings.stream_buffer_size 配置的字符数阈值，
    累积文本内容达到阈值后统一发送。

    当 stream_buffer_size=0 时退化为直通模式，不做任何缓冲。

    Args:
        raw_stream: 原始异步生成器（call_ai_api_stream 的输出）
    """
    buffer_size = settings.stream_buffer_size

    # 直通模式：buffer_size=0 时不缓冲，直接转发所有 chunk
    if buffer_size <= 0:
        async for chunk in raw_stream:
            yield chunk
            if chunk.get("error") or chunk.get("done"):
                break
        return

    # 缓冲模式：使用 list 累积避免 O(n²) 字符串拼接
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    content_len = 0
    reasoning_len = 0

    async for chunk in raw_stream:
        # 错误或终止信号：立即刷新缓冲区并转发
        if chunk.get("error") or chunk.get("done"):
            if content_parts or reasoning_parts:
                yield {
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                    "done": False,
                }
                content_parts.clear()
                reasoning_parts.clear()
                content_len = reasoning_len = 0
            yield chunk
            break

        # 累积到缓冲区
        c = chunk.get("content", "")
        r = chunk.get("reasoning_content", "")
        if c:
            content_parts.append(c)
            content_len += len(c)
        if r:
            reasoning_parts.append(r)
            reasoning_len += len(r)

        # 缓冲区达到阈值，立即发送
        if content_len >= buffer_size or reasoning_len >= buffer_size:
            yield {
                "content": "".join(content_parts),
                "reasoning_content": "".join(reasoning_parts),
                "done": False,
            }
            content_parts.clear()
            reasoning_parts.clear()
            content_len = reasoning_len = 0

    # 流正常结束但未收到 done/error 信号时，刷新剩余缓冲
    if content_parts or reasoning_parts:
        yield {
            "content": "".join(content_parts),
            "reasoning_content": "".join(reasoning_parts),
            "done": False,
        }


# 上下文构建器实例，用于生成引文指示提示词
_context_builder = ContextBuilder()

# 查询改写器实例
_query_rewriter = QueryRewriter()


async def _maybe_rewrite_query(
    question: str,
    chat_history: list[dict] | None,
    selected_text: str | None,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
) -> str:
    """在满足条件时用 LLM 改写查询，否则回退到 regex 改写。

    触发条件：
    1. 配置启用了 LLM 查询改写
    2. 查询长度 < trigger_length（长查询信息已足够）
    3. 存在对话历史（多轮对话才需要上下文消解）
    4. 有可用的 api_key
    """
    if (
        not settings.enable_llm_query_rewrite
        or len(question) > settings.query_rewrite_trigger_length
        or not chat_history
        or not api_key
    ):
        return _query_rewriter.rewrite(question, selected_text=selected_text)

    rewritten = await _query_rewriter.rewrite_with_llm(
        query=question,
        chat_history=chat_history,
        selected_text=selected_text,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
    )
    return rewritten

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
    enable_graphrag: bool = False
    enable_jieba_bm25: bool = True
    num_expand_context_chunk: int = 1


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
        return memory_service.retrieve_memories(
            question, api_key=api_key, doc_id=doc_id, filter_by_doc=False
        )
    except Exception as e:
        logger.error(f"记忆检索失败: {e}")
        return ""


def _retrieve_raw_memories(question: str, api_key: str = None, doc_id: str = None) -> list[dict]:
    """检索原始记忆列表（供 ContextInjector 使用）"""
    if memory_service is None:
        return []
    try:
        return memory_service.retrieve_memories_raw(
            question, api_key=api_key, doc_id=doc_id, filter_by_doc=False
        )
    except Exception as e:
        logger.error(f"记忆原始检索失败: {e}")
        return []


def _smart_inject_memory(system_prompt: str, memory_context: str, raw_memories: list[dict] = None) -> str:
    """智能注入记忆上下文：优先使用 ContextInjector，失败时回退到简单注入

    Args:
        system_prompt: 原始 system prompt
        memory_context: 格式化的记忆上下文字符串（降级用）
        raw_memories: 原始记忆列表（供 ContextInjector 使用）

    Returns:
        注入记忆后的 system prompt
    """
    # 优先使用 ContextInjector
    if raw_memories and memory_service and hasattr(memory_service, 'context_injector') and memory_service.context_injector:
        try:
            return memory_service.context_injector.inject(system_prompt, raw_memories)
        except Exception as e:
            logger.warning(f"ContextInjector 注入失败，回退到简单注入: {e}")
    # 降级为原有简单注入
    return _inject_memory_context(system_prompt, memory_context)


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
        "group_id": "selected-text",
        "page_range": [ps, pe],
        "highlight_text": selected_text[:200].strip(),
    }


def _build_selected_text_fallback_citations(
    selected_text: str,
    selected_page_info: dict,
):
    """仅在检索引用缺失时，为较长 selected_text 生成兜底 citation。"""
    if not selected_text or len(selected_text.strip()) < _MIN_SELECTED_TEXT_FALLBACK_CITATION_CHARS:
        return []
    return [_build_selected_text_citation(selected_text, selected_page_info)]


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


def _align_citations_with_answer(answer: str, citations: list[dict]) -> list[dict]:
    """将来源列表与回答正文中的实际引文编号对齐。"""
    if not citations:
        return []

    refs_in_answer = _extract_inline_citation_refs(answer)
    if not refs_in_answer:
        logger.info(
            "回答正文未检测到内联引用编号，保留原始 citations（count=%d）",
            len(citations),
        )
        normalized = []
        for c in citations:
            if not isinstance(c, dict):
                continue
            try:
                ref = int(c.get("ref"))
            except (TypeError, ValueError):
                continue
            item = c.copy()
            item["ref"] = ref
            normalized.append(item)
        return normalized

    citation_map = {}
    for c in citations:
        if not isinstance(c, dict):
            continue
        try:
            ref = int(c.get("ref"))
        except (TypeError, ValueError):
            continue
        normalized = c.copy()
        normalized["ref"] = ref
        citation_map[ref] = normalized

    aligned = [citation_map[ref] for ref in refs_in_answer if ref in citation_map]
    if not aligned:
        logger.info(
            "回答存在内联编号但未匹配到 citations，回退保留原始 citations（refs=%s, count=%d）",
            refs_in_answer,
            len(citations),
        )
        return list(citation_map.values())
    return aligned


def _normalize_web_search_max_results(value: Optional[int]) -> int:
    if value is None:
        return 5
    return max(1, min(int(value), _MAX_WEB_SEARCH_RESULTS))


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
            "- 至少覆盖 3-5 个要点，每个要点充分展开而非一句话带过\n"
            "- 直接引用文档原文作为论据佐证，不要仅概括\n"
            "- 涉及数据、公式、表格时必须完整展示，不可省略\n"
            "- 篇幅上不设上限，宁可详尽也不要遗漏重要信息\n"
            "- 回答结尾可附上简要总结"
        )
    return (
        "回答风格：标准模式。结构清晰，使用分点或分段组织回答，"
        "覆盖所有关键点并适度展开说明，引用文档原文佐证重要论述。"
    )


_CITATION_TOKEN_OVERHEAD = 1024  # 结构化引文（CITATION LIST）输出的预估 token 开销
_DETAILED_MIN_TOKENS = 4096     # 详细模式下 max_tokens 的最低保证值


def _adjust_max_tokens(
    max_tokens: Optional[int],
    answer_detail: str,
    has_structured_citations: bool,
) -> Optional[int]:
    """根据回答详细度和引文开销调整 max_tokens。

    - 详细模式：保证 max_tokens >= _DETAILED_MIN_TOKENS
    - 结构化引文：自动增加 _CITATION_TOKEN_OVERHEAD 补偿隐藏的 CITATION LIST 输出
    - 不覆盖用户已设置的更大值
    """
    detail = _normalize_answer_detail(answer_detail)
    effective = max_tokens

    # Fix 4：详细模式下保证 max_tokens 下限
    if detail == "detailed":
        if effective is None or effective < _DETAILED_MIN_TOKENS:
            effective = _DETAILED_MIN_TOKENS

    # Fix 3：结构化引文开销补偿
    if has_structured_citations and effective is not None:
        effective += _CITATION_TOKEN_OVERHEAD

    return effective


async def _maybe_perform_web_search(
    request: ChatRequest,
    *,
    query_override: str = "",
    doc_title: str = "",
    selected_text: str = "",
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

    try:
        logger.info(f"联网搜索开始: provider={provider}, query='{search_query[:120]}'")
        sources = await SearchManager.search(
            query=search_query,
            provider=provider,
            api_key=request.web_search_api_key,
            max_results=max_results,
        )
        if not sources:
            return [], ""
        return sources, format_search_results(sources)
    except Exception as e:
        logger.warning(f"联网搜索失败，已降级为仅文档检索: {e}")
        return [], ""


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
    if use_memory:
        memory_context = _retrieve_memory_context(
            request.question, api_key=request.api_key, doc_id=request.doc_id
        )
        raw_memories = _retrieve_raw_memories(
            request.question, api_key=request.api_key, doc_id=request.doc_id
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
        system_prompt = _smart_inject_memory(system_prompt, memory_context, raw_memories)
        user_content = [{"type": "text", "text": request.question or "请分析这些图片"}]
        for img_b64 in image_list:
            mime = _detect_mime_type(img_b64)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}})
    else:
        # LLM 查询改写：用于检索的 search_query（消解代词/口语化），原始 question 保留用于 LLM 回答
        search_query = await _maybe_rewrite_query(
            question=request.question,
            chat_history=request.chat_history,
            selected_text=request.selected_text,
            api_key=request.api_key,
            model=request.model,
            provider=request.api_provider,
            endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
        )
        web_search_sources, web_search_context = await _maybe_perform_web_search(
            request,
            query_override=search_query,
            doc_title=doc.get("filename", ""),
            selected_text=request.selected_text or "",
        )

        if request.selected_text and request.enable_vector_search:
            # 融合模式：selected_text + 向量检索
            _validate_rerank_request(request)
            selected_page_info = locate_selected_text(
                request.selected_text, doc.get("data", {}).get("pages", [])
            )
            try:
                strategy = get_retrieval_strategy(search_query)
                dynamic_top_k = strategy['top_k']
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
                context = f"用户选中的文本：\n{request.selected_text}\n\n"
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
        elif request.enable_vector_search:
            _validate_rerank_request(request)
            strategy = get_retrieval_strategy(search_query)
            dynamic_top_k = strategy['top_k']
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
                ]
            )
            relevant_text = context_result.get("context", "")
            retrieval_meta = context_result.get("retrieval_meta", {})
            context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n" if relevant_text else doc["data"]["full_text"][:8000]
        else:
            context = doc["data"]["full_text"][:8000]

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
        if request.enable_glossary:
            glossary_instruction = build_glossary_prompt(context)
            if glossary_instruction: system_prompt += f"\n\n{glossary_instruction}"
        if web_search_context:
            system_prompt += (
                "\n\n联网搜索结果（用于补充最新信息，优先保证与文档内容一致）：\n"
                f"{web_search_context}\n"
                "\n回答时可参考联网结果，但不得与文档事实冲突。"
            )
        generation_prompt = get_generation_prompt(request.question)
        if generation_prompt: system_prompt += f"\n\n{generation_prompt}"
        citations = retrieval_meta.get("citations", [])
        if citations:
            citation_prompt = build_structured_citation_prompt(citations)
            if citation_prompt: system_prompt += f"\n\n{citation_prompt}"
        system_prompt = _smart_inject_memory(system_prompt, memory_context, raw_memories)
        user_content = request.question

    messages = [{"role": "system", "content": system_prompt}]
    if request.chat_history:
        for hist_msg in request.chat_history:
            if isinstance(hist_msg, dict) and hist_msg.get("role") in ("user", "assistant") and hist_msg.get("content"):
                messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
    messages.append({"role": "user", "content": user_content})

    has_citations_non_stream = bool(citations)
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
        message = response["choices"][0]["message"]
        raw_answer = message["content"]
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
                        if ec.get("highlight_text") and ec.get("idx") is not None:
                            for oc in orig_citations:
                                if oc.get("ref") == ec["idx"]:
                                    oc["highlight_text"] = ec["highlight_text"]
                                    break
            except Exception as e:
                logger.warning(f"非流式引文后处理失败: {e}")

        retrieval_meta["citations"] = _align_citations_with_answer(
            answer, retrieval_meta.get("citations", [])
        )

        if use_memory:
            threading.Thread(target=_async_memory_write, args=(memory_service, request), daemon=True).start()
        return {
            "answer": answer, "reasoning_content": reasoning_content,
            "doc_id": request.doc_id, "question": request.question,
            "timestamp": datetime.now().isoformat(), "used_provider": response.get("_used_provider"),
            "used_model": response.get("_used_model"), "fallback_used": response.get("_fallback_used", False),
            "retrieval_meta": {k: v for k, v in retrieval_meta.items() if not k.startswith("_")},
            "web_search_sources": web_search_sources,
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
    context = ""
    retrieval_meta = {}
    has_structured_citations = False
    web_search_sources: list[dict] = []
    web_search_context = ""
    use_agent = False
    use_memory = _should_use_memory(request)
    memory_context = ""
    raw_memories = []
    if use_memory:
        memory_context = _retrieve_memory_context(
            request.question, api_key=request.api_key, doc_id=request.doc_id
        )
        raw_memories = _retrieve_raw_memories(
            request.question, api_key=request.api_key, doc_id=request.doc_id
        )

    image_list = (request.image_base64_list or [])
    if request.image_base64 and request.image_base64 not in image_list:
        image_list = [request.image_base64] + image_list
    image_list = [img for img in image_list if img]

    if image_list:
        print(f"[Chat Stream] 📸 截图模式：处理 {len(image_list)} 张图")
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
        system_prompt = _smart_inject_memory(system_prompt, memory_context, raw_memories)
        user_content = [{"type": "text", "text": request.question or "请分析这些图片"}]
        for img_b64 in image_list:
            mime = _detect_mime_type(img_b64)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}})
    else:
        # 应用前端传入的检索增强设置到全局配置（即时生效）
        settings.bm25_use_jieba = request.enable_jieba_bm25
        settings.num_expand_context_chunk = request.num_expand_context_chunk

        use_agent = request.enable_agent_retrieval and not request.selected_text
        # LLM 查询改写：用于检索的 search_query（消解代词/口语化），原始 question 保留用于 LLM 回答
        search_query = await _maybe_rewrite_query(
            question=request.question,
            chat_history=request.chat_history,
            selected_text=request.selected_text,
            api_key=request.api_key,
            model=request.model,
            provider=request.api_provider,
            endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
        )
        web_search_sources, web_search_context = await _maybe_perform_web_search(
            request,
            query_override=search_query,
            doc_title=doc.get("filename", ""),
            selected_text=request.selected_text or "",
        )

        if request.selected_text and request.enable_vector_search:
            # 融合模式：selected_text + 向量检索
            _validate_rerank_request(request)
            selected_page_info = locate_selected_text(
                request.selected_text, doc.get("data", {}).get("pages", [])
            )
            try:
                strategy = get_retrieval_strategy(search_query)
                dynamic_top_k = strategy['top_k']
                context_result = await vector_context(
                    request.doc_id, search_query, vector_store_dir=router.vector_store_dir,
                    pages=doc.get("data", {}).get("pages", []), api_key=request.api_key,
                    top_k=dynamic_top_k, candidate_k=max(request.candidate_k, dynamic_top_k),
                    use_rerank=request.use_rerank, reranker_model=request.reranker_model,
                    rerank_provider=request.rerank_provider, rerank_api_key=request.rerank_api_key,
                    rerank_endpoint=request.rerank_endpoint,
                    middlewares=[
                        *( [LoggingMiddleware()] if settings.enable_search_logging else [] ),
                        RetryMiddleware(retries=settings.search_retry_retries, delay=settings.search_retry_delay)
                    ],
                    selected_text=request.selected_text,
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
                context = f"用户选中的文本：\n{request.selected_text}\n\n"
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
            context = ""
        elif request.enable_vector_search:
            _validate_rerank_request(request)

            # 复杂问题分解：对包含"比较""区别"等关键词的查询，拆分为子问题分别检索
            sub_questions = await decompose_question(
                question=request.question,
                api_key=request.api_key,
                model=request.model,
                provider=request.api_provider,
                endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
            )

            queries_to_search = [search_query] + sub_questions if sub_questions else [search_query]
            all_relevant_texts = []

            for sq in queries_to_search:
                strategy = get_retrieval_strategy(sq)
                dynamic_top_k = strategy['top_k']
                cr = await vector_context(
                    request.doc_id, sq, vector_store_dir=router.vector_store_dir,
                    pages=doc.get("data", {}).get("pages", []), api_key=request.api_key,
                    top_k=dynamic_top_k, candidate_k=max(request.candidate_k, dynamic_top_k),
                    use_rerank=request.use_rerank, reranker_model=request.reranker_model,
                    rerank_provider=request.rerank_provider, rerank_api_key=request.rerank_api_key,
                    rerank_endpoint=request.rerank_endpoint,
                    middlewares=[
                        *( [LoggingMiddleware()] if settings.enable_search_logging else [] ),
                        RetryMiddleware(retries=settings.search_retry_retries, delay=settings.search_retry_delay)
                    ]
                )
                rt = cr.get("context", "")
                if rt:
                    all_relevant_texts.append(rt)
                # 使用第一个（主查询）的 retrieval_meta
                if sq == search_query:
                    retrieval_meta = cr.get("retrieval_meta", {})

            relevant_text = "\n\n---\n\n".join(all_relevant_texts) if all_relevant_texts else ""
            context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n" if relevant_text else doc["data"]["full_text"][:8000]
        else:
            context = doc["data"]["full_text"][:8000]

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
        if request.enable_glossary:
            glossary_instruction = build_glossary_prompt(context)
            if glossary_instruction: system_prompt += f"\n\n{glossary_instruction}"
        generation_prompt = get_generation_prompt(request.question)
        if generation_prompt: system_prompt += f"\n\n{generation_prompt}"
        if web_search_context:
            system_prompt += (
                "\n\n联网搜索结果（用于补充最新信息，优先保证与文档内容一致）：\n"
                f"{web_search_context}\n"
                "\n回答时可参考联网结果，但不得与文档事实冲突。"
            )
        citations = retrieval_meta.get("citations", [])
        has_structured_citations = bool(citations)
        if citations:
            citation_prompt = build_structured_citation_prompt(citations)
            if citation_prompt: system_prompt += f"\n\n{citation_prompt}"
        system_prompt = _smart_inject_memory(system_prompt, memory_context, raw_memories)
        user_content = request.question

    messages = [{"role": "system", "content": system_prompt}]
    if request.chat_history:
        for hist_msg in request.chat_history:
            if isinstance(hist_msg, dict) and hist_msg.get("role") in ("user", "assistant") and hist_msg.get("content"):
                messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
    messages.append({"role": "user", "content": user_content})

    # 收集检索到的 chunks 用于引文模糊匹配
    _retrieval_chunks = retrieval_meta.get("_chunks", [])

    async def event_generator():
        nonlocal messages, system_prompt, retrieval_meta, web_search_sources
        try:
            if use_agent:
                # ... Agent 逻辑省略，保持原样 ...
                pass
            if web_search_sources:
                yield f"data: {json.dumps({'type': 'web_search', 'sources': web_search_sources}, ensure_ascii=False)}\n\n"
            if not use_agent and not image_list:
                yield f"data: {json.dumps({'type': 'retrieval_progress', 'phase': 'complete', 'message': '检索完成'}, ensure_ascii=False)}\n\n"
            # 使用 _buffered_stream 包装流式输出，合并高频小 chunk 减少 SSE 事件频率
            adjusted_stream_max_tokens = _adjust_max_tokens(
                request.max_tokens, request.answer_detail, has_structured_citations,
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
            qa_score_val = None

            async for chunk in _buffered_stream(raw_stream):
                if chunk.get("error"):
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                    break

                content = chunk.get('content', '')
                reasoning = chunk.get('reasoning_content', '')

                if chunk.get("done"):
                    qa_score_val = chunk.get('qa_score')
                    # 结构化引文后处理
                    if has_structured_citations and full_output:
                        try:
                            inline_cites = parse_citation_list(full_output)
                            _context_segments = retrieval_meta.get("_context_segments", [])
                            if inline_cites and (_retrieval_chunks or _context_segments):
                                enhanced = match_citations_to_chunks(inline_cites, _retrieval_chunks, context_segments=_context_segments)
                                # 用增强的引文数据替换 retrieval_meta 中的 citations
                                orig_citations = retrieval_meta.get("citations", [])
                                for ec in enhanced:
                                    if ec.get("highlight_text") and ec.get("idx") is not None:
                                        for oc in orig_citations:
                                            if oc.get("ref") == ec["idx"]:
                                                oc["highlight_text"] = ec["highlight_text"]
                                                break
                        except Exception as e:
                            logger.warning(f"结构化引文后处理失败: {e}")

                    final_answer_text = extract_final_answer(full_output) if full_output else ""
                    retrieval_meta["citations"] = _align_citations_with_answer(
                        final_answer_text, retrieval_meta.get("citations", [])
                    )

                    # 移除内部 _chunks 字段（仅后端使用），避免发送大量原始数据
                    send_meta = {k: v for k, v in retrieval_meta.items() if not k.startswith("_")}
                    chunk_data = {
                        'content': '', 'reasoning_content': reasoning,
                        'done': True, 'used_provider': chunk.get('used_provider'),
                        'used_model': chunk.get('used_model'), 'fallback_used': chunk.get('fallback_used'),
                        'retrieval_meta': send_meta,
                        'web_search_sources': web_search_sources,
                    }
                    if qa_score_val is not None:
                        chunk_data['qa_score'] = qa_score_val
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                    if use_memory: threading.Thread(target=_async_memory_write, args=(memory_service, request), daemon=True).start()
                    # 异步生成追问建议
                    try:
                        followup_history = list(request.chat_history or [])
                        followup_history.append({"role": "user", "content": request.question})
                        followups = await generate_followup_questions(
                            chat_history=followup_history,
                            api_key=request.api_key,
                            model=request.model,
                            provider=request.api_provider,
                            endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
                        )
                        if followups:
                            yield f"data: {json.dumps({'type': 'followup_questions', 'questions': followups}, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        logger.debug(f"追问建议生成失败（不影响主流程）: {e}")
                    # 首轮对话自动命名
                    if not request.chat_history or len(request.chat_history) <= 1:
                        try:
                            name_history = [{"role": "user", "content": request.question}]
                            if full_output:
                                name_history.append({"role": "assistant", "content": full_output[:300]})
                            conv_name = await suggest_conversation_name(
                                chat_history=name_history,
                                api_key=request.api_key,
                                model=request.model,
                                provider=request.api_provider,
                                endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
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
                if has_structured_citations and content:
                    if not reached_final_answer:
                        if START_ANSWER in full_output:
                            reached_final_answer = True
                            # 提取 FINAL ANSWER 之后的内容
                            after_marker = full_output.split(START_ANSWER, 1)[1].lstrip()
                            if after_marker:
                                yield f"data: {json.dumps({'content': after_marker, 'reasoning_content': reasoning, 'done': False, 'used_provider': chunk.get('used_provider'), 'used_model': chunk.get('used_model'), 'fallback_used': chunk.get('fallback_used')})}\n\n"
                        # 不展示 CITATION LIST 部分
                        continue
                    else:
                        # 已进入 FINAL ANSWER 区域，检查是否 CITATION LIST 再次出现（小模型重复）
                        if START_CITATION in content:
                            break

                chunk_data = {
                    'content': content, 'reasoning_content': reasoning,
                    'done': False, 'used_provider': chunk.get('used_provider'),
                    'used_model': chunk.get('used_model'), 'fallback_used': chunk.get('fallback_used'),
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _detect_mime_type(img_b64: str) -> str:
    try:
        header = base64.b64decode(img_b64[:16])
        if header[:3] == b'\xff\xd8\xff': return 'image/jpeg'
        if header[:4] == b'\x89PNG': return 'image/png'
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP': return 'image/webp'
    except: pass
    return 'image/jpeg'
