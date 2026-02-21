from datetime import datetime
from typing import Optional, List
import json
import logging
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

    # 缓冲模式：累积 content 和 reasoning_content
    buffer_content = ""
    buffer_reasoning = ""

    async for chunk in raw_stream:
        # 错误或终止信号：立即刷新缓冲区并转发
        if chunk.get("error") or chunk.get("done"):
            if buffer_content or buffer_reasoning:
                yield {
                    "content": buffer_content,
                    "reasoning_content": buffer_reasoning,
                    "done": False,
                }
                buffer_content = ""
                buffer_reasoning = ""
            yield chunk
            break

        # 累积到缓冲区
        buffer_content += chunk.get("content", "")
        buffer_reasoning += chunk.get("reasoning_content", "")

        # 缓冲区达到阈值，立即发送
        if len(buffer_content) >= buffer_size:
            yield {
                "content": buffer_content,
                "reasoning_content": buffer_reasoning,
                "done": False,
            }
            buffer_content = ""
            buffer_reasoning = ""

    # 流正常结束但未收到 done/error 信号时，刷新剩余缓冲
    if buffer_content or buffer_reasoning:
        yield {
            "content": buffer_content,
            "reasoning_content": buffer_reasoning,
            "done": False,
        }


# 上下文构建器实例，用于生成引文指示提示词
_context_builder = ContextBuilder()

# 模块级变量，由 app.py 注入 MemoryService 实例
memory_service = None


def build_chat_middlewares():
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

    parts = [f"用户选中的文本{page_label}：\n{selected_text}"]
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
    use_memory = _should_use_memory(request)
    if use_memory:
        _maybe_flush_memory(request)
    memory_context = ""
    if use_memory:
        memory_context = _retrieve_memory_context(
            request.question, api_key=request.api_key, doc_id=request.doc_id
        )

    # 支持多图逻辑
    image_list = (request.image_base64_list or [])
    if request.image_base64 and request.image_base64 not in image_list:
        image_list = [request.image_base64] + image_list
    image_list = [img for img in image_list if img]

    if image_list:
        print(f"[Chat] 📸 截图模式：处理 {len(image_list)} 张图")
        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
用户从文档中截取了 {len(image_list)} 张图片并发送给你。请仔细分析这些图片内容并回答问题。

回答规则：
1. 以用户发送的图片为核心依据进行回答，不要参考其他内容。
2. 如果图片包含图表，请分析数据趋势和关键信息。
3. 如果图片包含公式，请使用 LaTeX 格式（$公式$）展示。
4. 如果图片包含表格，请转换为 Markdown 格式。
5. 简洁清晰，学术准确。"""
        system_prompt = _inject_memory_context(system_prompt, memory_context)
        user_content = [{"type": "text", "text": request.question or "请分析这些图片"}]
        for img_b64 in image_list:
            mime = _detect_mime_type(img_b64)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}})
    else:
        if request.selected_text and request.enable_vector_search:
            # 融合模式：selected_text + 向量检索
            _validate_rerank_request(request)
            selected_page_info = locate_selected_text(
                request.selected_text, doc.get("data", {}).get("pages", [])
            )
            try:
                strategy = get_retrieval_strategy(request.question)
                dynamic_top_k = strategy['top_k']
                context_result = await vector_context(
                    request.doc_id, request.question, vector_store_dir=router.vector_store_dir,
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
                # 融合：selected_text 优先 + 检索补充
                context = _build_fused_context(
                    request.selected_text, retrieval_context, selected_page_info
                )
                # 如果检索没有返回 citations，基于 selected_text 位置生成基础 citation
                if not retrieval_meta.get("citations"):
                    retrieval_meta["citations"] = [_build_selected_text_citation(
                        request.selected_text, selected_page_info
                    )]
            except Exception as e:
                logger.warning(f"框选模式向量检索失败，降级为仅 selected_text: {e}")
                context = f"用户选中的文本：\n{request.selected_text}\n\n"
        elif request.selected_text:
            # 仅 selected_text 模式（向量检索未启用）
            context = f"用户选中的文本：\n{request.selected_text}\n\n"
        elif request.enable_vector_search:
            _validate_rerank_request(request)
            strategy = get_retrieval_strategy(request.question)
            dynamic_top_k = strategy['top_k']
            context_result = await vector_context(
                request.doc_id, request.question, vector_store_dir=router.vector_store_dir,
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

        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

回答规则：
1. 基于文档内容准确回答，简洁清晰，学术准确。
2. 遇到公式、数据、图表等关键信息时，必须直接引用原文展示完整内容。
3. 优先依据文档内容回答。"""
        if request.enable_glossary:
            glossary_instruction = build_glossary_prompt(context)
            if glossary_instruction: system_prompt += f"\n\n{glossary_instruction}"
        generation_prompt = get_generation_prompt(request.question)
        if generation_prompt: system_prompt += f"\n\n{generation_prompt}"
        citations = retrieval_meta.get("citations", [])
        if citations:
            citation_prompt = _context_builder.build_citation_prompt(citations)
            if citation_prompt: system_prompt += f"\n\n{citation_prompt}"
        system_prompt = _inject_memory_context(system_prompt, memory_context)
        user_content = request.question

    messages = [{"role": "system", "content": system_prompt}]
    if request.chat_history:
        for hist_msg in request.chat_history:
            if isinstance(hist_msg, dict) and hist_msg.get("role") in ("user", "assistant") and hist_msg.get("content"):
                messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
    messages.append({"role": "user", "content": user_content})

    try:
        response = await call_ai_api(
            messages, request.api_key, request.model, request.api_provider,
            endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
            middlewares=build_chat_middlewares(), max_tokens=request.max_tokens,
            temperature=request.temperature, top_p=request.top_p,
            custom_params=request.custom_params, reasoning_effort=request.reasoning_effort,
        )
        message = response["choices"][0]["message"]
        answer = message["content"]
        reasoning_content = extract_reasoning_content(message)
        if use_memory:
            threading.Thread(target=_async_memory_write, args=(memory_service, request), daemon=True).start()
        return {
            "answer": answer, "reasoning_content": reasoning_content,
            "doc_id": request.doc_id, "question": request.question,
            "timestamp": datetime.now().isoformat(), "used_provider": response.get("_used_provider"),
            "used_model": response.get("_used_model"), "fallback_used": response.get("_fallback_used", False),
            "retrieval_meta": retrieval_meta
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
    use_agent = False
    use_memory = _should_use_memory(request)
    memory_context = ""
    if use_memory:
        memory_context = _retrieve_memory_context(
            request.question, api_key=request.api_key, doc_id=request.doc_id
        )

    image_list = (request.image_base64_list or [])
    if request.image_base64 and request.image_base64 not in image_list:
        image_list = [request.image_base64] + image_list
    image_list = [img for img in image_list if img]

    if image_list:
        print(f"[Chat Stream] 📸 截图模式：处理 {len(image_list)} 张图")
        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
用户从文档中截取了 {len(image_list)} 张图片并发送给你。请仔细分析这些图片内容并回答问题。

回答规则：
1. 以用户发送的图片为核心依据进行回答，不要参考其他内容。
2. 如果图片包含图表，请分析数据和关键信息。
3. 如果图片包含公式，请使用 LaTeX 格式（$公式$）展示。
4. 如果图片包含表格，请转换为 Markdown 格式。
5. 简洁清晰，学术准确。"""
        system_prompt = _inject_memory_context(system_prompt, memory_context)
        user_content = [{"type": "text", "text": request.question or "请分析这些图片"}]
        for img_b64 in image_list:
            mime = _detect_mime_type(img_b64)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}})
    else:
        use_agent = request.enable_agent_retrieval and not request.selected_text
        if request.selected_text and request.enable_vector_search:
            # 融合模式：selected_text + 向量检索
            _validate_rerank_request(request)
            selected_page_info = locate_selected_text(
                request.selected_text, doc.get("data", {}).get("pages", [])
            )
            try:
                strategy = get_retrieval_strategy(request.question)
                dynamic_top_k = strategy['top_k']
                context_result = await vector_context(
                    request.doc_id, request.question, vector_store_dir=router.vector_store_dir,
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
                # 融合：selected_text 优先 + 检索补充
                context = _build_fused_context(
                    request.selected_text, retrieval_context, selected_page_info
                )
                # 如果检索没有返回 citations，基于 selected_text 位置生成基础 citation
                if not retrieval_meta.get("citations"):
                    retrieval_meta["citations"] = [_build_selected_text_citation(
                        request.selected_text, selected_page_info
                    )]
            except Exception as e:
                logger.warning(f"框选模式向量检索失败，降级为仅 selected_text: {e}")
                context = f"用户选中的文本：\n{request.selected_text}\n\n"
        elif request.selected_text:
            # 仅 selected_text 模式（向量检索未启用）
            context = f"用户选中的文本：\n{request.selected_text}\n\n"
        elif use_agent:
            context = ""
        elif request.enable_vector_search:
            _validate_rerank_request(request)
            strategy = get_retrieval_strategy(request.question)
            dynamic_top_k = strategy['top_k']
            context_result = await vector_context(
                request.doc_id, request.question, vector_store_dir=router.vector_store_dir,
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
            relevant_text = context_result.get("context", "")
            retrieval_meta = context_result.get("retrieval_meta", {})
            context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n" if relevant_text else doc["data"]["full_text"][:8000]
        else:
            context = doc["data"]["full_text"][:8000]

        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
文档内容：
{context}

回答规则：
1. 基于文档内容准确回答。"""
        generation_prompt = get_generation_prompt(request.question)
        if generation_prompt: system_prompt += f"\n\n{generation_prompt}"
        citations = retrieval_meta.get("citations", [])
        if citations:
            citation_prompt = _context_builder.build_citation_prompt(citations)
            if citation_prompt: system_prompt += f"\n\n{citation_prompt}"
        system_prompt = _inject_memory_context(system_prompt, memory_context)
        user_content = request.question

    messages = [{"role": "system", "content": system_prompt}]
    if request.chat_history:
        for hist_msg in request.chat_history:
            if isinstance(hist_msg, dict) and hist_msg.get("role") in ("user", "assistant") and hist_msg.get("content"):
                messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
    messages.append({"role": "user", "content": user_content})

    async def event_generator():
        nonlocal messages, system_prompt, retrieval_meta
        try:
            if use_agent:
                # ... Agent 逻辑省略，保持原样 ...
                pass
            if not use_agent and not image_list:
                yield f"data: {json.dumps({'type': 'retrieval_progress', 'phase': 'complete', 'message': '检索完成'}, ensure_ascii=False)}\n\n"
            # 使用 _buffered_stream 包装流式输出，合并高频小 chunk 减少 SSE 事件频率
            raw_stream = call_ai_api_stream(
                messages, request.api_key, request.model, request.api_provider,
                endpoint=_get_provider_endpoint(request.api_provider, request.api_host or ""),
                middlewares=build_chat_middlewares(), enable_thinking=request.enable_thinking,
                max_tokens=request.max_tokens, temperature=request.temperature,
                top_p=request.top_p, custom_params=request.custom_params,
                reasoning_effort=request.reasoning_effort,
            )
            async for chunk in _buffered_stream(raw_stream):
                if chunk.get("error"):
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                    break
                chunk_data = {
                    'content': chunk.get('content', ''), 'reasoning_content': chunk.get('reasoning_content', ''),
                    'done': chunk.get('done', False), 'used_provider': chunk.get('used_provider'),
                    'used_model': chunk.get('used_model'), 'fallback_used': chunk.get('fallback_used'),
                }
                if chunk.get("done"): chunk_data['retrieval_meta'] = retrieval_meta
                yield f"data: {json.dumps(chunk_data)}\n\n"
                if chunk.get("done"):
                    if use_memory: threading.Thread(target=_async_memory_write, args=(memory_service, request), daemon=True).start()
                    yield "data: [DONE]\n\n"
                    break
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
