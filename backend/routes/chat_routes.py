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
from services.retrieval_agent import RetrievalAgent
from services.retrieval_tools import DocContext
from services.glossary_service import glossary_service, build_glossary_prompt
from services.table_service import protect_markdown_tables, restore_markdown_tables
from services.query_analyzer import get_retrieval_strategy
from services.preset_service import get_generation_prompt
from services.context_builder import ContextBuilder
from models.provider_registry import PROVIDER_CONFIG
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
    top_k: int = 10  # 增加到10，获取更多上下文
    candidate_k: int = 20
    use_rerank: bool = False
    reranker_model: Optional[str] = None
    rerank_provider: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    doc_store_key: Optional[str] = None
    # 新增：术语库和表格保护选项
    enable_glossary: bool = True  # 是否启用术语库
    protect_tables: bool = True   # 是否保护表格结构
    # 深度思考模式
    enable_thinking: bool = False  # 是否开启深度思考
    # 模型参数（前端可调，None 表示不传，由模型使用默认值）
    max_tokens: Optional[int] = None  # 最大输出 token 数
    temperature: Optional[float] = None  # 温度参数
    top_p: Optional[float] = None  # 核采样参数
    custom_params: Optional[dict] = None  # 自定义参数 {key: value}，直接透传给 API
    reasoning_effort: Optional[str] = None  # 深度思考力度（'low'|'medium'|'high'）
    stream_output: bool = True  # 是否流式输出
    # 多轮对话历史（需求 3.2）
    chat_history: Optional[List[dict]] = None  # [{"role": "user"|"assistant", "content": "..."}]
    # 记忆功能开关（需求 5.4）
    enable_memory: bool = True  # 是否启用记忆功能
    # Agent 多轮检索（需求 P0）
    enable_agent_retrieval: bool = False  # 是否启用多轮 Agent 检索


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
    """检索记忆上下文，异常时返回空字符串（需求 5.1, 5.5）
    
    Args:
        question: 用户问题
        api_key: API 密钥
        doc_id: 当前文档 ID，用于文档相关性加权
    """
    if memory_service is None:
        return ""
    try:
        # 使用文档相关性加权，但不过滤（保留跨文档记忆）
        return memory_service.retrieve_memories(
            question, api_key=api_key, doc_id=doc_id, filter_by_doc=False
        )
    except Exception as e:
        logger.error(f"记忆检索失败: {e}")
        return ""


def _async_memory_write(svc, request):
    """异步记忆写入：提取 QA 摘要 + 更新关键词（需求 5.3）"""
    try:
        # 提取 QA 摘要（传入 LLM 参数用于记忆提炼）
        if request.doc_id:
            # 构建完整对话历史（包含当前问题）
            history = list(request.chat_history or [])
            history.append({"role": "user", "content": request.question})
            svc.save_qa_summary(
                request.doc_id,
                history,
                api_key=getattr(request, "api_key", None),
                model=getattr(request, "model", None),
                api_provider=getattr(request, "api_provider", None),
            )
        # 更新关键词统计
        svc.update_keywords(request.question)
    except Exception as e:
        logger.error(f"异步记忆写入失败: {e}")


# 跟踪已 flush 过的 doc_id，防止同一会话重复 flush
_flushed_sessions: set = set()


def _maybe_flush_memory(request) -> None:
    """当 chat_history 较长时，提前触发一次记忆写入（借鉴 OpenClaw memoryFlush）

    防止长会话中间轮次的重要信息丢失。每个 doc_id 每次会话只 flush 一次。
    
    优化点：
    1. 使用精确的 token 估算（考虑中英文差异）
    2. 基于配置化阈值触发
    3. 支持禁用开关
    """
    if memory_service is None:
        return
    
    # 检查是否启用记忆刷新
    if not settings.memory_flush_enabled:
        return
    
    history = getattr(request, "chat_history", None)
    if not history:
        return
    
    doc_id = getattr(request, "doc_id", "")
    if not doc_id or doc_id in _flushed_sessions:
        return

    # 使用精确的 token 估算（考虑中英文差异）
    from services.token_budget import TokenBudget
    budget = TokenBudget()
    
    total_tokens = 0
    for msg in history:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if content:
                total_tokens += budget.estimate_tokens(content)
    
    # 检查是否达到阈值
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
    """判断是否应启用记忆功能（需求 5.4）"""
    return (
        settings.memory_enabled
        and getattr(request, "enable_memory", True)
        and memory_service is not None
    )


def _inject_memory_context(system_prompt: str, memory_context: str) -> str:
    """将记忆上下文注入 system prompt（需求 5.2）
    格式：在文档内容之后、回答规则之前插入记忆段落"""
    if not memory_context:
        return system_prompt
    # 在"回答规则："之前插入记忆上下文
    marker = "\n回答规则："
    if marker in system_prompt:
        idx = system_prompt.index(marker)
        return (
            system_prompt[:idx]
            + f"\n\n用户历史记忆：\n{memory_context}"
            + system_prompt[idx:]
        )
    # 如果没有找到标记，追加到末尾
    return system_prompt + f"\n\n用户历史记忆：\n{memory_context}"


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

    # Compaction 前自动 flush：长会话提前保存记忆（借鉴 OpenClaw）
    if use_memory:
        _maybe_flush_memory(request)

    # 记忆检索：在构建 system prompt 之前执行（需求 5.1）
    memory_context = ""
    if use_memory:
        memory_context = _retrieve_memory_context(
            request.question, api_key=request.api_key, doc_id=request.doc_id
        )

    # 截图模式：跳过向量检索，使用 vision 专用精简 prompt，让模型专注分析图片
    if request.image_base64:
        print(f"[Chat] 📸 截图模式：跳过向量检索，使用 vision 专用 prompt (model={request.model}, provider={request.api_provider})")
        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
用户从文档中截取了一张图片并发送给你。请仔细分析用户发送的图片内容并回答问题。

回答规则：
1. 以用户发送的图片为核心依据进行回答，不要参考其他内容。
2. 如果图片包含图表，请分析数据趋势和关键信息。
3. 如果图片包含公式，请使用 LaTeX 格式（$公式$）展示。
4. 如果图片包含表格，请转换为 Markdown 格式。
5. 简洁清晰，学术准确。"""

        # 截图模式也注入记忆上下文（需求 5.2）
        system_prompt = _inject_memory_context(system_prompt, memory_context)

        user_content = [
            {"type": "text", "text": request.question or "请分析这张图片"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}}
        ]
    else:
        # 非截图模式：正常的文本检索流程
        if request.selected_text:
            context = f"用户选中的文本：\n{request.selected_text}\n\n"
        elif request.enable_vector_search:
            _validate_rerank_request(request)
            
            # 智能分析查询类型，动态调整top_k
            strategy = get_retrieval_strategy(request.question)
            dynamic_top_k = strategy['top_k']
            
            print(f"[Chat] 查询类型: {strategy['query_type']}, 动态top_k: {dynamic_top_k}, 原因: {strategy['reasoning']}")
            
            # vector_context 返回包含 context 和 retrieval_meta 的字典
            context_result = await vector_context(
                request.doc_id,
                request.question,
                vector_store_dir=router.vector_store_dir,
                pages=doc.get("data", {}).get("pages", []),
                api_key=request.api_key,
                top_k=dynamic_top_k,  # 使用动态计算的top_k
                candidate_k=max(request.candidate_k, dynamic_top_k),
                use_rerank=request.use_rerank,
                reranker_model=request.reranker_model,
                rerank_provider=request.rerank_provider,
                rerank_api_key=request.rerank_api_key,
                rerank_endpoint=request.rerank_endpoint,
                middlewares=[
                    *( [LoggingMiddleware()] if settings.enable_search_logging else [] ),
                    RetryMiddleware(retries=settings.search_retry_retries, delay=settings.search_retry_delay),
                    ErrorCaptureMiddleware()
                ]
            )
            relevant_text = context_result.get("context", "")
            retrieval_meta = context_result.get("retrieval_meta", {})
            if relevant_text:
                context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n"
            else:
                context = doc["data"]["full_text"][:8000]
        else:
            context = doc["data"]["full_text"][:8000]

        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

回答规则：
1. 基于文档内容准确回答，简洁清晰，学术准确。
2. 不要声明你是否具备外部工具/联网等能力，不要输出与回答无关的免责声明。
3. 优先依据文档内容回答；若文档信息不足，请基于常识给出概览性解答并明确不确定之处。
4. 遇到公式、数据、图表等关键信息时，必须直接引用原文展示完整内容，不要仅概括描述。
   - 例如用户问"有什么公式"时，应直接展示公式的完整表达式。
   - 对于数学公式，优先使用LaTeX格式展示（$公式$）。
5. 不要说"根据您提供的有限片段"、"基于片段"等暗示信息不足的措辞，直接回答问题。"""

        # 集成术语库 - 在 system_prompt 中注入术语指令
        if request.enable_glossary:
            glossary_instruction = build_glossary_prompt(context)
            if glossary_instruction:
                system_prompt += f"\n\n{glossary_instruction}"

        # 检测生成类查询（思维导图/流程图），注入对应系统提示词
        generation_prompt = get_generation_prompt(request.question)
        if generation_prompt:
            system_prompt += f"\n\n{generation_prompt}"

        # 引文追踪：如果 retrieval_meta 中包含 citations，追加引文指示提示词
        citations = retrieval_meta.get("citations", [])
        if citations:
            citation_prompt = _context_builder.build_citation_prompt(citations)
            if citation_prompt:
                system_prompt += f"\n\n{citation_prompt}"

        # 注入记忆上下文到 system prompt（需求 5.2）
        system_prompt = _inject_memory_context(system_prompt, memory_context)

        user_content = request.question

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # 插入多轮对话历史（需求 3.2：位于 system prompt 之后、当前用户消息之前）
    if request.chat_history:
        for hist_msg in request.chat_history:
            if isinstance(hist_msg, dict) and hist_msg.get("role") in ("user", "assistant") and hist_msg.get("content"):
                messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
    messages.append({"role": "user", "content": user_content})

    middlewares = build_chat_middlewares()
    try:
        response = await call_ai_api(
            messages,
            request.api_key,
            request.model,
            request.api_provider,
            endpoint=PROVIDER_CONFIG.get(request.api_provider, {}).get("endpoint", ""),
            middlewares=middlewares,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            custom_params=request.custom_params,
            reasoning_effort=request.reasoning_effort,
        )
        message = response["choices"][0]["message"]
        answer = message["content"]
        reasoning_content = extract_reasoning_content(message)

        # 异步触发记忆写入（需求 5.3）
        if use_memory:
            threading.Thread(
                target=_async_memory_write,
                args=(memory_service, request),
                daemon=True,
            ).start()

        return {
            "answer": answer,
            "reasoning_content": reasoning_content,
            "doc_id": request.doc_id,
            "question": request.question,
            "timestamp": datetime.now().isoformat(),
            "used_provider": response.get("_used_provider"),
            "used_model": response.get("_used_model"),
            "fallback_used": response.get("_fallback_used", False),
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

    # 记忆检索：在构建 system prompt 之前执行（需求 5.1）
    memory_context = ""
    if use_memory:
        memory_context = _retrieve_memory_context(
            request.question, api_key=request.api_key, doc_id=request.doc_id
        )

    # 截图模式：跳过向量检索，使用 vision 专用精简 prompt，让模型专注分析图片
    if request.image_base64:
        print(f"[Chat Stream] 📸 截图模式：跳过向量检索，使用 vision 专用 prompt (model={request.model}, provider={request.api_provider})")
        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
用户从文档中截取了一张图片并发送给你。请仔细分析用户发送的图片内容并回答问题。

回答规则：
1. 以用户发送的图片为核心依据进行回答，不要参考其他内容。
2. 如果图片包含图表，请分析数据和关键信息。
3. 如果图片包含公式，请使用 LaTeX 格式（$公式$）展示。
4. 如果图片包含表格，请转换为 Markdown 格式。
5. 简洁清晰，学术准确。"""

        # 截图模式也注入记忆上下文（需求 5.2）
        system_prompt = _inject_memory_context(system_prompt, memory_context)

        user_content = [
            {"type": "text", "text": request.question or "请分析这张图片"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}}
        ]
    else:
        # 非截图模式：正常的文本检索流程
        # Agent 模式标志：后续在 event_generator 中使用
        use_agent = request.enable_agent_retrieval and not request.selected_text

        if request.selected_text:
            context = f"用户选中的文本：\n{request.selected_text}\n\n"
        elif use_agent:
            # Agent 模式：上下文由 event_generator 中的 RetrievalAgent 动态生成
            # 此处仅设置占位，实际上下文在流式生成器中填充
            context = ""
        elif request.enable_vector_search:
            _validate_rerank_request(request)
            
            # 智能分析查询类型，动态调整top_k
            strategy = get_retrieval_strategy(request.question)
            dynamic_top_k = strategy['top_k']
            
            print(f"[Chat Stream] 查询类型: {strategy['query_type']}, 动态top_k: {dynamic_top_k}, 原因: {strategy['reasoning']}")
            
            # vector_context 返回包含 context 和 retrieval_meta 的字典
            context_result = await vector_context(
                request.doc_id,
                request.question,
                vector_store_dir=router.vector_store_dir,
                pages=doc.get("data", {}).get("pages", []),
                api_key=request.api_key,
                top_k=dynamic_top_k,  # 使用动态计算的top_k
                candidate_k=max(request.candidate_k, dynamic_top_k),
                use_rerank=request.use_rerank,
                reranker_model=request.reranker_model,
                rerank_provider=request.rerank_provider,
                rerank_api_key=request.rerank_api_key,
                rerank_endpoint=request.rerank_endpoint,
                middlewares=[
                    *( [LoggingMiddleware()] if settings.enable_search_logging else [] ),
                    RetryMiddleware(retries=settings.search_retry_retries, delay=settings.search_retry_delay)
                ]
            )
            relevant_text = context_result.get("context", "")
            retrieval_meta = context_result.get("retrieval_meta", {})
            if relevant_text:
                context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n"
            else:
                context = doc["data"]["full_text"][:8000]
        else:
            context = doc["data"]["full_text"][:8000]

        system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

回答规则：
1. 基于文档内容准确回答，简洁清晰，学术准确。
2. 不要声明你是否具备外部工具/联网等能力，不要输出与回答无关的免责声明。
3. 优先依据文档内容回答；若文档信息不足，请基于常识给出概览性解答并明确不确定之处。
4. 遇到公式、数据、图表等关键信息时，必须直接引用原文展示完整内容，不要仅概括描述。
   - 例如用户问"有什么公式"时，应直接展示公式的完整表达式。
   - 对于数学公式，优先使用LaTeX格式展示（$公式$）。
5. 不要说"根据您提供的有限片段"、"基于片段"等暗示信息不足的措辞，直接回答问题。"""

        # 检测生成类查询（思维导图/流程图），注入对应系统提示词
        generation_prompt = get_generation_prompt(request.question)
        if generation_prompt:
            system_prompt += f"\n\n{generation_prompt}"

        # 引文追踪：如果 retrieval_meta 中包含 citations，追加引文指示提示词
        citations = retrieval_meta.get("citations", [])
        if citations:
            citation_prompt = _context_builder.build_citation_prompt(citations)
            if citation_prompt:
                system_prompt += f"\n\n{citation_prompt}"

        # 注入记忆上下文到 system prompt（需求 5.2）
        system_prompt = _inject_memory_context(system_prompt, memory_context)

        user_content = request.question

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # 插入多轮对话历史（需求 3.2：位于 system prompt 之后、当前用户消息之前）
    if request.chat_history:
        for hist_msg in request.chat_history:
            if isinstance(hist_msg, dict) and hist_msg.get("role") in ("user", "assistant") and hist_msg.get("content"):
                messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
    messages.append({"role": "user", "content": user_content})

    async def event_generator():
        nonlocal messages, system_prompt, retrieval_meta
        try:
            # ============ Agent 多轮检索模式 ============
            if use_agent:
                from services.semantic_group_service import SemanticGroupService
                from services.embedding_service import _get_semantic_groups_dir

                # 构建 DocContext
                full_text = doc.get("data", {}).get("full_text", "")
                pages = doc.get("data", {}).get("pages", [])
                chunks = doc.get("data", {}).get("chunks", [])
                if not chunks:
                    # 从 pages 提取 chunks
                    chunks = [p.get("content", "") for p in pages if p.get("content")]

                # 尝试加载语义意群
                groups_dir = _get_semantic_groups_dir()
                group_svc = SemanticGroupService()
                semantic_groups = group_svc.load_groups(request.doc_id, groups_dir) or []

                doc_ctx = DocContext(
                    doc_id=request.doc_id,
                    full_text=full_text,
                    chunks=chunks,
                    pages=pages,
                    semantic_groups=semantic_groups,
                    vector_store_dir=getattr(router, "vector_store_dir", ""),
                    api_key=request.api_key or "",
                )

                agent = RetrievalAgent(
                    api_key=request.api_key or "",
                    model=request.model,
                    provider=request.api_provider,
                    endpoint=PROVIDER_CONFIG.get(request.api_provider, {}).get("endpoint", ""),
                    max_rounds=settings.agent_max_rounds,
                    temperature=settings.agent_planner_temperature,
                )

                agent_context = ""
                async for event in agent.run(
                    question=request.question,
                    doc_ctx=doc_ctx,
                    doc_name=doc.get("filename", ""),
                ):
                    if event["type"] == "retrieval_progress":
                        # 向前端发送检索进度
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    elif event["type"] == "retrieval_complete":
                        agent_context = event.get("context", "")
                        retrieval_meta["agent_search_history"] = event.get("search_history", [])
                        retrieval_meta["agent_detail"] = event.get("detail", [])

                # 用 Agent 获取的上下文重建 system_prompt
                if agent_context:
                    system_prompt = f"""你是专业的PDF文档智能助手。用户正在查看文档"{doc["filename"]}"。
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{agent_context}

回答规则：
1. 基于文档内容准确回答，简洁清晰，学术准确。
2. 不要声明你是否具备外部工具/联网等能力，不要输出与回答无关的免责声明。
3. 优先依据文档内容回答；若文档信息不足，请基于常识给出概览性解答并明确不确定之处。
4. 遇到公式、数据、图表等关键信息时，必须直接引用原文展示完整内容，不要仅概括描述。
   - 例如用户问"有什么公式"时，应直接展示公式的完整表达式。
   - 对于数学公式，优先使用LaTeX格式展示（$公式$）。
5. 不要说"根据您提供的有限片段"、"基于片段"等暗示信息不足的措辞，直接回答问题。"""

                    # 重新注入记忆上下文
                    system_prompt = _inject_memory_context(system_prompt, memory_context)

                    # 检测生成类查询
                    generation_prompt = get_generation_prompt(request.question)
                    if generation_prompt:
                        system_prompt += f"\n\n{generation_prompt}"

                    # 重建 messages
                    messages = [{"role": "system", "content": system_prompt}]
                    if request.chat_history:
                        for hist_msg in request.chat_history:
                            if isinstance(hist_msg, dict) and hist_msg.get("role") in ("user", "assistant") and hist_msg.get("content"):
                                messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
                    messages.append({"role": "user", "content": request.question})

            # ============ 非 Agent 模式的检索进度反馈 ============
            if not use_agent and not request.image_base64:
                yield f"data: {json.dumps({'type': 'retrieval_progress', 'phase': 'complete', 'message': '检索完成，正在生成回答...'}, ensure_ascii=False)}\n\n"

            # ============ 流式 LLM 回答 ============
            middlewares = build_chat_middlewares()
            async for chunk in call_ai_api_stream(
                messages,
                request.api_key,
                request.model,
                request.api_provider,
                endpoint=PROVIDER_CONFIG.get(request.api_provider, {}).get("endpoint", ""),
                middlewares=middlewares,
                enable_thinking=request.enable_thinking,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                custom_params=request.custom_params,
                reasoning_effort=request.reasoning_effort,
            ):
                if chunk.get("error"):
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                    break
                chunk_data = {
                    'content': chunk.get('content', ''),
                    'reasoning_content': chunk.get('reasoning_content', ''),
                    'done': chunk.get('done', False),
                    'used_provider': chunk.get('used_provider'),
                    'used_model': chunk.get('used_model'),
                    'fallback_used': chunk.get('fallback_used'),
                }
                # 在最后一个 chunk 中附带 retrieval_meta（含 citations）
                if chunk.get("done"):
                    chunk_data['retrieval_meta'] = retrieval_meta
                yield f"data: {json.dumps(chunk_data)}\n\n"
                if chunk.get("done"):
                    # 异步触发记忆写入（需求 5.3）
                    if use_memory:
                        threading.Thread(
                            target=_async_memory_write,
                            args=(memory_service, request),
                            daemon=True,
                        ).start()
                    yield "data: [DONE]\n\n"
                    break

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
