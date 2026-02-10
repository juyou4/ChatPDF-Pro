from datetime import datetime
from typing import Optional, List
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.chat_service import call_ai_api, call_ai_api_stream, extract_reasoning_content
from services.vector_service import vector_context
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

router = APIRouter()

# 上下文构建器实例，用于生成引文指示提示词
_context_builder = ContextBuilder()


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
    # 模型参数（前端可调）
    max_tokens: int = 8192  # 最大输出 token 数
    temperature: float = 0.7  # 温度参数
    top_p: float = 1.0  # 核采样参数
    stream_output: bool = True  # 是否流式输出
    # 多轮对话历史（需求 3.2）
    chat_history: Optional[List[dict]] = None  # [{"role": "user"|"assistant", "content": "..."}]


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
            top_p=request.top_p
        )
        message = response["choices"][0]["message"]
        answer = message["content"]
        reasoning_content = extract_reasoning_content(message)

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
        try:
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
                top_p=request.top_p
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
                    yield "data: [DONE]\n\n"
                    break

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
