from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.chat_service import call_ai_api, call_ai_api_stream
from services.vector_service import vector_context
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
    top_k: int = 5
    candidate_k: int = 20
    use_rerank: bool = False
    reranker_model: Optional[str] = None
    rerank_provider: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    doc_store_key: Optional[str] = None


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
    if use_rerank and provider and provider.lower() in {"cohere", "jina"} and not api_key:
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
    if request.selected_text:
        context = f"用户选中的文本：\n{request.selected_text}\n\n"
    elif request.enable_vector_search:
        _validate_rerank_request(request)
        relevant_text = await vector_context(
            request.doc_id,
            request.question,
            vector_store_dir=router.vector_store_dir,
            pages=doc.get("data", {}).get("pages", []),
            api_key=request.api_key,
            top_k=request.top_k,
            candidate_k=max(request.candidate_k, request.top_k),
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
        if relevant_text:
            context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n"
        else:
            context = doc["data"]["full_text"][:8000]
    else:
        context = doc["data"]["full_text"][:8000]

    system_prompt = f"""你是专业的文档分析助手。用户上传了一份PDF文档。

文档名称：{doc["filename"]}
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

请根据文档内容准确回答用户的问题。如果文档中没有相关信息，请明确告知。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.question}
    ]

    middlewares = build_chat_middlewares()
    try:
        response = await call_ai_api(
            messages,
            request.api_key,
            request.model,
            request.api_provider,
            endpoint=PROVIDER_CONFIG.get(request.api_provider, {}).get("endpoint", ""),
            middlewares=middlewares
        )
        answer = response["choices"][0]["message"]["content"]

        return {
            "answer": answer,
            "doc_id": request.doc_id,
            "question": request.question,
            "timestamp": datetime.now().isoformat(),
            "used_provider": response.get("_used_provider"),
            "used_model": response.get("_used_model"),
            "fallback_used": response.get("_fallback_used", False)
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
    if request.selected_text:
        context = f"用户选中的文本：\n{request.selected_text}\n\n"
    elif request.enable_vector_search:
        _validate_rerank_request(request)
        relevant_text = await vector_context(
            request.doc_id,
            request.question,
            vector_store_dir=router.vector_store_dir,
            pages=doc.get("data", {}).get("pages", []),
            api_key=request.api_key,
            top_k=request.top_k,
            candidate_k=max(request.candidate_k, request.top_k),
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
        if relevant_text:
            context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n"
        else:
            context = doc["data"]["full_text"][:8000]
    else:
        context = doc["data"]["full_text"][:8000]

    system_prompt = f"""你是专业的文档分析助手。用户上传了一份PDF文档。

文档名称：{doc["filename"]}
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

请根据文档内容准确回答用户的问题。如果文档中没有相关信息，请明确告知。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.question}
    ]

    async def event_generator():
        try:
            middlewares = build_chat_middlewares()
            async for chunk in call_ai_api_stream(
                messages,
                request.api_key,
                request.model,
                request.api_provider,
                endpoint=PROVIDER_CONFIG.get(request.api_provider, {}).get("endpoint", ""),
                middlewares=middlewares
            ):
                if chunk.get("error"):
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                    break
                yield f"data: {json.dumps({'content': chunk.get('content', ''), 'done': chunk.get('done', False), 'used_provider': chunk.get('used_provider'), 'used_model': chunk.get('used_model'), 'fallback_used': chunk.get('fallback_used')})}\n\n"
                if chunk.get("done"):
                    yield "data: [DONE]\n\n"
                    break

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
