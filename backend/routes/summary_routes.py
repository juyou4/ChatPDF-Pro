from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.provider_registry import PROVIDER_CONFIG
from services.chat_service import call_ai_api
from utils.middleware import (
    LoggingMiddleware,
    RetryMiddleware,
    ErrorCaptureMiddleware,
    TimeoutMiddleware,
    FallbackMiddleware,
    DegradeOnErrorMiddleware,
)
from config import settings

router = APIRouter()


class SummaryRequest(BaseModel):
    doc_id: str
    api_key: Optional[str] = None
    model: str
    api_provider: str


def build_summary_middlewares():
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


@router.post("/summary")
async def generate_summary(request: SummaryRequest):
    if not hasattr(router, "documents_store"):
        raise HTTPException(status_code=500, detail="文档存储未初始化")
    if request.doc_id not in router.documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = router.documents_store[request.doc_id]
    full_text = doc["data"]["full_text"]

    system_prompt = """你是专业的文档摘要专家。请为文档生成简洁的摘要。

要求：
1. 提取文档的核心观点和关键信息
2. 摘要应简洁明了，长度在200-500字
3. 使用要点形式组织内容
4. 生成3-5个相关问题建议"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请为以下文档生成摘要：\n\n{full_text[:8000]}"}
    ]

    try:
        endpoint = PROVIDER_CONFIG.get(request.api_provider, {}).get("endpoint", "")
        middlewares = build_summary_middlewares()
        response = await call_ai_api(
            messages,
            request.api_key,
            request.model,
            request.api_provider,
            endpoint=endpoint,
            middlewares=middlewares
        )
        summary = response["choices"][0]["message"]["content"]

        questions_messages = [
            {
                "role": "system",
                "content": "根据文档内容，生成5个有价值的问题。只输出问题列表，每行一个问题。"
            },
            {"role": "user", "content": f"文档内容：\n{full_text[:4000]}"}
        ]

        questions_response = await call_ai_api(
            questions_messages,
            request.api_key,
            request.model,
            request.api_provider,
            endpoint=endpoint,
            middlewares=middlewares
        )
        suggested_questions = questions_response["choices"][0]["message"]["content"].split("\n")
        suggested_questions = [q.strip() for q in suggested_questions if q.strip()]

        return {
            "summary": summary,
            "suggested_questions": suggested_questions[:5],
            "doc_id": request.doc_id,
            "timestamp": datetime.now().isoformat(),
            "used_provider": response.get("_used_provider"),
            "used_model": response.get("_used_model"),
            "fallback_used": response.get("_fallback_used", False)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"摘要生成失败: {str(e)}")
