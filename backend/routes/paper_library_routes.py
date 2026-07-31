"""Local-only paper library subscription API."""

from __future__ import annotations

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.paper_library_service import PaperLibraryService
from services.paper_subscription_discovery_service import discover_subscription_papers
from config import settings

router = APIRouter(prefix="/paper-library", tags=["paper-library"])
paper_library_service: PaperLibraryService | None = None
documents_store: dict = {}


def _service() -> PaperLibraryService:
    if paper_library_service is None:
        raise HTTPException(status_code=503, detail="论文库服务未初始化")
    return paper_library_service


class SubscriptionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    query: str = Field(..., min_length=1, max_length=500)
    keywords: list[str] = Field(default_factory=list, max_length=64)
    enabled: bool = True


class SubscriptionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    query: Optional[str] = Field(default=None, min_length=1, max_length=500)
    keywords: Optional[list[str]] = Field(default=None, max_length=64)
    enabled: Optional[bool] = None


class InterestFeedback(BaseModel):
    subscription_id: str = Field(..., min_length=1, max_length=80)
    paper_id: str = Field(..., min_length=1, max_length=80)
    relevance: Literal["relevant", "not_relevant"]
    novelty: Literal["new", "known", "unsure"]
    reason_codes: list[str] = Field(default_factory=list, max_length=12)


@router.get("/subscriptions")
async def list_subscriptions():
    return {"subscriptions": _service().list_subscriptions()}


@router.post("/subscriptions")
async def create_subscription(request: SubscriptionCreate):
    try:
        return _service().create_subscription(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.patch("/subscriptions/{subscription_id}")
async def update_subscription(subscription_id: str, request: SubscriptionUpdate):
    try:
        result = _service().update_subscription(
            subscription_id,
            request.model_dump(exclude_none=True),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail="订阅不存在")
    return result


@router.delete("/subscriptions/{subscription_id}")
async def delete_subscription(subscription_id: str):
    if not _service().delete_subscription(subscription_id):
        raise HTTPException(status_code=404, detail="订阅不存在")
    return {"deleted": True}


@router.post("/feedback")
async def record_interest_feedback(request: InterestFeedback):
    try:
        return _service().record_feedback(**request.model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="订阅不存在")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/process-new")
async def process_new_papers():
    return _service().process_new_documents(documents_store)


@router.post("/refresh")
async def refresh_subscriptions(limit_per_subscription: int = 20):
    """Explicit network refresh; never runs as part of chat or document upload."""
    subscriptions = [item for item in _service().list_subscriptions() if item.get("enabled")]
    discoveries = await asyncio.gather(*[
        discover_subscription_papers(
            str(subscription.get("query") or ""),
            semantic_scholar_api_key=settings.paper_metadata_semantic_scholar_api_key,
            limit=limit_per_subscription,
            timeout_seconds=settings.paper_metadata_hydration_timeout_seconds,
        )
        for subscription in subscriptions
    ])
    candidates: dict[str, dict] = {}
    diagnostics = []
    for subscription, discovery in zip(subscriptions, discoveries):
        diagnostics.append({
            "subscription_id": subscription.get("subscription_id"),
            "providers": discovery.get("providers") or {},
            "candidate_count": len(discovery.get("candidates") or []),
        })
        for candidate in discovery.get("candidates") or []:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("metadata"), dict):
                continue
            candidate_id = str(candidate.get("candidate_id") or "")
            if candidate_id:
                candidates[candidate_id] = {
                    "filename": candidate_id,
                    "paper_metadata": candidate["metadata"],
                    "data": {"parse_manifest": {}},
                }
    result = _service().process_new_documents(candidates)
    return {
        **result,
        "refresh": "explicit_network",
        "diagnostics": diagnostics,
    }


@router.get("/feed")
async def list_feed(subscription_id: str = "", limit: int = 50):
    return {"items": _service().list_feed(subscription_id=subscription_id, limit=limit)}


@router.delete("/data")
async def clear_paper_library():
    _service().clear()
    return {"cleared": True}
