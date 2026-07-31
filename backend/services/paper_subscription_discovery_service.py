"""Fail-open metadata discovery for explicit paper-library refreshes."""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

import httpx


def _text(value: Any, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _candidate_id(provider: str, metadata: dict[str, Any]) -> str:
    stable = str(metadata.get("doi") or metadata.get("arxiv_id") or metadata.get("title") or "")
    return f"external:{provider}:{hashlib.sha1(stable.casefold().encode('utf-8')).hexdigest()[:20]}"


def _crossref_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    message = payload.get("message") if isinstance(payload, dict) else {}
    items = message.get("items") if isinstance(message, dict) else []
    candidates = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        title = _text((item.get("title") or [""])[0], 300)
        if not title:
            continue
        authors = []
        for author in item.get("author") or []:
            if isinstance(author, dict):
                name = _text(" ".join(filter(None, (author.get("given"), author.get("family")))), 120)
                if name:
                    authors.append(name)
        date_parts = (item.get("published-print") or item.get("published-online") or {}).get("date-parts") or []
        try:
            year = int(date_parts[0][0])
        except (TypeError, ValueError, IndexError):
            year = None
        metadata = {
            "title": title,
            "authors": authors[:20],
            "year": year,
            "doi": _text(item.get("DOI"), 300).lower(),
            "venue": _text((item.get("container-title") or [""])[0], 200),
            "external_url": _text(item.get("URL"), 1200),
            "discovery_provider": "crossref",
        }
        candidates.append({
            "candidate_id": _candidate_id("crossref", metadata),
            "metadata": metadata,
        })
    return candidates


def _s2_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        title = _text(item.get("title"), 300)
        if not title:
            continue
        external_ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        metadata = {
            "title": title,
            "authors": [
                _text(author.get("name"), 120)
                for author in (item.get("authors") or [])
                if isinstance(author, dict) and _text(author.get("name"), 120)
            ][:20],
            "year": int(item.get("year")) if str(item.get("year") or "").isdigit() else None,
            "doi": _text(external_ids.get("DOI"), 300).lower(),
            "arxiv_id": _text(external_ids.get("ArXiv"), 120),
            "venue": _text(item.get("venue"), 200),
            "abstract_preview": _text(item.get("abstract"), 1200),
            "external_url": _text(item.get("url"), 1200),
            "discovery_provider": "semantic_scholar",
        }
        candidates.append({
            "candidate_id": _candidate_id("semantic_scholar", metadata),
            "metadata": metadata,
        })
    return candidates


async def discover_subscription_papers(
    query: str,
    *,
    semantic_scholar_api_key: str = "",
    limit: int = 20,
    timeout_seconds: float = 8.0,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Search two metadata providers in parallel after an explicit refresh."""
    query = _text(query, 500)
    if not query:
        return {"candidates": [], "providers": {}, "error": "empty_query"}
    bounded_limit = max(1, min(int(limit or 20), 50))
    owns_client = client is None
    active_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(max(1.0, float(timeout_seconds or 8.0))),
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": "ChatPDF/2 paper-library"},
    )
    provider_status: dict[str, dict[str, Any]] = {}

    async def crossref():
        response = await active_client.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": query, "rows": bounded_limit},
        )
        response.raise_for_status()
        return _crossref_candidates(response.json())

    async def semantic_scholar():
        headers = {"x-api-key": semantic_scholar_api_key} if semantic_scholar_api_key else None
        response = await active_client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": bounded_limit,
                "fields": "title,authors,year,venue,abstract,externalIds,url",
            },
            headers=headers,
        )
        response.raise_for_status()
        return _s2_candidates(response.json())

    async def run(name: str, call):
        try:
            values = await call()
            provider_status[name] = {"status": "ok", "candidate_count": len(values)}
            return values
        except Exception as exc:
            provider_status[name] = {"status": "failed", "error": type(exc).__name__}
            return []

    try:
        batches = await asyncio.gather(
            run("crossref", crossref),
            run("semantic_scholar", semantic_scholar),
        )
    finally:
        if owns_client:
            await active_client.aclose()

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in [item for batch in batches for item in batch]:
        metadata = candidate.get("metadata") if isinstance(candidate, dict) else None
        if not isinstance(metadata, dict):
            continue
        key = str(metadata.get("doi") or metadata.get("arxiv_id") or "").casefold()
        if not key:
            key = re.sub(r"\W+", "", str(metadata.get("title") or "").casefold())
        if not key:
            continue
        current = deduped.get(key)
        if current is None or len(metadata) > len(current.get("metadata") or {}):
            deduped[key] = candidate
    return {
        "query": query,
        "candidates": list(deduped.values())[:bounded_limit],
        "providers": provider_status,
    }
