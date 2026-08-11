"""将联网搜索结果转换为有边界、与查询相关的正文证据摘录。

搜索引擎只负责发现链接，标题和短摘要不足以支撑可靠回答。本模块通过现有
URL 读取器并发读取少量公开页面，选择与实际查询匹配的段落；单个来源失败
或正文离题时保留原搜索摘要，不影响其他来源。
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import ipaddress
import re
import time
from typing import Any
from urllib.parse import urlparse

from services import url_loader_service


_MAX_FETCH_SOURCES = 3
_PER_SOURCE_CHAR_BUDGET = 2400
_TOTAL_CHAR_BUDGET = 6400
_RAW_CONTENT_LIMIT = 200_000
_CACHE_TTL_SECONDS = 15 * 60
_CACHE_MAX_ITEMS = 64
_MIN_PASSAGE_CHARS = 48
_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "of", "on", "or", "the", "to", "via", "what", "when", "where",
    "which", "who", "why", "with",
}
_RAW_CONTENT_CACHE: OrderedDict[str, tuple[float, dict]] = OrderedDict()


def _safe_public_url(value: Any) -> str:
    """只接受公开 HTTP(S) URL，拒绝本机、内网和保留地址。"""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""
    hostname = parsed.hostname.strip(".").casefold()
    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
        or hostname.endswith(".internal")
    ):
        return ""
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return raw
    if not address.is_global:
        return ""
    return raw


def _tokens(text: Any) -> set[str]:
    source = str(text or "").casefold()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]{2,}", source)
        if token not in _ENGLISH_STOPWORDS
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", source):
        tokens.add(sequence)
        tokens.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return tokens


def _compact(text: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(text or "").casefold())


def _quoted_anchors(query: str) -> list[str]:
    return [
        match.strip()
        for match in re.findall(r'"([^"\r\n]{6,})"', str(query or ""))
        if match.strip()
    ]


def _matches_quoted_anchor(query: str, passage: str) -> bool:
    anchors = _quoted_anchors(query)
    if not anchors:
        return True
    compact_passage = _compact(passage)
    passage_tokens = _tokens(passage)
    for anchor in anchors:
        compact_anchor = _compact(anchor)
        if compact_anchor and compact_anchor in compact_passage:
            return True
        anchor_tokens = _tokens(anchor)
        if not anchor_tokens:
            continue
        overlap = len(anchor_tokens & passage_tokens)
        if overlap >= max(2, (len(anchor_tokens) * 3 + 3) // 4):
            return True
    return False


def _passage_score(query: str, passage: str) -> float:
    if not _matches_quoted_anchor(query, passage):
        return 0.0
    query_tokens = _tokens(query)
    passage_tokens = _tokens(passage)
    if not query_tokens or not passage_tokens:
        return 0.0
    overlap = len(query_tokens & passage_tokens)
    coverage = overlap / max(1, len(query_tokens))
    density = overlap / max(1, min(len(passage_tokens), 80))
    score = coverage * 0.82 + density * 0.18
    compact_passage = _compact(passage)
    if any(_compact(anchor) in compact_passage for anchor in _quoted_anchors(query)):
        score += 0.45
    return score


def _candidate_passages(content: str) -> list[tuple[int, str]]:
    normalized = str(content or "").replace("\x00", "")[:_RAW_CONTENT_LIMIT]
    blocks = re.split(r"\n\s*\n+", normalized)
    passages: list[tuple[int, str]] = []
    order = 0
    for block in blocks:
        cleaned = re.sub(r"[ \t]+", " ", block).strip()
        if len(cleaned) < _MIN_PASSAGE_CHARS:
            continue
        # 将超长段落切成有重叠的窗口，避免导航或转录文本吃掉单来源预算。
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + 1200)
            if end < len(cleaned):
                boundary = max(cleaned.rfind("。", start, end), cleaned.rfind(". ", start, end))
                if boundary > start + 500:
                    end = boundary + 1
            window = cleaned[start:end].strip()
            if len(window) >= _MIN_PASSAGE_CHARS:
                passages.append((order, window))
                order += 1
            if end >= len(cleaned):
                break
            start = max(end - 120, start + 1)
    return passages


def select_relevant_excerpt(query: str, content: str, *, max_chars: int = _PER_SOURCE_CHAR_BUDGET) -> str:
    """选择少量高相关段落，再按原网页顺序拼接。"""
    candidates = _candidate_passages(content)
    if not candidates or max_chars <= 0:
        return ""
    ranked = sorted(
        (( _passage_score(query, passage), order, passage) for order, passage in candidates),
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    )
    selected = [item for item in ranked[:5] if item[0] > 0]
    if not selected:
        return ""
    selected.sort(key=lambda item: item[1])
    parts: list[str] = []
    used = 0
    for _score, _order, passage in selected:
        remaining = max_chars - used
        if remaining <= 0:
            break
        piece = passage[:remaining].rstrip()
        if not piece:
            continue
        parts.append(piece)
        used += len(piece) + 2
    return "\n\n".join(parts).strip()


def _cache_get(url: str) -> dict | None:
    cached = _RAW_CONTENT_CACHE.get(url)
    if not cached:
        return None
    created_at, payload = cached
    if time.monotonic() - created_at > _CACHE_TTL_SECONDS:
        _RAW_CONTENT_CACHE.pop(url, None)
        return None
    _RAW_CONTENT_CACHE.move_to_end(url)
    return dict(payload)


def _cache_put(url: str, payload: dict) -> None:
    _RAW_CONTENT_CACHE[url] = (time.monotonic(), dict(payload))
    _RAW_CONTENT_CACHE.move_to_end(url)
    while len(_RAW_CONTENT_CACHE) > _CACHE_MAX_ITEMS:
        _RAW_CONTENT_CACHE.popitem(last=False)


async def _read_url(url: str, timeout_seconds: float) -> dict:
    cached = _cache_get(url)
    if cached is not None:
        return cached
    payload = await url_loader_service.fetch_url_content(url, timeout=timeout_seconds)
    bounded = {
        "title": str(payload.get("title") or "")[:500],
        "content": str(payload.get("content") or "")[:_RAW_CONTENT_LIMIT],
        "url": url,
    }
    _cache_put(url, bounded)
    return bounded


def _already_has_substantial_content(item: dict) -> bool:
    snippet = str(item.get("snippet") or "").strip()
    return len(snippet) >= 900


def _is_academic_metadata(item: dict) -> bool:
    return str(item.get("snippet") or "").lstrip().casefold().startswith("metadata provider:")


async def enrich_web_results(
    query: str,
    results: list[dict],
    *,
    max_sources: int = _MAX_FETCH_SOURCES,
    timeout_seconds: float = 8.0,
    per_source_chars: int = _PER_SOURCE_CHAR_BUDGET,
    total_chars: int = _TOTAL_CHAR_BUDGET,
) -> tuple[list[dict], dict]:
    """读取候选网页，仅在正文足够相关时替换搜索摘要。"""
    enriched = [dict(item) for item in (results or []) if isinstance(item, dict)]
    diagnostic = {
        "attempted": 0,
        "fetched": 0,
        "enriched": 0,
        "failed": 0,
        "skipped_private": 0,
        "irrelevant_content": 0,
    }
    eligible: list[tuple[int, str]] = []
    for index, item in enumerate(enriched):
        item.setdefault("evidence_type", "search_snippet")
        if _is_academic_metadata(item):
            item["evidence_type"] = "academic_metadata"
            continue
        if _already_has_substantial_content(item):
            item["evidence_type"] = "provider_content"
            continue
        url = _safe_public_url(item.get("url"))
        if not url:
            if item.get("url"):
                diagnostic["skipped_private"] += 1
                item["content_status"] = "unsafe_url"
            continue
        if len(eligible) < max(0, int(max_sources)):
            eligible.append((index, url))

    diagnostic["attempted"] = len(eligible)
    if not eligible:
        return enriched, diagnostic

    tasks = [_read_url(url, timeout_seconds) for _index, url in eligible]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    remaining_total = max(0, int(total_chars))

    for (index, _url), outcome in zip(eligible, outcomes):
        item = enriched[index]
        if isinstance(outcome, BaseException):
            diagnostic["failed"] += 1
            item["content_status"] = "fetch_failed"
            continue
        diagnostic["fetched"] += 1
        body = str(outcome.get("content") or "")
        excerpt_budget = min(max(0, int(per_source_chars)), remaining_total)
        excerpt = select_relevant_excerpt(query, body, max_chars=excerpt_budget)
        original_snippet = str(item.get("snippet") or "").strip()
        excerpt_score = _passage_score(query, excerpt)
        snippet_score = _passage_score(query, original_snippet)
        if not excerpt or excerpt_score < max(0.055, snippet_score * 0.7):
            diagnostic["irrelevant_content"] += 1
            item["content_status"] = "irrelevant_content"
            continue
        if original_snippet:
            item["search_snippet"] = original_snippet[:600]
        item["snippet"] = excerpt
        item["evidence_type"] = "webpage_excerpt"
        item["content_status"] = "fetched"
        item["content_chars"] = len(excerpt)
        diagnostic["enriched"] += 1
        remaining_total = max(0, remaining_total - len(excerpt))

    return enriched, diagnostic
