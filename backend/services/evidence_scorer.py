"""证据级 LLM 评分（paper-qa 风格 RCS：Retrieve → Compress → Score）。

设计目标：
- 每条证据输出 summary + relevance_score(0-10)
- 不相关时 summary 留空且 score=0，避免硬写摘要
- exact table / cost-anchor 证据直通，不经打分
- 超时或失败整体降级，不阻塞 Agent
- 候选不足时跳过打分，节省 LLM 往返
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

from services.structured_json import parse_json_object

logger = logging.getLogger(__name__)

EXACT_BYPASS_CHUNK_TYPES = frozenset(
    {"table", "table_row", "table_cell", "caption"}
)
EXACT_BYPASS_MARKERS = (
    "[structured table",
    "numeric_table_exact",
    "table_row_boundary",
    "table_row_raw_text",
    "table_row_evidence",
    "cost-anchor",
    "cost_anchor",
)

SUMMARY_SYSTEM_PROMPT = (
    "Provide a summary of the relevant information that could help answer the "
    "question based on the excerpt. Your summary, combined with many others, "
    "will be given to the model to generate an answer. Respond with the following "
    "JSON format:\n\n"
    '{{\n  "summary": "...",\n  "relevance_score": 0-10\n}}\n\n'
    "where `summary` is relevant information from the text - {summary_length} words. "
    "`relevance_score` is an integer 0-10 for the relevance of `summary` to the question.\n\n"
    "The excerpt may or may not contain relevant information. "
    "If not, leave `summary` empty, and make `relevance_score` be 0."
)

SUMMARY_USER_PROMPT = (
    "Excerpt from {citation}\n\n---\n\n{text}\n\n---\n\nQuestion: {question}"
)

# The cache is intentionally invalidated when either scoring prompt or the
# completion contract changes.  Before completion outcomes were enforced,
# syntactically valid JSON produced with ``finish_reason=length`` could have
# been cached as a normal score; keep those entries outside the new key space.
EVIDENCE_SCORE_CACHE_CONTRACT_VERSION = "completion-outcome-v1"
EVIDENCE_SCORE_PROMPT_VERSION = hashlib.sha256(
    (
        EVIDENCE_SCORE_CACHE_CONTRACT_VERSION
        + "\n"
        + SUMMARY_SYSTEM_PROMPT
        + "\n"
        + SUMMARY_USER_PROMPT
    ).encode("utf-8")
).hexdigest()[:16]

DEFAULT_SUMMARY_LENGTH = "about 40"
DEFAULT_EVIDENCE_K = 10
DEFAULT_ANSWER_MAX_SOURCES = 8
DEFAULT_HIGH_SCORE = 7
DEFAULT_MID_SCORE = 4
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MIN_CANDIDATES = 3
DEFAULT_MAX_CONCURRENCY = 5
MAX_EVIDENCE_CHARS = 2200


@dataclass
class ScoredEvidence:
    """One scored evidence unit ready for sufficiency / context assembly."""

    evidence_id: str
    text: str
    summary: str = ""
    relevance_score: int = 0
    bypass: bool = False
    source_kind: str = "search"  # search | fetch
    group_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["meta"] = dict(self.meta or {})
        return payload


def is_exact_evidence_bypass(
    text: str = "",
    meta: Optional[dict] = None,
) -> bool:
    """Return True when the unit must skip LLM scoring (numeric grounding)."""
    meta = meta if isinstance(meta, dict) else {}
    chunk_type = str(meta.get("chunk_type") or "").strip().lower()
    if chunk_type in EXACT_BYPASS_CHUNK_TYPES:
        return True
    if any(meta.get(key) not in (None, "", [], {}) for key in (
        "table_id",
        "table_bundle_id",
        "evidence_unit_id",
        "numeric_table_exact_context_row_text",
        "table_row_boundary_text",
        "table_row_raw_text",
        "table_row_evidence",
    )):
        return True
    haystack = f"{text}\n{meta}".lower()
    return any(marker in haystack for marker in EXACT_BYPASS_MARKERS)


def _passage_identity_token(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text or text.casefold() in {"none", "null"}:
        return ""
    return text


def evidence_identity(
    text: str,
    *,
    meta: Optional[dict] = None,
    group_id: str = "",
    fallback_scope: str = "",
) -> str:
    """Passage identity for scoring. Prefer chunk ids over layout/section ids.

    Citation anchors often stamp the same block_id / context_id onto every
    Methods hit on a page. Using those as exclusive identity makes collect +
    score-tier logic treat a whole section as one intro snippet.
    """
    meta = meta if isinstance(meta, dict) else {}
    for key in (
        "chunk_id",
        "child_chunk_id",
        "evidence_unit_id",
        "evidence_id",
    ):
        value = _passage_identity_token(meta.get(key))
        if value:
            return f"{key}:{value}"
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    if normalized:
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        return f"text:{digest}"
    gid = _passage_identity_token(group_id or meta.get("group_id"))
    if gid:
        return f"group:{gid}"
    return f"scope:{fallback_scope or 'unknown'}"


def _evidence_cache_enabled() -> bool:
    try:
        from services.evidence_cache import is_cache_enabled
        return is_cache_enabled()
    except Exception:
        return False


def _clamp_score(value: Any) -> int:
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, score))


def _parse_score_payload(content: Any) -> tuple[str, int]:
    try:
        parsed = parse_json_object(content, allow_partial=False)
    except Exception:
        # 不能从损坏的 JSON 里猜第一个数字。它可能来自摘要正文，而不是
        # relevance_score；失败时返回 0，让原始检索分数承担排序职责。
        return "", 0

    summary = str(parsed.get("summary") or "").strip()
    score = _clamp_score(parsed.get("relevance_score"))
    if score <= 0:
        return "", 0
    return re.sub(r"\s+", " ", summary).strip()[:600], score


def _response_content(response: Any) -> str:
    if not isinstance(response, dict):
        return str(response or "")
    if response.get("error"):
        return ""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


async def score_single_evidence(
    *,
    question: str,
    text: str,
    citation: str = "document",
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    summary_length: str = DEFAULT_SUMMARY_LENGTH,
    call_ai_api: Optional[Callable[..., Any]] = None,
) -> tuple[str, int]:
    """Score one evidence unit; returns (summary, relevance_score)."""
    excerpt = re.sub(r"\s+", " ", str(text or "")).strip()
    if not excerpt or not api_key:
        return "", 0
    excerpt = excerpt[:MAX_EVIDENCE_CHARS]
    if call_ai_api is None:
        from services.chat_service import call_ai_api as _call_ai_api
        call_ai_api = _call_ai_api

    system = SUMMARY_SYSTEM_PROMPT.format(summary_length=summary_length or DEFAULT_SUMMARY_LENGTH)
    user = SUMMARY_USER_PROMPT.format(
        citation=citation or "document",
        text=excerpt,
        question=str(question or "").strip(),
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        from services.completion_outcome import require_publishable_completion

        response = await call_ai_api(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint or "",
            max_tokens=220,
            temperature=0.0,
            purpose="evidence_score",
        )
        try:
            require_publishable_completion(response, operation="evidence score")
        except Exception:
            response = None
        summary, score = _parse_score_payload(_response_content(response))
        if summary or score > 0:
            return summary, score
        # One retry with the parse failure hint, matching paper-qa behaviour.
        retry_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Previous response was not valid JSON. "
                    'Reply only with {"summary":"...","relevance_score":0-10}.'
                ),
            }
        ]
        response = await call_ai_api(
            messages=retry_messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint or "",
            max_tokens=220,
            temperature=0.0,
            purpose="evidence_score_retry",
        )
        require_publishable_completion(response, operation="evidence score retry")
        return _parse_score_payload(_response_content(response))
    except Exception as exc:
        logger.debug("[EvidenceScorer] single score failed: %s", exc)
        return "", 0


def collect_score_candidates(
    *,
    search_results: Sequence[str],
    fetched_content: Optional[dict[str, dict]] = None,
    extract_meta: Optional[Callable[[str], dict]] = None,
    evidence_k: int = DEFAULT_EVIDENCE_K,
) -> list[ScoredEvidence]:
    """Build a bounded candidate list from search hits and fetched groups."""
    candidates: list[ScoredEvidence] = []
    seen: set[str] = set()
    fetched_content = fetched_content or {}

    for idx, chunk in enumerate(search_results or []):
        text = str(chunk or "").strip()
        if not text:
            continue
        meta = extract_meta(text) if callable(extract_meta) else {}
        meta = meta if isinstance(meta, dict) else {}
        body = str(meta.get("text") or text or "").strip()
        evidence_id = evidence_identity(body, meta=meta, fallback_scope=f"search:{idx}")
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        candidates.append(
            ScoredEvidence(
                evidence_id=evidence_id,
                text=body or text,
                bypass=is_exact_evidence_bypass(body or text, meta),
                source_kind="search",
                group_id=str(meta.get("group_id") or ""),
                meta=dict(meta),
            )
        )
        if len(candidates) >= max(1, int(evidence_k or DEFAULT_EVIDENCE_K)):
            break

    remaining = max(0, max(1, int(evidence_k or DEFAULT_EVIDENCE_K)) - len(candidates))
    for group_id, data in list((fetched_content or {}).items())[: remaining + 4]:
        if remaining <= 0:
            break
        payload = data if isinstance(data, dict) else {}
        text = str(payload.get("text") or "").strip()
        if not text:
            continue
        meta = {
            "group_id": group_id,
            "context_id": payload.get("context_id") or group_id,
            "evidence_id": payload.get("evidence_id") or f"{group_id}:fetch",
            "page_range": payload.get("page_range") or [],
            "granularity": payload.get("granularity") or "digest",
        }
        evidence_id = evidence_identity(text, meta=meta, group_id=str(group_id), fallback_scope=f"fetch:{group_id}")
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        candidates.append(
            ScoredEvidence(
                evidence_id=evidence_id,
                text=text,
                bypass=is_exact_evidence_bypass(text, meta),
                source_kind="fetch",
                group_id=str(group_id),
                meta=meta,
            )
        )
        remaining -= 1

    return candidates


async def score_evidence_batch(
    *,
    question: str,
    candidates: Sequence[ScoredEvidence],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
    call_ai_api: Optional[Callable[..., Any]] = None,
    doc_id: str = "",
    parse_generation: str = "",
    document_source_hash: str = "",
    block_index_hash: str = "",
) -> dict[str, Any]:
    """Score a batch of candidates with total timeout and exact-bypass protection.

    Returns a diagnostics-friendly dict:
    {
      "applied": bool,
      "reason": str,
      "scored": list[ScoredEvidence],
      "by_id": dict[str, ScoredEvidence],
      "high_score_count": int,
      "elapsed_ms": float,
    }
    """
    items = [item for item in candidates if isinstance(item, ScoredEvidence)]
    result: dict[str, Any] = {
        "applied": False,
        "reason": "",
        "scored": [],
        "by_id": {},
        "high_score_count": 0,
        "mid_score_count": 0,
        "dropped_count": 0,
        "bypass_count": 0,
        "cache_hits": 0,
        "elapsed_ms": 0.0,
        "timeout": False,
    }
    if not items:
        result["reason"] = "empty_candidates"
        return result
    if not api_key or not model:
        result["reason"] = "missing_credentials"
        result["scored"] = list(items)
        result["by_id"] = {item.evidence_id: item for item in items}
        return result

    scoreable = [item for item in items if not item.bypass]
    bypassed = [item for item in items if item.bypass]
    result["bypass_count"] = len(bypassed)
    # Exact units always keep full text and a synthetic high score for gates.
    for item in bypassed:
        item.relevance_score = 10
        item.summary = ""
        item.bypass = True

    # 缓存复用：同一 (解析身份, 问题, 证据) 已打过分就不再花一次 LLM。
    # 命中的条目直接带着 summary/score 出列，剩下的才进入打分池。
    cached_items: list[ScoredEvidence] = []
    if _evidence_cache_enabled():
        from services.evidence_cache import get_evidence_cache, make_cache_key

        cache = get_evidence_cache()
        still_scoreable: list[ScoredEvidence] = []
        for item in scoreable:
            key = make_cache_key(
                question=question,
                evidence_id=item.evidence_id,
                doc_id=doc_id,
                parse_generation=parse_generation,
                document_source_hash=document_source_hash,
                block_index_hash=block_index_hash,
                evidence_text=item.text,
                provider=provider,
                model=model,
                endpoint=endpoint,
                prompt_version=EVIDENCE_SCORE_PROMPT_VERSION,
            )
            hit = cache.get(key) if key else None
            if hit is None:
                item.meta["_score_cache_key"] = key
                still_scoreable.append(item)
                continue
            item.summary, item.relevance_score = hit[0], hit[1]
            item.meta["score_from_cache"] = True
            cached_items.append(item)
        scoreable = still_scoreable
        result["cache_hits"] = len(cached_items)
        if cached_items:
            logger.debug(
                "[EvidenceScorer] 证据评分缓存命中 %d/%d", len(cached_items),
                len(cached_items) + len(scoreable),
            )

    if not scoreable and cached_items:
        # 全部命中缓存：零 LLM 调用就能给出完整结果
        merged = cached_items + bypassed
        merged.sort(key=lambda item: (0 if item.bypass else 1, -int(item.relevance_score or 0)))
        result["applied"] = True
        result["reason"] = "all_cached"
        result["scored"] = merged
        result["by_id"] = {item.evidence_id: item for item in merged}
        result["high_score_count"] = sum(
            1 for item in merged
            if item.bypass or int(item.relevance_score or 0) >= DEFAULT_HIGH_SCORE
        )
        return result

    if len(scoreable) < max(1, int(min_candidates or DEFAULT_MIN_CANDIDATES)):
        # Keep bypass units but skip LLM when the non-exact pool is too small.
        merged = list(scoreable) + cached_items + bypassed
        result["scored"] = merged
        result["by_id"] = {item.evidence_id: item for item in merged}
        result["reason"] = "below_min_candidates"
        result["high_score_count"] = sum(1 for item in merged if item.relevance_score >= DEFAULT_HIGH_SCORE or item.bypass)
        return result

    started = time.perf_counter()
    semaphore = asyncio.Semaphore(max(1, min(int(max_concurrency or DEFAULT_MAX_CONCURRENCY), 5)))

    async def _score_one(item: ScoredEvidence) -> ScoredEvidence:
        async with semaphore:
            citation = (
                str(item.meta.get("group_id") or item.group_id or item.meta.get("page") or "document")
            )
            summary, score = await score_single_evidence(
                question=question,
                text=item.text,
                citation=citation,
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                call_ai_api=call_ai_api,
            )
            item.summary = summary
            item.relevance_score = score
            cache_key = item.meta.pop("_score_cache_key", "")
            if cache_key and (summary or score > 0):
                # 只缓存有内容的结果：空摘要+0分可能是这次调用失败，
                # 缓存下来会把一次偶发失败变成长期错误。
                from services.evidence_cache import get_evidence_cache

                get_evidence_cache().put(cache_key, summary, score)
            return item

    try:
        scored_items = await asyncio.wait_for(
            asyncio.gather(*[_score_one(item) for item in scoreable], return_exceptions=False),
            timeout=max(0.05, float(timeout_s or DEFAULT_TIMEOUT_S)),
        )
    except asyncio.TimeoutError:
        result["timeout"] = True
        result["reason"] = "scoring_timeout"
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        # Fail open: return original candidates without applied scoring.
        result["scored"] = list(scoreable) + cached_items + bypassed
        result["by_id"] = {item.evidence_id: item for item in result["scored"]}
        logger.info("[EvidenceScorer] batch timed out after %.1fs; falling back", timeout_s)
        return result
    except Exception as exc:
        result["reason"] = f"scoring_error:{type(exc).__name__}"
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["scored"] = list(scoreable) + cached_items + bypassed
        result["by_id"] = {item.evidence_id: item for item in result["scored"]}
        logger.debug("[EvidenceScorer] batch failed: %s", exc)
        return result

    merged = list(scored_items) + cached_items + bypassed
    # Prefer high scores first while preserving bypass units near the front.
    merged.sort(
        key=lambda item: (
            0 if item.bypass else 1,
            -int(item.relevance_score or 0),
        )
    )
    high = sum(1 for item in merged if item.bypass or int(item.relevance_score or 0) >= DEFAULT_HIGH_SCORE)
    mid = sum(
        1
        for item in merged
        if (not item.bypass)
        and DEFAULT_MID_SCORE <= int(item.relevance_score or 0) < DEFAULT_HIGH_SCORE
    )
    dropped = sum(
        1
        for item in merged
        if (not item.bypass) and int(item.relevance_score or 0) < DEFAULT_MID_SCORE
    )
    result.update(
        {
            "applied": True,
            "reason": "scored",
            "scored": merged,
            "by_id": {item.evidence_id: item for item in merged},
            "high_score_count": high,
            "mid_score_count": mid,
            "dropped_count": dropped,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    )
    return result


def tier_evidence_for_context(
    scored: Sequence[ScoredEvidence],
    *,
    high_score: int = DEFAULT_HIGH_SCORE,
    mid_score: int = DEFAULT_MID_SCORE,
    answer_max_sources: int = DEFAULT_ANSWER_MAX_SOURCES,
) -> list[tuple[str, ScoredEvidence]]:
    """Apply score tiers for final context consumption.

    - bypass / score >= high: full text
    - mid <= score < high: summary only (fallback to truncated text)
    - score < mid: drop
    """
    selected: list[tuple[str, ScoredEvidence]] = []
    limit = max(1, int(answer_max_sources or DEFAULT_ANSWER_MAX_SOURCES))
    for item in scored:
        if len(selected) >= limit:
            break
        score = int(item.relevance_score or 0)
        if item.bypass or score >= high_score:
            text = str(item.text or "").strip()
            if text:
                selected.append((text, item))
            continue
        if score >= mid_score:
            summary = str(item.summary or "").strip()
            text = summary or re.sub(r"\s+", " ", str(item.text or "")).strip()[:500]
            if text:
                # Keep provenance header if the original had one.
                selected.append((f"[相关摘要 score={score}]\n{text}", item))
            continue
        # dropped
    return selected


def sufficiency_from_scores(
    scored: Sequence[ScoredEvidence],
    *,
    high_score: int = DEFAULT_HIGH_SCORE,
    min_high: int = 2,
    min_sources: int = 2,
) -> dict[str, Any]:
    """Derive a score-weighted sufficiency hint for the agent gate."""
    high_items = [
        item
        for item in scored
        if item.bypass or int(item.relevance_score or 0) >= high_score
    ]
    mid_or_better = [
        item
        for item in scored
        if item.bypass or int(item.relevance_score or 0) >= DEFAULT_MID_SCORE
    ]
    high_count = len(high_items)
    independent = len({item.evidence_id for item in mid_or_better})
    if high_count >= max(1, min_high) and independent >= max(1, min_sources):
        level = "sufficient"
    elif high_count >= 1 or independent >= 1:
        level = "maybe_sufficient"
    else:
        level = "insufficient"
    return {
        "level": level,
        "high_score_count": high_count,
        "independent_scored_count": independent,
        "scored_count": len(list(scored or [])),
        "min_high": max(1, min_high),
        "score_threshold": high_score,
    }
