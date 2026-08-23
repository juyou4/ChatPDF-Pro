"""同会话多文档 fan-out 聚合。

不改动单文档索引结构：对每个 doc 独立取上下文，再按文档前缀合并，
供跨文档比较 / 综述类问题使用。完整 Agent 仍默认单文档；本模块提供
确定性聚合层，作为 paper-qa 跨文档能力的轻量第一步。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

RetrieverFn = Callable[[str, str, str], Awaitable[dict[str, Any]]]
# retriever(doc_id, doc_name, question) -> {
#   "context": str,
#   "detail": list[dict],
#   "citations": list[dict],
#   "error": str?,
# }


@dataclass
class DocFanoutInput:
    doc_id: str
    doc_name: str = ""
    rank_hint: float = 0.0
    work_id: str = ""
    version_rank: int = 0


@dataclass
class DocFanoutResult:
    doc_id: str
    doc_name: str
    context: str = ""
    detail: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    error: str = ""
    char_count: int = 0
    work_id: str = ""
    version_rank: int = 0
    citation_authorization: dict[str, Any] = field(default_factory=dict)


def _safe_name(doc_id: str, doc_name: str = "") -> str:
    name = re.sub(r"\s+", " ", str(doc_name or "")).strip()
    if name:
        return name[:80]
    return str(doc_id or "document")[:64]


def prefix_context_with_doc(doc_name: str, context: str) -> str:
    text = str(context or "").strip()
    if not text:
        return ""
    label = _safe_name("", doc_name) if doc_name else "document"
    return f"【文档: {label}】\n{text}"


def canonical_work_id(metadata: dict[str, Any] | None, *, fallback: str = "") -> str:
    """Return a stable paper-family identity without treating venue rank as truth."""
    data = metadata if isinstance(metadata, dict) else {}
    doi = str(data.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    arxiv_id = str(data.get("arxiv_id") or "").strip().lower()
    if arxiv_id:
        canonical_arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        return f"arxiv:{canonical_arxiv_id}"
    title = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(data.get("title") or "").lower())
    if len(title) >= 12:
        return f"title:{hashlib.sha1(title.encode('utf-8')).hexdigest()[:16]}"
    return f"doc:{str(fallback or '').strip()}" if fallback else ""


def document_version_rank(doc_name: str, metadata: dict[str, Any] | None = None) -> int:
    text = " ".join((
        str(doc_name or ""),
        str((metadata or {}).get("arxiv_id") or "") if isinstance(metadata, dict) else "",
    ))
    versions = [int(value) for value in re.findall(r"(?:^|[^a-z0-9])v(\d{1,3})(?!\d)", text, re.IGNORECASE)]
    return max(versions, default=0)


def deduplicate_document_versions(
    documents: Sequence[DocFanoutInput],
) -> tuple[list[DocFanoutInput], list[dict[str, Any]]]:
    """Keep the newest explicitly versioned copy of each canonical paper."""
    selected: dict[str, tuple[int, DocFanoutInput]] = {}
    order: list[str] = []
    skipped: list[dict[str, Any]] = []
    for position, doc in enumerate(documents or []):
        doc_id = str(getattr(doc, "doc_id", "") or "").strip()
        if not doc_id:
            continue
        work_id = str(getattr(doc, "work_id", "") or f"doc:{doc_id}").strip()
        current = selected.get(work_id)
        candidate_key = (
            int(getattr(doc, "version_rank", 0) or 0),
            float(getattr(doc, "rank_hint", 0.0) or 0.0),
            -position,
        )
        if current is None:
            selected[work_id] = (position, doc)
            order.append(work_id)
            continue
        previous_position, previous = current
        previous_key = (
            int(getattr(previous, "version_rank", 0) or 0),
            float(getattr(previous, "rank_hint", 0.0) or 0.0),
            -previous_position,
        )
        if candidate_key > previous_key:
            skipped.append({
                "doc_id": previous.doc_id,
                "kept_doc_id": doc.doc_id,
                "work_id": work_id,
                "reason": "older_version",
            })
            selected[work_id] = (position, doc)
        else:
            skipped.append({
                "doc_id": doc.doc_id,
                "kept_doc_id": previous.doc_id,
                "work_id": work_id,
                "reason": "older_version",
            })
    return [selected[key][1] for key in order], skipped


def _citation_text(item: dict[str, Any]) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(
            item.get("source_text")
            or item.get("context_segment_text")
            or item.get("display_text")
            or item.get("highlight_text")
            or item.get("text")
            or item.get("chunk")
            or ""
        ),
    ).strip()


def _namespaced_citation(
    item: dict[str, Any],
    *,
    doc_id: str,
    doc_name: str,
    ref: int,
) -> dict[str, Any]:
    entry = dict(item)
    namespace = f"doc:{doc_id}"
    original_id = ""
    for field in ("evidence_id", "chunk_id", "child_chunk_id", "block_id", "context_id"):
        value = entry.get(field)
        if value in (None, False, ""):
            continue
        token = str(value).strip()
        if token:
            original_id = token
            break
    if not original_id:
        original_id = hashlib.sha1(_citation_text(entry).encode("utf-8")).hexdigest()[:20]
    namespaced_id = f"{namespace}:{original_id}"
    entry.update({
        "ref": ref,
        "doc_id": doc_id,
        "doc_name": doc_name,
        "citation_namespace": namespace,
        "original_evidence_id": original_id,
        "evidence_id": namespaced_id,
    })
    if not entry.get("context_id"):
        entry["context_id"] = namespaced_id
    text = _citation_text(entry)
    entry.setdefault("source_text", text)
    entry.setdefault("display_text", text)
    entry.setdefault("highlight_text", text)
    entry.setdefault("context_segment_text", text)
    page = entry.get("page") or entry.get("page_number")
    if page and not entry.get("page_range"):
        entry["page_range"] = [page]
    return entry


_CONFLICT_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", re.IGNORECASE)
_CONFLICT_NUMBER_RE = re.compile(r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)?%?(?![a-z0-9])", re.IGNORECASE)
_CONFLICT_STOPWORDS = {
    "this", "that", "with", "from", "were", "was", "the", "and", "for", "result",
    "results", "table", "figure", "方法", "实验", "结果", "本文", "模型", "数据",
}


def group_potential_conflicts(citations: Sequence[dict], *, limit: int = 8) -> list[dict[str, Any]]:
    """Conservatively group cross-paper numeric statements that may disagree."""
    candidates: list[dict[str, Any]] = []
    for item in citations or []:
        if not isinstance(item, dict):
            continue
        text = _citation_text(item)
        numbers = tuple(sorted(set(_CONFLICT_NUMBER_RE.findall(text))))
        tokens = {
            token.casefold()
            for token in _CONFLICT_TOKEN_RE.findall(text)
            if token.casefold() not in _CONFLICT_STOPWORDS
        }
        if not numbers or len(tokens) < 2:
            continue
        candidates.append({"item": item, "text": text, "numbers": numbers, "tokens": tokens})

    groups: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            left_doc = str(left["item"].get("doc_id") or "")
            right_doc = str(right["item"].get("doc_id") or "")
            if not left_doc or left_doc == right_doc or left["numbers"] == right["numbers"]:
                continue
            union = left["tokens"] | right["tokens"]
            overlap = left["tokens"] & right["tokens"]
            if not union or len(overlap) < 2 or len(overlap) / len(union) < 0.4:
                continue
            pair = tuple(sorted((str(left["item"].get("evidence_id")), str(right["item"].get("evidence_id")))))
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            groups.append({
                "status": "potential_conflict",
                "shared_terms": sorted(overlap)[:8],
                "evidence": [
                    {
                        "doc_id": candidate["item"].get("doc_id"),
                        "doc_name": candidate["item"].get("doc_name"),
                        "ref": candidate["item"].get("ref"),
                        "evidence_id": candidate["item"].get("evidence_id"),
                        "numbers": list(candidate["numbers"]),
                        "text": candidate["text"][:500],
                    }
                    for candidate in (left, right)
                ],
            })
            if len(groups) >= max(1, int(limit or 8)):
                return groups
    return groups


def merge_fanout_contexts(
    results: Sequence[DocFanoutResult],
    *,
    max_total_chars: int = 24000,
    per_doc_chars: int = 8000,
    citation_ref_start: int = 1,
) -> dict[str, Any]:
    """Merge per-doc contexts with document name prefixes and citation rebasing."""
    parts: list[str] = []
    details: list[dict] = []
    citations: list[dict] = []
    diagnostics: list[dict] = []
    authorization: dict[str, Any] = {"enforced": True, "authorized": {}}
    used_chars = 0
    successful = 0
    next_ref = max(1, int(citation_ref_start or 1))

    for result in results:
        name = _safe_name(result.doc_id, result.doc_name)
        diag = {
            "doc_id": result.doc_id,
            "doc_name": name,
            "error": result.error,
            "char_count": 0,
            "included": False,
        }
        if result.error and not result.context:
            diagnostics.append(diag)
            continue
        name = _safe_name(result.doc_id, result.doc_name)
        result_citations: list[dict[str, Any]] = []
        citation_lines: list[str] = []
        for item in result.citations or []:
            if not isinstance(item, dict):
                continue
            citation = _namespaced_citation(
                item,
                doc_id=result.doc_id,
                doc_name=name,
                ref=next_ref,
            )
            text = _citation_text(citation)
            if not text:
                continue
            next_ref += 1
            result_citations.append(citation)
            citation_lines.append(f"[{citation['ref']}] {text}")
        raw = "\n\n".join(citation_lines) or str(result.context or "").strip()
        if not raw:
            diagnostics.append(diag)
            continue
        clipped = raw[: max(500, int(per_doc_chars or 8000))]
        block = prefix_context_with_doc(name, clipped)
        if used_chars + len(block) > max(2000, int(max_total_chars or 24000)) and parts:
            diag["error"] = diag.get("error") or "budget_exhausted"
            diagnostics.append(diag)
            continue
        parts.append(block)
        used_chars += len(block)
        successful += 1
        diag["char_count"] = len(clipped)
        diag["included"] = True
        diagnostics.append(diag)

        for item in result.detail or []:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry["doc_id"] = result.doc_id
            entry["doc_name"] = name
            details.append(entry)
        citations.extend(result_citations)
        for citation in result_citations:
            for field in ("evidence_id", "block_id", "chunk_id", "child_chunk_id", "context_id"):
                value = str(citation.get(field) or "").strip()
                if value:
                    authorization["authorized"].setdefault(field, []).append(value)
        result_authorization = result.citation_authorization or {}
        authorized = result_authorization.get("authorized") if isinstance(result_authorization, dict) else {}
        if isinstance(authorized, dict):
            for field, values in authorized.items():
                if not isinstance(values, list):
                    continue
                target = authorization["authorized"].setdefault(str(field), [])
                target.extend(str(value) for value in values if str(value or "").strip())

    return {
        "context": "\n\n".join(parts),
        "detail": details,
        "citations": citations,
        "doc_count": len(list(results or [])),
        "successful_doc_count": successful,
        "diagnostics": diagnostics,
        "total_chars": used_chars,
        "conflict_groups": group_potential_conflicts(citations),
        "citation_authorization": {
            "enforced": True,
            "authorized": {
                field: sorted(set(values))
                for field, values in authorization["authorized"].items()
                if values
            },
        },
    }


async def fanout_retrieve(
    *,
    question: str,
    documents: Sequence[DocFanoutInput],
    retriever: RetrieverFn,
    max_concurrency: int = 3,
    max_total_chars: int = 24000,
    per_doc_chars: int = 8000,
    citation_ref_start: int = 1,
) -> dict[str, Any]:
    """Run retriever on each document concurrently and merge results."""
    requested_docs = [item for item in documents if str(getattr(item, "doc_id", "") or "").strip()]
    docs, skipped_versions = deduplicate_document_versions(requested_docs)
    if not docs:
        return {
            "context": "",
            "detail": [],
            "citations": [],
            "doc_count": 0,
            "successful_doc_count": 0,
            "diagnostics": [],
            "total_chars": 0,
            "error": "empty_documents",
        }

    semaphore = asyncio.Semaphore(max(1, min(int(max_concurrency or 3), 5)))

    async def _one(doc: DocFanoutInput) -> DocFanoutResult:
        doc_id = str(doc.doc_id).strip()
        doc_name = _safe_name(doc_id, doc.doc_name)
        async with semaphore:
            try:
                payload = await retriever(doc_id, doc_name, question)
            except Exception as exc:
                logger.warning("[MultiDocFanout] retrieve failed doc=%s: %s", doc_id, exc)
                return DocFanoutResult(doc_id=doc_id, doc_name=doc_name, error=str(exc)[:240])
            payload = payload if isinstance(payload, dict) else {}
            context = str(payload.get("context") or "").strip()
            return DocFanoutResult(
                doc_id=doc_id,
                doc_name=doc_name,
                context=context,
                detail=list(payload.get("detail") or []) if isinstance(payload.get("detail"), list) else [],
                citations=list(payload.get("citations") or []) if isinstance(payload.get("citations"), list) else [],
                error=str(payload.get("error") or "").strip(),
                char_count=len(context),
                work_id=str(doc.work_id or f"doc:{doc_id}"),
                version_rank=int(doc.version_rank or 0),
                citation_authorization=(payload.get("citation_authorization") or {})
                if isinstance(payload.get("citation_authorization"), dict)
                else {},
            )

    results = await asyncio.gather(*[_one(doc) for doc in docs])
    merged = merge_fanout_contexts(
        results,
        max_total_chars=max_total_chars,
        per_doc_chars=per_doc_chars,
        citation_ref_start=citation_ref_start,
    )
    merged["requested_doc_count"] = len(requested_docs)
    merged["version_deduplication"] = skipped_versions
    merged["question"] = str(question or "").strip()
    return merged


def normalize_request_doc_ids(
    primary_doc_id: str,
    extra_doc_ids: Optional[Sequence[str]] = None,
    *,
    max_docs: int = 5,
) -> list[str]:
    """Deduplicate and bound doc ids, always keeping the primary first."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in [primary_doc_id, *(extra_doc_ids or [])]:
        doc_id = str(raw or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)
        if len(ordered) >= max(1, int(max_docs or 5)):
            break
    return ordered
