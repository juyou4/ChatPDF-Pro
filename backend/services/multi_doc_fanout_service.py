"""同会话多文档 fan-out 聚合。

不改动单文档索引结构：对每个 doc 独立取上下文，再按文档前缀合并，
供跨文档比较 / 综述类问题使用。完整 Agent 仍默认单文档；本模块提供
确定性聚合层，作为 paper-qa 跨文档能力的轻量第一步。
"""

from __future__ import annotations

import asyncio
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


@dataclass
class DocFanoutResult:
    doc_id: str
    doc_name: str
    context: str = ""
    detail: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    error: str = ""
    char_count: int = 0


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


def merge_fanout_contexts(
    results: Sequence[DocFanoutResult],
    *,
    max_total_chars: int = 24000,
    per_doc_chars: int = 8000,
) -> dict[str, Any]:
    """Merge per-doc contexts with document name prefixes and citation rebasing."""
    parts: list[str] = []
    details: list[dict] = []
    citations: list[dict] = []
    diagnostics: list[dict] = []
    used_chars = 0
    successful = 0

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
        raw = str(result.context or "").strip()
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
        for item in result.citations or []:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry["doc_id"] = result.doc_id
            entry["doc_name"] = name
            # Keep page anchors; UI can show doc_name prefix.
            citations.append(entry)

    return {
        "context": "\n\n".join(parts),
        "detail": details,
        "citations": citations,
        "doc_count": len(list(results or [])),
        "successful_doc_count": successful,
        "diagnostics": diagnostics,
        "total_chars": used_chars,
    }


async def fanout_retrieve(
    *,
    question: str,
    documents: Sequence[DocFanoutInput],
    retriever: RetrieverFn,
    max_concurrency: int = 3,
    max_total_chars: int = 24000,
    per_doc_chars: int = 8000,
) -> dict[str, Any]:
    """Run retriever on each document concurrently and merge results."""
    docs = [item for item in documents if str(getattr(item, "doc_id", "") or "").strip()]
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
            )

    results = await asyncio.gather(*[_one(doc) for doc in docs])
    merged = merge_fanout_contexts(
        results,
        max_total_chars=max_total_chars,
        per_doc_chars=per_doc_chars,
    )
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
