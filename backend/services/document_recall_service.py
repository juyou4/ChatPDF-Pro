"""Resolve previously uploaded documents for explicit cross-document reading.

The resolver intentionally returns candidates instead of silently changing a
chat request.  A title can be ambiguous, and a user must choose the companion
documents before their content enters the current context.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from services.document_parse_state import read_parse_manifest


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def list_recallable_documents(
    documents: Mapping[str, Any],
    *,
    query: str = "",
    exclude_doc_id: str = "",
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Rank lightweight document descriptors by title/id match and recency."""
    needle = _normalize(query)
    query_tokens = set(_tokens(needle))
    excluded = str(exclude_doc_id or "").strip()
    rows: list[tuple[tuple[float, float, str], dict[str, Any]]] = []

    for raw_doc_id, raw_doc in (documents or {}).items():
        doc_id = str(raw_doc_id or "").strip()
        document = raw_doc if isinstance(raw_doc, Mapping) else {}
        if not doc_id or doc_id == excluded or not document:
            continue
        filename = str(document.get("filename") or doc_id).strip() or doc_id
        title = Path(filename).stem or filename
        normalized_title = _normalize(title)
        normalized_id = _normalize(doc_id)
        title_tokens = set(_tokens(normalized_title))
        score = 0.0
        reasons: list[str] = []

        if needle:
            if needle == normalized_id:
                score += 120.0
                reasons.append("doc_id_exact")
            if needle == normalized_title:
                score += 110.0
                reasons.append("title_exact")
            elif needle and needle in normalized_title:
                score += 80.0
                reasons.append("title_phrase")
            overlap = len(query_tokens & title_tokens)
            if overlap:
                score += overlap * 18.0
                reasons.append("title_tokens")
            if not score:
                continue

        manifest = read_parse_manifest(document, doc_id=doc_id)
        data = document.get("data") if isinstance(document.get("data"), Mapping) else {}
        uploaded_at = str(document.get("upload_time") or "")
        recency = _timestamp(uploaded_at)
        rows.append((
            (-score, -recency, filename.casefold()),
            {
                "doc_id": doc_id,
                "filename": filename[:160],
                "title": title[:160],
                "parse_status": str(manifest.get("status") or ""),
                "parse_route": str(manifest.get("resolved_route") or manifest.get("requested_route") or ""),
                "parse_ready": str(manifest.get("status") or "").strip().lower() == "ready",
                "total_pages": _safe_positive_int(data.get("total_pages") or document.get("total_pages")),
                "upload_time": uploaded_at,
                "score": round(score, 2),
                "match_reasons": reasons,
            },
        ))

    rows.sort(key=lambda item: item[0])
    return [row for _sort_key, row in rows[:max(1, min(int(limit or 8), 20))]]


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _tokens(value: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(str(value or "").casefold()) if len(token) >= 2]


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(str(value or "").replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _safe_positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = ["list_recallable_documents"]
