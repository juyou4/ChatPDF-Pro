"""Isolated paper subscriptions, interest feedback, and incremental processing.

This store never participates in chat retrieval, answer confidence, or factual
verification. It only builds a local metadata feed from newly seen paper IDs.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping
import uuid

from services.multi_doc_fanout_service import canonical_work_id, document_version_rank

PAPER_LIBRARY_VERSION = "paper-library-v1"
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _tokens(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        for token in _TOKEN_RE.findall(str(value or "")):
            normalized = token.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
    return result[:64]


def _paper_id(doc_id: str, metadata: Mapping[str, Any], source_hash: str = "") -> str:
    work_id = canonical_work_id(dict(metadata or {}), fallback=doc_id)
    stable = f"{work_id}\0{source_hash or doc_id}"
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:32]


class PaperLibraryService:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir) / "paper_library"
        self.state_path = self.root / "state.json"
        self._lock = threading.RLock()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "version": PAPER_LIBRARY_VERSION,
            "subscriptions": [],
            "feedback": [],
            "seen_papers": {},
            "feed": [],
            "updated_at": "",
        }

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("version") == PAPER_LIBRARY_VERSION:
                return value
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return self._empty_state()

    def _save(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        state["version"] = PAPER_LIBRARY_VERSION
        state["updated_at"] = _now()
        temporary = self.state_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def list_subscriptions(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._load().get("subscriptions") or [])

    def create_subscription(
        self,
        *,
        name: str,
        query: str,
        keywords: list[str] | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        clean_name = _clean(name, 120)
        clean_query = _clean(query, 500)
        terms = _tokens([clean_query, *(keywords or [])])
        if not clean_name or not clean_query or not terms:
            raise ValueError("订阅名称、查询和至少一个有效关键词不能为空")
        timestamp = _now()
        record = {
            "subscription_id": uuid.uuid4().hex,
            "name": clean_name,
            "query": clean_query,
            "keywords": terms,
            "enabled": bool(enabled),
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self._lock:
            state = self._load()
            state.setdefault("subscriptions", []).append(record)
            self._save(state)
        return deepcopy(record)

    def update_subscription(self, subscription_id: str, changes: Mapping[str, Any]) -> dict[str, Any] | None:
        subscription_id = _clean(subscription_id, 80)
        with self._lock:
            state = self._load()
            for record in state.get("subscriptions") or []:
                if record.get("subscription_id") != subscription_id:
                    continue
                if "name" in changes:
                    name = _clean(changes.get("name"), 120)
                    if not name:
                        raise ValueError("订阅名称不能为空")
                    record["name"] = name
                if "query" in changes or "keywords" in changes:
                    query = _clean(changes.get("query", record.get("query")), 500)
                    keywords = changes.get("keywords", record.get("keywords") or [])
                    terms = _tokens([query, *(keywords if isinstance(keywords, list) else [])])
                    if not query or not terms:
                        raise ValueError("订阅查询和关键词不能为空")
                    record["query"] = query
                    record["keywords"] = terms
                if "enabled" in changes:
                    record["enabled"] = bool(changes.get("enabled"))
                record["updated_at"] = _now()
                self._save(state)
                return deepcopy(record)
        return None

    def delete_subscription(self, subscription_id: str) -> bool:
        subscription_id = _clean(subscription_id, 80)
        with self._lock:
            state = self._load()
            before = len(state.get("subscriptions") or [])
            state["subscriptions"] = [
                item for item in (state.get("subscriptions") or [])
                if item.get("subscription_id") != subscription_id
            ]
            state["feed"] = [
                item for item in (state.get("feed") or [])
                if item.get("subscription_id") != subscription_id
            ]
            changed = len(state["subscriptions"]) != before
            if changed:
                self._save(state)
            return changed

    def record_feedback(
        self,
        *,
        subscription_id: str,
        paper_id: str,
        relevance: str,
        novelty: str,
        reason_codes: list[str] | None = None,
    ) -> dict[str, Any]:
        relevance = _clean(relevance, 40).lower()
        novelty = _clean(novelty, 40).lower()
        if relevance not in {"relevant", "not_relevant"}:
            raise ValueError("relevance 必须是 relevant 或 not_relevant")
        if novelty not in {"new", "known", "unsure"}:
            raise ValueError("novelty 必须是 new、known 或 unsure")
        record = {
            "feedback_id": uuid.uuid4().hex,
            "subscription_id": _clean(subscription_id, 80),
            "paper_id": _clean(paper_id, 80),
            "relevance": relevance,
            "novelty": novelty,
            "reason_codes": _tokens(reason_codes or [])[:12],
            "created_at": _now(),
        }
        if not record["subscription_id"] or not record["paper_id"]:
            raise ValueError("subscription_id 和 paper_id 不能为空")
        with self._lock:
            state = self._load()
            if not any(
                item.get("subscription_id") == record["subscription_id"]
                for item in (state.get("subscriptions") or [])
            ):
                raise KeyError("subscription_not_found")
            state.setdefault("feedback", []).append(record)
            state["feedback"] = state["feedback"][-5000:]
            self._save(state)
        return deepcopy(record)

    @staticmethod
    def _metadata_for_document(doc_id: str, doc: Mapping[str, Any]) -> tuple[dict, str, int]:
        metadata = dict(doc.get("paper_metadata") or {})
        hydration = doc.get("paper_metadata_hydration")
        if isinstance(hydration, Mapping) and isinstance(hydration.get("metadata"), Mapping):
            for field, value in hydration["metadata"].items():
                if value not in (None, "", [], {}):
                    metadata[field] = value
        data = doc.get("data") if isinstance(doc.get("data"), Mapping) else {}
        manifest = data.get("parse_manifest") if isinstance(data.get("parse_manifest"), Mapping) else {}
        source_hash = _clean(manifest.get("source_hash") or metadata.get("document_source_hash"), 256)
        filename = _clean(doc.get("filename") or doc_id, 240)
        return metadata, source_hash, document_version_rank(filename, metadata)

    @staticmethod
    def _relevance(subscription: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[float, list[str]]:
        keywords = [str(value).casefold() for value in (subscription.get("keywords") or [])]
        haystack = " ".join((
            _clean(metadata.get("title"), 500),
            _clean(metadata.get("abstract_preview"), 1200),
            _clean(metadata.get("venue"), 300),
        )).casefold()
        matches = [keyword for keyword in keywords if keyword and keyword in haystack]
        score = len(matches) / max(1, len(keywords))
        return round(score, 4), matches[:16]

    def process_new_documents(self, documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Process unseen works or newer explicit versions only."""
        with self._lock:
            state = self._load()
            subscriptions = [item for item in (state.get("subscriptions") or []) if item.get("enabled")]
            seen = state.setdefault("seen_papers", {})
            feed = state.setdefault("feed", [])
            processed: list[dict[str, Any]] = []
            skipped: list[dict[str, Any]] = []
            for doc_id, doc in (documents or {}).items():
                if not isinstance(doc, Mapping):
                    continue
                metadata, source_hash, version_rank = self._metadata_for_document(str(doc_id), doc)
                if not metadata.get("title"):
                    skipped.append({"doc_id": str(doc_id), "reason": "missing_metadata"})
                    continue
                work_id = canonical_work_id(metadata, fallback=str(doc_id))
                paper_id = _paper_id(str(doc_id), metadata, source_hash)
                previous = seen.get(work_id) if isinstance(seen.get(work_id), dict) else None
                if previous and int(previous.get("version_rank") or 0) >= version_rank:
                    skipped.append({"doc_id": str(doc_id), "paper_id": paper_id, "reason": "already_processed"})
                    continue
                novelty = "new_version" if previous else "new_work"
                seen[work_id] = {
                    "paper_id": paper_id,
                    "doc_id": str(doc_id),
                    "source_hash": source_hash,
                    "version_rank": version_rank,
                    "first_seen_at": previous.get("first_seen_at") if previous else _now(),
                    "last_seen_at": _now(),
                }
                matches = []
                for subscription in subscriptions:
                    relevance_score, matched_keywords = self._relevance(subscription, metadata)
                    if relevance_score <= 0:
                        continue
                    item = {
                        "feed_id": uuid.uuid4().hex,
                        "subscription_id": subscription.get("subscription_id"),
                        "paper_id": paper_id,
                        "work_id": work_id,
                        "doc_id": str(doc_id),
                        "title": _clean(metadata.get("title"), 300),
                        "authors": [
                            _clean(value, 120) for value in (metadata.get("authors") or [])[:12]
                        ],
                        "year": metadata.get("year"),
                        "doi": _clean(metadata.get("doi"), 300),
                        "arxiv_id": _clean(metadata.get("arxiv_id"), 120),
                        "external_url": _clean(metadata.get("external_url"), 1200),
                        "discovery_provider": _clean(metadata.get("discovery_provider"), 80),
                        "relevance_score": relevance_score,
                        "matched_keywords": matched_keywords,
                        "novelty": novelty,
                        "created_at": _now(),
                    }
                    feed.append(item)
                    matches.append({
                        "subscription_id": item["subscription_id"],
                        "relevance_score": relevance_score,
                    })
                processed.append({
                    "doc_id": str(doc_id),
                    "paper_id": paper_id,
                    "work_id": work_id,
                    "novelty": novelty,
                    "matches": matches,
                })
            state["feed"] = feed[-5000:]
            self._save(state)
            return {
                "processed_count": len(processed),
                "skipped_count": len(skipped),
                "processed": processed,
                "skipped": skipped,
                "subscription_count": len(subscriptions),
                "isolation": "paper_library_only",
            }

    def list_feed(self, *, subscription_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            feed = list(self._load().get("feed") or [])
        if subscription_id:
            feed = [item for item in feed if item.get("subscription_id") == subscription_id]
        return deepcopy(list(reversed(feed[-max(1, min(int(limit or 50), 200)):])) )

    def clear(self) -> None:
        with self._lock:
            self._save(self._empty_state())
