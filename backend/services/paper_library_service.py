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
            # 文件内容哈希 → work_id。canonical_work_id 依赖 DOI/arXiv/标题，
            # 元数据缺失或解析出入时同一个文件会被认成两篇；内容哈希是兜底。
            "seen_source_hashes": {},
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
    def _keyword_weights(
        feedback: list[Any],
        feed: list[Any],
        subscription_id: str,
        keywords: list[str],
    ) -> dict[str, float]:
        """从历史反馈学习关键词权重。

        反馈只保存 ``paper_id``，而关键词命中记录在 feed 条目里，两者按
        ``paper_id`` 关联即可统计每个关键词分别出现在「相关 / 不相关」论文
        中的次数——全程不需要论文正文，隔离性不变。

        权重取 Beta(1,1) 后验均值并平移到 (0.5, 1.5)：没有任何反馈时恒为
        1.0，此时打分与未学习前逐位相同；持续被判无关的关键词最多降到半权，
        持续相关的最多升到 1.5 倍。有界是刻意的，避免少数几次反馈就让某个
        词独占排序。
        """
        subscription_id = _clean(subscription_id, 80)
        matched_by_paper: dict[str, set[str]] = {}
        for item in feed:
            if not isinstance(item, Mapping):
                continue
            if _clean(item.get("subscription_id"), 80) != subscription_id:
                continue
            paper_id = _clean(item.get("paper_id"), 80)
            if not paper_id:
                continue
            bucket = matched_by_paper.setdefault(paper_id, set())
            for value in item.get("matched_keywords") or []:
                token = str(value or "").casefold().strip()
                if token:
                    bucket.add(token)

        positive: dict[str, int] = {}
        negative: dict[str, int] = {}
        for record in feedback:
            if not isinstance(record, Mapping):
                continue
            if _clean(record.get("subscription_id"), 80) != subscription_id:
                continue
            hits = matched_by_paper.get(_clean(record.get("paper_id"), 80))
            if not hits:
                continue
            bucket = positive if record.get("relevance") == "relevant" else negative
            for token in hits:
                bucket[token] = bucket.get(token, 0) + 1

        weights: dict[str, float] = {}
        for value in keywords:
            token = str(value or "").casefold().strip()
            if not token:
                continue
            pos = positive.get(token, 0)
            neg = negative.get(token, 0)
            weights[token] = round(0.5 + (1 + pos) / (2 + pos + neg), 4)
        return weights

    @staticmethod
    def _relevance(
        subscription: Mapping[str, Any],
        metadata: Mapping[str, Any],
        weights: Mapping[str, float] | None = None,
    ) -> tuple[float, list[str], bool]:
        keywords = [str(value).casefold() for value in (subscription.get("keywords") or [])]
        haystack = " ".join((
            _clean(metadata.get("title"), 500),
            _clean(metadata.get("abstract_preview"), 1200),
            _clean(metadata.get("venue"), 300),
        )).casefold()
        matches = [keyword for keyword in keywords if keyword and keyword in haystack]
        weight_map = weights or {}
        total = sum(weight_map.get(keyword, 1.0) for keyword in keywords if keyword)
        hit = sum(weight_map.get(keyword, 1.0) for keyword in matches)
        score = hit / total if total > 0 else 0.0
        informed = any(
            abs(weight_map.get(keyword, 1.0) - 1.0) > 1e-9
            for keyword in keywords
            if keyword
        )
        return round(score, 4), matches[:16], informed

    def process_new_documents(self, documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Process unseen works or newer explicit versions only."""
        with self._lock:
            state = self._load()
            subscriptions = [item for item in (state.get("subscriptions") or []) if item.get("enabled")]
            seen = state.setdefault("seen_papers", {})
            seen_hashes = state.setdefault("seen_source_hashes", {})
            feed = state.setdefault("feed", [])
            # 权重在本批入库前一次性算好：同一批内新写入的 feed 条目还没有反馈，
            # 不应影响本批打分，否则同批文档的先后顺序会改变各自的分数。
            history = list(state.get("feedback") or [])
            weights_by_subscription = {
                str(subscription.get("subscription_id") or ""): self._keyword_weights(
                    history,
                    feed,
                    str(subscription.get("subscription_id") or ""),
                    [str(value) for value in (subscription.get("keywords") or [])],
                )
                for subscription in subscriptions
            }
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
                # 同一文件改名重传、或元数据解析出入导致 work_id 漂移时，内容哈希
                # 仍然相同。不同版本是不同文件，哈希天然不同，不会被这条误杀。
                known_work_for_hash = str(seen_hashes.get(source_hash) or "") if source_hash else ""
                if known_work_for_hash and known_work_for_hash != work_id:
                    skipped.append({
                        "doc_id": str(doc_id),
                        "paper_id": paper_id,
                        "reason": "duplicate_source_hash",
                    })
                    continue
                novelty = "new_version" if previous else "new_work"
                if source_hash:
                    seen_hashes[source_hash] = work_id
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
                    relevance_score, matched_keywords, feedback_informed = self._relevance(
                        subscription,
                        metadata,
                        weights_by_subscription.get(str(subscription.get("subscription_id") or "")),
                    )
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
                        # 让界面能说明这条分数是否已被历史反馈调整过，避免用户
                        # 觉得反馈石沉大海。
                        "feedback_informed": feedback_informed,
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
