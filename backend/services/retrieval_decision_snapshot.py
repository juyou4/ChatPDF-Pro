"""Candidate-level retrieval decision snapshots.

The retrieval stack intentionally keeps its ranking and filtering logic in the
existing services.  This module records the decisions made by that stack so a
quality investigation can answer *why* a candidate was or was not used without
persisting document text or provider secrets.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Iterable, Mapping, Optional


RETRIEVAL_SNAPSHOT_VERSION = 1
_MAX_CANDIDATES = 256
_MAX_STAGES_PER_CANDIDATE = 32
_MAX_ID_LENGTH = 180
_MAX_SECTION_LENGTH = 240


def new_retrieval_run_id() -> str:
    """Return a per-retrieval-run identifier safe to expose in diagnostics."""

    return f"retrieval-{uuid.uuid4().hex}"


def _text(value: Any, limit: int = _MAX_ID_LENGTH) -> str:
    return str(value or "").strip()[:limit]


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_page(value: Any) -> int:
    try:
        page = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return page if page > 0 else 0


def _page_range(item: Mapping[str, Any]) -> list[int]:
    raw = item.get("page_range")
    if isinstance(raw, (list, tuple)) and raw:
        try:
            start = int(raw[0])
            end = int(raw[1] if len(raw) > 1 else raw[0])
        except (TypeError, ValueError):
            start = end = 0
        if start > 0 and end > 0:
            return [min(start, end), max(start, end)]
    page = _positive_page(item.get("page"))
    return [page, page] if page else []


def normalize_retrieval_identity(identity: Optional[Mapping[str, Any]]) -> dict[str, str]:
    """Normalize the parser identity without accepting arbitrary request data."""

    value = identity if isinstance(identity, Mapping) else {}
    route = _text(
        value.get("parser_route")
        or value.get("route")
        or value.get("parse_route"),
        32,
    ).lower()
    generation = _text(
        value.get("parse_generation") or value.get("generation"),
        256,
    )
    source_hash = _text(
        value.get("document_source_hash") or value.get("source_hash"),
        256,
    )
    status = "bound" if route and generation and source_hash else "unavailable"
    return {
        "route": route,
        "generation": generation,
        "source_hash": source_hash,
        "identity_status": status,
    }


def _candidate_key(item: Mapping[str, Any]) -> dict[str, Any]:
    raw_text = str(item.get("raw_chunk_text") or item.get("chunk") or item.get("text") or "")
    text_hash = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()[:24]
    page_range = _page_range(item)
    return {
        "page_range": page_range,
        "chunk_id": item.get("chunk_id"),
        "child_chunk_id": item.get("child_chunk_id"),
        "parent_id": item.get("parent_id"),
        "block_id": _text(item.get("block_id")),
        "evidence_id": _text(item.get("evidence_id")),
        "context_id": _text(item.get("context_id")),
        "group_id": _text(item.get("group_id") or item.get("semantic_group_id")),
        "text_hash": text_hash if raw_text else "",
    }


def stable_candidate_id(
    doc_id: str,
    item: Mapping[str, Any],
    identity: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build a stable, opaque candidate id for one parse identity."""

    normalized_identity = normalize_retrieval_identity(identity)
    payload = {
        "doc_id": _text(doc_id, 128),
        "identity": normalized_identity,
        "key": _candidate_key(item),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
    return f"cand-{digest}"


def _candidate_identity_fields(item: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    page_range = _page_range(item)
    if page_range:
        fields["page_range"] = page_range
        fields["page"] = page_range[0]
    for key in (
        "chunk_id",
        "child_chunk_id",
        "parent_id",
        "block_id",
        "evidence_id",
        "context_id",
        "group_id",
    ):
        value = item.get(key)
        if key == "group_id" and not value:
            value = item.get("semantic_group_id")
        if value not in (None, "", [], {}):
            fields[key] = _text(value) if key not in {"chunk_id", "child_chunk_id", "parent_id"} else value
    section = _text(item.get("section_path") or item.get("chunk_heading"), _MAX_SECTION_LENGTH)
    if section:
        fields["section_path"] = section
    chunk_type = _text(item.get("chunk_type") or item.get("block_type"), 64).lower()
    if chunk_type:
        fields["chunk_type"] = chunk_type
    return fields


def _sources(item: Mapping[str, Any]) -> list[str]:
    raw = item.get("retrieval_sources")
    values: list[str] = []
    if isinstance(raw, (list, tuple)):
        values.extend(_text(value, 48).lower() for value in raw)
    elif raw:
        values.extend(_text(value, 48).lower() for value in str(raw).replace(",", "+").split("+"))
    if item.get("bm25") and "bm25" not in values:
        values.append("bm25")
    if item.get("rrf_score") is not None and "rrf" not in values:
        values.append("rrf")
    if item.get("semantic_group_id") and "semantic_group" not in values:
        values.append("semantic_group")
    if not values:
        values.append("vector")
    return list(dict.fromkeys(value for value in values if value))[:12]


class RetrievalDecisionSnapshot:
    """Mutable in-request snapshot with deterministic, idempotent updates."""

    def __init__(
        self,
        doc_id: str = "",
        *,
        identity: Optional[Mapping[str, Any]] = None,
        retrieval_run_id: Optional[str] = None,
    ) -> None:
        self.doc_id = _text(doc_id, 128)
        self.retrieval_run_id = _text(retrieval_run_id or new_retrieval_run_id(), 80)
        self.identity = normalize_retrieval_identity(identity)
        self._candidates: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._stage_order: list[str] = []
        self._context_stats: dict[str, Any] = {}

    def bind_identity(self, identity: Optional[Mapping[str, Any]]) -> None:
        incoming = normalize_retrieval_identity(identity)
        current = self.identity
        if current.get("identity_status") == "unavailable":
            self.identity = incoming
            return
        if incoming.get("identity_status") == "unavailable":
            return
        if any(current.get(key) != incoming.get(key) for key in ("route", "generation", "source_hash")):
            self.identity = {
                **current,
                "identity_status": "mismatch",
                "expected_route": current.get("route", ""),
                "expected_generation": current.get("generation", ""),
                "expected_source_hash": current.get("source_hash", ""),
                "actual_route": incoming.get("route", ""),
                "actual_generation": incoming.get("generation", ""),
                "actual_source_hash": incoming.get("source_hash", ""),
            }
            return
        self.identity = incoming

    def candidate_id(self, item: Mapping[str, Any]) -> str:
        return stable_candidate_id(self.doc_id, item, self.identity)

    def _ensure_candidate(self, item: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        candidate_id = str(item.get("candidate_id") or item.get("retrieval_candidate_id") or self.candidate_id(item))
        record = self._candidates.get(candidate_id)
        if record is None:
            record = {
                "candidate_id": candidate_id,
                "doc_id": self.doc_id,
                **_candidate_identity_fields(item),
                "retrieval_sources": _sources(item),
                "stages": [],
                "context_included": False,
                "citation_used": False,
                "claim_support": None,
            }
            self._candidates[candidate_id] = record
        else:
            for source in _sources(item):
                if source not in record["retrieval_sources"]:
                    record["retrieval_sources"].append(source)
        return candidate_id, record

    def _record_stage(
        self,
        record: dict[str, Any],
        stage: str,
        *,
        rank: Optional[int] = None,
        score: Any = None,
        included: bool,
        reason: str,
    ) -> None:
        normalized_stage = _text(stage, 64).lower() or "unknown"
        if normalized_stage not in self._stage_order:
            self._stage_order.append(normalized_stage)
        event = {
            "stage": normalized_stage,
            "included": bool(included),
            "reason": _text(reason, 120) or ("included" if included else "excluded"),
        }
        if rank is not None:
            try:
                event["rank"] = int(rank)
            except (TypeError, ValueError):
                pass
        numeric_score = _finite_number(score)
        if numeric_score is not None:
            event["score"] = round(numeric_score, 8)
        key = (
            event["stage"],
            event.get("rank"),
            event.get("score"),
            event["included"],
            event["reason"],
        )
        existing = {
            (
                item.get("stage"),
                item.get("rank"),
                item.get("score"),
                item.get("included"),
                item.get("reason"),
            )
            for item in record["stages"]
        }
        if key not in existing and len(record["stages"]) < _MAX_STAGES_PER_CANDIDATE:
            record["stages"].append(event)

    def record_candidates(
        self,
        items: Iterable[Mapping[str, Any]],
        stage: str,
        *,
        score_keys: tuple[str, ...] = ("combined_score", "rerank_score", "rrf_score", "similarity", "score"),
        reason: str = "candidate_retrieved",
    ) -> list[str]:
        ids: list[str] = []
        for rank, item in enumerate(items or [], start=1):
            if not isinstance(item, Mapping):
                continue
            if len(self._candidates) >= _MAX_CANDIDATES and self.candidate_id(item) not in self._candidates:
                continue
            candidate_id, record = self._ensure_candidate(item)
            score = next((item.get(key) for key in score_keys if item.get(key) is not None), None)
            self._record_stage(record, stage, rank=rank, score=score, included=True, reason=reason)
            ids.append(candidate_id)
        return ids

    def record_transition(
        self,
        before: Iterable[Mapping[str, Any]],
        after: Iterable[Mapping[str, Any]],
        stage: str,
        *,
        included_reason: str = "passed",
        excluded_reason: str = "filtered_by_stage",
    ) -> None:
        before_items = [item for item in (before or []) if isinstance(item, Mapping)]
        after_items = [item for item in (after or []) if isinstance(item, Mapping)]
        after_ids = set(self.record_candidates(after_items, stage, reason=included_reason))
        before_ids = set()
        for rank, item in enumerate(before_items, start=1):
            if len(self._candidates) >= _MAX_CANDIDATES and self.candidate_id(item) not in self._candidates:
                continue
            candidate_id, record = self._ensure_candidate(item)
            before_ids.add(candidate_id)
            if candidate_id not in after_ids:
                score = next((item.get(key) for key in ("combined_score", "rerank_score", "rrf_score", "similarity", "score") if item.get(key) is not None), None)
                self._record_stage(record, stage, rank=rank, score=score, included=False, reason=excluded_reason)

    def mark_context(self, segments: Iterable[Mapping[str, Any]], *, token_budget: int = 0, token_used: int = 0) -> None:
        matched: set[str] = set()
        for segment in segments or []:
            if not isinstance(segment, Mapping):
                continue
            requested_id = str(segment.get("candidate_id") or segment.get("retrieval_candidate_id") or "")
            if requested_id and requested_id in self._candidates:
                matched.add(requested_id)
                continue
            segment_key = _candidate_key(segment)
            for candidate_id, record in self._candidates.items():
                if any(
                    segment_key.get(key) and record.get(key) == segment_key.get(key)
                    for key in ("evidence_id", "block_id", "context_id", "chunk_id")
                ) or (
                    segment_key.get("group_id") and record.get("group_id") == segment_key.get("group_id")
                ):
                    matched.add(candidate_id)
        for candidate_id in matched:
            record = self._candidates[candidate_id]
            record["context_included"] = True
            self._record_stage(record, "context", included=True, reason="included_in_final_context")
        for candidate_id, record in self._candidates.items():
            if candidate_id not in matched:
                self._record_stage(record, "token_budget", included=False, reason="not_in_final_context")
        self._context_stats = {
            "candidate_count": len(self._candidates),
            "context_candidate_count": len(matched),
            "token_budget": max(0, int(token_budget or 0)),
            "token_used": max(0, int(token_used or 0)),
            "token_budget_status": (
                "within_budget"
                if not token_budget or token_used <= token_budget
                else "truncated"
            ),
        }

    def mark_citations(
        self,
        citations: Iterable[Mapping[str, Any]],
        *,
        claim_bindings: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> None:
        citation_ids: set[str] = set()
        ref_to_id: dict[str, set[str]] = {}
        for citation in citations or []:
            if not isinstance(citation, Mapping):
                continue
            if citation.get("authorized") is False or str(
                citation.get("authorization_status") or ""
            ).strip().lower() in {"unauthorized", "filtered", "rejected"}:
                continue
            candidate_id = str(citation.get("candidate_id") or citation.get("retrieval_candidate_id") or "")
            if not candidate_id:
                candidate_id = self._find_candidate_id(citation)
            if candidate_id in self._candidates:
                citation_ids.add(candidate_id)
            ref = str(citation.get("ref") or citation.get("source_ref") or "")
            if ref and candidate_id:
                ref_to_id.setdefault(ref, set()).add(candidate_id)
        for candidate_id, record in self._candidates.items():
            if candidate_id in citation_ids:
                record["citation_used"] = True
                self._record_stage(record, "citation", included=True, reason="authorized_citation")
            elif record.get("context_included"):
                self._record_stage(record, "citation", included=False, reason="not_cited")

        for binding in claim_bindings or []:
            if not isinstance(binding, Mapping):
                continue
            status = _text(binding.get("status"), 32).lower() or "uncertain"
            refs = [str(value) for value in (binding.get("refs") or [])]
            for ref in refs:
                for candidate_id in ref_to_id.get(ref, set()):
                    record = self._candidates.get(candidate_id)
                    if record is None:
                        continue
                    record["claim_support"] = status
                    self._record_stage(
                        record,
                        "claim_support",
                        included=status in {"supported", "confirmed"},
                        reason=f"claim_{status}",
                    )

    def _find_candidate_id(self, item: Mapping[str, Any]) -> str:
        candidate_id = self.candidate_id(item)
        if candidate_id in self._candidates:
            return candidate_id
        for key in ("evidence_id", "block_id", "context_id", "chunk_id", "group_id"):
            value = item.get(key)
            if value in (None, ""):
                continue
            for existing_id, record in self._candidates.items():
                if str(record.get(key) or "") == str(value):
                    return existing_id
        return ""

    def summary(self) -> dict[str, Any]:
        candidates = list(self._candidates.values())
        return {
            "schema_version": RETRIEVAL_SNAPSHOT_VERSION,
            "retrieval_run_id": self.retrieval_run_id,
            "identity_status": self.identity.get("identity_status", "unavailable"),
            "route": self.identity.get("route", ""),
            "stage_order": list(self._stage_order),
            "candidate_count": len(candidates),
            "context_candidate_count": sum(1 for item in candidates if item.get("context_included")),
            "citation_candidate_count": sum(1 for item in candidates if item.get("citation_used")),
            "claim_support_count": sum(1 for item in candidates if item.get("claim_support") in {"supported", "confirmed"}),
            "context": dict(self._context_stats),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RETRIEVAL_SNAPSHOT_VERSION,
            "retrieval_run_id": self.retrieval_run_id,
            "doc_id": self.doc_id,
            "identity": deepcopy(self.identity),
            "stage_order": list(self._stage_order),
            "candidates": deepcopy(list(self._candidates.values())),
            "context": deepcopy(self._context_stats),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> "RetrievalDecisionSnapshot":
        payload = value if isinstance(value, Mapping) else {}
        snapshot = cls(
            str(payload.get("doc_id") or ""),
            identity=payload.get("identity") if isinstance(payload.get("identity"), Mapping) else None,
            retrieval_run_id=str(payload.get("retrieval_run_id") or "") or None,
        )
        snapshot._stage_order = [
            _text(item, 64).lower()
            for item in (payload.get("stage_order") or [])
            if _text(item, 64)
        ][:64]
        for item in payload.get("candidates") or []:
            if not isinstance(item, Mapping):
                continue
            candidate_id = _text(item.get("candidate_id"), 80)
            if not candidate_id:
                continue
            snapshot._candidates[candidate_id] = deepcopy(dict(item))
            snapshot._candidates[candidate_id].setdefault("stages", [])
            snapshot._candidates[candidate_id].setdefault("context_included", False)
            snapshot._candidates[candidate_id].setdefault("citation_used", False)
            snapshot._candidates[candidate_id].setdefault("claim_support", None)
        snapshot._context_stats = deepcopy(payload.get("context") or {})
        return snapshot


def merge_decision_snapshots(
    left: Optional[Mapping[str, Any]],
    right: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge snapshots from retries or multi-query fan-out idempotently."""

    if not isinstance(left, Mapping):
        return deepcopy(dict(right)) if isinstance(right, Mapping) else {}
    if not isinstance(right, Mapping):
        return deepcopy(dict(left))
    first = RetrievalDecisionSnapshot.from_dict(left)
    second = RetrievalDecisionSnapshot.from_dict(right)
    if first.retrieval_run_id == second.retrieval_run_id:
        for candidate_id, record in second._candidates.items():
            if candidate_id not in first._candidates:
                first._candidates[candidate_id] = record
                continue
            target = first._candidates[candidate_id]
            existing = {
                json.dumps(stage, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                for stage in target.get("stages", [])
            }
            for stage in record.get("stages", []):
                encoded = json.dumps(stage, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                if encoded not in existing and len(target["stages"]) < _MAX_STAGES_PER_CANDIDATE:
                    target["stages"].append(deepcopy(stage))
            target["context_included"] = bool(target.get("context_included") or record.get("context_included"))
            target["citation_used"] = bool(target.get("citation_used") or record.get("citation_used"))
            target["claim_support"] = target.get("claim_support") or record.get("claim_support")
        for stage in second._stage_order:
            if stage not in first._stage_order:
                first._stage_order.append(stage)
        if second._context_stats:
            first._context_stats = dict(second._context_stats)
        return first.to_dict()

    merged = first.to_dict()
    merged["retrieval_run_ids"] = list(dict.fromkeys([
        str(left.get("retrieval_run_id") or ""),
        str(right.get("retrieval_run_id") or ""),
    ]))
    merged["identity"] = normalize_retrieval_identity(left.get("identity") if isinstance(left.get("identity"), Mapping) else {})
    if merged["identity"].get("identity_status") == "bound" and normalize_retrieval_identity(
        right.get("identity") if isinstance(right.get("identity"), Mapping) else {}
    ) != merged["identity"]:
        merged["identity"]["identity_status"] = "mixed"
    existing_ids = {str(item.get("candidate_id")) for item in merged.get("candidates") or [] if isinstance(item, Mapping)}
    for item in second.to_dict().get("candidates") or []:
        if isinstance(item, Mapping) and str(item.get("candidate_id")) not in existing_ids:
            merged.setdefault("candidates", []).append(deepcopy(dict(item)))
    merged["stage_order"] = list(dict.fromkeys([
        *(merged.get("stage_order") or []),
        *(second.to_dict().get("stage_order") or []),
    ]))
    merged["summary"] = RetrievalDecisionSnapshot.from_dict(merged).summary()
    return merged


def mark_snapshot_citations(
    snapshot: Optional[Mapping[str, Any]],
    citations: Iterable[Mapping[str, Any]],
    *,
    claim_bindings: Optional[Iterable[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        return {}
    value = RetrievalDecisionSnapshot.from_dict(snapshot)
    value.mark_citations(citations, claim_bindings=claim_bindings)
    return value.to_dict()


def sanitize_candidate_snapshot(snapshot: Optional[Mapping[str, Any]], *, include_candidates: bool = False) -> dict[str, Any]:
    """Return a public/debug view; never expose the parse source hash."""

    if not isinstance(snapshot, Mapping):
        return {}
    value = RetrievalDecisionSnapshot.from_dict(snapshot)
    result = value.summary()
    result["identity_status"] = value.identity.get("identity_status", "unavailable")
    if include_candidates:
        result["candidates"] = deepcopy(list(value._candidates.values()))
    return result
