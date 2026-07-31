"""Request-scoped citation authorization helpers.

An Agent may only cite evidence that a successful tool call exposed during the
current request.  The ledger deliberately stores stable provenance identifiers,
never document text, so it remains cheap and does not create another cache.
"""

from __future__ import annotations

from typing import Any, Iterable


CITATION_AUTHORIZATION_POLICY = "tool_result_evidence_v1"

# These are concrete evidence identities.  Coarse labels such as page and table
# number are intentionally excluded: they could authorize a different row or
# region that the Agent did not actually read.
CITATION_IDENTITY_FIELDS = (
    "evidence_id",
    "block_id",
    "chunk_id",
    "child_chunk_id",
    "parent_id",
    "asset_id",
    "analyzed_asset_id",
    "visual_evidence_id",
    "evidence_unit_id",
    "context_id",
    "group_id",
)


def _identity_value(value: Any) -> str:
    return str(value or "").strip()


def extract_citation_identity_values(records: Iterable[Any]) -> dict[str, set[str]]:
    """Collect field-scoped stable evidence IDs from returned metadata records."""
    collected = {field: set() for field in CITATION_IDENTITY_FIELDS}
    for record in records:
        if not isinstance(record, dict):
            continue
        for field in CITATION_IDENTITY_FIELDS:
            value = _identity_value(record.get(field))
            if value:
                collected[field].add(value)
    return collected


def normalize_citation_authorization(value: Any) -> dict:
    """Normalize an internal ledger snapshot without accepting text payloads."""
    raw = value if isinstance(value, dict) else {}
    raw_allowed = raw.get("authorized")
    raw_allowed = raw_allowed if isinstance(raw_allowed, dict) else {}
    authorized: dict[str, set[str]] = {}
    for field in CITATION_IDENTITY_FIELDS:
        values = raw_allowed.get(field)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple, set)):
            values = []
        authorized[field] = {
            normalized
            for item in values
            if (normalized := _identity_value(item))
        }
    return {
        "policy": str(raw.get("policy") or "").strip(),
        "enforced": bool(raw.get("enforced")),
        "authorized": authorized,
        "tool_counts": {
            str(key): max(0, int(item or 0))
            for key, item in (raw.get("tool_counts") or {}).items()
            if str(key).strip()
        } if isinstance(raw.get("tool_counts"), dict) else {},
    }


def citation_is_authorized(citation: Any, authorization: Any) -> bool:
    """Return whether a citation shares an exact, field-matched returned ID."""
    normalized = normalize_citation_authorization(authorization)
    if not normalized["enforced"]:
        return True
    if not isinstance(citation, dict):
        return False
    for field, allowed in normalized["authorized"].items():
        value = _identity_value(citation.get(field))
        if value and value in allowed:
            return True
    return False


def filter_authorized_citations(
    citations: Iterable[Any],
    authorization: Any,
    *,
    rebase_refs: bool = False,
) -> tuple[list[dict], dict]:
    """Drop citations not backed by this request's successful tool outputs."""
    normalized = normalize_citation_authorization(authorization)
    original = [dict(item) for item in citations or [] if isinstance(item, dict)]
    if not normalized["enforced"]:
        return original, {
            "enforced": False,
            "input_count": len(original),
            "kept_count": len(original),
            "filtered_count": 0,
        }
    kept = [item for item in original if citation_is_authorized(item, normalized)]
    if rebase_refs:
        rebased: list[dict] = []
        for index, item in enumerate(kept, start=1):
            updated = dict(item)
            try:
                source_ref = int(updated.get("source_ref") or updated.get("ref") or index)
            except (TypeError, ValueError):
                source_ref = index
            updated["source_ref"] = source_ref
            updated["ref"] = index
            rebased.append(updated)
        kept = rebased
    return kept, {
        "enforced": True,
        "input_count": len(original),
        "kept_count": len(kept),
        "filtered_count": max(0, len(original) - len(kept)),
    }


def filter_authorized_context_segments(
    segments: Iterable[Any],
    authorization: Any,
) -> tuple[list[dict], dict]:
    """Apply the same identity fence before context segments can recover citations."""
    return filter_authorized_citations(segments, authorization, rebase_refs=False)


def citation_authorization_summary(authorization: Any) -> dict:
    """Expose aggregate diagnostics without leaking the ledger's IDs to clients."""
    normalized = normalize_citation_authorization(authorization)
    authorized_count = sum(len(values) for values in normalized["authorized"].values())
    return {
        "policy": normalized["policy"] or CITATION_AUTHORIZATION_POLICY,
        "enforced": normalized["enforced"],
        "authorized_identity_count": authorized_count,
        "tool_counts": dict(normalized["tool_counts"]),
    }
