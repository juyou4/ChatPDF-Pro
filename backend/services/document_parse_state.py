"""Pure state helpers for a document's primary parsing route.

The active parsing route is deliberately separate from parser artifacts.  A
document may have historical OCR, ODL, or MinerU artifacts, but all user-facing
features must consume the single route recorded by ``parse_manifest``.  The
manifest belongs in ``document[\"data\"][\"parse_manifest\"]``.

This module does not read files, start jobs, or mutate a document.  Route
handlers can therefore use it both before upload and when a background parser
publishes a new generation.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping


DOCUMENT_PARSE_STATE_VERSION = 1

PARSE_ROUTE_LOCAL = "local"
PARSE_ROUTE_MINERU = "mineru"
PARSE_ROUTE_AUTO = "auto"
PARSE_ROUTES = frozenset({PARSE_ROUTE_LOCAL, PARSE_ROUTE_MINERU, PARSE_ROUTE_AUTO})

PARSE_STATUS_PENDING = "pending"
PARSE_STATUS_QUEUED = "queued"
PARSE_STATUS_RUNNING = "running"
PARSE_STATUS_READY = "ready"
PARSE_STATUS_FAILED = "failed"
PARSE_STATUS_CANCELLED = "cancelled"
PARSE_STATUSES = frozenset({
    PARSE_STATUS_PENDING,
    PARSE_STATUS_QUEUED,
    PARSE_STATUS_RUNNING,
    PARSE_STATUS_READY,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_CANCELLED,
})


_ROUTE_ALIASES = {
    "": PARSE_ROUTE_AUTO,
    "auto": PARSE_ROUTE_AUTO,
    "automatic": PARSE_ROUTE_AUTO,
    "smart": PARSE_ROUTE_AUTO,
    "default": PARSE_ROUTE_AUTO,
    "自动": PARSE_ROUTE_AUTO,
    "local": PARSE_ROUTE_LOCAL,
    "native": PARSE_ROUTE_LOCAL,
    "pdf": PARSE_ROUTE_LOCAL,
    "pdf_native": PARSE_ROUTE_LOCAL,
    "original": PARSE_ROUTE_LOCAL,
    "odl": PARSE_ROUTE_LOCAL,
    "yolo": PARSE_ROUTE_LOCAL,
    "本地": PARSE_ROUTE_LOCAL,
    "原始": PARSE_ROUTE_LOCAL,
    "mineru": PARSE_ROUTE_MINERU,
    "mineru_rag": PARSE_ROUTE_MINERU,
    "mineru_deep": PARSE_ROUTE_MINERU,
    "mineru_deep_parse": PARSE_ROUTE_MINERU,
    "deep": PARSE_ROUTE_MINERU,
    "deep_parse": PARSE_ROUTE_MINERU,
    "深度": PARSE_ROUTE_MINERU,
    "深度解析": PARSE_ROUTE_MINERU,
}

_STATUS_ALIASES = {
    "": PARSE_STATUS_PENDING,
    "pending": PARSE_STATUS_PENDING,
    "planned": PARSE_STATUS_PENDING,
    "selected": PARSE_STATUS_PENDING,
    "draft": PARSE_STATUS_PENDING,
    "queued": PARSE_STATUS_QUEUED,
    "waiting": PARSE_STATUS_QUEUED,
    "running": PARSE_STATUS_RUNNING,
    "parsing": PARSE_STATUS_RUNNING,
    "processing": PARSE_STATUS_RUNNING,
    "ready": PARSE_STATUS_READY,
    "complete": PARSE_STATUS_READY,
    "completed": PARSE_STATUS_READY,
    "success": PARSE_STATUS_READY,
    "succeeded": PARSE_STATUS_READY,
    "failed": PARSE_STATUS_FAILED,
    "error": PARSE_STATUS_FAILED,
    "cancelled": PARSE_STATUS_CANCELLED,
    "canceled": PARSE_STATUS_CANCELLED,
}

_ALLOWED_TRANSITIONS = {
    PARSE_STATUS_PENDING: {
        PARSE_STATUS_PENDING,
        PARSE_STATUS_QUEUED,
        PARSE_STATUS_RUNNING,
        PARSE_STATUS_FAILED,
        PARSE_STATUS_CANCELLED,
    },
    PARSE_STATUS_QUEUED: {
        PARSE_STATUS_PENDING,
        PARSE_STATUS_QUEUED,
        PARSE_STATUS_RUNNING,
        PARSE_STATUS_FAILED,
        PARSE_STATUS_CANCELLED,
    },
    PARSE_STATUS_RUNNING: {
        # A process restart can preserve a completed parser artifact while
        # losing in-memory embedding credentials.  The route then returns to
        # pending at ``awaiting_rag_index`` instead of publishing a mixed run.
        PARSE_STATUS_PENDING,
        PARSE_STATUS_RUNNING,
        PARSE_STATUS_READY,
        PARSE_STATUS_FAILED,
        PARSE_STATUS_CANCELLED,
    },
    PARSE_STATUS_READY: {
        PARSE_STATUS_PENDING,
        PARSE_STATUS_QUEUED,
        PARSE_STATUS_RUNNING,
        PARSE_STATUS_READY,
    },
    PARSE_STATUS_FAILED: {
        PARSE_STATUS_PENDING,
        PARSE_STATUS_QUEUED,
        PARSE_STATUS_RUNNING,
        PARSE_STATUS_FAILED,
        PARSE_STATUS_CANCELLED,
    },
    PARSE_STATUS_CANCELLED: {
        PARSE_STATUS_PENDING,
        PARSE_STATUS_QUEUED,
        PARSE_STATUS_RUNNING,
        PARSE_STATUS_FAILED,
        PARSE_STATUS_CANCELLED,
    },
}


def normalize_parse_route(value: Any, *, default: str = PARSE_ROUTE_AUTO, strict: bool = False) -> str:
    """Return one of ``local``, ``mineru``, or ``auto``.

    Existing documents contain historical provider names such as ``pdf_native``
    and ``mineru_rag``.  They normalize to the two user-visible primary routes;
    YOLO and ODL remain local-route implementation details.
    """
    fallback = _normalize_route_default(default)
    if value is None:
        return fallback
    normalized = _ROUTE_ALIASES.get(str(value).strip().lower())
    if normalized:
        return normalized
    if strict:
        raise ValueError(f"Unsupported document parse route: {value!r}")
    return fallback


def normalize_parse_status(value: Any, *, default: str = PARSE_STATUS_PENDING, strict: bool = False) -> str:
    """Return a canonical parse lifecycle status."""
    fallback = _normalize_status_default(default)
    if value is None:
        return fallback
    normalized = _STATUS_ALIASES.get(str(value).strip().lower())
    if normalized:
        return normalized
    if strict:
        raise ValueError(f"Unsupported document parse status: {value!r}")
    return fallback


def derive_source_hash(value: Any) -> str:
    """Derive a stable SHA-256 source hash without reading from the filesystem.

    Upload handlers should pass the original PDF bytes.  Structured values are
    supported for legacy documents whose original bytes were not retained in
    the calling context.
    """
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, bytearray):
        payload = bytes(value)
    elif isinstance(value, memoryview):
        payload = value.tobytes()
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_parse_generation() -> str:
    """Return an opaque generation token for a new end-to-end parse run."""
    return f"parse-{uuid.uuid4().hex}"


def build_parse_manifest(
    *,
    doc_id: str,
    route: Any = PARSE_ROUTE_AUTO,
    source_hash: str = "",
    generation: str | None = None,
    status: Any = PARSE_STATUS_PENDING,
    resolved_route: Any = None,
    stage: str = "",
    error: str = "",
    now: str | datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned primary-route manifest.

    ``route`` is the user choice captured before upload.  A non-auto route is
    resolved immediately to itself.  ``auto`` stays unresolved until the upload
    handler applies its product policy (MinerU first when configured, otherwise
    local), preventing a provisional output from being mistaken for a ready
    document on another route.
    """
    requested_route = normalize_parse_route(route, strict=True)
    normalized_status = normalize_parse_status(status, strict=True)
    normalized_resolved = _normalize_resolved_route(resolved_route)
    if requested_route != PARSE_ROUTE_AUTO:
        if normalized_resolved and normalized_resolved != requested_route:
            raise ValueError(
                "An explicit primary parse route cannot resolve to a different parser"
            )
        normalized_resolved = requested_route
    if normalized_status == PARSE_STATUS_READY and not normalized_resolved:
        raise ValueError("A ready parse manifest requires a resolved primary route")

    timestamp = _timestamp(now)
    safe_doc_id = str(doc_id or "")
    safe_source_hash = str(source_hash or "").strip() or derive_source_hash({
        "doc_id": safe_doc_id,
        "route": requested_route,
    })
    return {
        "schema_version": DOCUMENT_PARSE_STATE_VERSION,
        "doc_id": safe_doc_id,
        # ``route`` remains concise for API consumers. ``requested_route`` is
        # explicit for future migrations and avoids overloading resolved_route.
        "route": requested_route,
        "requested_route": requested_route,
        "resolved_route": normalized_resolved or "",
        "generation": str(generation or new_parse_generation()),
        "source_hash": safe_source_hash,
        "status": normalized_status,
        "stage": str(stage or _default_stage(normalized_status)),
        "error": str(error or ""),
        "created_at": timestamp,
        "updated_at": timestamp,
        "metadata": dict(metadata or {}),
    }


def read_parse_manifest(document_or_data: Mapping[str, Any] | None, *, doc_id: str = "") -> dict[str, Any]:
    """Read and normalize a document manifest without mutating the input.

    For documents created before this state layer, a stable legacy manifest is
    inferred from existing ``data`` fields.  This lets callers progressively
    adopt the new contract without a bulk rewrite of saved documents.
    """
    document, data, inferred_doc_id = _split_document_data(document_or_data, doc_id)
    raw_manifest = data.get("parse_manifest")
    if isinstance(raw_manifest, Mapping):
        return _normalize_manifest(raw_manifest, data=data, doc_id=inferred_doc_id)
    # Accept the internal key during rollout, but write only parse_manifest.
    raw_manifest = data.get("document_parse_state")
    if isinstance(raw_manifest, Mapping):
        return _normalize_manifest(raw_manifest, data=data, doc_id=inferred_doc_id)
    return _legacy_manifest(document, data, inferred_doc_id)


def attach_parse_manifest(document: Mapping[str, Any] | None, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow document copy with the normalized manifest attached.

    The function intentionally does not mutate the supplied document, so a
    stale background job cannot modify an in-memory document by accident.
    """
    source = dict(document or {})
    raw_data = source.get("data")
    data = dict(raw_data) if isinstance(raw_data, Mapping) else {}
    inferred_doc_id = str(source.get("doc_id") or source.get("id") or "")
    normalized = _normalize_manifest(manifest, data=data, doc_id=inferred_doc_id)
    data["parse_manifest"] = normalized
    source["data"] = data
    return source


def transition_parse_manifest(
    manifest: Mapping[str, Any],
    status: Any,
    *,
    stage: str | None = None,
    error: str | None = None,
    resolved_route: Any = None,
    now: str | datetime | None = None,
) -> dict[str, Any]:
    """Return the next immutable manifest state after a validated transition."""
    current = _normalize_manifest(manifest, data={}, doc_id=str((manifest or {}).get("doc_id") or ""))
    next_status = normalize_parse_status(status, strict=True)
    current_status = current["status"]
    if next_status not in _ALLOWED_TRANSITIONS[current_status]:
        raise ValueError(f"Invalid parse manifest transition: {current_status} -> {next_status}")

    requested_route = current["requested_route"]
    current_resolved = current["resolved_route"]
    if resolved_route is None:
        next_resolved = current_resolved
    else:
        next_resolved = _normalize_resolved_route(resolved_route)
        if not next_resolved:
            raise ValueError("A resolved primary route must be local or mineru")
        if requested_route != PARSE_ROUTE_AUTO and next_resolved != requested_route:
            raise ValueError(
                "An explicit primary parse route cannot resolve to a different parser"
            )
        if current_resolved and next_resolved != current_resolved:
            raise ValueError("Changing parser route requires a new parse generation")

    if next_status == PARSE_STATUS_READY and not next_resolved:
        raise ValueError("A ready parse manifest requires a resolved primary route")

    updated = dict(current)
    updated["status"] = next_status
    updated["resolved_route"] = next_resolved
    updated["stage"] = str(stage if stage is not None else _default_stage(next_status))
    if error is not None:
        updated["error"] = str(error)
    elif next_status in {PARSE_STATUS_QUEUED, PARSE_STATUS_RUNNING, PARSE_STATUS_READY}:
        updated["error"] = ""
    updated["updated_at"] = _timestamp(now)
    if next_status == PARSE_STATUS_RUNNING and not updated.get("started_at"):
        updated["started_at"] = updated["updated_at"]
    if next_status == PARSE_STATUS_READY:
        updated["completed_at"] = updated["updated_at"]
    return updated


def start_new_parse_generation(
    manifest: Mapping[str, Any] | None,
    *,
    route: Any = None,
    source_hash: str | None = None,
    generation: str | None = None,
    now: str | datetime | None = None,
) -> dict[str, Any]:
    """Create a pending generation for a reparse or a deliberate route switch."""
    current = _normalize_manifest(manifest or {}, data={}, doc_id=str((manifest or {}).get("doc_id") or ""))
    selected_route = current["requested_route"] if route is None else route
    selected_hash = current["source_hash"] if source_hash is None else source_hash
    return build_parse_manifest(
        doc_id=current["doc_id"],
        route=selected_route,
        source_hash=str(selected_hash or ""),
        generation=generation,
        status=PARSE_STATUS_PENDING,
        stage="selected",
        now=now,
        metadata=current.get("metadata") if isinstance(current.get("metadata"), Mapping) else None,
    )


def is_parse_prepared(manifest: Mapping[str, Any] | None, *, route: Any = None) -> bool:
    """Whether a manifest represents a ready, fully resolved parse generation."""
    if not isinstance(manifest, Mapping):
        return False
    current = _normalize_manifest(manifest, data={}, doc_id=str(manifest.get("doc_id") or ""))
    if current["status"] != PARSE_STATUS_READY:
        return False
    if not current["generation"] or not current["source_hash"]:
        return False
    resolved_route = current["resolved_route"]
    if resolved_route not in {PARSE_ROUTE_LOCAL, PARSE_ROUTE_MINERU}:
        return False
    if route is None:
        return True
    expected = normalize_parse_route(route, strict=True)
    return expected == PARSE_ROUTE_AUTO or expected == resolved_route


def is_parse_ready(manifest: Mapping[str, Any] | None, *, route: Any = None) -> bool:
    """Compatibility alias for callers that use 'ready' terminology."""
    return is_parse_prepared(manifest, route=route)


def matches_parse_generation(
    manifest: Mapping[str, Any] | None,
    *,
    generation: str,
    source_hash: str | None = None,
) -> bool:
    """Check whether a worker result still belongs to the active generation."""
    if not isinstance(manifest, Mapping) or not generation:
        return False
    current = _normalize_manifest(manifest, data={}, doc_id=str(manifest.get("doc_id") or ""))
    if current["generation"] != str(generation):
        return False
    return source_hash is None or current["source_hash"] == str(source_hash)


PARSE_ARTIFACT_IDENTITY_KEYS = (
    "parse_generation",
    "document_source_hash",
    "block_index_hash",
)


def artifact_parse_identity(source: Mapping[str, Any] | None) -> dict[str, str]:
    """Read the parse identity an artifact recorded about itself.

    Accepts every envelope this repo actually persists: a vector pickle (whose
    identity lives under ``index_meta``), a block index, and a GraphRAG identity
    file. Missing fields come back as empty strings so callers never have to
    distinguish "absent" from "blank" — both mean "cannot prove it belongs here".
    """
    if not isinstance(source, Mapping):
        return {key: "" for key in PARSE_ARTIFACT_IDENTITY_KEYS}
    nested = source.get("index_meta")
    record: Mapping[str, Any] = nested if isinstance(nested, Mapping) else source
    return {
        "parse_generation": str(record.get("parse_generation") or "").strip(),
        "document_source_hash": str(record.get("document_source_hash") or "").strip(),
        "block_index_hash": str(
            record.get("block_index_hash") or record.get("block_index_revision") or ""
        ).strip(),
    }


def parse_identity_matches(
    artifact: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    *,
    block_index: Mapping[str, Any] | None = None,
    require_block_index_hash: bool = True,
) -> bool:
    """The single admission rule for any artifact bound to a parse run.

    Every consumer used to carry its own copy of this comparison and they had
    drifted: some checked ``(parse_generation, document_source_hash)`` only, so
    a parser repair that rebuilt the block tree *within* one generation left
    their artifacts admissible while the reading UI had already moved on — the
    same document then answered from stale block ids and section paths.

    ``block_index`` is the currently published block tree. Pass it whenever one
    exists; omitting it keeps a legacy artifact that predates immersive reading
    blocks usable. ``require_block_index_hash`` decides what to do when the
    published block tree carries no revision hash at all: reject (the default,
    matching the RAG index gate) or fall through on the generation pair alone.
    """
    if not isinstance(manifest, Mapping):
        return False
    expected_generation = str(manifest.get("generation") or "").strip()
    expected_source_hash = str(manifest.get("source_hash") or "").strip()
    if not expected_generation or not expected_source_hash:
        return False

    stored = artifact_parse_identity(artifact)
    if stored["parse_generation"] != expected_generation:
        return False
    if stored["document_source_hash"] != expected_source_hash:
        return False

    if not isinstance(block_index, Mapping):
        return True

    published = artifact_parse_identity(block_index)
    if (
        published["parse_generation"] != expected_generation
        or published["document_source_hash"] != expected_source_hash
    ):
        return False
    if not published["block_index_hash"]:
        return not require_block_index_hash
    return stored["block_index_hash"] == published["block_index_hash"]


def _normalize_manifest(raw: Mapping[str, Any], *, data: Mapping[str, Any], doc_id: str) -> dict[str, Any]:
    raw = raw if isinstance(raw, Mapping) else {}
    inferred_route = _legacy_route(data)
    requested_route = normalize_parse_route(
        raw.get("requested_route", raw.get("route", inferred_route)),
        default=inferred_route,
    )
    status = normalize_parse_status(raw.get("status"), default=PARSE_STATUS_PENDING)
    raw_resolved = raw.get("resolved_route", raw.get("active_route", raw.get("parser_route")))
    resolved_route = _normalize_resolved_route(raw_resolved)
    if requested_route != PARSE_ROUTE_AUTO:
        resolved_route = requested_route
    elif status == PARSE_STATUS_READY and not resolved_route:
        # A legacy ready record predates route selection. Infer a stable local
        # or MinerU source rather than treating the old document as ambiguous.
        resolved_route = inferred_route

    safe_doc_id = str(raw.get("doc_id") or doc_id or "")
    source_hash = str(raw.get("source_hash") or _legacy_source_hash(data, safe_doc_id)).strip()
    generation = str(raw.get("generation") or _legacy_generation(
        safe_doc_id,
        source_hash,
        requested_route,
    ))
    created_at = str(raw.get("created_at") or raw.get("updated_at") or _timestamp(None))
    updated_at = str(raw.get("updated_at") or created_at)
    metadata = raw.get("metadata")
    normalized = {
        "schema_version": DOCUMENT_PARSE_STATE_VERSION,
        "doc_id": safe_doc_id,
        "route": requested_route,
        "requested_route": requested_route,
        "resolved_route": resolved_route or "",
        "generation": generation,
        "source_hash": source_hash,
        "status": status,
        "stage": str(raw.get("stage") or _default_stage(status)),
        "error": str(raw.get("error") or ""),
        "created_at": created_at,
        "updated_at": updated_at,
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }
    for timestamp_key in ("started_at", "completed_at"):
        if raw.get(timestamp_key):
            normalized[timestamp_key] = str(raw[timestamp_key])
    return normalized


def _legacy_manifest(document: Mapping[str, Any], data: Mapping[str, Any], doc_id: str) -> dict[str, Any]:
    route = _legacy_route(data)
    source_hash = _legacy_source_hash(data, doc_id)
    status = PARSE_STATUS_READY if _legacy_document_has_parse_output(data) else PARSE_STATUS_PENDING
    # Legacy documents do not persist parser lifecycle timestamps.  Reading a
    # derived manifest must nevertheless be deterministic: status polling and
    # cache keys may read it repeatedly in the same request window.
    legacy_timestamp = str(
        data.get("parse_manifest_created_at")
        or document.get("upload_time")
        or data.get("upload_time")
        or "1970-01-01T00:00:00+00:00"
    )
    manifest = build_parse_manifest(
        doc_id=doc_id,
        route=route,
        source_hash=source_hash,
        generation=_legacy_generation(doc_id, source_hash, route),
        status=status,
        stage="legacy_ready" if status == PARSE_STATUS_READY else "selected",
        now=legacy_timestamp,
        metadata={"legacy_inferred": True},
    )
    # Top-level status data exists in a few earlier documents. Preserve it only
    # when it maps cleanly to this state model.
    top_level_status = document.get("parse_status") if isinstance(document, Mapping) else None
    if top_level_status is not None:
        manifest["status"] = normalize_parse_status(top_level_status, default=manifest["status"])
        manifest["stage"] = _default_stage(manifest["status"])
    return manifest


def _split_document_data(
    value: Mapping[str, Any] | None,
    explicit_doc_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    document = value if isinstance(value, Mapping) else {}
    nested_data = document.get("data")
    if isinstance(nested_data, Mapping):
        data = nested_data
    else:
        # Accept a data mapping directly, which keeps service call sites small.
        data = document
    doc_id = str(explicit_doc_id or document.get("doc_id") or document.get("id") or data.get("doc_id") or "")
    return document, data, doc_id


def _legacy_route(data: Mapping[str, Any]) -> str:
    candidates = [
        data.get("parse_route"),
        data.get("requested_parse_route"),
        data.get("rag_index_source"),
        data.get("block_index_source"),
        data.get("parse_provider"),
        data.get("extraction_method"),
    ]
    artifact = data.get("parse_artifact")
    if isinstance(artifact, Mapping):
        candidates.append(artifact.get("provider"))
    for candidate in candidates:
        normalized = normalize_parse_route(candidate, default=PARSE_ROUTE_AUTO)
        if normalized != PARSE_ROUTE_AUTO:
            return normalized
    return PARSE_ROUTE_LOCAL


def _legacy_source_hash(data: Mapping[str, Any], doc_id: str) -> str:
    candidates = [
        data.get("source_hash"),
        data.get("parse_source_hash"),
        data.get("rag_source_hash"),
    ]
    artifact = data.get("parse_artifact")
    if isinstance(artifact, Mapping):
        candidates.append(artifact.get("source_hash"))
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    # This fallback is only for old documents. New upload code must hash the
    # original PDF bytes before any local parser or OCR path can alter text.
    return derive_source_hash({
        "doc_id": doc_id,
        "full_text": data.get("full_text") or "",
        "pages": data.get("pages") or [],
    })


def _legacy_generation(doc_id: str, source_hash: str, route: str) -> str:
    identity = derive_source_hash({
        "doc_id": doc_id,
        "source_hash": source_hash,
        "route": route,
    })
    return f"legacy-{identity[:24]}"


def _legacy_document_has_parse_output(data: Mapping[str, Any]) -> bool:
    if str(data.get("full_text") or "").strip():
        return True
    pages = data.get("pages")
    if isinstance(pages, list) and pages:
        return True
    return False


def _normalize_resolved_route(value: Any) -> str:
    if value is None or not str(value).strip():
        return ""
    route = normalize_parse_route(value, strict=True)
    if route == PARSE_ROUTE_AUTO:
        raise ValueError("A resolved primary route cannot be auto")
    return route


def _normalize_route_default(default: str) -> str:
    normalized = _ROUTE_ALIASES.get(str(default or "").strip().lower())
    if normalized:
        return normalized
    raise ValueError(f"Unsupported document parse route default: {default!r}")


def _normalize_status_default(default: str) -> str:
    normalized = _STATUS_ALIASES.get(str(default or "").strip().lower())
    if normalized:
        return normalized
    raise ValueError(f"Unsupported document parse status default: {default!r}")


def _default_stage(status: str) -> str:
    return {
        PARSE_STATUS_PENDING: "selected",
        PARSE_STATUS_QUEUED: "queued",
        PARSE_STATUS_RUNNING: "parsing",
        PARSE_STATUS_READY: "ready",
        PARSE_STATUS_FAILED: "failed",
        PARSE_STATUS_CANCELLED: "cancelled",
    }[status]


def _timestamp(value: str | datetime | None) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
    else:
        value = datetime.now(timezone.utc)
    # 毫秒精度带时区，避免 JS Date 把 6 位微秒或无时区 ISO 解析成 Invalid / UTC。
    return value.isoformat(timespec="milliseconds")
