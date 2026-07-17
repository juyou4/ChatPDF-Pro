"""Structured, parse-bound visual supplements for the local reading route."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any


VISUAL_SUPPLEMENT_SCHEMA_VERSION = "v2"
VISUAL_SUPPLEMENT_SOURCE = "visual_vlm"
VISUAL_SUPPLEMENT_BLOCK_TYPE = "visual_enrichment"
VISUAL_SUPPLEMENT_DEFAULT_PURPOSE = "figure_description"
VISUAL_SUPPLEMENT_FIGURE_ANALYSIS_PROMPT_VERSION = "figure-analysis-v1"
VISUAL_SUPPLEMENT_FIGURE_ON_DEMAND_PROMPT_VERSION = "figure-on-demand-v1"
VISUAL_SUPPLEMENT_PAGE_RECOVERY_PROMPT_VERSION = "page-visual-recovery-v1"
VISUAL_SUPPLEMENT_DEFAULT_PROMPT_VERSION = VISUAL_SUPPLEMENT_FIGURE_ANALYSIS_PROMPT_VERSION
VISUAL_SUPPLEMENT_PROMPT_SUITE_IDENTITY = "|".join((
    VISUAL_SUPPLEMENT_FIGURE_ANALYSIS_PROMPT_VERSION,
    VISUAL_SUPPLEMENT_FIGURE_ON_DEMAND_PROMPT_VERSION,
    VISUAL_SUPPLEMENT_PAGE_RECOVERY_PROMPT_VERSION,
))
VISUAL_SUPPLEMENT_COMMIT_SCHEMA_VERSION = "v2"
VISUAL_SUPPLEMENT_COMMIT_MARKER_KEY = "visual_supplement_commit"
DEFAULT_VISUAL_EVIDENCE_SNAPSHOT_LIMIT = 8
_ACTIVE_PROMPT_VERSIONS_BY_PURPOSE = {
    "figure_description": frozenset({
        VISUAL_SUPPLEMENT_FIGURE_ANALYSIS_PROMPT_VERSION,
        VISUAL_SUPPLEMENT_FIGURE_ON_DEMAND_PROMPT_VERSION,
    }),
    "scan_region_recognition": frozenset({
        VISUAL_SUPPLEMENT_PAGE_RECOVERY_PROMPT_VERSION,
    }),
}


def _text(value: Any, limit: int = 1600) -> str:
    normalized = " ".join(str(value or "").split())
    return normalized[:limit].strip()


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _stable_hash(payload: dict[str, Any], length: int = 24) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _bbox_hash(value: Any) -> str:
    normalized = _bbox(value)
    if not normalized:
        return ""
    return _stable_hash({"bbox": [round(item, 4) for item in normalized]}, 24)


def _confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _normalized_visual_risk(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    reasons = [
        _text(reason, 80)
        for reason in (value.get("reasons") or [])[:12]
        if _text(reason, 80)
    ] if isinstance(value.get("reasons"), (list, tuple)) else []
    try:
        page = max(0, int(value.get("page") or 0))
    except (TypeError, ValueError):
        page = 0
    risk = {
        "should_enrich": bool(value.get("should_enrich")),
        "score": _confidence(value.get("score")),
        "level": _text(value.get("level"), 16),
        "reasons": reasons,
        "page": page,
        "bbox_hash": _text(value.get("bbox_hash"), 64),
        "source": _text(value.get("source"), 40),
        "structure_confidence": _confidence(value.get("structure_confidence")),
        "ocr_confidence": _confidence(value.get("ocr_confidence")),
    }
    return risk if risk["level"] or risk["reasons"] or risk["score"] is not None else None


def _identity_matches(envelope: dict[str, Any], parse_identity: dict[str, Any]) -> bool:
    if str(parse_identity.get("parser_route") or "").lower() != "local":
        return False
    return (
        str(envelope.get("parse_generation") or "")
        == str(parse_identity.get("parse_generation") or "")
        and str(envelope.get("document_source_hash") or "")
        == str(parse_identity.get("document_source_hash") or "")
    )


def _active_visual_supplement_envelope(
    data: dict[str, Any], parse_identity: dict[str, Any]
) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    envelope = data.get("visual_supplements")
    if not isinstance(envelope, dict) or envelope.get("schema_version") != VISUAL_SUPPLEMENT_SCHEMA_VERSION:
        return None
    if not _identity_matches(envelope, parse_identity):
        return None
    return envelope


def active_visual_supplements(
    data: dict[str, Any],
    parse_identity: dict[str, Any],
    *,
    require_committed: bool = True,
) -> list[dict[str, Any]]:
    """Return supplements owned by the active local parse generation.

    Public readers must not observe a VLM response while its matching reading
    block index is still being published.  The publisher is the only caller
    allowed to opt into the short-lived staging view.
    """
    envelope = _active_visual_supplement_envelope(data, parse_identity)
    if not envelope or (
        require_committed
        and not visual_supplements_are_committed(data, parse_identity=parse_identity)
    ):
        return []
    return _current_visual_evidence_items(envelope)


def visual_supplement_revision(
    data: dict[str, Any],
    parse_identity: dict[str, Any],
    *,
    require_committed: bool = True,
) -> str:
    """Return the published revision, or an explicit publisher staging revision."""
    envelope = _active_visual_supplement_envelope(data, parse_identity)
    if not envelope or (
        require_committed
        and not visual_supplements_are_committed(data, parse_identity=parse_identity)
    ):
        return ""
    return _active_visual_revision(envelope)


def _commit_marker_for_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    revision = str(envelope.get("revision") or "").strip()
    if not revision:
        return {}
    return {
        "schema_version": VISUAL_SUPPLEMENT_COMMIT_SCHEMA_VERSION,
        "parser_route": "local",
        "parse_generation": str(envelope.get("parse_generation") or ""),
        "document_source_hash": str(envelope.get("document_source_hash") or ""),
        "visual_supplement_revision": revision,
        "visual_model_identity": str(envelope.get("visual_model_identity") or ""),
    }


def _commit_marker_matches_envelope(marker: Any, envelope: dict[str, Any]) -> bool:
    if not isinstance(marker, dict):
        return False
    expected = _commit_marker_for_envelope(envelope)
    return bool(expected) and all(str(marker.get(key) or "") == str(value or "") for key, value in expected.items())


def mark_visual_supplements_committed(
    data: dict[str, Any], *, parse_identity: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Record that the current local visual envelope was published to retrieval."""
    envelope = _active_visual_supplement_envelope(data, parse_identity)
    if not envelope:
        return False, {}

    marker = _commit_marker_for_envelope(envelope)
    if not marker:
        return False, {}
    existing = data.get(VISUAL_SUPPLEMENT_COMMIT_MARKER_KEY)
    if _commit_marker_matches_envelope(existing, envelope):
        return False, dict(marker)

    marker["committed_at"] = time.time()
    data[VISUAL_SUPPLEMENT_COMMIT_MARKER_KEY] = marker
    return True, dict(marker)


def visual_supplements_are_committed(
    data: dict[str, Any], *, parse_identity: dict[str, Any]
) -> bool:
    """Whether the active local envelope has a matching durable publication marker."""
    envelope = _active_visual_supplement_envelope(data, parse_identity)
    return bool(envelope and _commit_marker_matches_envelope(
        data.get(VISUAL_SUPPLEMENT_COMMIT_MARKER_KEY), envelope
    ))


def _normalized_visual_evidence(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    item_id = _text(item.get("id"), 160)
    try:
        page = int(item.get("page") or 0)
    except (TypeError, ValueError):
        return None
    bbox = _bbox(item.get("bbox"))
    analysis = _text(item.get("analysis"))
    text = _text(item.get("text") or analysis)
    if not item_id or page <= 0 or not bbox or not text:
        return None

    evidence: dict[str, Any] = {
        "id": item_id,
        "page": page,
        "bbox": bbox,
        "bbox_hash": _text(item.get("bbox_hash"), 64) or _bbox_hash(bbox),
        "text": text,
        "source": VISUAL_SUPPLEMENT_SOURCE,
        "route": _text(item.get("route"), 24) or "local",
        "block_type": _text(item.get("block_type"), 64) or VISUAL_SUPPLEMENT_BLOCK_TYPE,
        "purpose": _text(item.get("purpose"), 80) or VISUAL_SUPPLEMENT_DEFAULT_PURPOSE,
        "prompt_version": (
            _text(item.get("prompt_version"), 80)
            or VISUAL_SUPPLEMENT_DEFAULT_PROMPT_VERSION
        ),
        "confidence": _confidence(item.get("confidence")),
    }
    figure_id = _text(item.get("figure_id"), 160)
    caption = _text(item.get("caption"), 400)
    render_mode = _text(item.get("render_mode"), 32)
    if figure_id:
        evidence["figure_id"] = figure_id
    if caption:
        evidence["caption"] = caption
    if analysis:
        evidence["analysis"] = analysis
    if render_mode:
        evidence["render_mode"] = render_mode
    visual_risk = _normalized_visual_risk(item.get("visual_risk"))
    if visual_risk:
        evidence["visual_risk"] = visual_risk

    provider = _text(item.get("provider"), 80)
    model_name = _text(item.get("model"), 160)

    model = item.get("visual_model")
    if isinstance(model, dict):
        normalized_model = {
            "identity": _text(model.get("identity"), 80),
            "provider": _text(model.get("provider"), 80),
            "model": _text(model.get("model"), 160),
        }
        if any(normalized_model.values()):
            evidence["visual_model"] = normalized_model
            provider = provider or normalized_model["provider"]
            model_name = model_name or normalized_model["model"]
    evidence["provider"] = provider
    evidence["model"] = model_name
    return evidence


def _visual_prompt_is_current(item: dict[str, Any]) -> bool:
    purpose = _text(item.get("purpose"), 80) or VISUAL_SUPPLEMENT_DEFAULT_PURPOSE
    allowed = _ACTIVE_PROMPT_VERSIONS_BY_PURPOSE.get(purpose)
    if allowed is None:
        return True
    prompt_version = (
        _text(item.get("prompt_version"), 80)
        or VISUAL_SUPPLEMENT_DEFAULT_PROMPT_VERSION
    )
    return prompt_version in allowed


def _current_visual_evidence_items(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        normalized
        for item in envelope.get("items") or []
        if (normalized := _normalized_visual_evidence(item)) is not None
        and _visual_prompt_is_current(normalized)
    ]
    items.sort(
        key=lambda item: (
            int(item.get("page") or 0),
            float((item.get("bbox") or [0, 0, 0, 0])[1]),
            str(item.get("id") or ""),
        )
    )
    return items


def _active_visual_revision(envelope: dict[str, Any]) -> str:
    items = _current_visual_evidence_items(envelope)
    if not items:
        return ""
    return _stable_hash({
        "schema_version": VISUAL_SUPPLEMENT_SCHEMA_VERSION,
        "parser_route": "local",
        "parse_generation": str(envelope.get("parse_generation") or ""),
        "document_source_hash": str(envelope.get("document_source_hash") or ""),
        "visual_model_identity": str(envelope.get("visual_model_identity") or ""),
        "items": items,
    }, 24)


def committed_visual_evidence_snapshot(
    data: dict[str, Any],
    *,
    parse_identity: dict[str, Any],
    limit: int = DEFAULT_VISUAL_EVIDENCE_SNAPSHOT_LIMIT,
) -> list[dict[str, Any]]:
    """Return bounded, normalized visual evidence only after matching publication."""
    envelope = _active_visual_supplement_envelope(data, parse_identity)
    if not envelope:
        return []
    if not visual_supplements_are_committed(data, parse_identity=parse_identity):
        return []
    try:
        bounded_limit = max(0, int(limit))
    except (TypeError, ValueError):
        return []
    if not bounded_limit:
        return []

    revision = _active_visual_revision(envelope)
    if not revision:
        return []
    evidence = [
        {
            **item,
            "visual_supplement_revision": revision,
        }
        for item in _current_visual_evidence_items(envelope)
    ]
    evidence.sort(
        key=lambda item: (
            int(item["page"]),
            float(item["bbox"][1]),
            str(item["id"]),
        )
    )
    return evidence[:bounded_limit]


def committed_visual_evidence_for_document(
    document: dict[str, Any],
    *,
    limit: int = DEFAULT_VISUAL_EVIDENCE_SNAPSHOT_LIMIT,
) -> list[dict[str, Any]]:
    """Read a local document's committed visual evidence without route imports."""
    if not isinstance(document, dict):
        return []
    data = document.get("data")
    if not isinstance(data, dict):
        return []
    manifest = data.get("parse_manifest")
    if not isinstance(manifest, dict):
        return []
    return committed_visual_evidence_snapshot(
        data,
        parse_identity={
            "parser_route": str(manifest.get("resolved_route") or ""),
            "parse_generation": str(manifest.get("generation") or ""),
            "document_source_hash": str(manifest.get("source_hash") or ""),
        },
        limit=limit,
    )


def build_visual_supplement(
    *,
    figure_id: str,
    page: int,
    bbox: Any,
    caption: str,
    analysis: str,
    visual_model_identity: str,
    provider: str,
    model: str,
    render_mode: str,
    purpose: str = VISUAL_SUPPLEMENT_DEFAULT_PURPOSE,
    confidence: Any = None,
    prompt_version: str = VISUAL_SUPPLEMENT_DEFAULT_PROMPT_VERSION,
    route: str = "local",
) -> dict[str, Any] | None:
    """Normalize a VLM figure reading into a stable, block-index-ready item."""
    normalized_bbox = _bbox(bbox)
    normalized_analysis = _text(analysis)
    page_number = int(page or 0)
    if not normalized_bbox or page_number <= 0 or not normalized_analysis:
        return None
    normalized_caption = _text(caption, 400)
    normalized_route = _text(route, 24).lower() or "local"
    if normalized_route != "local":
        return None
    normalized_purpose = _text(purpose, 80) or VISUAL_SUPPLEMENT_DEFAULT_PURPOSE
    normalized_prompt_version = (
        _text(prompt_version, 80) or VISUAL_SUPPLEMENT_DEFAULT_PROMPT_VERSION
    )
    normalized_bbox_hash = _bbox_hash(normalized_bbox)
    normalized_provider = _text(provider, 80)
    normalized_model = _text(model, 160)
    stable = {
        "figure_id": _text(figure_id, 160),
        "page": page_number,
        "bbox_hash": normalized_bbox_hash,
        "purpose": normalized_purpose,
        "route": normalized_route,
    }
    item_id = f"visual_vlm_{_stable_hash(stable, 18)}"
    content = normalized_analysis
    if normalized_caption and normalized_caption not in content:
        content = f"{normalized_caption}. {content}"
    return {
        "id": item_id,
        "figure_id": stable["figure_id"],
        "page": page_number,
        "bbox": normalized_bbox,
        "bbox_hash": normalized_bbox_hash,
        "caption": normalized_caption,
        "analysis": normalized_analysis,
        "text": _text(content),
        "source": VISUAL_SUPPLEMENT_SOURCE,
        "route": normalized_route,
        "block_type": VISUAL_SUPPLEMENT_BLOCK_TYPE,
        "purpose": normalized_purpose,
        "confidence": _confidence(confidence),
        "provider": normalized_provider,
        "model": normalized_model,
        "prompt_version": normalized_prompt_version,
        "visual_model": {
            "identity": _text(visual_model_identity, 80),
            "provider": normalized_provider,
            "model": normalized_model,
        },
        "render_mode": _text(render_mode, 32),
    }


def upsert_visual_supplements(
    data: dict[str, Any],
    *,
    parse_identity: dict[str, Any],
    visual_model_identity: str,
    items: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Merge current-model VLM items and return whether the envelope changed."""
    if str(parse_identity.get("parser_route") or "").lower() != "local":
        return False, {}

    parse_generation = str(parse_identity.get("parse_generation") or "").strip()
    document_source_hash = str(parse_identity.get("document_source_hash") or "").strip()
    if not parse_generation or not document_source_hash:
        return False, {}

    existing = data.get("visual_supplements") if isinstance(data, dict) else None
    existing_items: list[dict[str, Any]] = []
    if (
        isinstance(existing, dict)
        and _identity_matches(existing, parse_identity)
        and str(existing.get("visual_model_identity") or "") == str(visual_model_identity or "")
    ):
        existing_items = _current_visual_evidence_items(existing)

    by_id = {str(item.get("id") or ""): item for item in existing_items if item.get("id")}
    for item in items:
        normalized = _normalized_visual_evidence(item)
        if normalized is None or not _visual_prompt_is_current(normalized):
            continue
        by_id[str(normalized["id"])] = normalized
    merged_items = sorted(
        by_id.values(),
        key=lambda item: (
            int(item.get("page") or 0),
            float((item.get("bbox") or [0, 0, 0, 0])[1]),
            str(item.get("id") or ""),
        ),
    )
    revision_payload = {
        "schema_version": VISUAL_SUPPLEMENT_SCHEMA_VERSION,
        "parser_route": "local",
        "parse_generation": parse_generation,
        "document_source_hash": document_source_hash,
        "visual_model_identity": str(visual_model_identity or ""),
        "items": merged_items,
    }
    revision = _stable_hash(revision_payload, 24)
    envelope = {
        **revision_payload,
        "revision": revision,
        "updated_at": time.time(),
    }
    previous_revision = str(existing.get("revision") or "") if isinstance(existing, dict) else ""
    changed = previous_revision != revision
    if changed:
        data["visual_supplements"] = envelope
    return changed, envelope
