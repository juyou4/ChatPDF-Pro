"""Durable, parse-identity-bound visual assets derived from MinerU blocks.

The primary MinerU block index remains the source of truth.  This module only
derives reusable figure/table render metadata from that index; it never changes
block ids, parser route, parse generation, or source hash.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from schemas.figure_schema import LogicalFigureSchema


MINERU_VISUAL_ASSET_SCHEMA_VERSION = "mineru_visual_assets.v1"
MINERU_VISUAL_ASSET_KEY = "mineru_visual_assets"


def build_mineru_visual_asset_envelope(block_index: dict[str, Any] | None) -> dict[str, Any]:
    """Create a stable visual-asset envelope from a validated MinerU index.

    The caller is responsible for publishing the returned envelope together
    with the document record.  Returning an empty mapping instead of raising
    keeps visual derivation non-critical to the primary parse publication.
    """
    identity = _identity_from(block_index)
    if not _is_mineru_identity(identity):
        return {}

    try:
        from services.figure_extraction import build_mineru_logical_figures_from_block_index

        figures = build_mineru_logical_figures_from_block_index(block_index or {})
    except Exception:
        return {}

    source_blocks = _source_block_lookup(block_index or {})
    assets: list[dict[str, Any]] = []
    for figure in figures:
        asset = _asset_from_figure(figure, source_blocks, identity)
        if asset:
            assets.append(asset)

    assets.sort(key=lambda item: (
        int(item.get("page") or 0),
        float((item.get("bbox") or [0.0, 0.0, 0.0, 0.0])[1]),
        str(item.get("asset_id") or ""),
    ))
    revision = _stable_hash({
        "schema_version": MINERU_VISUAL_ASSET_SCHEMA_VERSION,
        "parser_route": identity["route"],
        "parse_generation": identity["generation"],
        "document_source_hash": identity["source_hash"],
        "assets": [
            {
                "asset_id": asset["asset_id"],
                "kind": asset["kind"],
                "page_spans": asset["page_spans"],
                "bbox": asset["bbox"],
                "panel_bboxes": asset["panel_bboxes"],
                "caption": asset["caption"],
                "table_html": asset["table_html"],
                "source_block_id": asset["source_block_id"],
                "source_block_ids": asset["source_block_ids"],
                "render_ref": asset["render_ref"],
            }
            for asset in assets
        ],
    })
    return {
        "schema_version": MINERU_VISUAL_ASSET_SCHEMA_VERSION,
        "parser_route": identity["route"],
        "parse_generation": identity["generation"],
        "document_source_hash": identity["source_hash"],
        "revision": revision,
        "assets": assets,
        "asset_count": len(assets),
    }


def active_mineru_visual_asset_envelope(
    data: dict[str, Any] | None,
    *,
    parse_identity: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return only an envelope that belongs to the current MinerU parse."""
    if not isinstance(data, dict):
        return None
    envelope = data.get(MINERU_VISUAL_ASSET_KEY)
    expected = _identity_from(parse_identity)
    if not isinstance(envelope, dict) or not _is_mineru_identity(expected):
        return None
    if envelope.get("schema_version") != MINERU_VISUAL_ASSET_SCHEMA_VERSION:
        return None
    actual = _identity_from(envelope)
    if actual != expected:
        return None
    assets = envelope.get("assets")
    if not isinstance(assets, list):
        return None
    return envelope


def resolve_mineru_visual_asset_envelope(
    data: dict[str, Any] | None,
    *,
    block_index: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Read a persisted envelope, lazily deriving a compatible legacy view.

    The lazy path is intentionally not written back from a request reader.  A
    new MinerU publication persists it; older documents still get the same
    canonical geometry without a read-path mutation.
    """
    expected = _identity_from(block_index)
    persisted = active_mineru_visual_asset_envelope(data, parse_identity=expected)
    if persisted is not None:
        return persisted
    derived = build_mineru_visual_asset_envelope(block_index)
    return derived or None


def active_mineru_visual_assets(
    data: dict[str, Any] | None,
    *,
    parse_identity: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    envelope = active_mineru_visual_asset_envelope(data, parse_identity=parse_identity)
    if envelope is None:
        return []
    identity = _identity_from(envelope)
    result = [
        copy.deepcopy(asset)
        for asset in envelope.get("assets") or []
        if _asset_matches_identity(asset, identity)
    ]
    result.sort(key=lambda item: (
        int(item.get("page") or 0),
        float((item.get("bbox") or [0.0, 0.0, 0.0, 0.0])[1]),
        str(item.get("asset_id") or ""),
    ))
    return result


def logical_figures_from_mineru_visual_assets(
    envelope: dict[str, Any] | None,
) -> list[LogicalFigureSchema]:
    """Adapt persisted assets back to the overview's shared figure schema."""
    if not isinstance(envelope, dict) or envelope.get("schema_version") != MINERU_VISUAL_ASSET_SCHEMA_VERSION:
        return []
    identity = _identity_from(envelope)
    if not _is_mineru_identity(identity):
        return []

    figures: list[LogicalFigureSchema] = []
    for asset in envelope.get("assets") or []:
        if not _asset_matches_identity(asset, identity):
            continue
        page = _positive_int(asset.get("page"))
        body_bbox = _bbox(asset.get("bbox"))
        if page <= 0 or not body_bbox:
            continue
        figure_id = _text(asset.get("source_block_id") or asset.get("asset_id"), 240)
        if not figure_id:
            continue
        figures.append(LogicalFigureSchema(
            figure_id=figure_id,
            page_idx=page - 1,
            figure_index=_text(asset.get("figure_id"), 240) or None,
            caption_text=_text(asset.get("caption"), 1600),
            body_bbox_page_pts=body_bbox,
            full_bbox_page_pts=_bbox(asset.get("full_bbox")) or body_bbox,
            panel_bboxes_page_pts=_panel_bboxes(asset.get("panel_bboxes")),
            source="mineru",
            confidence=_confidence(asset.get("confidence")) or 0.0,
            source_metadata={
                "adapter": "mineru_visual_assets",
                "asset_id": _text(asset.get("asset_id"), 240),
                "source_block_id": _text(asset.get("source_block_id"), 240),
                "render_ref": copy.deepcopy(asset.get("render_ref") or {}),
            },
        ))
    figures.sort(key=lambda item: (
        item.page_idx,
        float((item.body_bbox_page_pts or [0.0, 0.0, 0.0, 0.0])[1]),
        item.figure_id,
    ))
    return figures


def _asset_from_figure(
    figure: LogicalFigureSchema,
    source_blocks: dict[str, dict[str, Any]],
    identity: dict[str, str],
) -> dict[str, Any] | None:
    page = int(figure.page_idx) + 1
    bbox = _bbox(figure.body_bbox_page_pts) or _bbox(figure.full_bbox_page_pts)
    if page <= 0 or not bbox:
        return None
    source_block_id = _text(figure.figure_id, 240)
    source_block = source_blocks.get(source_block_id) or {}
    source_block_ids = _component_block_ids(
        figure,
        source_block_id=source_block_id,
        source_blocks=source_blocks,
    )
    kind = "table" if str(source_block.get("type") or "").strip().lower() == "table" else "figure"
    panel_bboxes = _panel_bboxes(figure.panel_bboxes_page_pts)
    full_bbox = _bbox(figure.full_bbox_page_pts) or bbox
    caption = _text(figure.caption_text, 1800)
    source_text = _text(source_block.get("text"), 12000)
    table_html = _table_html(source_block, source_text) if kind == "table" else ""
    if not caption and kind == "table":
        caption = _text(source_text, 1800)
    figure_id = _text(figure.figure_index, 240) or source_block_id
    asset_id = "mva_" + _stable_hash({
        "route": identity["route"],
        "generation": identity["generation"],
        "source_hash": identity["source_hash"],
        "kind": kind,
        "page": page,
        "source_block_id": source_block_id,
        "source_block_ids": source_block_ids,
        "figure_id": figure_id,
        "bbox": bbox,
        "panel_bboxes": panel_bboxes,
    }, length=32)
    render_ref = {
        "mode": "panel_composite" if len(panel_bboxes) >= 2 else "bbox_crop",
        "page": page,
        "bbox": list(bbox),
    }
    if panel_bboxes:
        render_ref["panel_bboxes"] = copy.deepcopy(panel_bboxes)
    return {
        "asset_id": asset_id,
        "figure_id": figure_id,
        "kind": kind,
        "page": page,
        "page_spans": [page],
        "bbox": list(bbox),
        "full_bbox": list(full_bbox),
        "panel_bboxes": panel_bboxes,
        "caption": caption,
        "text": _join_text(caption, source_text if kind == "table" else ""),
        "table_html": table_html,
        "source_block_id": source_block_id,
        "source_block_ids": source_block_ids,
        "render_ref": render_ref,
        "source": "mineru_visual_assets",
        "route": identity["route"],
        "parser_route": identity["route"],
        "generation": identity["generation"],
        "parse_generation": identity["generation"],
        "source_hash": identity["source_hash"],
        "document_source_hash": identity["source_hash"],
        "confidence": _confidence(figure.confidence),
    }


def _source_block_lookup(block_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for page in block_index.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            block_id = _text(block.get("block_id") or block.get("id"), 240)
            if block_id:
                lookup[block_id] = block
    return lookup


def _component_block_ids(
    figure: LogicalFigureSchema,
    *,
    source_block_id: str,
    source_blocks: dict[str, dict[str, Any]],
) -> list[str]:
    """Keep the original MinerU panel ids so consumers can collapse them."""
    ids: list[str] = [source_block_id] if source_block_id else []
    metadata = figure.source_metadata if isinstance(figure.source_metadata, dict) else {}
    for value in metadata.get("merged_from") or []:
        block_id = _text(value, 240)
        if block_id and block_id not in ids:
            ids.append(block_id)

    panel_keys = {
        tuple(round(value, 3) for value in bbox)
        for bbox in _panel_bboxes(figure.panel_bboxes_page_pts)
    }
    if panel_keys:
        for block_id, block in source_blocks.items():
            if str(block.get("type") or "").strip().lower() != "figure":
                continue
            bbox = _bbox(block.get("bbox"))
            if bbox and tuple(round(value, 3) for value in bbox) in panel_keys and block_id not in ids:
                ids.append(block_id)
    return ids


def _identity_from(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {"route": "", "generation": "", "source_hash": ""}
    route = _text(
        value.get("parser_route") or value.get("resolved_route") or value.get("route"),
        32,
    ).lower()
    if not route and _text(value.get("source"), 80).lower() == "mineru_vlm":
        route = "mineru"
    return {
        "route": route,
        "generation": _text(value.get("parse_generation") or value.get("generation"), 160),
        "source_hash": _text(
            value.get("document_source_hash") or value.get("source_hash"),
            256,
        ).lower(),
    }


def _is_mineru_identity(identity: dict[str, str]) -> bool:
    return bool(
        identity.get("route") == "mineru"
        and identity.get("generation")
        and identity.get("source_hash")
    )


def _asset_matches_identity(asset: Any, identity: dict[str, str]) -> bool:
    if not isinstance(asset, dict):
        return False
    return bool(
        _text(asset.get("asset_id"), 240)
        and _text(asset.get("route") or asset.get("parser_route"), 32).lower() == identity["route"]
        and _text(asset.get("generation") or asset.get("parse_generation"), 160) == identity["generation"]
        and _text(asset.get("source_hash") or asset.get("document_source_hash"), 256).lower() == identity["source_hash"]
    )


def _table_html(block: dict[str, Any], fallback: str) -> str:
    for key in ("table_html", "html", "markdown"):
        value = _text(block.get(key), 20000)
        if value:
            return value
    return fallback if fallback.lstrip().lower().startswith("<table") else ""


def _text(value: Any, limit: int = 1600) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _join_text(*values: str) -> str:
    return "\n".join(dict.fromkeys(value for value in values if value)).strip()


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _panel_bboxes(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        return []
    panels: list[list[float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for raw_bbox in value:
        bbox = _bbox(raw_bbox)
        if not bbox:
            continue
        key = tuple(round(item, 3) for item in bbox)
        if key not in seen:
            panels.append(bbox)
            seen.add(key)
    return panels


def _confidence(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _stable_hash(payload: dict[str, Any], length: int = 24) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
