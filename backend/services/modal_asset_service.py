"""构建与当前解析身份绑定的请求内模态资产视图。"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any


MODAL_ASSET_INDEX_VERSION = "v1"
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 8

_BASE_KINDS = {"figure", "table", "formula"}
_VISUAL_KIND = "visual_enrichment"
_KIND_ALIASES = {
    "fig": "figure",
    "figure": "figure",
    "image": "figure",
    "picture": "figure",
    "chart": "figure",
    "table": "table",
    "formula": "formula",
    "equation": "formula",
    "eq": "formula",
    "visual": _VISUAL_KIND,
    "visual_enrichment": _VISUAL_KIND,
}

_REFERENCE_NUMBER_PATTERN = r"\d+(?:\s*\.\s*\d+)*(?:[a-z]|\s*\([a-z]\))?"
_REFERENCE_PATTERN = re.compile(
    r"(?P<label>fig(?:ure)?s?|figs?|图表|图|tables?|tabs?|表格|表|"
    r"equations?|eqs?|formulas?|公式|方程|式)"
    r"\s*(?:no\.?\s*)?[:：#.]?\s*"
    rf"(?P<numbers>{_REFERENCE_NUMBER_PATTERN}(?:\s*(?:,|，|、|;|；|&|and|和|与|及|"
    rf"to|through|至|到|[-–—])\s*{_REFERENCE_NUMBER_PATTERN})*)",
    re.IGNORECASE,
)
_REFERENCE_TOKEN_PATTERN = re.compile(_REFERENCE_NUMBER_PATTERN, re.IGNORECASE)
_REFERENCE_RANGE_PATTERN = re.compile(
    rf"^\s*(?P<start>\d+)\s*(?:to|through|至|到|[-–—])\s*(?P<end>\d+)\s*$",
    re.IGNORECASE,
)
_PAGE_PATTERN = re.compile(
    r"(?:第\s*(?P<zh>\d+)\s*页|\bpage\s*(?P<en>\d+)\b|\bp\.?\s*(?P<short>\d+)\b)",
    re.IGNORECASE,
)
_CAPTION_PREFIX_PATTERN = re.compile(
    r"^\s*(?:fig(?:ure)?|图表|图|table|tab(?:le)?|表格|表)\s*\d+",
    re.IGNORECASE,
)
_VISUAL_QUERY_PATTERN = re.compile(
    r"(?:图片|图像|图表|插图|曲线图|柱状图|折线图|散点图|流程图|示意图|截图|"
    r"表格|公式|方程|坐标轴|图例|布局|排版|视觉|左上角|右上角|左下角|右下角|"
    r"\b(?:figure|fig|image|picture|chart|plot|diagram|table|formula|equation|layout|"
    r"screenshot|visual|axis|legend)s?\b)",
    re.IGNORECASE,
)
_FIGURE_QUERY_PATTERN = re.compile(
    r"(?:图片|图像|图表|插图|曲线图|柱状图|折线图|散点图|流程图|示意图|截图|"
    r"坐标轴|图例|"
    r"\b(?:figure|fig|image|picture|chart|plot|diagram|curve|screenshot|axis|legend)s?\b)",
    re.IGNORECASE,
)

_SEARCH_STOPWORDS = {
    "a", "an", "and", "are", "in", "is", "of", "on", "page", "show", "the", "what",
    "figure", "fig", "table", "formula", "equation", "图", "表", "页", "图片", "图表", "表格",
}


def build_modal_asset_index(
    *,
    block_index: dict | None,
    visual_evidence: list[dict] | None = None,
) -> dict:
    """从块索引和请求内视觉证据构建纯内存资产索引。"""
    identity = _parse_identity(block_index)
    result = _empty_index(identity)
    if not isinstance(block_index, dict):
        return result

    pages = block_index.get("pages")
    if not isinstance(pages, list):
        return result

    sections = _section_lookup(block_index.get("outline"))
    assets: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    supplements: dict[str, dict[str, Any]] = {}
    block_to_asset: dict[str, dict[str, Any]] = {}

    for page_record in pages:
        if not isinstance(page_record, dict):
            continue
        page_number = _positive_int(page_record.get("page"))
        blocks = page_record.get("blocks")
        if page_number <= 0 or not isinstance(blocks, list):
            continue

        for block in blocks:
            if not isinstance(block, dict):
                continue
            raw_kind, kind = _block_kind(block)
            if kind == _VISUAL_KIND:
                supplement = _normalize_supplement(block, page_number, identity)
                if supplement:
                    _upsert_supplement(supplements, supplement)
                continue
            if kind in _BASE_KINDS:
                asset = _asset_from_block(
                    block,
                    page=page_number,
                    kind=kind,
                    source_kind=raw_kind,
                    identity=identity,
                    sections=sections,
                )
                assets.append(asset)
                if asset["block_id"]:
                    block_to_asset[asset["block_id"]] = asset
                continue
            if _normalized_token(block.get("type")) == "caption":
                caption = _normalize_caption_block(block, page_number)
                if caption:
                    captions.append(caption)

    for evidence in visual_evidence or []:
        supplement = _normalize_supplement(evidence, 0, identity)
        if supplement and _evidence_matches_identity(supplement, identity):
            _upsert_supplement(supplements, supplement)

    for supplement in sorted(supplements.values(), key=_supplement_sort_key):
        target = _find_supplement_target(supplement, assets, block_to_asset)
        if target is None:
            asset = _asset_from_supplement(supplement, identity, sections, block_to_asset)
            assets.append(asset)
            if asset["block_id"]:
                block_to_asset.setdefault(asset["block_id"], asset)
            continue
        _merge_supplement(target, supplement, identity)

    _bind_captions(captions, assets, block_to_asset)
    for asset in assets:
        asset["relations"] = _dedupe_relations(asset.get("relations"))
        asset["references"] = sorted(_asset_references(asset))
        asset["visual_provenance"] = _dedupe_provenance(asset.get("visual_provenance"))

    assets.sort(key=_asset_sort_key)
    result["assets"] = assets
    result["asset_count"] = len(assets)
    result["page_count"] = len({asset["page"] for asset in assets if asset["page"] > 0})
    result["index_id"] = _stable_hash({
        "version": MODAL_ASSET_INDEX_VERSION,
        "route": identity["route"],
        "generation": identity["generation"],
        "source_hash": identity["source_hash"],
        "revision": identity["revision"],
        "asset_ids": [asset["asset_id"] for asset in assets],
    })
    return result


def search_modal_assets(
    index: dict,
    *,
    query: str = "",
    reference: str = "",
    page: int = 0,
    kinds: list[str] | None = None,
    limit: int = 5,
) -> list[dict]:
    """按编号、页码和文字证据对模态资产进行确定性排序。"""
    if not isinstance(index, dict):
        return []
    assets = index.get("assets")
    if not isinstance(assets, list):
        return []

    bounded_limit = _bounded_limit(limit)
    if bounded_limit <= 0:
        return []
    requested_page = _positive_int(page)
    allowed_kinds = _normalize_kind_filter(kinds)
    query_text = _clean_text(query, 2400)
    reference_text = _clean_text(reference, 400)
    requested_references = _extract_references(f"{reference_text} {query_text}")
    loose_reference = _loose_key(reference_text)
    query_pages = _extract_pages(f"{reference_text} {query_text}")
    terms = _search_terms(query_text)

    ranked: list[tuple[float, int, float, float, str, dict[str, Any]]] = []
    for raw_asset in assets:
        if not isinstance(raw_asset, dict):
            continue
        kind = _normalize_kind(raw_asset.get("kind"))
        asset_page = _positive_int(raw_asset.get("page"))
        if allowed_kinds is not None and kind not in allowed_kinds:
            continue
        if requested_page > 0 and asset_page != requested_page:
            continue
        if requested_references and not (requested_references & _asset_references(raw_asset)):
            continue

        score, reasons = _score_asset(
            raw_asset,
            query=query_text,
            reference=reference_text,
            references=requested_references,
            loose_reference=loose_reference,
            query_pages=query_pages,
            terms=terms,
        )
        if requested_page > 0:
            score += 30.0
            reasons.append("page_filter")
        result = copy.deepcopy(raw_asset)
        result["score"] = round(score, 6)
        result["match_reasons"] = reasons
        bbox = _normalize_bbox(raw_asset.get("bbox")) or [float("inf")] * 4
        ranked.append((
            -score,
            asset_page or 10**9,
            bbox[1],
            bbox[0],
            str(raw_asset.get("asset_id") or ""),
            result,
        ))

    has_text_intent = bool(query_text or reference_text)
    total_asset_count = sum(1 for item in assets if isinstance(item, dict))
    if has_text_intent and ranked and not any(item[0] < 0 for item in ranked):
        if total_asset_count == 1 and len(ranked) == 1 and looks_like_visual_query(f"{reference_text} {query_text}"):
            ranked[0][5]["match_reasons"] = ["single_visual_fallback"]
        else:
            return []

    ranked.sort(key=lambda item: item[:5])
    return [item[5] for item in ranked[:bounded_limit]]


def looks_like_figure_query(query: str) -> bool:
    """Return whether a question explicitly asks about a Figure-like asset."""
    normalized = _clean_text(query, 2400)
    if not normalized:
        return False
    if any(reference.startswith("figure:") for reference in _extract_references(normalized)):
        return True
    return bool(_FIGURE_QUERY_PATTERN.search(normalized))


def looks_like_visual_query(query: str) -> bool:
    """判断问题是否明确需要图像、表格、公式或页面视觉证据。"""
    normalized = _clean_text(query, 2400)
    if not normalized:
        return False
    if _extract_references(normalized):
        return True
    if _VISUAL_QUERY_PATTERN.search(normalized):
        return True
    return bool(
        _extract_pages(normalized)
        and re.search(r"(?:位置|区域|上方|下方|左侧|右侧|角落|看起来|显示|标注|where|position|region)", normalized, re.IGNORECASE)
    )


def detect_query_modalities(query: str) -> list[str]:
    """返回稳定、有序的查询模态，不执行任何视觉模型调用。"""
    normalized = _clean_text(query, 2400)
    if not normalized:
        return ["text"]

    references = _extract_references(normalized)
    modalities: list[str] = []
    if (
        any(item.startswith("figure:") for item in references)
        or _FIGURE_QUERY_PATTERN.search(normalized)
    ):
        modalities.append("figure")
    if (
        any(item.startswith("table:") for item in references)
        or re.search(r"(?:表格|\btable?s?\b)", normalized, re.IGNORECASE)
    ):
        modalities.append("table")
    if (
        any(item.startswith("formula:") for item in references)
        or re.search(r"(?:公式|方程|\b(?:formula|equation)s?\b)", normalized, re.IGNORECASE)
    ):
        modalities.append("formula")
    if re.search(
        r"(?:布局|排版|位置|区域|上方|下方|左侧|右侧|角落|"
        r"\b(?:layout|position|region|top|bottom|left|right)\b)",
        normalized,
        re.IGNORECASE,
    ):
        modalities.append("layout")
    if not modalities:
        return ["text"]
    return modalities


def _empty_index(identity: dict[str, str]) -> dict[str, Any]:
    return {
        "version": MODAL_ASSET_INDEX_VERSION,
        "route": identity["route"],
        "generation": identity["generation"],
        "source_hash": identity["source_hash"],
        "revision": identity["revision"],
        "parser_route": identity["route"],
        "parse_generation": identity["generation"],
        "document_source_hash": identity["source_hash"],
        "visual_supplement_revision": identity["revision"],
        "source": identity["source"],
        "assets": [],
        "asset_count": 0,
        "page_count": 0,
        "index_id": "",
    }


def _parse_identity(block_index: Any) -> dict[str, str]:
    if not isinstance(block_index, dict):
        return {"route": "", "generation": "", "source_hash": "", "revision": "", "source": ""}
    source = _clean_text(block_index.get("source"), 80)
    route = _clean_text(block_index.get("parser_route") or block_index.get("route"), 32).lower()
    if not route and source.lower() == "mineru_vlm":
        route = "mineru"
    return {
        "route": route,
        "generation": _clean_text(block_index.get("parse_generation") or block_index.get("generation"), 160),
        "source_hash": _clean_text(block_index.get("document_source_hash") or block_index.get("source_hash"), 256),
        "revision": _clean_text(block_index.get("visual_supplement_revision") or block_index.get("revision"), 160),
        "source": source,
    }


def _section_lookup(value: Any) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    if not isinstance(value, list):
        return sections
    for item in value:
        if not isinstance(item, dict):
            continue
        section_id = _clean_text(item.get("section_id"), 160)
        if section_id:
            sections[section_id] = {
                "section_id": section_id,
                "title": _clean_text(item.get("title"), 400),
                "page": _positive_int(item.get("page")),
            }
    return sections


def _block_kind(block: dict[str, Any]) -> tuple[str, str]:
    candidates = [block.get("block_type"), block.get("type"), block.get("kind")]
    for value in candidates:
        raw = _normalized_token(value)
        if raw == _VISUAL_KIND:
            return raw, _VISUAL_KIND
    for value in candidates:
        raw = _normalized_token(value)
        kind = _normalize_kind(raw)
        if kind in _BASE_KINDS:
            return raw, kind
    return "", ""


def _asset_from_block(
    block: dict[str, Any],
    *,
    page: int,
    kind: str,
    source_kind: str,
    identity: dict[str, str],
    sections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    block_id = _clean_text(block.get("block_id") or block.get("id"), 240)
    owner_block_id = _clean_text(block.get("owner_block_id"), 240) or block_id
    figure_id = _clean_text(
        block.get("figure_id") or block.get("table_id") or block.get("equation_id"),
        240,
    )
    bbox = _normalize_bbox(block.get("bbox")) or []
    text = _first_text(block, ("text", "content", "latex", "html"), 6000)
    caption = _first_text(
        block,
        ("caption", "caption_text", "figure_caption", "table_caption", "image_caption"),
        1200,
    )
    if not caption and kind in {"figure", "table"} and _looks_like_caption(text):
        caption = text
    description = _first_text(block, ("description", "analysis", "visual_description"), 5000)
    source = _clean_text(block.get("source"), 80) or identity["source"]
    route = _clean_text(block.get("route"), 32).lower() or identity["route"]
    section_id = _clean_text(block.get("section_id"), 160)
    asset_id = _asset_id(
        identity,
        kind=kind,
        page=page,
        block_id=block_id,
        figure_id=figure_id,
        bbox=bbox,
        text=text,
    )
    asset = _asset_shell(
        asset_id=asset_id,
        kind=kind,
        source_kind=source_kind or kind,
        page=page,
        bbox=bbox,
        owner_block_id=owner_block_id,
        block_id=block_id,
        figure_id=figure_id,
        text=text,
        caption=caption,
        description=description,
        source=source,
        route=route,
        confidence=_normalize_confidence(block.get("confidence")),
        identity=identity,
        section_id=section_id,
        sections=sections,
    )
    visual_model = block.get("visual_model")
    if source.lower() == "mineru_vlm" or isinstance(visual_model, dict):
        asset["visual_provenance"].append(_base_visual_provenance(block, asset))
    explicit_parent = _clean_text(block.get("derived_from") or block.get("source_block_id"), 240)
    if explicit_parent:
        _add_relation(asset, "derived_from", source_id=asset_id, target_id=explicit_parent, target_kind="block")
    return asset


def _asset_from_supplement(
    supplement: dict[str, Any],
    identity: dict[str, str],
    sections: dict[str, dict[str, Any]],
    block_to_asset: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence_id = supplement["evidence_id"]
    owner_block_id = supplement["owner_block_id"]
    owner = block_to_asset.get(owner_block_id)
    section_id = supplement["section_id"] or (owner.get("section_id", "") if owner else "")
    asset_id = _asset_id(
        identity,
        kind=_VISUAL_KIND,
        page=supplement["page"],
        block_id=evidence_id,
        figure_id=supplement["figure_id"],
        bbox=supplement["bbox"],
        text=supplement["text"],
    )
    asset = _asset_shell(
        asset_id=asset_id,
        kind=_VISUAL_KIND,
        source_kind=_VISUAL_KIND,
        page=supplement["page"],
        bbox=supplement["bbox"],
        owner_block_id=owner_block_id,
        block_id=evidence_id,
        figure_id=supplement["figure_id"],
        text=supplement["text"],
        caption=supplement["caption"],
        description=supplement["description"],
        source=supplement["source"],
        route=supplement["route"] or identity["route"],
        confidence=supplement["confidence"],
        identity=identity,
        section_id=section_id,
        sections=sections,
    )
    asset["visual_provenance"].append(_supplement_provenance(supplement, identity))
    if owner_block_id:
        target_id = owner["asset_id"] if owner else owner_block_id
        _add_relation(
            asset,
            "derived_from",
            source_id=evidence_id,
            target_id=target_id,
            target_kind="asset" if owner else "block",
            target_block_id=owner_block_id,
        )
    return asset


def _asset_shell(
    *,
    asset_id: str,
    kind: str,
    source_kind: str,
    page: int,
    bbox: list[float],
    owner_block_id: str,
    block_id: str,
    figure_id: str,
    text: str,
    caption: str,
    description: str,
    source: str,
    route: str,
    confidence: float | None,
    identity: dict[str, str],
    section_id: str,
    sections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    asset: dict[str, Any] = {
        "asset_id": asset_id,
        "kind": kind,
        "source_kind": source_kind,
        "page": page,
        "bbox": list(bbox),
        "owner_block_id": owner_block_id,
        "block_id": block_id,
        "figure_id": figure_id,
        "text": _join_text(text, caption, description),
        "caption": caption,
        "description": description,
        "source": source,
        "route": route,
        "generation": identity["generation"],
        "source_hash": identity["source_hash"],
        "revision": identity["revision"],
        "confidence": confidence,
        "section_id": section_id,
        "section_title": _clean_text((sections.get(section_id) or {}).get("title"), 400),
        "visual_provenance": [],
        "relations": [],
        "references": [],
    }
    if page > 0:
        _add_relation(
            asset,
            "located_on",
            source_id=asset_id,
            target_id=f"page:{page}",
            target_kind="page",
            page=page,
        )
    if section_id:
        section = sections.get(section_id) or {}
        _add_relation(
            asset,
            "belongs_to_section",
            source_id=asset_id,
            target_id=section_id,
            target_kind="section",
            title=_clean_text(section.get("title"), 400),
        )
    return asset


def _normalize_supplement(
    item: Any,
    fallback_page: int,
    identity: dict[str, str],
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    raw_kind, kind = _block_kind(item)
    visual_marker = bool(item.get("visual_enhancement") or item.get("visual_evidence_id"))
    if kind != _VISUAL_KIND and not visual_marker:
        return None
    page = _positive_int(item.get("page")) or fallback_page
    evidence_id = _clean_text(
        item.get("visual_evidence_id") or item.get("id") or item.get("block_id") or item.get("evidence_id"),
        240,
    )
    text = _first_text(item, ("text", "analysis", "description", "content"), 6000)
    if page <= 0 or not evidence_id or not text:
        return None
    caption = _first_text(item, ("caption", "caption_text", "figure_caption", "table_caption"), 1200)
    description = _first_text(item, ("analysis", "description", "visual_description"), 5000)
    if not description and text != caption:
        description = text
    model = item.get("visual_model") if isinstance(item.get("visual_model"), dict) else {}
    route = _clean_text(item.get("route"), 32).lower() or identity["route"]
    return {
        "evidence_id": evidence_id,
        "page": page,
        "bbox": _normalize_bbox(item.get("bbox") or item.get("figure_bbox")) or [],
        "owner_block_id": _clean_text(
            item.get("owner_block_id") or item.get("source_block_id") or item.get("derived_from"),
            240,
        ),
        "figure_id": _clean_text(item.get("figure_id") or item.get("table_id"), 240),
        "section_id": _clean_text(item.get("section_id"), 160),
        "text": text,
        "caption": caption,
        "description": description,
        "source": _clean_text(item.get("visual_source") or item.get("source"), 80) or "visual_vlm",
        "route": route,
        "generation": _clean_text(item.get("parse_generation") or item.get("generation"), 160),
        "source_hash": _clean_text(item.get("document_source_hash") or item.get("source_hash"), 256),
        "revision": _clean_text(item.get("visual_supplement_revision") or item.get("revision"), 160),
        "confidence": _normalize_confidence(item.get("confidence")),
        "provider": _clean_text(item.get("provider") or model.get("provider"), 120),
        "model": _clean_text(item.get("model") or model.get("model"), 240),
        "visual_model": copy.deepcopy(model),
        "prompt_version": _clean_text(item.get("prompt_version"), 160),
        "purpose": _clean_text(item.get("purpose"), 120),
        "render_mode": _clean_text(item.get("render_mode"), 80),
        "bbox_hash": _clean_text(item.get("bbox_hash"), 160),
        "source_kind": raw_kind or _VISUAL_KIND,
    }


def _evidence_matches_identity(evidence: dict[str, Any], identity: dict[str, str]) -> bool:
    for evidence_key, identity_key in (
        ("route", "route"),
        ("generation", "generation"),
        ("source_hash", "source_hash"),
        ("revision", "revision"),
    ):
        evidence_value = _clean_text(evidence.get(evidence_key), 256).lower()
        identity_value = _clean_text(identity.get(identity_key), 256).lower()
        if evidence_value and identity_value and evidence_value != identity_value:
            return False
    return True


def _upsert_supplement(target: dict[str, dict[str, Any]], item: dict[str, Any]) -> None:
    evidence_id = item["evidence_id"]
    existing = target.get(evidence_id)
    if existing is None:
        target[evidence_id] = item
        return
    merged = dict(existing)
    for key, value in item.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    target[evidence_id] = merged


def _find_supplement_target(
    supplement: dict[str, Any],
    assets: list[dict[str, Any]],
    block_to_asset: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    owner_block_id = supplement["owner_block_id"]
    if owner_block_id and owner_block_id in block_to_asset:
        return block_to_asset[owner_block_id]

    figure_key = _loose_key(supplement["figure_id"])
    if figure_key:
        matches = [
            asset for asset in assets
            if asset["kind"] in _BASE_KINDS and _loose_key(asset.get("figure_id")) == figure_key
        ]
        page_matches = [asset for asset in matches if asset["page"] == supplement["page"]]
        if len(page_matches) == 1:
            return page_matches[0]
        if len(matches) == 1:
            return matches[0]

    bbox = supplement["bbox"]
    if not bbox:
        return None
    candidates: list[tuple[float, dict[str, Any]]] = []
    for asset in assets:
        if asset["kind"] not in _BASE_KINDS or asset["page"] != supplement["page"] or not asset["bbox"]:
            continue
        overlap = _bbox_overlap(bbox, asset["bbox"])
        if overlap >= 0.72:
            candidates.append((overlap, asset))
    candidates.sort(key=lambda item: (-item[0], item[1]["asset_id"]))
    if not candidates:
        return None
    if len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= 0.05:
        return candidates[0][1]
    return None


def _merge_supplement(
    asset: dict[str, Any],
    supplement: dict[str, Any],
    identity: dict[str, str],
) -> None:
    asset["figure_id"] = asset["figure_id"] or supplement["figure_id"]
    asset["caption"] = asset["caption"] or supplement["caption"]
    asset["description"] = _join_text(asset["description"], supplement["description"])
    asset["text"] = _join_text(asset["text"], supplement["caption"], supplement["description"], supplement["text"])
    asset["confidence"] = _best_confidence(asset.get("confidence"), supplement.get("confidence"))
    asset["visual_provenance"].append(_supplement_provenance(supplement, identity))
    _add_relation(
        asset,
        "derived_from",
        source_id=supplement["evidence_id"],
        target_id=asset["asset_id"],
        target_kind="asset",
        target_block_id=asset["block_id"],
    )
    if supplement["caption"]:
        _add_relation(
            asset,
            "caption_of",
            source_id=supplement["evidence_id"],
            target_id=asset["asset_id"],
            target_kind=asset["kind"],
            target_block_id=asset["block_id"],
        )


def _normalize_caption_block(block: dict[str, Any], page: int) -> dict[str, Any] | None:
    text = _first_text(block, ("text", "caption", "content"), 1600)
    block_id = _clean_text(block.get("block_id") or block.get("id"), 240)
    if not text or not block_id:
        return None
    return {
        "block_id": block_id,
        "page": page,
        "bbox": _normalize_bbox(block.get("bbox")) or [],
        "text": text,
        "figure_id": _clean_text(block.get("figure_id") or block.get("table_id"), 240),
        "owner_block_id": _clean_text(block.get("owner_block_id") or block.get("caption_of"), 240),
        "references": _extract_references(text),
    }


def _bind_captions(
    captions: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    block_to_asset: dict[str, dict[str, Any]],
) -> None:
    for caption in captions:
        target = None
        owner_block_id = caption["owner_block_id"]
        if owner_block_id:
            target = block_to_asset.get(owner_block_id)
        if target is None and caption["figure_id"]:
            key = _loose_key(caption["figure_id"])
            matches = [
                asset for asset in assets
                if asset["page"] == caption["page"] and _loose_key(asset.get("figure_id")) == key
            ]
            if len(matches) == 1:
                target = matches[0]
        if target is None and caption["references"]:
            matches = [
                asset for asset in assets
                if asset["page"] == caption["page"]
                and caption["references"] & _asset_references(asset)
            ]
            if len(matches) == 1:
                target = matches[0]
        if target is None and _CAPTION_PREFIX_PATTERN.search(caption["text"]):
            target = _unique_nearby_caption_target(caption, assets)
        if target is None:
            continue
        target["caption"] = target["caption"] or caption["text"]
        target["text"] = _join_text(target["text"], caption["text"])
        _add_relation(
            target,
            "caption_of",
            source_id=caption["block_id"],
            target_id=target["asset_id"],
            target_kind=target["kind"],
            target_block_id=target["block_id"],
        )


def _unique_nearby_caption_target(
    caption: dict[str, Any],
    assets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    bbox = caption["bbox"]
    if not bbox:
        return None
    ranked: list[tuple[float, dict[str, Any]]] = []
    for asset in assets:
        asset_bbox = asset["bbox"]
        if asset["page"] != caption["page"] or asset["kind"] not in {"figure", "table"} or not asset_bbox:
            continue
        horizontal_overlap = _axis_overlap(bbox[0], bbox[2], asset_bbox[0], asset_bbox[2])
        if horizontal_overlap < 0.5:
            continue
        vertical_gap = max(0.0, asset_bbox[1] - bbox[3], bbox[1] - asset_bbox[3])
        height = max(1.0, bbox[3] - bbox[1])
        if vertical_gap > max(48.0, height * 2.5):
            continue
        ranked.append((vertical_gap - horizontal_overlap, asset))
    ranked.sort(key=lambda item: (item[0], item[1]["asset_id"]))
    if not ranked:
        return None
    if len(ranked) == 1 or ranked[1][0] - ranked[0][0] >= 8.0:
        return ranked[0][1]
    return None


def _base_visual_provenance(block: dict[str, Any], asset: dict[str, Any]) -> dict[str, Any]:
    model = block.get("visual_model") if isinstance(block.get("visual_model"), dict) else {}
    return {
        "role": "parser",
        "evidence_id": asset["block_id"],
        "source": asset["source"],
        "route": asset["route"],
        "revision": asset["revision"],
        "provider": _clean_text(block.get("provider") or model.get("provider"), 120),
        "model": _clean_text(block.get("model") or model.get("model"), 240),
        "visual_model": copy.deepcopy(model),
        "prompt_version": _clean_text(block.get("prompt_version"), 160),
        "purpose": _clean_text(block.get("purpose"), 120),
        "render_mode": _clean_text(block.get("render_mode"), 80),
        "bbox_hash": _clean_text(block.get("bbox_hash"), 160),
        "confidence": _normalize_confidence(block.get("confidence")),
    }


def _supplement_provenance(
    supplement: dict[str, Any],
    identity: dict[str, str],
) -> dict[str, Any]:
    return {
        "role": "enrichment",
        "evidence_id": supplement["evidence_id"],
        "source": supplement["source"],
        "route": supplement["route"] or identity["route"],
        "revision": supplement["revision"] or identity["revision"],
        "provider": supplement["provider"],
        "model": supplement["model"],
        "visual_model": copy.deepcopy(supplement["visual_model"]),
        "prompt_version": supplement["prompt_version"],
        "purpose": supplement["purpose"],
        "render_mode": supplement["render_mode"],
        "bbox_hash": supplement["bbox_hash"],
        "confidence": supplement["confidence"],
    }


def _score_asset(
    asset: dict[str, Any],
    *,
    query: str,
    reference: str,
    references: set[str],
    loose_reference: str,
    query_pages: set[int],
    terms: set[str],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    asset_references = _asset_references(asset)
    exact_references = references & asset_references
    if exact_references:
        score += 100.0 + min(3, len(exact_references)) * 5.0
        reasons.append("explicit_reference")
    searchable_reference = _loose_key(" ".join((
        str(asset.get("figure_id") or ""),
        str(asset.get("caption") or ""),
        str(asset.get("asset_id") or ""),
    )))
    if loose_reference and loose_reference in searchable_reference:
        score += 35.0
        reasons.append("reference_text")

    if query_pages and asset["page"] in query_pages:
        score += 24.0
        reasons.append("page")

    caption_terms = _search_terms(str(asset.get("caption") or ""))
    description_terms = _search_terms(str(asset.get("description") or ""))
    text_terms = _search_terms(str(asset.get("text") or ""))
    caption_hits = terms & caption_terms
    description_hits = terms & description_terms
    text_hits = terms & text_terms
    if caption_hits:
        score += 8.0 * len(caption_hits)
        reasons.append("caption_terms")
    if description_hits:
        score += 4.0 * len(description_hits)
        reasons.append("description_terms")
    residual_hits = text_hits - caption_hits - description_hits
    if residual_hits:
        score += 2.0 * len(residual_hits)
        reasons.append("text_terms")

    normalized_query = _loose_key(query)
    normalized_caption = _loose_key(asset.get("caption"))
    normalized_description = _loose_key(asset.get("description"))
    if len(normalized_query) >= 4 and (
        normalized_query in normalized_caption or normalized_query in normalized_description
    ):
        score += 18.0
        reasons.append("exact_phrase")

    confidence = _normalize_confidence(asset.get("confidence"))
    if confidence is not None and score > 0:
        score += confidence
    if score > 0 and looks_like_visual_query(query) and asset.get("visual_provenance"):
        score += 0.5
    return score, reasons


def _asset_references(asset: dict[str, Any]) -> set[str]:
    stored = asset.get("references")
    references = set(stored) if isinstance(stored, list) else set()
    for value in (
        asset.get("figure_id"),
        asset.get("caption"),
        asset.get("description"),
        asset.get("text"),
    ):
        references.update(_extract_references(str(value or "")))
    return references


def _extract_references(value: Any) -> set[str]:
    text = _clean_text(value, 8000)
    references: set[str] = set()
    for match in _REFERENCE_PATTERN.finditer(text):
        kind = _reference_kind(match.group("label"))
        if not kind:
            continue
        numbers = match.group("numbers")
        range_match = _REFERENCE_RANGE_PATTERN.fullmatch(numbers)
        if range_match:
            start = int(range_match.group("start"))
            end = int(range_match.group("end"))
            if abs(end - start) <= 32:
                step = 1 if end >= start else -1
                references.update(
                    f"{kind}:{number}"
                    for number in range(start, end + step, step)
                )
                continue
        for token_match in _REFERENCE_TOKEN_PATTERN.finditer(numbers):
            number = re.sub(r"[^0-9a-z]+", "", token_match.group(0).lower())
            if number:
                references.add(f"{kind}:{number}")
    return references


def _reference_kind(label: str) -> str:
    normalized = _normalized_token(label)
    if normalized in {"fig", "figs", "figure", "figures", "图", "图表"}:
        return "figure"
    if normalized in {"table", "tables", "tab", "tabs", "表", "表格"}:
        return "table"
    if normalized in {
        "equation", "equations", "eq", "eqs", "formula", "formulas", "公式", "方程", "式"
    }:
        return "formula"
    return ""


def _extract_pages(value: Any) -> set[int]:
    pages: set[int] = set()
    for match in _PAGE_PATTERN.finditer(_clean_text(value, 4000)):
        page = _positive_int(match.group("zh") or match.group("en") or match.group("short"))
        if page > 0:
            pages.add(page)
    return pages


def _search_terms(value: Any) -> set[str]:
    text = _clean_text(value, 8000).lower()
    terms = {
        token for token in re.findall(r"[a-z][a-z0-9_\-]*|\d+(?:\.\d+)?", text)
        if len(token) > 1 and token not in _SEARCH_STOPWORDS
    }
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            continue
        if len(run) <= 8:
            terms.add(run)
        terms.update(run[index:index + 2] for index in range(len(run) - 1))
    return terms - _SEARCH_STOPWORDS


def _normalize_kind_filter(kinds: Any) -> set[str] | None:
    if kinds is None:
        return None
    if not isinstance(kinds, list):
        return set()
    return {_normalize_kind(item) for item in kinds if _normalize_kind(item)}


def _normalize_kind(value: Any) -> str:
    return _KIND_ALIASES.get(_normalized_token(value), "")


def _normalized_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9_\u4e00-\u9fff]+", "_", str(value or "").strip().lower()).strip("_")


def _normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def _bbox_overlap(left: list[float], right: list[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    iou = intersection / union if union > 0 else 0.0
    containment = intersection / min(left_area, right_area) if min(left_area, right_area) > 0 else 0.0
    return max(iou, containment)


def _axis_overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    denominator = min(left_end - left_start, right_end - right_start)
    return intersection / denominator if denominator > 0 else 0.0


def _asset_id(
    identity: dict[str, str],
    *,
    kind: str,
    page: int,
    block_id: str,
    figure_id: str,
    bbox: list[float],
    text: str,
) -> str:
    digest = _stable_hash({
        "route": identity["route"],
        "generation": identity["generation"],
        "source_hash": identity["source_hash"],
        "kind": kind,
        "page": page,
        "block_id": block_id,
        "figure_id": figure_id,
        "bbox": bbox,
        "text": text[:240],
    }, 20)
    return f"asset:{kind}:{digest}"


def _add_relation(
    asset: dict[str, Any],
    relation_type: str,
    *,
    source_id: str,
    target_id: str,
    target_kind: str,
    **metadata: Any,
) -> None:
    if not source_id or not target_id:
        return
    relation = {
        "type": relation_type,
        "source_id": source_id,
        "target_id": target_id,
        "target_kind": target_kind,
    }
    relation.update({key: value for key, value in metadata.items() if value not in (None, "", [], {})})
    asset.setdefault("relations", []).append(relation)


def _dedupe_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    by_key: dict[str, dict[str, Any]] = {}
    for relation in value:
        if not isinstance(relation, dict):
            continue
        key = json.dumps(relation, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        by_key[key] = relation
    return [by_key[key] for key in sorted(by_key)]


def _dedupe_provenance(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    by_key: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        key = _clean_text(item.get("evidence_id"), 240) or _stable_hash(item)
        by_key[key] = item
    return [by_key[key] for key in sorted(by_key)]


def _asset_sort_key(asset: dict[str, Any]) -> tuple[Any, ...]:
    bbox = asset.get("bbox") or [float("inf")] * 4
    kind_order = {"figure": 0, "table": 1, "formula": 2, _VISUAL_KIND: 3}
    return (
        _positive_int(asset.get("page")) or 10**9,
        float(bbox[1]),
        float(bbox[0]),
        kind_order.get(str(asset.get("kind") or ""), 9),
        str(asset.get("asset_id") or ""),
    )


def _supplement_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    bbox = item.get("bbox") or [float("inf")] * 4
    return (item.get("page", 0), float(bbox[1]), float(bbox[0]), item.get("evidence_id", ""))


def _looks_like_caption(value: str) -> bool:
    return bool(value and _CAPTION_PREFIX_PATTERN.search(value[:180]))


def _first_text(item: dict[str, Any], keys: tuple[str, ...], limit: int) -> str:
    for key in keys:
        text = _clean_text(item.get(key), limit)
        if text:
            return text
    return ""


def _join_text(*values: Any, limit: int = 8000) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value, limit)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            parts.append(text)
    return "\n".join(parts)[:limit].strip()


def _clean_text(value: Any, limit: int) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    normalized = " ".join(str(value).replace("\x00", " ").split())
    return normalized[: max(0, int(limit))].strip()


def _loose_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


def _positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _bounded_limit(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = DEFAULT_SEARCH_LIMIT
    return max(0, min(MAX_SEARCH_LIMIT, number))


def _normalize_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return round(max(0.0, min(1.0, float(value))), 6)
    except (TypeError, ValueError):
        return None


def _best_confidence(left: Any, right: Any) -> float | None:
    values = [value for value in (_normalize_confidence(left), _normalize_confidence(right)) if value is not None]
    return max(values) if values else None


def _stable_hash(value: Any, length: int = 24) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
