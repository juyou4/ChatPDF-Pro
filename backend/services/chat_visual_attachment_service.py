"""Trusted, generation-bound visual attachments for chat answers.

The language model never supplies an image path.  Attachments are selected from
the canonical modal asset index, rendered from the source PDF, and materialized
as immutable JPEG files before a chat response is published.  Keeping the
rendered file makes historical answers independent from the document's current
parse generation.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable

import fitz
from PIL import Image

from services.figure_render import crop_figure_image, render_panel_composite_image
from services.modal_asset_service import search_modal_assets


CHAT_VISUAL_ATTACHMENT_VERSION = "chat_visual_attachment.v1"
CHAT_VISUAL_RENDER_VERSION = "chat_visual_render.v1"

_SAFE_ATTACHMENT_ID_RE = re.compile(r"^va_[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INLINE_CITATION_RE = re.compile(r"(?<!!)\[(\d{1,3})\](?!\()|【(\d{1,3})】")
_VISUAL_REQUEST_RE = re.compile(
    r"(?:图\s*[一二三四五六七八九十\d]+|表\s*[一二三四五六七八九十\d]+|"
    r"图片|图像|图表|架构图|流程图|示意图|曲线|坐标图|柱状图|饼图|热力图|"
    r"figure|fig\.?|image|diagram|chart|plot|table)",
    re.IGNORECASE,
)
_EXPLICIT_VISUAL_REF_RE = re.compile(
    r"(?:(?P<fig>(?<![A-Za-z0-9_])figure|(?<![A-Za-z0-9_])fig\.?|图)|"
    r"(?P<tab>(?<![A-Za-z0-9_])table|(?<![A-Za-z0-9_])tab\.?|表))"
    r"\s*[.:：#_-]*\s*(?P<num>[a-z]?\d+(?:[.\-]\d+)?[a-z]?)",
    re.IGNORECASE,
)
_BARE_VISUAL_ID_RE = re.compile(
    r"^(?:fig(?:ure)?|tab(?:le)?|图|表)?[_:\-\s#]*(?P<num>[a-z]?\d+(?:[.\-]\d+)?[a-z]?)$",
    re.IGNORECASE,
)
_MAX_ATTACHMENTS = 2
_MAX_RENDER_EDGE = 1800
_MAX_RENDER_PIXELS = 6_000_000
_CACHE_LOCK = threading.RLock()


class ChatVisualAttachmentError(ValueError):
    """A fixed-code error safe for route-level translation."""


def question_requests_visual_attachment(question: str) -> bool:
    """Return whether the user question itself asked to see a figure or table."""
    return bool(_VISUAL_REQUEST_RE.search(str(question or "")))


def build_chat_visual_attachments(
    *,
    data_dir: Path | str,
    doc_id: str,
    pdf_path: Path | str | None,
    modal_asset_index: dict,
    citations: Iterable[dict] | None,
    question: str,
    answer: str,
    max_items: int = _MAX_ATTACHMENTS,
) -> list[dict[str, Any]]:
    """Select cited visuals and materialize immutable answer attachments."""
    identity = _validated_index_identity(modal_asset_index)
    selected = _select_assets(
        modal_asset_index,
        citations=list(citations or []),
        question=question,
        answer=answer,
        max_items=max_items,
    )
    if not selected:
        return []
    source_pdf = Path(pdf_path) if pdf_path else None
    if source_pdf is None or not source_pdf.is_file():
        return []
    if _file_sha256(source_pdf) != identity["source_hash"]:
        return []
    attachments: list[dict[str, Any]] = []
    for asset, citation_refs, evidence_mode in selected:
        try:
            attachments.append(
                _materialize_asset(
                    data_dir=Path(data_dir),
                    doc_id=str(doc_id or ""),
                    pdf_path=source_pdf,
                    identity=identity,
                    asset=asset,
                    citation_refs=citation_refs,
                    evidence_mode=evidence_mode,
                )
            )
        except ChatVisualAttachmentError:
            continue
    return attachments


def load_chat_visual_attachment(
    *,
    data_dir: Path | str,
    doc_id: str,
    parse_generation: str,
    attachment_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a cached attachment without consulting the current generation."""
    safe_attachment_id = str(attachment_id or "").strip().lower()
    if not _SAFE_ATTACHMENT_ID_RE.fullmatch(safe_attachment_id):
        raise ChatVisualAttachmentError("invalid_attachment_id")
    expected_generation = str(parse_generation or "").strip()
    if not expected_generation or len(expected_generation) > 160:
        raise ChatVisualAttachmentError("invalid_parse_generation")

    root = _document_cache_dir(Path(data_dir), str(doc_id or ""))
    manifest_path = root / f"{safe_attachment_id}.json"
    image_path = root / f"{safe_attachment_id}.jpg"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ChatVisualAttachmentError("attachment_not_found") from exc
    if not isinstance(manifest, dict) or not image_path.is_file():
        raise ChatVisualAttachmentError("attachment_not_found")
    if (
        manifest.get("version") != CHAT_VISUAL_ATTACHMENT_VERSION
        or str(manifest.get("doc_id") or "") != str(doc_id or "")
        or str(manifest.get("parse_generation") or "") != expected_generation
        or str(manifest.get("attachment_id") or "") != safe_attachment_id
    ):
        raise ChatVisualAttachmentError("attachment_identity_mismatch")
    if not _manifest_image_is_valid(manifest, image_path):
        raise ChatVisualAttachmentError("attachment_corrupt")
    return image_path, manifest


def _validated_index_identity(index: Any) -> dict[str, str]:
    if not isinstance(index, dict):
        raise ChatVisualAttachmentError("invalid_modal_asset_index")
    identity = {
        "route": str(index.get("route") or index.get("parser_route") or "").strip().lower(),
        "generation": str(index.get("generation") or index.get("parse_generation") or "").strip(),
        "source_hash": str(index.get("source_hash") or index.get("document_source_hash") or "").strip().lower(),
        "revision": str(index.get("revision") or index.get("visual_supplement_revision") or "").strip(),
    }
    if (
        identity["route"] not in {"local", "mineru"}
        or not identity["generation"]
        or len(identity["generation"]) > 160
        or not _SHA256_RE.fullmatch(identity["source_hash"])
        or not any(key in index for key in ("revision", "visual_supplement_revision"))
    ):
        raise ChatVisualAttachmentError("incomplete_parse_identity")
    return identity


def _select_assets(
    index: dict,
    *,
    citations: list[dict],
    question: str,
    answer: str,
    max_items: int,
) -> list[tuple[dict, list[int], str]]:
    assets = [asset for asset in (index.get("assets") or []) if _displayable_asset(asset)]
    if not assets:
        return []
    bounded_limit = max(0, min(int(max_items or 0), _MAX_ATTACHMENTS))
    if bounded_limit <= 0:
        return []

    inline_refs = {
        int(left or right)
        for left, right in _INLINE_CITATION_RE.findall(str(answer or ""))
        if (left or right)
    }
    visual_requested = bool(_VISUAL_REQUEST_RE.search(f"{question}\n{answer}"))
    ranked: dict[str, dict[str, Any]] = {}
    for position, citation in enumerate(citations):
        if not isinstance(citation, dict):
            continue
        ref = _positive_int(citation.get("display_ref") or citation.get("ref"))
        if inline_refs and ref not in inline_refs:
            continue
        runtime_analyzed = bool(
            citation.get("runtime_visual_analysis")
            or citation.get("analyzed_asset_id")
            or str(citation.get("retrieval_type") or "") == "agent_visual_analysis"
        )
        if not visual_requested and not runtime_analyzed:
            continue
        asset = _match_citation_asset(citation, assets)
        if asset is None:
            continue
        asset_id = str(asset.get("asset_id") or "")
        entry = ranked.setdefault(asset_id, {
            "asset": asset,
            "refs": [],
            "runtime_analyzed": False,
            "position": position,
        })
        if ref > 0 and ref not in entry["refs"]:
            entry["refs"].append(ref)
        entry["runtime_analyzed"] = entry["runtime_analyzed"] or runtime_analyzed

    if question_requests_visual_attachment(question):
        for asset in _match_question_assets(index, assets, question, bounded_limit):
            asset_id = str(asset.get("asset_id") or "")
            if not asset_id or asset_id in ranked:
                continue
            ranked[asset_id] = {
                "asset": asset,
                "refs": [],
                "runtime_analyzed": False,
                "position": 10**6 + len(ranked),
            }
            if len(ranked) >= bounded_limit:
                break

    ordered = sorted(
        ranked.values(),
        key=lambda item: (
            not item["runtime_analyzed"],
            item["position"],
            _positive_int(item["asset"].get("page")),
            str(item["asset"].get("asset_id") or ""),
        ),
    )
    return [
        (
            item["asset"],
            item["refs"],
            "vlm_verified" if item["runtime_analyzed"] else "parser_visual",
        )
        for item in ordered[:bounded_limit]
    ]


def _match_question_assets(
    index: dict,
    assets: list[dict],
    question: str,
    max_items: int,
) -> list[dict]:
    """Locate the asked figure/table even when citations only carry surrounding text."""
    requested = _question_visual_keys(question)
    if requested:
        matched = [
            asset
            for asset in assets
            if _asset_visual_keys(asset) & requested
        ]
        matched.sort(
            key=lambda asset: (
                _positive_int(asset.get("page")),
                str(asset.get("asset_id") or ""),
            )
        )
        return matched[:max_items]

    selected: list[dict] = []
    seen: set[str] = set()
    for asset in search_modal_assets(index, query=question, limit=max_items):
        if not _displayable_asset(asset):
            continue
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id or asset_id in seen:
            continue
        seen.add(asset_id)
        selected.append(asset)
        if len(selected) >= max_items:
            break
    return selected


def _question_visual_keys(question: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for match in _EXPLICIT_VISUAL_REF_RE.finditer(str(question or "")):
        number = _normalized_visual_number(match.group("num"))
        if not number:
            continue
        keys.add(("table" if match.group("tab") else "figure", number))
    return keys


def _asset_visual_keys(asset: dict) -> set[tuple[str, str]]:
    kind = str(asset.get("kind") or "").strip().lower()
    if kind not in {"figure", "table"}:
        return set()
    keys: set[tuple[str, str]] = set()
    blob = " ".join(
        str(asset.get(field) or "")
        for field in ("figure_id", "caption", "description", "text")
    )
    for match in _EXPLICIT_VISUAL_REF_RE.finditer(blob):
        number = _normalized_visual_number(match.group("num"))
        if not number:
            continue
        keys.add(("table" if match.group("tab") else "figure", number))
    bare = _BARE_VISUAL_ID_RE.fullmatch(str(asset.get("figure_id") or "").strip())
    if bare:
        number = _normalized_visual_number(bare.group("num"))
        if number:
            keys.add((kind, number))
    return keys


def _normalized_visual_number(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower()).lstrip("0") or ""


def _match_citation_asset(citation: dict, assets: list[dict]) -> dict | None:
    identifiers = [
        str(citation.get(key) or "").strip()
        for key in ("analyzed_asset_id", "asset_id")
        if str(citation.get(key) or "").strip()
    ]
    for identifier in identifiers:
        for asset in assets:
            if str(asset.get("asset_id") or "").strip() == identifier:
                return asset

    figure_id = str(citation.get("figure_id") or citation.get("table_id") or "").strip().casefold()
    block_id = str(citation.get("block_id") or citation.get("evidence_block_id") or "").strip()
    page = _citation_page(citation)
    bbox = _valid_bbox(citation.get("figure_bbox") or citation.get("bbox"))
    candidates: list[tuple[float, dict]] = []
    for asset in assets:
        if page and _positive_int(asset.get("page")) != page:
            continue
        score = 0.0
        if figure_id and str(asset.get("figure_id") or "").strip().casefold() == figure_id:
            score += 10.0
        if block_id and block_id in {
            str(asset.get("block_id") or ""),
            str(asset.get("owner_block_id") or ""),
        }:
            score += 8.0
        if bbox:
            score += 5.0 * _bbox_iou(bbox, _valid_bbox(asset.get("bbox")))
        if score > 0:
            candidates.append((score, asset))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], str(item[1].get("asset_id") or "")))
    return candidates[0][1]


def _displayable_asset(asset: Any) -> bool:
    if not isinstance(asset, dict):
        return False
    if str(asset.get("kind") or "").strip().lower() not in {"figure", "table"}:
        return False
    return bool(
        str(asset.get("asset_id") or "").strip()
        and _positive_int(asset.get("page")) > 0
        and _valid_bbox(asset.get("bbox"))
    )


def _materialize_asset(
    *,
    data_dir: Path,
    doc_id: str,
    pdf_path: Path,
    identity: dict[str, str],
    asset: dict,
    citation_refs: list[int],
    evidence_mode: str,
) -> dict[str, Any]:
    if not doc_id:
        raise ChatVisualAttachmentError("missing_document_id")
    if not _asset_matches_identity(asset, identity):
        raise ChatVisualAttachmentError("asset_identity_mismatch")
    bbox = _valid_bbox(asset.get("bbox"))
    page = _positive_int(asset.get("page"))
    if not bbox or page <= 0:
        raise ChatVisualAttachmentError("invalid_asset_geometry")
    panel_bboxes = _valid_panel_bboxes(asset.get("panel_bboxes"))
    render_ref = asset.get("render_ref") if isinstance(asset.get("render_ref"), dict) else {}

    payload = {
        "version": CHAT_VISUAL_ATTACHMENT_VERSION,
        "render_version": CHAT_VISUAL_RENDER_VERSION,
        "doc_id": doc_id,
        "asset_id": str(asset.get("asset_id") or ""),
        "kind": str(asset.get("kind") or "").strip().lower(),
        "page": page,
        "bbox": [round(value, 3) for value in bbox],
        "panel_bboxes": [
            [round(value, 3) for value in panel]
            for panel in panel_bboxes
        ],
        "render_ref": _safe_render_ref(render_ref),
        "route": identity["route"],
        "parse_generation": identity["generation"],
        "document_source_hash": identity["source_hash"],
        "visual_supplement_revision": identity["revision"],
    }
    attachment_id = f"va_{_stable_hash(payload)[:32]}"
    root = _document_cache_dir(data_dir, doc_id)
    image_path = root / f"{attachment_id}.jpg"
    manifest_path = root / f"{attachment_id}.json"

    with _CACHE_LOCK:
        cached = _read_matching_manifest(manifest_path, image_path, payload, attachment_id)
        if cached is None:
            image_bytes, width, height = _render_asset(
                pdf_path,
                page,
                bbox,
                panel_bboxes=panel_bboxes,
            )
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            manifest = {
                **payload,
                "attachment_id": attachment_id,
                "width": width,
                "height": height,
                "mime_type": "image/jpeg",
                "image_sha256": image_sha256,
            }
            root.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(image_path, image_bytes)
            _atomic_write_json(manifest_path, manifest)
        else:
            manifest = cached

    caption = _compact_text(
        asset.get("caption")
        or asset.get("description")
        or asset.get("text"),
        360,
    )
    figure_id = _compact_text(asset.get("figure_id"), 120)
    label = figure_id or ("表格" if payload["kind"] == "table" else "图")
    return {
        "version": CHAT_VISUAL_ATTACHMENT_VERSION,
        "attachment_id": attachment_id,
        "asset_id": payload["asset_id"],
        "kind": payload["kind"],
        "label": label,
        "caption": caption,
        "figure_id": figure_id,
        "page": page,
        "bbox": payload["bbox"],
        "panel_bboxes": payload["panel_bboxes"],
        "render_ref": payload["render_ref"],
        "coordinate_space": "pdf_top_left_points",
        "route": identity["route"],
        "parse_generation": identity["generation"],
        "document_source_hash": identity["source_hash"],
        "visual_supplement_revision": identity["revision"],
        "citation_refs": sorted(set(citation_refs)),
        "evidence_mode": evidence_mode,
        "width": manifest["width"],
        "height": manifest["height"],
        "mime_type": "image/jpeg",
    }


def _asset_matches_identity(asset: dict, identity: dict[str, str]) -> bool:
    return (
        str(asset.get("route") or "").strip().lower() == identity["route"]
        and str(asset.get("generation") or "").strip() == identity["generation"]
        and str(asset.get("source_hash") or "").strip().lower() == identity["source_hash"]
        and str(asset.get("revision") or "").strip() == identity["revision"]
    )


def _render_asset(
    pdf_path: Path,
    page: int,
    bbox: list[float],
    *,
    panel_bboxes: list[list[float]] | None = None,
) -> tuple[bytes, int, int]:
    try:
        pdf_doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ChatVisualAttachmentError("pdf_open_failed") from exc
    try:
        if page > pdf_doc.page_count:
            raise ChatVisualAttachmentError("asset_page_out_of_range")
        valid_panels = _valid_panel_bboxes(panel_bboxes)
        if len(valid_panels) >= 2:
            image_bytes, width, height = render_panel_composite_image(
                pdf_doc,
                page - 1,
                valid_panels,
                dpi=180,
                output_format="jpeg",
                jpg_quality=90,
                padding=5,
            )
        else:
            page_rect = pdf_doc[page - 1].rect
            padded = fitz.Rect(
                max(page_rect.x0, bbox[0] - 5.0),
                max(page_rect.y0, bbox[1] - 5.0),
                min(page_rect.x1, bbox[2] + 5.0),
                min(page_rect.y1, bbox[3] + 5.0),
            )
            if padded.is_empty or padded.width <= 0 or padded.height <= 0:
                raise ChatVisualAttachmentError("invalid_asset_geometry")
            image_bytes, width, height = crop_figure_image(
                pdf_doc,
                page - 1,
                [padded.x0, padded.y0, padded.x1, padded.y1],
                dpi=180,
                output_format="jpeg",
                jpg_quality=90,
            )
    except ChatVisualAttachmentError:
        raise
    except Exception as exc:
        raise ChatVisualAttachmentError("render_failed") from exc
    finally:
        pdf_doc.close()
    return _bound_render(image_bytes, width, height)


def _valid_panel_bboxes(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        return []
    panels: list[list[float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for raw_bbox in value:
        bbox = _valid_bbox(raw_bbox)
        if not bbox:
            continue
        key = tuple(round(item, 3) for item in bbox)
        if key not in seen:
            panels.append(bbox)
            seen.add(key)
    return panels


def _safe_render_ref(value: dict[str, Any]) -> dict[str, Any]:
    mode = str(value.get("mode") or "").strip()[:80]
    result: dict[str, Any] = {"mode": mode} if mode else {}
    page = _positive_int(value.get("page"))
    if page:
        result["page"] = page
    return result


def _bound_render(image_bytes: bytes, width: int, height: int) -> tuple[bytes, int, int]:
    pixels = max(1, int(width) * int(height))
    scale = min(
        1.0,
        _MAX_RENDER_EDGE / float(max(1, width, height)),
        math.sqrt(_MAX_RENDER_PIXELS / float(pixels)),
    )
    if scale >= 0.999:
        return image_bytes, int(width), int(height)
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            target = (
                max(1, int(round(source.width * scale))),
                max(1, int(round(source.height * scale))),
            )
            resized = source.convert("RGB").resize(target, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            resized.save(output, format="JPEG", quality=88, optimize=True)
            return output.getvalue(), target[0], target[1]
    except Exception as exc:
        raise ChatVisualAttachmentError("render_resize_failed") from exc


def _read_matching_manifest(
    manifest_path: Path,
    image_path: Path,
    payload: dict,
    attachment_id: str,
) -> dict | None:
    if not manifest_path.is_file() or not image_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    expected = {**payload, "attachment_id": attachment_id}
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if not _manifest_image_is_valid(manifest, image_path):
        return None
    return manifest


def _manifest_image_is_valid(manifest: dict, image_path: Path) -> bool:
    expected_hash = str(manifest.get("image_sha256") or "").strip().lower()
    if not _SHA256_RE.fullmatch(expected_hash):
        return False
    try:
        return _file_sha256(image_path) == expected_hash
    except OSError:
        return False


def _document_cache_dir(data_dir: Path, doc_id: str) -> Path:
    if not doc_id or len(doc_id) > 240:
        raise ChatVisualAttachmentError("invalid_document_id")
    key = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:32]
    return data_dir / "chat_visual_assets" / key


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        except OSError:
            pass


def _atomic_write_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return parsed if parsed > 0 else 0


def _citation_page(citation: dict) -> int:
    page_range = citation.get("page_range")
    if isinstance(page_range, (list, tuple)) and page_range:
        return _positive_int(page_range[0])
    return _positive_int(citation.get("page"))


def _valid_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        bbox = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return []
    if not all(math.isfinite(item) for item in bbox):
        return []
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else []


def _bbox_iou(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
