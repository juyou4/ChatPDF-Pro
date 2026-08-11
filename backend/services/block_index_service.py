"""Document block index for immersive reading.

Builds a stable page/block/outline structure from the original PDF when
available, with a text-only fallback for imported non-PDF documents.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import io
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from services.document_block_roles import (
    annotate_block_role,
    is_post_reference_template_artifact,
    is_reference_heading,
    looks_like_affiliation_or_author_line,
)
from services.visual_supplement_service import (
    active_visual_supplements,
    visual_supplement_revision,
    visual_supplements_are_committed,
)

logger = logging.getLogger(__name__)

BLOCK_INDEX_VERSION = 12
BLOCK_INDEX_REVISION_VERSION = 1
_BLOCK_INDEX_SAVE_LOCKS: dict[str, threading.RLock] = {}
_BLOCK_INDEX_SAVE_LOCKS_LOCK = threading.Lock()
_GENERIC_TOC_TITLES = {
    "全文",
    "全文书签",
    "full text",
    "fulltext",
    "document",
    "article",
    "paper",
    "contents",
    "content",
}

_RE_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.?\s+)(.+)$")
_RE_ROMAN_HEADING = re.compile(r"^\s*([IVXLCM]+)\.\s+(.+)$")
_RE_ALPHA_HEADING = re.compile(r"^\s*([A-Z])\.\s+(.+)$")
_RE_CANONICAL_HEADING = re.compile(
    r"^\s*(abstract|introduction|background|related\s+work|method|methods|methodology|"
    r"approach|experiments?|evaluation|results?|discussion|conclusion|limitations?|"
    r"preliminar(?:y|ies)|implementation|future\s+work|references|acknowledg(?:e)?ments?|"
    r"appendix|supplementary\s+material)\s*$",
    re.IGNORECASE,
)
_RE_CAPTION = re.compile(
    r"^\s*(fig(?:ure)?\.?|图|table|表)\s*([0-9]+|[ivxlcdm]+)\b",
    re.IGNORECASE,
)
_RE_TABLE_LABEL = re.compile(r"^\s*(table|表)\s*([0-9]+|[ivxlcdm]+)\s*[:.\-]?\s*$", re.IGNORECASE)
_RE_RAW_TABLE_MARKUP = re.compile(r"</?(?:table|thead|tbody|tfoot|tr|td|th)\b", re.IGNORECASE)
_RE_DECIMAL_TOKEN = re.compile(r"(?<![A-Za-z])[-+]?\d+\.\d+(?![A-Za-z])")
_RE_PUBLICATION_HEADER_CUE = re.compile(
    r"\b(vol\.?|no\.?|pp\.?|transactions?|journal|proceedings|conference|copyright|"
    r"authorized|licensed|downloaded|doi|issn|isbn)\b",
    re.IGNORECASE,
)
_RE_ALGORITHM_LINE = re.compile(
    r"^\s*(input|output|require|ensure|initialize|initialise|update|repeat|return|for\s+each|for\s+"
    r"|while\s+|if\s+|else\b|end\b|stage\s*\d+)\b",
    re.IGNORECASE,
)
_RE_REFERENCE_ENTRY = re.compile(
    r"^\s*(?:\[\d+\]|\d+[\.)])\s+"
    r"(?:[A-Z][A-Za-z'\-]+,\s+(?:[A-Z]\.|[A-Z][A-Za-z'\-]+)|"
    r"(?:[A-Z]\.\s*)?[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.?))",
    re.IGNORECASE,
)
_RE_KEYWORDS_LINE = re.compile(r"^\s*keywords?\s*[:：]\s+.+", re.IGNORECASE)
_RE_AUTHOR_INITIAL_REFERENCE = re.compile(
    r"^\s*\d+[\.)]\s+[A-Z][A-Za-z'\-]+,\s+(?:[A-Z]\.\s*)+",
    re.IGNORECASE,
)


def get_block_index_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "block_indexes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_block_index_path(data_dir: Path | str, doc_id: str) -> Path:
    return get_block_index_dir(data_dir) / f"{doc_id}.json"


def stamp_block_index_revision(index: dict[str, Any]) -> dict[str, Any]:
    """Attach a stable content revision without making timestamps part of identity.

    Parse generation identifies the selected parser run, but an adapter repair can
    legitimately rebuild the block tree within that same generation. Downstream
    caches must therefore also bind to the actual published structure.
    """
    if not isinstance(index, dict):
        return index
    payload = {
        "revision_version": BLOCK_INDEX_REVISION_VERSION,
        "version": index.get("version"),
        "source": index.get("source"),
        "parser_route": index.get("parser_route"),
        "parse_generation": index.get("parse_generation"),
        "document_source_hash": index.get("document_source_hash"),
        "visual_supplement_revision": index.get("visual_supplement_revision"),
        "pages": index.get("pages") or [],
        "outline": index.get("outline") or [],
        "mineru_structure_version": (
            (index.get("mineru_meta") or {}).get("structure_version")
            if isinstance(index.get("mineru_meta"), dict)
            else None
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    revision = hashlib.sha256(encoded).hexdigest()[:24]
    index["block_index_revision_version"] = BLOCK_INDEX_REVISION_VERSION
    index["block_index_revision"] = revision
    # Keep the legacy-friendly name explicit for API clients and cache keys.
    index["block_index_hash"] = revision
    return index


def _get_block_index_save_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _BLOCK_INDEX_SAVE_LOCKS_LOCK:
        return _BLOCK_INDEX_SAVE_LOCKS.setdefault(key, threading.RLock())


def load_block_index(data_dir: Path | str, doc_id: str) -> dict[str, Any] | None:
    path = get_block_index_path(data_dir, doc_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != BLOCK_INDEX_VERSION:
            return None
        return data
    except Exception as exc:
        logger.warning("[BlockIndex] Failed to load %s: %s", path, exc)
        return None


def save_block_index(
    data_dir: Path | str,
    doc_id: str,
    index: dict[str, Any],
    *,
    preserve_active_source: bool = True,
    current_doc: dict[str, Any] | None = None,
    include_uncommitted_visual_supplements: bool = False,
) -> bool:
    """Durably publish a block index and report a failed write to callers.

    The index is consumed by reading, outlines, translations, and the visual
    supplement publication fence.  A partial or silently failed write must not
    be treated as a successful parser publication.
    """
    path = get_block_index_path(data_dir, doc_id)
    with _get_block_index_save_lock(path):
        stamp_block_index_revision(index)
        # Building a local index can take long enough for a MinerU publication
        # to replace the document's parse identity. Re-read that identity at
        # the publication point instead of trusting the one captured before
        # the build started, otherwise a stale builder can destroy the current
        # route's otherwise valid index.
        if isinstance(current_doc, dict):
            current_identity = _document_parse_identity(
                current_doc,
                include_uncommitted_visual_supplements=include_uncommitted_visual_supplements,
            )
            if current_identity.get("invalid_manifest") or not _block_index_matches_parse_identity(
                index,
                current_identity,
            ):
                logger.info("[BlockIndex] discard stale publication for %s", doc_id)
                return False
        active = load_block_index(data_dir, doc_id)
        if (
            preserve_active_source
            and isinstance(active, dict)
            and str(active.get("source") or "").strip().lower() == "mineru_vlm"
            and str(index.get("source") or "").strip().lower() != "mineru_vlm"
            and not _block_index_matches_parse_identity(
                active,
                {
                    "parser_route": str(index.get("parser_route") or ""),
                    "parse_generation": str(index.get("parse_generation") or ""),
                    "document_source_hash": str(index.get("document_source_hash") or ""),
                    "full_route": bool(index.get("full_route")),
                },
            )
        ):
            logger.info("[BlockIndex] discard stale local publication for %s", doc_id)
            return False

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                temp_path = f.name
                json.dump(index, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
            return True
        except Exception as exc:
            logger.warning("[BlockIndex] Failed to save %s: %s", path, exc)
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass
            return False


def ensure_block_index(
    *,
    doc_id: str,
    doc: dict[str, Any],
    data_dir: Path | str,
    pdf_path: Path | str | None = None,
    force_rebuild: bool = False,
    preserve_active_source: bool = True,
    include_uncommitted_visual_supplements: bool = False,
) -> dict[str, Any]:
    parse_identity = _document_parse_identity(
        doc,
        include_uncommitted_visual_supplements=include_uncommitted_visual_supplements,
    )
    if parse_identity.get("invalid_manifest"):
        raise RuntimeError("文档解析身份不完整，拒绝复用无法确认来源的阅读块缓存")
    cached = load_block_index(data_dir, doc_id)
    has_uncommitted_visual_staging = _has_uncommitted_visual_staging(doc, parse_identity)
    if (
        parse_identity.get("parser_route") == "mineru"
        and parse_identity.get("full_route")
        and not cached
    ):
        # A full MinerU document cannot safely fall back to locally inferred
        # blocks after its MinerU cache has been removed or corrupted.
        raise RuntimeError("MinerU 全程解析的阅读块尚未发布")
    if cached:
        primary_identity_matches = _block_index_matches_primary_parse_identity(
            cached,
            parse_identity,
        )
        visual_revision_changed = (
            primary_identity_matches
            and str(cached.get("visual_supplement_revision") or "")
            != str(parse_identity.get("visual_supplement_revision") or "")
        )
        if visual_revision_changed and not has_uncommitted_visual_staging:
            cached = _reconcile_cached_visual_blocks(
                cached,
                doc=doc,
                parse_identity=parse_identity,
            )
            if not save_block_index(
                data_dir,
                doc_id,
                cached,
                current_doc=doc,
            ):
                raise RuntimeError("过期视觉补充索引迁移失败")
        cached_matches_parse = _block_index_matches_parse_identity(cached, parse_identity)
        cached = _maybe_upgrade_mineru_structure_cache(
            cached=cached,
            cached_matches_parse=cached_matches_parse,
            parse_identity=parse_identity,
            doc_id=doc_id,
            doc=doc,
            data_dir=data_dir,
            pdf_path=pdf_path,
            preserve_active_source=preserve_active_source,
            include_uncommitted_visual_supplements=include_uncommitted_visual_supplements,
        )
        # A publisher may have written a staged visual block index immediately
        # before it durably records the matching commit marker. Do not expose
        # those blocks, but also do not overwrite the staged file with a base
        # index: the publisher needs that exact revision to become visible as
        # soon as its document commit succeeds.
        if (
            has_uncommitted_visual_staging
            and str(cached.get("source") or "").strip().lower() != "mineru_vlm"
            and str(cached.get("parse_generation") or "") == str(parse_identity.get("parse_generation") or "")
            and str(cached.get("document_source_hash") or "")
            == str(parse_identity.get("document_source_hash") or "")
        ):
            return _without_uncommitted_visual_blocks(cached)
        # A re-upload of the same PDF keeps its document id, but deliberately
        # creates a new parse generation.  Local readers must rebuild instead
        # of retaining a prior MinerU block tree for the new local route.
        if parse_identity.get("parser_route") == "local" and not cached_matches_parse:
            force_rebuild = True
            preserve_active_source = False
        # MinerU-first documents have no correct local fallback.  They stay
        # gated until their worker publishes a matching MinerU block index.
        if (
            parse_identity.get("parser_route") == "mineru"
            and parse_identity.get("full_route")
            and not cached_matches_parse
        ):
            raise RuntimeError("MinerU 全程解析的阅读块尚未发布")
        # AI-outline regeneration currently uses ``force_rebuild`` to bypass its
        # own cache. Do not let that incidental request replace a completed
        # MinerU layout with a second, local PDF interpretation. A caller that
        # intentionally switches the document back to local parsing must opt in.
        if not force_rebuild or (
            preserve_active_source
            and str(cached.get("source") or "").strip().lower() == "mineru_vlm"
        ):
            return cached

    index = build_block_index(
        doc_id=doc_id,
        doc=doc,
        pdf_path=pdf_path,
        include_uncommitted_visual_supplements=include_uncommitted_visual_supplements,
    )
    if not save_block_index(
        data_dir,
        doc_id,
        index,
        preserve_active_source=preserve_active_source,
        current_doc=doc,
        include_uncommitted_visual_supplements=include_uncommitted_visual_supplements,
    ):
        raise RuntimeError("阅读块索引写入失败或已被新解析路线替代")
    return index


def _maybe_upgrade_mineru_structure_cache(
    *,
    cached: dict[str, Any],
    cached_matches_parse: bool,
    parse_identity: dict[str, Any],
    doc_id: str,
    doc: dict[str, Any],
    data_dir: Path | str,
    pdf_path: Path | str | None,
    preserve_active_source: bool,
    include_uncommitted_visual_supplements: bool,
) -> dict[str, Any]:
    """Rebuild old MinerU structure metadata from the matching raw result."""
    if (
        not cached_matches_parse
        or str(cached.get("source") or "").strip().lower() != "mineru_vlm"
    ):
        return cached

    try:
        from services.mineru_block_index_service import (
            MINERU_STRUCTURE_VERSION,
            MinerUResultUnreadable,
            build_block_index_from_mineru_payload,
            load_mineru_result,
        )
    except Exception as exc:
        logger.warning("[BlockIndex] MinerU structure upgrader unavailable for %s: %s", doc_id, exc)
        return cached

    original_cached = cached
    cached, repaired_table_count = _repair_legacy_mineru_table_markup(cached)
    if repaired_table_count:
        logger.info(
            "[BlockIndex] repaired %s legacy MinerU table block(s) for %s",
            repaired_table_count,
            doc_id,
        )

    def publish_legacy_table_repair() -> dict[str, Any]:
        """Persist a table-only repair or keep every consumer on the old revision.

        Returning a request-local Markdown table while the on-disk block index
        (and therefore the vector index identity) still contained raw HTML
        created a split brain: reading/outline used the repaired text but chat
        continued to retrieve the old chunks.  A successful save stamps a new
        block revision, so the existing RAG artifact is automatically rejected
        by the shared parse-identity gate until it is rebuilt.  If the save
        fails, return the original cache instead of exposing an unpublishable
        in-memory variant.
        """
        if not repaired_table_count:
            return cached
        if save_block_index(
            data_dir,
            doc_id,
            cached,
            preserve_active_source=preserve_active_source,
            current_doc=doc,
            include_uncommitted_visual_supplements=include_uncommitted_visual_supplements,
        ):
            logger.info(
                "[BlockIndex] published legacy MinerU table repair for %s; existing RAG revision is stale",
                doc_id,
            )
            return cached
        logger.warning(
            "[BlockIndex] legacy MinerU table repair was not persisted for %s; retaining old cache revision",
            doc_id,
        )
        return original_cached

    mineru_meta = cached.get("mineru_meta")
    cached_structure_version = (
        mineru_meta.get("structure_version")
        if isinstance(mineru_meta, dict)
        else None
    )
    if cached_structure_version == MINERU_STRUCTURE_VERSION:
        return publish_legacy_table_repair()

    has_parse_identity = bool(
        parse_identity.get("parse_generation")
        and parse_identity.get("document_source_hash")
    )
    try:
        payload = load_mineru_result(
            data_dir,
            doc_id,
            parse_generation=parse_identity.get("parse_generation") if has_parse_identity else None,
            document_source_hash=parse_identity.get("document_source_hash") if has_parse_identity else None,
            require_identity=has_parse_identity,
        )
    except MinerUResultUnreadable as exc:
        # This is an *optional* structure upgrade, so a damaged payload must not
        # take the reader down — but it also must not look like "no payload".
        logger.error(
            "[BlockIndex] raw MinerU result is unreadable for %s; keeping old structure cache: %s",
            doc_id,
            exc,
        )
        return publish_legacy_table_repair()
    if payload is None:
        logger.warning(
            "[BlockIndex] matching raw MinerU result unavailable; retaining old structure cache for %s",
            doc_id,
        )
        return publish_legacy_table_repair()

    try:
        rebuilt = build_block_index_from_mineru_payload(
            doc_id=doc_id,
            doc=doc,
            payload=payload,
            pdf_path=pdf_path,
        )
    except Exception as exc:
        logger.warning("[BlockIndex] MinerU structure cache rebuild failed for %s: %s", doc_id, exc)
        return publish_legacy_table_repair()

    if not save_block_index(
        data_dir,
        doc_id,
        rebuilt,
        preserve_active_source=preserve_active_source,
        current_doc=doc,
        include_uncommitted_visual_supplements=include_uncommitted_visual_supplements,
    ):
        logger.warning("[BlockIndex] MinerU structure cache publication failed for %s", doc_id)
        return cached

    logger.info(
        "[BlockIndex] upgraded MinerU structure cache for %s to version %s",
        doc_id,
        MINERU_STRUCTURE_VERSION,
    )
    return rebuilt


def _repair_legacy_mineru_table_markup(cached: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Build a request-safe view when an old cache still contains table HTML.

    The structure upgrader normally rebuilds from the matching raw MinerU
    payload. Some imported/older documents no longer have that payload, so the
    cached index still needs a non-destructive fallback. Keep its old structure
    version so a real rebuild can happen later if the raw payload reappears.
    """
    pages = cached.get("pages") if isinstance(cached, dict) else None
    if not isinstance(pages, list):
        return cached, 0

    candidates = [
        block
        for page in pages
        if isinstance(page, dict)
        for block in (page.get("blocks") or [])
        if isinstance(block, dict)
        and _RE_RAW_TABLE_MARKUP.search(str(block.get("text") or ""))
    ]
    if not candidates:
        return cached, 0

    from services.mineru_text_normalizer import normalize_table_text

    repaired = deepcopy(cached)
    repaired_count = 0
    for page in repaired.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            source = str(block.get("text") or "")
            if not _RE_RAW_TABLE_MARKUP.search(source):
                continue
            normalized = normalize_table_text(source)
            if not normalized or normalized == source:
                continue
            block["text"] = normalized
            if len(normalized) > 2400:
                block["display_text"] = _limit_text(normalized, 2400)
            else:
                block.pop("display_text", None)
            repaired_count += 1
    return repaired, repaired_count


def _has_uncommitted_visual_staging(doc: dict[str, Any], parse_identity: dict[str, Any]) -> bool:
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, dict) or not parse_identity:
        return False
    return bool(
        active_visual_supplements(
            data,
            parse_identity,
            require_committed=False,
        )
        and not visual_supplements_are_committed(data, parse_identity=parse_identity)
    )


def _without_uncommitted_visual_blocks(index: dict[str, Any]) -> dict[str, Any]:
    """Return a request-local base view while retaining a staged file on disk."""
    safe = deepcopy(index)
    safe["visual_supplement_revision"] = ""
    for page in safe.get("pages") or []:
        if not isinstance(page, dict):
            continue
        page["blocks"] = [
            block
            for block in page.get("blocks") or []
            if not (isinstance(block, dict) and block.get("visual_enhancement"))
        ]
    return safe


def _reconcile_cached_visual_blocks(
    index: dict[str, Any],
    *,
    doc: dict[str, Any],
    parse_identity: dict[str, Any],
) -> dict[str, Any]:
    """Replace only the additive visual layer while preserving parser output.

    Prompt or completion-contract upgrades can invalidate committed visual
    supplements without changing the primary MinerU/local generation. Treating
    that as a parser mismatch makes a full-route MinerU document unreadable.
    Remove the obsolete additive blocks, then attach only currently publishable
    visual evidence.
    """
    reconciled = _without_uncommitted_visual_blocks(index)
    data = doc.get("data") if isinstance(doc, dict) else None
    if (
        str(parse_identity.get("visual_supplement_revision") or "").strip()
        and isinstance(data, dict)
    ):
        return stage_visual_supplements_on_block_index(
            reconciled,
            data=data,
            parse_identity=parse_identity,
        )
    return stamp_block_index_revision(reconciled)


def build_block_index(
    *,
    doc_id: str,
    doc: dict[str, Any],
    pdf_path: Path | str | None = None,
    include_uncommitted_visual_supplements: bool = False,
) -> dict[str, Any]:
    data = doc.get("data", {}) if isinstance(doc, dict) else {}
    pages: list[dict[str, Any]] = []
    toc_items: list[dict[str, Any]] = []
    source = "fallback"
    canonical_source_pages = data.get("pages", []) if isinstance(data, dict) else []
    canonical_pages = _build_pages_from_text(canonical_source_pages)
    canonical_source = _canonical_block_index_source(data, canonical_source_pages)

    pdf_file = Path(pdf_path) if pdf_path else None
    if pdf_file and pdf_file.exists():
        try:
            pages, toc_items = _build_pages_from_pdf(pdf_file)
            source = "pdf_native"
        except Exception as exc:
            logger.warning("[BlockIndex] PDF build failed for %s: %s", doc_id, exc)

    # ``data.pages`` is the document's canonical text representation after
    # local extraction, OCR and optional ODL cleanup. Scanned PDFs still have
    # physical pages, so checking only ``if not pages`` loses usable OCR text.
    if not pages or _should_prefer_canonical_pages(pages, canonical_pages):
        pages = canonical_pages
        source = canonical_source

    parse_identity = _document_parse_identity(
        doc,
        include_uncommitted_visual_supplements=include_uncommitted_visual_supplements,
    )
    _mark_repeated_page_artifacts(pages)
    _inject_visual_blocks(
        pages,
        data,
        parse_identity,
        include_uncommitted_visual_supplements=include_uncommitted_visual_supplements,
    )
    outline = _build_outline(toc_items, pages)
    _assign_sections(pages, outline)
    _annotate_block_roles(pages, outline)
    _exclude_post_reference_template_blocks(pages)

    return stamp_block_index_revision({
        "version": BLOCK_INDEX_VERSION,
        "doc_id": doc_id,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "pages": pages,
        "outline": outline,
        "visual_supplement_revision": str(parse_identity.get("visual_supplement_revision") or ""),
        **parse_identity,
    })


def stage_visual_supplements_on_block_index(
    block_index: dict[str, Any],
    *,
    data: dict[str, Any],
    parse_identity: dict[str, Any],
) -> dict[str, Any]:
    """Add staged visual semantic blocks without rebuilding the primary index.

    MinerU's index must not be regenerated through the PDF-native builder just
    to publish a visual description. This function copies the published index,
    adds only parse-bound supplement blocks, then restamps its revision.
    """
    if not isinstance(block_index, dict) or not isinstance(data, dict):
        raise ValueError("视觉补充索引输入无效")
    route = str(parse_identity.get("parser_route") or "").strip().lower()
    generation = str(parse_identity.get("parse_generation") or "").strip()
    source_hash = str(parse_identity.get("document_source_hash") or "").strip()
    if route not in {"local", "mineru"} or not generation or not source_hash:
        raise ValueError("视觉补充解析身份无效")
    if (
        str(block_index.get("parser_route") or "").strip().lower() != route
        or str(block_index.get("parse_generation") or "").strip() != generation
        or str(block_index.get("document_source_hash") or "").strip() != source_hash
    ):
        raise ValueError("视觉补充不能写入其他解析代际")

    staged = deepcopy(block_index)
    effective_identity = dict(parse_identity)
    effective_identity["visual_supplement_revision"] = visual_supplement_revision(
        data,
        effective_identity,
        require_committed=False,
    )
    if not effective_identity["visual_supplement_revision"]:
        return staged
    _inject_visual_blocks(
        staged.get("pages") or [],
        data,
        effective_identity,
        include_uncommitted_visual_supplements=True,
    )
    _assign_sections(staged.get("pages") or [], staged.get("outline") or [])
    _annotate_block_roles(staged.get("pages") or [], staged.get("outline") or [])
    staged["visual_supplement_revision"] = effective_identity["visual_supplement_revision"]
    return stamp_block_index_revision(staged)


def _document_parse_identity(
    doc: dict[str, Any],
    *,
    include_uncommitted_visual_supplements: bool = False,
) -> dict[str, Any]:
    data = doc.get("data", {}) if isinstance(doc, dict) else {}
    manifest = data.get("parse_manifest") if isinstance(data, dict) else None
    if not isinstance(manifest, dict) and isinstance(data, dict):
        manifest = data.get("document_parse_state")
    if not isinstance(manifest, dict):
        # 真正没有 manifest 的历史文档继续沿用旧缓存；现代文档一旦写入
        # manifest，就必须具备完整身份，不能再走 fail-open。
        return {}
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    if metadata.get("legacy_inferred"):
        return {}
    route = str(manifest.get("resolved_route") or "").strip().lower()
    generation = str(manifest.get("generation") or "").strip()
    source_hash = str(manifest.get("source_hash") or "").strip()
    if not route or not generation or not source_hash:
        return {"invalid_manifest": True}
    identity = {
        "parser_route": route,
        "parse_generation": generation,
        "document_source_hash": source_hash,
        "full_route": bool(metadata.get("full_route")),
    }
    identity["visual_supplement_revision"] = visual_supplement_revision(
        data,
        identity,
        require_committed=not include_uncommitted_visual_supplements,
    )
    return identity


def _block_index_matches_primary_parse_identity(
    index: dict[str, Any],
    identity: dict[str, Any],
) -> bool:
    if not identity:
        # Existing documents predate parse generations. Keep their cached block
        # index usable until they are explicitly reparsed.
        return True
    if (
        str(index.get("parse_generation") or "") != str(identity.get("parse_generation") or "")
        or str(index.get("document_source_hash") or "") != str(identity.get("document_source_hash") or "")
    ):
        return False
    route = str(identity.get("parser_route") or "")
    source = str(index.get("source") or "").strip().lower()
    if route == "mineru":
        return source == "mineru_vlm"
    if route == "local":
        return source != "mineru_vlm"
    return True


def _block_index_matches_parse_identity(index: dict[str, Any], identity: dict[str, Any]) -> bool:
    if not _block_index_matches_primary_parse_identity(index, identity):
        return False
    if not identity:
        return True
    return (
        str(index.get("visual_supplement_revision") or "")
        == str(identity.get("visual_supplement_revision") or "")
    )


def active_block_index_revision(
    index: dict[str, Any] | None,
    doc: dict[str, Any] | None,
) -> str | None:
    """Return the revision only when a modern document owns this index.

    ``None`` is intentionally distinct from an empty string: it means a
    modern parse has no trustworthy published block snapshot and consumers
    must not reuse an artifact that was built from an earlier structure.  An
    empty string remains the compatibility value for documents created before
    parse identities and block revisions existed.
    """
    identity = _document_parse_identity(doc if isinstance(doc, dict) else {})
    if identity.get("invalid_manifest"):
        return None
    if not identity:
        return ""
    if not isinstance(index, dict) or not _block_index_matches_parse_identity(index, identity):
        return None
    revision = str(
        index.get("block_index_hash") or index.get("block_index_revision") or ""
    ).strip()
    return revision or None


def _build_pages_from_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import fitz

    pdf_doc = fitz.open(str(pdf_path))
    try:
        toc_items = [
            {"level": int(level), "title": str(title).strip(), "page": int(page)}
            for level, title, page in pdf_doc.get_toc(simple=True)
            if str(title).strip() and int(page) > 0
        ]

        pages: list[dict[str, Any]] = []
        for page_idx in range(len(pdf_doc)):
            page = pdf_doc[page_idx]
            text_dict = page.get_text("dict") or {}
            layout_regions = _detect_page_layout_regions(page)
            page_blocks = _extract_page_text_blocks(
                text_dict=text_dict,
                page_number=page_idx + 1,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                layout_regions=layout_regions,
            )
            pages.append({
                "page": page_idx + 1,
                "width_pts": float(page.rect.width),
                "height_pts": float(page.rect.height),
                "blocks": page_blocks,
                "layout_regions": layout_regions,
            })
        return pages, toc_items
    finally:
        pdf_doc.close()


def _extract_page_text_blocks(
    text_dict: dict[str, Any],
    page_number: int,
    page_width: float = 612.0,
    page_height: float = 792.0,
    layout_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw_blocks = text_dict.get("blocks", []) if isinstance(text_dict, dict) else []
    font_sizes = _collect_font_sizes(raw_blocks)
    median_font = median(font_sizes) if font_sizes else 10.0
    body_font_key = _dominant_page_body_font_key(raw_blocks, median_font)

    blocks: list[dict[str, Any]] = []
    seq = 0
    for raw in raw_blocks:
        if raw.get("type") != 0:
            continue
        lines = raw.get("lines", []) or []
        for group_lines, forced_heading in _split_text_block_line_groups(
            lines,
            median_font=median_font,
            body_font_key=body_font_key,
            page_width=page_width,
        ):
            for region_lines, layout_category in _split_line_group_by_layout_regions(group_lines, layout_regions or []):
                text = _block_text(region_lines)
                if len(text) < 2:
                    continue

                bbox = _merge_bboxes(
                    span.get("bbox")
                    for line in region_lines
                    for span in (line.get("spans", []) or [])
                ) or _normalize_bbox(raw.get("bbox"))
                if not bbox:
                    continue

                max_font = max(_collect_font_sizes([{"lines": region_lines}]) or [median_font])
                is_bold = _line_is_bold(region_lines)
                block_type, heading_level = _classify_text_block(
                    text,
                    max_font,
                    median_font,
                    bbox=bbox,
                    page_width=page_width,
                    page_height=page_height,
                    lines=region_lines,
                )
                if forced_heading and block_type != "heading" and _looks_like_heading_text(text):
                    block_type = "heading"
                    heading_level = _embedded_heading_level(text)
                block = {
                    "block_id": f"p{page_number}_b{seq}",
                    "type": block_type,
                    "bbox": bbox,
                    "text": _limit_text(text, 2400),
                    "section_id": None,
                    "font_size": round(max_font, 2),
                    "is_bold": is_bold,
                    "line_count": len([line for line in region_lines if line.get("spans")]),
                    "font_key": _dominant_font_key(region_lines),
                    "line_anchors": _build_line_anchors(region_lines),
                }
                if forced_heading:
                    block["split_from_mixed_block"] = True
                if layout_category:
                    block["layout_region_hint"] = layout_category
                    if layout_category != _dominant_line_layout_category(group_lines, layout_regions or []):
                        block["split_from_layout_region"] = True
                if heading_level:
                    block["level"] = heading_level
                blocks.append(block)
                seq += 1

    _demote_adjacent_table_titles(blocks)
    _apply_layout_regions(blocks, layout_regions or [], page_width, page_height)
    return blocks


def _build_pages_from_text(source_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for page_idx, page_data in enumerate(source_pages or []):
        if not isinstance(page_data, dict):
            continue
        page_num = int(page_data.get("page") or page_idx + 1)
        text = str(page_data.get("content") or page_data.get("text") or "")
        page_source = _canonical_page_source(page_data)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks = []
        y = 72.0
        available_height = 660.0
        step = max(32.0, available_height / max(len(paragraphs), 1))
        for idx, paragraph in enumerate(paragraphs):
            block_type, heading_level = _classify_text_block(paragraph, 12.0, 10.0)
            block = {
                "block_id": f"p{page_num}_b{idx}",
                "type": block_type,
                "bbox": [54.0, y, 558.0, min(760.0, y + max(22.0, step * 0.75))],
                "text": _limit_text(paragraph, 2400),
                "section_id": None,
                "source": page_source,
            }
            if heading_level:
                block["level"] = heading_level
            blocks.append(block)
            y += step
        pages.append({
            "page": page_num,
            "width_pts": 612.0,
            "height_pts": 792.0,
            "blocks": blocks,
            "source": page_source,
        })
    return pages


def _canonical_page_source(page_data: dict[str, Any]) -> str:
    """Return the per-page text source without inventing a MinerU identity."""
    raw_source = str(page_data.get("source") or "").strip().lower()
    aliases = {
        "pdf": "pdf_native",
        "pdfplumber": "pdf_native",
        "pymupdf": "pdf_native",
        "local": "pdf_native",
        "local_pdf": "pdf_native",
    }
    return aliases.get(raw_source, raw_source or "text_fallback")


def _canonical_block_index_source(data: dict[str, Any], source_pages: list[dict[str, Any]]) -> str:
    """Describe the active canonical text path at document level.

    ODL can retain individual OCR pages, so prefer its explicit hybrid marker
    over reducing a mixed page set to an opaque generic fallback.
    """
    extraction_method = str(data.get("extraction_method") or "").strip().lower()
    if extraction_method in {"odl", "odl_hybrid"}:
        return extraction_method

    page_sources = {
        _canonical_page_source(page)
        for page in source_pages or []
        if isinstance(page, dict) and str(page.get("content") or page.get("text") or "").strip()
    }
    if not page_sources:
        return "text_fallback"
    if len(page_sources) == 1:
        return next(iter(page_sources))
    if "odl" in page_sources and "ocr" in page_sources:
        return "odl_hybrid"
    if "ocr" in page_sources:
        return "ocr"
    if "odl" in page_sources:
        return "odl"
    return "text_fallback"


def _page_text_metrics(pages: list[dict[str, Any]]) -> dict[str, int]:
    """Measure text coverage before figure/table blocks are injected."""
    characters = 0
    pages_with_text = 0
    for page in pages or []:
        page_characters = 0
        for block in page.get("blocks", []) or []:
            if str(block.get("type") or "").lower() in {"artifact", "figure", "image", "table"}:
                continue
            text = " ".join(str(block.get("text") or "").split())
            page_characters += len(text)
        characters += page_characters
        if page_characters >= 8:
            pages_with_text += 1
    return {"characters": characters, "pages_with_text": pages_with_text}


def _should_prefer_canonical_pages(
    native_pages: list[dict[str, Any]],
    canonical_pages: list[dict[str, Any]],
) -> bool:
    """Use OCR/ODL canonical text only when native PDF text is unusable.

    This deliberately does not make ODL a blanket replacement for native PDF
    layout. It switches only for empty or clearly under-covered native text,
    preserving native geometry on ordinary born-digital PDFs.
    """
    canonical = _page_text_metrics(canonical_pages)
    if canonical["characters"] < 8 or canonical["pages_with_text"] == 0:
        return False

    native = _page_text_metrics(native_pages)
    if native["characters"] < 8 or native["pages_with_text"] == 0:
        return True

    # A short native text layer often contains only a page number or a stray
    # glyph on scanned pages. Require a material canonical corpus before
    # treating a non-empty PDF layer as weak.
    if (
        canonical["characters"] >= 16
        and native["characters"] < 32
        and native["characters"] < canonical["characters"]
    ):
        return True
    if canonical["characters"] >= 80 and native["characters"] * 4 < canonical["characters"]:
        return True
    return (
        canonical["characters"] >= 80
        and native["pages_with_text"] * 2 < canonical["pages_with_text"]
    )


def _mark_repeated_page_artifacts(pages: list[dict[str, Any]]) -> None:
    """Demote repeated top/bottom blocks such as journal headers and footers."""
    candidates: dict[str, list[tuple[int, float, dict[str, Any]]]] = {}
    for page in pages:
        page_num = int(page.get("page") or 1)
        page_height = float(page.get("height_pts") or 792.0)
        for block in page.get("blocks", []) or []:
            text = str(block.get("text") or "").strip()
            bbox = _normalize_bbox(block.get("bbox"))
            if not text or not bbox:
                continue
            y_mid = ((bbox[1] + bbox[3]) / 2.0) / max(page_height, 1.0)
            if 0.18 < y_mid < 0.88:
                continue
            norm = _normalize_repeated_artifact_text(text)
            if len(norm) < 10:
                continue
            candidates.setdefault(norm, []).append((page_num, y_mid, block))

    for entries in candidates.values():
        page_nums = {page_num for page_num, _y_mid, _block in entries}
        if len(page_nums) < 3:
            continue
        entries = sorted(entries, key=lambda item: item[1])
        clusters: list[list[tuple[int, float, dict[str, Any]]]] = []
        for entry in entries:
            if not clusters or abs(clusters[-1][-1][1] - entry[1]) > 0.018:
                clusters.append([entry])
            else:
                clusters[-1].append(entry)
        for cluster in clusters:
            if len({page_num for page_num, _y_mid, _block in cluster}) < 3:
                continue
            for _page_num, _y_mid, block in cluster:
                block["type"] = "artifact"
                block.pop("level", None)


def _normalize_repeated_artifact_text(text: str) -> str:
    value = " ".join(str(text or "").lower().split())
    value = re.sub(r"\b\d+\b", "#", value)
    value = re.sub(r"[^a-z\u4e00-\u9fff#]+", " ", value)
    return " ".join(value.split())


def _inject_visual_blocks(
    pages: list[dict[str, Any]],
    data: dict[str, Any],
    parse_identity: dict[str, Any],
    *,
    include_uncommitted_visual_supplements: bool = False,
) -> None:
    base_block_counts = {
        id(page): len(page.get("blocks") or [])
        for page in pages
    }
    page_map = {int(page.get("page", 0)): page for page in pages}
    existing_ids: set[str] = {
        str(block.get("block_id"))
        for page in pages
        for block in page.get("blocks", [])
        if block.get("block_id")
    }

    for fig in data.get("logical_figures", []) or []:
        if not isinstance(fig, dict):
            continue
        page_num = int(fig.get("page_idx", 0)) + 1
        bbox = _normalize_bbox(fig.get("full_bbox_page_pts")) or _normalize_bbox(fig.get("body_bbox_page_pts"))
        if not bbox or page_num not in page_map:
            continue
        block_id = str(fig.get("figure_id") or f"figure_p{page_num}_{len(existing_ids)}")
        _append_visual_block(
            page_map[page_num],
            existing_ids,
            block_id=block_id,
            block_type="figure",
            bbox=bbox,
            text=str(fig.get("caption_text") or fig.get("caption") or "Figure").strip(),
        )

    for fig in data.get("figures", []) or []:
        if not isinstance(fig, dict):
            continue
        page_num = int(fig.get("page") or 0)
        if page_num not in page_map:
            continue
        bbox = (
            _normalize_bbox(fig.get("group_bbox"))
            or _normalize_bbox(fig.get("figure_bbox"))
            or _normalize_bbox(fig.get("caption_bbox"))
            or _normalize_bbox(fig.get("bbox"))
        )
        if not bbox:
            continue
        block_id = str(fig.get("figure_id") or f"figure_p{page_num}_{len(existing_ids)}")
        _append_visual_block(
            page_map[page_num],
            existing_ids,
            block_id=block_id,
            block_type="figure",
            bbox=bbox,
            text=str(fig.get("caption") or fig.get("label") or "Figure").strip(),
        )

    for img in data.get("images", []) or []:
        if not isinstance(img, dict):
            continue
        page_num = int(img.get("page") or 0)
        bbox = _normalize_bbox(img.get("bbox"))
        if not bbox or page_num not in page_map:
            continue
        block_id = str(img.get("id") or f"image_p{page_num}_{len(existing_ids)}")
        _append_visual_block(
            page_map[page_num],
            existing_ids,
            block_id=block_id,
            block_type="figure",
            bbox=bbox,
            text="Image",
        )

    # VLM output is intentionally additive and only active for the current
    # parse generation. It never replaces local OCR/PDF text or MinerU blocks.
    for supplement in active_visual_supplements(
        data,
        parse_identity,
        require_committed=not include_uncommitted_visual_supplements,
    ):
        page_num = int(supplement.get("page") or 0)
        bbox = _normalize_bbox(supplement.get("bbox"))
        if page_num not in page_map or not bbox:
            continue
        item_id = str(supplement.get("id") or "").strip()
        if not item_id:
            continue
        _append_visual_block(
            page_map[page_num],
            existing_ids,
            block_id=item_id,
            block_type="caption",
            bbox=bbox,
            text=str(supplement.get("text") or supplement.get("analysis") or "").strip(),
            extra={
                "visual_enhancement": True,
                "block_type": str(supplement.get("block_type") or "visual_enrichment"),
                "visual_source": str(supplement.get("source") or "visual_vlm"),
                "route": str(supplement.get("route") or "local"),
                "purpose": str(supplement.get("purpose") or "figure_description"),
                "confidence": supplement.get("confidence"),
                "prompt_version": str(supplement.get("prompt_version") or ""),
                "bbox_hash": str(supplement.get("bbox_hash") or ""),
                "provider": str(supplement.get("provider") or ""),
                "model": str(supplement.get("model") or ""),
                "visual_supplement_revision": str(parse_identity.get("visual_supplement_revision") or ""),
                "visual_model": dict(supplement.get("visual_model") or {}),
                "figure_id": str(supplement.get("figure_id") or ""),
            },
        )

    for page in pages:
        blocks = list(page.get("blocks") or [])
        base_count = min(base_block_counts.get(id(page), 0), len(blocks))
        base_blocks = blocks[:base_count]
        visual_blocks = sorted(
            blocks[base_count:],
            key=lambda b: (float((b.get("bbox") or [0, 0, 0, 0])[1]), float((b.get("bbox") or [0, 0, 0, 0])[0])),
        )
        page["blocks"] = [*base_blocks, *visual_blocks]


def _append_visual_block(
    page: dict[str, Any],
    existing_ids: set[str],
    *,
    block_id: str,
    block_type: str,
    bbox: list[float],
    text: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if block_id in existing_ids:
        return
    existing_ids.add(block_id)
    block = {
        "block_id": block_id,
        "type": block_type,
        "bbox": bbox,
        "text": _limit_text(text or block_type, 1000),
        "section_id": None,
    }
    if extra:
        block.update(extra)
    page.setdefault("blocks", []).append(block)


def _demote_adjacent_table_titles(blocks: list[dict[str, Any]]) -> None:
    """将 TABLE 标签后的表题降级为 caption，避免进入章节大纲。"""
    for idx, block in enumerate(blocks[:-1]):
        text = str(block.get("text") or "").strip()
        if not _RE_TABLE_LABEL.match(text):
            continue

        next_block = blocks[idx + 1]
        next_text = str(next_block.get("text") or "").strip()
        if not next_text or len(next_text) > 220:
            continue
        if _looks_like_numeric_measurement_line(next_text):
            continue
        if not _looks_like_table_title(next_text):
            continue

        label_bbox = _normalize_bbox(block.get("bbox"))
        title_bbox = _normalize_bbox(next_block.get("bbox"))
        if label_bbox and title_bbox:
            y_gap = title_bbox[1] - label_bbox[3]
            if y_gap < -4 or y_gap > 42:
                continue

        next_block["type"] = "caption"
        next_block["caption_role"] = "table_title"
        next_block.pop("level", None)


def _build_outline(toc_items: list[dict[str, Any]], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outline: list[dict[str, Any]] = []
    if toc_items:
        for idx, item in enumerate(toc_items):
            page_num = item.get("page", 1)
            title = _normalize_outline_title(item.get("title", ""))
            if not title or _looks_like_non_section_noise(title):
                continue
            if int(page_num or 1) == 1 and _looks_like_front_matter_title(title):
                continue
            first_block = _first_block_id_on_page(pages, page_num)
            outline.append({
                "section_id": f"s{len(outline) + 1}",
                "title": _limit_text(title, 180),
                "level": max(1, min(int(item.get("level", 1)), 6)),
                "page": page_num,
                "first_block": first_block,
                "source": "toc",
            })
        if _toc_outline_quality_ok(outline):
            return _merge_bare_outline_labels(outline)

    body_heading_seen = False
    references_seen = False
    mineru_appendix_mode = ""
    for page in pages:
        page_num = int(page.get("page", 1))
        page_width = float(page.get("width_pts") or 612.0)
        page_height = float(page.get("height_pts") or 792.0)
        for block in page.get("blocks", []):
            if block.get("type") != "heading":
                continue
            title = _normalize_outline_title(block.get("text"))
            if not title or len(title) > 180:
                # Record every deliberate drop. A heading block that leaves this
                # loop with neither an outline entry nor an exclusion reason is a
                # *silent* loss, and that is the only thing the structure
                # diagnostics can safely fail closed on.
                block["structure_exclusion_reason"] = "unusable_heading_title"
                continue
            is_mineru_heading = str(block.get("source") or "").strip().lower() == "mineru_vlm"
            has_mineru_structure_signal = is_mineru_heading and str(
                block.get("heading_source") or ""
            ).strip().lower() in {
                "mineru_text_level",
                "mineru_type",
                "mineru_run_in_numbered",
            }
            if references_seen and _looks_like_post_references_noise(title):
                block["type"] = "artifact"
                if str(block.get("source") or "").strip().lower() == "mineru_vlm":
                    block["structure_exclusion_reason"] = "post_references_artifact"
                block.pop("level", None)
                continue
            explicit_appendix_level = _mineru_appendix_heading_level(title)
            bare_appendix_level = _mineru_appendix_heading_level(title, allow_bare_letter=True)
            is_bare_appendix_heading = (
                bare_appendix_level is not None
                and explicit_appendix_level is None
            )
            if references_seen and is_mineru_heading:
                if not mineru_appendix_mode:
                    if explicit_appendix_level is not None:
                        mineru_appendix_mode = "explicit"
                    elif is_bare_appendix_heading:
                        mineru_appendix_mode = "bare"
                    else:
                        block["type"] = "artifact"
                        block["structure_exclusion_reason"] = "post_references_artifact"
                        block.pop("level", None)
                        continue
                elif mineru_appendix_mode == "bare" and not is_bare_appendix_heading:
                    block["type"] = "artifact"
                    block["structure_exclusion_reason"] = "post_references_artifact"
                    block.pop("level", None)
                    continue
                if explicit_appendix_level is not None or bare_appendix_level is not None:
                    block["level"] = explicit_appendix_level or bare_appendix_level
            # The native PDF route must remain conservative because a line
            # starting with a large number is often a citation or footer.
            # MinerU's explicit text_level/type signal is stronger structural
            # evidence, though: a textbook can legitimately contain chapter
            # 51+ and those anchors must not disappear from the unified index.
            if _looks_like_non_section_noise(title) and not has_mineru_structure_signal:
                block["structure_exclusion_reason"] = "non_section_noise"
                continue
            # 同一条 MinerU 豁免（见上方注释）也适用于这里：run-in 判定是给原生
            # PDF 路线用的段落启发式，而 MinerU 侧的真正 run-in 段落早在
            # ``_split_run_in_numbered_heading_blocks`` 就被拆成标题 + 正文了。
            # 对一个带 text_level/type 显式结构信号的标题再判一次，只会把长标题、
            # 以句号结尾的标题、含 ". 大写" 的标题白白丢出大纲。
            if _looks_like_run_in_paragraph_heading(title) and not has_mineru_structure_signal:
                block["structure_exclusion_reason"] = "run_in_paragraph_heading"
                continue
            if page_num == 1 and not body_heading_seen and _looks_like_front_matter_title(title):
                block["structure_exclusion_reason"] = "front_matter_title"
                continue
            if block.get("source") != "mineru_vlm" and _looks_like_layout_noise(title, _normalize_bbox(block.get("bbox")), page_width, page_height):
                block["structure_exclusion_reason"] = "layout_noise"
                continue
            body_heading_seen = True
            if is_reference_heading(title):
                references_seen = True
            outline.append({
                "section_id": f"s{len(outline) + 1}",
                "title": title,
                "level": max(1, min(int(block.get("level") or 2), 6)),
                "page": page_num,
                "first_block": block.get("block_id"),
                "source": "heading",
            })

    if outline:
        # Do not silently truncate a long paper or textbook. Every heading is
        # a section anchor; truncating here assigns all later blocks to the
        # last retained section and poisons every downstream consumer.
        merged = _merge_bare_outline_labels(outline)
        # A merged-away heading keeps no outline entry of its own, so tag its
        # block too — otherwise the diagnostics below would read a legitimate
        # bare-label merge as silent structure loss.
        surviving_blocks = {str(item.get("first_block") or "") for item in merged}
        merged_away = {
            str(item.get("first_block") or "")
            for item in outline
            if str(item.get("first_block") or "") not in surviving_blocks
        }
        if merged_away:
            for page in pages:
                for block in page.get("blocks", []):
                    if isinstance(block, dict) and str(block.get("block_id") or "") in merged_away:
                        block.setdefault("structure_exclusion_reason", "merged_into_previous_heading")
        return merged

    first_page = pages[0].get("page", 1) if pages else 1
    return [{
        "section_id": "s1",
        "title": "全文",
        "level": 1,
        "page": first_page,
        "first_block": _first_block_id_on_page(pages, first_page),
        "source": "fallback",
    }]


def _mineru_appendix_heading_level(
    title: str,
    *,
    allow_bare_letter: bool = False,
) -> int | None:
    normalized = " ".join(str(title or "").split())
    if re.match(r"^(?:appendix|appendices|supplementary\s+material)\b", normalized, re.IGNORECASE):
        match = re.match(r"^(?:appendix\s+)?[A-Z](?:\.(\d+(?:\.\d+)*))?(?:[.)]?\s+|$)", normalized)
        return 1 + len(match.group(1).split(".")) if match and match.group(1) else 1
    if not allow_bare_letter:
        return None
    match = re.match(r"^[A-Z](?:\.(\d+(?:\.\d+)*))?(?:[.)]?\s+|$)", normalized)
    if not match:
        return None
    return 1 + len(match.group(1).split(".")) if match.group(1) else 1


def _toc_outline_quality_ok(outline: list[dict[str, Any]]) -> bool:
    if not outline:
        return False
    titles = [str(item.get("title") or "").strip() for item in outline]
    if len(outline) == 1:
        return False
    useful_titles = [
        title for title in titles
        if title and re.sub(r"\s+", " ", title.lower()).strip() not in _GENERIC_TOC_TITLES
    ]
    return len(useful_titles) >= 2


def _merge_bare_outline_labels(outline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    idx = 0
    while idx < len(outline):
        item = dict(outline[idx])
        title = str(item.get("title") or "").strip()
        next_item = dict(outline[idx + 1]) if idx + 1 < len(outline) else None
        if next_item and _outline_items_should_merge(title, item, next_item):
            next_title = str(next_item.get("title") or "").strip()
            item["title"] = _limit_text(f"{title} {next_title}".strip(), 180)
            idx += 1
            title = str(item.get("title") or "").strip()
            next_item = dict(outline[idx + 1]) if idx + 1 < len(outline) else None
            if next_item and _outline_items_should_merge(title, item, next_item):
                next_title = str(next_item.get("title") or "").strip()
                item["title"] = _limit_text(f"{title} {next_title}".strip(), 180)
                idx += 1
        merged.append(item)
        idx += 1

    for new_idx, item in enumerate(merged):
        item["section_id"] = f"s{new_idx + 1}"
    return merged


def _outline_items_should_merge(title: str, item: dict[str, Any], next_item: dict[str, Any]) -> bool:
    next_title = str(next_item.get("title") or "").strip()
    if not title or not next_title:
        return False
    if int(item.get("page") or 0) != int(next_item.get("page") or -1):
        return False
    if int(item.get("level") or 1) != int(next_item.get("level") or 1):
        return False
    if _RE_CANONICAL_HEADING.match(next_title) or _RE_NUMBERED_HEADING.match(next_title) or _RE_ROMAN_HEADING.match(next_title) or _RE_ALPHA_HEADING.match(next_title):
        return False
    if re.fullmatch(r"[A-Z](?:\.\d+)?|\d+(?:\.\d+)*", title):
        return True
    if re.match(r"^[A-Z]\.\d+\s+", title) and re.search(r"\b(of|for|with|and|or|to|from|Object)$", title):
        return True
    return False


def _assign_sections(pages: list[dict[str, Any]], outline: list[dict[str, Any]]) -> None:
    anchors: list[tuple[int, int, str]] = []
    block_order: dict[str, tuple[int, int]] = {}
    for page in pages:
        page_num = int(page.get("page", 1))
        for idx, block in enumerate(page.get("blocks", [])):
            block_id = block.get("block_id")
            if block_id:
                block_order[str(block_id)] = (page_num, idx)

    for item in outline:
        block_id = item.get("first_block")
        if block_id and str(block_id) in block_order:
            page_num, block_idx = block_order[str(block_id)]
        else:
            page_num, block_idx = int(item.get("page", 1)), 0
        anchors.append((page_num, block_idx, str(item.get("section_id"))))

    anchors.sort(key=lambda x: (x[0], x[1]))
    if not anchors:
        return

    current = anchors[0][2]
    anchor_pos = 0
    for page in pages:
        page_num = int(page.get("page", 1))
        for idx, block in enumerate(page.get("blocks", [])):
            while anchor_pos + 1 < len(anchors) and (page_num, idx) >= (anchors[anchor_pos + 1][0], anchors[anchor_pos + 1][1]):
                anchor_pos += 1
                current = anchors[anchor_pos][2]
            block["section_id"] = current


def _annotate_block_roles(pages: list[dict[str, Any]], outline: list[dict[str, Any]]) -> None:
    """Persist semantic roles once section anchors are available.

    The retrieval layer deliberately trusts persisted roles so it does not
    have to re-infer bibliography and front-matter semantics on each query.
    """
    section_titles = {
        str(item.get("section_id") or ""): str(item.get("title") or "")
        for item in outline
        if isinstance(item, dict) and str(item.get("section_id") or "")
    }
    for page in pages:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            section_title = section_titles.get(str(block.get("section_id") or ""), "")
            block.update(annotate_block_role(block, section_title=section_title))


def _exclude_post_reference_template_blocks(pages: list[dict[str, Any]]) -> None:
    """Fence terminal submission templates off from the preceding appendix.

    A checklist heading is not enough on its own: its following instruction
    paragraphs otherwise inherit the last valid appendix section id and get
    summarized as source material.
    """
    references_seen = False
    template_started = False
    for page in pages:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            text = str(block.get("text") or "")
            if str(block.get("type") or "").lower() == "heading" and is_reference_heading(text):
                references_seen = True
            if references_seen and is_post_reference_template_artifact(text):
                template_started = True
            if not template_started:
                continue
            block["exclude_from_reading"] = True
            block["post_reference_template_artifact"] = True
            block["content_role"] = "artifact"
            block["role_confidence"] = 1.0
            block["role_reason"] = "post_reference_template"
            block["block_role_version"] = 1


def _first_block_id_on_page(pages: list[dict[str, Any]], page_num: int) -> str | None:
    for page in pages:
        if int(page.get("page", 0)) != int(page_num):
            continue
        for block in page.get("blocks", []):
            if block.get("block_id"):
                return block["block_id"]
    return None


def _detect_page_layout_regions(page: Any) -> list[dict[str, Any]]:
    """YOLO 可用时检测页面版面区域，失败静默降级到纯文本路径。"""
    try:
        from PIL import Image
        from services.layout_service import detect_layout, is_available, pixel_bbox_to_page_pts

        if not is_available():
            return []

        scale = 2.0
        import fitz
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        detections = detect_layout(image, conf=0.18)
        regions = []
        for det in detections or []:
            bbox = det.get("bbox")
            if not bbox:
                continue
            page_bbox = _normalize_bbox(pixel_bbox_to_page_pts(
                bbox,
                pix.width,
                pix.height,
                float(page.rect.width),
                float(page.rect.height),
            ))
            if not page_bbox:
                continue
            regions.append({
                "category_id": int(det.get("category_id", -1)),
                "category_name": str(det.get("category_name") or ""),
                "bbox": page_bbox,
                "score": float(det.get("score") or 0),
            })
        return regions
    except Exception as exc:
        logger.debug("[BlockIndex] DocLayout-YOLO layout detection skipped: %s", exc)
        return []


def _apply_layout_regions(
    blocks: list[dict[str, Any]],
    layout_regions: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> None:
    if not blocks or not layout_regions:
        return

    if _layout_regions_are_usable(layout_regions):
        for block in blocks:
            bbox = _normalize_bbox(block.get("bbox"))
            text = str(block.get("text") or "").strip()
            if not bbox or not text:
                continue
            region = _best_layout_region_for_bbox(bbox, layout_regions)
            if not region:
                block["layout_region"] = "Outside"
                block["layout_excluded_from_outline"] = True
                block["type"] = "artifact"
                block.pop("level", None)
                continue
            _apply_yolo_region_type(block, region, page_height)
        return

    title_regions = [r for r in layout_regions if r.get("category_name") == "Title"]
    text_regions = [r for r in layout_regions if r.get("category_name") == "Text"]
    table_regions = [r for r in layout_regions if r.get("category_name") == "TableBody"]
    image_regions = [r for r in layout_regions if r.get("category_name") == "ImageBody"]
    abandon_regions = [r for r in layout_regions if r.get("category_name") == "Abandon"]
    footnote_regions = [r for r in layout_regions if r.get("category_name") in {"TableFootnote", "EquationNumber"}]
    excluded_regions = table_regions + image_regions + abandon_regions + footnote_regions

    for block in blocks:
        bbox = _normalize_bbox(block.get("bbox"))
        text = str(block.get("text") or "").strip()
        if not bbox or not text:
            continue

        excluded = _best_overlapping_region(bbox, excluded_regions)
        if excluded and _block_should_demote_for_layout(block, excluded):
            block["layout_region"] = excluded.get("category_name")
            block["layout_score"] = excluded.get("score")
            block["layout_excluded_from_outline"] = True
            block.pop("level", None)
            if excluded.get("category_name") in {"Abandon", "TableFootnote", "EquationNumber"}:
                block["type"] = "artifact"
            elif _looks_like_region_label_noise(text) or block.get("type") == "heading":
                block["type"] = "artifact"
            elif excluded.get("category_name") == "TableBody":
                block["type"] = "table"
            elif excluded.get("category_name") == "ImageBody":
                block["type"] = "figure"
            continue

        text_region = _best_overlapping_region(bbox, text_regions)
        if text_region and _block_should_demote_inside_text_region(block, text_region):
            block["layout_region"] = "Text"
            block["layout_score"] = text_region.get("score")
            block["layout_excluded_from_outline"] = True
            block["type"] = "paragraph"
            block.pop("level", None)
            continue

        title = _best_overlapping_region(bbox, title_regions)
        if title and _block_should_promote_for_layout(block, title, page_width, page_height):
            block["layout_region"] = "Title"
            block["layout_score"] = title.get("score")
            block["type"] = "heading"
            block["level"] = _layout_title_level(text, bbox, page_height)
            block["layout_promoted"] = True


def _best_overlapping_region(
    bbox: list[float],
    regions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for region in regions:
        region_bbox = _normalize_bbox(region.get("bbox"))
        if not region_bbox:
            continue
        score = _bbox_intersection_ratio(bbox, region_bbox)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, region)
    return best[1] if best else None


_YOLO_TEXTLIKE_CATEGORIES = {
    "Title",
    "Text",
    "Abandon",
    "ImageCaption",
    "TableCaption",
    "TableFootnote",
    "EquationNumber",
    "InterlineEquation",
}
_YOLO_KNOWN_CATEGORIES = _YOLO_TEXTLIKE_CATEGORIES | {"ImageBody", "TableBody"}
_YOLO_STRICT_TEXT_CATEGORIES = {"Title", "Text", "Abandon"}


def _layout_regions_are_usable(layout_regions: list[dict[str, Any]]) -> bool:
    return any(_layout_region_category(region) in _YOLO_STRICT_TEXT_CATEGORIES for region in layout_regions or [])


def _layout_region_category(region: dict[str, Any] | None) -> str:
    return str((region or {}).get("category_name") or "").strip()


def _best_layout_region_for_bbox(bbox: list[float], regions: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: tuple[float, float, dict[str, Any]] | None = None
    for region in regions or []:
        category = _layout_region_category(region)
        if category not in _YOLO_KNOWN_CATEGORIES:
            continue
        region_bbox = _normalize_bbox(region.get("bbox"))
        if not region_bbox:
            continue
        coverage = _bbox_intersection_ratio(bbox, region_bbox)
        center_inside = _bbox_center_inside(bbox, region_bbox)
        if coverage < 0.10 and not center_inside:
            continue
        priority = _layout_category_priority(category)
        score = coverage + (0.35 if center_inside else 0.0) + priority
        model_score = float(region.get("score") or 0)
        if best is None or (score, model_score) > (best[0], best[1]):
            best = (score, model_score, region)
    return best[2] if best else None


def _layout_category_priority(category: str) -> float:
    if category == "Abandon":
        return 0.08
    if category == "Title":
        return 0.06
    if category in {"TableBody", "ImageBody"}:
        return 0.04
    if category in {"TableCaption", "ImageCaption"}:
        return 0.03
    return 0.0


def _apply_yolo_region_type(block: dict[str, Any], region: dict[str, Any], page_height: float) -> None:
    category = _layout_region_category(region)
    block["layout_region"] = category
    block["layout_score"] = region.get("score")
    block["layout_primary_classifier"] = "doclayout_yolo"

    if category in {"Abandon", "TableFootnote", "EquationNumber"}:
        block["type"] = "artifact"
        block["layout_excluded_from_outline"] = True
        block.pop("level", None)
        return

    if category == "Text":
        block["type"] = "paragraph"
        block["layout_excluded_from_outline"] = True
        block.pop("level", None)
        return

    if category == "Title":
        text = str(block.get("text") or "").strip()
        block["type"] = "heading"
        block["level"] = _layout_title_level(text, _normalize_bbox(block.get("bbox")) or [], page_height)
        block.pop("layout_excluded_from_outline", None)
        block["layout_promoted"] = True
        return

    if category == "TableBody":
        block["type"] = "table"
        block["layout_excluded_from_outline"] = True
        block.pop("level", None)
        return

    if category == "ImageBody":
        block["type"] = "figure"
        block["layout_excluded_from_outline"] = True
        block.pop("level", None)
        return

    if category in {"TableCaption", "ImageCaption"}:
        block["type"] = "caption"
        block["layout_excluded_from_outline"] = True
        block.pop("level", None)
        return

    if category == "InterlineEquation":
        block["type"] = "paragraph"
        block["layout_excluded_from_outline"] = True
        block.pop("level", None)


def _block_should_demote_for_layout(block: dict[str, Any], region: dict[str, Any]) -> bool:
    bbox = _normalize_bbox(block.get("bbox"))
    region_bbox = _normalize_bbox(region.get("bbox"))
    if not bbox or not region_bbox:
        return False
    coverage = _bbox_intersection_ratio(bbox, region_bbox)
    center_inside = _bbox_center_inside(bbox, region_bbox)
    text = str(block.get("text") or "").strip()
    if coverage >= 0.45 or center_inside:
        return True
    return block.get("type") == "heading" and coverage >= 0.25 and _looks_like_region_label_noise(text)


def _block_should_promote_for_layout(
    block: dict[str, Any],
    region: dict[str, Any],
    page_width: float,
    page_height: float,
) -> bool:
    bbox = _normalize_bbox(block.get("bbox"))
    region_bbox = _normalize_bbox(region.get("bbox"))
    if not bbox or not region_bbox:
        return False
    text = str(block.get("text") or "").strip()
    if not _looks_like_embedded_heading_line(text):
        return False
    if _looks_like_layout_noise(text, bbox, page_width, page_height):
        return False
    coverage = _bbox_intersection_ratio(bbox, region_bbox)
    return coverage >= 0.18 or _bbox_center_inside(bbox, region_bbox)


def _block_should_demote_inside_text_region(block: dict[str, Any], region: dict[str, Any]) -> bool:
    if block.get("type") != "heading" or not block.get("split_from_mixed_block"):
        return False
    text = " ".join(str(block.get("text") or "").split())
    if not text:
        return True
    if _RE_CANONICAL_HEADING.match(text) or _RE_NUMBERED_HEADING.match(text) or _RE_ROMAN_HEADING.match(text) or _RE_ALPHA_HEADING.match(text):
        return False
    if len(text.split()) > 6:
        return False
    bbox = _normalize_bbox(block.get("bbox"))
    region_bbox = _normalize_bbox(region.get("bbox"))
    if not bbox or not region_bbox:
        return False
    return _bbox_center_inside(bbox, region_bbox) or _bbox_intersection_ratio(bbox, region_bbox) >= 0.60


def _layout_title_level(text: str, bbox: list[float], page_height: float) -> int:
    explicit = _embedded_heading_level(text)
    stripped = " ".join(str(text or "").split())
    if (
        _RE_CANONICAL_HEADING.match(stripped)
        or _RE_NUMBERED_HEADING.match(stripped)
        or _RE_ROMAN_HEADING.match(stripped)
        or _RE_ALPHA_HEADING.match(stripped)
    ):
        return explicit
    if bbox and bbox[1] < page_height * 0.20 and len(text.split()) >= 5:
        return 1
    return 2


def _looks_like_region_label_noise(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if _looks_like_algorithm_line(stripped) or _looks_like_numeric_measurement_line(stripped):
        return True
    if _RE_CAPTION.match(stripped):
        return False
    words = stripped.split()
    if len(words) <= 4:
        return True
    if len(stripped) <= 40 and not stripped.endswith("."):
        return True
    return False


def _split_line_group_by_layout_regions(
    lines: list[dict[str, Any]],
    layout_regions: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], str]]:
    valid_lines = [line for line in lines or [] if _line_text(line)]
    if not valid_lines:
        return []
    if not _layout_regions_are_usable(layout_regions):
        return [(valid_lines, "")]

    groups: list[tuple[list[dict[str, Any]], str]] = []
    current_lines: list[dict[str, Any]] = []
    current_category = ""
    for line in valid_lines:
        category = _line_layout_category(line, layout_regions)
        if current_lines and category != current_category:
            groups.append((current_lines, current_category))
            current_lines = []
        current_lines.append(line)
        current_category = category
    if current_lines:
        groups.append((current_lines, current_category))
    return groups


def _line_layout_category(line: dict[str, Any], layout_regions: list[dict[str, Any]]) -> str:
    bbox = _line_bbox(line)
    if not bbox:
        return ""
    region = _best_layout_region_for_bbox(bbox, layout_regions)
    return _layout_region_category(region)


def _dominant_line_layout_category(lines: list[dict[str, Any]], layout_regions: list[dict[str, Any]]) -> str:
    if not _layout_regions_are_usable(layout_regions):
        return ""
    weights: dict[str, int] = {}
    for line in lines or []:
        category = _line_layout_category(line, layout_regions)
        if not category:
            continue
        weights[category] = weights.get(category, 0) + max(1, len(_line_text(line)))
    if not weights:
        return ""
    return max(weights.items(), key=lambda item: item[1])[0]


def _bbox_intersection_ratio(inner: list[float], outer: list[float]) -> float:
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return intersection / area


def _bbox_center_inside(inner: list[float], outer: list[float]) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _split_text_block_line_groups(
    lines: list[dict[str, Any]],
    *,
    median_font: float,
    body_font_key: str,
    page_width: float,
) -> list[tuple[list[dict[str, Any]], bool]]:
    valid_lines = [line for line in lines or [] if _line_text(line)]
    if len(valid_lines) <= 1:
        return [(valid_lines, False)] if valid_lines else []

    groups: list[tuple[list[dict[str, Any]], bool]] = []
    idx = 0
    while idx < len(valid_lines) - 1:
        line = valid_lines[idx]
        remaining = valid_lines[idx + 1:]
        if not _line_should_split_as_heading(
            line,
            remaining,
            median_font=median_font,
            body_font_key=body_font_key,
            page_width=page_width,
        ):
            break
        groups.append(([line], True))
        idx += 1

    if not groups:
        return [(valid_lines, False)]
    if idx < len(valid_lines):
        groups.append((valid_lines[idx:], False))
    return groups


def _line_should_split_as_heading(
    line: dict[str, Any],
    remaining_lines: list[dict[str, Any]],
    *,
    median_font: float,
    body_font_key: str,
    page_width: float,
) -> bool:
    text = _line_text(line)
    if not text or not _looks_like_embedded_heading_line(text):
        return False
    if _looks_like_layout_noise(text, _line_bbox(line), page_width, 792.0):
        return False

    remaining_text = " ".join(_line_text(item) for item in remaining_lines if _line_text(item))
    if len(remaining_text) < 24:
        return False

    if _RE_CANONICAL_HEADING.match(text) or _RE_NUMBERED_HEADING.match(text) or _RE_ROMAN_HEADING.match(text) or _RE_ALPHA_HEADING.match(text):
        return True

    line_font = _dominant_font_key([line])
    next_font = _dominant_font_key(remaining_lines[: min(3, len(remaining_lines))])
    font_differs_from_body = bool(line_font and body_font_key and line_font != body_font_key)
    font_differs_from_next = bool(line_font and next_font and line_font != next_font)
    size = _line_max_font(line, median_font)
    size_signal = median_font > 0 and size >= median_font * 1.08
    weight_signal = _line_is_bold([line])
    return (font_differs_from_body or font_differs_from_next or size_signal or weight_signal) and _looks_like_heading_text(text)


def _looks_like_embedded_heading_line(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped or len(stripped) > 140:
        return False
    if _RE_CAPTION.match(stripped) or _looks_like_numeric_measurement_line(stripped) or _looks_like_algorithm_line(stripped):
        return False
    if _RE_CANONICAL_HEADING.match(stripped) or _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return True
    words = stripped.split()
    if not 1 <= len(words) <= 12:
        return False
    return _looks_like_heading_text(stripped)


def _embedded_heading_level(text: str) -> int:
    stripped = " ".join(str(text or "").split())
    numbered = _RE_NUMBERED_HEADING.match(stripped)
    if numbered:
        return max(1, min(numbered.group(1).count(".") + 1, 4))
    if _RE_ROMAN_HEADING.match(stripped) or _RE_CANONICAL_HEADING.match(stripped):
        return 1
    if _RE_ALPHA_HEADING.match(stripped):
        return 2
    return 2


def _classify_text_block(
    text: str,
    max_font: float,
    median_font: float,
    *,
    bbox: list[float] | None = None,
    page_width: float = 612.0,
    page_height: float = 792.0,
    lines: list[dict[str, Any]] | None = None,
) -> tuple[str, int | None]:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return "paragraph", None
    if _RE_CAPTION.match(stripped):
        return "caption", None
    if _looks_like_numeric_measurement_line(stripped):
        return "paragraph", None
    if _looks_like_algorithm_line(stripped):
        return "paragraph", None

    heading_level = _heading_level(
        stripped,
        max_font,
        median_font,
        bbox=bbox,
        page_width=page_width,
        page_height=page_height,
        lines=lines,
    )
    if heading_level:
        return "heading", heading_level
    return "paragraph", None


def _heading_level(
    text: str,
    max_font: float,
    median_font: float,
    *,
    bbox: list[float] | None = None,
    page_width: float = 612.0,
    page_height: float = 792.0,
    lines: list[dict[str, Any]] | None = None,
) -> int | None:
    if len(text) > 180:
        return None
    if _looks_like_algorithm_line(text):
        return None
    if _looks_like_non_section_noise(text):
        return None
    if _looks_like_layout_noise(text, bbox, page_width, page_height):
        return None

    numbered = _RE_NUMBERED_HEADING.match(text)
    if numbered:
        number = numbered.group(1)
        try:
            first_number = int(number.split(".")[0])
        except ValueError:
            first_number = 0
        suffix = numbered.group(2).strip()
        if first_number > 50:
            return None
        if len(suffix) > 80:
            return None
        if re.match(r"^\d{3,}(?:\.|$)", number) and _looks_like_publication_header_footer(text):
            return None
        return max(1, min(number.count(".") + 1, 4))

    if _RE_ROMAN_HEADING.match(text):
        return 1

    if _RE_ALPHA_HEADING.match(text):
        return 2

    if _RE_CANONICAL_HEADING.match(text):
        return 1

    alpha = re.sub(r"[^A-Za-z]", "", text)
    if (
        3 <= len(alpha) <= 80
        and alpha.isupper()
        and len(text.split()) <= 10
        and not re.match(r"^\s*\d", text)
    ):
        return 1

    if (
        median_font > 0
        and max_font >= median_font * 1.42
        and _looks_like_standalone_heading(text, bbox, page_width, lines)
    ):
        return 1 if max_font >= median_font * 1.55 else 2

    if (
        _line_is_bold(lines)
        and _looks_like_standalone_heading(text, bbox, page_width, lines)
        and _looks_like_heading_text(text)
    ):
        return 2

    return None


def _looks_like_layout_noise(
    text: str,
    bbox: list[float] | None,
    page_width: float,
    page_height: float,
) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if re.fullmatch(r"[\d.,:/()%+\-=×xX<>~]+", compact):
        return True
    if _looks_like_numeric_measurement_line(text):
        return True
    if _looks_like_algorithm_line(text):
        return True
    if _looks_like_non_section_noise(text):
        return True
    if re.match(r"^\s*\d{3,}\s+[A-Z]", text) and _looks_like_publication_header_footer(text):
        return True
    digit_ratio = sum(ch.isdigit() for ch in compact) / max(len(compact), 1)
    symbol_ratio = sum(not ch.isalnum() for ch in compact) / max(len(compact), 1)
    if digit_ratio > 0.45 and symbol_ratio > 0.12:
        return True
    if bbox:
        x0, y0, x1, y1 = bbox
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        if x0 < page_width * 0.03 or x1 > page_width * 0.97:
            return True
        if y0 < page_height * 0.08 and _looks_like_publication_header_footer(text):
            return True
        if y1 > page_height * 0.94 and _looks_like_publication_header_footer(text):
            return True
        if height > page_height * 0.28 and width < page_width * 0.10:
            return True
    return False


def _looks_like_publication_header_footer(text: str) -> bool:
    compact = " ".join(str(text or "").split())
    if not compact:
        return False
    if re.match(r"^\s*\d+\s+[A-Z]", compact):
        return True
    return bool(_RE_PUBLICATION_HEADER_CUE.search(compact))


def _looks_like_algorithm_line(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    if _RE_ALGORITHM_LINE.match(stripped):
        return True
    if re.search(r"\b(stage|step)\s*\d+\b", stripped, re.IGNORECASE) and re.search(r"\b(update|initialize|input|output)\b", stripped, re.IGNORECASE):
        return True
    return False


def _looks_like_non_section_noise(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if _RE_CANONICAL_HEADING.match(stripped):
        return False
    if _RE_KEYWORDS_LINE.match(stripped):
        return True
    numbered = _RE_NUMBERED_HEADING.match(stripped)
    if numbered:
        try:
            first_number = int(numbered.group(1).split(".")[0])
        except ValueError:
            first_number = 0
        if first_number > 50 or len(numbered.group(2).strip()) > 80:
            return True
    if _looks_like_reference_entry(stripped):
        return True
    if _looks_like_affiliation_or_author_line(stripped):
        return True
    return False


def _looks_like_front_matter_title(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if _RE_CANONICAL_HEADING.match(stripped):
        return False
    if _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return False
    return True


def _normalize_outline_title(value: Any) -> str:
    title = " ".join(str(value or "").split())
    title = re.sub(r"^(\d+(?:\.\d+)*|[A-Z]|[IVXLCM]+)\s+(.+)$", r"\1 \2", title)
    return _limit_text(title, 180)


def _looks_like_run_in_paragraph_heading(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    if _RE_CANONICAL_HEADING.match(stripped) or _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return False
    words = stripped.split()
    if len(words) > 12:
        return True
    if stripped.endswith(".") and len(words) >= 4:
        return True
    if re.search(r"\.\s+[A-Z]", stripped):
        return True
    if len(words) >= 9 and re.search(r"\[[0-9,\s]+\].*\b(is|are|was|were|has|have)\b", stripped, re.IGNORECASE):
        return True
    if len(words) >= 9 and re.search(r"\b(is|are|was|were)\s+(a|an|the|more)\b", stripped, re.IGNORECASE):
        return True
    return False


def _looks_like_reference_entry(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if _RE_REFERENCE_ENTRY.match(stripped):
        return True
    if _RE_AUTHOR_INITIAL_REFERENCE.match(stripped):
        return True
    numbered = _RE_NUMBERED_HEADING.match(stripped)
    if numbered:
        try:
            first_number = int(numbered.group(1).split(".")[0])
        except ValueError:
            first_number = 0
        if first_number > 50:
            return True
        if len(numbered.group(2)) > 80 and re.search(r"\bet\s+al\.?\b|,\s+[A-Z]\.", stripped, re.IGNORECASE):
            return True
    return bool(re.match(r"^\s*\d+[\.)]?\s+", stripped) and re.search(r"\bet\s+al\.?\b", stripped, re.IGNORECASE))


def _looks_like_post_references_noise(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if re.match(r"^\s*(?:[a-z0-9-]+\s+)?(?:paper\s+)?checklist\b", stripped, re.IGNORECASE):
        return True
    if _RE_CANONICAL_HEADING.match(stripped):
        return False
    return bool(_looks_like_reference_entry(stripped) or re.match(r"^\s*(?:\[\d+\]|\d+[\.)])\s+", stripped))


def _looks_like_affiliation_or_author_line(text: str) -> bool:
    return looks_like_affiliation_or_author_line(text)


def _looks_like_numeric_measurement_line(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    if _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return False
    decimal_tokens = _RE_DECIMAL_TOKEN.findall(stripped)
    if not decimal_tokens:
        return False
    alpha_tokens = re.findall(r"[A-Za-z]{2,}", stripped)
    number_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", stripped)
    if len(alpha_tokens) <= 2 and (len(number_tokens) >= 2 or len(stripped.split()) <= 8):
        return True
    compact = re.sub(r"\s+", "", stripped)
    digit_ratio = sum(ch.isdigit() for ch in compact) / max(len(compact), 1)
    return len(alpha_tokens) <= 3 and digit_ratio > 0.35 and len(stripped.split()) <= 12


def _looks_like_table_title(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    words = stripped.split()
    if not 2 <= len(words) <= 18:
        return False
    alpha = re.sub(r"[^A-Za-z]", "", stripped)
    if alpha and alpha.isupper() and len(alpha) >= 6:
        return True
    if stripped.endswith("."):
        return False
    return _looks_like_heading_text(stripped)


def _looks_like_standalone_heading(
    text: str,
    bbox: list[float] | None,
    page_width: float,
    lines: list[dict[str, Any]] | None,
) -> bool:
    words = text.split()
    if not 1 <= len(words) <= 12:
        return False
    if "\n" in text and len([line for line in text.splitlines() if line.strip()]) > 2:
        return False
    if not bbox:
        return True

    x0, _y0, x1, _y1 = bbox
    width = max(1.0, x1 - x0)
    center = (x0 + x1) / 2
    is_centered = abs(center - page_width / 2) <= page_width * 0.18
    is_near_text_start = page_width * 0.06 <= x0 <= page_width * 0.28
    is_not_tiny_label = width >= page_width * 0.10
    if not is_not_tiny_label:
        return False

    boldish = False
    for line in lines or []:
        for span in line.get("spans", []) or []:
            font = str(span.get("font") or "").lower()
            flags = int(span.get("flags") or 0)
            if flags & 16 or _font_has_weight_signal(font):
                boldish = True
                break
        if boldish:
            break
    return (is_centered or is_near_text_start) and (boldish or len(words) <= 6)


def _dominant_page_body_font_key(raw_blocks: list[dict[str, Any]], median_font: float) -> str:
    weighted: dict[str, int] = {}
    fallback: dict[str, int] = {}
    for block in raw_blocks or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                key = _normalize_font_key(span.get("font"))
                if not key:
                    continue
                text = str(span.get("text") or "")
                weight = max(1, len(text.strip()))
                fallback[key] = fallback.get(key, 0) + weight
                try:
                    size = float(span.get("size", 0) or 0)
                except (TypeError, ValueError):
                    size = 0
                if median_font > 0 and 0 < size <= median_font * 1.15:
                    weighted[key] = weighted.get(key, 0) + weight
    pool = weighted or fallback
    if not pool:
        return ""
    return max(pool.items(), key=lambda item: item[1])[0]


def _dominant_font_key(lines: list[dict[str, Any]] | None) -> str:
    weighted: dict[str, int] = {}
    for line in lines or []:
        for span in line.get("spans", []) or []:
            key = _normalize_font_key(span.get("font"))
            if not key:
                continue
            text = str(span.get("text") or "")
            weighted[key] = weighted.get(key, 0) + max(1, len(text.strip()))
    if not weighted:
        return ""
    return max(weighted.items(), key=lambda item: item[1])[0]


def _normalize_font_key(font: Any) -> str:
    value = str(font or "").strip().lower()
    if not value:
        return ""
    if "+" in value:
        value = value.split("+", 1)[1]
    return re.sub(r"\s+", "", value)


def _font_has_weight_signal(font: Any) -> bool:
    value = str(font or "").lower()
    return bool(re.search(r"(bold|semibold|semi-bold|demibold|demi-bold|extrabold|extra-bold|black|heavy|medium)", value))


def _line_text(line: dict[str, Any]) -> str:
    pieces = []
    for span in line.get("spans", []) or []:
        text = str(span.get("text") or "")
        if text:
            pieces.append(text)
    return _clean_pdf_text(" ".join("".join(pieces).split()))


def _line_bbox(line: dict[str, Any]) -> list[float] | None:
    return _merge_bboxes(span.get("bbox") for span in line.get("spans", []) or [])


def _line_max_font(line: dict[str, Any], fallback: float) -> float:
    sizes = []
    for span in line.get("spans", []) or []:
        try:
            size = float(span.get("size", 0) or 0)
        except (TypeError, ValueError):
            continue
        if size > 0:
            sizes.append(size)
    return max(sizes or [fallback])


def _line_is_bold(lines: list[dict[str, Any]] | None) -> bool:
    for line in lines or []:
        for span in line.get("spans", []) or []:
            font = str(span.get("font") or "").lower()
            flags = int(span.get("flags") or 0)
            if flags & 16:
                return True
            if _font_has_weight_signal(font):
                return True
    return False


def _looks_like_heading_text(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped or stripped.endswith((".", ",", ";", ":")):
        return False
    if _looks_like_algorithm_line(stripped):
        return False
    if _RE_CANONICAL_HEADING.match(stripped):
        return True
    if _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return True
    if _RE_CAPTION.match(stripped):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", stripped)
    if not words:
        return False
    small_words = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "via", "with"}
    content_words = [word for word in words if word.lower() not in small_words]
    if not content_words:
        return False
    capitalized = sum(1 for word in content_words if word[:1].isupper())
    if capitalized / max(len(content_words), 1) >= 0.6:
        return True
    if len(content_words) <= 4 and stripped[:1].isupper():
        return True
    return False


def _block_text(lines: list[dict[str, Any]]) -> str:
    line_texts: list[str] = []
    for line in lines:
        pieces = []
        for span in line.get("spans", []) or []:
            text = str(span.get("text") or "")
            if text:
                pieces.append(text)
        line_text = " ".join("".join(pieces).split())
        if line_text:
            line_texts.append(line_text)
    return _clean_pdf_text("\n".join(line_texts))


def _clean_pdf_text(text: Any) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", value)
    return value.strip()


def _collect_font_sizes(raw_blocks: list[dict[str, Any]]) -> list[float]:
    sizes: list[float] = []
    for block in raw_blocks:
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                try:
                    size = float(span.get("size", 0))
                except (TypeError, ValueError):
                    continue
                if size > 0:
                    sizes.append(size)
    return sizes


def _normalize_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def _merge_bboxes(bboxes: Any) -> list[float] | None:
    valid = [_normalize_bbox(bbox) for bbox in bboxes]
    valid = [bbox for bbox in valid if bbox]
    if not valid:
        return None
    return [
        round(min(b[0] for b in valid), 2),
        round(min(b[1] for b in valid), 2),
        round(max(b[2] for b in valid), 2),
        round(max(b[3] for b in valid), 2),
    ]


def _build_line_anchors(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for line in lines or []:
        spans = line.get("spans", []) if isinstance(line, dict) else []
        text = _block_text([line]) if spans else ""
        bbox = _merge_bboxes(
            span.get("bbox")
            for span in spans
            if isinstance(span, dict)
        )
        if text and bbox:
            anchors.append({"text": _limit_text(text, 800), "bbox": bbox})
    return anchors


def _limit_text(text: Any, max_len: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len].rstrip() + "..."
