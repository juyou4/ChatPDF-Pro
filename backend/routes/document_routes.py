import io
import asyncio
import os
import glob
import hashlib
import json
import logging
import pickle
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import PyPDF2
import pdfplumber
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request

from services.vector_service import create_index
from services.url_loader_service import fetch_url_content
from services.multi_format_loader import is_supported_format, extract_from_file
from services.block_index_service import ensure_block_index, load_block_index, save_block_index
from services.mineru_block_index_service import (
    MINERU_BLOCK_INDEX_SOURCE,
    build_block_index_from_mineru_payload,
    get_mineru_result_path,
    load_mineru_result,
    save_mineru_result,
)
from services.mineru_text_normalizer import (
    MINERU_RAG_INDEX_SOURCE,
    normalize_mineru_for_rag,
    utc_now_iso,
    validate_mineru_rag_data,
)
from services.document_parse_artifact import (
    artifact_reference,
    build_document_parse_artifact,
    derive_table_geometry_capabilities,
    persist_document_parse_artifact,
)
from services.document_parse_state import (
    PARSE_ROUTE_AUTO,
    PARSE_ROUTE_LOCAL,
    PARSE_ROUTE_MINERU,
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_QUEUED,
    PARSE_STATUS_READY,
    PARSE_STATUS_RUNNING,
    build_parse_manifest,
    derive_source_hash,
    is_parse_prepared,
    matches_parse_generation,
    normalize_parse_route,
    read_parse_manifest,
    transition_parse_manifest,
)
from services.document_job_store import (
    load_document_job,
    persist_document_job,
    recover_interrupted_document_job,
)
from services.embedding_service import (
    _build_semantic_group_index_async,
    _build_semantic_group_index,
    _index_cache,
    get_embedding_function,
)
from services.semantic_group_store import (
    deactivate_generation,
    publish_generation,
    semantic_group_paths,
    validate_semantic_group_artifacts,
)
from services.ai_cache_state import load_ai_cache_generation, rotate_ai_cache_generation
from services.block_translation_service import (
    MAX_BLOCKS_PER_REQUEST,
    get_cached_translations,
    get_translation_cache_path,
    save_translation_cache,
    translate_blocks,
)
from services.reading_outline_service import (
    get_or_create_reading_outline,
    get_reading_outline_path,
    save_reading_outline,
)
from services.section_outline_service import (
    get_or_create_section_outline,
    get_section_outline_path,
    save_section_outline,
)
from services.table_visual_metadata import build_table_visual_metadata
from services.table_visual_verifier import get_table_visual_verification_status
from runtime_mode import runtime
from services.ocr_service import (
    is_ocr_available,
    detect_pdf_quality,
    ocr_pdf,
    get_ocr_service,
    _ocr_registry,
    _document_parser_registry,
    _find_poppler,
    _save_online_ocr_config,
    _load_online_ocr_config,
    _mask_api_key,
    apply_ocr_result_to_pages,
    get_ocr_provider_usage,
    record_ocr_provider_use,
    select_ocr_target_pages,
    validate_external_ocr_service_url,
    MistralAdapter,
    MinerUAdapter,
    Doc2XAdapter,
    WorkerOCRAdapter,
    MinerUDirectAdapter,
)
from services.layout_service import (
    configure_yolo_model_path,
    download_yolo_model,
    get_yolo_model_status,
    reset_yolo_model_config,
)
from models.model_detector import normalize_embedding_model_id
from models.model_id_resolver import resolve_model_id
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# 目录策略与 app.py 保持一致：
# - desktop: 使用 runtime.data_dir（由 Electron 传入）
# - server: 使用项目根目录 data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if runtime.is_desktop:
    DATA_DIR = Path(runtime.data_dir)
else:
    DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
UPLOAD_DIR = DATA_DIR / "uploads"

# Legacy paths from the old layout (stored under backend/)
LEGACY_BACKEND_DATA_DIR = BACKEND_ROOT / "data"
LEGACY_BACKEND_DOCS_DIR = LEGACY_BACKEND_DATA_DIR / "docs"
LEGACY_BACKEND_VECTOR_STORE_DIR = LEGACY_BACKEND_DATA_DIR / "vector_stores"
LEGACY_BACKEND_UPLOAD_DIR = BACKEND_ROOT / "uploads"
LEGACY_PROJECT_UPLOAD_DIR = PROJECT_ROOT / "uploads"

documents_store = {}
_INDEX_STATUS_LOCK = threading.Lock()
_DOCUMENT_INDEX_STATUS: dict[str, dict] = {}
_DEEP_PARSE_LOCK = threading.Lock()
_DEEP_PARSE_TASKS: dict[str, dict] = {}
_DEEP_PARSE_CANCEL_EVENTS: dict[str, threading.Event] = {}
_DEEP_PARSE_JOB_TYPE = "mineru_deep_parse"
try:
    _DEEP_PARSE_CONCURRENCY = max(1, min(8, int(os.getenv("CHATPDF_MINERU_DEEP_PARSE_CONCURRENCY", "2"))))
except ValueError:
    _DEEP_PARSE_CONCURRENCY = 2
_DEEP_PARSE_SEMAPHORE = threading.BoundedSemaphore(_DEEP_PARSE_CONCURRENCY)
_DOCUMENT_OPERATION_LOCKS_LOCK = threading.Lock()
_DOCUMENT_OPERATION_LOCKS: dict[str, threading.Lock] = {}
_DOCUMENT_PUBLICATION_LOCKS_LOCK = threading.Lock()
_DOCUMENT_PUBLICATION_LOCKS: dict[str, threading.RLock] = {}


def _get_document_operation_lock(doc_id: str) -> threading.Lock:
    with _DOCUMENT_OPERATION_LOCKS_LOCK:
        return _DOCUMENT_OPERATION_LOCKS.setdefault(doc_id, threading.Lock())


def _get_document_publication_lock(doc_id: str) -> threading.RLock:
    """Serialize short, externally visible parse-generation publications.

    MinerU parsing and embedding construction can take minutes, so uploads
    must not wait for the whole operation lock.  This separate re-entrant lock
    only covers the final artifact/document swap and the upload's manifest
    replacement, making the generation check and publication atomic together.
    """
    with _DOCUMENT_PUBLICATION_LOCKS_LOCK:
        return _DOCUMENT_PUBLICATION_LOCKS.setdefault(doc_id, threading.RLock())


def _normalize_page_keys(data: dict):
    """Ensure every page has both 'text' and 'content' keys for compatibility."""
    for page in data.get("data", {}).get("pages", []):
        if "text" not in page and "content" in page:
            page["text"] = page["content"]
        elif "content" not in page and "text" in page:
            page["content"] = page["text"]


def _clean_control_text(text: str) -> str:
    if not text:
        return ""
    return ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\t\n\r')


def _clean_nested_text(value):
    if isinstance(value, str):
        return _clean_control_text(value)
    if isinstance(value, dict):
        return {key: _clean_nested_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_nested_text(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clean_nested_text(item) for item in value)
    return value


def _clean_structured_table_bundle(bundle: dict) -> dict:
    if not isinstance(bundle, dict):
        return {}
    cleaned_bundle = dict(bundle)
    for key in (
        "bundle_text",
        "table_caption",
        "table_header",
        "table_body_markdown",
        "table_markdown",
        "html_table",
        "table_footnote",
        "table_id",
    ):
        if key in cleaned_bundle:
            cleaned_bundle[key] = _clean_control_text(str(cleaned_bundle.get(key) or ""))
    if "evidence_units" in cleaned_bundle:
        cleaned_bundle["evidence_units"] = _clean_nested_text(cleaned_bundle.get("evidence_units"))
    # 视觉核验身份只附加到清洗后的副本，不能修改 MinerU/ODL 原始 bundle。
    cleaned_bundle.update(build_table_visual_metadata(cleaned_bundle))
    return cleaned_bundle


def _clean_structured_table_bundles(bundles: list[dict]) -> list[dict]:
    cleaned_bundles = []
    for bundle in bundles or []:
        if not isinstance(bundle, dict):
            continue
        cleaned_bundles.append(_clean_structured_table_bundle(bundle))
    return cleaned_bundles


def _backfill_bundle_evidence_units_from_pages(
    bundles: list[dict],
    pages: list[dict],
) -> list[dict]:
    """当顶层 structured_table_bundles 缺少 evidence_units 时，从 page.table_bundles 回填。"""
    evidence_units_by_key: dict[tuple[str, str], list] = {}

    for page in pages or []:
        if not isinstance(page, dict):
            continue
        for bundle in page.get("table_bundles", []) or []:
            if not isinstance(bundle, dict):
                continue
            evidence_units = bundle.get("evidence_units")
            if not evidence_units:
                continue

            bundle_id = str(bundle.get("bundle_id") or "").strip()
            table_id = _clean_control_text(str(bundle.get("table_id") or "")).strip()
            if bundle_id:
                evidence_units_by_key.setdefault(("bundle_id", bundle_id), evidence_units)
            if table_id:
                evidence_units_by_key.setdefault(("table_id", table_id), evidence_units)

    hydrated_bundles = []
    for bundle in bundles or []:
        if not isinstance(bundle, dict):
            continue
        hydrated_bundle = dict(bundle)
        if not hydrated_bundle.get("evidence_units"):
            bundle_id = str(hydrated_bundle.get("bundle_id") or "").strip()
            table_id = _clean_control_text(str(hydrated_bundle.get("table_id") or "")).strip()
            if bundle_id and ("bundle_id", bundle_id) in evidence_units_by_key:
                hydrated_bundle["evidence_units"] = evidence_units_by_key[("bundle_id", bundle_id)]
            elif table_id and ("table_id", table_id) in evidence_units_by_key:
                hydrated_bundle["evidence_units"] = evidence_units_by_key[("table_id", table_id)]
        # evidence 回填会改变表格源内容，重新派生缓存失效所需的身份元数据。
        hydrated_bundle.update(build_table_visual_metadata(hydrated_bundle))
        hydrated_bundles.append(hydrated_bundle)
    return hydrated_bundles


def _clean_page_texts(pages: list[dict]) -> list[dict]:
    cleaned_pages = []
    for page in pages or []:
        if not isinstance(page, dict):
            continue
        cleaned_page = dict(page)
        raw_text = cleaned_page.get("text", cleaned_page.get("content", ""))
        cleaned_text = _clean_control_text(raw_text)
        cleaned_page["text"] = cleaned_text
        cleaned_page["content"] = cleaned_text
        if isinstance(cleaned_page.get("table_bundles"), list):
            cleaned_page["table_bundles"] = _clean_structured_table_bundles(cleaned_page.get("table_bundles", []))
        cleaned_pages.append(cleaned_page)
    return cleaned_pages


def _merge_odl_pages_with_existing_ocr(
    existing_pages: list[dict],
    odl_pages: list[dict],
) -> tuple[list[dict], bool]:
    """Use ODL for native pages without discarding successful page OCR."""
    existing_by_page = {
        int(page.get("page") or index + 1): page
        for index, page in enumerate(existing_pages or [])
        if isinstance(page, dict)
    }
    merged_pages: list[dict] = []
    preserved_ocr = False
    seen_pages: set[int] = set()
    for index, odl_page in enumerate(odl_pages or []):
        if not isinstance(odl_page, dict):
            continue
        page_num = int(odl_page.get("page") or index + 1)
        seen_pages.add(page_num)
        existing = existing_by_page.get(page_num)
        if existing and existing.get("source") == "ocr":
            merged = dict(odl_page)
            merged["text"] = existing.get("text", existing.get("content", ""))
            merged["content"] = existing.get("content", existing.get("text", ""))
            merged["source"] = "ocr"
            if existing.get("ocr_backend"):
                merged["ocr_backend"] = existing["ocr_backend"]
            merged_pages.append(merged)
            preserved_ocr = True
        else:
            merged_pages.append(dict(odl_page))
    for page_num, existing in sorted(existing_by_page.items()):
        if page_num not in seen_pages:
            merged_pages.append(dict(existing))
    merged_pages.sort(key=lambda page: int(page.get("page") or 0))
    return merged_pages, preserved_ocr


def save_document(doc_id: str, data: dict):
    try:
        file_path = DOCS_DIR / f"{doc_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            import json
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("Saved document %s to %s", doc_id, file_path)
    except Exception as e:
        logger.warning("Error saving document %s: %s", doc_id, e)


def load_documents():
    logger.info("Loading documents from disk...")
    count = 0
    for file_path in glob.glob(str(DOCS_DIR / "*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
                doc_id = os.path.splitext(os.path.basename(file_path))[0]
                _normalize_page_keys(data)
                documents_store[doc_id] = data
                count += 1
        except Exception as e:
            logger.warning("Error loading document from %s: %s", file_path, e)
    logger.info("Loaded %s documents.", count)


def _resolve_document_pdf_path(doc: dict) -> Path | None:
    """Resolve a stored document's PDF file path when it has one."""
    pdf_url = (doc or {}).get("pdf_url") or ""
    if not pdf_url:
        return None
    pdf_name = pdf_url.split("/")[-1]
    if not pdf_name:
        return None
    pdf_path = UPLOAD_DIR / pdf_name
    return pdf_path if pdf_path.exists() else None


def _read_document_parse_manifest(doc_id: str, doc: dict | None = None) -> dict:
    """Return the primary parser contract, including a safe legacy fallback."""
    target = doc if isinstance(doc, dict) else documents_store.get(doc_id)
    return read_parse_manifest(target or {}, doc_id=doc_id)


class _SupersededParseGeneration(RuntimeError):
    """Raised when a background publisher no longer owns a document run."""


class _SupersededAICacheGeneration(RuntimeError):
    """Raised when a cache publisher predates an explicit cache clear."""


def _parse_generation_matches_current_document(
    doc_id: str,
    *,
    parse_generation: str,
    document_source_hash: str = "",
) -> bool:
    """Return whether a worker still owns the current document generation."""
    if not parse_generation:
        return False
    manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    return matches_parse_generation(
        manifest,
        generation=parse_generation,
        source_hash=document_source_hash or None,
    )


def _require_current_parse_generation(
    doc_id: str,
    *,
    parse_generation: str,
    document_source_hash: str = "",
) -> dict:
    """Return the active manifest or refuse an obsolete publication."""
    manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    if not parse_generation or not matches_parse_generation(
        manifest,
        generation=parse_generation,
        source_hash=document_source_hash or None,
    ):
        raise _SupersededParseGeneration("文档解析路线已更新，已拒绝发布旧代际结果")
    return manifest


def _build_parse_bound_cache_writer(
    doc_id: str,
    *,
    parse_generation: str,
    document_source_hash: str,
    persist,
    ai_cache_generation: str | None = None,
):
    """创建“身份复核 + 缓存落盘”不可分割的短事务。"""
    expected_ai_cache_generation = (
        load_ai_cache_generation(DATA_DIR, doc_id)
        if ai_cache_generation is None
        else str(ai_cache_generation)
    )

    def write(payload):
        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=document_source_hash,
            )
            current_ai_cache_generation = load_ai_cache_generation(DATA_DIR, doc_id)
            if current_ai_cache_generation != expected_ai_cache_generation:
                raise _SupersededAICacheGeneration(
                    "AI 缓存已清理，已拒绝发布清理前的旧结果"
                )
            return persist(payload)

    return write


def _write_document_parse_manifest(
    doc_id: str,
    manifest: dict,
    *,
    doc: dict | None = None,
    persist: bool = True,
) -> dict:
    """Persist one canonical parser state without replacing unrelated document data."""
    target = doc if isinstance(doc, dict) else documents_store.get(doc_id)
    if not isinstance(target, dict):
        raise RuntimeError("文档记录不存在，无法更新解析状态")
    data = target.setdefault("data", {})
    data["parse_manifest"] = dict(manifest)
    if persist:
        save_document(doc_id, target)
    return data["parse_manifest"]


def _transition_document_parse_manifest(
    doc_id: str,
    status: str,
    *,
    stage: str | None = None,
    error: str | None = None,
    doc: dict | None = None,
    metadata: dict | None = None,
    persist: bool = True,
) -> dict:
    target = doc if isinstance(doc, dict) else documents_store.get(doc_id)
    current = _read_document_parse_manifest(doc_id, target)
    updated = transition_parse_manifest(current, status, stage=stage, error=error)
    if metadata:
        merged_metadata = dict(updated.get("metadata") or {})
        merged_metadata.update(metadata)
        updated["metadata"] = merged_metadata
    return _write_document_parse_manifest(doc_id, updated, doc=target, persist=persist)


def _transition_current_full_mineru_manifest(
    doc_id: str,
    status: str,
    *,
    parse_generation: str,
    document_source_hash: str,
    stage: str,
    error: str | None = None,
    expected_statuses: set[str] | None = None,
) -> dict | None:
    """Transition only the active full-route MinerU generation under its publication lock."""
    if not parse_generation or not document_source_hash:
        return None
    with _get_document_publication_lock(doc_id):
        doc = documents_store.get(doc_id)
        manifest = _read_document_parse_manifest(doc_id, doc)
        if not matches_parse_generation(
            manifest,
            generation=parse_generation,
            source_hash=document_source_hash,
        ):
            return None
        if not _is_full_mineru_parse_manifest(manifest):
            return None
        if expected_statuses is not None and manifest.get("status") not in expected_statuses:
            return None
        return _transition_document_parse_manifest(
            doc_id,
            status,
            stage=stage,
            error=error,
            doc=doc,
        )


def _document_parse_gate_message(manifest: dict) -> str:
    route = str(
        manifest.get("resolved_route")
        or manifest.get("requested_route")
        or manifest.get("route")
        or "auto"
    )
    stage = str(manifest.get("stage") or "")
    if route == PARSE_ROUTE_MINERU and (manifest.get("metadata") or {}).get("full_route"):
        if stage == "awaiting_rag_index":
            return "MinerU 已完成版面解析，正在等待问答索引发布；请继续完成 MinerU 索引重建"
        return "当前文档正在按 MinerU 全程解析，完成前不能使用阅读、速览、翻译或问答"
    return "当前文档解析尚未完成，请稍后重试"


def _require_document_parse_ready(doc_id: str, doc: dict | None = None) -> dict:
    manifest = _read_document_parse_manifest(doc_id, doc)
    if not is_parse_prepared(manifest):
        raise HTTPException(status_code=409, detail=_document_parse_gate_message(manifest))
    return manifest


def _parse_manifest_index_matches(doc_id: str, manifest: dict) -> bool:
    """Whether the active vector pair belongs to the document's active parse run."""
    if not _vector_index_ready(doc_id):
        return False
    metadata = _read_vector_index_meta(doc_id)
    if bool((manifest.get("metadata") or {}).get("legacy_inferred")):
        return True
    index_meta = metadata.get("index_meta") or {}
    return (
        str(index_meta.get("parse_generation") or "") == str(manifest.get("generation") or "")
        and str(index_meta.get("document_source_hash") or "") == str(manifest.get("source_hash") or "")
    )


def _warm_block_index(doc_id: str) -> None:
    """Best-effort block index build; upload/search must not fail because of it."""
    try:
        doc = documents_store.get(doc_id)
        if not doc:
            return
        ensure_block_index(
            doc_id=doc_id,
            doc=doc,
            data_dir=DATA_DIR,
            pdf_path=_resolve_document_pdf_path(doc),
        )
    except Exception as exc:
        logger.warning("[BlockIndex] warm build failed for %s: %s", doc_id, exc)


def _vector_index_ready(doc_id: str) -> bool:
    return (VECTOR_STORE_DIR / f"{doc_id}.index").exists() and (VECTOR_STORE_DIR / f"{doc_id}.pkl").exists()


def _vector_index_paths(doc_id: str, base_dir: Path | None = None) -> tuple[Path, Path]:
    root = base_dir or VECTOR_STORE_DIR
    return root / f"{doc_id}.index", root / f"{doc_id}.pkl"


def _semantic_group_paths(doc_id: str) -> dict[str, Path]:
    return semantic_group_paths(DATA_DIR / "semantic_groups", doc_id)


def _semantic_group_backup_path(doc_id: str, source: str, kind: str) -> Path:
    root = DATA_DIR / "semantic_groups"
    safe_source = _safe_index_source_name(source)
    suffix = {
        "json": "semantic.json",
        "index": "semantic.index",
        "pkl": "semantic.pkl",
    }.get(kind, f"semantic.{kind}")
    return root / f"{doc_id}.{safe_source}.bak.{suffix}"


def _read_vector_index_meta(doc_id: str, base_dir: Path | None = None) -> dict:
    _index_path, chunks_path = _vector_index_paths(doc_id, base_dir)
    if not chunks_path.exists():
        return {}
    try:
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        logger.warning("[RagIndex] failed to read vector pkl meta for %s: %s", doc_id, exc)
        return {}
    if not isinstance(data, dict):
        return {"index_source": "pdf_native"}
    index_meta = data.get("index_meta") if isinstance(data.get("index_meta"), dict) else {}
    return {
        "index_source": data.get("index_source") or "pdf_native",
        "source_hash": data.get("source_hash") or "",
        "rebuilt_at": data.get("rebuilt_at") or "",
        "previous_index_source": data.get("previous_index_source") or "",
        "normalizer_version": data.get("normalizer_version") or "",
        "parse_generation": index_meta.get("parse_generation") or "",
        "document_source_hash": index_meta.get("document_source_hash") or "",
        "index_meta": index_meta,
        "chunk_count": len(data.get("chunks") or []),
        "table_chunk_count": sum(
            1
            for item in (data.get("chunk_metadata") or [])
            if isinstance(item, dict) and item.get("structured_table_bundle")
        ),
    }


def _get_rag_index_status(doc_id: str) -> dict:
    ready = _vector_index_ready(doc_id)
    meta = _read_vector_index_meta(doc_id) if ready else {}
    source = meta.get("index_source") or ("pdf_native" if ready else "")
    parse_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    matches_active_parse = _parse_manifest_index_matches(doc_id, parse_manifest) if ready else False
    with _INDEX_STATUS_LOCK:
        lifecycle = dict(_DOCUMENT_INDEX_STATUS.get(doc_id) or {})
    lifecycle_matches_active_parse = bool(lifecycle) and matches_parse_generation(
        parse_manifest,
        generation=str(lifecycle.get("parse_generation") or ""),
        source_hash=str(lifecycle.get("document_source_hash") or ""),
    )
    if not lifecycle_matches_active_parse:
        lifecycle = {}
    if not lifecycle:
        if ready and matches_active_parse:
            lifecycle = {"status": "ready", "stage": "ready", "error": ""}
        elif ready:
            lifecycle = {
                "status": "stale",
                "stage": "parse_generation_mismatch",
                "error": "现有问答索引不属于当前解析代际",
            }
        else:
            lifecycle = {"status": "missing", "stage": "not_started", "error": ""}
    return {
        "status": str(lifecycle.get("status") or "missing"),
        "stage": str(lifecycle.get("stage") or "not_started"),
        "error": str(lifecycle.get("error") or ""),
        "ready": ready and matches_active_parse,
        "artifact_ready": ready,
        "index_source": source,
        "source_hash": meta.get("source_hash", ""),
        "rebuilt_at": meta.get("rebuilt_at", ""),
        "previous_index_source": meta.get("previous_index_source", ""),
        "normalizer_version": meta.get("normalizer_version", ""),
        "parse_generation": meta.get("parse_generation", ""),
        "document_source_hash": meta.get("document_source_hash", ""),
        "matches_active_parse": matches_active_parse,
        "chunk_count": meta.get("chunk_count", 0),
        "table_chunk_count": meta.get("table_chunk_count", 0),
        "can_rollback": bool(_load_complete_rag_backup_manifest(doc_id, "pdf_native")),
    }


def _set_document_index_status(
    doc_id: str,
    status: str,
    *,
    stage: str = "",
    error: str = "",
    parse_generation: str | None = None,
    document_source_hash: str | None = None,
) -> None:
    manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    with _INDEX_STATUS_LOCK:
        _DOCUMENT_INDEX_STATUS[doc_id] = {
            "doc_id": doc_id,
            "status": status,
            "stage": stage,
            "error": error,
            "parse_generation": str(parse_generation or manifest.get("generation") or ""),
            "document_source_hash": str(document_source_hash or manifest.get("source_hash") or ""),
            "updated_at": datetime.now().isoformat(),
        }


def _get_document_index_status(doc_id: str) -> dict:
    with _INDEX_STATUS_LOCK:
        current = dict(_DOCUMENT_INDEX_STATUS.get(doc_id) or {})
    parse_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    current_matches_parse = bool(current) and matches_parse_generation(
        parse_manifest,
        generation=str(current.get("parse_generation") or ""),
        source_hash=str(current.get("document_source_hash") or ""),
    )
    if current and not current_matches_parse:
        current = {}
    vector_matches_parse = _parse_manifest_index_matches(doc_id, parse_manifest)
    if _vector_index_ready(doc_id) and vector_matches_parse:
        current.update({
            "doc_id": doc_id,
            "status": "ready",
            "stage": "ready",
            "error": "",
        })
    elif _vector_index_ready(doc_id) and not vector_matches_parse:
        if current.get("status") not in {"queued", "running"}:
            current = {
                "doc_id": doc_id,
                "status": "stale",
                "stage": "parse_generation_mismatch",
                "error": "现有问答索引不属于当前解析代际",
            }
    elif not current:
        current = {
            "doc_id": doc_id,
            "status": "missing",
            "stage": "not_started",
            "error": "",
        }
    current["vector_ready"] = _vector_index_ready(doc_id) and vector_matches_parse
    current["vector_artifact_ready"] = _vector_index_ready(doc_id)
    current["parse_manifest"] = parse_manifest
    current["rag_index"] = _get_rag_index_status(doc_id)
    return current


def _build_document_indexes(
    doc_id: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_api_host: Optional[str],
    summary_api_key: Optional[str],
) -> None:
    document_lock = _get_document_operation_lock(doc_id)
    document_lock.acquire()
    parse_manifest: dict = {}
    try:
        doc = documents_store.get(doc_id)
        if not doc:
            raise RuntimeError("文档记录不存在，无法构建索引")

        parse_manifest = _read_document_parse_manifest(doc_id, doc)
        parse_generation = str(parse_manifest.get("generation") or "")
        parse_source_hash = str(parse_manifest.get("source_hash") or "")
        if not is_parse_prepared(parse_manifest):
            _set_document_index_status(
                doc_id,
                "queued",
                stage="waiting_for_primary_parser",
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )
            return

        _set_document_index_status(
            doc_id,
            "running",
            stage="block_index",
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        )
        _warm_block_index(doc_id)

        if _parse_manifest_index_matches(doc_id, parse_manifest):
            _set_document_index_status(
                doc_id,
                "ready",
                stage="ready",
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )
            return

        data = doc.get("data") or {}
        _set_document_index_status(
            doc_id,
            "running",
            stage="vector_index",
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        )
        create_index(
            doc_id,
            data.get("full_text", ""),
            str(VECTOR_STORE_DIR),
            embedding_model,
            embedding_api_key,
            embedding_api_host,
            pages=data.get("pages"),
            structured_table_bundles=data.get("structured_table_bundles"),
            summary_api_key=summary_api_key,
            index_source=(
                MINERU_RAG_INDEX_SOURCE
                if parse_manifest.get("resolved_route") == PARSE_ROUTE_MINERU
                else "pdf_native"
            ),
            index_meta={
                "source_hash": data.get("rag_source_hash") or parse_manifest.get("source_hash", ""),
                "document_source_hash": parse_manifest.get("source_hash", ""),
                "parse_generation": parse_manifest.get("generation", ""),
                "parser_route": parse_manifest.get("resolved_route", ""),
            },
            build_semantic_groups=False,
        )
        # Do not let the embedding service publish a detached background
        # semantic generation after this document switches to MinerU. Build and
        # publish it while the same document lock is held instead.
        semantic_stage = DATA_DIR / "semantic_groups" / "_tmp" / f"{doc_id}.local.{uuid.uuid4().hex}"
        try:
            semantic_stage.mkdir(parents=True, exist_ok=True)
            semantic_rebuild = _prepare_semantic_group_rebuild(
                doc_id,
                VECTOR_STORE_DIR,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
                embedding_api_host=embedding_api_host,
                summary_api_key=summary_api_key,
            )
            semantic_result = _build_semantic_group_index(
                doc_id,
                semantic_rebuild["chunks"],
                data.get("pages") or [],
                semantic_rebuild["embed_fn"],
                semantic_rebuild["api_key"],
                model=semantic_rebuild["model"],
                provider=semantic_rebuild["provider"],
                endpoint=semantic_rebuild["endpoint"],
                output_dir=str(semantic_stage),
                raise_on_error=True,
            )
            semantic_validation = _validate_temp_semantic_groups(doc_id, semantic_stage, semantic_result)
            with _get_document_publication_lock(doc_id):
                _require_current_parse_generation(
                    doc_id,
                    parse_generation=parse_generation,
                    document_source_hash=parse_source_hash,
                )
                _publish_temp_semantic_groups(
                    doc_id,
                    semantic_stage,
                    semantic_validation,
                    source_hash=parse_source_hash,
                    transaction_id=parse_generation,
                )
        except Exception as semantic_exc:
            # Semantic groups enhance retrieval but must never leave an older
            # generation active when the current parser route has changed.
            logger.warning("[Upload] semantic groups unavailable for %s: %s", doc_id, semantic_exc)
        finally:
            shutil.rmtree(semantic_stage, ignore_errors=True)
        current_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
        if not matches_parse_generation(
            current_manifest,
            generation=str(parse_manifest.get("generation") or ""),
            source_hash=str(parse_manifest.get("source_hash") or ""),
        ):
            raise RuntimeError("索引构建期间文档解析路线已切换，已拒绝发布旧代际状态")
        _set_document_index_status(
            doc_id,
            "ready",
            stage="ready",
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        )
        logger.info("[Upload] background index ready for %s", doc_id)
    except Exception as exc:
        logger.exception("[Upload] background index failed for %s: %s", doc_id, exc)
        _set_document_index_status(
            doc_id,
            "failed",
            stage="failed",
            error=str(exc),
            parse_generation=str(parse_manifest.get("generation") or ""),
            document_source_hash=str(parse_manifest.get("source_hash") or ""),
        )
    finally:
        document_lock.release()


def _queue_document_indexes(
    doc_id: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_api_host: Optional[str],
    summary_api_key: Optional[str],
) -> dict:
    parse_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    if not is_parse_prepared(parse_manifest):
        _set_document_index_status(doc_id, "queued", stage="waiting_for_primary_parser")
        return _get_document_index_status(doc_id)
    current = _get_document_index_status(doc_id)
    if current.get("status") in {"queued", "running", "ready"}:
        return current

    _set_document_index_status(doc_id, "queued", stage="queued")
    thread = threading.Thread(
        target=_build_document_indexes,
        args=(doc_id, embedding_model, embedding_api_key, embedding_api_host, summary_api_key),
        name=f"chatpdf-index-{doc_id[:8]}",
        daemon=True,
    )
    thread.start()
    return _get_document_index_status(doc_id)


def _mineru_configured() -> bool:
    config = _load_online_ocr_config("mineru")
    access_mode = str(config.get("access_mode") or "worker").strip().lower()
    worker_url = str(config.get("worker_url") or "").strip()
    token_mode = str(config.get("token_mode") or "frontend").strip().lower()
    token = str(config.get("token") or "").strip()
    if access_mode == "direct":
        return bool(token)
    return bool(worker_url and (token_mode == "worker" or token))


def _validate_mineru_access(config: dict) -> tuple[bool, str]:
    """Fast preflight check before uploading a PDF to MinerU."""
    import httpx

    access_mode = str(config.get("access_mode") or "worker").strip().lower()
    token = str(config.get("token") or "").strip()
    if access_mode == "direct":
        if not token:
            return False, "直连模式下必须提供 MinerU Token"
        base_url = str(config.get("base_url") or "https://mineru.net/api/v4").strip()
        try:
            base_url = validate_external_ocr_service_url(
                base_url,
                service_name="MinerU API Base URL",
            ).rstrip("/")
        except ValueError as exc:
            return False, str(exc)
        try:
            with httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0)) as client:
                response = client.post(
                    f"{base_url}/file-urls/batch",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "enable_formula": False,
                        "enable_table": False,
                        "language": "ch",
                        "files": [{
                            "name": "health_check.pdf",
                            "is_ocr": True,
                        }],
                    },
                )
            if response.status_code in (401, 403):
                return False, "MinerU Token 无效或已过期"
            if not response.is_success:
                return False, f"MinerU API 不可达 (HTTP {response.status_code})"
            data = response.json()
            if data.get("code") != 0:
                return False, f"MinerU Token 验证失败: {data.get('msg') or data}"
            return True, "MinerU 直连 API 可达且 Token 有效"
        except httpx.TimeoutException:
            return False, "连接超时，请检查网络或 Base URL"
        except httpx.ConnectError:
            return False, "连接失败，请检查网络或 Base URL"
        except httpx.RequestError:
            return False, "网络连接失败，请检查网络设置"
        except Exception as exc:
            return False, f"MinerU 直连验证失败: {exc}"

    worker_url = str(config.get("worker_url") or "").strip()
    auth_key = str(config.get("auth_key") or "").strip()
    token_mode = str(config.get("token_mode") or "frontend").strip().lower()
    if not worker_url:
        return False, "Worker 代理模式下必须提供 Worker URL"
    try:
        worker_url = validate_external_ocr_service_url(
            worker_url,
            service_name="MinerU Worker",
        )
    except ValueError as exc:
        return False, str(exc)

    try:
        with httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0)) as client:
            headers = {}
            if auth_key:
                headers["X-Auth-Key"] = auth_key
            health_resp = client.get(f"{worker_url}/health", headers=headers)
            if health_resp.status_code in (401, 403):
                return False, "Auth Key 无效，请检查 Auth Key 是否正确"
            if health_resp.status_code == 404:
                return False, "MinerU Worker 路由不存在，请检查是否部署了 pb-ocr-proxy 的 /health 和 /mineru/upload 路由"
            if not health_resp.is_success:
                return False, f"Worker 不可达 (HTTP {health_resp.status_code})"

            if token_mode == "frontend":
                if not token:
                    return False, "前端透传模式下必须提供 MinerU Token"
                token_headers = dict(headers)
                token_headers["X-MinerU-Key"] = token
                token_resp = client.get(f"{worker_url}/mineru/result/__health__", headers=token_headers)
                if token_resp.status_code in (401, 403):
                    return False, "MinerU Token 无效或缺失，请检查 Token 是否正确"
                if token_resp.status_code == 404:
                    return False, "MinerU Worker 缺少 /mineru/result/__health__ 路由，请重新部署 pb-ocr-proxy"
                if not token_resp.is_success:
                    return False, f"MinerU Token 验证失败 (HTTP {token_resp.status_code})"
            return True, "MinerU Worker 可达"
    except httpx.TimeoutException:
        return False, "连接超时，请检查 Worker URL 是否正确"
    except httpx.ConnectError:
        return False, "连接失败，请检查 Worker URL 是否正确"
    except httpx.RequestError:
        return False, "网络连接失败，请检查网络设置"
    except Exception as exc:
        return False, f"MinerU Worker 验证失败: {exc}"


def _clear_block_dependent_ai_cache(doc_id: str, doc: dict | None = None) -> list[str]:
    """Delete AI artifacts that bind to old block ids after deep parsing."""
    removed: list[str] = []
    cache_paths = {
        "reading_outline": get_reading_outline_path(DATA_DIR, doc_id),
        "section_outline": get_section_outline_path(DATA_DIR, doc_id),
        "block_translations": get_translation_cache_path(DATA_DIR, doc_id),
    }
    for name, path in cache_paths.items():
        try:
            if path.exists():
                path.unlink()
                removed.append(name)
        except Exception as exc:
            logger.warning("[DeepParse] 删除 %s 缓存失败 doc=%s path=%s err=%s", name, doc_id, path, exc)

    # 速览图表解读缓存是内存态（doc["data"]["logical_figures*"]），不是文件，
    # 需要单独失效，否则深度解析完成后速览仍会命中旧的 pdf_native/caption_only 结果。
    target_doc = doc if doc is not None else documents_store.get(doc_id)
    if isinstance(target_doc, dict):
        doc_data = target_doc.get("data")
        if isinstance(doc_data, dict) and doc_data.pop("logical_figures_status", None) is not None:
            doc_data.pop("logical_figures_meta", None)
            doc_data.pop("logical_figures", None)
            removed.append("logical_figures")

    # GraphRAG persists under doc_id, while a re-upload of the same PDF keeps
    # that id but starts a different parse generation. Remove both its in-memory
    # registry and on-disk graph before a new route can be published.
    try:
        from services.graphrag import INSTANCES as graphrag_instances, BUILD_PROGRESS as graphrag_progress

        graphrag_instances.pop(doc_id, None)
        graphrag_progress.pop(doc_id, None)
        graph_dir = Path(settings.graphrag_working_dir) / doc_id
        if graph_dir.exists():
            shutil.rmtree(graph_dir, ignore_errors=True)
            removed.append("graphrag")
    except Exception as exc:
        logger.warning("[ParseRoute] 删除 GraphRAG 缓存失败 doc=%s err=%s", doc_id, exc)

    return removed


def _set_deep_parse_status(doc_id: str, status: str, *, stage: str = "", error: str = "", **extra) -> None:
    with _DEEP_PARSE_LOCK:
        current = dict(_DEEP_PARSE_TASKS.get(doc_id) or {})
        current.update({
            "doc_id": doc_id,
            "provider": "mineru",
            "job_type": _DEEP_PARSE_JOB_TYPE,
            "status": status,
            "stage": stage,
            "error": error,
            "created_at": current.get("created_at") or datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **extra,
        })
        _DEEP_PARSE_TASKS[doc_id] = current
    persist_document_job(DATA_DIR, _DEEP_PARSE_JOB_TYPE, doc_id, current)


def _retire_superseded_mineru_job(doc_id: str) -> None:
    """Stop a same-PDF MinerU job before replacing its parse manifest.

    The cancellation event prevents any remaining remote polling from doing
    more local work. The generation fence at publication time is still the
    authoritative protection, because a remote response can race with this
    best-effort cancellation.
    """
    with _DEEP_PARSE_LOCK:
        current = dict(_DEEP_PARSE_TASKS.get(doc_id) or {})
        cancel_event = _DEEP_PARSE_CANCEL_EVENTS.get(doc_id)
        if cancel_event:
            cancel_event.set()

    if current.get("status") in {"queued", "running"}:
        _set_deep_parse_status(
            doc_id,
            "cancelled",
            stage="superseded",
            error="",
            message="同一 PDF 已重新上传，旧 MinerU 解析任务已失效",
            superseded=True,
        )


def _get_deep_parse_status(doc_id: str) -> dict:
    with _DEEP_PARSE_LOCK:
        current = dict(_DEEP_PARSE_TASKS.get(doc_id) or {})
    if not current:
        persisted = load_document_job(DATA_DIR, _DEEP_PARSE_JOB_TYPE, doc_id)
        if persisted:
            current = recover_interrupted_document_job(
                DATA_DIR,
                _DEEP_PARSE_JOB_TYPE,
                doc_id,
                persisted,
                updated_at=datetime.now().isoformat(),
            )
            with _DEEP_PARSE_LOCK:
                _DEEP_PARSE_TASKS[doc_id] = dict(current)

    parse_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    legacy_parse = _is_legacy_parse_manifest(parse_manifest)
    if current and not legacy_parse:
        # Persisted job records are keyed by doc_id, while same-PDF uploads
        # intentionally keep that id. Do not surface an old MinerU job as the
        # state of the newly selected primary route.
        if not matches_parse_generation(
            parse_manifest,
            generation=str(current.get("parse_generation") or ""),
            source_hash=str(current.get("document_source_hash") or "") or None,
        ):
            current = {}

    mineru_config = _load_online_ocr_config("mineru")
    access_mode = str(mineru_config.get("access_mode") or "worker").strip().lower()
    if legacy_parse:
        mineru_result_exists = get_mineru_result_path(DATA_DIR, doc_id).exists()
    else:
        mineru_result_exists = bool(
            load_mineru_result(
                DATA_DIR,
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
                require_identity=True,
            )
        )
    block_index = load_block_index(DATA_DIR, doc_id)
    block_matches_parse = legacy_parse or (
        isinstance(block_index, dict)
        and str(block_index.get("parse_generation") or "") == str(parse_manifest.get("generation") or "")
        and str(block_index.get("document_source_hash") or "") == str(parse_manifest.get("source_hash") or "")
    )
    if not block_matches_parse:
        block_index = None
    active_source = block_index.get("source") if isinstance(block_index, dict) else ""
    active_mineru = active_source == MINERU_BLOCK_INDEX_SOURCE
    parse_ready = is_parse_prepared(parse_manifest)
    is_full_mineru_route = _is_full_mineru_parse_manifest(parse_manifest)

    if not current:
        waiting_for_full_route = active_mineru and is_full_mineru_route and not parse_ready
        status = "running" if waiting_for_full_route else ("ready" if active_mineru else "idle")
        current = {
            "doc_id": doc_id,
            "provider": "mineru",
            "status": status,
            "stage": str(parse_manifest.get("stage") or "building_rag_index") if waiting_for_full_route else ("ready" if active_mineru else "not_started"),
            "error": "",
        }

    current.update({
        "configured": _mineru_configured(),
        "access_mode": access_mode,
        "mineru_result_exists": mineru_result_exists,
        "active_source": active_source,
        "active_mineru": active_mineru,
    })
    rag_index = _get_rag_index_status(doc_id)
    current["rag_index"] = rag_index
    current["parse_manifest"] = parse_manifest
    current["parse_ready"] = parse_ready
    if (
        active_mineru
        and current.get("status") not in {"queued", "running", "failed"}
        and not (is_full_mineru_route and not parse_ready)
    ):
        current["status"] = "ready"
        current["stage"] = "ready"

    current.update(_assess_deep_parse_recommendation(doc_id, active_mineru, block_index, rag_index=rag_index))
    return current


def _assess_deep_parse_recommendation(
    doc_id: str,
    active_mineru: bool,
    block_index: dict | None,
    rag_index: dict | None = None,
) -> dict:
    """基于本地解析质量给出"是否建议深度解析"的信号，不做自动上传。

    质量门只读已有信号（上传阶段算好的 extraction_quality、本地大纲候选数），
    不重新跑一遍质量评估，成本几乎为零。已经在用 MinerU 结果时不再建议。
    """
    doc = documents_store.get(doc_id)
    parse_manifest = _read_document_parse_manifest(doc_id, doc)
    if (
        not _is_legacy_parse_manifest(parse_manifest)
        and parse_manifest.get("resolved_route") == PARSE_ROUTE_LOCAL
    ):
        return {
            "recommend_deep_parse": False,
            "recommend_reason": "",
            "recommend_rag_index_rebuild": False,
            "recommend_rag_index_reason": "",
            "parse_route_locked": True,
        }

    if active_mineru:
        rag_index = rag_index if isinstance(rag_index, dict) else _get_rag_index_status(doc_id)
        rag_source = str(rag_index.get("index_source") or ("pdf_native" if rag_index.get("ready") else "")).strip()
        recommend_rag_rebuild = bool(rag_index.get("ready") and rag_source != MINERU_RAG_INDEX_SOURCE)
        return {
            "recommend_deep_parse": False,
            "recommend_reason": "",
            "recommend_rag_index_rebuild": recommend_rag_rebuild,
            "recommend_rag_index_reason": (
                "MinerU 深度解析已完成，但问答索引仍使用本地 PDF 解析；建议重建问答索引以启用结构化表格证据"
                if recommend_rag_rebuild else ""
            ),
        }

    doc_data = doc.get("data") if isinstance(doc, dict) else None
    extraction_quality = str((doc_data or {}).get("extraction_quality") or "unknown")
    total_pages = int((doc_data or {}).get("total_pages") or 0)

    if extraction_quality == "poor":
        return {
            "recommend_deep_parse": True,
            "recommend_reason": "本地文本提取质量较差（可能是扫描件或版式复杂），建议用 MinerU 深度解析改善阅读体验",
            "recommend_rag_index_rebuild": False,
            "recommend_rag_index_reason": "",
        }

    # load_block_index 对版本不匹配的缓存会直接返回 None（见 block_index_service.py），
    # 所以 block_index 非 None 才代表"当前代码版本刚构建出的新鲜结果"。缓存过旧或
    # 尚未构建时 block_index 也是 None，这种"未知"不能当成"大纲为空"来推荐深度解析，
    # 否则纯粹因为缓存版本升级就会对一堆本来解析得很好的旧文档触发误报。
    if isinstance(block_index, dict):
        outline_count = len(block_index.get("outline") or [])
        if outline_count == 0 and total_pages > 3:
            return {
                "recommend_deep_parse": True,
                "recommend_reason": "本地未能识别出章节大纲，建议用 MinerU 深度解析重建带坐标的阅读结构",
                "recommend_rag_index_rebuild": False,
                "recommend_rag_index_reason": "",
            }

    return {
        "recommend_deep_parse": False,
        "recommend_reason": "",
        "recommend_rag_index_rebuild": False,
        "recommend_rag_index_reason": "",
    }


def _make_mineru_adapter(config: dict, access_mode: str):
    if access_mode == "direct":
        return MinerUDirectAdapter(
            token=config.get("token", ""),
            base_url=config.get("base_url", "https://mineru.net/api/v4"),
            enable_ocr=config.get("enable_ocr", False),
            enable_formula=config.get("enable_formula", True),
            enable_table=config.get("enable_table", True),
            model_version=config.get("model_version", "vlm"),
        )
    return MinerUAdapter(
        worker_url=config.get("worker_url", ""),
        auth_key=config.get("auth_key", ""),
        token=config.get("token", ""),
        token_mode=config.get("token_mode", "frontend"),
        enable_ocr=config.get("enable_ocr", False),
        enable_formula=config.get("enable_formula", True),
        enable_table=config.get("enable_table", True),
        model_version=config.get("model_version", "vlm"),
    )


def _run_mineru_deep_parse(
    doc_id: str,
    cancel_event: threading.Event,
    remote_job: Optional[dict] = None,
    parse_generation: str = "",
    full_route_options: Optional[dict] = None,
) -> None:
    acquired_slot = False
    acquired_document_lock = False
    parser_attempted = False
    parser_outcome_recorded = False
    full_mineru_route = False
    parse_source_hash = ""
    document_lock = _get_document_operation_lock(doc_id)

    def _worker_matches_current_generation() -> bool:
        """Keep an older same-PDF job from touching a newer parse generation."""
        if not parse_generation:
            return True
        current_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
        return matches_parse_generation(
            current_manifest,
            generation=parse_generation,
            source_hash=parse_source_hash or None,
        )

    def _set_worker_status(status: str, *, stage: str = "", error: str = "", **extra) -> bool:
        """Publish progress only while this worker still owns the document generation."""
        if not _worker_matches_current_generation():
            return False
        if parse_generation:
            extra.setdefault("parse_generation", parse_generation)
            extra.setdefault("document_source_hash", parse_source_hash)
        _set_deep_parse_status(doc_id, status, stage=stage, error=error, **extra)
        return True

    try:
        initial_doc = documents_store.get(doc_id)
        initial_manifest = _read_document_parse_manifest(doc_id, initial_doc)
        parse_source_hash = str(initial_manifest.get("source_hash") or "")
        if parse_generation and not matches_parse_generation(
            initial_manifest,
            generation=parse_generation,
            source_hash=parse_source_hash,
        ):
            logger.info("[DeepParse] skip stale MinerU worker for %s generation=%s", doc_id, parse_generation)
            return
        full_mineru_route = bool(
            full_route_options is not None
            or _is_full_mineru_parse_manifest(initial_manifest)
        )
        if full_mineru_route and initial_manifest.get("status") in {PARSE_STATUS_PENDING, PARSE_STATUS_QUEUED}:
            _transition_document_parse_manifest(
                doc_id,
                PARSE_STATUS_RUNNING,
                stage="mineru_parsing",
                doc=initial_doc,
                metadata={"full_route": True},
            )
        _set_worker_status("queued", stage="waiting_for_slot", message="等待 MinerU 解析槽位")
        _DEEP_PARSE_SEMAPHORE.acquire()
        acquired_slot = True
        if cancel_event.is_set():
            _set_worker_status("cancelled", stage="cancelled", message="MinerU 深度解析已取消")
            return
        _set_worker_status("queued", stage="waiting_for_document_lock", message="等待文档解析锁")
        document_lock.acquire()
        acquired_document_lock = True
        if not _worker_matches_current_generation():
            logger.info("[DeepParse] skip stale MinerU worker after lock for %s generation=%s", doc_id, parse_generation)
            return
        if cancel_event.is_set():
            _set_worker_status("cancelled", stage="cancelled", message="MinerU 深度解析已取消")
            return
        doc = documents_store.get(doc_id)
        if not doc:
            raise RuntimeError("文档记录不存在")

        pdf_path = _resolve_document_pdf_path(doc)
        if not pdf_path or not pdf_path.exists():
            raise RuntimeError("当前文档没有可用于 MinerU 深度解析的 PDF 原文件")

        config = _load_online_ocr_config("mineru")
        access_mode = str((remote_job or {}).get("access_mode") or config.get("access_mode") or "worker").strip().lower()
        adapter = _make_mineru_adapter(config, access_mode)
        if not adapter.is_available():
            raise RuntimeError("MinerU 未配置或不可用，请先在 OCR 设置中配置 Worker/直连模式和 Token")

        def _on_mineru_progress(progress: dict) -> None:
            if not isinstance(progress, dict):
                return
            stage = str(progress.get("stage") or "running")
            message = str(progress.get("message") or "")
            extra = {
                key: value
                for key, value in progress.items()
                if key not in {"stage", "message"}
            }
            extra.setdefault("model_version", config.get("model_version", "vlm"))
            _set_worker_status(
                "running",
                stage=stage,
                message=message,
                access_mode=access_mode,
                **extra,
            )

        if remote_job and remote_job.get("batch_id"):
            _set_worker_status(
                "running", stage="resuming", message="恢复 MinerU 远端任务",
                access_mode=access_mode, batch_id=remote_job["batch_id"], data_id=remote_job.get("data_id", ""),
                recovered_after_restart=True,
            )
            parser_attempted = True
            payload = adapter.resume_batch(
                str(remote_job["batch_id"]), data_id=str(remote_job.get("data_id") or ""),
                progress_callback=_on_mineru_progress, cancel_event=cancel_event,
            )
        else:
            _set_worker_status("running", stage="uploading", message="准备上传 PDF 到 MinerU")
            pdf_bytes = pdf_path.read_bytes()
            parser_attempted = True
            payload = adapter.analyze_pdf(pdf_bytes, progress_callback=_on_mineru_progress, cancel_event=cancel_event)
        record_ocr_provider_use("mineru", outcome="success", operation="document_parse")
        parser_outcome_recorded = True
        payload.setdefault("model_version", config.get("model_version", "vlm"))
        with _get_document_publication_lock(doc_id):
            if not _worker_matches_current_generation():
                logger.info("[DeepParse] discard stale MinerU payload for %s generation=%s", doc_id, parse_generation)
                return
            if cancel_event.is_set():
                _set_worker_status("cancelled", stage="cancelled", message="MinerU 深度解析已取消")
                return
            save_mineru_result(
                DATA_DIR,
                doc_id,
                payload,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )

        _set_worker_status("running", stage="building_index", message="重建阅读块和大纲")
        if cancel_event.is_set():
            _set_worker_status("cancelled", stage="cancelled", message="MinerU 深度解析已取消")
            return
        block_index = build_block_index_from_mineru_payload(
            doc_id=doc_id,
            doc=doc,
            payload=payload,
            pdf_path=pdf_path,
        )
        removed: list[str] = []
        waiting_for_rag_rebuild = False
        with _get_document_publication_lock(doc_id):
            if not _worker_matches_current_generation():
                logger.info("[DeepParse] discard stale MinerU block index for %s generation=%s", doc_id, parse_generation)
                return
            if cancel_event.is_set():
                _set_worker_status("cancelled", stage="cancelled", message="MinerU 深度解析已取消")
                return
            save_block_index(DATA_DIR, doc_id, block_index)
            current_doc = documents_store.get(doc_id)
            removed = _clear_block_dependent_ai_cache(doc_id, current_doc)

            if full_mineru_route:
                _transition_document_parse_manifest(
                    doc_id,
                    PARSE_STATUS_RUNNING,
                    stage="building_rag_index",
                    doc=current_doc,
                    metadata={
                        "full_route": True,
                        "block_source": MINERU_BLOCK_INDEX_SOURCE,
                        "figure_source": "mineru_deep_parse",
                    },
                )
                waiting_for_rag_rebuild = full_route_options is None

        if full_mineru_route:
            if waiting_for_rag_rebuild:
                # A restarted worker can safely resume the remote MinerU job,
                # but secrets for the embedding service are intentionally not
                # persisted. Leave this generation unpublished until the UI
                # supplies a fresh RAG rebuild request.
                with _get_document_publication_lock(doc_id):
                    if not _worker_matches_current_generation():
                        return
                    _transition_document_parse_manifest(
                        doc_id,
                        PARSE_STATUS_PENDING,
                        stage="awaiting_rag_index",
                        doc=documents_store.get(doc_id),
                    )
                _set_worker_status(
                    "ready",
                    stage="awaiting_rag_index",
                    message="MinerU 版面解析完成，等待使用当前 Embedding 配置发布问答索引",
                    active_source=MINERU_BLOCK_INDEX_SOURCE,
                    active_mineru=True,
                    access_mode=access_mode,
                )
                return
            _rebuild_mineru_rag_index_unlocked(
                doc_id,
                embedding_model=str(full_route_options.get("embedding_model") or "local-minilm"),
                embedding_api_key=full_route_options.get("embedding_api_key"),
                embedding_api_host=full_route_options.get("embedding_api_host"),
                summary_api_key=full_route_options.get("summary_api_key"),
                summary_model=str(full_route_options.get("summary_model") or "gpt-4o-mini"),
                summary_provider=str(full_route_options.get("summary_provider") or "openai"),
                summary_api_host=str(full_route_options.get("summary_api_host") or ""),
                expected_parse_generation=parse_generation,
                expected_document_source_hash=parse_source_hash,
            )

        block_count = sum(len(page.get("blocks") or []) for page in block_index.get("pages", []))
        outline_count = len(block_index.get("outline") or [])
        figure_count = sum(
            1
            for page in block_index.get("pages", [])
            for block in (page.get("blocks") or [])
            if block.get("type") in ("figure", "table")
        )
        _set_worker_status(
            "ready",
            stage="ready",
            block_count=block_count,
            outline_count=outline_count,
            figure_count=figure_count,
            cache_removed=removed,
            active_source=MINERU_BLOCK_INDEX_SOURCE,
            active_mineru=True,
            access_mode=access_mode,
            model_version=config.get("model_version", "vlm"),
        )
        logger.info("[DeepParse] MinerU deep parse ready for %s: blocks=%s outline=%s", doc_id, block_count, outline_count)
    except _SupersededParseGeneration:
        logger.info("[DeepParse] MinerU worker superseded for %s generation=%s", doc_id, parse_generation)
        return
    except Exception as exc:
        if parser_attempted and not parser_outcome_recorded and not cancel_event.is_set():
            record_ocr_provider_use("mineru", outcome="failure", operation="document_parse")
            parser_outcome_recorded = True
        if cancel_event.is_set() or "已取消" in str(exc):
            logger.info("[DeepParse] MinerU deep parse cancelled for %s", doc_id)
            if full_mineru_route:
                try:
                    _transition_current_full_mineru_manifest(
                        doc_id,
                        PARSE_STATUS_CANCELLED,
                        parse_generation=parse_generation,
                        document_source_hash=parse_source_hash,
                        stage="cancelled",
                        error="",
                        expected_statuses={
                            PARSE_STATUS_PENDING,
                            PARSE_STATUS_QUEUED,
                            PARSE_STATUS_RUNNING,
                            PARSE_STATUS_FAILED,
                            PARSE_STATUS_CANCELLED,
                        },
                    )
                except Exception:
                    logger.debug("[DeepParse] failed to mark parse manifest cancelled for %s", doc_id)
            _set_worker_status("cancelled", stage="cancelled", message="MinerU 深度解析已取消", error="")
            return
        logger.exception("[DeepParse] MinerU deep parse failed for %s: %s", doc_id, exc)
        if full_mineru_route and _worker_matches_current_generation():
            try:
                _transition_document_parse_manifest(
                    doc_id,
                    PARSE_STATUS_FAILED,
                    stage="failed",
                    error=str(exc),
                )
            except Exception:
                logger.debug("[DeepParse] failed to mark parse manifest failed for %s", doc_id)
        _set_worker_status("failed", stage="failed", error=str(exc))
    finally:
        if acquired_document_lock:
            document_lock.release()
        if acquired_slot:
            _DEEP_PARSE_SEMAPHORE.release()
        with _DEEP_PARSE_LOCK:
            # A same-PDF re-upload gets a new parse generation and cancel
            # event. An old worker must never remove that new worker's handle.
            if _DEEP_PARSE_CANCEL_EVENTS.get(doc_id) is cancel_event:
                _DEEP_PARSE_CANCEL_EVENTS.pop(doc_id, None)


def _queue_mineru_deep_parse(
    doc_id: str,
    *,
    force: bool = False,
    full_route_options: Optional[dict] = None,
) -> dict:
    doc = documents_store.get(doc_id)
    parse_manifest = _read_document_parse_manifest(doc_id, doc)
    parse_generation = str(parse_manifest.get("generation") or "")
    parse_source_hash = str(parse_manifest.get("source_hash") or "")
    current = _get_deep_parse_status(doc_id)
    # ``_get_deep_parse_status`` intentionally hides an old task from a new
    # document generation. Queueing still has to retire that task's concrete
    # cancellation handle, even though it is no longer user-visible.
    with _DEEP_PARSE_LOCK:
        previous_task = dict(_DEEP_PARSE_TASKS.get(doc_id) or {})
        previous_cancel_event = _DEEP_PARSE_CANCEL_EVENTS.get(doc_id)
        previous_generation = str(previous_task.get("parse_generation") or "")
        if previous_cancel_event and previous_generation and previous_generation != parse_generation:
            previous_cancel_event.set()
    current_matches_generation = bool(
        parse_generation
        and str(current.get("parse_generation") or "") == parse_generation
        and str(current.get("document_source_hash") or "") in {"", parse_source_hash}
    )
    if current.get("status") in {"queued", "running"} and current_matches_generation:
        return current
    if current.get("active_mineru") and not force and current_matches_generation:
        return current

    full_mineru_route = bool(
        full_route_options is not None or _is_full_mineru_parse_manifest(parse_manifest)
    )
    if full_mineru_route and parse_manifest.get("status") == PARSE_STATUS_FAILED:
        requeued_manifest = _transition_current_full_mineru_manifest(
            doc_id,
            PARSE_STATUS_QUEUED,
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
            stage="mineru_queued",
            error="",
            expected_statuses={PARSE_STATUS_FAILED},
        )
        if requeued_manifest is None:
            # Another upload or retry won the publication lock. Do not start a
            # worker for the stale failed snapshot read above.
            return _get_deep_parse_status(doc_id)

    # Document ids are content hashes. Re-uploading the same PDF intentionally
    # creates a new parse generation, so its worker must not be suppressed by
    # an older job still waiting on MinerU or the document lock.
    if current.get("status") in {"queued", "running"} and not current_matches_generation:
        with _DEEP_PARSE_LOCK:
            previous_cancel_event = _DEEP_PARSE_CANCEL_EVENTS.get(doc_id)
            if previous_cancel_event:
                previous_cancel_event.set()

    cancel_event = threading.Event()
    with _DEEP_PARSE_LOCK:
        _DEEP_PARSE_CANCEL_EVENTS[doc_id] = cancel_event
    _set_deep_parse_status(
        doc_id,
        "queued",
        stage="queued",
        error="",
        job_id=f"mineru-{uuid.uuid4().hex}",
        parse_generation=parse_generation,
        document_source_hash=parse_source_hash,
        full_route=full_mineru_route,
    )
    thread = threading.Thread(
        target=_run_mineru_deep_parse,
        args=(doc_id, cancel_event, None, parse_generation, full_route_options),
        name=f"chatpdf-mineru-{doc_id[:8]}",
        daemon=True,
    )
    thread.start()
    return _get_deep_parse_status(doc_id)


def resume_pending_mineru_deep_parse_jobs() -> list[dict]:
    """Resume persisted remote MinerU batches after a process restart."""
    jobs_dir = DATA_DIR / "document_jobs" / _DEEP_PARSE_JOB_TYPE
    if not jobs_dir.exists():
        return []
    resumed: list[dict] = []
    for job_path in jobs_dir.glob("*.json"):
        try:
            record = json.loads(job_path.read_text(encoding="utf-8"))
            doc_id = str(record.get("doc_id") or "")
            if (
                not doc_id or doc_id not in documents_store
                or record.get("status") not in {"queued", "running"}
                or not record.get("batch_id")
            ):
                continue
            cancel_event = threading.Event()
            with _DEEP_PARSE_LOCK:
                if doc_id in _DEEP_PARSE_CANCEL_EVENTS:
                    continue
                _DEEP_PARSE_CANCEL_EVENTS[doc_id] = cancel_event
                _DEEP_PARSE_TASKS[doc_id] = dict(record)
            thread = threading.Thread(
                target=_run_mineru_deep_parse,
                args=(doc_id, cancel_event, record, str(record.get("parse_generation") or ""), None),
                name=f"chatpdf-mineru-resume-{doc_id[:8]}",
                daemon=True,
            )
            thread.start()
            resumed.append({"doc_id": doc_id, "batch_id": record["batch_id"]})
        except Exception as exc:
            logger.error("[DeepParse] failed to resume persisted MinerU job %s: %s", job_path, exc)
    return resumed


def _cancel_mineru_deep_parse(doc_id: str) -> dict:
    current = _get_deep_parse_status(doc_id)
    if current.get("status") not in {"queued", "running"}:
        return current
    parse_generation = str(current.get("parse_generation") or "")
    document_source_hash = str(current.get("document_source_hash") or "")
    with _DEEP_PARSE_LOCK:
        event = _DEEP_PARSE_CANCEL_EVENTS.get(doc_id)
        if event:
            event.set()
    _transition_current_full_mineru_manifest(
        doc_id,
        PARSE_STATUS_CANCELLED,
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
        stage="cancelled",
        error="",
        expected_statuses={
            PARSE_STATUS_PENDING,
            PARSE_STATUS_QUEUED,
            PARSE_STATUS_RUNNING,
            PARSE_STATUS_FAILED,
        },
    )
    remote_cancel = {"attempted": False, "state": "not_requested"}
    batch_id = str(current.get("batch_id") or "")
    if batch_id:
        config = _load_online_ocr_config("mineru")
        access_mode = str(current.get("access_mode") or config.get("access_mode") or "worker").strip().lower()
        try:
            remote_cancel = _make_mineru_adapter(config, access_mode).cancel_batch(
                batch_id,
                data_id=str(current.get("data_id") or ""),
            )
        except Exception as exc:
            remote_cancel = {"attempted": True, "state": "error", "detail": str(exc)}
    _set_deep_parse_status(
        doc_id, "cancelled", stage="cancelled", message="MinerU 深度解析已取消", error="",
        remote_cancel=remote_cancel,
    )
    return _get_deep_parse_status(doc_id)


def _estimate_text_tokens(text: str) -> int:
    # Conservative mixed CJK/English estimate for user-facing cost hints.
    return max(1, int(len(text or "") / 2.2))


def _estimate_chunk_count(text: str, *, chunk_chars: int = 1200, overlap_chars: int = 200) -> int:
    value = str(text or "")
    if not value.strip():
        return 0
    step = max(1, chunk_chars - overlap_chars)
    return max(1, int((len(value) + step - 1) / step))


def _safe_index_source_name(source: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", source or "pdf_native").strip("_") or "pdf_native"


def _backup_current_vector_index(doc_id: str, source: str) -> dict:
    index_path, pkl_path = _vector_index_paths(doc_id)
    if not index_path.exists() or not pkl_path.exists():
        return {"backed_up": False}
    safe_source = _safe_index_source_name(source)
    backup_index = VECTOR_STORE_DIR / f"{doc_id}.{safe_source}.bak.index"
    backup_pkl = VECTOR_STORE_DIR / f"{doc_id}.{safe_source}.bak.pkl"
    shutil.copy2(index_path, backup_index)
    shutil.copy2(pkl_path, backup_pkl)
    return {
        "backed_up": True,
        "source": safe_source,
        "index_path": str(backup_index),
        "pkl_path": str(backup_pkl),
    }


def _backup_current_semantic_groups(doc_id: str, source: str) -> dict:
    paths = _semantic_group_paths(doc_id)
    backed_up: dict[str, str] = {}
    existing = {kind: path for kind, path in paths.items() if path.exists()}
    if not existing:
        return {"backed_up": False}
    for kind, path in existing.items():
        backup_path = _semantic_group_backup_path(doc_id, source, kind)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        backed_up[kind] = str(backup_path)
    return {
        "backed_up": True,
        "source": _safe_index_source_name(source),
        "paths": backed_up,
    }


def _remove_current_semantic_groups(doc_id: str) -> dict:
    try:
        result = deactivate_generation(DATA_DIR / "semantic_groups", doc_id)
    except Exception as exc:
        logger.warning("[RagIndex] failed to deactivate semantic groups for %s: %s", doc_id, exc)
        result = {"deactivated": False, "removed": [], "error": str(exc)}
    _index_cache.invalidate(doc_id)
    return result


def _publish_temp_semantic_groups(
    doc_id: str,
    temp_dir: Path,
    validation: dict,
    *,
    source_hash: str = "",
    transaction_id: str = "",
) -> dict:
    if validation.get("status") == "disabled":
        return _remove_current_semantic_groups(doc_id)
    result = publish_generation(
        DATA_DIR / "semantic_groups",
        doc_id,
        temp_dir,
        source_hash=source_hash,
        transaction_id=transaction_id,
    )
    _index_cache.invalidate(doc_id)
    return result


def _validate_temp_semantic_groups(doc_id: str, temp_dir: Path, result: dict) -> dict:
    status = str((result or {}).get("status") or "failed")
    if status == "disabled":
        return {"valid": True, "status": status, "paths": []}
    expected = [temp_dir / f"{doc_id}.json", temp_dir / f"{doc_id}_groups.index", temp_dir / f"{doc_id}_groups.pkl"]
    missing = [str(path) for path in expected if not path.exists() or path.stat().st_size <= 0]
    if status != "ready" or missing:
        raise RuntimeError(f"临时 semantic groups 不完整: status={status}, missing={missing}")
    artifact_validation = validate_semantic_group_artifacts(
        {"json": expected[0], "index": expected[1], "pkl": expected[2]},
        doc_id,
    )
    if not artifact_validation["valid"]:
        raise RuntimeError(
            "临时 semantic groups 无法加载: "
            + ", ".join(artifact_validation["errors"])
        )
    return {
        "valid": True,
        "status": status,
        "paths": [str(path) for path in expected],
        "group_count": artifact_validation["group_count"],
    }


def _restore_semantic_group_backup(
    doc_id: str,
    source: str,
    *,
    source_hash: str = "",
    transaction_id: str = "",
) -> dict:
    root = DATA_DIR / "semantic_groups"
    staged_dir = root / "_restore" / f"{_safe_index_source_name(doc_id)}.{uuid.uuid4().hex}"
    restored: dict[str, str] = {}
    for kind, target_path in _semantic_group_paths(doc_id).items():
        backup_path = _semantic_group_backup_path(doc_id, source, kind)
        if not backup_path.exists():
            continue
        staged_dir.mkdir(parents=True, exist_ok=True)
        target = staged_dir / target_path.name
        shutil.copy2(backup_path, target)
        restored[kind] = str(target)
    if len(restored) == 3:
        published = publish_generation(
            root,
            doc_id,
            staged_dir,
            source_hash=source_hash,
            transaction_id=transaction_id,
        )
        restored = dict(published["paths"])
    elif restored:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise RuntimeError("语义组备份不完整，拒绝发布部分恢复结果")
    else:
        _remove_current_semantic_groups(doc_id)
    _index_cache.invalidate(doc_id)
    return {"restored": bool(restored), "paths": restored}


def _document_backup_path(doc_id: str, source: str) -> Path:
    return DOCS_DIR / f"{doc_id}.{_safe_index_source_name(source)}.bak.doc.json"


def _backup_current_document_data(doc_id: str, source: str) -> dict:
    doc = documents_store.get(doc_id)
    if not isinstance(doc, dict):
        return {"backed_up": False}
    path = _document_backup_path(doc_id, source)
    try:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        return {"backed_up": True, "source": _safe_index_source_name(source), "path": str(path)}
    except Exception as exc:
        logger.warning("[RagIndex] failed to backup document data for %s: %s", doc_id, exc)
        return {"backed_up": False, "error": str(exc)}


def _rag_backup_manifest_path(doc_id: str, source: str) -> Path:
    return DATA_DIR / "rag_transactions" / f"{doc_id}.{_safe_index_source_name(source)}.manifest.json"


def _rag_transaction_journal_path(doc_id: str) -> Path:
    return DATA_DIR / "rag_transactions" / "pending" / f"{_safe_index_source_name(doc_id)}.json"


def _write_rag_transaction_journal(doc_id: str, state: str, *, source: str, manifest_path: str, error: str = "") -> dict:
    """Persist the source-switch phase before moving to the next artifact."""
    path = _rag_transaction_journal_path(doc_id)
    payload = {
        "schema_version": 1,
        "doc_id": doc_id,
        "source": _safe_index_source_name(source),
        "state": state,
        "manifest_path": manifest_path,
        "updated_at": utc_now_iso(),
    }
    if error:
        payload["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temp_path), str(path))
    return {**payload, "path": str(path)}


def _backup_current_rag_state(doc_id: str, source: str) -> dict:
    """Snapshot all active artifacts before a source switch."""
    vector = _backup_current_vector_index(doc_id, source)
    document = _backup_current_document_data(doc_id, source)
    semantic = _backup_current_semantic_groups(doc_id, source)
    complete = bool(vector.get("backed_up") and document.get("backed_up"))
    manifest = {
        "schema_version": 1,
        "doc_id": doc_id,
        "source": _safe_index_source_name(source),
        "created_at": utc_now_iso(),
        "complete": complete,
        "vector": vector,
        "document": document,
        "semantic_groups": semantic,
    }
    path = _rag_backup_manifest_path(doc_id, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
    manifest["path"] = str(path)
    return manifest


def _load_complete_rag_backup_manifest(doc_id: str, source: str) -> dict:
    path = _rag_backup_manifest_path(doc_id, source)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(manifest, dict) or not manifest.get("complete"):
        return {}
    vector = manifest.get("vector") or {}
    document = manifest.get("document") or {}
    if not (Path(str(vector.get("index_path") or "")).exists() and Path(str(vector.get("pkl_path") or "")).exists() and Path(str(document.get("path") or "")).exists()):
        return {}
    return manifest


def _is_full_mineru_parse_manifest(manifest: dict) -> bool:
    return bool(
        manifest.get("resolved_route") == PARSE_ROUTE_MINERU
        and (manifest.get("metadata") or {}).get("full_route")
    )


def _is_legacy_parse_manifest(manifest: dict) -> bool:
    return bool((manifest.get("metadata") or {}).get("legacy_inferred"))


def _require_mineru_route_compatibility(doc_id: str, doc: dict) -> dict:
    """Limit the old block-only MinerU path to documents created before routes."""
    manifest = _read_document_parse_manifest(doc_id, doc)
    if _is_full_mineru_parse_manifest(manifest) or _is_legacy_parse_manifest(manifest):
        return manifest
    raise HTTPException(
        status_code=409,
        detail=(
            "当前文档已按本地解析路线发布。为避免阅读、大纲、翻译和问答来源不一致，"
            "请重新上传并在上传前选择 MinerU 全程解析。"
        ),
    )


def _load_mineru_result_for_manifest(doc_id: str, manifest: dict) -> dict | None:
    """Load only the raw MinerU payload belonging to this document run."""
    legacy_parse = _is_legacy_parse_manifest(manifest)
    return load_mineru_result(
        DATA_DIR,
        doc_id,
        parse_generation=(None if legacy_parse else str(manifest.get("generation") or "")),
        document_source_hash=(None if legacy_parse else str(manifest.get("source_hash") or "")),
        require_identity=not legacy_parse,
    )


def _mark_full_mineru_parse_ready(
    doc_id: str,
    *,
    expected_parse_generation: str = "",
    expected_document_source_hash: str = "",
) -> dict:
    """Open a MinerU-first document only after all primary artifacts publish."""
    if expected_parse_generation:
        _require_current_parse_generation(
            doc_id,
            parse_generation=expected_parse_generation,
            document_source_hash=expected_document_source_hash,
        )
    doc = documents_store.get(doc_id)
    if not isinstance(doc, dict):
        raise RuntimeError("文档记录不存在，无法发布 MinerU 解析结果")
    manifest = _read_document_parse_manifest(doc_id, doc)
    if not _is_full_mineru_parse_manifest(manifest):
        return manifest
    if manifest.get("status") in {PARSE_STATUS_FAILED, PARSE_STATUS_CANCELLED}:
        raise RuntimeError("已终止的 MinerU 解析不能直接发布")
    if manifest.get("status") in {PARSE_STATUS_PENDING, PARSE_STATUS_QUEUED}:
        manifest = _transition_document_parse_manifest(
            doc_id,
            PARSE_STATUS_RUNNING,
            stage="building_rag_index",
            doc=doc,
        )
    if manifest.get("status") != PARSE_STATUS_READY:
        manifest = _transition_document_parse_manifest(
            doc_id,
            PARSE_STATUS_READY,
            stage="ready",
            doc=doc,
        )
    return manifest


def _apply_mineru_rag_document_data(
    doc_id: str,
    normalized: dict,
    *,
    expected_parse_generation: str = "",
    expected_document_source_hash: str = "",
) -> None:
    if expected_parse_generation:
        _require_current_parse_generation(
            doc_id,
            parse_generation=expected_parse_generation,
            document_source_hash=expected_document_source_hash,
        )
    doc = documents_store.get(doc_id)
    if not isinstance(doc, dict):
        raise RuntimeError("文档记录不存在，无法切换问答数据源")
    data = dict(doc.get("data") or {})
    pages = []
    for page in normalized.get("pages") or []:
        if not isinstance(page, dict):
            continue
        page_copy = dict(page)
        content = str(page_copy.get("content") or page_copy.get("text") or "")
        page_copy["content"] = content
        page_copy["text"] = content
        page_copy["source"] = MINERU_RAG_INDEX_SOURCE
        pages.append(page_copy)

    structured_table_bundles = normalized.get("structured_table_bundles") or []
    parse_manifest = _read_document_parse_manifest(doc_id, doc)
    data.update({
        "full_text": normalized.get("full_text", ""),
        "pages": pages,
        "total_pages": len(pages) or data.get("total_pages", 0),
        "structured_table_bundles": structured_table_bundles,
        "structured_table_count": len(structured_table_bundles),
        "rag_index_source": MINERU_RAG_INDEX_SOURCE,
        "rag_source_hash": normalized.get("source_hash", ""),
        "rag_normalizer_version": normalized.get("normalizer_version", ""),
        "rag_quality_report": normalized.get("quality_report") or {},
        "parse_artifact": normalized.get("parse_artifact") or data.get("parse_artifact") or {},
        "extraction_method": data.get("extraction_method", "pdf_native"),
    })
    if _is_full_mineru_parse_manifest(parse_manifest):
        # Keep the route gated while the staged semantic index is published.
        # A full MinerU route must not expose its new text before every primary
        # consumer can resolve the same generation.
        metadata = dict(parse_manifest.get("metadata") or {})
        metadata.update({
            "text_source": "mineru",
            "block_source": MINERU_BLOCK_INDEX_SOURCE,
            "rag_source": MINERU_RAG_INDEX_SOURCE,
            "figure_source": "mineru_deep_parse",
            "parser_source_hash": normalized.get("source_hash", ""),
            "normalizer_version": normalized.get("normalizer_version", ""),
        })
        parse_manifest["metadata"] = metadata
        data["parse_manifest"] = parse_manifest
    doc["data"] = data
    _normalize_page_keys(doc)
    save_document(doc_id, doc)


def _restore_document_backup(doc_id: str, source: str = "pdf_native") -> dict:
    path = _document_backup_path(doc_id, source)
    if not path.exists():
        return {"restored": False}
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            restored_doc = json.load(f)
        _normalize_page_keys(restored_doc)
        documents_store[doc_id] = restored_doc
        save_document(doc_id, restored_doc)
        return {"restored": True, "path": str(path)}
    except Exception as exc:
        logger.warning("[RagIndex] failed to restore document backup for %s: %s", doc_id, exc)
        return {"restored": False, "error": str(exc)}


def _replace_vector_index_from_temp(doc_id: str, temp_dir: Path) -> None:
    temp_index, temp_pkl = _vector_index_paths(doc_id, temp_dir)
    if not temp_index.exists() or not temp_pkl.exists():
        raise RuntimeError("临时问答索引未生成完整 index/pkl 文件")
    index_path, pkl_path = _vector_index_paths(doc_id)
    os.replace(str(temp_index), str(index_path))
    os.replace(str(temp_pkl), str(pkl_path))
    _index_cache.invalidate(doc_id)


def _validate_temp_vector_index(doc_id: str, temp_dir: Path) -> tuple[bool, list[str]]:
    _index_path, chunks_path = _vector_index_paths(doc_id, temp_dir)
    failures: list[str] = []
    if not chunks_path.exists():
        return False, ["temp_pkl_missing"]
    try:
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        return False, [f"temp_pkl_unreadable:{exc}"]
    if not isinstance(data, dict):
        return False, ["temp_pkl_legacy_shape"]
    if data.get("index_source") != MINERU_RAG_INDEX_SOURCE:
        failures.append("temp_index_source_not_mineru")
    chunks = data.get("chunks") or []
    if not chunks:
        failures.append("temp_chunks_empty")
    for chunk in chunks:
        if re.search(r"</?(?:table|thead|tbody|tfoot|tr|td|th)\b", str(chunk or ""), re.IGNORECASE):
            failures.append("html_tag_in_temp_chunks")
            break
    metadata = data.get("chunk_metadata") or []
    for item in metadata:
        if not isinstance(item, dict):
            continue
        body = str(item.get("table_body_markdown") or "")
        if body and re.search(r"</?(?:table|thead|tbody|tfoot|tr|td|th)\b", body, re.IGNORECASE):
            failures.append("html_tag_in_temp_table_markdown")
            break
    return not failures, failures


def _prepare_semantic_group_rebuild(
    doc_id: str,
    temp_dir: Path,
    *,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_api_host: Optional[str],
    summary_api_key: Optional[str],
    summary_model: str = "gpt-4o-mini",
    summary_provider: str = "openai",
    summary_api_host: str = "",
) -> dict:
    """Prepare the post-swap semantic group rebuild from the validated temp index.

    MinerU rebuild writes the vector index into a temp directory first, so
    create_index(..., build_semantic_groups=False) cannot build semantic groups
    yet. Preparing here keeps failures before the active index is replaced.
    """
    _index_path, chunks_path = _vector_index_paths(doc_id, temp_dir)
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("临时问答索引格式异常，无法准备意群索引重建")
    chunks = [str(chunk or "") for chunk in (data.get("chunks") or []) if str(chunk or "").strip()]
    if not chunks:
        raise RuntimeError("临时问答索引分块为空，无法准备意群索引重建")
    effective_embedding_model = data.get("embedding_model") or embedding_model
    embed_fn = get_embedding_function(effective_embedding_model, embedding_api_key, embedding_api_host)
    return {
        "chunks": chunks,
        "embed_fn": embed_fn,
        # Do not silently reuse the embedding key for semantic-group summaries.
        # Rebuild/evaluation jobs often use a dedicated embedding provider whose
        # key cannot call chat completions; passing it here makes the background
        # task spend minutes failing before falling back.  A missing summary key
        # intentionally selects SemanticGroupService's deterministic truncation.
        "api_key": summary_api_key,
        "model": summary_model or "gpt-4o-mini",
        "provider": summary_provider or "openai",
        "endpoint": summary_api_host or "",
        "embedding_model": effective_embedding_model,
    }


def _restore_vector_index_backup(doc_id: str, source: str = "pdf_native") -> dict:
    """恢复完整 RAG 快照，并通过顶层 ``restored`` 返回统一结果。"""
    safe_source = _safe_index_source_name(source)
    backup_index = VECTOR_STORE_DIR / f"{doc_id}.{safe_source}.bak.index"
    backup_pkl = VECTOR_STORE_DIR / f"{doc_id}.{safe_source}.bak.pkl"
    if not backup_index.exists() or not backup_pkl.exists():
        raise RuntimeError("没有可回退的本地问答索引备份")
    index_path, pkl_path = _vector_index_paths(doc_id)
    shutil.copy2(backup_index, index_path)
    shutil.copy2(backup_pkl, pkl_path)
    doc_restore = _restore_document_backup(doc_id, safe_source)
    restored_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    legacy_manifest = _is_legacy_parse_manifest(restored_manifest)
    semantic_restore = _restore_semantic_group_backup(
        doc_id,
        safe_source,
        source_hash=("" if legacy_manifest else str(restored_manifest.get("source_hash") or "")),
        transaction_id=("" if legacy_manifest else str(restored_manifest.get("generation") or "")),
    )
    backup_manifest = _load_complete_rag_backup_manifest(doc_id, safe_source)
    semantic_required = bool((backup_manifest.get("semantic_groups") or {}).get("backed_up"))
    restored = bool(
        doc_restore.get("restored")
        and (semantic_restore.get("restored") or not semantic_required)
    )
    _index_cache.invalidate(doc_id)
    _set_document_index_status(doc_id, "ready", stage="ready")
    status = _get_rag_index_status(doc_id)
    status["restored"] = restored
    status["document_restore"] = doc_restore
    status["semantic_group_restore"] = semantic_restore
    return status


def _document_pdf_page_sizes(doc_id: str) -> dict[int, list[float]]:
    """Return the original PDF page dimensions for parser bbox conversion."""
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        return {}
    try:
        import fitz

        document = fitz.open(str(pdf_path))
        try:
            return {
                page_index + 1: [float(page.rect.width), float(page.rect.height)]
                for page_index, page in enumerate(document)
            }
        finally:
            document.close()
    except Exception as exc:
        logger.warning("[RagIndex] failed to read PDF page sizes for %s: %s", doc_id, exc)
        return {}


def _rebuild_mineru_rag_index(
    doc_id: str,
    *,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_api_host: Optional[str],
    summary_api_key: Optional[str],
    summary_model: str = "gpt-4o-mini",
    summary_provider: str = "openai",
    summary_api_host: str = "",
    expected_parse_generation: str = "",
    expected_document_source_hash: str = "",
) -> dict:
    """Rebuild under the same document lock used by MinerU deep parsing."""
    document_lock = _get_document_operation_lock(doc_id)
    if not document_lock.acquire(blocking=False):
        raise RuntimeError("该文档正在执行 MinerU 深度解析或索引重建，请稍后再试")
    try:
        return _rebuild_mineru_rag_index_unlocked(
            doc_id,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            embedding_api_host=embedding_api_host,
            summary_api_key=summary_api_key,
            summary_model=summary_model,
            summary_provider=summary_provider,
            summary_api_host=summary_api_host,
            expected_parse_generation=expected_parse_generation,
            expected_document_source_hash=expected_document_source_hash,
        )
    finally:
        document_lock.release()


def _rebuild_mineru_rag_index_unlocked(
    doc_id: str,
    *,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_api_host: Optional[str],
    summary_api_key: Optional[str],
    summary_model: str = "gpt-4o-mini",
    summary_provider: str = "openai",
    summary_api_host: str = "",
    expected_parse_generation: str = "",
    expected_document_source_hash: str = "",
) -> dict:
    if doc_id not in documents_store:
        raise RuntimeError("文档记录不存在")
    doc = documents_store[doc_id]
    parse_manifest = _read_document_parse_manifest(doc_id, doc)
    legacy_parse = _is_legacy_parse_manifest(parse_manifest)
    expected_parse_generation = str(
        expected_parse_generation or parse_manifest.get("generation") or ""
    )
    expected_document_source_hash = str(
        expected_document_source_hash or parse_manifest.get("source_hash") or ""
    )
    _require_current_parse_generation(
        doc_id,
        parse_generation=expected_parse_generation,
        document_source_hash=expected_document_source_hash,
    )
    payload = _load_mineru_result_for_manifest(doc_id, parse_manifest)
    if not payload:
        raise RuntimeError("当前解析代际没有可用的 MinerU 原始结果，请重新执行 MinerU 解析")

    original_data = doc.get("data") or {}
    previous_meta = _read_vector_index_meta(doc_id)
    previous_source = previous_meta.get("index_source") or "pdf_native"
    had_previous_rag = _vector_index_ready(doc_id)

    normalized = normalize_mineru_for_rag(payload, page_sizes=_document_pdf_page_sizes(doc_id))
    ok, failures = validate_mineru_rag_data(
        normalized,
        original_full_text=original_data.get("full_text", ""),
    )
    quality_report = dict(normalized.get("quality_report") or {})
    if failures:
        quality_report["failure_reasons"] = sorted(set((quality_report.get("failure_reasons") or []) + failures))
        raise RuntimeError(f"MinerU 问答索引重建失败，已保留原索引: {', '.join(failures)}")

    artifact = build_document_parse_artifact(
        doc_id=doc_id,
        provider="mineru",
        provider_version=str(normalized.get("normalizer_version") or ""),
        pages=normalized.get("pages") or [],
        tables=normalized.get("structured_table_bundles") or [],
        warnings=(normalized.get("quality_report") or {}).get("warnings") or [],
        capabilities={
            "per_page_text": True,
            "document_structure": True,
            "structured_tables": True,
            "figures": True,
            **derive_table_geometry_capabilities(normalized.get("structured_table_bundles") or []),
        },
        source_hash=str(normalized.get("source_hash") or ""),
        raw_ref=f"mineru_results/{doc_id}.json",
    )
    artifact_path = persist_document_parse_artifact(DATA_DIR, artifact)
    normalized["parse_artifact"] = {
        "schema_version": artifact["schema_version"],
        "provider": artifact["provider"],
        "source_hash": artifact["source_hash"],
        "ref": artifact_reference(DATA_DIR, artifact_path),
    }

    full_text = normalized.get("full_text", "")
    pages = artifact["pages"]
    structured_table_bundles = artifact["tables"]
    if not full_text or not pages:
        raise RuntimeError("MinerU 规范化结果为空，已保留原索引")

    temp_dir = VECTOR_STORE_DIR / "_tmp" / f"{doc_id}.mineru"
    temp_semantic_dir = DATA_DIR / "semantic_groups" / "_tmp" / f"{doc_id}.mineru"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    if temp_semantic_dir.exists():
        shutil.rmtree(temp_semantic_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_semantic_dir.mkdir(parents=True, exist_ok=True)

    _set_document_index_status(doc_id, "running", stage="rebuilding_rag_index")
    backup = {}
    doc_backup = {}
    semantic_backup = {}
    semantic_cleanup = {}
    replaced_current_index = False
    try:
        create_index(
            doc_id,
            full_text,
            str(temp_dir),
            embedding_model,
            embedding_api_key,
            embedding_api_host,
            pages=pages,
            structured_table_bundles=structured_table_bundles,
            summary_api_key=summary_api_key,
            index_source=MINERU_RAG_INDEX_SOURCE,
            index_meta={
                "source_hash": normalized.get("source_hash", ""),
                "document_source_hash": parse_manifest.get("source_hash", ""),
                "parse_generation": parse_manifest.get("generation", ""),
                "parser_route": parse_manifest.get("resolved_route", ""),
                "rebuilt_at": utc_now_iso(),
                "previous_index_source": previous_source,
                "normalizer_version": normalized.get("normalizer_version", ""),
                "parse_artifact_ref": normalized["parse_artifact"]["ref"],
            },
            build_semantic_groups=False,
        )
        temp_meta = _read_vector_index_meta(doc_id, temp_dir)
        if temp_meta.get("index_source") != MINERU_RAG_INDEX_SOURCE:
            raise RuntimeError("临时索引缺少 MinerU 来源标记")
        temp_ok, temp_failures = _validate_temp_vector_index(doc_id, temp_dir)
        if not temp_ok:
            raise RuntimeError(f"MinerU 问答索引质量门失败，已保留原索引: {', '.join(temp_failures)}")
        semantic_rebuild = _prepare_semantic_group_rebuild(
            doc_id,
            temp_dir,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            embedding_api_host=embedding_api_host,
            summary_api_key=summary_api_key,
            summary_model=summary_model,
            summary_provider=summary_provider,
            summary_api_host=summary_api_host,
        )
        semantic_result = _build_semantic_group_index(
            doc_id, semantic_rebuild["chunks"], pages, semantic_rebuild["embed_fn"], semantic_rebuild["api_key"],
            model=semantic_rebuild["model"], provider=semantic_rebuild["provider"], endpoint=semantic_rebuild["endpoint"],
            output_dir=str(temp_semantic_dir), raise_on_error=True,
        )
        semantic_validation = _validate_temp_semantic_groups(doc_id, temp_semantic_dir, semantic_result)
        # The expensive temp build above intentionally runs without holding
        # the upload path. The irreversible swap below is short and guarded
        # by both the publication lock and the active parse identity.
        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=expected_parse_generation,
                document_source_hash=expected_document_source_hash,
            )
            if had_previous_rag:
                backup_manifest = _backup_current_rag_state(doc_id, previous_source)
                if not backup_manifest.get("complete"):
                    raise RuntimeError("无法创建完整的当前 RAG 快照，已取消切换")
                transaction_journal = _write_rag_transaction_journal(
                    doc_id,
                    "prepared",
                    source=previous_source,
                    manifest_path=str(backup_manifest.get("path") or ""),
                )
            else:
                # A MinerU-first upload has no local vector pair to back up. It is
                # still a transaction, but there is no previous generation to
                # restore on failure.
                backup_manifest = {
                    "complete": False,
                    "fresh_document": True,
                    "source": previous_source,
                    "vector": {},
                    "document": {},
                    "semantic_groups": {},
                }
                transaction_journal = {}
            backup = dict(backup_manifest.get("vector") or {})
            doc_backup = dict(backup_manifest.get("document") or {})
            semantic_backup = dict(backup_manifest.get("semantic_groups") or {})
            _replace_vector_index_from_temp(doc_id, temp_dir)
            replaced_current_index = True
            if had_previous_rag:
                _write_rag_transaction_journal(
                    doc_id,
                    "vector_swapped",
                    source=previous_source,
                    manifest_path=str(backup_manifest.get("path") or ""),
                )
            _apply_mineru_rag_document_data(
                doc_id,
                normalized,
                expected_parse_generation=("" if legacy_parse else expected_parse_generation),
                expected_document_source_hash=("" if legacy_parse else expected_document_source_hash),
            )
            if had_previous_rag:
                _write_rag_transaction_journal(
                    doc_id,
                    "document_swapped",
                    source=previous_source,
                    manifest_path=str(backup_manifest.get("path") or ""),
                )
            semantic_cleanup = _publish_temp_semantic_groups(
                doc_id,
                temp_semantic_dir,
                semantic_validation,
                source_hash=("" if legacy_parse else expected_document_source_hash),
                transaction_id=("" if legacy_parse else expected_parse_generation),
            )
            semantic_cleanup["staged_validation"] = semantic_validation
            _mark_full_mineru_parse_ready(
                doc_id,
                expected_parse_generation=("" if legacy_parse else expected_parse_generation),
                expected_document_source_hash=("" if legacy_parse else expected_document_source_hash),
            )
            if had_previous_rag:
                transaction_journal = _write_rag_transaction_journal(
                    doc_id,
                    "committed",
                    source=previous_source,
                    manifest_path=str(backup_manifest.get("path") or ""),
                )
            _set_document_index_status(doc_id, "ready", stage="ready")
    except _SupersededParseGeneration:
        # A newer upload owns the document now. Never mark that new route as a
        # failed MinerU rebuild and never restore an older route over it.
        logger.info("[RagIndex] discard superseded MinerU publication for %s", doc_id)
        raise
    except Exception as exc:
        with _get_document_publication_lock(doc_id):
            still_owns_document = (
                _is_legacy_parse_manifest(
                    _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
                )
                if legacy_parse
                else _parse_generation_matches_current_document(
                    doc_id,
                    parse_generation=expected_parse_generation,
                    document_source_hash=expected_document_source_hash,
                )
            )
            if still_owns_document:
                if had_previous_rag and replaced_current_index and _load_complete_rag_backup_manifest(doc_id, previous_source):
                    try:
                        _restore_vector_index_backup(doc_id, previous_source)
                        transaction_journal = _write_rag_transaction_journal(
                            doc_id,
                            "rolled_back",
                            source=previous_source,
                            manifest_path=str((locals().get("backup_manifest") or {}).get("path") or ""),
                            error=str(exc),
                        )
                    except Exception as restore_exc:
                        logger.error(
                            "[RagIndex] failed to restore previous index after MinerU rebuild error for %s: %s",
                            doc_id,
                            restore_exc,
                        )
                _set_document_index_status(doc_id, "failed", stage="rebuilding_rag_index_failed", error="MinerU 问答索引重建失败，已保留原索引")
            else:
                logger.info("[RagIndex] skip rollback for superseded parse generation doc=%s", doc_id)
        raise exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(temp_semantic_dir, ignore_errors=True)

    return {
        "status": "ready",
        "message": "MinerU 问答索引已重建",
        "rag_index": _get_rag_index_status(doc_id),
        "quality_report": quality_report,
        "normalized": {
            "page_count": len(pages),
            "full_text_chars": len(full_text),
            "estimated_embedding_tokens": _estimate_text_tokens(full_text),
            "estimated_chunk_count": _estimate_chunk_count(full_text),
            "structured_table_count": len(structured_table_bundles),
            "parse_artifact": normalized["parse_artifact"],
        },
        "backup": {
            **backup,
            "manifest": backup_manifest if 'backup_manifest' in locals() else {},
            "transaction": transaction_journal if 'transaction_journal' in locals() else {},
            "document": doc_backup,
            "semantic_groups": semantic_backup,
            "semantic_group_cleanup": semantic_cleanup,
        },
        "semantic_group_rebuild": {
            "queued": False,
            "status": semantic_validation.get("status"),
            "chunk_count": len(semantic_rebuild["chunks"]),
            "embedding_model": semantic_rebuild["embedding_model"],
        },
    }


def migrate_legacy_storage():
    """Move files from old backend/* paths to project root if needed."""
    migrations = [
        (LEGACY_BACKEND_DOCS_DIR, DOCS_DIR, "*.json"),
        (LEGACY_BACKEND_VECTOR_STORE_DIR, VECTOR_STORE_DIR, "*.index"),
        (LEGACY_BACKEND_VECTOR_STORE_DIR, VECTOR_STORE_DIR, "*.pkl"),
        (LEGACY_BACKEND_UPLOAD_DIR, UPLOAD_DIR, "*.pdf"),
        (LEGACY_PROJECT_UPLOAD_DIR, UPLOAD_DIR, "*.pdf"),
    ]

    for src_dir, dest_dir, pattern in migrations:
        if not src_dir.exists() or src_dir.resolve() == dest_dir.resolve():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src_file in src_dir.glob(pattern):
            dest_file = dest_dir / src_file.name
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)


def generate_doc_id(content: str | bytes | bytearray | memoryview) -> str:
    """Generate a stable document identity.

    PDFs use their original bytes so OCR/ODL/MinerU output can never make two
    unrelated scans share a document id. Text-only imports retain the legacy
    text hash for backwards-compatible storage paths.
    """
    if isinstance(content, memoryview):
        content = content.tobytes()
    if isinstance(content, (bytes, bytearray)):
        return hashlib.sha256(bytes(content)).hexdigest()
    return hashlib.md5(str(content).encode("utf-8")).hexdigest()


def _build_pending_mineru_document_data(pdf_bytes: bytes) -> dict:
    """Keep only display-safe PDF metadata until the selected MinerU route publishes."""
    total_pages = 0
    try:
        total_pages = len(PyPDF2.PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception as exc:
        logger.warning("[Upload] failed to read PDF page count before MinerU parse: %s", exc)
    return {
        "full_text": "",
        "pages": [],
        "total_pages": total_pages,
        "images": [],
        "figures": [],
        "image_count": 0,
        "ocr_used": False,
        "ocr_backend": None,
        "extraction_quality": "pending_mineru",
        "extraction_method": "mineru_pending",
        "structured_table_bundles": [],
        "structured_table_count": 0,
    }


def _build_upload_parse_manifest(
    doc_id: str,
    *,
    parse_route: str,
    resolved_route: str | None = None,
    pdf_bytes: bytes,
    status: str,
    stage: str,
    metadata: dict | None = None,
) -> dict:
    selected_route = resolved_route or (parse_route if parse_route != PARSE_ROUTE_AUTO else PARSE_ROUTE_LOCAL)
    return build_parse_manifest(
        doc_id=doc_id,
        route=parse_route,
        resolved_route=selected_route,
        source_hash=derive_source_hash(pdf_bytes),
        status=status,
        stage=stage,
        metadata=metadata or {},
    )


def _start_mineru_full_route_upload(
    *,
    doc_id: str,
    filename: str,
    pdf_bytes: bytes,
    requested_route: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_api_host: Optional[str],
    summary_api_key: Optional[str],
    auto_selected: bool = False,
) -> dict:
    """Persist a pending MinerU-first document and queue its atomic publication."""
    if not _mineru_configured():
        raise HTTPException(
            status_code=400,
            detail="已选择 MinerU 全程解析，但 MinerU 尚未配置或不可用",
        )

    pending_data = _build_pending_mineru_document_data(pdf_bytes)
    pending_data["parse_manifest"] = _build_upload_parse_manifest(
        doc_id,
        parse_route=requested_route,
        resolved_route=PARSE_ROUTE_MINERU,
        pdf_bytes=pdf_bytes,
        status=PARSE_STATUS_QUEUED,
        stage="queued_mineru",
        metadata={
            "full_route": True,
            "auto_selected": bool(auto_selected),
            "text_source": "pending",
            "block_source": "pending",
            "rag_source": "pending",
            "figure_source": "pending",
        },
    )
    pdf_filename = f"{doc_id}.pdf"
    pdf_path = UPLOAD_DIR / pdf_filename
    with open(pdf_path, "wb") as handle:
        handle.write(pdf_bytes)

    # Replacing a same-byte PDF creates a new parse generation. Keep that
    # replacement atomic with old MinerU publication so a late worker cannot
    # write its blocks/RAG/document data into this pending route.
    with _get_document_publication_lock(doc_id):
        _retire_superseded_mineru_job(doc_id)
        _clear_block_dependent_ai_cache(doc_id, documents_store.get(doc_id))
        _remove_current_semantic_groups(doc_id)
        documents_store[doc_id] = {
            "filename": filename,
            "upload_time": datetime.now().isoformat(),
            "data": pending_data,
            "pdf_url": f"/uploads/{pdf_filename}",
        }
        _normalize_page_keys(documents_store[doc_id])
        save_document(doc_id, documents_store[doc_id])
        _set_document_index_status(doc_id, "queued", stage="waiting_for_mineru")
    deep_status = _queue_mineru_deep_parse(
        doc_id,
        full_route_options={
            "embedding_model": embedding_model,
            "embedding_api_key": embedding_api_key,
            "embedding_api_host": embedding_api_host,
            "summary_api_key": summary_api_key,
        },
    )
    return {
        "message": "PDF 已上传，正在执行 MinerU 全程解析",
        "doc_id": doc_id,
        "filename": filename,
        "total_pages": pending_data["total_pages"],
        "total_chars": 0,
        "image_count": 0,
        "pdf_url": f"/uploads/{pdf_filename}",
        "ocr_used": False,
        "ocr_backend": None,
        "extraction_quality": "pending_mineru",
        "extraction_method": "mineru_pending",
        "parse_manifest": pending_data["parse_manifest"],
        "deep_parse": deep_status,
        "indexing_status": "waiting_for_mineru",
    }


def _ocr_result_has_success(ocr_result) -> bool:
    pages = getattr(ocr_result, "pages", None)
    return any(getattr(page, "success", False) for page in pages or [])


def extract_text_from_pdf(
    pdf_file,
    pdf_bytes: Optional[bytes] = None,
    enable_ocr: str = "auto",
    ocr_backend: str = "auto",
    extract_images: bool = True,
    ocr_dpi: int = 200,
    ocr_language: str = "chi_sim+eng",
    ocr_quality_threshold: int = 60,
):
    """
    从 PDF 中提取文本和图片，支持可选的 OCR 回退
    参考 paper-burner-x 实现，支持多栏检测、图片提取、分批处理、智能段落合并
    
    Features:
    - P0: 多栏检测 (detect_columns) - 双栏论文支持
    - P0: 逐页质量评估 (assess_page_quality) - 按页决定是否OCR
    - P0: 图片提取与过滤 - 跳过装饰图标，保留有意义的图片
    - P1: 分批处理大文档 - 每50页一批，避免内存溢出
    - P1: 自适应阈值 - 基于中位数字符高度/宽度
    - P1: 保守的垃圾过滤 - 白名单保护公式/引用
    - P2: 智能段落合并 - 根据句号、大写、列表标记判断换段
    - P2: 元数据保留 - page, block_id, bbox, source, quality_score
    
    Args:
        pdf_file: pdfplumber 使用的文件对象
        pdf_bytes: PDF 原始字节（OCR 需要）
        enable_ocr: OCR 模式 - "auto"（自动检测）、"always"（始终启用）或 "never"（禁用）
        ocr_backend: OCR 后端 - "auto"、"tesseract"、"paddleocr"、"mistral"、"mineru" 或 "doc2x"
        extract_images: 是否从 PDF 中提取图片
        ocr_dpi: OCR 图像转换分辨率（DPI），默认 200
        ocr_language: OCR 语言设置（Tesseract 语言代码），默认 "chi_sim+eng"
        ocr_quality_threshold: 页面质量阈值（0-100），低于此值触发 OCR，默认 60
    
    Returns:
        包含 full_text、pages、total_pages、images 和 OCR 元数据的字典
    """
    import re
    import base64
    import time
    from statistics import median
    
    # ==================== 配置常量 ====================
    BATCH_SIZE = 50  # 每批处理页数
    BATCH_SLEEP = 0.3  # 批间休息时间(秒)
    
    # 图片过滤配置
    MIN_IMAGE_SIZE = 50  # 提高到50px，过滤更多小图标
    MAX_ASPECT_RATIO = 10  # 降低到10，过滤长条形图片
    MIN_ASPECT_RATIO = 0.1  # 提高到0.1
    MAX_IMAGE_DIMENSION = 800  # 图片最大尺寸，超过会压缩
    IMAGE_QUALITY = 75  # JPEG压缩质量
    
    # ==================== 白名单模式 ====================
    # 保护公式、引用、特殊格式不被误判为乱码
    WHITELIST_PATTERNS = [
        r'^\s*\[\d+\]',           # 引用 [1], [23]
        r'^\s*\(\d+\)',           # 引用 (1), (23)
        r'^\s*Fig\.\s*\d+',       # Figure 引用
        r'^\s*Table\s*\d+',       # Table 引用
        r'^\s*Eq\.\s*\d+',        # Equation 引用
        r'^\s*§\s*\d+',           # Section 符号
        r'[α-ωΑ-Ω∑∏∫∂∇±×÷≤≥≠≈∞∈∉⊂⊃∪∩]',  # 数学/希腊符号
        r'\$.*\$',               # LaTeX 行内公式
        r'\\[a-zA-Z]+',          # LaTeX 命令
        r'^\s*\d+\.\s+',         # 编号列表 1. 2. 3.
        r'^\s*[a-z]\)\s+',       # 编号列表 a) b) c)
        r'^\s*•\s+',             # 项目符号
        r'^\s*-\s+',             # 破折号列表
        r'https?://',            # URL
        r'[a-zA-Z0-9._%+-]+@',   # Email
    ]
    
    def extract_text_from_dict(text_dict: dict) -> str:
        """
        从 PyMuPDF 的 dict 格式中提取文本
        参考 paper-burner-x 的 _extractTextFromPage 实现
        
        核心逻辑：
        1. 遍历所有文本项（字符/单词）
        2. 根据 Y 坐标变化检测换行
        3. 根据 X 坐标间距决定是否添加空格
        """
        if not text_dict or "blocks" not in text_dict:
            return ""
        
        text_items = []
        
        # 遍历所有块
        for block in text_dict["blocks"]:
            if block.get("type") != 0:  # 0 = text block
                continue
            
            # 遍历块中的所有行
            for line in block.get("lines", []):
                # 遍历行中的所有 span
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                    
                    # 获取位置信息
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    x0, y0, x1, y1 = bbox
                    
                    text_items.append({
                        "text": text,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "width": x1 - x0
                    })
        
        if not text_items:
            return ""
        
        # 按 Y 坐标排序（从上到下），然后按 X 坐标排序（从左到右）
        text_items.sort(key=lambda item: (round(item["y0"] / 5) * 5, item["x0"]))
        
        # 重建文本
        result = ""
        last_y = None
        last_x_end = None
        
        for item in text_items:
            text = item["text"]
            y = item["y0"]
            x_start = item["x0"]
            x_end = item["x1"]
            
            # 检测换行（Y 坐标变化超过阈值）
            if last_y is not None and abs(y - last_y) > 5:
                result += '\n'
                last_x_end = None
            
            # 检测是否需要添加空格（X 坐标间距）
            if last_x_end is not None:
                # 估算空格宽度为字符宽度的 30%
                space_width = item["width"] * 0.3 if item["width"] > 0 else 3
                gap = x_start - last_x_end
                
                if gap > space_width:
                    result += ' '
            
            result += text
            last_y = y
            last_x_end = x_end
        
        return result.strip()
    
    def clean_text(text: str) -> str:
        """保守清理文本，只移除真正的乱码字符"""
        if not text:
            return ""
        # 只移除 NULL 字符和真正的控制字符，保留换行/制表
        cleaned = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\t\n\r')
        # 移除连续的替换字符
        cleaned = re.sub(r'[\ufffd]{2,}', '', cleaned)
        return cleaned
    
    def matches_whitelist(line: str) -> bool:
        """检查是否匹配白名单模式"""
        for pattern in WHITELIST_PATTERNS:
            if re.search(pattern, line):
                return True
        return False
    
    def is_garbage_line(line: str) -> bool:
        """保守的乱码检测，白名单优先"""
        if not line or len(line) < 2:
            return False
        
        # 白名单保护
        if matches_whitelist(line):
            return False
        
        # 统计不可打印字符
        bad_chars = sum(1 for ch in line if ord(ch) < 32 and ch not in '\t\n\r')
        # 统计替换字符和私用区字符
        weird_chars = sum(1 for ch in line if ch == '\ufffd' or 0xE000 <= ord(ch) <= 0xF8FF)
        # NULL 字符
        null_chars = line.count('\u0000')
        
        total_bad = bad_chars + weird_chars + null_chars
        # 提高阈值，更保守
        return total_bad / len(line) > 0.3
    
    def get_adaptive_thresholds(blocks: list) -> dict:
        """基于中位数计算自适应阈值"""
        if not blocks:
            return {"line_height": 12, "char_width": 8, "column_gap": 50}
        
        heights = []
        widths = []
        for block in blocks:
            if len(block) >= 7 and block[6] == 0:  # 文本块
                h = block[3] - block[1]  # y1 - y0
                w = block[2] - block[0]  # x1 - x0
                if h > 0:
                    heights.append(h)
                if w > 0:
                    widths.append(w)
        
        med_height = median(heights) if heights else 12
        med_width = median(widths) if widths else 100
        
        return {
            "line_height": med_height,
            "char_width": med_width / 10 if med_width > 0 else 8,
            "column_gap": med_width * 0.3,  # 栏间距约为块宽度的30%
            "line_tolerance": med_height * 0.5  # 同行容差
        }
    
    def detect_columns(blocks: list, page_width: float) -> list:
        """检测多栏布局，返回栏边界列表"""
        if not blocks or page_width <= 0:
            return [(0, page_width)]
        
        # 收集所有文本块的X坐标
        x_positions = []
        for block in blocks:
            if len(block) >= 7 and block[6] == 0:
                x_positions.append(block[0])  # x0
                x_positions.append(block[2])  # x1
        
        if not x_positions:
            return [(0, page_width)]
        
        # 分析X坐标分布，寻找明显的间隙
        x_positions.sort()
        
        # 计算相邻X坐标的间隙
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > page_width * 0.1:  # 间隙超过页宽10%
                gaps.append((x_positions[i-1], x_positions[i], gap))
        
        # 如果有明显的中间间隙，判定为双栏
        mid_point = page_width / 2
        for left, right, gap in gaps:
            if abs((left + right) / 2 - mid_point) < page_width * 0.15:
                # 间隙在页面中间附近
                return [(0, left + gap * 0.1), (right - gap * 0.1, page_width)]
        
        return [(0, page_width)]
    
    def sort_blocks_by_columns(blocks: list, columns: list, thresholds: dict) -> list:
        """按栏排序文本块：先按栏，栏内按Y再按X"""
        if not blocks:
            return []
        
        def get_column_index(block):
            x_center = (block[0] + block[2]) / 2
            for i, (col_left, col_right) in enumerate(columns):
                if col_left <= x_center <= col_right:
                    return i
            return 0
        
        # 为每个块添加栏索引
        blocks_with_col = [(block, get_column_index(block)) for block in blocks]
        
        # 排序：栏索引 -> Y坐标 -> X坐标
        line_tol = thresholds.get("line_tolerance", 6)
        sorted_blocks = sorted(
            blocks_with_col,
            key=lambda x: (x[1], round(x[0][1] / line_tol) * line_tol, x[0][0])
        )
        
        return [block for block, _ in sorted_blocks]
    
    def assess_page_quality(page_text: str, block_count: int, quality_threshold: int = 60) -> dict:
        """评估单页提取质量
        
        Args:
            page_text: 页面文本内容
            block_count: 文本块数量
            quality_threshold: 质量阈值（0-100），低于此值判定为需要 OCR
        """
        if not page_text:
            return {"score": 0, "needs_ocr": True, "reason": "empty_page"}
        
        text_len = len(page_text)
        
        # 计算各种指标
        null_ratio = page_text.count('\u0000') / text_len if text_len > 0 else 0
        weird_ratio = sum(1 for ch in page_text if ch == '\ufffd' or 0xE000 <= ord(ch) <= 0xF8FF) / text_len if text_len > 0 else 0
        
        # 有效字符比例
        valid_chars = sum(1 for ch in page_text if ch.isalnum() or ch in ' \t\n.,;:!?-()[]{}"\'' or '\u4e00' <= ch <= '\u9fff')
        valid_ratio = valid_chars / text_len if text_len > 0 else 0
        
        # 计算质量分数 (0-100)
        score = 100
        score -= null_ratio * 200
        score -= weird_ratio * 150
        score -= (1 - valid_ratio) * 50
        
        # 文本密度检查
        if block_count > 0 and text_len / block_count < 10:
            score -= 20
        
        score = max(0, min(100, score))
        
        needs_ocr = score < quality_threshold
        reason = "good" if score >= 80 else ("acceptable" if score >= quality_threshold else "poor_quality")
        
        return {
            "score": round(score, 1),
            "needs_ocr": needs_ocr,
            "reason": reason,
            "null_ratio": round(null_ratio, 3),
            "valid_ratio": round(valid_ratio, 3)
        }

    FIGURE_PATTERNS = [
        r'^图\s*(\d+)([a-zA-Z]?)',
        r'^Figure\s+(\d+)([a-zA-Z]?)',
        r'^Fig\.?\s+(\d+)([a-zA-Z]?)',
    ]

    FIGURE_CAPTION_PATTERNS = [
        r'(图\s*\d+[a-zA-Z]?)',
        r'(Figure\s+\d+[a-zA-Z]?)',
        r'(Fig\.?\s+\d+[a-zA-Z]?)',
    ]

    def _parse_figure_number(figure_num: str) -> tuple:
        """
        解析 figure 编号，返回 (base_number, sub_id)
        例如:
            "1" -> ("1", None)
            "1a" -> ("1", "a")
            "1A" -> ("1", "A")
        """
        if not figure_num:
            return ("", None)

        import re
        # 支持 "1a", "1A", "1.1" 等格式
        match = re.match(r'^(\d+)([a-zA-Z]?)$', figure_num.strip())
        if match:
            base = match.group(1)
            sub = match.group(2) if match.group(2) else None
            if sub:
                sub = sub.lower()
            return (base, sub)

        # 尝试直接解析纯数字
        try:
            return (str(int(figure_num)), None)
        except (ValueError, TypeError):
            return (figure_num, None)

    def _extract_figure_captions_from_dict(
        text_dict: dict,
        page_num: int,
        page_width: float = 0,
        page_height: float = 0,
    ) -> list:
        """从 PyMuPDF 的 dict 格式中检测 figure 标题"""
        if not text_dict or "blocks" not in text_dict:
            return []

        import re
        figures = []

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_text = ""
                line_bbox = None

                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text:
                        line_text += text
                        if line_bbox is None:
                            line_bbox = span.get("bbox")
                        else:
                            cur_bbox = span.get("bbox", [0, 0, 0, 0])
                            line_bbox = [
                                min(line_bbox[0], cur_bbox[0]),
                                min(line_bbox[1], cur_bbox[1]),
                                max(line_bbox[2], cur_bbox[2]),
                                max(line_bbox[3], cur_bbox[3])
                            ]

                line_text = line_text.strip()
                if not line_text:
                    continue

                for pattern in FIGURE_PATTERNS:
                    match = re.match(pattern, line_text, re.IGNORECASE)
                    if match:
                        # 解析 base_number 和 sub_id
                        raw_num = match.group(1)
                        sub_id_raw = match.group(2) if match.group(2) else ""
                        base_number, sub_id = _parse_figure_number(raw_num + sub_id_raw)

                        # 构建 display_label
                        if sub_id:
                            display_label = f"Figure {base_number}{sub_id}"
                        else:
                            display_label = f"Figure {base_number}"

                        figures.append({
                            "figure_number": base_number,  # 主编号，用于分组
                            "raw_number": raw_num + sub_id_raw,  # 原始编号，如 "1a"
                            "base_number": base_number,  # 主编号 "1"
                            "sub_id": sub_id,  # 子图标识 "a" or None
                            "display_label": display_label,
                            "label": line_text[:50],
                            "caption": line_text[:100],  # 保存完整caption
                            "page": page_num,
                            "bbox": line_bbox or [0, 0, 0, 0],
                            "caption_bbox": line_bbox or [0, 0, 0, 0],
                            "page_width": page_width,
                            "page_height": page_height,
                        })
                        break

        return figures

    def _normalize_bbox(bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    def _merge_bboxes(bboxes):
        valid = [_normalize_bbox(b) for b in bboxes]
        valid = [b for b in valid if b]
        if not valid:
            return None
        return [
            min(b[0] for b in valid),
            min(b[1] for b in valid),
            max(b[2] for b in valid),
            max(b[3] for b in valid),
        ]

    def _expand_bbox(bbox, page_width, page_height, x_ratio=0.04, y_ratio=0.03):
        normalized = _normalize_bbox(bbox)
        if not normalized:
            return None
        x0, y0, x1, y1 = normalized
        x_pad = max(12.0, page_width * x_ratio)
        y_pad = max(10.0, page_height * y_ratio)
        return [
            max(0.0, x0 - x_pad),
            max(0.0, y0 - y_pad),
            min(page_width, x1 + x_pad),
            min(page_height, y1 + y_pad),
        ]

    def _build_band_bbox(page_width, page_height, upper_bound, lower_bound):
        if page_width <= 0 or page_height <= 0:
            return None
        y0 = max(0.0, upper_bound)
        y1 = min(page_height, lower_bound)
        if y1 <= y0:
            return None
        min_height = min(page_height, max(page_height * 0.16, 120.0))
        if y1 - y0 < min_height:
            y0 = max(0.0, y1 - min_height)
        return [
            max(0.0, page_width * 0.04),
            y0,
            min(page_width, page_width * 0.96),
            y1,
        ]

    def _group_figures_by_base_number(figures: list) -> list:
        """
        将 figures 按 (page, base_number) 分组
        支持 synthetic parent：当只有子图(1a,1b)没有主图(1)时，自动生成 group

        返回: list of FigureGroup dicts
        """
        from collections import defaultdict

        # 按 (page, base_number) 分组
        groups = defaultdict(lambda: {
            "sub_figures": [],
            "has_parent": False,
            "parent_figure": None
        })

        for fig in figures:
            base = fig.get("base_number", "")
            page = fig.get("page", 1)
            key = (page, base)

            if fig.get("sub_id"):  # 子图，如 "1a" 中的 "a"
                groups[key]["sub_figures"].append(fig)
            else:  # 主图，如 "Figure 1"
                groups[key]["has_parent"] = True
                groups[key]["parent_figure"] = fig

        # 构建最终 group 列表
        result = []
        for (page, base), data in sorted(groups.items()):
            sub_figures = data["sub_figures"]

            if data["has_parent"]:
                # 有主图
                parent = data["parent_figure"]
                group = {
                    "group_id": f"fig-{base}",
                    "base_number": base,
                    "page": page,
                    "caption": parent.get("caption"),
                    "label": parent.get("display_label", f"Figure {base}"),
                    "sub_figures": sub_figures,
                    "is_synthetic": False,
                }
            else:
                # 无显式主图，synthetic parent
                # 合并所有子图的 caption
                merged_caption = "; ".join(
                    s.get("caption", "") for s in sub_figures if s.get("caption")
                )
                # 构建显示标签
                if sub_figures:
                    labels = [s.get("display_label", "") for s in sub_figures]
                    display_label = ", ".join(filter(None, labels))
                else:
                    display_label = f"Figure {base}"

                group = {
                    "group_id": f"fig-{base}",
                    "base_number": base,
                    "page": page,
                    "caption": merged_caption if merged_caption else None,
                    "label": display_label,
                    "sub_figures": sub_figures,
                    "is_synthetic": True,
                }
            result.append(group)

        return result

    def _match_figures_with_images(figures: list, images: list) -> list:
        """
        将 figure 标题与同页的图片进行空间匹配，返回结构化的 figures 列表。

        改进版：采用 group-first 策略
        1. 先将 figures 按 base_number 分组（支持子图 1a,1b 合并到主图 1）
        2. 按 group 匹配图片
        3. 返回增强的 figure 数据（包含 sub_figures, is_synthetic 等）
        """
        if not figures:
            return []

        # Step 1: 将 figures 按 base_number 分组
        figure_groups = _group_figures_by_base_number(figures)

        result = []
        page_to_images = {}
        for img in images:
            p = img.get("page", 1)
            if p not in page_to_images:
                page_to_images[p] = []
            page_to_images[p].append(img)

        for page_num in page_to_images:
            page_to_images[page_num] = sorted(
                page_to_images[page_num],
                key=lambda img: (
                    (_normalize_bbox(img.get("bbox")) or [0, 0, 0, 0])[1],
                    (_normalize_bbox(img.get("bbox")) or [0, 0, 0, 0])[0],
                    img.get("id", ""),
                )
            )

        # region agent log: debug figure-image matching
        debug_entries = []
        # endregion agent log

        # Step 2: 按 group 处理
        for group in figure_groups:
            page = group["page"]
            base_number = group["base_number"]
            page_images = page_to_images.get(page, [])

            # 用于跟踪本页面已经分配给前面 group 的图片，避免重复分配
            already_matched_image_ids = set()

            # 获取该页所有 figure（包括子图）对应的 caption bboxes
            all_captions = []

            # 主图的 caption
            if group.get("caption"):
                # 从原始 figures 中找对应的 caption bbox
                for fig in figures:
                    if fig.get("page") == page and fig.get("base_number") == base_number and not fig.get("sub_id"):
                        all_captions.append(fig)
                        break

            # 子图的 captions
            for sf in group.get("sub_figures", []):
                if sf.get("page") == page:
                    all_captions.append(sf)

            # 按 caption 位置排序
            all_captions = sorted(
                all_captions,
                key=lambda f: (_normalize_bbox(f.get("caption_bbox") or f.get("bbox")) or [0, 0, 0, 0])[1]
            )

            # 遍历每个 caption，收集匹配的 image_ids
            group_image_ids = []
            group_bboxes = []

            for idx, caption_fig in enumerate(all_captions):
                fig_bbox = _normalize_bbox(caption_fig.get("caption_bbox") or caption_fig.get("bbox")) or [0, 0, 0, 0]
                caption_top = fig_bbox[1]
                caption_bottom = fig_bbox[3]
                page_width = caption_fig.get("page_width", 0) or 612
                page_height = caption_fig.get("page_height", 0) or 792

                # 确定搜索窗口
                prev_bottom = 0.0
                if idx > 0:
                    prev_bbox = _normalize_bbox(all_captions[idx - 1].get("caption_bbox") or all_captions[idx - 1].get("bbox"))
                    if prev_bbox:
                        prev_bottom = prev_bbox[3]

                band_top = max(0.0, prev_bottom + 6.0)
                band_bottom = max(band_top + 1.0, caption_top - 4.0)

                matched_images = []
                matched_bboxes = []

                for img in page_images:
                    img_id = img.get("id")
                    if img_id in already_matched_image_ids:
                        continue

                    img_bbox = _normalize_bbox(img.get("bbox")) or [0, 0, 0, 0]
                    img_y0 = img_bbox[1]
                    img_y1 = img_bbox[3]
                    img_center_y = (img_y0 + img_y1) / 2 if img_y1 > img_y0 else img_y0

                    in_window = (
                        img_center_y >= band_top and
                        img_center_y <= caption_bottom + 8.0 and
                        img_y0 <= caption_bottom + 24.0
                    )
                    if in_window:
                        matched_images.append(img_id)
                        matched_bboxes.append(img_bbox)

                    already_matched_image_ids.update(matched_images)

                group_image_ids.extend(matched_images)
                group_bboxes.extend(matched_bboxes)

            # 去重 image_ids
            unique_image_ids = list(dict.fromkeys(group_image_ids))

            # 合并 group_bboxes 生成 group_bbox
            group_bbox = _merge_bboxes(group_bboxes)
            if group_bbox:
                group_bbox = _expand_bbox(group_bbox, page_width or 612, page_height or 792, x_ratio=0.05, y_ratio=0.04)

            # 获取 caption bbox（使用第一个 caption 的位置）
            primary_caption_bbox = None
            if all_captions:
                primary_caption_bbox = all_captions[0].get("caption_bbox") or all_captions[0].get("bbox")

            # 构建返回结果
            result.append({
                "figure_id": group["group_id"],
                "number": base_number,
                "label": group.get("label", f"Figure {base_number}"),
                "caption": group.get("caption"),
                "page": page,
                "image_ids": unique_image_ids,
                "group_bbox": group_bbox,  # 联合 bbox
                "caption_bbox": primary_caption_bbox,
                "sub_figures": group.get("sub_figures", []),
                "is_synthetic": group.get("is_synthetic", False),
                "page_width": page_width or 612,
                "page_height": page_height or 792,
            })

            # region agent log
            try:
                debug_entries.append({
                    "group_id": group["group_id"],
                    "base_number": base_number,
                    "page": page,
                    "is_synthetic": group.get("is_synthetic", False),
                    "sub_figures_count": len(group.get("sub_figures", [])),
                    "matched_image_ids": unique_image_ids,
                    "group_bbox": group_bbox,
                })
            except Exception:
                pass
            # endregion agent log

        # region agent log: write debug log to NDJSON file
        if debug_entries:
            try:
                import json as _json
                import time as _time

                log_record = {
                    "id": f"log_{int(_time.time() * 1000)}",
                    "timestamp": int(_time.time() * 1000),
                    "location": "routes/document_routes.py:_match_figures_with_images",
                    "message": "figure-group matching debug (enhanced)",
                    "data": {
                        "figures_count": len(figures),
                        "images_count": len(images),
                        "groups_count": len(figure_groups),
                        "entries": debug_entries,
                    },
                    "runId": "initial",
                    "hypothesisId": "H1-H3",
                }

                with open(r"e:\Project\.cursor\debug.log", "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(log_record, ensure_ascii=False) + "\n")
            except Exception:
                pass
        # endregion agent log

        return result

    def extract_with_pymupdf(pdf_bytes: bytes, extract_images: bool = True) -> tuple:
        """
        使用 PyMuPDF 进行字符级文本提取，参考 paper-burner-x 实现
        核心改进：
        1. 使用 get_text("dict") 获取字符级坐标
        2. 按 Y 坐标检测换行，按 X 坐标间距添加空格
        3. 精确控制文本重建，避免空格丢失
        4. 检测 figure 标题（图1 / Figure 1 等）并与图片关联
        返回: (pages, full_text, page_qualities, all_images, figures, error)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None, None, None, [], [], "PyMuPDF not installed"
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        full_text_parts = []
        page_qualities = []
        all_images = []  # 存储所有提取的图片
        all_figures = []  # 存储所有检测到的 figure 标题
        
        total_pages = len(doc)
        total_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.debug("[PDF] Processing %s pages in %s batches", total_pages, total_batches)
        
        for batch_idx in range(total_batches):
            start_page = batch_idx * BATCH_SIZE
            end_page = min((batch_idx + 1) * BATCH_SIZE, total_pages)
            
            logger.debug(
                "[PDF] Batch %s/%s: pages %s-%s",
                batch_idx + 1,
                total_batches,
                start_page + 1,
                end_page,
            )
            
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height
                
                # ==================== 字符级文本提取（参考 paper-burner-x）====================
                # 使用 get_text("dict") 获取详细的文本结构
                text_dict = {"blocks": []}
                try:
                    text_dict = page.get_text("dict")
                    page_text = extract_text_from_dict(text_dict)
                except Exception as dict_err:
                    # 如果 dict 模式失败，回退到简单的 text 模式
                    logger.debug(
                        "[PDF] Page %s dict extraction failed, fallback to text mode: %s",
                        page_num + 1,
                        dict_err,
                    )
                    page_text = page.get_text("text")
                
                # 清理文本
                page_text = clean_text(page_text)

                # ==================== 表格检测与 Markdown 注入 ====================
                page_table_bundles = []
                try:
                    from services.table_aware_service import (
                        bind_nearest_table_captions,
                        extract_table_caption_candidates_from_text_dict,
                        extract_tables_from_page,
                        inject_tables_into_text,
                    )
                    page_tables = extract_tables_from_page(page, page_text, page_num + 1)
                    if page_tables:
                        page_table_bundles = [
                            table.get("structured_bundle")
                            for table in page_tables
                            if isinstance(table, dict) and table.get("structured_bundle")
                        ]
                        caption_candidates = extract_table_caption_candidates_from_text_dict(text_dict, page_num + 1)
                        if caption_candidates and page_table_bundles:
                            bound_bundles = bind_nearest_table_captions(page_table_bundles, caption_candidates)
                            if len(bound_bundles) == len(page_table_bundles):
                                page_table_bundles = bound_bundles
                                bundle_iter = iter(bound_bundles)
                                for table in page_tables:
                                    if not isinstance(table, dict) or not table.get("structured_bundle"):
                                        continue
                                    bound = next(bundle_iter, None)
                                    if isinstance(bound, dict):
                                        table["structured_bundle"] = bound
                                        if bound.get("table_body_markdown"):
                                            table["markdown"] = bound["table_body_markdown"]
                        page_text = inject_tables_into_text(page_text, page_tables)
                except Exception as table_err:
                    pass  # 表格检测失败不影响主流程

                # ==================== Figure 标题检测 ====================
                page_figures = _extract_figure_captions_from_dict(
                    text_dict,
                    page_num + 1,
                    page_width,
                    page_height,
                )
                all_figures.extend(page_figures)

                # ==================== 图片提取 ====================
                page_images = []
                if extract_images:
                    try:
                        image_list = page.get_images(full=True)
                        for img_idx, img_info in enumerate(image_list):
                            try:
                                xref = img_info[0]
                                base_image = doc.extract_image(xref)
                                
                                if not base_image:
                                    continue
                                
                                img_width = base_image.get("width", 0)
                                img_height = base_image.get("height", 0)
                                
                                # 图片过滤
                                if img_width < MIN_IMAGE_SIZE or img_height < MIN_IMAGE_SIZE:
                                    continue  # 跳过装饰图标
                                
                                aspect_ratio = img_width / img_height if img_height > 0 else 0
                                if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                                    continue  # 跳过线条/分隔符
                                
                                # 获取图片数据
                                img_data = base_image.get("image")
                                img_ext = base_image.get("ext", "png")
                                
                                if img_data:
                                    # 压缩大图片
                                    if img_width > MAX_IMAGE_DIMENSION or img_height > MAX_IMAGE_DIMENSION:
                                        try:
                                            from PIL import Image
                                            import io as img_io
                                            
                                            img = Image.open(img_io.BytesIO(img_data))
                                            img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                                            
                                            buffer = img_io.BytesIO()
                                            if img.mode in ('RGBA', 'P'):
                                                img = img.convert('RGB')
                                            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
                                            img_data = buffer.getvalue()
                                            img_ext = "jpg"
                                        except Exception as resize_err:
                                            logger.debug("[PDF] Image resize failed: %s", resize_err)
                                    
                                    img_id = f"page{page_num + 1}_img{img_idx + 1}"
                                    img_base64 = base64.b64encode(img_data).decode('utf-8')

                                    img_bbox = None
                                    try:
                                        img_rects = page.get_image_rects(xref)
                                        if img_rects:
                                            rect = img_rects[0]
                                            img_bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                                    except Exception:
                                        pass

                                    page_images.append({
                                        "id": img_id,
                                        "data": f"data:image/{img_ext};base64,{img_base64}",
                                        "width": img_width,
                                        "height": img_height,
                                        "page": page_num + 1,
                                        "bbox": img_bbox
                                    })
                                    
                                    # 不在文本中插入图片引用，避免干扰RAG检索
                                    # 图片信息已经单独存储在 all_images 数组中
                                    
                            except Exception as img_err:
                                # 单个图片提取失败不影响整体
                                pass
                        
                        all_images.extend(page_images)
                        
                    except Exception as img_extract_err:
                        logger.debug(
                            "[PDF] Page %s image extraction failed: %s",
                            page_num + 1,
                            img_extract_err,
                        )
                
                # 评估页面质量（使用传入的质量阈值）
                quality = assess_page_quality(page_text, 1, ocr_quality_threshold)  # block_count设为1，因为我们不再使用blocks
                page_qualities.append(quality)
                
                page_payload = {
                    "page": page_num + 1,
                    "content": page_text,
                    "quality_score": quality["score"],
                    "image_count": len(page_images),
                    "source": "pymupdf_dict"
                }
                if page_table_bundles:
                    page_payload["table_bundles"] = page_table_bundles
                pages.append(page_payload)
                full_text_parts.append(page_text)
            
            # 批间休息，释放内存
            if batch_idx < total_batches - 1:
                time.sleep(BATCH_SLEEP)
        
        doc.close()

        # 基于图片位置匹配 figure 标题，生成 figures 元数据
        figures = _match_figures_with_images(all_figures, all_images)

        return pages, '\n\n'.join(full_text_parts), page_qualities, all_images, figures, None
    
    def extract_with_pdfplumber(pdf_file) -> tuple:
        """使用 pdfplumber 的 chars 进行坐标级文本提取，带自适应阈值"""
        pdf_file.seek(0)
        
        with pdfplumber.open(pdf_file) as pdf:
            pages = []
            full_text_parts = []
            page_qualities = []
            
            total_pages = len(pdf.pages)
            total_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(total_batches):
                start_page = batch_idx * BATCH_SIZE
                end_page = min((batch_idx + 1) * BATCH_SIZE, total_pages)
                
                for i in range(start_page, end_page):
                    page = pdf.pages[i]
                    chars = page.chars
                    page_width = page.width
                    
                    if not chars:
                        quality = {"score": 0, "needs_ocr": True, "reason": "no_chars"}
                        page_qualities.append(quality)
                        pages.append({
                            "page": i + 1,
                            "content": "",
                            "quality_score": 0,
                            "source": "pdfplumber"
                        })
                        continue
                    
                    # 计算自适应阈值
                    char_heights = [c.get('height', 10) for c in chars if c.get('height')]
                    char_widths = [c.get('width', 5) for c in chars if c.get('width')]
                    med_height = median(char_heights) if char_heights else 10
                    med_width = median(char_widths) if char_widths else 5
                    
                    line_tolerance = med_height * 0.4
                    space_threshold = med_width * 1.5
                    
                    # 按Y坐标分组，然后按X坐标排序
                    lines = {}
                    for char in chars:
                        if not char.get('text') or ord(char['text']) < 32:
                            continue
                        
                        y = round(char['top'] / line_tolerance) * line_tolerance
                        if y not in lines:
                            lines[y] = []
                        lines[y].append((char['x0'], char['text'], char.get('width', med_width)))
                    
                    # 按Y坐标排序，然后每行按X坐标排序
                    page_lines = []
                    for y in sorted(lines.keys()):
                        line_chars = sorted(lines[y], key=lambda c: c[0])
                        
                        # 智能添加空格
                        line_text = ""
                        last_x_end = None
                        for x, ch, w in line_chars:
                            if last_x_end is not None:
                                gap = x - last_x_end
                                if gap > space_threshold:
                                    line_text += " "
                            line_text += ch
                            last_x_end = x + w
                        
                        if line_text.strip() and not is_garbage_line(line_text):
                            page_lines.append(clean_text(line_text))
                    
                    page_text = '\n'.join(page_lines)
                    
                    # 评估质量（使用传入的质量阈值）
                    quality = assess_page_quality(page_text, len(set(c.get('block', 0) for c in chars)), ocr_quality_threshold)
                    page_qualities.append(quality)
                    
                    pages.append({
                        "page": i + 1,
                        "content": page_text,
                        "quality_score": quality["score"],
                        "source": "pdfplumber"
                    })
                    full_text_parts.append(page_text)
                
                # 批间休息
                if batch_idx < total_batches - 1:
                    time.sleep(BATCH_SLEEP)
        
        return pages, '\n\n'.join(full_text_parts), page_qualities, [], None
    
    def heuristic_rebuild(text: str, is_cjk: bool = False) -> str:
        """
        智能段落合并与启发式文本重建
        完全参考 paper-burner-x 的 _heuristicRebuild 实现
        """
        if not text:
            return ""
        
        rebuilt = text
        
        # 先保护图片引用，避免被文本处理规则破坏
        image_refs = []
        def save_image_ref(match):
            placeholder = f"__IMG_PLACEHOLDER_{len(image_refs)}__"
            image_refs.append(match.group(0))
            return placeholder
        rebuilt = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', save_image_ref, rebuilt)
        
        # 1. 修复被断开的单词（英文连字符换行）
        # 匹配：字母-空格-换行-小写字母 -> 字母字母
        rebuilt = re.sub(r'([a-zA-Z])-\s*\n\s*([a-z])', r'\1\2', rebuilt)
        
        # 2. 合并被打断的句子
        # 如果行尾不是句号等结束符，且下一行不是大写/数字/特殊字符开头，则合并
        rebuilt = re.sub(r'([^\n.!?。！？])\n([a-z\u4e00-\u9fff])', r'\1 \2', rebuilt)
        
        # 3. 修复中文标点符号周围的空格
        rebuilt = re.sub(r'\s+([，。！？；：、）】」』])', r'\1', rebuilt)
        rebuilt = re.sub(r'([（【「『])\s+', r'\1', rebuilt)
        
        # 4. 修复英文标点符号
        # 标点后应有空格（如果后面是字母），但要排除邮箱、网址、缩写等情况
        # 不处理 . 因为它可能是邮箱、网址、缩写
        rebuilt = re.sub(r'([,!?;:])([a-zA-Z])', r'\1 \2', rebuilt)
        # 移除标点前的多余空格
        rebuilt = re.sub(r'\s+([,.!?;:])', r'\1', rebuilt)
        
        # 5. 规范化空白字符
        # 多个空格变成一个
        rebuilt = re.sub(r' {2,}', ' ', rebuilt)
        # 保留段落分隔（最多2个换行）
        rebuilt = re.sub(r'\n{3,}', '\n\n', rebuilt)
        
        # 6. 修复常见的格式问题
        # 修复：数字. 后面应该有空格（列表项）
        rebuilt = re.sub(r'(\d+)\.\s*([a-zA-Z\u4e00-\u9fff])', r'\1. \2', rebuilt)
        # 修复：括号内不应有首尾空格
        rebuilt = re.sub(r'\(\s+', '(', rebuilt)
        rebuilt = re.sub(r'\s+\)', ')', rebuilt)
        
        # 7. 智能段落识别（参考 paper-burner-x）
        lines = rebuilt.split('\n')
        paragraphs = []
        current_para = ''
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line == '':
                if current_para:
                    paragraphs.append(current_para.strip())
                    current_para = ''
                continue
            
            # 判断是否应该换段
            should_break = (
                current_para == '' or  # 当前段落为空
                re.match(r'^#{1,6}\s', line) or  # 标题
                re.match(r'^[\-\*\+]\s', line) or  # 无序列表
                re.match(r'^\d+\.\s', line) or  # 有序列表
                line.startswith('__IMG_PLACEHOLDER_') or  # 图片占位符
                # 上一段以句号结束且本行首字母大写或中文
                (re.search(r'[.!?。！？]\s*$', current_para) and re.match(r'^[A-Z\u4e00-\u9fff]', line))
            )
            
            if should_break:
                if current_para:
                    paragraphs.append(current_para.strip())
                current_para = line
            else:
                # 合并到当前段落，总是加空格（因为我们已经在字符级提取时处理了空格）
                current_para += ' ' + line
        
        if current_para:
            paragraphs.append(current_para.strip())
        
        rebuilt = '\n\n'.join(paragraphs)
        
        # 恢复图片引用
        for idx, ref in enumerate(image_refs):
            rebuilt = rebuilt.replace(f"__IMG_PLACEHOLDER_{idx}__", ref)
        
        return rebuilt.strip()
    
    def detect_language(text: str) -> str:
        """检测文本主要语言"""
        if not text:
            return "en"
        cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        return "cjk" if cjk_count / len(text) > 0.1 else "en"
    
    # ==================== 主提取逻辑 ====================
    pages = None
    full_text = ""
    page_qualities = None
    all_images = []
    extraction_method = None
    err = "pdf_bytes not provided"
    
    # 优先使用 PyMuPDF
    figures = []
    if pdf_bytes:
        pages, full_text, page_qualities, all_images, figures, err = extract_with_pymupdf(pdf_bytes, extract_images)
        if pages is not None:
            extraction_method = "pymupdf"
            logger.info(
                "[PDF] Using PyMuPDF extraction, %s pages, %s images, %s figures",
                len(pages),
                len(all_images),
                len(figures),
            )

    # 如果 PyMuPDF 失败，回退到 pdfplumber
    if pages is None:
        logger.warning("[PDF] PyMuPDF failed (%s), falling back to pdfplumber", err)
        pages, full_text, page_qualities, all_images, err = extract_with_pdfplumber(pdf_file)
        extraction_method = "pdfplumber"
        figures = []  # pdfplumber 暂不提取 figures
    
    # 检测语言并应用启发式重建
    is_cjk = detect_language(full_text) == "cjk"
    full_text = heuristic_rebuild(full_text, is_cjk)
    for page in pages:
        page["content"] = heuristic_rebuild(page["content"], is_cjk)
    structured_table_bundles = [
        bundle
        for page in pages
        if isinstance(page, dict)
        for bundle in (page.get("table_bundles") or [])
        if isinstance(bundle, dict)
    ]
    try:
        from services.table_aware_service import merge_pdf_native_structured_table_bundles
        structured_table_bundles = merge_pdf_native_structured_table_bundles(structured_table_bundles)
    except Exception as merge_err:
        logger.debug("[PDF] pdf_native structured table merge skipped: %s", merge_err)
    structured_table_bundles = _clean_structured_table_bundles(structured_table_bundles)
    
    # 获取总页数
    pdf_file.seek(0)
    reader = PyPDF2.PdfReader(pdf_file)
    total_pages = len(reader.pages)
    
    # 计算整体质量分数
    avg_quality = sum(q["score"] for q in page_qualities) / len(page_qualities) if page_qualities else 50
    pages_needing_ocr = [i for i, q in enumerate(page_qualities) if q.get("needs_ocr")] if page_qualities else []
    
    result = {
        "full_text": full_text,
        "total_pages": total_pages,
        "pages": pages,
        "images": all_images,  # 新增：提取的图片列表
        "figures": figures,  # 新增：检测到的 figure 标题列表
        "image_count": len(all_images),
        "ocr_used": False,
        "ocr_backend": None,
        "extraction_quality": "good" if avg_quality >= 80 else ("acceptable" if avg_quality >= 60 else "poor"),
        "extraction_method": extraction_method,
        "avg_quality_score": round(avg_quality, 1),
        "pages_needing_ocr": pages_needing_ocr,
        "structured_table_bundles": structured_table_bundles,
        "structured_table_count": len(structured_table_bundles),
    }
    
    ocr_target_pages = select_ocr_target_pages(enable_ocr, total_pages, pages_needing_ocr)

    if not ocr_target_pages:
        logger.debug("[PDF] 无需执行 OCR 或 OCR 已禁用 (mode=%s, avg_quality=%.1f)", enable_ocr, avg_quality)
        return result
    
    # 通过注册表获取 OCR 适配器。优先使用本次上传请求的设置，缺省时回退到后端全局配置。
    selected_ocr_backend = (ocr_backend or settings.ocr_backend or "auto").strip().lower()
    legacy_structured_ocr_warning = ""
    if selected_ocr_backend in {"mineru", "doc2x"}:
        # These adapters currently return whole-document Markdown and cannot
        # safely populate only the failed pages. Keep their configuration for
        # deep parsing, but never use them in the page-level replacement path.
        legacy_structured_ocr_warning = (
            f"{selected_ocr_backend} 已从逐页 OCR 降级为本地自动 OCR；"
            "请使用深度解析入口获取结构化结果"
        )
        selected_ocr_backend = "auto"
    adapter = _ocr_registry.get_adapter(selected_ocr_backend)
    if adapter is None:
        logger.warning(
            "[PDF] 需要对 %s 页执行 OCR，但后端不可用: %s",
            len(ocr_target_pages),
            selected_ocr_backend,
        )
        result["ocr_error"] = f"OCR 后端不可用: {selected_ocr_backend}"
        result["ocr_warning"] = "；".join(part for part in (legacy_structured_ocr_warning, f"OCR 后端不可用: {selected_ocr_backend}") if part)
        return result
    
    if pdf_bytes is None:
        logger.warning("[PDF] 需要 OCR 但未提供 pdf_bytes")
        result["ocr_error"] = "无法执行 OCR：缺少 PDF 原始数据"
        result["ocr_warning"] = "无法执行 OCR：缺少 PDF 原始数据"
        return result
    
    # 使用适配器系统执行逐页 OCR
    logger.info("[PDF] 开始逐页 OCR，共 %s 页，后端: %s", len(ocr_target_pages), adapter.name)
    primary_outcome_recorded = False
    try:
        # 调用适配器的 ocr_pages()，仅传入需要 OCR 的页码列表
        ocr_result = adapter.ocr_pages(
            pdf_bytes=pdf_bytes,
            page_numbers=ocr_target_pages,
            dpi=ocr_dpi
        )
        if not _ocr_result_has_success(ocr_result):
            record_ocr_provider_use(adapter.name, outcome="failure", operation="page_ocr")
            primary_outcome_recorded = True
            raise RuntimeError("OCR 后端未返回任何成功页面")
        record_ocr_provider_use(adapter.name, outcome="success", operation="page_ocr")
        primary_outcome_recorded = True
        
        apply_ocr_result_to_pages(
            result,
            pages,
            ocr_result,
            ocr_target_pages,
            heuristic_rebuild,
            is_cjk=is_cjk,
        )
        if legacy_structured_ocr_warning:
            current_warning = str(result.get("ocr_warning") or "").strip()
            result["ocr_warning"] = "；".join(part for part in (legacy_structured_ocr_warning, current_warning) if part)
        
        # 处理部分页面 OCR 失败的警告信息
        if ocr_result.failed_pages:
            failed_info = ", ".join(str(p) for p in ocr_result.failed_pages)
            logger.warning("[PDF] OCR 警告: 部分页面 OCR 失败（页码: %s）", failed_info)
        
        logger.info(
            "[PDF] OCR 完成。已使用: %s，目标页面: %s，后端: %s",
            result["ocr_used"],
            ocr_target_pages,
            ocr_result.backend,
        )
        
        # 存储 MinerU 版面分析提取的 figure 数据（供速览 Figure Pipeline 使用）
        if hasattr(ocr_result, 'layout_figures') and ocr_result.layout_figures:
            logger.info("[PDF] MinerU 版面分析: 存储 %s 个 figure 区域", len(ocr_result.layout_figures))
        
    except Exception as e:
        if not primary_outcome_recorded:
            record_ocr_provider_use(adapter.name, outcome="failure", operation="page_ocr")
            primary_outcome_recorded = True
        # 在线 OCR 失败时，尝试回退到本地 OCR 引擎
        if adapter.name in _ocr_registry._ONLINE_ADAPTERS:
            logger.warning("[PDF] 在线 OCR (%s) 失败，尝试回退到本地引擎: %s", adapter.name, e)
            local_adapter = _ocr_registry.get_local_adapter(exclude=[adapter.name])
            if local_adapter is not None:
                fallback_outcome_recorded = False
                try:
                    logger.info("[PDF] 回退到本地 OCR 引擎: %s", local_adapter.name)
                    ocr_result = local_adapter.ocr_pages(
                        pdf_bytes=pdf_bytes,
                        page_numbers=ocr_target_pages,
                        dpi=ocr_dpi
                    )
                    if not _ocr_result_has_success(ocr_result):
                        record_ocr_provider_use(
                            local_adapter.name,
                            outcome="failure",
                            operation="page_ocr",
                            fallback=True,
                        )
                        fallback_outcome_recorded = True
                        raise RuntimeError("本地 OCR 回退未返回任何成功页面")
                    record_ocr_provider_use(
                        local_adapter.name,
                        outcome="success",
                        operation="page_ocr",
                        fallback=True,
                    )
                    fallback_outcome_recorded = True

                    apply_ocr_result_to_pages(
                        result,
                        pages,
                        ocr_result,
                        ocr_target_pages,
                        heuristic_rebuild,
                        is_cjk=is_cjk,
                    )

                    if result.get("ocr_used"):
                        result["ocr_warning"] = (
                            f"在线 OCR ({adapter.name}) 失败，已回退到本地引擎 ({local_adapter.name})"
                        )
                        logger.info(
                            "[PDF] 在线 OCR 回退成功: %s -> %s",
                            adapter.name,
                            local_adapter.name,
                        )
                    else:
                        result["ocr_warning"] = (
                            f"在线 OCR ({adapter.name}) 失败，本地引擎 ({local_adapter.name}) "
                            "未产生可用 OCR 文本，已保留原始提取文本"
                        )
                        logger.warning(
                            "[PDF] 在线 OCR 回退未产生可用文本: %s -> %s",
                            adapter.name,
                            local_adapter.name,
                        )
                except Exception as fallback_err:
                    if not fallback_outcome_recorded:
                        record_ocr_provider_use(
                            local_adapter.name,
                            outcome="failure",
                            operation="page_ocr",
                            fallback=True,
                        )
                    logger.error("[PDF] 本地 OCR 回退也失败: %s", fallback_err)
                    result["ocr_error"] = str(e)
                    result["ocr_warning"] = (
                        f"在线 OCR ({adapter.name}) 和本地 OCR 回退均失败: {str(e)}"
                    )
            else:
                logger.warning("[PDF] 在线 OCR 失败且无可用的本地 OCR 引擎用于回退")
                result["ocr_error"] = str(e)
                result["ocr_warning"] = (
                    f"在线 OCR ({adapter.name}) 失败且无可用的本地 OCR 引擎: {str(e)}"
                )
        else:
            logger.warning("[PDF] OCR 失败: %s", e)
            result["ocr_error"] = str(e)
            result["ocr_warning"] = f"OCR 处理异常: {str(e)}"
    
    return result


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    embedding_model: str = Form("local-minilm"),
    embedding_api_key: Optional[str] = Form(None),
    embedding_api_host: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    enable_ocr: Optional[str] = Form(None),
    ocr_backend: Optional[str] = Form(None),
    parse_route: Optional[str] = Form(None),
):
    """
    上传并处理 PDF 文件
    
    Args:
        file: 要上传的 PDF 文件
        embedding_model: 文本嵌入模型
        embedding_api_key: 云端嵌入模型的 API 密钥
        embedding_api_host: 自定义 API 地址
        api_key: 语义意群摘要使用的 LLM API 密钥（可选，默认回退到 embedding_api_key）
        enable_ocr: OCR 模式 - "auto"（自动检测）、"always"（始终启用）或 "never"（禁用）。
                    缺失时使用后端配置中的 ocr_default_mode 默认值。
        ocr_backend: OCR 后端。缺失时使用后端配置中的 ocr_backend 默认值。
        parse_route: 主解析路线 - auto（本地优先）、local 或 mineru。
    """
    filename_lower = file.filename.lower()
    is_pdf = filename_lower.endswith('.pdf')
    is_multi_format = is_supported_format(file.filename)

    if not is_pdf and not is_multi_format:
        supported = "PDF, DOCX, XLSX, TXT, MD, CSV"
        raise HTTPException(status_code=400, detail=f"不支持的文件格式，支持: {supported}")

    try:
        content = await file.read()
        try:
            requested_parse_route = normalize_parse_route(
                parse_route,
                default=PARSE_ROUTE_AUTO,
                strict=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # 多格式文档处理（非 PDF）
        if is_multi_format and not is_pdf:
            if requested_parse_route == PARSE_ROUTE_MINERU:
                raise HTTPException(status_code=400, detail="MinerU 全程解析当前仅支持 PDF 文件")
            import tempfile
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                normalized_model = normalize_embedding_model_id(embedding_model)
                if not normalized_model:
                    raise HTTPException(status_code=400, detail=f"Embedding模型 '{embedding_model}' 未配置")
                embedding_model = normalized_model

                extracted_data = extract_from_file(tmp_path, file.filename)
                doc_id = generate_doc_id(extracted_data["full_text"])
                extracted_data["parse_manifest"] = build_parse_manifest(
                    doc_id=doc_id,
                    route=requested_parse_route,
                    resolved_route=PARSE_ROUTE_LOCAL,
                    source_hash=derive_source_hash(content),
                    status=PARSE_STATUS_READY,
                    stage="local_ready",
                    metadata={
                        "full_route": False,
                        "text_source": str(extracted_data.get("source_type") or "file"),
                        "block_source": "canonical_pages",
                        "rag_source": "pdf_native",
                        "figure_source": "none",
                    },
                )

                _clear_block_dependent_ai_cache(doc_id, documents_store.get(doc_id))
                _remove_current_semantic_groups(doc_id)
                documents_store[doc_id] = {
                    "filename": file.filename,
                    "upload_time": datetime.now().isoformat(),
                    "data": extracted_data,
                    "pdf_url": None,
                }
                save_document(doc_id, documents_store[doc_id])
                summary_api_key = ((api_key or "").strip() or (embedding_api_key or "").strip() or None)
                index_status = _queue_document_indexes(
                    doc_id,
                    embedding_model,
                    embedding_api_key,
                    embedding_api_host,
                    summary_api_key,
                )
                return {
                    "message": "文档上传成功",
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "total_pages": extracted_data["total_pages"],
                    "total_chars": len(extracted_data["full_text"]),
                    "source_type": extracted_data.get("source_type", "unknown"),
                    "indexing_status": index_status.get("status", "queued"),
                }
            finally:
                os.unlink(tmp_path)

        pdf_file = io.BytesIO(content)

        normalized_model = normalize_embedding_model_id(embedding_model)
        if not normalized_model:
            raise HTTPException(status_code=400, detail=f"Embedding模型 '{embedding_model}' 未配置或格式不正确（建议使用 provider:model 格式）")
        embedding_model = normalized_model

        # 桌面模式下本地模型不可用，提前拦截
        if runtime.is_desktop and ('local' in embedding_model.lower().split(':')[0] or embedding_model in ('local-minilm',)):
            raise HTTPException(
                status_code=400,
                detail="桌面版不支持本地 Embedding 模型，请在设置中选择远程 Embedding 服务（如 OpenAI、硅基流动等）并配置 API Key"
            )

        # PDF identity is derived from the original bytes before any parser can
        # alter text. This keeps re-parses in one document lineage while the
        # parse manifest distinguishes generations.
        doc_id = generate_doc_id(content)
        summary_api_key = ((api_key or "").strip() or (embedding_api_key or "").strip() or None)
        if requested_parse_route == PARSE_ROUTE_MINERU:
            return _start_mineru_full_route_upload(
                doc_id=doc_id,
                filename=file.filename,
                pdf_bytes=content,
                requested_route=requested_parse_route,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
                embedding_api_host=embedding_api_host,
                summary_api_key=summary_api_key,
            )

        # 当 enable_ocr 参数缺失时，回退到配置中的默认值
        ocr_mode = enable_ocr if enable_ocr is not None else settings.ocr_default_mode
        ocr_backend_name = ocr_backend if ocr_backend is not None else settings.ocr_backend

        # 使用配置中的 OCR 参数提取文本
        extracted_data = extract_text_from_pdf(
            pdf_file,
            pdf_bytes=content,
            enable_ocr=ocr_mode,
            ocr_backend=ocr_backend_name,
            ocr_dpi=settings.ocr_dpi,
            ocr_language=settings.ocr_language,
            ocr_quality_threshold=settings.ocr_quality_threshold,
        )

        # Page OCR has a different contract from MinerU/ODL document parsing.
        # Keep Mistral's page-level result in the common artifact namespace
        # before an optional ODL merge replaces the active page representation.
        page_ocr_artifact_ref = {}
        if extracted_data.get("ocr_used") and extracted_data.get("ocr_backend") == "mistral":
            mistral_pages = [
                dict(page) for page in (extracted_data.get("pages") or [])
                if isinstance(page, dict) and page.get("ocr_backend") == "mistral"
            ]
            if mistral_pages:
                artifact = build_document_parse_artifact(
                    doc_id=doc_id,
                    provider="mistral",
                    provider_version="mistral-ocr-latest",
                    pages=mistral_pages,
                    tables=[],
                    warnings=[str(extracted_data.get("ocr_warning") or "")],
                    capabilities={
                        "per_page_text": True,
                        "document_structure": False,
                        "structured_tables": False,
                        "table_geometry": False,
                        "figures": False,
                    },
                )
                artifact_path = persist_document_parse_artifact(DATA_DIR, artifact)
                page_ocr_artifact_ref = {
                    "schema_version": artifact["schema_version"],
                    "provider": artifact["provider"],
                    "source_hash": artifact["source_hash"],
                    "ref": artifact_reference(DATA_DIR, artifact_path),
                }

        pdf_filename = f"{doc_id}.pdf"
        pdf_path = UPLOAD_DIR / pdf_filename
        with open(pdf_path, "wb") as f:
            f.write(content)

        pdf_url = f"/uploads/{pdf_filename}"

        # ── ODL 去脏合并层 ──────────────────────────────────────────────────
        # PDF 已落盘，尝试用 OpenDataLoader 解析，得到过滤了 header/footer/
        # caption/image 脏块的干净文本；失败则保留 PyMuPDF 优先、pdfplumber
        # 回退以及可选逐页 OCR 已得到的结果。
        try:
            from services.odl_parser_service import parse_pdf_odl, is_odl_available
            if is_odl_available():
                odl_result = parse_pdf_odl(str(pdf_path))
                if odl_result:
                    cleaned_odl_pages = _clean_page_texts(odl_result.get("pages", []))
                    cleaned_odl_bundles = _clean_structured_table_bundles(odl_result.get("structured_table_bundles", []))
                    cleaned_odl_bundles = _backfill_bundle_evidence_units_from_pages(
                        cleaned_odl_bundles,
                        cleaned_odl_pages,
                    )
                    cleaned_odl_full_text = _clean_control_text(odl_result.get("full_text", ""))
                    if not cleaned_odl_full_text and cleaned_odl_pages:
                        cleaned_odl_full_text = "\n\n".join(
                            page.get("text", "") for page in cleaned_odl_pages if page.get("text", "").strip()
                        )
                    merged_pages, preserved_ocr = _merge_odl_pages_with_existing_ocr(
                        extracted_data.get("pages") or [],
                        cleaned_odl_pages,
                    )
                    merged_full_text = "\n\n".join(
                        str(page.get("content") or page.get("text") or "").strip()
                        for page in merged_pages
                        if str(page.get("content") or page.get("text") or "").strip()
                    )
                    extracted_data["full_text"] = merged_full_text or cleaned_odl_full_text or extracted_data.get("full_text", "")
                    extracted_data["pages"] = merged_pages or extracted_data.get("pages", [])
                    extracted_data["total_pages"] = odl_result["total_pages"]
                    extracted_data["extraction_method"] = "odl_hybrid" if preserved_ocr else "odl"
                    extracted_data["odl_preserved_ocr_pages"] = preserved_ocr
                    extracted_data["odl_element_count"] = odl_result.get("odl_element_count", 0)
                    extracted_data["odl_kept_count"] = odl_result.get("odl_kept_count", 0)
                    extracted_data["odl_soft_kept_caption_count"] = odl_result.get("odl_soft_kept_caption_count", 0)
                    extracted_data["structured_table_bundles"] = cleaned_odl_bundles
                    extracted_data["structured_table_count"] = odl_result.get("structured_table_count", 0)
                    extracted_data["extraction_quality"] = "odl_hybrid" if preserved_ocr else odl_result.get("extraction_quality", "odl_clean")
        except Exception as _odl_err:
            logger.warning(f"[Upload] ODL 合并失败，使用已有提取结果: {_odl_err}")
        # ────────────────────────────────────────────────────────────────────

        # Auto never publishes the provisional local extraction as a second
        # route. When quality is poor and MinerU has been configured, switch
        # directly to the same full MinerU publication path as an explicit
        # MinerU upload.
        if (
            requested_parse_route == PARSE_ROUTE_AUTO
            and str(extracted_data.get("extraction_quality") or "").lower() == "poor"
            and _mineru_configured()
        ):
            return _start_mineru_full_route_upload(
                doc_id=doc_id,
                filename=file.filename,
                pdf_bytes=content,
                requested_route=requested_parse_route,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
                embedding_api_host=embedding_api_host,
                summary_api_key=summary_api_key,
                auto_selected=True,
            )

        extracted_data["parse_manifest"] = _build_upload_parse_manifest(
            doc_id,
            parse_route=requested_parse_route,
            resolved_route=PARSE_ROUTE_LOCAL,
            pdf_bytes=content,
            status=PARSE_STATUS_READY,
            stage="local_ready",
            metadata={
                "full_route": False,
                "auto_selected": requested_parse_route == PARSE_ROUTE_AUTO,
                "text_source": str(extracted_data.get("extraction_method") or "pdf_native"),
                "block_source": "canonical_pages",
                "rag_source": "pdf_native",
                "figure_source": "pdf_native",
            },
        )

        # A local re-upload is also a route publication. Retire an old MinerU
        # run under the same short lock used by its final artifact swap.
        with _get_document_publication_lock(doc_id):
            _retire_superseded_mineru_job(doc_id)
            _clear_block_dependent_ai_cache(doc_id, documents_store.get(doc_id))
            _remove_current_semantic_groups(doc_id)
            documents_store[doc_id] = {
                "filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "data": extracted_data,
                "pdf_url": pdf_url
            }
            if page_ocr_artifact_ref:
                documents_store[doc_id]["data"]["parse_artifacts"] = [page_ocr_artifact_ref]
            if str(extracted_data.get("extraction_method") or "").startswith("odl"):
                artifact = build_document_parse_artifact(
                    doc_id=doc_id,
                    provider="odl",
                    provider_version="odl-v1",
                    pages=extracted_data.get("pages") or [],
                    tables=extracted_data.get("structured_table_bundles") or [],
                    warnings=[str(extracted_data.get("ocr_warning") or "")],
                    capabilities={
                        "per_page_text": True,
                        "document_structure": True,
                        "structured_tables": True,
                        "figures": False,
                        **derive_table_geometry_capabilities(extracted_data.get("structured_table_bundles") or []),
                    },
                )
                artifact_path = persist_document_parse_artifact(DATA_DIR, artifact)
                documents_store[doc_id]["data"]["parse_artifact"] = {
                    "schema_version": artifact["schema_version"],
                    "provider": artifact["provider"],
                    "source_hash": artifact["source_hash"],
                    "ref": artifact_reference(DATA_DIR, artifact_path),
                }
                documents_store[doc_id]["data"].setdefault("parse_artifacts", []).append(
                    documents_store[doc_id]["data"]["parse_artifact"]
                )
            _normalize_page_keys(documents_store[doc_id])
            save_document(doc_id, documents_store[doc_id])

        summary_api_key = ((api_key or "").strip() or (embedding_api_key or "").strip() or None)
        index_status = _queue_document_indexes(
            doc_id,
            embedding_model,
            embedding_api_key,
            embedding_api_host,
            summary_api_key,
        )

        response = {
            "message": "PDF上传成功",
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": extracted_data["total_pages"],
            "total_chars": len(extracted_data["full_text"]),
            "image_count": extracted_data.get("image_count", 0),
            "pdf_url": pdf_url,
            "ocr_used": extracted_data.get("ocr_used", False),
            "ocr_backend": extracted_data.get("ocr_backend"),
            "extraction_quality": extracted_data.get("extraction_quality", "unknown"),
            "extraction_method": extracted_data.get("extraction_method", "unknown"),
            "parse_manifest": extracted_data.get("parse_manifest", {}),
            "indexing_status": index_status.get("status", "queued"),
        }
        if extracted_data.get("extraction_method") == "odl":
            response["odl_element_count"] = extracted_data.get("odl_element_count", 0)
            response["odl_kept_count"] = extracted_data.get("odl_kept_count", 0)
            response["odl_soft_kept_caption_count"] = extracted_data.get("odl_soft_kept_caption_count", 0)
            response["structured_table_count"] = extracted_data.get("structured_table_count", 0)
        
        if extracted_data.get("ocr_error"):
            response["ocr_warning"] = extracted_data["ocr_error"]
        
        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"PDF处理失败: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF处理失败: {str(e)}")


@router.post("/documents/url")
async def import_url(
    request: Request,
):
    """将网页 URL 转为文档并索引到向量库

    请求体 JSON:
        url: 目标网页 URL
        embedding_model: 文本嵌入模型
        embedding_api_key: 云端嵌入模型的 API 密钥（可选）
        embedding_api_host: 自定义 API 地址（可选）
        api_key: 语义意群摘要使用的 LLM API 密钥（可选，默认回退到 embedding_api_key）
    """
    try:
        body = await request.json()
        url = body.get("url", "").strip()
        embedding_model = body.get("embedding_model", "local-minilm")
        embedding_api_key = (body.get("embedding_api_key") or "").strip() or None
        embedding_api_host = body.get("embedding_api_host")
        api_key = (body.get("api_key") or "").strip() or None

        if not url:
            raise HTTPException(status_code=400, detail="URL 不能为空")

        normalized_model = normalize_embedding_model_id(embedding_model)
        if not normalized_model:
            raise HTTPException(status_code=400, detail=f"Embedding模型 '{embedding_model}' 未配置")
        embedding_model = normalized_model

        # 抓取网页内容
        result = await fetch_url_content(url)
        title = result["title"]
        content = result["content"]

        if not content or len(content) < 10:
            raise HTTPException(status_code=400, detail="网页内容为空或过短")

        doc_id = generate_doc_id(content)

        # 构建与 PDF 文档兼容的数据结构
        extracted_data = {
            "full_text": content,
            "total_pages": 1,
            "pages": [{"page": 1, "text": content}],
            "source_type": "url",
            "source_url": url,
        }
        extracted_data["parse_manifest"] = build_parse_manifest(
            doc_id=doc_id,
            route=PARSE_ROUTE_LOCAL,
            resolved_route=PARSE_ROUTE_LOCAL,
            source_hash=derive_source_hash(content),
            status=PARSE_STATUS_READY,
            stage="local_ready",
            metadata={
                "full_route": False,
                "text_source": "url",
                "block_source": "canonical_pages",
                "rag_source": "url",
                "figure_source": "none",
            },
        )

        _clear_block_dependent_ai_cache(doc_id, documents_store.get(doc_id))
        _remove_current_semantic_groups(doc_id)
        documents_store[doc_id] = {
            "filename": f"🌐 {title[:60]}",
            "upload_time": datetime.now().isoformat(),
            "data": extracted_data,
            "pdf_url": None,
        }

        save_document(doc_id, documents_store[doc_id])

        create_index(
            doc_id, content, str(VECTOR_STORE_DIR),
            embedding_model, embedding_api_key, embedding_api_host,
            pages=extracted_data["pages"],
            summary_api_key=api_key or embedding_api_key,
            index_source="url",
            index_meta={
                "source_hash": extracted_data["parse_manifest"]["source_hash"],
                "document_source_hash": extracted_data["parse_manifest"]["source_hash"],
                "parse_generation": extracted_data["parse_manifest"]["generation"],
                "parser_route": PARSE_ROUTE_LOCAL,
            },
        )

        return {
            "message": "URL 导入成功",
            "doc_id": doc_id,
            "filename": f"🌐 {title[:60]}",
            "title": title,
            "url": url,
            "total_chars": len(content),
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL 导入失败: {str(e)}")


@router.get("/document/{doc_id}")
async def get_document(doc_id: str, include_content: bool = True):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    parse_manifest = _read_document_parse_manifest(doc_id, doc)
    response = {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "upload_time": doc["upload_time"],
        "total_pages": doc["data"]["total_pages"],
        "total_chars": len(doc["data"]["full_text"]),
        "image_count": doc["data"].get("image_count", 0),
        "pdf_url": doc.get("pdf_url"),
        "ocr_used": doc["data"].get("ocr_used", False),
        "ocr_backend": doc["data"].get("ocr_backend"),
        "extraction_quality": doc["data"].get("extraction_quality", "unknown"),
        "extraction_method": doc["data"].get("extraction_method", "unknown"),
        "parse_manifest": parse_manifest,
        "parse_ready": is_parse_prepared(parse_manifest),
        "indexing": _get_document_index_status(doc_id),
    }
    if include_content:
        _require_document_parse_ready(doc_id, doc)
        response["pages"] = doc["data"]["pages"]
        response["images"] = doc["data"].get("images", [])
    return response


@router.get("/document/{doc_id}/index-status")
async def get_document_index_status(doc_id: str):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    return _get_document_index_status(doc_id)


@router.get("/documents/{doc_id}/table-visual-verifications/{task_id}")
async def get_document_table_visual_verification_status(doc_id: str, task_id: str):
    """Return the persisted post-retrieval visual verification task state."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    status = get_table_visual_verification_status(doc_id, task_id)
    if not status:
        raise HTTPException(status_code=404, detail="表格视觉核验任务未找到")
    return status


def recover_pending_rag_transactions() -> list[dict]:
    """Rollback interrupted source switches after the document store has loaded."""
    pending_dir = DATA_DIR / "rag_transactions" / "pending"
    if not pending_dir.exists():
        return []
    recovered: list[dict] = []
    terminal_states = {"committed", "rolled_back"}
    for journal_path in pending_dir.glob("*.json"):
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            doc_id = str(journal.get("doc_id") or "")
            source = str(journal.get("source") or "pdf_native")
            if not doc_id or journal.get("state") in terminal_states:
                continue
            if not _load_complete_rag_backup_manifest(doc_id, source):
                logger.error("[RagIndex] pending transaction lacks a complete backup: %s", journal_path)
                continue
            snapshot = _restore_vector_index_backup(doc_id, source)
            document = snapshot.get("document_restore") or {}
            semantic_groups = snapshot.get("semantic_group_restore") or {}
            if not snapshot.get("restored"):
                raise RuntimeError(
                    "RAG transaction recovery incomplete: "
                    f"snapshot={snapshot.get('restored')}, "
                    f"document={document.get('restored')}, "
                    f"semantic={semantic_groups.get('restored')}"
                )
            _write_rag_transaction_journal(
                doc_id,
                "rolled_back",
                source=source,
                manifest_path=str(journal.get("manifest_path") or ""),
            )
            recovered.append({
                "doc_id": doc_id,
                "state": "rolled_back",
                "vector": snapshot,
                "document": document,
                "semantic_groups": semantic_groups,
            })
        except Exception as exc:
            logger.error("[RagIndex] failed to recover pending transaction %s: %s", journal_path, exc)
    return recovered


@router.get("/documents/{doc_id}/deep-parse/status")
async def get_document_deep_parse_status(doc_id: str):
    """Return current MinerU deep-parse task status for this document."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    return _get_deep_parse_status(doc_id)


@router.post("/documents/{doc_id}/deep-parse")
async def start_document_deep_parse(request: Request, doc_id: str):
    """Manually run MinerU deep parsing and replace the current block index."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    try:
        body = await request.json()
    except Exception:
        body = {}
    provider = str(body.get("provider") or "mineru").strip().lower()
    if provider != "mineru":
        raise HTTPException(status_code=400, detail="当前仅支持 MinerU 深度解析")

    if not _mineru_configured():
        raise HTTPException(status_code=400, detail="请先在 OCR 设置中配置 MinerU Worker/直连模式和 Token")

    doc = documents_store[doc_id]
    parse_manifest = _require_mineru_route_compatibility(doc_id, doc)
    pdf_path = _resolve_document_pdf_path(doc)
    if not pdf_path or not pdf_path.exists():
        raise HTTPException(status_code=400, detail="当前文档没有可用于 MinerU 深度解析的 PDF 原文件")

    access_ok, access_message = _validate_mineru_access(_load_online_ocr_config("mineru"))
    if not access_ok:
        raise HTTPException(status_code=400, detail=access_message)

    force = bool(body.get("force", False))
    return _queue_mineru_deep_parse(doc_id, force=force)


@router.post("/documents/{doc_id}/deep-parse/cancel")
async def cancel_document_deep_parse(doc_id: str):
    """Stop local waiting for an in-flight MinerU deep-parse task."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    return _cancel_mineru_deep_parse(doc_id)


@router.post("/documents/{doc_id}/deep-parse/rebuild")
async def rebuild_document_deep_parse_index(doc_id: str):
    """Rebuild block index from the cached MinerU raw payload without re-uploading.

    用于适配器逻辑（如页码映射、类型识别）修复后，让已解析过的文档
    不必重新烧一次云端解析即可用上修复。
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    parse_manifest = _require_mineru_route_compatibility(doc_id, doc)
    payload = _load_mineru_result_for_manifest(doc_id, parse_manifest)
    if not payload:
        raise HTTPException(status_code=409, detail="当前解析代际没有可用的 MinerU 解析结果，请重新执行深度解析")

    pdf_path = _resolve_document_pdf_path(doc)
    try:
        block_index = build_block_index_from_mineru_payload(
            doc_id=doc_id,
            doc=doc,
            payload=payload,
            pdf_path=pdf_path,
        )
        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
            )
            save_block_index(DATA_DIR, doc_id, block_index)
            removed = _clear_block_dependent_ai_cache(doc_id, documents_store.get(doc_id))
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新执行 MinerU 阅读块重建")
    except Exception as exc:
        logger.exception("[DeepParse] rebuild-from-cache failed for %s: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail=f"从缓存重建索引失败: {exc}")

    block_count = sum(len(page.get("blocks") or []) for page in block_index.get("pages", []))
    outline_count = len(block_index.get("outline") or [])
    figure_count = sum(
        1
        for page in block_index.get("pages", [])
        for block in (page.get("blocks") or [])
        if block.get("type") in ("figure", "table")
    )
    with _get_document_publication_lock(doc_id):
        _require_current_parse_generation(
            doc_id,
            parse_generation=str(parse_manifest.get("generation") or ""),
            document_source_hash=str(parse_manifest.get("source_hash") or ""),
        )
        _set_deep_parse_status(
            doc_id,
            "ready",
            stage="ready",
            block_count=block_count,
            outline_count=outline_count,
            figure_count=figure_count,
            cache_removed=removed,
            active_source=MINERU_BLOCK_INDEX_SOURCE,
            active_mineru=True,
            message="已从缓存的 MinerU 结果重建索引",
            parse_generation=str(parse_manifest.get("generation") or ""),
            document_source_hash=str(parse_manifest.get("source_hash") or ""),
        )
    return _get_deep_parse_status(doc_id)


@router.get("/documents/{doc_id}/rag-index/status")
async def get_document_rag_index_status(doc_id: str):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    return _get_rag_index_status(doc_id)


@router.post("/documents/{doc_id}/rag-index/rebuild")
async def rebuild_document_rag_index_from_mineru(request: Request, doc_id: str):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    parse_manifest = _require_mineru_route_compatibility(doc_id, doc)

    try:
        body = await request.json()
    except Exception:
        body = {}

    payload = _load_mineru_result_for_manifest(doc_id, parse_manifest)
    if not payload:
        raise HTTPException(status_code=409, detail="当前解析代际没有可用的 MinerU 解析结果，请重新执行深度解析")

    normalized = normalize_mineru_for_rag(payload, page_sizes=_document_pdf_page_sizes(doc_id))
    ok, failures = validate_mineru_rag_data(
        normalized,
        original_full_text=(doc.get("data") or {}).get("full_text", ""),
    )
    estimate = {
        "page_count": len(normalized.get("pages") or []),
        "full_text_chars": len(normalized.get("full_text") or ""),
        "estimated_embedding_tokens": _estimate_text_tokens(normalized.get("full_text") or ""),
        "estimated_chunk_count": _estimate_chunk_count(normalized.get("full_text") or ""),
        "structured_table_count": len(normalized.get("structured_table_bundles") or []),
        "source_hash": normalized.get("source_hash", ""),
        "normalizer_version": normalized.get("normalizer_version", ""),
    }
    if body.get("estimate_only"):
        return {
            "status": "estimated",
            "can_rebuild": ok,
            "estimate": estimate,
            "quality_report": normalized.get("quality_report") or {},
            "quality_failures": failures,
            "rag_index": _get_rag_index_status(doc_id),
        }
    if not ok:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "MinerU 问答索引重建失败，已保留原索引",
                "quality_failures": failures,
                "quality_report": normalized.get("quality_report") or {},
            },
        )

    embedding_model = str(body.get("embedding_model") or "local-minilm").strip() or "local-minilm"
    embedding_api_key = (body.get("embedding_api_key") or body.get("api_key") or "").strip() or None
    embedding_api_host = (body.get("embedding_api_host") or body.get("api_host") or "").strip() or None
    summary_api_key = (body.get("summary_api_key") or "").strip() or None
    summary_model = str(body.get("summary_model") or body.get("model") or "gpt-4o-mini").strip() or "gpt-4o-mini"
    summary_provider = str(body.get("summary_provider") or body.get("provider") or "openai").strip() or "openai"
    summary_api_host = (body.get("summary_api_host") or body.get("api_host") or "").strip()

    try:
        return await asyncio.to_thread(
            _rebuild_mineru_rag_index,
            doc_id,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            embedding_api_host=embedding_api_host,
            summary_api_key=summary_api_key,
            summary_model=summary_model,
            summary_provider=summary_provider,
            summary_api_host=summary_api_host,
            expected_parse_generation=str(parse_manifest.get("generation") or ""),
            expected_document_source_hash=str(parse_manifest.get("source_hash") or ""),
        )
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新发起 MinerU 索引重建")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[RagIndex] MinerU rebuild failed for %s: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/documents/{doc_id}/rag-index/rollback")
async def rollback_document_rag_index(doc_id: str):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    parse_manifest = _read_document_parse_manifest(doc_id, documents_store[doc_id])
    if _is_full_mineru_parse_manifest(parse_manifest):
        raise HTTPException(
            status_code=409,
            detail="MinerU 全程解析不支持只回退问答索引；请重新选择本地路线并重新解析整份文档",
        )
    try:
        manifest = _load_complete_rag_backup_manifest(doc_id, "pdf_native")
        if not manifest:
            raise HTTPException(status_code=409, detail="没有完整的本地 RAG 回滚快照")
        snapshot = _restore_vector_index_backup(doc_id, "pdf_native")
        document = snapshot.get("document_restore") or {}
        semantic_groups = snapshot.get("semantic_group_restore") or {}
        if not snapshot.get("restored"):
            raise RuntimeError("RAG 回滚快照恢复不完整")
        return {
            "status": "ready",
            "message": "已回退到本地问答索引",
            "rag_index": snapshot,
            "document": document,
            "semantic_groups": semantic_groups,
            "manifest": manifest,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[RagIndex] rollback failed for %s: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/documents/{doc_id}/blocks")
async def get_document_blocks(doc_id: str, force_rebuild: bool = False):
    """Return page-level blocks, outline and PDF bbox anchors for immersive reading."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    _require_document_parse_ready(doc_id, doc)
    try:
        return ensure_block_index(
            doc_id=doc_id,
            doc=doc,
            data_dir=DATA_DIR,
            pdf_path=_resolve_document_pdf_path(doc),
            force_rebuild=force_rebuild,
        )
    except Exception as exc:
        logger.exception("[BlockIndex] build failed for %s", doc_id)
        raise HTTPException(status_code=500, detail=f"块索引生成失败: {exc}")


@router.get("/documents/{doc_id}/reading-outline")
async def get_document_reading_outline(doc_id: str, force: bool = False):
    """Return cached/fallback AI reading outline with evidence block bindings."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    parse_manifest = _require_document_parse_ready(doc_id, doc)
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
        force_rebuild=force,
    )
    try:
        return await get_or_create_reading_outline(
            data_dir=DATA_DIR,
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            force=force,
            cache_writer=_build_parse_bound_cache_writer(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
                persist=lambda outline: save_reading_outline(DATA_DIR, doc_id, outline),
            ),
        )
    except _SupersededAICacheGeneration:
        raise HTTPException(status_code=409, detail="AI 缓存已清理，请重新生成阅读总结")
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新生成阅读总结")


@router.post("/documents/{doc_id}/reading-outline")
async def create_document_reading_outline(
    request: Request,
    doc_id: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_host: Optional[str] = None,
    force: bool = False,
):
    """Generate or return the structured reading outline used by immersive reading."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    try:
        body = await request.json()
    except Exception:
        body = {}

    force = bool(body.get("force", force))
    api_key, model, provider, api_host = _resolve_overview_runtime_params(
        request,
        api_key,
        model,
        provider,
        api_host,
    )
    if not api_key:
        merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
        prov = merged.get(provider, {})
        api_key = (prov.get("api_key") or "").strip()

    doc = documents_store[doc_id]
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
        force_rebuild=force,
    )

    try:
        return await get_or_create_reading_outline(
            data_dir=DATA_DIR,
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=_get_overview_provider_endpoint(provider, api_host),
            force=force,
            cache_writer=_build_parse_bound_cache_writer(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
                persist=lambda outline: save_reading_outline(DATA_DIR, doc_id, outline),
            ),
        )
    except _SupersededAICacheGeneration:
        raise HTTPException(status_code=409, detail="AI 缓存已清理，请重新生成阅读总结")
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新生成阅读总结")


@router.get("/documents/{doc_id}/section-outline")
async def get_document_section_outline(doc_id: str, force: bool = False):
    """返回 PDF 书签或确定性的启发式章节大纲。"""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    parse_manifest = _require_document_parse_ready(doc_id, doc)
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
        force_rebuild=force,
    )
    try:
        return await get_or_create_section_outline(
            data_dir=DATA_DIR,
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            force=force,
            cache_writer=_build_parse_bound_cache_writer(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
                persist=lambda outline: save_section_outline(DATA_DIR, doc_id, outline),
            ),
        )
    except _SupersededAICacheGeneration:
        raise HTTPException(status_code=409, detail="AI 缓存已清理，请重新生成章节大纲")
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新生成章节大纲")


@router.post("/documents/{doc_id}/section-outline")
async def create_document_section_outline(
    request: Request,
    doc_id: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_host: Optional[str] = None,
    force: bool = False,
):
    """为左侧“大纲”生成或返回原文章节树。"""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    try:
        body = await request.json()
    except Exception:
        body = {}

    force = bool(body.get("force", force))
    api_key, model, provider, api_host = _resolve_overview_runtime_params(
        request,
        api_key,
        model,
        provider,
        api_host,
    )
    if not api_key:
        merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
        prov = merged.get(provider, {})
        api_key = (prov.get("api_key") or "").strip()

    doc = documents_store[doc_id]
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
        force_rebuild=force,
    )

    try:
        return await get_or_create_section_outline(
            data_dir=DATA_DIR,
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=_get_overview_provider_endpoint(provider, api_host),
            force=force,
            cache_writer=_build_parse_bound_cache_writer(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
                persist=lambda outline: save_section_outline(DATA_DIR, doc_id, outline),
            ),
        )
    except _SupersededAICacheGeneration:
        raise HTTPException(status_code=409, detail="AI 缓存已清理，请重新生成章节大纲")
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新生成章节大纲")


@router.get("/document/{doc_id}/thumbnail/{page}")
async def get_page_thumbnail(doc_id: str, page: int):
    """按需生成 PDF 页面缩略图

    使用 pymupdf 渲染指定页面为 40dpi 缩略图，返回 base64 编码的 JPEG。
    单页缩略图约 5-15KB。

    Args:
        doc_id: 文档 ID
        page: 页码（1-indexed）
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    pdf_url = doc.get("pdf_url")
    if not pdf_url:
        raise HTTPException(status_code=400, detail="该文档无 PDF 文件（可能是 URL 导入的文档）")

    pdf_path = UPLOAD_DIR / pdf_url.split("/")[-1]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF 文件不存在")

    try:
        import fitz
        import base64

        pdf_doc = fitz.open(str(pdf_path))
        if page < 1 or page > len(pdf_doc):
            pdf_doc.close()
            raise HTTPException(status_code=400, detail=f"页码超出范围 (1-{len(pdf_doc)})")

        pdf_page = pdf_doc[page - 1]
        # 40dpi 缩略图，体积小且足够预览
        pix = pdf_page.get_pixmap(dpi=40)
        img_bytes = pix.tobytes("jpeg")
        pdf_doc.close()

        b64 = base64.b64encode(img_bytes).decode("ascii")
        return {
            "doc_id": doc_id,
            "page": page,
            "thumbnail": f"data:image/jpeg;base64,{b64}",
            "width": pix.width,
            "height": pix.height,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缩略图生成失败: {str(e)}")


_GRAPHRAG_PARSE_IDENTITY_FILE = "chatpdf_parse_identity.json"


def _graphrag_parse_identity(manifest: dict) -> dict:
    """Return the parser generation a GraphRAG artifact is allowed to serve."""
    return {
        "parse_generation": str(manifest.get("generation") or ""),
        "document_source_hash": str(manifest.get("source_hash") or ""),
    }


def _graphrag_build_matches_active_parse(
    doc_id: str,
    *,
    parse_generation: str,
    document_source_hash: str,
) -> bool:
    return matches_parse_generation(
        _read_document_parse_manifest(doc_id, documents_store.get(doc_id)),
        generation=parse_generation,
        source_hash=document_source_hash,
    )


def _bind_graphrag_progress_identity(progress, *, parse_generation: str, document_source_hash: str):
    """将内存构建进度绑定到发起构建时的主解析代际。"""
    progress.parse_generation = str(parse_generation or "")
    progress.document_source_hash = str(document_source_hash or "")
    return progress


def _graphrag_progress_matches_parse(progress, manifest: dict) -> bool:
    return bool(
        progress
        and str(getattr(progress, "parse_generation", "") or "")
        == str(manifest.get("generation") or "")
        and str(getattr(progress, "document_source_hash", "") or "")
        == str(manifest.get("source_hash") or "")
    )


def _graphrag_identity_path(working_dir: str | Path) -> Path:
    return Path(working_dir) / _GRAPHRAG_PARSE_IDENTITY_FILE


def _graphrag_index_matches_parse(working_dir: str | Path, manifest: dict) -> bool:
    expected = _graphrag_parse_identity(manifest)
    if not expected["parse_generation"] or not expected["document_source_hash"]:
        return False
    try:
        stored = json.loads(_graphrag_identity_path(working_dir).read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        str(stored.get("parse_generation") or "") == expected["parse_generation"]
        and str(stored.get("document_source_hash") or "") == expected["document_source_hash"]
    )


def _write_graphrag_parse_identity(working_dir: str | Path, manifest: dict) -> None:
    path = _graphrag_identity_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = _graphrag_parse_identity(manifest)
    if not identity["parse_generation"] or not identity["document_source_hash"]:
        raise RuntimeError("GraphRAG 缺少文档解析代际，无法发布索引")
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(identity, ensure_ascii=False), encoding="utf-8")
    os.replace(str(temp_path), str(path))


def _new_graphrag_staging_dir(doc_id: str, parse_generation: str) -> str:
    root = Path(settings.graphrag_working_dir) / "_staging" / doc_id
    return str(root / f"{parse_generation}.{uuid.uuid4().hex}")


def _publish_graphrag_staging_dir(staging_dir: str | Path, working_dir: str | Path) -> None:
    """Swap a completed generation into the active GraphRAG location."""
    staging_path = Path(staging_dir)
    active_path = Path(working_dir)
    backup_path = active_path.with_name(f".{active_path.name}.{uuid.uuid4().hex}.backup")
    moved_active = False
    try:
        if active_path.exists():
            os.replace(str(active_path), str(backup_path))
            moved_active = True
        os.replace(str(staging_path), str(active_path))
    except Exception:
        if moved_active and backup_path.exists() and not active_path.exists():
            os.replace(str(backup_path), str(active_path))
        raise
    finally:
        if backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)


@router.post("/document/{doc_id}/graphrag/build")
async def build_graphrag_index(doc_id: str, request: Request):
    """为文档构建 GraphRAG 知识图谱索引

    请求体 JSON:
        api_key: LLM API 密钥
        model: LLM 模型名
        api_provider: LLM 提供商
        api_host: LLM API 地址（可选）
        embedding_model: Embedding 模型名（可选）
        embedding_api_key: Embedding API 密钥（可选）
        embedding_api_host: Embedding API 地址（可选）
        force_rebuild: 是否强制重建（可选，默认 false）

    说明：该端点由前端「GraphRAG 知识图谱」按钮显式触发，调用即表示用户同意
    构建。是否开启全局 settings.enable_graphrag 只影响 /chat 路径是否把图谱
    上下文拼进 prompt，不再作为构建入口的硬门槛，避免勾选框常年半成品。
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    parse_manifest = _require_document_parse_ready(doc_id, doc)
    parse_generation = str(parse_manifest.get("generation") or "")
    parse_source_hash = str(parse_manifest.get("source_hash") or "")
    full_text = doc.get("data", {}).get("full_text", "")
    if not full_text or len(full_text) < 50:
        raise HTTPException(status_code=400, detail="文档内容过短，无法构建知识图谱")

    # 文档级构建锁：防止同一文档并发构建
    from services.graphrag import get_build_lock, INSTANCES as _GRAPHRAG_INSTANCES, BUILD_PROGRESS as _GRAPHRAG_BUILD_PROGRESS
    lock = get_build_lock(doc_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="该文档正在构建中，请勿重复提交")

    staging_dir = ""
    build_progress = None
    try:
        body = await request.json()
        api_key = body.get("api_key", "")
        model = body.get("model", "")
        provider = body.get("api_provider", "")
        api_host = body.get("api_host", "")
        embedding_model = body.get("embedding_model", "")
        embedding_api_key = body.get("embedding_api_key", "")
        embedding_api_host = body.get("embedding_api_host", "")
        force_rebuild = body.get("force_rebuild", False)

        provider_lower = (provider or "").lower()
        if not model:
            raise HTTPException(status_code=400, detail="GraphRAG 构建需要 model")
        if not api_key and provider_lower not in {"ollama", "local"}:
            raise HTTPException(status_code=400, detail="GraphRAG 构建需要 api_key")

        from services.graphrag import GraphRAG, GraphRAGConfig, BuildProgress

        # 解析 endpoint
        endpoint = ""
        if api_host:
            host = api_host.strip().rstrip('/')
            endpoint = f"{host}/chat/completions" if not host.endswith('/chat/completions') else host

        # 解析 embedding endpoint
        embed_endpoint = ""
        if embedding_api_host:
            host = embedding_api_host.strip().rstrip('/')
            if host.endswith('/embeddings') or host.endswith('/v1'):
                embed_endpoint = host
            else:
                embed_endpoint = f"{host}/v1"

        working_dir = os.path.join(settings.graphrag_working_dir, doc_id)

        embedding_registry_key = None
        embedding_config = None
        if embedding_model:
            embedding_registry_key, embedding_config = resolve_model_id(embedding_model)

        resolved_embedding_model = embedding_model or model
        resolved_embedding_provider = provider
        resolved_embedding_dim = 1536
        resolved_embedding_base_url = ""

        if embedding_config:
            resolved_embedding_model = (
                embedding_config.get("model_name")
                or embedding_registry_key
                or resolved_embedding_model
            )
            resolved_embedding_provider = embedding_config.get("provider") or provider
            resolved_embedding_dim = int(embedding_config.get("dimension") or 1536)
            resolved_embedding_base_url = (embedding_config.get("base_url") or "").strip().rstrip('/')
        elif embedding_model:
            logger.warning(
                "[GraphRAG] 未找到 embedding 模型注册信息，沿用请求参数: %s",
                embedding_model,
            )

        config = GraphRAGConfig(
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            embedding_api_key=embedding_api_key or api_key,
            embedding_model=resolved_embedding_model,
            embedding_provider=resolved_embedding_provider,
            embedding_endpoint=embed_endpoint or resolved_embedding_base_url or endpoint.replace("/chat/completions", ""),
            embedding_dim=resolved_embedding_dim,
        )

        # A GraphRAG cache is valid only for the document parse generation that
        # created it. Same-PDF re-uploads intentionally retain ``doc_id``.
        # Without this identity check a graph from a former local/MinerU route
        # could be loaded merely because its model configuration still matches.
        has_persisted_index = GraphRAG.has_persisted_index(working_dir)
        if (
            not force_rebuild
            and has_persisted_index
            and _graphrag_index_matches_parse(working_dir, parse_manifest)
        ):
            disk_meta = GraphRAG.load_metadata(working_dir)
            from services.graphrag.graphrag import _compute_config_hash
            current_hash = _compute_config_hash(
                config, settings.graphrag_chunk_token_size, settings.graphrag_max_gleaning
            )
            if disk_meta and disk_meta.status == "done" and disk_meta.config_hash == current_hash:
                # 索引已存在且配置未变，直接从磁盘加载
                rag = await GraphRAG.load_from_disk(
                    working_dir=working_dir,
                    config=config,
                    chunk_token_size=settings.graphrag_chunk_token_size,
                    entity_extract_max_gleaning=settings.graphrag_max_gleaning,
                    best_model_max_async=settings.graphrag_max_async,
                    cheap_model_max_async=settings.graphrag_max_async,
                )
                if rag is not None:
                    with _get_document_publication_lock(doc_id):
                        _require_current_parse_generation(
                            doc_id,
                            parse_generation=parse_generation,
                            document_source_hash=parse_source_hash,
                        )
                        if not _graphrag_index_matches_parse(working_dir, parse_manifest):
                            raise _SupersededParseGeneration("GraphRAG 索引不属于当前解析代际")
                        loaded_progress = _bind_graphrag_progress_identity(
                            rag.get_build_progress(),
                            parse_generation=parse_generation,
                            document_source_hash=parse_source_hash,
                        )
                        _GRAPHRAG_INSTANCES[doc_id] = rag
                        _GRAPHRAG_BUILD_PROGRESS[doc_id] = loaded_progress
                    return {
                        "message": "GraphRAG 索引已存在，从磁盘加载",
                        "doc_id": doc_id,
                        "stats": rag.stats(),
                        "loaded_from_disk": True,
                    }
        # 旧图谱在 staging 构建成功前继续可用；最终发布会在短锁内替换。

        # 构建新索引
        # Build into a generation-specific staging directory. The parser route
        # may change while LLM extraction is running; direct writes to the
        # active doc_id directory would let that old task resurrect stale data.
        staging_dir = _new_graphrag_staging_dir(doc_id, parse_generation)
        rag = GraphRAG(
            working_dir=staging_dir,
            config=config,
            chunk_token_size=settings.graphrag_chunk_token_size,
            entity_extract_max_gleaning=settings.graphrag_max_gleaning,
            best_model_max_async=settings.graphrag_max_async,
            cheap_model_max_async=settings.graphrag_max_async,
        )

        # 注册构建进度（供 progress 端点读取）
        build_progress = _bind_graphrag_progress_identity(
            rag.get_build_progress(),
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        )
        _GRAPHRAG_BUILD_PROGRESS[doc_id] = build_progress

        await rag.ainsert(full_text)

        if not _graphrag_build_matches_active_parse(
            doc_id,
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        ):
            if _GRAPHRAG_BUILD_PROGRESS.get(doc_id) is rag.get_build_progress():
                _GRAPHRAG_BUILD_PROGRESS.pop(doc_id, None)
            raise HTTPException(status_code=409, detail="文档解析路线已更新，已丢弃旧 GraphRAG 构建")

        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )
            _write_graphrag_parse_identity(staging_dir, parse_manifest)
            # 身份文件写入与目录交换之间也保持 fail-closed。
            _require_current_parse_generation(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )
            _publish_graphrag_staging_dir(staging_dir, working_dir)
            staging_dir = ""

        # Storage objects keep absolute paths. Reload after the directory swap
        # so the shared registry never points back to the moved staging path.
        published_rag = await GraphRAG.load_from_disk(
            working_dir=working_dir,
            config=config,
            chunk_token_size=settings.graphrag_chunk_token_size,
            entity_extract_max_gleaning=settings.graphrag_max_gleaning,
            best_model_max_async=settings.graphrag_max_async,
            cheap_model_max_async=settings.graphrag_max_async,
        )
        if published_rag is None:
            if not _graphrag_build_matches_active_parse(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            ):
                raise HTTPException(status_code=409, detail="文档解析路线已更新，已丢弃旧 GraphRAG 构建")
            raise RuntimeError("GraphRAG 发布后无法重新加载索引")

        with _get_document_publication_lock(doc_id):
            current_manifest = _require_current_parse_generation(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )
            if not _graphrag_index_matches_parse(working_dir, current_manifest):
                raise _SupersededParseGeneration("GraphRAG 发布结果不属于当前解析代际")
            stats = published_rag.stats()
            published_progress = _bind_graphrag_progress_identity(
                published_rag.get_build_progress(),
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )
            # 缓存实例以便查询时复用（存到模块级 registry，跨 router 共享）
            _GRAPHRAG_INSTANCES[doc_id] = published_rag
            _GRAPHRAG_BUILD_PROGRESS[doc_id] = published_progress

        return {
            "message": "GraphRAG 索引构建完成",
            "doc_id": doc_id,
            "stats": stats,
            "loaded_from_disk": False,
        }

    except _SupersededParseGeneration:
        if _GRAPHRAG_BUILD_PROGRESS.get(doc_id) is build_progress:
            _GRAPHRAG_BUILD_PROGRESS.pop(doc_id, None)
        raise HTTPException(status_code=409, detail="文档解析路线已更新，已丢弃旧 GraphRAG 构建")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GraphRAG] 构建失败: {e}", exc_info=True)
        # 记录失败到进度表
        if _GRAPHRAG_BUILD_PROGRESS.get(doc_id) is build_progress:
            prog = build_progress
            prog.status = "failed"
            prog.last_error = str(e)
        raise HTTPException(status_code=500, detail=f"GraphRAG 构建失败: {str(e)}")
    finally:
        if staging_dir:
            shutil.rmtree(staging_dir, ignore_errors=True)
        lock.release()


@router.get("/document/{doc_id}/graphrag/stats")
async def get_graphrag_stats(doc_id: str):
    """获取文档的 GraphRAG 索引统计信息

    优先从内存 INSTANCES 读取，若不存在则尝试从磁盘加载。
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    from services.graphrag import INSTANCES as _GRAPHRAG_INSTANCES, BUILD_PROGRESS as _GRAPHRAG_BUILD_PROGRESS

    # 1. 内存中已有实例
    if doc_id in _GRAPHRAG_INSTANCES:
        rag = _GRAPHRAG_INSTANCES[doc_id]
        working_dir = os.path.join(settings.graphrag_working_dir, doc_id)
        if _graphrag_index_matches_parse(working_dir, parse_manifest):
            return {"doc_id": doc_id, "stats": rag.stats()}
        _GRAPHRAG_INSTANCES.pop(doc_id, None)
        _GRAPHRAG_BUILD_PROGRESS.pop(doc_id, None)

    # 2. 尝试从磁盘加载元数据（不需要 api_key 也能读取统计）
    working_dir = os.path.join(settings.graphrag_working_dir, doc_id)
    from services.graphrag import GraphRAG
    disk_meta = GraphRAG.load_metadata(working_dir)
    if disk_meta is not None and _graphrag_index_matches_parse(working_dir, parse_manifest):
        return {
            "doc_id": doc_id,
            "stats": {
                "working_dir": working_dir,
                "num_nodes": disk_meta.num_nodes,
                "num_edges": disk_meta.num_edges,
                "num_docs": disk_meta.num_docs,
                "num_chunks": disk_meta.num_chunks,
                "build_meta": disk_meta.to_dict(),
            },
            "loaded_from_disk_meta": True,
        }

    raise HTTPException(status_code=404, detail="该文档未构建 GraphRAG 索引")


@router.get("/document/{doc_id}/graphrag/progress")
async def get_graphrag_build_progress(doc_id: str):
    """获取文档的 GraphRAG 构建进度"""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    from services.graphrag import BUILD_PROGRESS as _GRAPHRAG_BUILD_PROGRESS

    # 1. 内存中的实时进度
    if doc_id in _GRAPHRAG_BUILD_PROGRESS:
        prog = _GRAPHRAG_BUILD_PROGRESS[doc_id]
        if _graphrag_progress_matches_parse(prog, parse_manifest):
            return {"doc_id": doc_id, "progress": prog.to_dict()}
        _GRAPHRAG_BUILD_PROGRESS.pop(doc_id, None)

    # 2. 磁盘元数据
    working_dir = os.path.join(settings.graphrag_working_dir, doc_id)
    from services.graphrag import GraphRAG
    disk_meta = GraphRAG.load_metadata(working_dir)
    if disk_meta is not None and _graphrag_index_matches_parse(working_dir, parse_manifest):
        return {"doc_id": doc_id, "progress": disk_meta.to_dict()}

    raise HTTPException(status_code=404, detail="该文档未构建 GraphRAG 索引")


@router.delete("/document/{doc_id}/graphrag")
async def delete_graphrag_index(doc_id: str):
    """删除文档的 GraphRAG 索引（内存 + 磁盘）"""
    import shutil
    from services.graphrag import INSTANCES as _GRAPHRAG_INSTANCES, BUILD_PROGRESS as _GRAPHRAG_BUILD_PROGRESS

    # 从内存移除
    _GRAPHRAG_INSTANCES.pop(doc_id, None)
    _GRAPHRAG_BUILD_PROGRESS.pop(doc_id, None)

    # 删除磁盘数据
    working_dir = os.path.join(settings.graphrag_working_dir, doc_id)
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir, ignore_errors=True)
        return {"message": "GraphRAG 索引已删除", "doc_id": doc_id}

    raise HTTPException(status_code=404, detail="该文档未构建 GraphRAG 索引")


@router.get("/api/ocr/status")
async def get_ocr_status():
    """
    检查 OCR 可用性、后端状态和当前配置

    返回包含 OCR 后端可用性、Poppler 状态、当前配置和安装指引的完整状态信息。
    """
    status = is_ocr_available()

    # 使用 OCRRegistry 获取后端可用性
    available_backends = _ocr_registry.list_available()
    available_document_parsers = _document_parser_registry.list_available()
    backends = {
        "tesseract": available_backends.get("tesseract", False),
        "paddleocr": available_backends.get("paddleocr", False),
        "mistral": available_backends.get("mistral", False),  # 在线 OCR
        "mineru": available_document_parsers.get("mineru", False),  # 文档级深度解析
        "doc2x": available_document_parsers.get("doc2x", False),  # legacy 文档解析
    }

    # 检测 Poppler 可用性
    poppler_path = _find_poppler()
    poppler_available = poppler_path is not None

    # 自动逐页 OCR 仅推荐本地引擎；云端需要在上传时显式选择。
    recommended = None
    if backends.get("paddleocr"):
        recommended = "paddleocr"
    elif backends.get("tesseract"):
        recommended = "tesseract"

    # 构建在线 OCR 服务状态信息
    online_services = {}
    for provider in _SUPPORTED_ONLINE_OCR_PROVIDERS:
        provider_config = _load_online_ocr_config(provider)
        if provider in ("mineru", "doc2x"):
            # MinerU 支持 Worker 代理和直连 API；Doc2X 仍为 Worker 代理。
            access_mode = provider_config.get("access_mode", "worker")
            worker_url = provider_config.get("worker_url", "")
            token = provider_config.get("token", "")
            token_mode = provider_config.get("token_mode", "frontend")
            configured = bool(token) if provider == "mineru" and access_mode == "direct" else (
                bool(worker_url) and (token_mode == "worker" or bool(token))
            )
            adapter = _document_parser_registry.get_adapter(provider)
            available = adapter.is_available() if adapter else False
            online_services[provider] = {
                "configured": configured,
                "available": available,
                "access_mode": access_mode,
                "deprecated": provider == "doc2x",
                "usage": get_ocr_provider_usage(provider),
            }
        else:
            # Mistral 等直接 API 调用模式
            api_key = provider_config.get("api_key", "")
            base_url = provider_config.get("base_url", "")
            adapter = _ocr_registry.get_adapter(provider)
            available = adapter.is_available() if adapter else False
            online_services[provider] = {
                "configured": bool(api_key),
                "available": available,
            }

    # 从 AppSettings 读取当前 OCR 配置
    config = {
        "default_mode": settings.ocr_default_mode,
        "dpi": settings.ocr_dpi,
        "language": settings.ocr_language,
        "quality_threshold": settings.ocr_quality_threshold,
    }

    # 安装指引
    install_instructions = {
        "tesseract": "pip install pytesseract pdf2image && 安装 Tesseract-OCR",
        "paddleocr": "pip install paddleocr pdf2image",
    }

    # 当 Poppler 不可用时，在安装指引中标注 Poppler 缺失及其影响
    if not poppler_available:
        install_instructions["poppler"] = (
            "Poppler 未安装，PDF 转图像功能不可用，OCR 将无法正常工作。\n"
            "安装方式:\n"
            "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
            "  - macOS: brew install poppler\n"
            "  - Linux: sudo apt-get install poppler-utils"
        )

    return {
        "available": status["any"],
        "backends": backends,
        "page_ocr_backends": ["paddleocr", "tesseract", "mistral"],
        "deprecated_page_ocr_backends": ["mineru", "doc2x"],
        "provider_sunset": {
            "doc2x": {"deprecated": True, "replacement": "local_auto", "usage": get_ocr_provider_usage("doc2x")},
            "paddleocr": {"deprecated": True, "replacement": "paddleocr_vl", "usage": get_ocr_provider_usage("paddleocr")},
        },
        "poppler_available": poppler_available,
        "recommended": recommended,
        "config": config,
        "online_services": online_services,
        "install_instructions": install_instructions,
        # [T3] Figure Extraction 能力
        "figure_extraction": {
            "configured": online_services.get("mineru", {}).get("configured", False),
            "can_extract_figures": online_services.get("mineru", {}).get("available", False),
            "timeout_sec": settings.figure_extraction_timeout_sec,
        },
        "figure_preview": {
            "yolo": get_yolo_model_status(),
        },
    }


# 支持的在线 OCR 提供商列表
_SUPPORTED_ONLINE_OCR_PROVIDERS = {"mistral", "mineru", "doc2x"}


@router.get("/api/layout/yolo/status")
async def get_layout_yolo_status():
    """获取速览 YOLO 图表预览资源状态。"""
    return get_yolo_model_status()


@router.post("/api/layout/yolo/download")
async def download_layout_yolo_model(request: Request):
    """下载 YOLO 权重到默认用户数据目录或用户指定目录。"""
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        install_dir = (body.get("install_dir") or "").strip()
        force = bool(body.get("force", False))
        return download_yolo_model(install_dir=install_dir or None, force=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"下载 YOLO 权重失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/layout/yolo/config")
async def save_layout_yolo_config(request: Request):
    """保存用户手动指定的 YOLO 权重路径。"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")

    model_path = (body.get("model_path") or "").strip()
    if not model_path:
        raise HTTPException(status_code=400, detail="缺少 model_path 参数")

    try:
        return configure_yolo_model_path(model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"保存 YOLO 权重路径失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/layout/yolo/reset")
async def reset_layout_yolo_config():
    """清除用户自定义 YOLO 权重路径。"""
    try:
        return reset_yolo_model_config()
    except Exception as e:
        logger.error(f"重置 YOLO 权重配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ocr/online-config")
async def save_online_ocr_config(request: Request):
    """
    保存在线 OCR 服务配置

    支持 Mistral（API Key + Base URL）和 MinerU/Doc2X（Worker 代理模式）。
    持久化到本地配置文件，并重新注册对应的在线 OCR 适配器。

    请求体（Mistral）:
        {
            "provider": "mistral",
            "api_key": "sk-xxx...",
            "base_url": "https://api.mistral.ai"  // 可选
        }

    请求体（MinerU）:
        {
            "provider": "mineru",
            "worker_url": "https://your-worker.workers.dev",
            "auth_key": "your-auth-secret",  // 可选
            "token_mode": "frontend",  // "frontend" 或 "worker"
            "token": "your-mineru-token",  // token_mode 为 frontend 时必填
            "enable_ocr": true,  // 可选，默认 true
            "enable_formula": true,  // 可选，默认 true
            "enable_table": true  // 可选，默认 true
        }

    请求体（Doc2X）:
        {
            "provider": "doc2x",
            "worker_url": "https://your-worker.workers.dev",
            "auth_key": "your-auth-secret",  // 可选
            "token_mode": "frontend",  // "frontend" 或 "worker"
            "token": "your-doc2x-token"  // token_mode 为 frontend 时必填
        }

    响应:
        {"success": true, "message": "配置已保存"}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")

    provider = body.get("provider", "").strip()

    # 校验 provider 参数
    if not provider:
        raise HTTPException(status_code=400, detail="缺少 provider 参数")
    if provider not in _SUPPORTED_ONLINE_OCR_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 provider: {provider}，当前支持: {', '.join(sorted(_SUPPORTED_ONLINE_OCR_PROVIDERS))}",
        )

    # 根据 provider 类型构建配置字典
    if provider in ("mineru", "doc2x"):
        # Worker 代理模式配置
        existing_config = _load_online_ocr_config(provider)
        access_mode = body.get("access_mode", "worker").strip() if provider == "mineru" else "worker"
        if access_mode not in ("worker", "direct"):
            raise HTTPException(status_code=400, detail="access_mode 必须为 'worker' 或 'direct'")
        worker_url = body.get("worker_url", "").strip()
        auth_key = body.get("auth_key", "").strip()
        token_mode = body.get("token_mode", "frontend").strip()
        token = body.get("token", "").strip()
        if not auth_key:
            auth_key = existing_config.get("auth_key", "")
        if not token:
            token = existing_config.get("token", "")

        # 校验 worker_url 参数
        if access_mode == "worker" and not worker_url:
            raise HTTPException(status_code=400, detail="缺少 worker_url 参数")
        if worker_url:
            try:
                worker_url = validate_external_ocr_service_url(
                    worker_url,
                    service_name=f"{provider} Worker",
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # 校验 token_mode 参数
        if token_mode not in ("frontend", "worker"):
            raise HTTPException(status_code=400, detail="token_mode 必须为 'frontend' 或 'worker'")

        config: dict = {
            "access_mode": access_mode,
            "worker_url": worker_url,
            "auth_key": auth_key,
            "token_mode": token_mode,
            "token": token,
        }

        # MinerU 特有选项
        if provider == "mineru":
            base_url = body.get("base_url", "https://mineru.net/api/v4").strip() or "https://mineru.net/api/v4"
            try:
                base_url = validate_external_ocr_service_url(
                    base_url,
                    service_name="MinerU API Base URL",
                ).rstrip("/")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            config["base_url"] = base_url
            config["enable_ocr"] = body.get("enable_ocr", False)
            config["enable_formula"] = body.get("enable_formula", True)
            config["enable_table"] = body.get("enable_table", True)
            model_version = str(body.get("model_version") or "vlm").strip().lower()
            if model_version not in {"vlm", "pipeline"}:
                raise HTTPException(status_code=400, detail="model_version 必须为 'vlm' 或 'pipeline'")
            config["model_version"] = model_version
    else:
        # Mistral 等直接 API 调用模式
        api_key = body.get("api_key", "").strip()
        base_url = body.get("base_url", "").strip()
        current_config = _load_online_ocr_config(provider)

        if not api_key:
            api_key = str(current_config.get("api_key") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="缺少 api_key 参数，请先填写或保存 API Key")

        config = {"api_key": api_key}
        if base_url:
            try:
                base_url = validate_external_ocr_service_url(
                    base_url,
                    service_name="Mistral OCR Base URL",
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            config["base_url"] = base_url

    # 持久化配置到本地文件
    try:
        _save_online_ocr_config(provider, config)
    except Exception as e:
        logger.error(f"保存在线 OCR 配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置保存失败: {str(e)}")

    # 重新注册对应的在线 OCR 适配器
    try:
        if provider == "mistral":
            # 重新加载完整配置（合并默认值）
            full_config = _load_online_ocr_config("mistral")
            # 从注册表中移除旧的 mistral 适配器（如果存在）
            _ocr_registry._adapters.pop("mistral", None)
            # 创建新的 MistralAdapter 实例并注册
            new_adapter = MistralAdapter(
                api_key=full_config.get("api_key", ""),
                base_url=full_config.get("base_url", "https://api.mistral.ai"),
            )
            _ocr_registry.register(new_adapter)
            logger.info(f"MistralAdapter 已重新注册，可用: {new_adapter.is_available()}")
        elif provider == "mineru":
            # 重新加载完整配置
            full_config = _load_online_ocr_config("mineru")
            # MinerU belongs to the document-parser registry, never to page OCR.
            _document_parser_registry.unregister("mineru")
            # 创建新的 MinerUAdapter 实例并注册
            if full_config.get("access_mode") == "direct":
                new_adapter = MinerUDirectAdapter(
                    token=full_config.get("token", ""),
                    base_url=full_config.get("base_url", "https://mineru.net/api/v4"),
                    enable_ocr=full_config.get("enable_ocr", False),
                    enable_formula=full_config.get("enable_formula", True),
                    enable_table=full_config.get("enable_table", True),
                    model_version=full_config.get("model_version", "vlm"),
                )
            else:
                new_adapter = MinerUAdapter(
                    worker_url=full_config.get("worker_url", ""),
                    auth_key=full_config.get("auth_key", ""),
                    token=full_config.get("token", ""),
                    token_mode=full_config.get("token_mode", "frontend"),
                    enable_ocr=full_config.get("enable_ocr", False),
                    enable_formula=full_config.get("enable_formula", True),
                    enable_table=full_config.get("enable_table", True),
                    model_version=full_config.get("model_version", "vlm"),
                )
            _document_parser_registry.register(new_adapter)
            logger.info(f"MinerU 文档解析适配器已重新注册，可用: {new_adapter.is_available()}")
        elif provider == "doc2x":
            # 重新加载完整配置
            full_config = _load_online_ocr_config("doc2x")
            # Doc2X is retained only as a legacy document parser configuration.
            _document_parser_registry.unregister("doc2x")
            # 创建新的 Doc2XAdapter 实例并注册
            new_adapter = Doc2XAdapter(
                worker_url=full_config.get("worker_url", ""),
                auth_key=full_config.get("auth_key", ""),
                token=full_config.get("token", ""),
                token_mode=full_config.get("token_mode", "frontend"),
            )
            _document_parser_registry.register(new_adapter)
            logger.info(f"Doc2X 文档解析适配器已重新注册，可用: {new_adapter.is_available()}")
    except Exception as e:
        # 适配器注册失败不影响配置保存结果，仅记录警告
        logger.warning(f"重新注册在线 OCR 适配器失败: {e}")

    return {"success": True, "message": "配置已保存"}


@router.get("/api/ocr/online-config")
async def get_online_ocr_config():
    """
    获取在线 OCR 服务配置（敏感信息脱敏显示）

    返回各在线 OCR 提供商的配置状态，包括：
    - Mistral: API Key 是否已配置、脱敏后的 API Key 预览和 Base URL
    - MinerU/Doc2X: Worker URL、Auth Key/Token 配置状态和脱敏预览、Token Mode 及 MinerU 特有选项

    响应:
        {
            "mistral": {
                "api_key_configured": true,
                "api_key_preview": "sk-x...xxxx",
                "base_url": "https://api.mistral.ai"
            },
            "mineru": {
                "worker_url": "https://your-worker.workers.dev",
                "auth_key_configured": true,
                "auth_key_preview": "your...cret",
                "token_mode": "frontend",
                "token_configured": true,
                "token_preview": "your...oken",
                "enable_ocr": true,
                "enable_formula": true,
                "enable_table": true
            },
            "doc2x": {
                "worker_url": "",
                "auth_key_configured": false,
                "auth_key_preview": "",
                "token_mode": "frontend",
                "token_configured": false,
                "token_preview": ""
            }
        }
    """
    result = {}

    for provider in _SUPPORTED_ONLINE_OCR_PROVIDERS:
        config = _load_online_ocr_config(provider)

        if provider in ("mineru", "doc2x"):
            # Worker 代理模式：返回 worker_url、auth_key/token 脱敏信息
            worker_url = config.get("worker_url", "")
            access_mode = config.get("access_mode", "worker")
            base_url = config.get("base_url", "https://mineru.net/api/v4")
            auth_key = config.get("auth_key", "")
            token_mode = config.get("token_mode", "frontend")
            token = config.get("token", "")

            provider_result = {
                "access_mode": access_mode,
                "base_url": base_url,
                "worker_url": worker_url,
                "auth_key_configured": bool(auth_key),
                "auth_key_preview": _mask_api_key(auth_key),
                "token_mode": token_mode,
                "token_configured": bool(token),
                "token_preview": _mask_api_key(token),
            }

            # MinerU 特有选项
            if provider == "mineru":
                provider_result["enable_ocr"] = config.get("enable_ocr", False)
                provider_result["enable_formula"] = config.get("enable_formula", True)
                provider_result["enable_table"] = config.get("enable_table", True)
                provider_result["model_version"] = config.get("model_version", "vlm")

            result[provider] = provider_result
        else:
            # Mistral 等直接 API 调用模式
            api_key = config.get("api_key", "")
            base_url = config.get("base_url", "")

            result[provider] = {
                "api_key_configured": bool(api_key),
                "api_key_preview": _mask_api_key(api_key),
                "base_url": base_url,
            }

    return result


@router.post("/api/ocr/validate-key")
async def validate_ocr_key(request: Request):
    """
    验证在线 OCR 服务的 API Key / Worker 连接有效性

    - Mistral: 调用 GET /v1/files 接口验证 API Key
    - MinerU: 向 Worker URL 发送 GET 请求测试可达性和认证
    - Doc2X: 向 Worker URL 发送 GET 请求测试可达性和认证

    请求体（Mistral）:
        {
            "provider": "mistral",
            "api_key": "sk-xxx..."
        }

    请求体（MinerU/Doc2X）:
        {
            "provider": "mineru",
            "worker_url": "https://your-worker.workers.dev",
            "auth_key": "your-auth-secret"  // 可选
        }

    响应:
        {"valid": true, "message": "验证成功"}
        {"valid": false, "message": "验证失败原因"}
    """
    import httpx

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")

    provider = body.get("provider", "").strip()

    # 校验 provider 参数
    if not provider:
        raise HTTPException(status_code=400, detail="缺少 provider 参数")
    if provider not in _SUPPORTED_ONLINE_OCR_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 provider: {provider}，当前支持: {', '.join(sorted(_SUPPORTED_ONLINE_OCR_PROVIDERS))}",
        )

    # 根据 provider 执行验证
    if provider == "mistral":
        api_key = body.get("api_key", "").strip()

        # 优先验证当前表单里的 Base URL；未传时再回退到已保存配置。
        current_config = _load_online_ocr_config("mistral")
        if not api_key:
            api_key = str(current_config.get("api_key") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="缺少 api_key 参数，请先填写或保存 Mistral API Key")

        base_url = (
            body.get("base_url")
            or current_config.get("base_url", "")
            or "https://api.mistral.ai"
        ).strip().rstrip("/")
        try:
            base_url = validate_external_ocr_service_url(
                base_url,
                service_name="Mistral OCR Base URL",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            # 调用 Mistral API 的文件列表接口验证 Key 有效性
            with httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0)) as client:
                resp = client.get(
                    f"{base_url}/v1/files",
                    headers={"Authorization": f"Bearer {api_key}"},
                )

            if resp.status_code == 200:
                logger.info("Mistral API Key 验证成功")
                return {"valid": True, "message": "API Key 验证成功"}
            elif resp.status_code in (401, 403):
                logger.warning(f"Mistral API Key 验证失败: HTTP {resp.status_code}")
                return {"valid": False, "message": "API Key 无效或已过期"}
            else:
                # 其他 HTTP 错误也视为验证失败
                logger.warning(f"Mistral API Key 验证异常: HTTP {resp.status_code}")
                return {"valid": False, "message": f"验证失败，服务返回 HTTP {resp.status_code}"}

        except httpx.TimeoutException:
            logger.warning("Mistral API Key 验证超时")
            return {"valid": False, "message": "连接 Mistral 超时，请检查网络代理或 Base URL"}
        except httpx.ConnectError as e:
            logger.warning("Mistral API Key 验证连接失败: %s", e)
            return {"valid": False, "message": f"无法连接到 Mistral 服务：{str(e) or '连接失败'}"}
        except httpx.TransportError as e:
            logger.warning("Mistral API Key 验证传输错误: %s", e)
            return {"valid": False, "message": f"Mistral 网络/SSL 连接失败：{str(e) or '传输错误'}"}
        except httpx.RequestError as e:
            logger.warning(f"Mistral API Key 验证网络错误: {e}")
            return {"valid": False, "message": f"Mistral 请求失败：{str(e) or '网络错误'}"}

    elif provider in ("mineru", "doc2x"):
        # Worker 代理模式验证：分两步——先测试 Worker 可达性，再测试 Token 有效性
        current_config = _load_online_ocr_config(provider)
        access_mode = body.get("access_mode", current_config.get("access_mode", "worker")).strip() if provider == "mineru" else "worker"
        worker_url = body.get("worker_url", "").strip()
        auth_key = body.get("auth_key", "").strip()
        token = body.get("token", "").strip()
        token_mode = body.get("token_mode", current_config.get("token_mode", "frontend")).strip()
        base_url = body.get("base_url", current_config.get("base_url", "https://mineru.net/api/v4")).strip() or "https://mineru.net/api/v4"
        provider_label = "MinerU" if provider == "mineru" else "Doc2X"
        if not worker_url:
            worker_url = current_config.get("worker_url", "")
        if not auth_key:
            auth_key = current_config.get("auth_key", "")
        if not token:
            token = current_config.get("token", "")

        if provider == "mineru" and access_mode == "direct":
            if not token:
                raise HTTPException(status_code=400, detail="直连模式下必须提供 MinerU Token")
            try:
                base_url_clean = validate_external_ocr_service_url(
                    base_url,
                    service_name="MinerU API Base URL",
                ).rstrip("/")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            try:
                with httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0)) as client:
                    token_resp = client.post(
                        f"{base_url_clean}/file-urls/batch",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "enable_formula": False,
                            "enable_table": False,
                            "language": "ch",
                            "files": [{
                                "name": "health_check.pdf",
                                "is_ocr": True,
                            }],
                        },
                    )
                if token_resp.status_code in (401, 403):
                    return {"valid": False, "message": "MinerU Token 无效或已过期"}
                if not token_resp.is_success:
                    return {"valid": False, "message": f"MinerU API 不可达 (HTTP {token_resp.status_code})"}
                token_data = token_resp.json()
                if token_data.get("code") != 0:
                    return {"valid": False, "message": f"MinerU Token 验证失败: {token_data.get('msg') or token_data}"}
                logger.info("MinerU 直连 Token 验证成功")
                return {"valid": True, "message": "MinerU 直连 API 可达且 Token 有效"}
            except httpx.TimeoutException:
                logger.warning("MinerU 直连验证超时")
                return {"valid": False, "message": "连接超时，请检查网络或 Base URL"}
            except httpx.ConnectError:
                logger.warning("MinerU 直连连接失败")
                return {"valid": False, "message": "连接失败，请检查网络或 Base URL"}
            except httpx.RequestError as e:
                logger.warning(f"MinerU 直连验证网络错误: {e}")
                return {"valid": False, "message": "网络连接失败，请检查网络设置"}

        # 校验 worker_url 参数
        if not worker_url:
            raise HTTPException(status_code=400, detail="缺少 worker_url 参数")

        try:
            worker_url_clean = validate_external_ocr_service_url(
                worker_url,
                service_name=f"{provider} Worker",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            with httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0)) as client:
                # 第一步：测试 Worker 可达性（GET /health，仅带 Auth Key）
                health_headers = {}
                if auth_key:
                    health_headers["X-Auth-Key"] = auth_key

                health_resp = client.get(f"{worker_url_clean}/health", headers=health_headers)

                if health_resp.status_code in (401, 403):
                    return {"valid": False, "message": "Auth Key 无效，请检查 Auth Key 是否正确"}
                if not health_resp.is_success:
                    return {"valid": False, "message": f"Worker 不可达 (HTTP {health_resp.status_code})"}

                # 第二步：测试 Token 有效性（前端透传模式下）
                if token_mode == "frontend":
                    if not token:
                        return {"valid": False, "message": "前端透传模式下必须提供 Token"}

                    token_headers = dict(health_headers)
                    if provider == "mineru":
                        token_headers["X-MinerU-Key"] = token
                        token_test_url = f"{worker_url_clean}/mineru/result/__health__"
                    else:
                        token_headers["X-Doc2X-Key"] = token
                        token_test_url = f"{worker_url_clean}/doc2x/status/__health__"

                    token_resp = client.get(token_test_url, headers=token_headers)

                    if token_resp.status_code in (401, 403):
                        return {"valid": False, "message": "Token 无效或缺失，请检查 Token 是否正确"}

                    # 尝试解析 JSON 响应
                    try:
                        token_data = token_resp.json()
                        if not token_resp.is_success or not token_data.get("success", True):
                            err_msg = token_data.get("message") or token_data.get("error") or "未知错误"
                            return {"valid": False, "message": f"Token 验证失败: {err_msg}"}
                    except Exception:
                        # 非 JSON 响应但状态码正常也视为通过
                        if not token_resp.is_success:
                            return {"valid": False, "message": f"Token 验证失败 (HTTP {token_resp.status_code})"}

                    logger.info(f"{provider_label} Worker + Token 验证成功")
                    return {"valid": True, "message": f"{provider_label} Worker 可达且 Token 有效"}
                else:
                    # Worker 模式：只需验证 Worker 可达性
                    logger.info(f"{provider_label} Worker 验证成功（Worker 模式）")
                    return {"valid": True, "message": f"{provider_label} Worker 可达（Token 由 Worker 配置）"}

        except httpx.TimeoutException:
            logger.warning(f"{provider_label} Worker 验证超时")
            return {"valid": False, "message": "连接超时，请检查 Worker URL 是否正确"}
        except httpx.ConnectError:
            logger.warning(f"{provider_label} Worker 连接失败")
            return {"valid": False, "message": "连接失败，请检查 Worker URL 是否正确"}
        except httpx.RequestError as e:
            logger.warning(f"{provider_label} Worker 验证网络错误: {e}")
            return {"valid": False, "message": "网络连接失败，请检查网络设置"}

    # 不应到达此处，但作为安全兜底
    return {"valid": False, "message": f"暂不支持 {provider} 的验证"}


# initialize
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
migrate_legacy_storage()
load_documents()
recover_pending_rag_transactions()
resume_pending_mineru_deep_parse_jobs()


# ============ 速览（Overview）API ============

from services.overview_service import (
    clear_overview_cache,
    get_or_create_overview,
    create_overview_task,
    get_task_status,
    OverviewDepth,
)
from models.dynamic_store import load_dynamic_providers
from models.provider_registry import PROVIDER_CONFIG


def _get_overview_provider_endpoint(provider_id: str, api_host: str = "") -> str:
    """按优先级解析速览使用的聊天端点。"""
    if api_host and api_host.strip():
        host = api_host.strip().rstrip("/")
        if host.endswith("/chat/completions"):
            return host
        return f"{host}/chat/completions"

    dynamic = load_dynamic_providers()
    if provider_id in dynamic:
        return dynamic[provider_id].get("endpoint", "")

    return PROVIDER_CONFIG.get(provider_id, {}).get("endpoint", "")


def _resolve_overview_runtime_params(
    request: Request,
    api_key: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    api_host: Optional[str],
):
    resolved_provider = (provider or request.headers.get("X-ChatPDF-Provider") or "openai").strip() or "openai"
    resolved_model = (model or request.headers.get("X-ChatPDF-Model") or "gpt-4o").strip() or "gpt-4o"
    resolved_api_key = (api_key or request.headers.get("X-ChatPDF-Api-Key") or "").strip()
    resolved_api_host = (api_host or request.headers.get("X-ChatPDF-Api-Host") or "").strip()
    return resolved_api_key, resolved_model, resolved_provider, resolved_api_host


@router.get("/documents/{doc_id}/blocks/translations")
async def get_block_translations(
    request: Request,
    doc_id: str,
    target_lang: str = "zh",
    model: Optional[str] = None,
    provider: Optional[str] = None,
):
    """Return cached block translations for the immersive reading panel."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    _require_document_parse_ready(doc_id, doc)
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
    )
    resolved_provider = (provider or request.headers.get("X-ChatPDF-Provider") or "").strip() or None
    resolved_model = (model or request.headers.get("X-ChatPDF-Model") or "").strip() or None
    return get_cached_translations(
        data_dir=DATA_DIR,
        doc_id=doc_id,
        block_index=block_index,
        target_lang=target_lang,
        provider=resolved_provider,
        model=resolved_model,
        ai_cache_generation=load_ai_cache_generation(DATA_DIR, doc_id),
    )


@router.post("/documents/{doc_id}/blocks/translate")
async def translate_document_blocks(
    request: Request,
    doc_id: str,
    target_lang: str = "zh",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_host: Optional[str] = None,
    force: bool = False,
):
    """Translate selected block ids and cache the result."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    try:
        body = await request.json()
    except Exception:
        body = {}

    block_ids = body.get("block_ids", [])
    if isinstance(block_ids, str):
        block_ids = [item.strip() for item in block_ids.split(",") if item.strip()]
    if not isinstance(block_ids, list):
        raise HTTPException(status_code=400, detail="block_ids 必须是数组")
    block_ids = [str(item) for item in block_ids if str(item).strip()]
    if not block_ids:
        raise HTTPException(status_code=400, detail="缺少 block_ids")
    if len(block_ids) > MAX_BLOCKS_PER_REQUEST:
        block_ids = block_ids[:MAX_BLOCKS_PER_REQUEST]

    target_lang = body.get("target_lang") or target_lang
    force = bool(body.get("force", force))

    api_key, model, provider, api_host = _resolve_overview_runtime_params(
        request,
        api_key,
        model,
        provider,
        api_host,
    )
    if not api_key:
        merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
        prov = merged.get(provider, {})
        api_key = (prov.get("api_key") or "").strip()

    provider_lower = (provider or "").lower()
    if not api_key and provider_lower not in {"local", "ollama"}:
        raise HTTPException(status_code=400, detail="请先配置用于翻译的对话模型 API Key")

    doc = documents_store[doc_id]
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
    )
    ai_cache_generation = load_ai_cache_generation(DATA_DIR, doc_id)

    try:
        return await translate_blocks(
            data_dir=DATA_DIR,
            doc_id=doc_id,
            block_index=block_index,
            block_ids=block_ids,
            target_lang=target_lang,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=_get_overview_provider_endpoint(provider, api_host),
            force=force,
            ai_cache_generation=ai_cache_generation,
            cache_writer=_build_parse_bound_cache_writer(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
                persist=lambda envelope: save_translation_cache(DATA_DIR, doc_id, envelope),
                ai_cache_generation=ai_cache_generation,
            ),
        )
    except _SupersededAICacheGeneration:
        raise HTTPException(status_code=409, detail="AI 缓存已清理，请重新翻译当前阅读块")
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新翻译当前阅读块")
    except Exception as exc:
        logger.exception("[BlockTranslation] translate failed for %s", doc_id)
        raise HTTPException(status_code=502, detail=f"段落翻译失败: {exc}")


@router.post("/documents/{doc_id}/blocks/pretranslate")
async def pretranslate_document_blocks(
    request: Request,
    doc_id: str,
    target_lang: str = "zh",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_host: Optional[str] = None,
    force: bool = False,
    concurrency: int = 8,
):
    """批量预翻译文档块，由后端统一控制并发。"""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    try:
        body = await request.json()
    except Exception:
        body = {}

    block_ids = body.get("block_ids", [])
    if isinstance(block_ids, str):
        block_ids = [item.strip() for item in block_ids.split(",") if item.strip()]
    if not isinstance(block_ids, list):
        raise HTTPException(status_code=400, detail="block_ids 必须是数组")
    block_ids = [str(item) for item in block_ids if str(item).strip()]
    if not block_ids:
        raise HTTPException(status_code=400, detail="缺少 block_ids")

    target_lang = body.get("target_lang") or target_lang
    force = bool(body.get("force", force))
    try:
        concurrency = int(body.get("concurrency", concurrency) or 8)
    except Exception:
        concurrency = 8

    api_key, model, provider, api_host = _resolve_overview_runtime_params(
        request,
        api_key,
        model,
        provider,
        api_host,
    )
    if not api_key:
        merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
        prov = merged.get(provider, {})
        api_key = (prov.get("api_key") or "").strip()

    provider_lower = (provider or "").lower()
    if not api_key and provider_lower not in {"local", "ollama"}:
        raise HTTPException(status_code=400, detail="请先配置用于翻译的对话模型 API Key")

    doc = documents_store[doc_id]
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
    )
    ai_cache_generation = load_ai_cache_generation(DATA_DIR, doc_id)

    try:
        return await translate_blocks(
            data_dir=DATA_DIR,
            doc_id=doc_id,
            block_index=block_index,
            block_ids=block_ids,
            target_lang=target_lang,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=_get_overview_provider_endpoint(provider, api_host),
            force=force,
            max_blocks=None,
            concurrency=concurrency,
            ai_cache_generation=ai_cache_generation,
            cache_writer=_build_parse_bound_cache_writer(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
                persist=lambda envelope: save_translation_cache(DATA_DIR, doc_id, envelope),
                ai_cache_generation=ai_cache_generation,
            ),
        )
    except _SupersededAICacheGeneration:
        raise HTTPException(status_code=409, detail="AI 缓存已清理，请重新开始全文预翻译")
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新开始全文预翻译")
    except Exception as exc:
        logger.exception("[BlockTranslation] pretranslate failed for %s", doc_id)
        raise HTTPException(status_code=502, detail=f"全文预翻译失败: {exc}")


@router.delete("/documents/{doc_id}/ai-cache")
async def clear_document_ai_cache(doc_id: str):
    """清理当前文档的 AI 辅助缓存，不删除原始文件、向量库或对话历史。"""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    removed: list[str] = []
    with _get_document_publication_lock(doc_id):
        ai_cache_generation = rotate_ai_cache_generation(DATA_DIR, doc_id)
        cache_paths = {
            "reading_outline": get_reading_outline_path(DATA_DIR, doc_id),
            "section_outline": get_section_outline_path(DATA_DIR, doc_id),
            "block_translations": get_translation_cache_path(DATA_DIR, doc_id),
        }

        for name, path in cache_paths.items():
            try:
                if path.exists():
                    path.unlink()
                    removed.append(name)
            except Exception as exc:
                logger.warning("[AICache] 删除 %s 缓存失败 doc=%s path=%s err=%s", name, doc_id, path, exc)

    for depth in (OverviewDepth.BRIEF, OverviewDepth.STANDARD, OverviewDepth.DETAILED):
        for render_mode in ("raw", "yolo"):
            await clear_overview_cache(doc_id, depth, render_mode)
    removed.append("overview")

    return {
        "doc_id": doc_id,
        "removed": removed,
        "ai_cache_generation": ai_cache_generation,
    }


@router.post("/documents/{doc_id}/overview")
async def create_overview(
    request: Request,
    doc_id: str,
    depth: str = "standard",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_host: Optional[str] = None,
    figure_render_mode: str = "raw",
):
    """
    触发速览生成

    Args:
        doc_id: 文档 ID
        depth: 速览深度 brief(简介) / standard(标准) / detailed(详细)
        api_key: API Key（可选，默认使用配置）
        model: 模型名称（可选，默认 gpt-4o）
        provider: 模型提供商（可选，默认 openai）

    Returns:
        task_id: 任务 ID，用于轮询状态
        status: 任务状态
    """
    # 验证文档存在
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    # 验证深度参数
    valid_depths = [OverviewDepth.BRIEF, OverviewDepth.STANDARD, OverviewDepth.DETAILED]
    if depth not in valid_depths:
        depth = OverviewDepth.STANDARD

    api_key, model, provider, api_host = _resolve_overview_runtime_params(
        request,
        api_key,
        model,
        provider,
        api_host,
    )

    if not api_key:
        merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
        prov = merged.get(provider, {})
        api_key = (prov.get("api_key") or "").strip()

    task = await create_overview_task(
        doc_id,
        depth,
        api_key,
        model,
        provider,
        _get_overview_provider_endpoint(provider, api_host),
        figure_render_mode=figure_render_mode,
    )

    try:
        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
            )
            if (
                str(task.parse_generation or "") != str(parse_manifest.get("generation") or "")
                or str(task.document_source_hash or "") != str(parse_manifest.get("source_hash") or "")
            ):
                raise _SupersededParseGeneration("速览任务捕获了不同的解析代际")
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新创建速览任务")

    return {
        "task_id": task.task_id,
        "doc_id": doc_id,
        "depth": depth,
        "status": task.status
    }


@router.get("/documents/{doc_id}/overview/tasks/{task_id}")
async def get_overview_task_status(doc_id: str, task_id: str):
    """
    获取速览生成任务状态
    
    Args:
        doc_id: 文档 ID
        task_id: 任务 ID
    
    Returns:
        status: 任务状态 (pending/processing/completed/failed)
        result: 完成后返回速览数据
        error: 失败时返回错误信息
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    task = await get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    if task.doc_id != doc_id:
        raise HTTPException(status_code=400, detail="任务与文档不匹配")
    if (
        not _is_legacy_parse_manifest(parse_manifest)
        and (
        str(task.parse_generation or "") != str(parse_manifest.get("generation") or "")
        or str(task.document_source_hash or "") != str(parse_manifest.get("source_hash") or "")
        )
    ):
        raise HTTPException(status_code=409, detail="该速览任务属于旧的文档解析结果，已不再可用")
    
    response = {
        "task_id": task.task_id,
        "doc_id": task.doc_id,
        "depth": task.depth,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }
    
    if task.status == "completed" and task.result:
        response["result"] = task.result.model_dump()
    elif task.status == "failed":
        response["error"] = task.error
    
    return response


@router.get("/documents/{doc_id}/overview")
async def get_overview(
    request: Request,
    doc_id: str,
    depth: str = "standard",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_host: Optional[str] = None,
    use_mineru_figures: bool = False,
    figure_render_mode: str = "raw",
    force: bool = False,
):
    """
    获取速览（同步接口）

    如果速览未生成，会自动创建任务并等待完成。

    Args:
        doc_id: 文档 ID
        depth: 速览深度 brief(简介) / standard(标准) / detailed(详细)
        api_key: API Key（可选，默认使用配置）
        model: 模型名称（可选，默认 gpt-4o）
        provider: 模型提供商（可选，默认 openai）
        force: 是否强制绕过缓存重新生成

    Returns:
        速览数据
    """
    # 验证文档存在
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    # 验证深度参数
    valid_depths = [OverviewDepth.BRIEF, OverviewDepth.STANDARD, OverviewDepth.DETAILED]
    if depth not in valid_depths:
        depth = OverviewDepth.STANDARD

    api_key, model, provider, api_host = _resolve_overview_runtime_params(
        request,
        api_key,
        model,
        provider,
        api_host,
    )

    # 如果没传 api_key，从当前模型配置中获取（和对话逻辑一致）
    if not api_key:
        merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
        prov = merged.get(provider, {})
        api_key = (prov.get("api_key") or "").strip()

    logger.info(
        "[Overview-Route] doc=%s depth=%s use_mineru_figures=%s figure_render_mode=%s force=%s",
        doc_id,
        depth,
        use_mineru_figures,
        figure_render_mode,
        force,
    )
    try:
        overview = await get_or_create_overview(
            doc_id,
            depth,
            api_key,
            model,
            provider,
            _get_overview_provider_endpoint(provider, api_host),
            use_mineru_figures=use_mineru_figures,
            figure_render_mode=figure_render_mode,
            force=force,
        )
        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
            )
            if (
                str(overview.parse_generation or "") != str(parse_manifest.get("generation") or "")
                or str(overview.document_source_hash or "") != str(parse_manifest.get("source_hash") or "")
            ):
                raise _SupersededParseGeneration("速览结果不属于当前解析代际")
        return overview.model_dump()
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新生成速览")
    except TimeoutError:
        raise HTTPException(status_code=408, detail="速览生成超时，请稍后重试")
    except Exception as e:
        logger.error(f"获取速览失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取速览失败: {str(e)}")
