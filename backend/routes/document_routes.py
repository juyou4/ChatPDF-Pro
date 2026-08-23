import io
import asyncio
import inspect
import os
import glob
import hashlib
import ipaddress
import json
import logging
import pickle
import queue
import re
import shutil
import tempfile
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import PyPDF2
import pdfplumber
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse

from services.vector_service import (
    create_index as _vector_create_index,
    build_vector_index as _vector_build_vector_index,
)
from services.url_loader_service import fetch_url_content
from services.multi_format_loader import is_supported_format, extract_from_file
from services.block_index_service import (
    active_block_index_revision,
    build_block_index,
    ensure_block_index,
    load_block_index,
    save_block_index,
    stage_visual_supplements_on_block_index,
)
from services.block_evidence_service import (
    EVIDENCE_SCHEMA_VERSION,
    build_rag_source_from_block_index,
)
from services.block_inventory_service import enumerate_block_inventory
from services.mineru_block_index_service import (
    MINERU_BLOCK_INDEX_SOURCE,
    MinerUResultUnreadable,
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
    parse_identity_matches,
    read_parse_manifest,
    transition_parse_manifest,
)
from services.mineru_progress import derive_mineru_progress, utc_now_iso_ms
from services.document_job_store import (
    load_document_job,
    persist_document_job,
    recover_interrupted_document_job,
)
from services.task_event_ledger import (
    append_task_event,
    classify_error_code,
    get_task_event_ledger,
    normalize_diagnostic_reason,
    sanitize_task_shortfall,
)
from services.embedding_service import (
    KEYLESS_EMBEDDING_PROVIDERS,
    RAG_INDEX_VERSION,
    _canonicalize_embedding_identity,
    _extract_vector_semantic_identity,
    _resolve_verified_query_embedding_identity,
    _semantic_generation_identity_complete,
    _build_semantic_group_index_async,
    _build_semantic_group_index,
    _index_cache,
    get_document_publication_lock as _shared_document_publication_lock,
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
    backfill_block_summaries,
    get_cached_translations,
    get_translation_cache_path,
    save_translation_cache,
    translate_blocks,
)
from services.reading_outline_service import (
    READING_OUTLINE_PROMPT_VERSION,
    get_or_create_reading_outline,
    get_reading_outline_path,
    save_reading_outline,
)
from services.section_outline_service import (
    SECTION_OUTLINE_PROMPT_VERSION,
    get_or_create_section_outline,
    get_section_outline_path,
    save_section_outline,
)
from services.downstream_task_state import (
    build_downstream_task_identity,
    create_downstream_task,
    downstream_task_identity_matches,
    get_downstream_task_events,
    get_downstream_task,
    transition_downstream_task,
)
from services.document_recall_service import list_recallable_documents
from services.paper_metadata_hydration_service import (
    hydrate_paper_metadata,
    hydration_cache_identity,
)
from services.table_visual_metadata import build_table_visual_metadata
from services.table_visual_verifier import (
    clear_table_visual_verification_cache,
    get_table_visual_verification_status,
)
from services.visual_enrichment_service import reset_visual_document_state
from services.visual_document_enrichment_service import preflight_summary_visuals
from services.visual_model_service import resolve_visual_enrichment_policy
from services.chat_visual_attachment_service import (
    ChatVisualAttachmentError,
    load_chat_visual_attachment,
)
from services.document_parse_adapter import (
    DocumentParseSubmission,
    MinerUQualityError,
    MinerUDocumentParseAdapter,
    collect_mineru_block_validation,
    validate_mineru_block_index_quality,
)
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
    should_enable_mineru_ocr,
    format_local_ocr_unavailable_message,
    diagnose_local_ocr,
    normalize_mineru_model_version,
    create_mineru_direct_http_client,
    validate_external_ocr_service_url,
    validate_mineru_direct_api_base_url,
    MistralAdapter,
    MinerUAdapter,
    WorkerOCRAdapter,
    MinerUDirectAdapter,
)
from services.layout_service import (
    configure_yolo_model_path,
    download_yolo_model,
    get_yolo_model_status,
    reset_yolo_model_config,
)
from services.local_parser_addon_service import (
    get_local_parser_addon_status,
    start_local_parser_addon_install,
)
from models.model_detector import get_model_provider, normalize_embedding_model_id
from models.model_id_resolver import PROVIDER_BASE_URL_HINTS, resolve_model_id
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# 目录策略与 app.py 保持一致：显式 CHATPDF_DATA_DIR 优先；未配置时
# desktop 使用 Electron 数据目录，server 使用项目根目录 data/。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(runtime.data_dir)
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
try:
    _DOCUMENT_INDEX_MAX_PENDING = max(1, min(32, int(os.getenv("CHATPDF_DOCUMENT_INDEX_MAX_PENDING", "6"))))
except ValueError:
    _DOCUMENT_INDEX_MAX_PENDING = 6
_DOCUMENT_INDEX_ADMISSION = threading.BoundedSemaphore(_DOCUMENT_INDEX_MAX_PENDING)
_DEEP_PARSE_LOCK = threading.Lock()
_DEEP_PARSE_TASKS: dict[str, dict] = {}
_PAPER_METADATA_HYDRATION_TASKS: dict[str, asyncio.Task] = {}
_PAPER_METADATA_HYDRATION_STATUS: dict[str, dict] = {}
_DEEP_PARSE_CANCEL_EVENTS: dict[str, threading.Event] = {}
_DEEP_PARSE_JOB_TYPE = "mineru_deep_parse"
_DEEP_PARSE_TERMINAL_STATUSES = {"ready", "partial_ready", "failed", "cancelled"}
try:
    _DEEP_PARSE_CONCURRENCY = max(1, min(8, int(os.getenv("CHATPDF_MINERU_DEEP_PARSE_CONCURRENCY", "2"))))
except ValueError:
    _DEEP_PARSE_CONCURRENCY = 2
_DEEP_PARSE_SEMAPHORE = threading.BoundedSemaphore(_DEEP_PARSE_CONCURRENCY)
try:
    _DEEP_PARSE_QUEUE_SIZE = max(1, min(64, int(os.getenv("CHATPDF_MINERU_DEEP_PARSE_QUEUE_SIZE", "8"))))
except ValueError:
    _DEEP_PARSE_QUEUE_SIZE = 8
_DEEP_PARSE_QUEUE: queue.Queue[tuple[str, threading.Event, Optional[dict], str, Optional[dict]]] = queue.Queue(
    maxsize=_DEEP_PARSE_QUEUE_SIZE
)
_DEEP_PARSE_WORKERS_LOCK = threading.Lock()
_DEEP_PARSE_WORKERS: list[threading.Thread] = []
_DOCUMENT_OPERATION_LOCKS_LOCK = threading.Lock()
_DOCUMENT_OPERATION_LOCKS: dict[str, threading.Lock] = {}


def _bounded_env_int(name: str, default: int, maximum: int) -> int:
    try:
        return max(1, min(int(os.getenv(name, str(default))), maximum))
    except (TypeError, ValueError):
        return default


_MAX_UPLOAD_BYTES = _bounded_env_int("CHATPDF_MAX_UPLOAD_BYTES", 100 * 1024 * 1024, 512 * 1024 * 1024)
_MAX_PDF_PAGES = _bounded_env_int("CHATPDF_MAX_PDF_PAGES", 1000, 10_000)
_MAX_PRETRANSLATE_BLOCK_IDS = _bounded_env_int("CHATPDF_MAX_PRETRANSLATE_BLOCK_IDS", 2000, 10_000)
_PDF_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
    "text/pdf",
}
_GENERIC_BINARY_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}
_PDF_HEADER_SEARCH_BYTES = 1024


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
    return _shared_document_publication_lock(doc_id)


async def _read_upload_with_limit(upload: UploadFile, *, max_bytes: int = _MAX_UPLOAD_BYTES) -> bytes:
    """Read multipart uploads incrementally so one request cannot allocate unbounded memory."""
    chunks: list[bytes] = []
    total_bytes = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过大小限制（最大 {max_bytes // (1024 * 1024)} MB）",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_uploaded_content(
    *,
    filename: str,
    content_type: str | None,
    content: bytes,
) -> None:
    """Validate PDF media declarations and magic bytes without blocking other formats.

    A filename suffix alone is not a parser boundary.  Browsers may send an
    empty or generic binary MIME type, so those remain compatible, but an
    explicitly contradictory MIME type is rejected.  PDF allows an optional
    binary prefix before the header; PDF readers are required to find `%PDF-`
    near the start of the file, hence the bounded 1 KiB search.
    """
    if not str(filename or "").lower().endswith(".pdf"):
        return
    normalized_mime = str(content_type or "").split(";", 1)[0].strip().lower()
    if normalized_mime not in _PDF_MIME_TYPES | _GENERIC_BINARY_MIME_TYPES:
        raise HTTPException(status_code=415, detail="PDF 文件的 MIME 类型不匹配")
    if bytes(content or b"")[:_PDF_HEADER_SEARCH_BYTES].find(b"%PDF-") < 0:
        raise HTTPException(status_code=400, detail="PDF 文件头无效或文件内容损坏")


def _safe_graphrag_working_dir(doc_id: str) -> Path:
    """Return the active GraphRAG directory only when it stays under its root."""
    root = Path(settings.graphrag_working_dir).resolve()
    candidate = (root / str(doc_id)).resolve()
    if candidate.parent != root:
        raise HTTPException(status_code=400, detail="无效的文档 ID")
    return candidate


def _url_origin(url: str) -> tuple[str, str, int | None]:
    """Normalize a service origin so saved credentials never cross providers."""
    parsed = urlsplit((url or "").strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower().rstrip(".")
    if not scheme or not host:
        raise ValueError("服务 URL 缺少有效 origin")
    port = parsed.port
    if (scheme, port) in {("https", 443), ("http", 80)}:
        port = None
    return scheme, host, port


def _same_service_origin(left: str, right: str) -> bool:
    try:
        return _url_origin(left) == _url_origin(right)
    except ValueError:
        return False


def _credential_for_service_origin(
    *,
    supplied: str,
    saved: str,
    target_url: str,
    saved_url: str,
    credential_name: str,
) -> str:
    """Reuse a stored secret only for exactly the same upstream origin."""
    if supplied:
        return supplied
    if not saved:
        return ""
    if not _same_service_origin(target_url, saved_url):
        raise HTTPException(
            status_code=400,
            detail=(
                f"目标服务地址已变化；为避免泄露已保存的 {credential_name}，"
                f"请为该地址重新填写凭据"
            ),
        )
    return saved


def _normalize_optional_provider_id(provider: Optional[str]) -> Optional[str]:
    normalized = str(provider or "").strip()
    return normalized or None


def _infer_provider_id_from_endpoint_domain(endpoint: Optional[str]) -> Optional[str]:
    raw_endpoint = str(endpoint or "").strip()
    if not raw_endpoint:
        return None
    try:
        host = (urlsplit(raw_endpoint).hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return None
    if not host:
        return None

    configured_providers = {**PROVIDER_CONFIG, **load_dynamic_providers()}
    for provider_id, config in configured_providers.items():
        candidate_endpoint = str((config or {}).get("endpoint") or "").strip()
        if not candidate_endpoint:
            continue
        try:
            candidate_host = (urlsplit(candidate_endpoint).hostname or "").strip().lower().rstrip(".")
        except ValueError:
            continue
        if candidate_host and (host == candidate_host or host.endswith(f".{candidate_host}")):
            return _normalize_optional_provider_id(provider_id)

    for provider_id, hint in PROVIDER_BASE_URL_HINTS.items():
        normalized_hint = str(hint or "").strip().lower().rstrip(".")
        if normalized_hint and normalized_hint in host:
            return _normalize_optional_provider_id(provider_id)

    return None


def _resolve_graphrag_embedding_identity_or_400(
    *,
    embedding_model: Optional[str],
    embedding_provider: Optional[str],
    embedding_api_host: Optional[str],
) -> dict:
    requested_model = str(embedding_model or "").strip()
    requested_provider = _normalize_optional_provider_id(embedding_provider)
    requested_host = str(embedding_api_host or "").strip()
    if not requested_model:
        raise HTTPException(status_code=400, detail="GraphRAG 构建需要显式 embedding_model")
    if not requested_provider:
        raise HTTPException(status_code=400, detail="GraphRAG 构建需要显式 embedding_provider")

    prefixed_provider = _normalize_optional_provider_id(
        _embedding_provider_from_model(requested_model)
    )
    if prefixed_provider and prefixed_provider.casefold() != requested_provider.casefold():
        raise HTTPException(
            status_code=400,
            detail="embedding_model 与 embedding_provider 不一致",
        )

    registry_key, embedding_config = resolve_model_id(requested_model)
    resolved_config = embedding_config or {}
    # ``provider`` describes the protocol adapter (often ``openai``), while
    # ``provider_id`` is the credential owner selected in the UI.
    config_provider = _normalize_optional_provider_id(resolved_config.get("provider_id"))
    if config_provider and config_provider.casefold() != requested_provider.casefold():
        raise HTTPException(
            status_code=400,
            detail="embedding_provider 与模型注册配置不一致",
        )

    inferred_host_provider = _normalize_optional_provider_id(
        _infer_provider_id_from_endpoint_domain(requested_host)
    )
    if (
        requested_host
        and inferred_host_provider
        and inferred_host_provider.casefold() != requested_provider.casefold()
    ):
        raise HTTPException(
            status_code=400,
            detail="embedding_api_host 与 embedding_provider 不一致",
        )

    try:
        identity = _canonicalize_embedding_identity(
            registry_key or requested_model,
            embedding_provider=requested_provider,
            base_url=requested_host or None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"GraphRAG Embedding 配置无效：{exc}",
        ) from exc

    canonical_provider = _normalize_optional_provider_id(identity.get("provider"))
    if canonical_provider != requested_provider:
        raise HTTPException(
            status_code=400,
            detail="GraphRAG Embedding 身份无法与请求参数对齐",
        )

    if canonical_provider == "local":
        if requested_host:
            raise HTTPException(
                status_code=400,
                detail="本地 GraphRAG Embedding 不应提供 embedding_api_host",
            )
    elif not requested_host:
        raise HTTPException(
            status_code=400,
            detail="远程 GraphRAG Embedding 需要显式 embedding_api_host",
        )

    try:
        dimension = int(resolved_config.get("dimension") or 0)
    except (TypeError, ValueError):
        dimension = 0
    if dimension <= 0:
        raise HTTPException(
            status_code=400,
            detail="GraphRAG Embedding 模型缺少有效 dimension 配置",
        )
    identity["dimension"] = dimension
    return identity


def _require_explicit_rag_embedding_identity_or_400(
    *,
    embedding_model: Optional[str],
    embedding_provider: Optional[str],
    embedding_api_host: Optional[str],
    embedding_api_key: Optional[str],
    operation: str,
) -> dict:
    """Validate the complete, user-selected embedding identity for a RAG build.

    An index is only meaningful in the vector space it was built with.  In
    particular, a recovery or startup path must never guess ``local-minilm``
    after a document was previously indexed with a remote provider.
    """
    requested_model = str(embedding_model or "").strip()
    requested_provider = _normalize_optional_provider_id(embedding_provider)
    requested_host = str(embedding_api_host or "").strip()
    requested_key = str(embedding_api_key or "").strip()

    if not requested_model:
        raise HTTPException(status_code=400, detail=f"{operation}需要显式 embedding_model")
    if not requested_provider:
        raise HTTPException(status_code=400, detail=f"{operation}需要显式 embedding_provider")

    normalized_model = normalize_embedding_model_id(requested_model)
    if not normalized_model:
        raise HTTPException(
            status_code=400,
            detail=f"{operation}使用的 Embedding 模型未配置或已下线",
        )

    try:
        identity = _canonicalize_embedding_identity(
            normalized_model,
            embedding_provider=requested_provider,
            base_url=requested_host or None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{operation}的 Embedding 配置无效：{exc}",
        ) from exc

    provider = str(identity.get("provider") or "").strip()
    if provider != "local" and not requested_host:
        raise HTTPException(
            status_code=400,
            detail=f"远程 {operation}需要显式 embedding_api_host",
        )
    if provider not in KEYLESS_EMBEDDING_PROVIDERS and not requested_key:
        raise HTTPException(
            status_code=400,
            detail=f"远程 {operation}需要显式 embedding_api_key",
        )

    return {
        "model": str(identity.get("model") or "").strip(),
        "provider": provider,
        "api_host": str(identity.get("api_host") or "").strip(),
        "api_key": requested_key or None,
    }


def _embedding_provider_from_model(embedding_model: Optional[str]) -> Optional[str]:
    raw_model = str(embedding_model or "").strip()
    if ":" not in raw_model:
        return None
    provider_part, _model_part = raw_model.split(":", 1)
    provider = provider_part.strip()
    return provider or None


def _resolve_embedding_provider(
    embedding_model: Optional[str],
    embedding_provider: Optional[str],
) -> Optional[str]:
    return (
        _normalize_optional_provider_id(embedding_provider)
        or _embedding_provider_from_model(embedding_model)
    )


def _compose_provider_scoped_embedding_model(
    embedding_model: Optional[str],
    embedding_provider: Optional[str],
) -> str:
    normalized_model = str(embedding_model or "").strip()
    if not normalized_model:
        return ""
    resolved_provider = _resolve_embedding_provider(normalized_model, embedding_provider)
    if not resolved_provider:
        return normalized_model
    if ":" in normalized_model:
        current_provider, model_part = normalized_model.split(":", 1)
        if current_provider.strip().casefold() == resolved_provider.casefold():
            return normalized_model
        model_part = model_part.strip()
        if model_part:
            return f"{resolved_provider}:{model_part}"
        return normalized_model
    return f"{resolved_provider}:{normalized_model}"


def _callable_accepts_keyword(func, keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def _call_with_optional_keyword(func, keyword: str, value, /, *args, **kwargs):
    if value is not None and _callable_accepts_keyword(func, keyword):
        kwargs[keyword] = value
    return func(*args, **kwargs)


def build_vector_index(
    doc_id: str,
    full_text: str,
    vector_store_dir: str,
    embedding_model: str,
    api_key: Optional[str],
    api_host: Optional[str],
    pages: Optional[list] = None,
    evidence_chunks: Optional[list] = None,
    structured_table_bundles: Optional[list] = None,
    summary_api_key: Optional[str] = None,
    summary_model: str = "gpt-4o-mini",
    summary_provider: str = "openai",
    summary_api_host: str = "",
    index_source: str = "pdf_native",
    index_meta: Optional[dict] = None,
    build_semantic_groups: bool = True,
    embedding_provider: Optional[str] = None,
):
    requested_provider = _resolve_embedding_provider(embedding_model, embedding_provider)
    provider_scoped_model = _compose_provider_scoped_embedding_model(
        embedding_model,
        requested_provider,
    )
    return _call_with_optional_keyword(
        _vector_build_vector_index,
        "embedding_provider",
        requested_provider,
        doc_id,
        full_text,
        vector_store_dir,
        provider_scoped_model,
        api_key,
        None,
        pages=pages,
        evidence_chunks=evidence_chunks,
        structured_table_bundles=structured_table_bundles,
        summary_api_key=summary_api_key,
        summary_model=summary_model,
        summary_provider=summary_provider,
        summary_api_host=summary_api_host,
        index_source=index_source,
        index_meta=index_meta,
        build_semantic_groups=build_semantic_groups,
        embedding_api_host=api_host,
    )


def create_index(
    doc_id: str,
    full_text: str,
    vector_store_dir: str,
    embedding_model: str,
    api_key: Optional[str],
    api_host: Optional[str],
    pages: Optional[list] = None,
    evidence_chunks: Optional[list] = None,
    structured_table_bundles: Optional[list] = None,
    summary_api_key: Optional[str] = None,
    summary_model: str = "gpt-4o-mini",
    summary_provider: str = "openai",
    summary_api_host: str = "",
    index_source: str = "pdf_native",
    index_meta: Optional[dict] = None,
    build_semantic_groups: bool = True,
    embedding_provider: Optional[str] = None,
):
    requested_provider = _resolve_embedding_provider(embedding_model, embedding_provider)
    provider_scoped_model = _compose_provider_scoped_embedding_model(
        embedding_model,
        requested_provider,
    )
    return _call_with_optional_keyword(
        _vector_create_index,
        "embedding_provider",
        requested_provider,
        doc_id,
        full_text,
        vector_store_dir,
        provider_scoped_model,
        api_key,
        None,
        pages=pages,
        evidence_chunks=evidence_chunks,
        structured_table_bundles=structured_table_bundles,
        summary_api_key=summary_api_key,
        summary_model=summary_model,
        summary_provider=summary_provider,
        summary_api_host=summary_api_host,
        index_source=index_source,
        index_meta=index_meta,
        build_semantic_groups=build_semantic_groups,
        embedding_api_host=api_host,
    )


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
            if existing.get("ocr_attempts"):
                merged["ocr_attempts"] = deepcopy(existing["ocr_attempts"])
            if existing.get("ocr_last_error"):
                merged["ocr_last_error"] = existing["ocr_last_error"]
            merged_pages.append(merged)
            preserved_ocr = True
        else:
            merged = dict(odl_page)
            if existing:
                if existing.get("ocr_backend"):
                    merged["ocr_backend"] = existing["ocr_backend"]
                if existing.get("ocr_attempts"):
                    merged["ocr_attempts"] = deepcopy(existing["ocr_attempts"])
                if existing.get("ocr_last_error"):
                    merged["ocr_last_error"] = existing["ocr_last_error"]
            merged_pages.append(merged)
    for page_num, existing in sorted(existing_by_page.items()):
        if page_num not in seen_pages:
            merged_pages.append(dict(existing))
    merged_pages.sort(key=lambda page: int(page.get("page") or 0))
    return merged_pages, preserved_ocr


def save_document(doc_id: str, data: dict) -> bool:
    """Atomically persist document state used by parser publication fences."""
    temp_path: str | None = None
    try:
        file_path = DOCS_DIR / f"{doc_id}.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = f.name
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, file_path)
        logger.debug("Saved document %s to %s", doc_id, file_path)
        return True
    except Exception as e:
        logger.warning("Error saving document %s: %s", doc_id, e)
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
        return False


def _store_new_document_or_raise(
    doc_id: str,
    document: dict,
    *,
    previous: dict | None,
    message: str,
) -> None:
    """Persist a freshly uploaded document, rolling memory back if the write fails.

    ``save_document`` reports failure by returning False and logging a warning.
    Ignoring that leaves ``documents_store`` advertising a document that never
    reached disk: the upload answers 200, the record is gone after a restart,
    and ``resume_pending_mineru_deep_parse_jobs`` skips the doc entirely because
    it keys off ``documents_store`` — so an already-paid remote MinerU batch is
    never collected. Every other publication fence in this module already
    rolls back and raises on a failed write; these upload paths did not.
    """
    if save_document(doc_id, document):
        return
    if previous is None:
        documents_store.pop(doc_id, None)
    else:
        documents_store[doc_id] = previous
    raise RuntimeError(message)


def _is_document_backup_file(file_path: str | Path) -> bool:
    """Return whether a JSON file is an internal RAG rollback snapshot."""
    return Path(file_path).stem.lower().endswith(".bak.doc")


def _persist_inferred_parse_manifest(doc_id: str, document: dict) -> bool:
    """Freeze a legacy document identity before runtime services can mutate pages."""
    data = document.get("data") if isinstance(document, dict) else None
    if not isinstance(data, dict) or isinstance(data.get("parse_manifest"), dict):
        return False

    manifest = read_parse_manifest(document, doc_id=doc_id)
    metadata = dict(manifest.get("metadata") or {})
    if isinstance(data.get("document_parse_state"), dict):
        metadata["migrated_from_document_parse_state"] = True
    else:
        # Keep legacy_inferred for old block/MinerU compatibility, while the
        # persisted manifest makes its generation immutable from this point on.
        metadata["legacy_persisted"] = True
    manifest["metadata"] = metadata
    data["parse_manifest"] = manifest
    data.pop("document_parse_state", None)
    return True


def load_documents():
    logger.info("Loading documents from disk...")
    count = 0
    migrated_count = 0
    skipped_backup_count = 0
    for file_path in glob.glob(str(DOCS_DIR / "*.json")):
        if _is_document_backup_file(file_path):
            skipped_backup_count += 1
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
            doc_id = os.path.splitext(os.path.basename(file_path))[0]
            migrated = _persist_inferred_parse_manifest(doc_id, data)
            _normalize_page_keys(data)
            documents_store[doc_id] = data
            if migrated:
                if save_document(doc_id, data):
                    migrated_count += 1
                else:
                    logger.warning("Failed to persist inferred parse manifest for %s", doc_id)
            count += 1
        except Exception as e:
            logger.warning("Error loading document from %s: %s", file_path, e)
    logger.info(
        "Loaded %s documents (migrated=%s, skipped_backups=%s).",
        count,
        migrated_count,
        skipped_backup_count,
    )


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
    had_previous_manifest = "parse_manifest" in data
    previous_manifest = data.get("parse_manifest")
    data["parse_manifest"] = dict(manifest)
    if persist and not save_document(doc_id, target):
        if had_previous_manifest:
            data["parse_manifest"] = previous_manifest
        else:
            data.pop("parse_manifest", None)
        raise RuntimeError("解析状态写入失败")
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
    metadata: dict | None = None,
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
            metadata=metadata,
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


def _parse_manifest_index_matches(
    doc_id: str,
    manifest: dict,
    artifact_validation: dict | None = None,
) -> bool:
    """Whether the active vector pair belongs to the document's active parse run."""
    artifact_validation = artifact_validation or _inspect_vector_index_artifacts(doc_id, VECTOR_STORE_DIR)
    if not artifact_validation.get("valid"):
        return False
    metadata = _read_vector_index_meta(doc_id, artifact_validation=artifact_validation)
    if metadata.get("upgrade_required"):
        return False
    # A parser adapter repair can replace the block tree without changing the
    # selected parse generation.  Vector chunks, semantic groups and the
    # reading UI must all consume the same published block revision, otherwise
    # one document can answer from stale section ids after a rebuild.
    return parse_identity_matches(
        metadata.get("index_meta") or {},
        manifest,
        block_index=load_block_index(DATA_DIR, doc_id),
    )


def _active_rag_index_matches_current_parse(doc_id: str, manifest: dict | None = None) -> bool:
    """Return whether the active vector pair is safe to preserve as a rollback point."""
    current_manifest = manifest or _read_document_parse_manifest(
        doc_id,
        documents_store.get(doc_id),
    )
    # Documents predating parse manifests cannot prove a modern identity. Keep
    # their historical rollback behavior until they are explicitly re-parsed.
    if _is_legacy_parse_manifest(current_manifest):
        return _vector_index_ready(doc_id)
    return _parse_manifest_index_matches(doc_id, current_manifest)


def _warm_block_index(doc_id: str) -> dict | None:
    """Best-effort block index build; upload/search must not fail because of it."""
    try:
        doc = documents_store.get(doc_id)
        if not doc:
            return None
        return ensure_block_index(
            doc_id=doc_id,
            doc=doc,
            data_dir=DATA_DIR,
            pdf_path=_resolve_document_pdf_path(doc),
        )
    except Exception as exc:
        logger.warning("[BlockIndex] warm build failed for %s: %s", doc_id, exc)
        return None


_RAG_EXCLUDED_BLOCK_TYPES = {
    "artifact",
    "figure",
    "image",
    "table",
    "table_row",
    "table_cell",
}


def _rag_source_from_block_index(block_index: dict | None, data: dict) -> dict:
    """Build RAG input from the canonical reading blocks when possible."""
    fallback_pages = data.get("pages") or []
    fallback_text = str(data.get("full_text") or "")
    source = build_rag_source_from_block_index(block_index)
    if not str(source.get("full_text") or "").strip() or not source["pages"]:
        return {
            "full_text": fallback_text,
            "pages": fallback_pages,
            "evidence_chunks": [],
            "block_count": 0,
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        }
    return source


def _vector_index_ready(doc_id: str) -> bool:
    return bool(_inspect_vector_index_artifacts(doc_id, VECTOR_STORE_DIR).get("valid"))


def _vector_index_paths(doc_id: str, base_dir: Path | None = None) -> tuple[Path, Path]:
    root = base_dir or VECTOR_STORE_DIR
    return root / f"{doc_id}.index", root / f"{doc_id}.pkl"


def _inspect_vector_index_artifacts(
    doc_id: str,
    base_dir: Path,
    *,
    expected_source: str = "",
    expected_parse_generation: str = "",
    expected_document_source_hash: str = "",
    expected_block_index_hash: str = "",
    expected_content_source: str = "",
    expected_evidence_schema_version: int = 0,
) -> dict:
    """Load and verify a vector artifact pair instead of trusting file existence."""
    index_path, chunks_path = _vector_index_paths(doc_id, base_dir)
    failures: list[str] = []
    data = None
    chunks: list = []
    vector_count = 0
    vector_dimension = 0

    try:
        index_size = index_path.stat().st_size
    except FileNotFoundError:
        index_size = -1
    except OSError as exc:
        index_size = -1
        logger.warning("[RagIndex] index stat failed doc=%s error=%s", doc_id, exc)
        failures.append("index_stat_failed")
    try:
        chunks_size = chunks_path.stat().st_size
    except FileNotFoundError:
        chunks_size = -1
    except OSError as exc:
        chunks_size = -1
        logger.warning("[RagIndex] pkl stat failed doc=%s error=%s", doc_id, exc)
        failures.append("pkl_stat_failed")

    if index_size < 0:
        failures.append("index_missing")
    elif index_size == 0:
        failures.append("index_empty_file")
    if chunks_size < 0:
        failures.append("pkl_missing")
    elif chunks_size == 0:
        failures.append("pkl_empty_file")

    if chunks_size > 0:
        try:
            with open(chunks_path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            logger.warning("[RagIndex] pkl unreadable doc=%s error=%s", doc_id, exc)
            failures.append(f"pkl_unreadable:{type(exc).__name__}")

    if isinstance(data, dict):
        raw_chunks = data.get("chunks")
        if isinstance(raw_chunks, list):
            chunks = raw_chunks
        else:
            failures.append("chunks_invalid_shape")
        index_meta = data.get("index_meta") if isinstance(data.get("index_meta"), dict) else {}
        actual_source = str(data.get("index_source") or "").strip()
        if expected_source and actual_source != expected_source:
            failures.append(f"index_source_mismatch:{actual_source or 'missing'}")
        actual_generation = str(index_meta.get("parse_generation") or "").strip()
        actual_source_hash = str(index_meta.get("document_source_hash") or "").strip()
        actual_block_index_hash = str(index_meta.get("block_index_hash") or "").strip()
        actual_content_source = str(index_meta.get("content_source") or "").strip()
        try:
            actual_evidence_schema_version = int(index_meta.get("evidence_schema_version") or 0)
        except (TypeError, ValueError):
            actual_evidence_schema_version = 0
        if expected_parse_generation and actual_generation != expected_parse_generation:
            failures.append("parse_generation_mismatch")
        if expected_document_source_hash and actual_source_hash != expected_document_source_hash:
            failures.append("document_source_hash_mismatch")
        if expected_block_index_hash and actual_block_index_hash != expected_block_index_hash:
            failures.append("block_index_hash_mismatch")
        if expected_content_source and actual_content_source != expected_content_source:
            failures.append(
                f"content_source_mismatch:{actual_content_source or 'missing'}"
            )
        if (
            expected_evidence_schema_version
            and actual_evidence_schema_version != expected_evidence_schema_version
        ):
            failures.append("evidence_schema_version_mismatch")
    elif isinstance(data, list):
        chunks = data
        if (
            expected_source
            or expected_parse_generation
            or expected_document_source_hash
            or expected_block_index_hash
            or expected_content_source
            or expected_evidence_schema_version
        ):
            failures.append("legacy_pkl_missing_identity")
    elif data is not None:
        failures.append("pkl_invalid_shape")

    if not chunks:
        failures.append("chunks_empty")
    elif any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
        failures.append("chunks_contain_blank_text")

    if index_size > 0:
        try:
            import faiss

            persisted_index = faiss.read_index(str(index_path))
            vector_count = int(persisted_index.ntotal)
            vector_dimension = int(persisted_index.d)
        except Exception as exc:
            logger.warning("[RagIndex] FAISS index unreadable doc=%s error=%s", doc_id, exc)
            failures.append(f"index_unreadable:{type(exc).__name__}")

    if vector_count <= 0:
        failures.append("vector_count_empty")
    if vector_dimension <= 0:
        failures.append("vector_dimension_invalid")
    if chunks and vector_count > 0 and vector_count != len(chunks):
        failures.append(f"vector_chunk_count_mismatch:{vector_count}:{len(chunks)}")

    if isinstance(data, dict):
        stored_vector_count = data.get("vector_count")
        stored_vector_dimension = data.get("vector_dimension")
        if stored_vector_count is not None:
            try:
                if int(stored_vector_count) != vector_count:
                    failures.append("stored_vector_count_mismatch")
            except (TypeError, ValueError):
                failures.append("stored_vector_count_invalid")
        if stored_vector_dimension is not None:
            try:
                if int(stored_vector_dimension) != vector_dimension:
                    failures.append("stored_vector_dimension_mismatch")
            except (TypeError, ValueError):
                failures.append("stored_vector_dimension_invalid")
        build_validation = data.get("build_validation")
        if isinstance(build_validation, dict) and build_validation.get("valid") is not True:
            failures.append("build_validation_failed")
        try:
            index_version = int(data.get("index_version") or 0)
        except (TypeError, ValueError):
            index_version = 0
        if (
            index_version == RAG_INDEX_VERSION
            and not _semantic_generation_identity_complete(
                _extract_vector_semantic_identity(data)
            )
        ):
            failures.append("embedding_build_identity_incomplete")

    failures = list(dict.fromkeys(failures))
    return {
        "valid": not failures,
        "errors": failures,
        "chunk_count": len(chunks),
        "vector_count": vector_count,
        "vector_dimension": vector_dimension,
        "index_path": str(index_path),
        "chunks_path": str(chunks_path),
        "_data": data,
    }


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


def _read_vector_index_meta(
    doc_id: str,
    base_dir: Path | None = None,
    *,
    artifact_validation: dict | None = None,
) -> dict:
    artifact_validation = artifact_validation or _inspect_vector_index_artifacts(
        doc_id,
        base_dir or VECTOR_STORE_DIR,
    )
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
        return {
            "index_source": "pdf_native",
            "index_version": 0,
            "upgrade_required": True,
            "vector_count": artifact_validation.get("vector_count", 0),
            "vector_dimension": artifact_validation.get("vector_dimension", 0),
            "artifact_valid": bool(artifact_validation.get("valid")),
            "artifact_errors": list(artifact_validation.get("errors") or []),
        }
    index_meta = data.get("index_meta") if isinstance(data.get("index_meta"), dict) else {}
    try:
        index_version = int(data.get("index_version") or 0)
    except (TypeError, ValueError):
        index_version = 0
    parse_generation = str(index_meta.get("parse_generation") or "").strip()
    document_source_hash = str(index_meta.get("document_source_hash") or "").strip()
    parse_identity_complete = bool(parse_generation and document_source_hash)
    semantic_identity = _extract_vector_semantic_identity(data)
    semantic_identity_complete = _semantic_generation_identity_complete(semantic_identity)
    identity_complete = (
        parse_identity_complete and semantic_identity_complete
        if index_version == RAG_INDEX_VERSION
        else parse_identity_complete
    )
    try:
        expected_page_count = int(index_meta.get("expected_page_count") or 0)
    except (TypeError, ValueError):
        expected_page_count = 0
    try:
        coverage = float(index_meta.get("coverage") or 0.0)
    except (TypeError, ValueError):
        coverage = 0.0
    failed_pages = index_meta.get("failed_pages")
    if not isinstance(failed_pages, list):
        failed_pages = []
    return {
        "index_version": index_version,
        "expected_index_version": RAG_INDEX_VERSION,
        "upgrade_required": index_version != RAG_INDEX_VERSION or not identity_complete,
        "identity_complete": identity_complete,
        "parse_identity_complete": parse_identity_complete,
        "semantic_identity_complete": semantic_identity_complete,
        "index_source": data.get("index_source") or "pdf_native",
        "source_hash": data.get("source_hash") or "",
        "rebuilt_at": data.get("rebuilt_at") or "",
        "previous_index_source": data.get("previous_index_source") or "",
        "normalizer_version": data.get("normalizer_version") or "",
        "parse_generation": parse_generation,
        "document_source_hash": document_source_hash,
        "quality_status": str(index_meta.get("quality_status") or "success"),
        "expected_page_count": expected_page_count,
        "coverage": coverage,
        "failed_pages": list(failed_pages),
        "index_meta": index_meta,
        "chunk_count": len(data.get("chunks") or []),
        "vector_count": artifact_validation.get("vector_count", 0),
        "vector_dimension": artifact_validation.get("vector_dimension", 0),
        "artifact_valid": bool(artifact_validation.get("valid")),
        "artifact_errors": list(artifact_validation.get("errors") or []),
        "vector_build_id": semantic_identity.get("vector_build_id", ""),
        "embedding_identity_version": int(semantic_identity.get("embedding_identity_version") or 0),
        "embedding_model": semantic_identity.get("embedding_model", ""),
        "embedding_provider": semantic_identity.get("embedding_provider", ""),
        "embedding_api_host": semantic_identity.get("embedding_api_host", ""),
        "table_chunk_count": sum(
            1
            for item in (data.get("chunk_metadata") or [])
            if isinstance(item, dict) and item.get("structured_table_bundle")
        ),
    }


def _get_rag_index_status(doc_id: str, artifact_validation: dict | None = None) -> dict:
    if artifact_validation is None:
        with _INDEX_STATUS_LOCK:
            active_lifecycle = dict(_DOCUMENT_INDEX_STATUS.get(doc_id) or {})
        active_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
        lifecycle_matches_parse = bool(active_lifecycle) and matches_parse_generation(
            active_manifest,
            generation=str(active_lifecycle.get("parse_generation") or ""),
            source_hash=str(active_lifecycle.get("document_source_hash") or ""),
        )
        if lifecycle_matches_parse and active_lifecycle.get("status") in {"queued", "running"}:
            index_path, chunks_path = _vector_index_paths(doc_id)
            return {
                "status": str(active_lifecycle.get("status") or "queued"),
                "stage": str(active_lifecycle.get("stage") or "queued"),
                "error": str(active_lifecycle.get("error") or ""),
                "created_at": str(active_lifecycle.get("created_at") or ""),
                "started_at": str(active_lifecycle.get("started_at") or ""),
                "stage_started_at": str(active_lifecycle.get("stage_started_at") or ""),
                "updated_at": str(active_lifecycle.get("updated_at") or ""),
                "ready": False,
                "artifact_ready": index_path.exists() and chunks_path.exists(),
                "index_version": 0,
                "expected_index_version": RAG_INDEX_VERSION,
                "upgrade_required": True,
                "identity_complete": False,
                "parse_generation": str(active_lifecycle.get("parse_generation") or ""),
                "document_source_hash": str(active_lifecycle.get("document_source_hash") or ""),
                "matches_active_parse": False,
                "can_rollback": False,
            }
    artifact_validation = artifact_validation or _inspect_vector_index_artifacts(doc_id, VECTOR_STORE_DIR)
    ready = bool(artifact_validation.get("valid"))
    artifact_errors = list(artifact_validation.get("errors") or [])
    artifact_present = not (
        "index_missing" in artifact_errors
        and "pkl_missing" in artifact_errors
    )
    # Invalid artifacts remain unusable, but readable metadata is still useful
    # for distinguishing a stale index from one that was never built.
    meta = (
        _read_vector_index_meta(doc_id, artifact_validation=artifact_validation)
        if ready or artifact_present
        else {}
    )
    upgrade_required = bool(meta.get("upgrade_required"))
    identity_complete = bool(meta.get("identity_complete"))
    parse_identity_complete = bool(meta.get("parse_identity_complete"))
    semantic_identity_complete = bool(meta.get("semantic_identity_complete"))
    quality_status = str(meta.get("quality_status") or "success")
    source = meta.get("index_source") or ("pdf_native" if ready else "")
    parse_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    matches_active_parse = (
        _parse_manifest_index_matches(doc_id, parse_manifest, artifact_validation)
        if ready
        else False
    )
    with _INDEX_STATUS_LOCK:
        lifecycle = dict(_DOCUMENT_INDEX_STATUS.get(doc_id) or {})
    lifecycle_matches_active_parse = bool(lifecycle) and matches_parse_generation(
        parse_manifest,
        generation=str(lifecycle.get("parse_generation") or ""),
        source_hash=str(lifecycle.get("document_source_hash") or ""),
    )
    if not lifecycle_matches_active_parse:
        lifecycle = {}
    # Artifact validity is authoritative over a stale in-memory lifecycle entry.
    # Without this override a previous "ready" status can mask an identityless
    # index until the process restarts.
    preserve_active_lifecycle = str(lifecycle.get("status") or "") in {"queued", "running", "failed"}
    if ready and not identity_complete and not preserve_active_lifecycle:
        lifecycle = {
            "status": "stale",
            "stage": "parse_identity_missing",
            "error": "问答索引缺少解析身份，需要按当前解析结果重建",
        }
    elif ready and upgrade_required and not preserve_active_lifecycle:
        lifecycle = {
            "status": "stale",
            "stage": "index_version_mismatch",
            "error": "问答索引格式已升级，需要按当前解析结果重建",
        }
    elif (
        ready
        and matches_active_parse
        and quality_status == "partial_success"
        and not preserve_active_lifecycle
    ):
        lifecycle = {
            "status": "partial_ready",
            "stage": "partial_ready",
            "error": "部分页面解析失败，问答仅基于已成功解析的页面",
        }
    elif not lifecycle:
        if ready and matches_active_parse:
            lifecycle = {"status": "ready", "stage": "ready", "error": ""}
        elif ready:
            lifecycle = {
                "status": "stale",
                "stage": "parse_generation_mismatch",
                "error": "现有问答索引不属于当前解析代际",
            }
        elif artifact_present and not identity_complete and meta:
            lifecycle = {
                "status": "stale",
                "stage": "parse_identity_missing",
                "error": "问答索引缺少解析身份，需要按当前解析结果重建",
            }
        elif artifact_present and upgrade_required and meta:
            lifecycle = {
                "status": "stale",
                "stage": "index_version_mismatch",
                "error": "问答索引格式已升级，需要按当前解析结果重建",
            }
        elif artifact_present:
            lifecycle = {
                "status": "failed",
                "stage": "artifact_validation_failed",
                "error": "问答索引工件损坏或不完整，需要重新构建",
            }
        else:
            lifecycle = {"status": "missing", "stage": "not_started", "error": ""}
    return {
        "status": str(lifecycle.get("status") or "missing"),
        "stage": str(lifecycle.get("stage") or "not_started"),
        "error": str(lifecycle.get("error") or ""),
        "created_at": str(lifecycle.get("created_at") or ""),
        "started_at": str(lifecycle.get("started_at") or ""),
        "stage_started_at": str(lifecycle.get("stage_started_at") or ""),
        "updated_at": str(lifecycle.get("updated_at") or ""),
        "ready": ready and matches_active_parse and not upgrade_required,
        "artifact_ready": ready,
        "index_source": source,
        "source_hash": meta.get("source_hash", ""),
        "rebuilt_at": meta.get("rebuilt_at", ""),
        "previous_index_source": meta.get("previous_index_source", ""),
        "normalizer_version": meta.get("normalizer_version", ""),
        "index_version": meta.get("index_version", 0),
        "expected_index_version": meta.get("expected_index_version", RAG_INDEX_VERSION),
        "upgrade_required": upgrade_required,
        "identity_complete": identity_complete,
        "parse_identity_complete": parse_identity_complete,
        "semantic_identity_complete": semantic_identity_complete,
        "parse_generation": meta.get("parse_generation", ""),
        "document_source_hash": meta.get("document_source_hash", ""),
        "vector_build_id": meta.get("vector_build_id", ""),
        "embedding_identity_version": meta.get("embedding_identity_version", 0),
        "embedding_model": meta.get("embedding_model", ""),
        "embedding_provider": meta.get("embedding_provider", ""),
        "embedding_api_host": meta.get("embedding_api_host", ""),
        "quality_status": quality_status,
        "expected_page_count": meta.get("expected_page_count", 0),
        "coverage": meta.get("coverage", 0.0),
        "failed_pages": meta.get("failed_pages", []),
        "matches_active_parse": matches_active_parse,
        "chunk_count": meta.get("chunk_count", artifact_validation.get("chunk_count", 0)),
        "vector_count": meta.get("vector_count", artifact_validation.get("vector_count", 0)),
        "vector_dimension": meta.get("vector_dimension", artifact_validation.get("vector_dimension", 0)),
        "artifact_errors": meta.get("artifact_errors", artifact_errors),
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
    timestamp = utc_now_iso_ms()
    next_parse_generation = str(parse_generation or manifest.get("generation") or "")
    next_document_source_hash = str(document_source_hash or manifest.get("source_hash") or "")
    with _INDEX_STATUS_LOCK:
        current = dict(_DOCUMENT_INDEX_STATUS.get(doc_id) or {})
        if current and (
            str(current.get("parse_generation") or "") != next_parse_generation
            or str(current.get("document_source_hash") or "") != next_document_source_hash
        ):
            current = {}
        previous_stage = str(current.get("stage") or "").strip().lower()
        normalized_stage = str(stage or "").strip().lower()
        _DOCUMENT_INDEX_STATUS[doc_id] = {
            "doc_id": doc_id,
            "status": status,
            "stage": stage,
            "error": error,
            "parse_generation": next_parse_generation,
            "document_source_hash": next_document_source_hash,
            "created_at": str(current.get("created_at") or timestamp),
            "started_at": str(
                current.get("started_at")
                or (timestamp if status in {"queued", "running"} else "")
            ),
            "stage_started_at": str(
                timestamp
                if normalized_stage and normalized_stage != previous_stage
                else current.get("stage_started_at") or timestamp
            ),
            "updated_at": timestamp,
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
    if current.get("status") in {"queued", "running"}:
        index_path, chunks_path = _vector_index_paths(doc_id)
        artifact_present = index_path.exists() and chunks_path.exists()
        lifecycle = {
            "status": str(current.get("status") or "queued"),
            "stage": str(current.get("stage") or "queued"),
            "error": str(current.get("error") or ""),
            "ready": False,
            "artifact_ready": artifact_present,
            "index_version": 0,
            "expected_index_version": RAG_INDEX_VERSION,
            "upgrade_required": True,
            "identity_complete": False,
            "parse_generation": str(current.get("parse_generation") or ""),
            "document_source_hash": str(current.get("document_source_hash") or ""),
            "matches_active_parse": False,
        }
        return {
            **current,
            "vector_ready": False,
            "vector_artifact_ready": artifact_present,
            "parse_manifest": parse_manifest,
            "rag_index": lifecycle,
        }
    artifact_validation = _inspect_vector_index_artifacts(doc_id, VECTOR_STORE_DIR)
    artifact_ready = bool(artifact_validation.get("valid"))
    vector_matches_parse = _parse_manifest_index_matches(doc_id, parse_manifest, artifact_validation)
    vector_meta = (
        _read_vector_index_meta(doc_id, artifact_validation=artifact_validation)
        if artifact_ready
        else {}
    )
    rag_index_status = _get_rag_index_status(doc_id, artifact_validation)
    partial_ready = str(vector_meta.get("quality_status") or "") == "partial_success"
    if artifact_ready and vector_matches_parse:
        if current.get("status") not in {"queued", "running", "failed"}:
            current.update({
                "doc_id": doc_id,
                "status": "partial_ready" if partial_ready else "ready",
                "stage": "partial_ready" if partial_ready else "ready",
                "error": (
                    "部分页面解析失败，问答仅基于已成功解析的页面"
                    if partial_ready
                    else ""
                ),
            })
    elif artifact_ready and not vector_matches_parse:
        if current.get("status") not in {"queued", "running", "failed"}:
            current = {
                "doc_id": doc_id,
                "status": "stale",
                "stage": "parse_generation_mismatch",
                "error": "现有问答索引不属于当前解析代际",
            }
    elif not current:
        current = {
            "doc_id": doc_id,
            "status": rag_index_status.get("status", "missing"),
            "stage": rag_index_status.get("stage", "not_started"),
            "error": rag_index_status.get("error", ""),
        }
    current["vector_ready"] = artifact_ready and vector_matches_parse
    current["vector_artifact_ready"] = artifact_ready
    current["parse_manifest"] = parse_manifest
    current["rag_index"] = rag_index_status
    return current


def _build_document_indexes(
    doc_id: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_api_host: Optional[str],
    summary_api_key: Optional[str],
    embedding_provider: Optional[str] = None,
) -> None:
    document_lock = _get_document_operation_lock(doc_id)
    document_lock.acquire()
    parse_manifest: dict = {}
    failure_stage = "initializing"
    vector_stage = VECTOR_STORE_DIR / "_tmp" / f"{doc_id}.background.{uuid.uuid4().hex}"
    semantic_stage = DATA_DIR / "semantic_groups" / "_tmp" / f"{doc_id}.background.{uuid.uuid4().hex}"
    try:
        failure_stage = "loading_document"
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

        _require_current_parse_generation(
            doc_id,
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        )

        failure_stage = "building_block_index"
        _set_document_index_status(
            doc_id,
            "running",
            stage="block_index",
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        )
        block_index = _warm_block_index(doc_id)

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
        rag_source = _rag_source_from_block_index(block_index, data)
        if not str(rag_source.get("full_text") or "").strip() or not (rag_source.get("pages") or []):
            raise RuntimeError("当前解析代际没有可用于构建问答索引的正文")
        block_index_hash = str(
            (block_index or {}).get("block_index_hash")
            or (block_index or {}).get("block_index_revision")
            or ""
        ).strip()
        if not block_index_hash:
            raise RuntimeError("当前阅读块索引缺少内容修订，拒绝构建问答索引")
        expected_index_source = (
            MINERU_RAG_INDEX_SOURCE
            if parse_manifest.get("resolved_route") == PARSE_ROUTE_MINERU
            else "pdf_native"
        )
        vector_stage.mkdir(parents=True, exist_ok=True)
        semantic_stage.mkdir(parents=True, exist_ok=True)

        failure_stage = "building_vector_index_staging"
        _set_document_index_status(
            doc_id,
            "running",
            stage="building_vector_index_staging",
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        )
        background_index_meta = {
            "source_hash": data.get("rag_source_hash") or parse_manifest.get("source_hash", ""),
            "document_source_hash": parse_manifest.get("source_hash", ""),
            "parse_generation": parse_manifest.get("generation", ""),
            "parser_route": parse_manifest.get("resolved_route", ""),
            "content_source": "block_index_evidence" if rag_source["evidence_chunks"] else "document_full_text",
            "block_index_version": (block_index or {}).get("version", ""),
            "block_index_hash": block_index_hash,
            "evidence_schema_version": rag_source.get("evidence_schema_version", 0),
        }
        background_index_kwargs = {
            "pages": rag_source["pages"],
            "structured_table_bundles": data.get("structured_table_bundles"),
            "summary_api_key": summary_api_key,
            "index_source": expected_index_source,
            "index_meta": background_index_meta,
            "build_semantic_groups": False,
        }
        if rag_source["evidence_chunks"] and _callable_accepts_keyword(create_index, "evidence_chunks"):
            background_index_kwargs["evidence_chunks"] = rag_source["evidence_chunks"]
        _call_with_optional_keyword(
            create_index,
            "embedding_provider",
            _normalize_optional_provider_id(embedding_provider),
            doc_id,
            rag_source["full_text"],
            str(vector_stage),
            embedding_model,
            embedding_api_key,
            embedding_api_host,
            **background_index_kwargs,
        )

        failure_stage = "validating_vector_index_staging"
        vector_ok, vector_failures = _validate_temp_vector_index(
            doc_id,
            vector_stage,
            expected_source=expected_index_source,
            expected_parse_generation=parse_generation,
            expected_document_source_hash=parse_source_hash,
            expected_block_index_hash=block_index_hash,
            expected_content_source=background_index_meta["content_source"],
            expected_evidence_schema_version=background_index_meta["evidence_schema_version"],
        )
        if not vector_ok:
            raise RuntimeError("后台问答索引质量门失败: " + ", ".join(vector_failures))

        # Semantic groups are optional, but they may only be derived from the
        # validated vector staging pair for this exact parse generation.
        semantic_validation = None
        semantic_error = ""
        try:
            failure_stage = "building_semantic_groups_staging"
            _set_document_index_status(
                doc_id,
                "running",
                stage=failure_stage,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )
            semantic_rebuild = _prepare_semantic_group_rebuild(
                doc_id,
                vector_stage,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
                embedding_api_host=embedding_api_host,
                summary_api_key=summary_api_key,
                expected_source=expected_index_source,
                expected_parse_generation=parse_generation,
                expected_document_source_hash=parse_source_hash,
                expected_block_index_hash=block_index_hash,
                embedding_provider=embedding_provider,
            )
            semantic_result = _build_semantic_group_index(
                doc_id,
                semantic_rebuild["chunks"],
                rag_source["pages"],
                semantic_rebuild["embed_fn"],
                semantic_rebuild["api_key"],
                chunk_pages=semantic_rebuild["chunk_pages"],
                chunk_types=semantic_rebuild["chunk_types"],
                chunk_metadata=semantic_rebuild["chunk_metadata"],
                model=semantic_rebuild["model"],
                provider=semantic_rebuild["provider"],
                endpoint=semantic_rebuild["endpoint"],
                output_dir=str(semantic_stage),
                raise_on_error=True,
                semantic_identity=semantic_rebuild["semantic_identity"],
            )
            semantic_validation = _validate_temp_semantic_groups(
                doc_id,
                semantic_stage,
                semantic_result,
                expected_identity=semantic_rebuild["semantic_identity"],
                expected_vector_dimension=int(semantic_rebuild["semantic_identity"].get("vector_dimension") or 0),
            )
        except Exception as semantic_exc:
            semantic_error = str(semantic_exc)
            logger.warning("[Upload] semantic groups unavailable for %s: %s", doc_id, semantic_exc)

        failure_stage = "publishing_index_generation"
        _set_document_index_status(
            doc_id,
            "running",
            stage=failure_stage,
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        )
        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )
            _replace_vector_index_from_temp(doc_id, vector_stage)
            if semantic_validation:
                try:
                    _publish_temp_semantic_groups(
                        doc_id,
                        semantic_stage,
                        semantic_validation,
                        source_hash=parse_source_hash,
                        transaction_id=parse_generation,
                        semantic_identity=semantic_rebuild["semantic_identity"],
                    )
                except Exception as semantic_publish_exc:
                    semantic_error = str(semantic_publish_exc)
                    logger.warning(
                        "[Upload] semantic groups publish failed for %s: %s",
                        doc_id,
                        semantic_publish_exc,
                    )
                    _remove_current_semantic_groups(doc_id)
            else:
                # Never leave a previous parse generation's semantic groups
                # beside the newly published vector index.
                _remove_current_semantic_groups(doc_id)

        if not _vector_index_ready(doc_id) or not _parse_manifest_index_matches(doc_id, parse_manifest):
            raise RuntimeError("后台问答索引发布后验真失败")
        _set_document_index_status(
            doc_id,
            "ready",
            stage="ready",
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
        )
        if semantic_error:
            logger.warning("[Upload] background index ready without semantic groups for %s", doc_id)
        logger.info("[Upload] background index ready for %s", doc_id)
    except Exception as exc:
        logger.exception("[Upload] background index failed for %s: %s", doc_id, exc)
        _set_document_index_status(
            doc_id,
            "failed",
            stage=failure_stage,
            error=str(exc),
            parse_generation=str(parse_manifest.get("generation") or ""),
            document_source_hash=str(parse_manifest.get("source_hash") or ""),
        )
    finally:
        shutil.rmtree(vector_stage, ignore_errors=True)
        shutil.rmtree(semantic_stage, ignore_errors=True)
        document_lock.release()


def _queue_document_indexes(
    doc_id: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_api_host: Optional[str],
    summary_api_key: Optional[str],
    embedding_provider: Optional[str] = None,
) -> dict:
    parse_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    if not is_parse_prepared(parse_manifest):
        _set_document_index_status(doc_id, "queued", stage="waiting_for_primary_parser")
        return _get_document_index_status(doc_id)
    current = _get_document_index_status(doc_id)
    if current.get("status") in {"queued", "running", "ready"}:
        return current

    if not _DOCUMENT_INDEX_ADMISSION.acquire(blocking=False):
        _set_document_index_status(
            doc_id,
            "failed",
            stage="queue_full",
            error="文档索引任务繁忙，请稍后重试",
        )
        return _get_document_index_status(doc_id)

    _set_document_index_status(doc_id, "queued", stage="queued")
    def _run_admitted_index() -> None:
        try:
            _call_with_optional_keyword(
                _build_document_indexes,
                "embedding_provider",
                _normalize_optional_provider_id(embedding_provider),
                doc_id,
                embedding_model,
                embedding_api_key,
                embedding_api_host,
                summary_api_key,
            )
        finally:
            _DOCUMENT_INDEX_ADMISSION.release()

    try:
        thread = threading.Thread(
            target=_run_admitted_index,
            name=f"chatpdf-index-{doc_id[:8]}",
            daemon=True,
        )
        thread.start()
    except Exception:
        _DOCUMENT_INDEX_ADMISSION.release()
        _set_document_index_status(
            doc_id,
            "failed",
            stage="start_failed",
            error="文档索引后台任务启动失败",
        )
    return _get_document_index_status(doc_id)


def _document_has_indexable_content(doc: dict) -> bool:
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, dict):
        return False
    return bool(str(data.get("full_text") or "").strip() and isinstance(data.get("pages"), list) and data["pages"])


def queue_stale_document_index_upgrades() -> dict:
    """Report stale indexes at startup without changing their embedding space.

    Older code rebuilt every stale artifact with a hard-coded local model.  A
    backend restart could therefore replace a valid remote index and make the
    next query fail with an embedding-identity conflict.  Startup has no
    access to the user's live credential and must only report the work that
    needs an explicit, confirmed rebuild from the client.
    """
    candidates: list[str] = []
    for doc_id, doc in sorted(documents_store.items()):
        if not _document_has_indexable_content(doc):
            continue
        manifest = _read_document_parse_manifest(doc_id, doc)
        if not is_parse_prepared(manifest):
            continue
        status = _get_document_index_status(doc_id)
        if status.get("status") not in {"ready", "queued", "running"}:
            candidates.append(doc_id)

    return {
        "status": "explicit_rebuild_required" if candidates else "ready",
        "documents": candidates,
    }


def _mineru_configuration_error(config: dict | None = None) -> str:
    """Return an actionable configuration error, or an empty string when ready."""
    config = config if isinstance(config, dict) else _load_online_ocr_config("mineru")
    access_mode = str(config.get("access_mode") or "worker").strip().lower()
    worker_url = str(config.get("worker_url") or "").strip()
    token_mode = str(config.get("token_mode") or "frontend").strip().lower()
    token = str(config.get("token") or "").strip()
    if access_mode == "direct":
        return "" if token else "当前后端实例的 MinerU 直连 Token 为空，请在 OCR 设置中重新填写并保存"
    if access_mode != "worker":
        return "MinerU 访问模式无效，请在 OCR 设置中重新选择直连或 Worker"
    if not worker_url:
        return "当前后端实例的 MinerU Worker URL 为空，请在 OCR 设置中填写并保存"
    if token_mode == "frontend" and not token:
        return "当前后端实例的 MinerU Token 为空，请在 OCR 设置中重新填写并保存"
    return ""


def _mineru_configured() -> bool:
    return not _mineru_configuration_error()


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
            base_url = validate_mineru_direct_api_base_url(base_url).rstrip("/")
        except ValueError as exc:
            return False, str(exc)
        try:
            with create_mineru_direct_http_client(
                timeout_seconds=15.0,
                connect_timeout_seconds=10.0,
            ) as client:
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


def _clear_block_bound_reading_cache(doc_id: str) -> list[str]:
    """Delete artifacts whose evidence ids are coupled to the block index."""
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
    return removed


def _clear_block_dependent_ai_cache(doc_id: str, doc: dict | None = None) -> list[str]:
    """Delete AI artifacts that bind to old block ids after deep parsing."""
    # Rotate the write fence before deleting files. This covers same-generation
    # adapter rebuilds as well as route switches: a request that started with
    # the old block ids may finish later, but it can no longer republish its
    # result into the newly rebuilt document.
    removed: list[str] = []
    # Fail closed here. Warning and carrying on deleted the artifacts *without*
    # advancing the fence, which is strictly worse than doing nothing: the old
    # results are gone and a late worker from the previous generation can still
    # publish into the freshly rebuilt document. Callers all run inside a
    # publication lock and abort the publish on an exception.
    try:
        rotate_ai_cache_generation(DATA_DIR, doc_id)
    except Exception as exc:
        logger.error("[ParseRoute] 轮换 AI 缓存代际失败 doc=%s err=%s", doc_id, exc)
        raise RuntimeError("AI 缓存代际轮换失败，已中止解析产物发布") from exc
    removed.append("ai_cache_generation")
    removed.extend(_clear_block_bound_reading_cache(doc_id))
    try:
        from services.overview_service import invalidate_overview_work

        invalidate_overview_work(
            doc_id,
            status="superseded",
            reason="文档解析路线已更新，请重新生成速览",
        )
        removed.append("overview_tasks")
    except Exception as exc:
        logger.warning("[ParseRoute] 作废速览任务失败 doc=%s err=%s", doc_id, exc)

    # 速览图表解读缓存是内存态（doc["data"]["logical_figures*"]），不是文件，
    # 需要单独失效，否则深度解析完成后速览仍会命中旧的 pdf_native/caption_only 结果。
    target_doc = doc if doc is not None else documents_store.get(doc_id)
    if isinstance(target_doc, dict):
        doc_data = target_doc.get("data")
        if isinstance(doc_data, dict):
            if doc_data.pop("logical_figures_status", None) is not None:
                doc_data.pop("logical_figures_meta", None)
                doc_data.pop("logical_figures", None)
                removed.append("logical_figures")
            # MinerU visual assets are derived from the primary block index and
            # carry the same parse identity. A route/generation replacement
            # must never leave the former document's panel geometry visible.
            if doc_data.pop("mineru_visual_assets", None) is not None:
                removed.append("mineru_visual_assets")

    # GraphRAG persists under doc_id, while a re-upload of the same PDF keeps
    # that id but starts a different parse generation. Remove both its in-memory
    # registry and on-disk graph before a new route can be published.
    try:
        from services.graphrag import INSTANCES as graphrag_instances, BUILD_PROGRESS as graphrag_progress

        graphrag_instances.pop(doc_id, None)
        graphrag_progress.pop(doc_id, None)
        graph_dir = _safe_graphrag_working_dir(doc_id)
        if graph_dir.exists():
            shutil.rmtree(graph_dir, ignore_errors=True)
            removed.append("graphrag")
    except Exception as exc:
        logger.warning("[ParseRoute] 删除 GraphRAG 缓存失败 doc=%s err=%s", doc_id, exc)

    return removed


def _persist_mineru_visual_assets(
    doc_id: str,
    block_index: dict,
) -> bool:
    """Persist an additive visual view after the MinerU primary index publishes.

    A failed derived-asset write is logged and retried lazily by consumers; it
    does not roll back a valid primary MinerU parse. The envelope is fenced by
    the same route/generation/source-hash tuple as the block index.
    """
    try:
        from services.mineru_visual_asset_service import (
            MINERU_VISUAL_ASSET_KEY,
            build_mineru_visual_asset_envelope,
        )

        envelope = build_mineru_visual_asset_envelope(block_index)
    except Exception as exc:
        logger.warning("[MinerUVisualAsset] build skipped doc=%s error=%s", doc_id, exc)
        return False
    if not envelope:
        return False

    doc = documents_store.get(doc_id)
    if not isinstance(doc, dict) or not isinstance(doc.get("data"), dict):
        return False
    existing = doc["data"].get(MINERU_VISUAL_ASSET_KEY)
    if isinstance(existing, dict) and (
        str(existing.get("revision") or "") == str(envelope.get("revision") or "")
        and str(existing.get("parse_generation") or "") == str(envelope.get("parse_generation") or "")
        and str(existing.get("document_source_hash") or "") == str(envelope.get("document_source_hash") or "")
    ):
        return False

    candidate = deepcopy(doc)
    candidate_data = candidate.get("data")
    if not isinstance(candidate_data, dict):
        return False
    candidate_data[MINERU_VISUAL_ASSET_KEY] = envelope
    if not save_document(doc_id, candidate):
        logger.warning("[MinerUVisualAsset] persist failed doc=%s", doc_id)
        return False
    documents_store[doc_id] = candidate
    return True


def publish_logical_figure_cache(
    doc_id: str,
    *,
    parse_generation: str,
    document_source_hash: str,
    parser_route: str,
    figures: list[dict],
    metadata: dict,
    status: dict,
) -> bool:
    """Publish an in-memory logical-figure cache only for the active parse run.

    Figure extraction can finish after a document was re-uploaded or switched
    from local parsing to MinerU.  The cache is deliberately ephemeral, but a
    stale task must still never replace the current document's ``data`` map.
    """
    expected_route = str(parser_route or "").strip().lower()
    expected_generation = str(parse_generation or "").strip()
    expected_source_hash = str(document_source_hash or "").strip()
    with _get_document_publication_lock(doc_id):
        current = documents_store.get(doc_id)
        if not isinstance(current, dict):
            return False
        manifest = _read_document_parse_manifest(doc_id, current)
        if not matches_parse_generation(
            manifest,
            generation=expected_generation,
            source_hash=expected_source_hash or None,
        ):
            logger.info(
                "[FigureExtraction] discard stale cache doc=%s generation=%s",
                doc_id,
                expected_generation,
            )
            return False
        current_route = str(manifest.get("resolved_route") or "").strip().lower()
        if expected_route and current_route and current_route != expected_route:
            logger.info(
                "[FigureExtraction] discard route-mismatched cache doc=%s expected=%s current=%s",
                doc_id,
                expected_route,
                current_route,
            )
            return False

        candidate = deepcopy(current)
        candidate_data = candidate.get("data")
        if not isinstance(candidate_data, dict):
            return False
        candidate_data["logical_figures"] = deepcopy(figures)
        candidate_data["logical_figures_meta"] = deepcopy(metadata)
        candidate_data["logical_figures_status"] = deepcopy(status)
        # Logical figures are an in-memory overview cache. Keep the existing
        # no-disk-write policy while atomically replacing only the live record.
        documents_store[doc_id] = candidate
        return True


def publish_visual_supplements(
    doc_id: str,
    *,
    parse_generation: str,
    document_source_hash: str,
    visual_model_identity: str,
    items: list[dict],
) -> dict:
    """Publish additive VLM supplements without changing the main parser route.

    The document lock and parse-generation fence make this a small publication
    transaction: an obsolete VLM response cannot alter a document that has
    since been reparsed or switched to MinerU.
    """
    if not items:
        return {"published": False, "reason": "no_items", "revision": ""}

    from services.visual_supplement_service import (
        mark_visual_supplements_committed,
        upsert_visual_supplements,
        visual_supplements_are_committed,
    )

    with _get_document_publication_lock(doc_id):
        doc = documents_store.get(doc_id)
        if not isinstance(doc, dict):
            return {"published": False, "reason": "missing_document", "revision": ""}
        manifest = _require_current_parse_generation(
            doc_id,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
        )
        resolved_route = str(manifest.get("resolved_route") or "").strip().lower()
        if resolved_route not in {PARSE_ROUTE_LOCAL, PARSE_ROUTE_MINERU}:
            return {"published": False, "reason": "unsupported_parse_route", "revision": ""}

        data = doc.setdefault("data", {})
        if not isinstance(data, dict):
            raise RuntimeError("文档数据格式异常，无法发布视觉补充")

        # ``read_parse_manifest`` can infer a stable identity for legacy
        # documents, but the block/RAG readers intentionally consume only a
        # persisted modern manifest. Migrate under this publication lock so a
        # successful VLM publication is immediately consumable everywhere.
        had_persisted_manifest = "parse_manifest" in data
        before_manifest = data.get("parse_manifest")
        migrated_legacy_manifest = (
            not isinstance(before_manifest, dict)
            or _is_legacy_parse_manifest(manifest)
        )
        if migrated_legacy_manifest:
            migrated_metadata = dict(manifest.get("metadata") or {})
            migrated_metadata.pop("legacy_inferred", None)
            migrated_metadata["migrated_from_legacy"] = True
            migrated_manifest = dict(manifest)
            migrated_manifest["metadata"] = migrated_metadata
            data["parse_manifest"] = migrated_manifest
            manifest = migrated_manifest

        identity = {
            "parser_route": resolved_route,
            "parse_generation": str(manifest.get("generation") or ""),
            "document_source_hash": str(manifest.get("source_hash") or ""),
        }
        before = data.get("visual_supplements")
        before_commit = data.get("visual_supplement_commit")

        def restore_staged_document_data() -> None:
            if before is None:
                data.pop("visual_supplements", None)
            else:
                data["visual_supplements"] = before
            if before_commit is None:
                data.pop("visual_supplement_commit", None)
            else:
                data["visual_supplement_commit"] = before_commit
            if had_persisted_manifest:
                data["parse_manifest"] = before_manifest
            else:
                data.pop("parse_manifest", None)

        changed, envelope = upsert_visual_supplements(
            data,
            parse_identity=identity,
            visual_model_identity=visual_model_identity,
            items=items,
        )
        revision = str(envelope.get("revision") or "")
        if not changed:
            if visual_supplements_are_committed(data, parse_identity=identity):
                if migrated_legacy_manifest:
                    if not save_document(doc_id, doc):
                        restore_staged_document_data()
                        raise RuntimeError("旧文档解析身份迁移写入失败")
                    # The legacy block index has no modern identity. Rebuild
                    # it after the persisted migration instead of retaining a
                    # visually invisible legacy cache.
                    try:
                        ensure_block_index(
                            doc_id=doc_id,
                            doc=doc,
                            data_dir=DATA_DIR,
                            pdf_path=_resolve_document_pdf_path(doc),
                            force_rebuild=True,
                            preserve_active_source=False,
                        )
                    except Exception as exc:
                        # The persisted modern manifest is already correct;
                        # a later reader will retry its identity-bound build.
                        logger.warning(
                            "[VisualSupplement] legacy block-index migration deferred doc=%s: %s",
                            doc_id,
                            exc,
                        )
                    rotate_ai_cache_generation(DATA_DIR, doc_id)
                    _clear_block_bound_reading_cache(doc_id)
                return {
                    "published": False,
                    "reason": "unchanged",
                    "revision": revision,
                    "committed": True,
                }
        try:
            # Build the pending index in memory. Writing it to the live path
            # before the document commit lets unrelated readers race and
            # overwrite the staging revision with a base index.
            if resolved_route == PARSE_ROUTE_MINERU:
                active_block_index = load_block_index(DATA_DIR, doc_id)
                if not isinstance(active_block_index, dict):
                    raise RuntimeError("MinerU 阅读块索引不可用，无法发布视觉补充")
                staged_block_index = stage_visual_supplements_on_block_index(
                    active_block_index,
                    data=data,
                    parse_identity=identity,
                )
            else:
                staged_block_index = build_block_index(
                    doc_id=doc_id,
                    doc=doc,
                    pdf_path=_resolve_document_pdf_path(doc),
                    include_uncommitted_visual_supplements=True,
                )
        except Exception:
            restore_staged_document_data()
            raise

        # commit marker 只能先写入私有副本。聊天和搜索不会获取 publication
        # lock，若直接改共享对象，它们可能在文档落盘失败前读到未提交证据。
        committed_doc = deepcopy(doc)
        committed_data = committed_doc.get("data")
        if not isinstance(committed_data, dict):
            restore_staged_document_data()
            raise RuntimeError("文档数据格式异常，无法提交视觉补充")
        committed, marker = mark_visual_supplements_committed(
            committed_data,
            parse_identity=identity,
        )
        if not save_block_index(DATA_DIR, doc_id, staged_block_index):
            restore_staged_document_data()
            raise RuntimeError("视觉补充阅读块索引写入失败")

        if not marker or not save_document(doc_id, committed_doc):
            restore_staged_document_data()
            # The staged index was intentionally written before the marker.
            # Restore the active file to the old visible document state.
            try:
                ensure_block_index(
                    doc_id=doc_id,
                    doc=doc,
                    data_dir=DATA_DIR,
                    pdf_path=_resolve_document_pdf_path(doc),
                    force_rebuild=True,
                    preserve_active_source=False,
                )
            except Exception as restore_exc:
                logger.error(
                    "[VisualSupplement] block-index rollback failed doc=%s: %s",
                    doc_id,
                    restore_exc,
                )
            raise RuntimeError("视觉补充提交写入失败")

        # 两份文件均已持久化后再一次性替换共享引用。仍持有旧引用的并发
        # 请求只会看到 uncommitted staging，新的请求才会看到 commit marker。
        documents_store[doc_id] = committed_doc

        # Visual supplements only append evidence blocks. They do not replace
        # the parser route, generation, source text, or existing block ids, so
        # invalidating every block-bound artifact here throws away valid work
        # (notably page/full-document translations) when a background figure
        # analysis finishes. Translation records retain their own source and
        # block-signature checks; newly appended visual blocks simply miss the
        # cache and can be translated independently. A real parse publication
        # still goes through ``_clear_block_dependent_ai_cache`` above and
        # keeps the strict route/generation write fence.
        removed: list[str] = []
        result = {
            "published": True,
            "revision": revision,
            "committed": committed,
            "block_count": sum(len(page.get("blocks") or []) for page in staged_block_index.get("pages") or []),
            "removed": removed,
            "cache_policy": "preserved_additive",
        }
        if not changed:
            result["reason"] = "recovered_publication"
        return result


def _set_deep_parse_status(doc_id: str, status: str, *, stage: str = "", error: str = "", **extra) -> dict:
    """Atomically publish a deep-parse task transition.

    A remote poll may return after the user cancels a job. Terminal records are
    immutable for that job; only a newly created job id may replace them. The
    disk record is written while the same lock is held so a late writer cannot
    persist an older state after the in-memory task has already advanced.
    """
    with _DEEP_PARSE_LOCK:
        current = dict(_DEEP_PARSE_TASKS.get(doc_id) or {})
        current_status = str(current.get("status") or "").strip().lower()
        current_stage = str(current.get("stage") or "").strip().lower()
        current_job_id = str(current.get("job_id") or "").strip()
        incoming_job_id = str(extra.get("job_id") or "").strip()
        incoming_generation = str(extra.get("parse_generation") or "").strip()
        current_generation = str(current.get("parse_generation") or "").strip()
        starts_replacement_job = bool(
            status == "queued"
            and incoming_job_id
            and incoming_job_id != current_job_id
        )
        starts_new_generation = bool(
            incoming_generation
            and current_generation
            and incoming_generation != current_generation
        )
        if current_status in _DEEP_PARSE_TERMINAL_STATUSES and not starts_replacement_job and not starts_new_generation:
            return current
        if starts_replacement_job or starts_new_generation:
            # A retry is a distinct local task even when it resumes the same
            # remote batch. Do not let its elapsed time or remote percentage
            # inherit from the terminal task it replaces.
            for key in (
                "created_at",
                "started_at",
                "completed_at",
                "poll_attempt",
                "poll_total",
                "remote_state",
                "remote_progress_percent",
                "remote_progress_source",
                "remote_pages_completed",
                "remote_pages_total",
                "stage_started_at",
            ):
                current.pop(key, None)
        timestamp = utc_now_iso_ms()
        normalized_stage = str(stage or "").strip().lower()
        if normalized_stage and normalized_stage != current_stage:
            extra.setdefault("stage_started_at", timestamp)
        preserved_created_at = str(current.get("created_at") or "").strip()
        preserved_started_at = str(current.get("started_at") or "").strip()
        extra.pop("created_at", None)
        extra.pop("started_at", None)
        extra.pop("completed_at", None)
        extra.pop("progress", None)
        extra.pop("elapsed_seconds", None)
        current.update({
            "doc_id": doc_id,
            "provider": "mineru",
            "job_type": _DEEP_PARSE_JOB_TYPE,
            "status": status,
            "stage": stage,
            "error": error,
            "created_at": preserved_created_at or timestamp,
            "updated_at": timestamp,
            **extra,
        })
        quality_failures = current.get("quality_failures")
        shortfall: dict[str, object] = {}
        if isinstance(quality_failures, (list, tuple)) and quality_failures:
            shortfall = {
                "kind": "parse_quality",
                "code": "quality_gate_failed",
                "stage": stage or "normalize",
                "retryable": True,
                "failed_pages": current.get("failed_pages") or [],
                "reasons": [
                    normalize_diagnostic_reason(item)
                    for item in list(quality_failures)[:12]
                    if normalize_diagnostic_reason(item)
                ],
            }
        elif status in {"failed", "partial_ready"}:
            shortfall = {
                "kind": "parse",
                "code": classify_error_code(error) or ("partial_parse" if status == "partial_ready" else "parse_failed"),
                "stage": stage or status,
                "retryable": status == "failed",
                "failed_pages": current.get("failed_pages") or [],
            }
        clean_shortfall = sanitize_task_shortfall(shortfall)
        if clean_shortfall:
            current["shortfall"] = clean_shortfall
        elif status in {"ready", "succeeded"}:
            current.pop("shortfall", None)
        if preserved_started_at:
            current["started_at"] = preserved_started_at
        elif status in {"queued", "running"} and not current.get("started_at"):
            current["started_at"] = current.get("created_at") or timestamp
        if status in {"ready", "partial_ready"} and not current.get("completed_at"):
            current["completed_at"] = timestamp
        _DEEP_PARSE_TASKS[doc_id] = current
        try:
            persist_document_job(DATA_DIR, _DEEP_PARSE_JOB_TYPE, doc_id, current)
        except Exception as exc:
            logger.warning("[DeepParse] failed to persist task status for %s: %s", doc_id, exc)
        task_id = str(current.get("task_id") or current.get("job_id") or "").strip()
        if task_id:
            current["task_id"] = task_id
            try:
                append_task_event(
                    DATA_DIR,
                    task_id=task_id,
                    stage=stage or status,
                    status=status,
                    identity={
                        "route": "mineru",
                        "generation": current.get("parse_generation") or "",
                        "source_hash": current.get("document_source_hash") or "",
                    },
                    attempt=current.get("poll_attempt") or 0,
                    error_code=classify_error_code(error),
                    degraded_reason=(clean_shortfall or {}).get("code", ""),
                    shortfall=clean_shortfall,
                )
            except Exception as exc:
                logger.debug("[DeepParse] event ledger append failed task=%s: %s", task_id, exc)
        return dict(current)


def _deep_parse_worker_owns_task(doc_id: str, cancel_event: threading.Event) -> bool:
    """Prevent an old same-generation worker from publishing over a retry."""
    with _DEEP_PARSE_LOCK:
        active_event = _DEEP_PARSE_CANCEL_EVENTS.get(doc_id)
    return active_event is None or active_event is cancel_event


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


def _can_resume_direct_mineru_result_download(task: dict | None) -> bool:
    """Return whether a failed direct job can resume without reuploading its PDF."""
    if not isinstance(task, dict):
        return False
    error = str(task.get("error") or "").casefold()
    return bool(
        str(task.get("status") or "").casefold() == "failed"
        and str(task.get("access_mode") or "").casefold() == "direct"
        and str(task.get("batch_id") or "").strip()
        and "mineru zip 下载失败" in error
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
                updated_at=utc_now_iso_ms(),
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
        try:
            mineru_result_exists = bool(
                load_mineru_result(
                    DATA_DIR,
                    doc_id,
                    parse_generation=str(parse_manifest.get("generation") or ""),
                    document_source_hash=str(parse_manifest.get("source_hash") or ""),
                    require_identity=True,
                )
            )
        except MinerUResultUnreadable as exc:
            # 这里只是状态探针，不能因为产物损坏就让状态接口 500；
            # 但也不能当成"没有产物"静默带过。
            logger.error("[DeepParse] MinerU 原始产物不可读 doc=%s: %s", doc_id, exc)
            mineru_result_exists = False
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
    mineru_quality = (
        dict(block_index.get("mineru_meta") or {})
        if active_mineru and isinstance(block_index, dict)
        else {}
    )
    parse_ready = is_parse_prepared(parse_manifest)
    is_full_mineru_route = _is_full_mineru_parse_manifest(parse_manifest)

    if not current:
        waiting_for_full_route = active_mineru and is_full_mineru_route and not parse_ready
        status = (
            "running"
            if waiting_for_full_route
            else (_mineru_ready_status(block_index) if active_mineru else "idle")
        )
        current = {
            "doc_id": doc_id,
            "provider": "mineru",
            "status": status,
            "stage": (
                str(parse_manifest.get("stage") or "building_rag_index")
                if waiting_for_full_route
                else (_mineru_ready_status(block_index) if active_mineru else "not_started")
            ),
            "error": "",
            "created_at": str(parse_manifest.get("created_at") or ""),
            "started_at": str(parse_manifest.get("started_at") or ""),
            "parse_generation": str(parse_manifest.get("generation") or ""),
            "stage_started_at": str(
                parse_manifest.get("updated_at")
                or parse_manifest.get("started_at")
                or parse_manifest.get("created_at")
                or ""
            ),
        }

    if not current.get("created_at"):
        current["created_at"] = str(parse_manifest.get("created_at") or "")
    if not current.get("started_at"):
        current["started_at"] = str(parse_manifest.get("started_at") or "")
    if not current.get("stage_started_at"):
        current["stage_started_at"] = str(
            parse_manifest.get("updated_at")
            or parse_manifest.get("started_at")
            or parse_manifest.get("created_at")
            or ""
        )

    current.update({
        "configured": _mineru_configured(),
        "access_mode": access_mode,
        "mineru_result_exists": mineru_result_exists,
        "active_source": active_source,
        "active_mineru": active_mineru,
    })
    if _can_resume_direct_mineru_result_download(current):
        current.update({
            "resume_available": True,
            "resume_kind": "result_download",
        })
    else:
        current.pop("resume_available", None)
        current.pop("resume_kind", None)
    if mineru_quality:
        current.update({
            "quality_status": mineru_quality.get("quality_status", "success"),
            "expected_page_count": mineru_quality.get("expected_page_count", 0),
            "coverage": mineru_quality.get("coverage", 0.0),
            "failed_pages": mineru_quality.get("failed_pages") or [],
            "page_ledger": mineru_quality.get("page_ledger") or [],
        })
    rag_index = _get_rag_index_status(doc_id)
    current["rag_index"] = rag_index
    current["parse_manifest"] = parse_manifest
    current["parse_ready"] = parse_ready
    if (
        is_full_mineru_route
        and not parse_ready
        and str(parse_manifest.get("stage") or "").strip().lower() == "building_rag_index"
        and str(rag_index.get("status") or "").strip().lower() in {"queued", "running"}
    ):
        rag_stage = str(rag_index.get("stage") or "building_rag_index").strip().lower()
        current["status"] = "running"
        current["stage"] = rag_stage
        if not current.get("started_at"):
            current["started_at"] = str(
                rag_index.get("started_at")
                or rag_index.get("created_at")
                or current.get("created_at")
                or ""
            )
        if not current.get("created_at"):
            current["created_at"] = str(
                rag_index.get("created_at")
                or current.get("started_at")
                or ""
            )
        current["stage_started_at"] = str(
            rag_index.get("stage_started_at")
            or current.get("stage_started_at")
            or current.get("started_at")
            or ""
        )
        current["updated_at"] = str(
            rag_index.get("updated_at")
            or current.get("updated_at")
            or ""
        )
    if (
        active_mineru
        and current.get("status") not in {"queued", "running", "failed"}
        and not (is_full_mineru_route and not parse_ready)
    ):
        current["status"] = _mineru_ready_status(block_index)
        current["stage"] = _mineru_ready_status(block_index)

    current["progress"] = derive_mineru_progress(current)
    current.update(_assess_deep_parse_recommendation(doc_id, active_mineru, block_index, rag_index=rag_index))
    task_id = str(current.get("task_id") or current.get("job_id") or "").strip()
    if task_id:
        current["task_id"] = task_id
        ledger = get_task_event_ledger(DATA_DIR, task_id)
        current["events"] = ledger.get("events") or []
        if ledger.get("shortfall"):
            current["shortfall"] = ledger["shortfall"]
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


def _make_mineru_adapter(
    config: dict,
    access_mode: str,
    model_version: Optional[str] = None,
    enable_ocr: Optional[bool] = None,
):
    effective_model_version = normalize_mineru_model_version(
        model_version if model_version is not None else config.get("model_version")
    )
    if enable_ocr is None:
        enable_ocr = bool(config.get("enable_ocr", False))
    if access_mode == "direct":
        return MinerUDirectAdapter(
            token=config.get("token", ""),
            base_url=config.get("base_url", "https://mineru.net/api/v4"),
            enable_ocr=enable_ocr,
            enable_formula=config.get("enable_formula", True),
            enable_table=config.get("enable_table", True),
            model_version=effective_model_version,
        )
    return MinerUAdapter(
        worker_url=config.get("worker_url", ""),
        auth_key=config.get("auth_key", ""),
        token=config.get("token", ""),
        token_mode=config.get("token_mode", "frontend"),
        enable_ocr=enable_ocr,
        enable_formula=config.get("enable_formula", True),
        enable_table=config.get("enable_table", True),
        model_version=effective_model_version,
    )


def _deep_parse_worker_loop() -> None:
    """Run bounded MinerU jobs without creating one waiting thread per document."""
    while True:
        doc_id, cancel_event, remote_job, parse_generation, full_route_options = _DEEP_PARSE_QUEUE.get()
        try:
            if cancel_event.is_set():
                with _DEEP_PARSE_LOCK:
                    if _DEEP_PARSE_CANCEL_EVENTS.get(doc_id) is cancel_event:
                        _DEEP_PARSE_CANCEL_EVENTS.pop(doc_id, None)
                continue
            _run_mineru_deep_parse(
                doc_id,
                cancel_event,
                remote_job,
                parse_generation,
                full_route_options,
                acquire_slot=False,
            )
        except Exception:
            # _run_mineru_deep_parse already owns task status publication. This
            # is a final containment boundary so one unexpected worker fault
            # cannot terminate the fixed queue worker.
            logger.exception("[DeepParse] MinerU queue worker crashed for %s", doc_id)
        finally:
            _DEEP_PARSE_QUEUE.task_done()


def _ensure_deep_parse_workers() -> None:
    """Start at most the configured number of daemon queue workers."""
    with _DEEP_PARSE_WORKERS_LOCK:
        _DEEP_PARSE_WORKERS[:] = [
            worker for worker in _DEEP_PARSE_WORKERS if worker.is_alive()
        ]
        while len(_DEEP_PARSE_WORKERS) < _DEEP_PARSE_CONCURRENCY:
            worker = threading.Thread(
                target=_deep_parse_worker_loop,
                name=f"chatpdf-mineru-worker-{len(_DEEP_PARSE_WORKERS) + 1}",
                daemon=True,
            )
            worker.start()
            _DEEP_PARSE_WORKERS.append(worker)


def _enqueue_mineru_deep_parse(
    doc_id: str,
    cancel_event: threading.Event,
    remote_job: Optional[dict],
    parse_generation: str,
    full_route_options: Optional[dict],
) -> bool:
    """Queue one MinerU job or reject it before it consumes another thread."""
    try:
        _ensure_deep_parse_workers()
        _DEEP_PARSE_QUEUE.put_nowait(
            (doc_id, cancel_event, remote_job, parse_generation, full_route_options)
        )
        return True
    except queue.Full:
        return False
    except Exception as exc:
        logger.exception("[DeepParse] failed to enqueue MinerU job %s: %s", doc_id, exc)
        return False


def _run_mineru_deep_parse(
    doc_id: str,
    cancel_event: threading.Event,
    remote_job: Optional[dict] = None,
    parse_generation: str = "",
    full_route_options: Optional[dict] = None,
    *,
    acquire_slot: bool = True,
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
        if not _worker_matches_current_generation() or not _deep_parse_worker_owns_task(doc_id, cancel_event):
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
            _transition_current_full_mineru_manifest(
                doc_id,
                PARSE_STATUS_RUNNING,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
                stage="mineru_parsing",
                expected_statuses={PARSE_STATUS_PENDING, PARSE_STATUS_QUEUED},
                metadata={"full_route": True},
            )
        if acquire_slot:
            _set_worker_status("queued", stage="waiting_for_slot", message="等待 MinerU 解析槽位")
            while not cancel_event.is_set():
                if _DEEP_PARSE_SEMAPHORE.acquire(timeout=0.25):
                    acquired_slot = True
                    break
            if not acquired_slot:
                _set_worker_status("cancelled", stage="cancelled", message="MinerU 深度解析已取消")
                return
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
        queued_model_version = (remote_job or {}).get("model_version")
        model_version = normalize_mineru_model_version(
            queued_model_version
            if queued_model_version is not None
            else config.get("model_version")
        )
        doc_data = doc.get("data") if isinstance(doc, dict) else {}
        enable_ocr = should_enable_mineru_ocr(
            config,
            extraction_quality=(doc_data or {}).get("extraction_quality"),
            pages_needing_ocr=(doc_data or {}).get("pages_needing_ocr"),
            ocr_status=(doc_data or {}).get("ocr_status"),
        )
        if enable_ocr and not bool(config.get("enable_ocr")):
            logger.info(
                "[DeepParse] 本地文本质量不足，MinerU 自动开启扫描件 OCR doc_id=%s quality=%s ocr_status=%s",
                doc_id,
                (doc_data or {}).get("extraction_quality"),
                (doc_data or {}).get("ocr_status"),
            )
        transport = _make_mineru_adapter(
            config,
            access_mode,
            model_version,
            enable_ocr=enable_ocr,
        )
        adapter = MinerUDocumentParseAdapter(transport)
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
                if key not in {
                    "stage",
                    "message",
                    "created_at",
                    "started_at",
                    "completed_at",
                    "updated_at",
                    "stage_started_at",
                    "progress",
                    "elapsed_seconds",
                    "status",
                    "error",
                    "doc_id",
                }
            }
            extra.setdefault("model_version", model_version)
            _set_worker_status(
                "running",
                stage=stage,
                message=message,
                access_mode=access_mode,
                **extra,
            )

        if remote_job and remote_job.get("batch_id"):
            resuming_result_download = str(remote_job.get("resume_kind") or "") == "result_download"
            _set_worker_status(
                "running",
                stage="resuming_result_download" if resuming_result_download else "resuming",
                message="重新获取 MinerU 已完成的解析结果" if resuming_result_download else "恢复 MinerU 远端任务",
                access_mode=access_mode, batch_id=remote_job["batch_id"], data_id=remote_job.get("data_id", ""),
                recovered_after_restart=not resuming_result_download,
                resume_kind="result_download" if resuming_result_download else "",
            )
            parser_attempted = True
            submission = DocumentParseSubmission(
                provider="mineru",
                job_id=str(remote_job["batch_id"]),
                data_id=str(remote_job.get("data_id") or ""),
                access_mode=access_mode,
            )
        else:
            _set_worker_status("running", stage="uploading", message="准备上传 PDF 到 MinerU")
            pdf_bytes = pdf_path.read_bytes()
            parser_attempted = True
            submission = adapter.submit(
                pdf_bytes,
                progress_callback=_on_mineru_progress,
                cancel_event=cancel_event,
            )
        payload = adapter.poll(
            submission,
            progress_callback=_on_mineru_progress,
            cancel_event=cancel_event,
        )
        payload.setdefault("model_version", model_version)

        _set_worker_status("running", stage="building_index", message="重建阅读块和大纲")
        if cancel_event.is_set():
            _set_worker_status("cancelled", stage="cancelled", message="MinerU 深度解析已取消")
            return
        rag_normalized, quality_failures = _normalize_mineru_for_document(
            doc_id,
            payload,
            doc=doc,
        )
        if quality_failures:
            quality_message = (
                "MinerU 解析质量门失败，未发布任何结果: "
                + ", ".join(quality_failures)
            )
            if parser_attempted and not parser_outcome_recorded:
                record_ocr_provider_use("mineru", outcome="failure", operation="document_parse")
                parser_outcome_recorded = True
            if full_mineru_route and _worker_matches_current_generation():
                try:
                    _transition_current_full_mineru_manifest(
                        doc_id,
                        PARSE_STATUS_FAILED,
                        parse_generation=parse_generation,
                        document_source_hash=parse_source_hash,
                        stage="failed",
                        error=quality_message,
                        expected_statuses={
                            PARSE_STATUS_PENDING,
                            PARSE_STATUS_QUEUED,
                            PARSE_STATUS_RUNNING,
                        },
                    )
                except Exception:
                    logger.debug("[DeepParse] failed to mark quality-gated parse failed for %s", doc_id)
            _set_worker_status(
                "failed",
                stage="failed",
                error=quality_message,
                quality_status=rag_normalized.get("quality_status", "failed"),
                expected_page_count=rag_normalized.get("expected_page_count", 0),
                coverage=rag_normalized.get("coverage", 0.0),
                failed_pages=rag_normalized.get("failed_pages") or [],
                page_ledger=rag_normalized.get("page_ledger") or [],
                quality_failures=quality_failures,
                quality_report=rag_normalized.get("quality_report") or {},
            )
            return
        artifact = adapter.normalize(
            submission,
            payload,
            normalizer=lambda raw_payload: build_block_index_from_mineru_payload(
                doc_id=doc_id,
                doc=doc,
                payload=raw_payload,
                pdf_path=pdf_path,
            ),
        )
        block_index = _attach_mineru_quality_to_block_index(
            artifact.normalized,
            rag_normalized,
        )
        record_ocr_provider_use("mineru", outcome="success", operation="document_parse")
        parser_outcome_recorded = True
        mineru_quality = dict(block_index.get("mineru_meta") or {})
        removed: list[str] = []
        waiting_for_rag_rebuild = False
        with _get_document_publication_lock(doc_id):
            if not _worker_matches_current_generation():
                logger.info("[DeepParse] discard stale MinerU block index for %s generation=%s", doc_id, parse_generation)
                return
            if cancel_event.is_set():
                _set_worker_status("cancelled", stage="cancelled", message="MinerU 深度解析已取消")
                return
            def _publish_mineru_artifact(current_artifact):
                save_mineru_result(
                    DATA_DIR,
                    doc_id,
                    current_artifact.raw_payload,
                    parse_generation=parse_generation,
                    document_source_hash=parse_source_hash,
                )
                if not save_block_index(DATA_DIR, doc_id, current_artifact.normalized):
                    raise RuntimeError("MinerU 阅读块索引写入失败")
                return current_artifact.normalized

            block_index = adapter.publish(artifact, publisher=_publish_mineru_artifact)
            current_doc = documents_store.get(doc_id)
            removed = adapter.invalidate(
                invalidator=lambda: _clear_block_dependent_ai_cache(doc_id, current_doc)
            )
            if _persist_mineru_visual_assets(doc_id, block_index):
                removed.append("mineru_visual_assets_published")
            current_doc = documents_store.get(doc_id)

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
                        "quality_status": mineru_quality.get("quality_status", "success"),
                        "expected_page_count": mineru_quality.get("expected_page_count", 0),
                        "coverage": mineru_quality.get("coverage", 0.0),
                        "failed_pages": mineru_quality.get("failed_pages") or [],
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
                    _mineru_ready_status(block_index),
                    stage="awaiting_rag_index",
                    message="MinerU 版面解析完成，等待使用当前 Embedding 配置发布问答索引",
                    active_source=MINERU_BLOCK_INDEX_SOURCE,
                    active_mineru=True,
                    access_mode=access_mode,
                    quality_status=mineru_quality.get("quality_status", "success"),
                    expected_page_count=mineru_quality.get("expected_page_count", 0),
                    coverage=mineru_quality.get("coverage", 0.0),
                    failed_pages=mineru_quality.get("failed_pages") or [],
                    page_ledger=mineru_quality.get("page_ledger") or [],
                )
                return
            embedding_identity = _require_explicit_rag_embedding_identity_or_400(
                embedding_model=full_route_options.get("embedding_model"),
                embedding_provider=full_route_options.get("embedding_provider"),
                embedding_api_host=full_route_options.get("embedding_api_host"),
                embedding_api_key=full_route_options.get("embedding_api_key"),
                operation="MinerU 问答索引发布",
            )
            _rebuild_mineru_rag_index_unlocked(
                doc_id,
                embedding_model=embedding_identity["model"],
                embedding_api_key=embedding_identity["api_key"],
                embedding_api_host=embedding_identity["api_host"],
                embedding_provider=embedding_identity["provider"],
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
            _mineru_ready_status(block_index),
            stage=_mineru_ready_status(block_index),
            block_count=block_count,
            outline_count=outline_count,
            figure_count=figure_count,
            cache_removed=removed,
            active_source=MINERU_BLOCK_INDEX_SOURCE,
            active_mineru=True,
            access_mode=access_mode,
            model_version=model_version,
            quality_status=mineru_quality.get("quality_status", "success"),
            expected_page_count=mineru_quality.get("expected_page_count", 0),
            coverage=mineru_quality.get("coverage", 0.0),
            failed_pages=mineru_quality.get("failed_pages") or [],
            page_ledger=mineru_quality.get("page_ledger") or [],
            structure_degraded=bool(mineru_quality.get("structure_degraded")),
            silently_dropped_heading_count=int(
                mineru_quality.get("silently_dropped_heading_count") or 0
            ),
            outline_heading_count=int(mineru_quality.get("outline_heading_count") or 0),
        )
        logger.info("[DeepParse] MinerU deep parse ready for %s: blocks=%s outline=%s", doc_id, block_count, outline_count)
    except _SupersededParseGeneration:
        logger.info("[DeepParse] MinerU worker superseded for %s generation=%s", doc_id, parse_generation)
        return
    except MinerUQualityError as exc:
        if parser_attempted and not parser_outcome_recorded and not cancel_event.is_set():
            record_ocr_provider_use("mineru", outcome="failure", operation="document_parse")
            parser_outcome_recorded = True
        quality_meta = exc.as_status_dict()
        logger.warning("[DeepParse] MinerU quality gate failed for %s: %s", doc_id, exc)
        if full_mineru_route and _worker_matches_current_generation():
            try:
                _transition_current_full_mineru_manifest(
                    doc_id,
                    PARSE_STATUS_FAILED,
                    parse_generation=parse_generation,
                    document_source_hash=parse_source_hash,
                    stage="failed",
                    error=str(exc),
                    expected_statuses={
                        PARSE_STATUS_PENDING,
                        PARSE_STATUS_QUEUED,
                        PARSE_STATUS_RUNNING,
                    },
                    metadata=quality_meta,
                )
            except Exception:
                logger.debug("[DeepParse] failed to mark quality-gate manifest failed for %s", doc_id)
        _set_worker_status("failed", stage="failed", error=str(exc), **quality_meta)
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
                _transition_current_full_mineru_manifest(
                    doc_id,
                    PARSE_STATUS_FAILED,
                    parse_generation=parse_generation,
                    document_source_hash=parse_source_hash,
                    stage="failed",
                    error=str(exc),
                    expected_statuses={
                        PARSE_STATUS_PENDING,
                        PARSE_STATUS_QUEUED,
                        PARSE_STATUS_RUNNING,
                    },
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

    mineru_config = _load_online_ocr_config("mineru")
    access_mode = str(mineru_config.get("access_mode") or "worker").strip().lower()
    model_version = normalize_mineru_model_version(
        mineru_config.get("model_version")
    )
    task_snapshot = {
        "access_mode": access_mode,
        "model_version": model_version,
    }
    resume_existing_result = bool(
        force
        and access_mode == "direct"
        and _can_resume_direct_mineru_result_download(current)
    )
    if resume_existing_result:
        task_snapshot.update({
            "batch_id": str(current.get("batch_id") or ""),
            "data_id": str(current.get("data_id") or ""),
            "resume_kind": "result_download",
        })

    cancel_event = threading.Event()
    with _DEEP_PARSE_LOCK:
        _DEEP_PARSE_CANCEL_EVENTS[doc_id] = cancel_event
    _set_deep_parse_status(
        doc_id,
        "queued",
        stage="resuming_result_download" if resume_existing_result else "queued",
        error="",
        message=("正在重新获取 MinerU 已完成的解析结果，不会重新上传 PDF" if resume_existing_result else ""),
        job_id=f"mineru-{uuid.uuid4().hex}",
        parse_generation=parse_generation,
        document_source_hash=parse_source_hash,
        full_route=full_mineru_route,
        **task_snapshot,
    )
    if not _enqueue_mineru_deep_parse(
        doc_id,
        cancel_event,
        task_snapshot,
        parse_generation,
        full_route_options,
    ):
        with _DEEP_PARSE_LOCK:
            if _DEEP_PARSE_CANCEL_EVENTS.get(doc_id) is cancel_event:
                _DEEP_PARSE_CANCEL_EVENTS.pop(doc_id, None)
        queue_error = "MinerU 等待队列已满，请稍后重试"
        if full_mineru_route:
            _transition_current_full_mineru_manifest(
                doc_id,
                PARSE_STATUS_FAILED,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
                stage="queue_full",
                error=queue_error,
                expected_statuses={PARSE_STATUS_PENDING, PARSE_STATUS_QUEUED, PARSE_STATUS_RUNNING},
            )
        _set_deep_parse_status(
            doc_id,
            "failed",
            stage="queue_full",
            error=queue_error,
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
            full_route=full_mineru_route,
            **task_snapshot,
        )
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
            current_manifest = _read_document_parse_manifest(
                doc_id,
                documents_store.get(doc_id),
            )
            record_generation = str(record.get("parse_generation") or "").strip()
            record_source_hash = str(record.get("document_source_hash") or "").strip()
            if (
                not _is_legacy_parse_manifest(current_manifest)
                and (
                    not record_generation
                    or not matches_parse_generation(
                        current_manifest,
                        generation=record_generation,
                        source_hash=record_source_hash or None,
                    )
                )
            ):
                stale_record = dict(record)
                stale_record.update({
                    "status": "cancelled",
                    "stage": "superseded",
                    "error": "",
                    "message": "文档已重新解析，重启时拒绝恢复旧 MinerU 任务",
                    "superseded": True,
                    "updated_at": utc_now_iso_ms(),
                })
                try:
                    persist_document_job(DATA_DIR, _DEEP_PARSE_JOB_TYPE, doc_id, stale_record)
                except Exception as persist_exc:
                    logger.warning(
                        "[DeepParse] failed to persist superseded restart job for %s: %s",
                        doc_id,
                        persist_exc,
                    )
                logger.info(
                    "[DeepParse] skip stale restart job doc=%s generation=%s",
                    doc_id,
                    record_generation or "legacy",
                )
                continue
            cancel_event = threading.Event()
            with _DEEP_PARSE_LOCK:
                if doc_id in _DEEP_PARSE_CANCEL_EVENTS:
                    continue
                _DEEP_PARSE_CANCEL_EVENTS[doc_id] = cancel_event
                _DEEP_PARSE_TASKS[doc_id] = dict(record)
            parse_generation = str(record.get("parse_generation") or "")
            if _enqueue_mineru_deep_parse(
                doc_id,
                cancel_event,
                record,
                parse_generation,
                None,
            ):
                resumed.append({"doc_id": doc_id, "batch_id": record["batch_id"]})
                continue
            with _DEEP_PARSE_LOCK:
                if _DEEP_PARSE_CANCEL_EVENTS.get(doc_id) is cancel_event:
                    _DEEP_PARSE_CANCEL_EVENTS.pop(doc_id, None)
            _set_deep_parse_status(
                doc_id,
                "failed",
                stage="queue_full",
                error="MinerU 等待队列已满，请稍后重试",
                parse_generation=parse_generation,
                document_source_hash=str(record.get("document_source_hash") or ""),
                recovered_after_restart=True,
            )
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
        model_version = normalize_mineru_model_version(
            current.get("model_version")
            if current.get("model_version") is not None
            else config.get("model_version")
        )
        try:
            remote_cancel = _make_mineru_adapter(
                config,
                access_mode,
                model_version,
            ).cancel_batch(
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
    semantic_identity: Optional[dict] = None,
) -> dict:
    if validation.get("status") == "disabled":
        return _remove_current_semantic_groups(doc_id)
    result = publish_generation(
        DATA_DIR / "semantic_groups",
        doc_id,
        temp_dir,
        source_hash=source_hash,
        transaction_id=transaction_id,
        semantic_identity=semantic_identity,
    )
    _index_cache.invalidate(doc_id)
    return result


def _validate_temp_semantic_groups(
    doc_id: str,
    temp_dir: Path,
    result: dict,
    *,
    expected_identity: Optional[dict] = None,
    expected_vector_dimension: int | None = None,
) -> dict:
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
        expected_identity=expected_identity,
        expected_vector_dimension=expected_vector_dimension,
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
    semantic_identity: Optional[dict] = None,
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
        artifact_validation = validate_semantic_group_artifacts(
            {
                "json": staged_dir / f"{doc_id}.json",
                "index": staged_dir / f"{doc_id}_groups.index",
                "pkl": staged_dir / f"{doc_id}_groups.pkl",
            },
            doc_id,
            expected_identity=semantic_identity,
            expected_vector_dimension=int((semantic_identity or {}).get("vector_dimension") or 0) or None,
        )
        if not artifact_validation["valid"]:
            shutil.rmtree(staged_dir, ignore_errors=True)
            _remove_current_semantic_groups(doc_id)
            return {
                "restored": False,
                "degraded": True,
                "paths": {},
                "errors": list(artifact_validation.get("errors") or []),
            }
        published = publish_generation(
            root,
            doc_id,
            staged_dir,
            source_hash=source_hash,
            transaction_id=transaction_id,
            semantic_identity=semantic_identity,
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
    current_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    payload = {
        "schema_version": 2,
        "doc_id": doc_id,
        "source": _safe_index_source_name(source),
        "state": state,
        "manifest_path": manifest_path,
        "parse_generation": str(current_manifest.get("generation") or ""),
        "document_source_hash": str(current_manifest.get("source_hash") or ""),
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
    current_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    if not _active_rag_index_matches_current_parse(doc_id, current_manifest):
        return {
            "schema_version": 1,
            "doc_id": doc_id,
            "source": _safe_index_source_name(source),
            "complete": False,
            "parse_generation": str(current_manifest.get("generation") or ""),
            "document_source_hash": str(current_manifest.get("source_hash") or ""),
            "reason": "active_vector_parse_identity_mismatch",
            "vector": {"backed_up": False},
            "document": {"backed_up": False},
            "semantic_groups": {"backed_up": False},
        }
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
        "parse_generation": str(current_manifest.get("generation") or ""),
        "document_source_hash": str(current_manifest.get("source_hash") or ""),
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


def _rag_backup_matches_current_parse(
    doc_id: str,
    backup_manifest: dict,
    current_manifest: dict,
) -> bool:
    """Return whether a rollback snapshot belongs to the active parse run."""
    expected_generation = str(backup_manifest.get("parse_generation") or "").strip()
    expected_source_hash = str(backup_manifest.get("document_source_hash") or "").strip()

    # Backups created before the identity fields were added can still be
    # validated from their saved document record. Refuse the rollback when the
    # legacy snapshot is incomplete instead of restoring it speculatively.
    if not expected_generation:
        document_meta = backup_manifest.get("document") or {}
        backup_path = Path(str(document_meta.get("path") or ""))
        try:
            backup_document = json.loads(backup_path.read_text(encoding="utf-8"))
            backup_parse_manifest = _read_document_parse_manifest(doc_id, backup_document)
            expected_generation = str(backup_parse_manifest.get("generation") or "").strip()
            expected_source_hash = str(backup_parse_manifest.get("source_hash") or "").strip()
        except Exception:
            return False

    if not matches_parse_generation(
        current_manifest,
        generation=expected_generation,
        source_hash=expected_source_hash or None,
    ):
        return False

    if _is_legacy_parse_manifest(current_manifest):
        return True

    vector_meta = backup_manifest.get("vector") or {}
    backup_pkl_path = Path(str(vector_meta.get("pkl_path") or ""))
    try:
        with open(backup_pkl_path, "rb") as handle:
            backup_index_data = pickle.load(handle)
    except Exception:
        return False
    if not isinstance(backup_index_data, dict):
        return False
    return parse_identity_matches(
        backup_index_data.get("index_meta") or {},
        current_manifest,
        block_index=load_block_index(DATA_DIR, doc_id),
    )


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


def _normalize_mineru_for_document(
    doc_id: str,
    payload: dict,
    *,
    doc: dict | None = None,
) -> tuple[dict, list[str]]:
    """Normalize and quality-check MinerU output against the source PDF."""
    current_doc = doc if isinstance(doc, dict) else documents_store.get(doc_id)
    data = (current_doc or {}).get("data") or {}
    page_sizes = _document_pdf_page_sizes(doc_id)
    try:
        stored_page_count = max(0, int(data.get("total_pages") or 0))
    except (TypeError, ValueError):
        stored_page_count = 0
    expected_page_count = max(stored_page_count, max(page_sizes.keys(), default=0))
    normalized = normalize_mineru_for_rag(
        payload,
        page_sizes=page_sizes,
        expected_page_count=expected_page_count,
    )
    # Whole-route MinerU uploads intentionally clear the local extraction
    # text.  Its full.md is the independent source witness in that case; do
    # not let an empty upload-time full_text disable the text-coverage gate.
    original_full_text = str(data.get("full_text") or "").strip()
    if not original_full_text:
        original_full_text = str(payload.get("full_md") or "").strip()
    ok, failures = validate_mineru_rag_data(
        normalized,
        original_full_text=original_full_text,
    )
    block_validation_error = ""
    block_validation = {}
    candidate_block_index = None
    try:
        candidate_block_index = build_block_index_from_mineru_payload(
            doc_id=doc_id,
            doc=current_doc or {},
            payload=payload,
            pdf_path=_resolve_document_pdf_path(current_doc or {}),
        )
        validate_mineru_block_index_quality(candidate_block_index)
        block_validation = collect_mineru_block_validation(candidate_block_index)
    except MinerUQualityError as exc:
        block_validation_error = str(exc)
        block_validation = (
            collect_mineru_block_validation(candidate_block_index)
            if isinstance(candidate_block_index, dict)
            else exc.as_status_dict()
        )
        failures = list(failures) + ["block_index_invalid"]
        ok = False
    except Exception as exc:
        block_validation_error = str(exc)
        failures = list(failures) + ["block_index_invalid"]
        ok = False
    quality_report = dict(normalized.get("quality_report") or {})
    if block_validation:
        quality_report["block_validation"] = block_validation
        if block_validation.get("structure_degraded"):
            warnings = list(quality_report.get("warnings") or [])
            dropped = int(block_validation.get("silently_dropped_heading_count") or 0)
            warning = f"outline_heading_coverage_incomplete:silently_dropped={dropped}"
            if warning not in warnings:
                warnings.append(warning)
            quality_report["warnings"] = warnings
    if failures:
        quality_report["failure_reasons"] = sorted(set(
            list(quality_report.get("failure_reasons") or []) + list(failures)
        ))
        if block_validation_error:
            quality_report["block_validation_error"] = block_validation_error
    if block_validation or failures:
        normalized["quality_report"] = quality_report
    return normalized, ([] if ok else failures)


def _artifact_figures_from_block_index(block_index: dict | None) -> list[dict]:
    """Expose the actual MinerU figure anchors in the parse artifact.

    The artifact is an interchange contract, so it must not advertise figure
    support when the current block index contains no usable figure geometry.
    """
    figures: list[dict] = []
    if not isinstance(block_index, dict):
        return figures
    for page_index, page in enumerate(block_index.get("pages") or []):
        if not isinstance(page, dict):
            continue
        try:
            page_number = max(1, int(page.get("page") or page_index + 1))
        except (TypeError, ValueError):
            page_number = page_index + 1
        blocks = [item for item in page.get("blocks") or [] if isinstance(item, dict)]
        for block_index_on_page, block in enumerate(blocks):
            if str(block.get("type") or "").strip().lower() != "figure":
                continue
            bbox = block.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                normalized_bbox = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                continue
            if normalized_bbox[2] <= normalized_bbox[0] or normalized_bbox[3] <= normalized_bbox[1]:
                continue
            caption = str(block.get("text") or "").strip()
            if not caption:
                for candidate in blocks[block_index_on_page + 1:block_index_on_page + 3]:
                    if str(candidate.get("type") or "").strip().lower() == "caption":
                        caption = str(candidate.get("text") or "").strip()
                        if caption:
                            break
            figure_id = str(block.get("block_id") or f"p{page_number}_figure_{block_index_on_page}")
            figures.append({
                "figure_id": figure_id,
                "page": page_number,
                "bbox": normalized_bbox,
                "caption": caption,
                "source": str(block.get("source") or MINERU_BLOCK_INDEX_SOURCE),
                "block_id": figure_id,
            })
    return figures


def _attach_mineru_quality_to_block_index(block_index: dict, normalized: dict) -> dict:
    """Attach the shared page ledger while preserving block-specific signals."""
    meta = dict(block_index.get("mineru_meta") or {})
    # The block index measures semantic structure; the normalizer measures the
    # subset that is usable as textual RAG evidence. Keep both instead of
    # overwriting one percentage with the other.
    meta["semantic_covered_page_count"] = meta.get("covered_page_count", 0)
    meta["semantic_page_coverage"] = meta.get("coverage", 0.0)
    block_ledger = {
        int(item.get("page") or 0): item
        for item in (meta.get("page_ledger") or [])
        if isinstance(item, dict)
    }
    merged_ledger = []
    for item in normalized.get("page_ledger") or []:
        if not isinstance(item, dict):
            continue
        page_num = int(item.get("page") or 0)
        merged = dict(item)
        if page_num in block_ledger:
            merged["has_blocks"] = bool(block_ledger[page_num].get("has_blocks"))
        merged_ledger.append(merged)
    for key in (
        "quality_status",
        "expected_page_count",
        "observed_page_count",
        "covered_page_count",
        "coverage",
        "failed_pages",
        "unexpected_pages",
    ):
        meta[key] = normalized.get(key)
    meta["page_ledger"] = merged_ledger
    meta["rag_full_text_chars"] = len(str(normalized.get("full_text") or ""))
    meta["rag_page_coverage"] = normalized.get("coverage", 0.0)
    quality_report = normalized.get("quality_report")
    if isinstance(quality_report, dict):
        for key in (
            "raw_type_counts",
            "emitted_type_counts",
            "filtered_type_counts",
            "unknown_types",
            "table_count",
            "malformed_table_count",
        ):
            meta[key] = quality_report.get(key)
        if "warnings" in quality_report:
            meta["warnings"] = quality_report.get("warnings")
        if "block_validation" in quality_report:
            meta["block_validation"] = quality_report.get("block_validation")
    block_index["mineru_meta"] = meta
    return block_index


def _mineru_ready_status(value: dict) -> str:
    quality = value.get("quality_status")
    if not quality and isinstance(value.get("mineru_meta"), dict):
        quality = value["mineru_meta"].get("quality_status")
    return "partial_ready" if quality == "partial_success" else "ready"


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
    previous_data = doc.get("data")
    data = dict(doc.get("data") or {})
    normalized_pages: dict[int, dict] = {}
    for page in normalized.get("pages") or []:
        if not isinstance(page, dict):
            continue
        page_copy = dict(page)
        content = str(page_copy.get("content") or page_copy.get("text") or "")
        page_copy["content"] = content
        page_copy["text"] = content
        page_copy["source"] = MINERU_RAG_INDEX_SOURCE
        try:
            page_num = max(1, int(page_copy.get("page") or page_copy.get("page_index", 0) + 1))
        except (TypeError, ValueError):
            continue
        page_copy["page"] = page_num
        page_copy["page_index"] = page_num - 1
        normalized_pages[page_num] = page_copy

    try:
        previous_page_count = max(0, int(data.get("total_pages") or 0))
    except (TypeError, ValueError):
        previous_page_count = 0
    try:
        expected_page_count = max(0, int(normalized.get("expected_page_count") or 0))
    except (TypeError, ValueError):
        expected_page_count = 0
    total_pages = max(previous_page_count, expected_page_count)
    if total_pages <= 0:
        total_pages = max(normalized_pages.keys(), default=0)
    page_quality = {
        int(item.get("page") or 0): item
        for item in (normalized.get("page_ledger") or [])
        if isinstance(item, dict)
    }
    pages = []
    for page_num in range(1, total_pages + 1):
        page_copy = dict(normalized_pages.get(page_num) or {
            "page": page_num,
            "page_index": page_num - 1,
            "content": "",
            "text": "",
            "source": MINERU_RAG_INDEX_SOURCE,
        })
        ledger_entry = page_quality.get(page_num) or {}
        page_copy["parse_status"] = ledger_entry.get("status", "unknown")
        if ledger_entry.get("reason"):
            page_copy["parse_failure_reason"] = ledger_entry["reason"]
        pages.append(page_copy)

    structured_table_bundles = normalized.get("structured_table_bundles") or []
    parse_manifest = _read_document_parse_manifest(doc_id, doc)
    data.update({
        "full_text": normalized.get("full_text", ""),
        "pages": pages,
        "total_pages": total_pages,
        "structured_table_bundles": structured_table_bundles,
        "structured_table_count": len(structured_table_bundles),
        "rag_index_source": MINERU_RAG_INDEX_SOURCE,
        "rag_source_hash": normalized.get("source_hash", ""),
        "rag_normalizer_version": normalized.get("normalizer_version", ""),
        "rag_quality_report": normalized.get("quality_report") or {},
        "mineru_quality_status": normalized.get("quality_status", "success"),
        "mineru_expected_page_count": expected_page_count,
        "mineru_page_coverage": normalized.get("coverage", 0.0),
        "mineru_failed_pages": normalized.get("failed_pages") or [],
        "mineru_page_ledger": normalized.get("page_ledger") or [],
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
            "quality_status": normalized.get("quality_status", "success"),
            "expected_page_count": expected_page_count,
            "coverage": normalized.get("coverage", 0.0),
            "failed_pages": normalized.get("failed_pages") or [],
        })
        parse_manifest["metadata"] = metadata
        data["parse_manifest"] = parse_manifest
    doc["data"] = data
    _normalize_page_keys(doc)
    if not save_document(doc_id, doc):
        doc["data"] = previous_data
        raise RuntimeError("MinerU 问答数据写入失败")


def _restore_document_backup(doc_id: str, source: str = "pdf_native") -> dict:
    path = _document_backup_path(doc_id, source)
    if not path.exists():
        return {"restored": False}
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            restored_doc = json.load(f)
        _normalize_page_keys(restored_doc)
        previous_doc = documents_store.get(doc_id)
        documents_store[doc_id] = restored_doc
        if not save_document(doc_id, restored_doc):
            if previous_doc is None:
                documents_store.pop(doc_id, None)
            else:
                documents_store[doc_id] = previous_doc
            raise RuntimeError("文档备份恢复写入失败")
        return {"restored": True, "path": str(path)}
    except Exception as exc:
        logger.warning("[RagIndex] failed to restore document backup for %s: %s", doc_id, exc)
        return {"restored": False, "error": str(exc)}


def _replace_vector_index_from_temp(doc_id: str, temp_dir: Path) -> None:
    temp_index, temp_pkl = _vector_index_paths(doc_id, temp_dir)
    if not temp_index.exists() or not temp_pkl.exists():
        raise RuntimeError("临时问答索引未生成完整 index/pkl 文件")
    index_path, pkl_path = _vector_index_paths(doc_id)
    rollback_dir = temp_dir / f".swap-rollback-{uuid.uuid4().hex}"
    rollback_index = rollback_dir / index_path.name
    rollback_pkl = rollback_dir / pkl_path.name
    had_index = index_path.exists()
    had_pkl = pkl_path.exists()
    try:
        if had_index or had_pkl:
            rollback_dir.mkdir(parents=True, exist_ok=True)
        if had_index:
            shutil.copy2(index_path, rollback_index)
        if had_pkl:
            shutil.copy2(pkl_path, rollback_pkl)
        os.replace(str(temp_index), str(index_path))
        os.replace(str(temp_pkl), str(pkl_path))
    except Exception:
        # The pair has no multi-file OS primitive. Restore both members before
        # bubbling up so callers can run the larger transaction rollback.
        try:
            if had_index and rollback_index.exists():
                shutil.copy2(rollback_index, index_path)
            elif not had_index:
                index_path.unlink(missing_ok=True)
            if had_pkl and rollback_pkl.exists():
                shutil.copy2(rollback_pkl, pkl_path)
            elif not had_pkl:
                pkl_path.unlink(missing_ok=True)
        except Exception as restore_exc:
            logger.error("[RagIndex] failed to restore partial vector swap doc=%s: %s", doc_id, restore_exc)
        raise
    finally:
        shutil.rmtree(rollback_dir, ignore_errors=True)
    _index_cache.invalidate(doc_id)


def _cleanup_fresh_mineru_rag_publication(
    doc_id: str,
    document_snapshot: dict | None,
    *,
    semantic_publication: dict | None = None,
) -> dict:
    """Undo a failed first-time MinerU RAG publication.

    Source switches have a complete pre-existing RAG snapshot to restore.  A
    MinerU-first upload has no such vector pair, but it can still fail after
    the vector files and document data have been swapped.  Leaving those files
    behind makes a failed parse look partially published after a restart.
    """
    result: dict[str, object] = {
        "vectors_removed": [],
        "semantic_groups": {},
        "document_restored": False,
        "errors": [],
    }
    for path in _vector_index_paths(doc_id):
        try:
            if path.exists():
                path.unlink()
                result["vectors_removed"].append(str(path))
        except Exception as cleanup_exc:
            logger.error("[RagIndex] failed to remove fresh vector artifact %s: %s", path, cleanup_exc)
            result["errors"].append(f"vector:{path.name}:{cleanup_exc}")
    _index_cache.invalidate(doc_id)

    try:
        result["semantic_groups"] = _remove_current_semantic_groups(doc_id)
        # ``publish_generation`` stores its immutable files beneath a new
        # generation directory.  After removing the active manifest, delete
        # only the generation created by this failed publication; do not touch
        # unrelated history from another document.
        paths = (semantic_publication or {}).get("paths") if isinstance(semantic_publication, dict) else None
        if isinstance(paths, dict):
            generations_root = (DATA_DIR / "semantic_groups" / "generations").resolve()
            generation_dirs = set()
            for raw_path in paths.values():
                try:
                    path = Path(str(raw_path)).resolve()
                    if generations_root in path.parents:
                        generation_dirs.add(path.parent)
                except Exception:
                    continue
            for generation_dir in generation_dirs:
                shutil.rmtree(generation_dir, ignore_errors=True)
    except Exception as cleanup_exc:
        logger.error("[RagIndex] failed to clean fresh semantic artifacts for %s: %s", doc_id, cleanup_exc)
        result["errors"].append(f"semantic:{cleanup_exc}")

    if isinstance(document_snapshot, dict):
        current_document = documents_store.get(doc_id)
        restored_document = deepcopy(document_snapshot)
        documents_store[doc_id] = restored_document
        if save_document(doc_id, restored_document):
            result["document_restored"] = True
        else:
            if current_document is None:
                documents_store.pop(doc_id, None)
            else:
                documents_store[doc_id] = current_document
            result["errors"].append("document:restore_save_failed")
            logger.error("[RagIndex] failed to restore fresh document snapshot for %s", doc_id)
    return result


def _validate_temp_vector_index(
    doc_id: str,
    temp_dir: Path,
    *,
    expected_source: str = MINERU_RAG_INDEX_SOURCE,
    expected_parse_generation: str = "",
    expected_document_source_hash: str = "",
    expected_block_index_hash: str = "",
    expected_content_source: str = "",
    expected_evidence_schema_version: int = 0,
) -> tuple[bool, list[str]]:
    inspection = _inspect_vector_index_artifacts(
        doc_id,
        temp_dir,
        expected_source=expected_source,
        expected_parse_generation=expected_parse_generation,
        expected_document_source_hash=expected_document_source_hash,
        expected_block_index_hash=expected_block_index_hash,
        expected_content_source=expected_content_source,
        expected_evidence_schema_version=expected_evidence_schema_version,
    )
    failures: list[str] = [f"temp_{error}" for error in inspection.get("errors") or []]
    data = inspection.get("_data")
    if not isinstance(data, dict):
        failures.append("temp_pkl_legacy_shape")
        return False, list(dict.fromkeys(failures))
    try:
        index_version = int(data.get("index_version") or 0)
    except (TypeError, ValueError):
        index_version = 0
    if index_version != RAG_INDEX_VERSION:
        failures.append("temp_index_version_mismatch")
    chunks = data.get("chunks") or []
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
    failures = list(dict.fromkeys(failures))
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
    expected_source: str = "",
    expected_parse_generation: str = "",
    expected_document_source_hash: str = "",
    expected_block_index_hash: str = "",
    embedding_provider: Optional[str] = None,
) -> dict:
    """Prepare the post-swap semantic group rebuild from the validated temp index.

    MinerU rebuild writes the vector index into a temp directory first, so
    create_index(..., build_semantic_groups=False) cannot build semantic groups
    yet. Preparing here keeps failures before the active index is replaced.
    """
    resolved_temp_dir = temp_dir.resolve()
    active_vector_dir = VECTOR_STORE_DIR.resolve()
    staging_root = (VECTOR_STORE_DIR / "_tmp").resolve()
    if resolved_temp_dir == active_vector_dir:
        raise RuntimeError("禁止从活动向量目录准备 semantic groups，必须使用当前 staging 产物")
    try:
        resolved_temp_dir.relative_to(staging_root)
    except ValueError as exc:
        raise RuntimeError("semantic groups 只能从向量索引 staging 目录重建") from exc
    if not expected_source or not expected_parse_generation or not expected_document_source_hash:
        raise RuntimeError(
            "semantic groups 重建必须绑定 index_source、parse_generation 和 document_source_hash"
        )

    inspection = _inspect_vector_index_artifacts(
        doc_id,
        resolved_temp_dir,
        expected_source=expected_source,
        expected_parse_generation=expected_parse_generation,
        expected_document_source_hash=expected_document_source_hash,
        expected_block_index_hash=expected_block_index_hash,
    )
    if not inspection.get("valid"):
        raise RuntimeError(
            "临时问答索引未通过 semantic 重建身份门禁: "
            + ", ".join(inspection.get("errors") or ["unknown"])
        )
    data = inspection.get("_data")
    if not isinstance(data, dict):
        raise RuntimeError("临时问答索引格式异常，无法准备意群索引重建")
    semantic_identity = _extract_vector_semantic_identity(data)
    if not _semantic_generation_identity_complete(semantic_identity):
        raise RuntimeError("临时问答索引缺少完整 embedding/build 身份，拒绝生成 semantic groups")
    chunks = [str(chunk or "") for chunk in (data.get("chunks") or [])]
    if not chunks:
        raise RuntimeError("临时问答索引分块为空，无法准备意群索引重建")
    chunk_pages = list(data.get("chunk_pages") or [])
    chunk_types = list(data.get("chunk_types") or [])
    chunk_metadata = list(data.get("chunk_metadata") or [])
    chunk_pages = (chunk_pages[:len(chunks)] + [0] * len(chunks))[:len(chunks)]
    chunk_types = (chunk_types[:len(chunks)] + [""] * len(chunks))[:len(chunks)]
    chunk_metadata = (chunk_metadata[:len(chunks)] + [{} for _ in chunks])[:len(chunks)]
    requested_embedding_model = str(embedding_model or "").strip()
    requested_embedding_provider = _resolve_embedding_provider(
        requested_embedding_model,
        embedding_provider,
    )
    verified_embedding = _resolve_verified_query_embedding_identity(
        data,
        api_key=embedding_api_key,
        embedding_model=requested_embedding_model,
        embedding_provider=requested_embedding_provider,
        embedding_api_host=embedding_api_host,
    )
    effective_embedding_model = _compose_provider_scoped_embedding_model(
        verified_embedding["model"],
        verified_embedding["provider"],
    )
    embed_fn = get_embedding_function(
        effective_embedding_model,
        verified_embedding["api_key"],
        verified_embedding["api_host"],
        allow_model_fallback=False,
    )
    return {
        "chunks": chunks,
        "chunk_pages": chunk_pages,
        "chunk_types": chunk_types,
        "chunk_metadata": chunk_metadata,
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
        "index_source": str(data.get("index_source") or ""),
        "parse_generation": str((data.get("index_meta") or {}).get("parse_generation") or ""),
        "document_source_hash": str((data.get("index_meta") or {}).get("document_source_hash") or ""),
        "vector_count": int(inspection.get("vector_count") or 0),
        "semantic_identity": semantic_identity,
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
    restored_vector = _inspect_vector_index_artifacts(doc_id, VECTOR_STORE_DIR)
    restored_vector_identity = _extract_vector_semantic_identity(restored_vector.get("_data"))
    restored_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
    legacy_manifest = _is_legacy_parse_manifest(restored_manifest)
    semantic_restore = _restore_semantic_group_backup(
        doc_id,
        safe_source,
        source_hash=("" if legacy_manifest else str(restored_manifest.get("source_hash") or "")),
        transaction_id=("" if legacy_manifest else str(restored_manifest.get("generation") or "")),
        semantic_identity=(restored_vector_identity if _semantic_generation_identity_complete(restored_vector_identity) else None),
    )
    backup_manifest = _load_complete_rag_backup_manifest(doc_id, safe_source)
    semantic_required = bool((backup_manifest.get("semantic_groups") or {}).get("backed_up"))
    restored = bool(
        doc_restore.get("restored")
        and (
            semantic_restore.get("restored")
            or semantic_restore.get("degraded")
            or not semantic_required
        )
    )
    _index_cache.invalidate(doc_id)
    if not restored:
        _set_document_index_status(
            doc_id,
            "failed",
            stage="rollback_failed",
            error="MinerU 问答索引回滚不完整",
        )
        raise RuntimeError("MinerU 问答索引回滚不完整")
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


def _rollback_local_rag_index_if_current(
    doc_id: str,
    *,
    parse_generation: str,
    document_source_hash: str,
    previous_source: str,
    replaced_current_index: bool,
    error: str,
) -> bool:
    """Restore a failed local RAG build only while it still owns the document."""
    with _get_document_publication_lock(doc_id):
        if not _parse_generation_matches_current_document(
            doc_id,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
        ):
            logger.info(
                "[RagIndex] discard stale local rebuild rollback doc=%s generation=%s",
                doc_id,
                parse_generation,
            )
            return False

        backup_manifest = _load_complete_rag_backup_manifest(doc_id, previous_source)
        current_manifest = _read_document_parse_manifest(doc_id, documents_store.get(doc_id))
        backup_matches_current = (
            _is_legacy_parse_manifest(current_manifest)
            or _rag_backup_matches_current_parse(doc_id, backup_manifest, current_manifest)
        )
        if replaced_current_index and backup_manifest and backup_matches_current:
            try:
                _restore_vector_index_backup(doc_id, previous_source)
            except Exception as restore_exc:
                logger.error("[RagIndex] 本地索引升级回滚失败 doc=%s: %s", doc_id, restore_exc)
        elif replaced_current_index and backup_manifest:
            logger.warning(
                "[RagIndex] skip mismatched local rollback snapshot doc=%s generation=%s",
                doc_id,
                parse_generation,
            )
        _set_document_index_status(
            doc_id,
            "failed",
            stage="local_rag_index_upgrade_failed",
            error=error,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
        )
        return True


def _rebuild_local_rag_index(
    doc_id: str,
    *,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_api_host: Optional[str],
    summary_api_key: Optional[str],
    summary_model: str = "gpt-4o-mini",
    summary_provider: str = "openai",
    summary_api_host: str = "",
    embedding_provider: Optional[str] = None,
) -> dict:
    """用当前解析身份的 block_index 事务性升级本地问答索引。"""
    document_lock = _get_document_operation_lock(doc_id)
    if not document_lock.acquire(blocking=False):
        raise RuntimeError("该文档正在执行解析或索引重建，请稍后再试")

    temp_dir = VECTOR_STORE_DIR / "_tmp" / f"{doc_id}.local.{uuid.uuid4().hex}"
    temp_semantic_dir = DATA_DIR / "semantic_groups" / "_tmp" / f"{doc_id}.local.{uuid.uuid4().hex}"
    replaced_current_index = False
    previous_source = "pdf_native"
    parse_generation = ""
    document_source_hash = ""
    try:
        doc = documents_store.get(doc_id)
        if not isinstance(doc, dict):
            raise RuntimeError("文档记录不存在")
        parse_manifest = _read_document_parse_manifest(doc_id, doc)
        if parse_manifest.get("resolved_route") == PARSE_ROUTE_MINERU:
            raise RuntimeError("MinerU 文档必须使用 MinerU 结构化结果重建问答索引")
        parse_generation = str(parse_manifest.get("generation") or "")
        document_source_hash = str(parse_manifest.get("source_hash") or "")
        _require_current_parse_generation(
            doc_id,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
        )

        block_index = ensure_block_index(
            doc_id=doc_id,
            doc=doc,
            data_dir=DATA_DIR,
            pdf_path=_resolve_document_pdf_path(doc),
        )
        data = doc.get("data") or {}
        rag_source = _rag_source_from_block_index(block_index, data)
        if not rag_source["full_text"] or not rag_source["pages"]:
            raise RuntimeError("当前文档没有可用于重建问答索引的正文")
        block_index_hash = str(
            block_index.get("block_index_hash") or block_index.get("block_index_revision") or ""
        ).strip()
        if not block_index_hash:
            raise RuntimeError("当前阅读块索引缺少内容修订，拒绝构建问答索引")

        previous_meta = _read_vector_index_meta(doc_id)
        previous_source = str(previous_meta.get("index_source") or "pdf_native")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_semantic_dir.mkdir(parents=True, exist_ok=True)
        _set_document_index_status(doc_id, "running", stage="upgrading_local_rag_index")

        local_index_meta = {
            "source_hash": data.get("rag_source_hash") or document_source_hash,
            "document_source_hash": document_source_hash,
            "parse_generation": parse_generation,
            "parser_route": parse_manifest.get("resolved_route", ""),
            "rebuilt_at": utc_now_iso(),
            "previous_index_source": previous_source,
            "content_source": "block_index_evidence" if rag_source["evidence_chunks"] else "document_full_text",
            "block_index_version": block_index.get("version", ""),
            "block_index_hash": block_index_hash,
            "evidence_schema_version": rag_source.get("evidence_schema_version", 0),
        }
        local_index_kwargs = {
            "pages": rag_source["pages"],
            "structured_table_bundles": data.get("structured_table_bundles"),
            "summary_api_key": summary_api_key,
            "index_source": "pdf_native",
            "index_meta": local_index_meta,
            "build_semantic_groups": False,
        }
        if rag_source["evidence_chunks"] and _callable_accepts_keyword(create_index, "evidence_chunks"):
            local_index_kwargs["evidence_chunks"] = rag_source["evidence_chunks"]
        _call_with_optional_keyword(
            create_index,
            "embedding_provider",
            _normalize_optional_provider_id(embedding_provider),
            doc_id,
            rag_source["full_text"],
            str(temp_dir),
            embedding_model,
            embedding_api_key,
            embedding_api_host,
            **local_index_kwargs,
        )
        temp_ok, temp_failures = _validate_temp_vector_index(
            doc_id,
            temp_dir,
            expected_source="pdf_native",
            expected_parse_generation=parse_generation,
            expected_document_source_hash=document_source_hash,
            expected_block_index_hash=block_index_hash,
            expected_content_source=local_index_meta["content_source"],
            expected_evidence_schema_version=int(local_index_meta.get("evidence_schema_version") or 0),
        )
        if not temp_ok:
            raise RuntimeError("本地问答索引质量门失败: " + ", ".join(temp_failures))

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
            expected_source="pdf_native",
            expected_parse_generation=parse_generation,
            expected_document_source_hash=document_source_hash,
            expected_block_index_hash=block_index_hash,
            embedding_provider=embedding_provider,
        )
        semantic_result = _build_semantic_group_index(
            doc_id,
            semantic_rebuild["chunks"],
            rag_source["pages"],
            semantic_rebuild["embed_fn"],
            semantic_rebuild["api_key"],
            chunk_pages=semantic_rebuild["chunk_pages"],
            chunk_types=semantic_rebuild["chunk_types"],
            chunk_metadata=semantic_rebuild["chunk_metadata"],
            model=semantic_rebuild["model"],
            provider=semantic_rebuild["provider"],
            endpoint=semantic_rebuild["endpoint"],
            output_dir=str(temp_semantic_dir),
            raise_on_error=True,
            semantic_identity=semantic_rebuild["semantic_identity"],
        )
        semantic_validation = _validate_temp_semantic_groups(
            doc_id,
            temp_semantic_dir,
            semantic_result,
            expected_identity=semantic_rebuild["semantic_identity"],
            expected_vector_dimension=int(semantic_rebuild["semantic_identity"].get("vector_dimension") or 0),
        )

        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=document_source_hash,
            )
            had_previous_rag = _active_rag_index_matches_current_parse(doc_id, parse_manifest)
            backup_manifest = (
                _backup_current_rag_state(doc_id, previous_source)
                if had_previous_rag
                else {"complete": False, "reason": "active_vector_parse_identity_mismatch"}
            )
            if had_previous_rag and not backup_manifest.get("complete"):
                raise RuntimeError("无法创建完整的当前 RAG 快照，已取消升级")
            _replace_vector_index_from_temp(doc_id, temp_dir)
            replaced_current_index = True
            _publish_temp_semantic_groups(
                doc_id,
                temp_semantic_dir,
                semantic_validation,
                source_hash=document_source_hash,
                transaction_id=parse_generation,
                semantic_identity=semantic_rebuild["semantic_identity"],
            )
            _set_document_index_status(
                doc_id,
                "ready",
                stage="ready",
                parse_generation=parse_generation,
                document_source_hash=document_source_hash,
            )

        return {
            "status": "ready",
            "message": "本地问答索引已升级",
            "rag_index": _get_rag_index_status(doc_id),
            "normalized": {
                "page_count": len(rag_source["pages"]),
                "full_text_chars": len(rag_source["full_text"]),
                "estimated_embedding_tokens": _estimate_text_tokens(rag_source["full_text"]),
                "estimated_chunk_count": _estimate_chunk_count(rag_source["full_text"]),
                "structured_table_count": len(data.get("structured_table_bundles") or []),
                "narrative_block_count": rag_source["block_count"],
            },
        }
    except Exception as exc:
        _rollback_local_rag_index_if_current(
            doc_id,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
            previous_source=previous_source,
            replaced_current_index=replaced_current_index,
            error=str(exc),
        )
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(temp_semantic_dir, ignore_errors=True)
        document_lock.release()


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
    embedding_provider: Optional[str] = None,
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
            embedding_provider=embedding_provider,
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
    embedding_provider: Optional[str] = None,
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

    previous_meta = _read_vector_index_meta(doc_id)
    previous_source = previous_meta.get("index_source") or "pdf_native"

    normalized, failures = _normalize_mineru_for_document(doc_id, payload, doc=doc)
    quality_report = dict(normalized.get("quality_report") or {})
    if failures:
        raise RuntimeError(f"MinerU 问答索引重建失败，已保留原索引: {', '.join(failures)}")
    ready_status = _mineru_ready_status(normalized)
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
    )
    if str(block_index.get("source") or "").strip().lower() != MINERU_BLOCK_INDEX_SOURCE:
        # A legacy manual rebuild can predate the persisted reading index.  It
        # still has the current identity-checked raw payload, so build an
        # in-memory structured source rather than falling back to page text.
        block_index = build_block_index_from_mineru_payload(
            doc_id=doc_id,
            doc=doc,
            payload=payload,
            pdf_path=_resolve_document_pdf_path(doc),
        )
    rag_source = _rag_source_from_block_index(block_index, doc.get("data") or {})
    artifact_figures = _artifact_figures_from_block_index(block_index)
    block_index_hash = str(
        block_index.get("block_index_hash") or block_index.get("block_index_revision") or ""
    ).strip()
    if not block_index_hash:
        raise RuntimeError("当前 MinerU 阅读块索引缺少内容修订，拒绝构建问答索引")

    had_previous_rag = _active_rag_index_matches_current_parse(doc_id, parse_manifest)
    fresh_document_snapshot = deepcopy(doc) if not had_previous_rag else None

    artifact = build_document_parse_artifact(
        doc_id=doc_id,
        provider="mineru",
        provider_version=str(normalized.get("normalizer_version") or ""),
        pages=normalized.get("pages") or [],
        tables=normalized.get("structured_table_bundles") or [],
        figures=artifact_figures,
        warnings=(normalized.get("quality_report") or {}).get("warnings") or [],
        capabilities={
            "per_page_text": True,
            "document_structure": True,
            "structured_tables": True,
            "figures": bool(artifact_figures),
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

    full_text = rag_source.get("full_text") or normalized.get("full_text", "")
    pages = rag_source.get("pages") or artifact["pages"]
    evidence_chunks = rag_source.get("evidence_chunks") or []
    structured_table_bundles = artifact["tables"]
    if not full_text or not pages:
        raise RuntimeError("MinerU 规范化结果为空，已保留原索引")
    # Publish the same page text that feeds the vector index.  Structured
    # tables remain in their dedicated bundle contract; this prevents the
    # visible document, retrieval and downstream overview from drifting back
    # to three independently flattened representations.
    normalized["full_text"] = full_text
    normalized["pages"] = [dict(page) for page in pages]
    normalized["evidence_schema_version"] = rag_source.get("evidence_schema_version", 0)

    temp_dir = VECTOR_STORE_DIR / "_tmp" / f"{doc_id}.mineru"
    temp_semantic_dir = DATA_DIR / "semantic_groups" / "_tmp" / f"{doc_id}.mineru"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    if temp_semantic_dir.exists():
        shutil.rmtree(temp_semantic_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_semantic_dir.mkdir(parents=True, exist_ok=True)

    _set_document_index_status(
        doc_id,
        "running",
        stage="building_vector_index",
        parse_generation=expected_parse_generation,
        document_source_hash=expected_document_source_hash,
    )
    backup = {}
    doc_backup = {}
    semantic_backup = {}
    semantic_cleanup = {}
    fresh_cleanup = {}
    replaced_current_index = False
    try:
        mineru_index_meta = {
            "source_hash": normalized.get("source_hash", ""),
            "document_source_hash": parse_manifest.get("source_hash", ""),
            "parse_generation": parse_manifest.get("generation", ""),
            "parser_route": parse_manifest.get("resolved_route", ""),
            "rebuilt_at": utc_now_iso(),
            "previous_index_source": previous_source,
            "normalizer_version": normalized.get("normalizer_version", ""),
            "parse_artifact_ref": normalized["parse_artifact"]["ref"],
            "quality_status": normalized.get("quality_status", "success"),
            "expected_page_count": normalized.get("expected_page_count", 0),
            "coverage": normalized.get("coverage", 0.0),
            "failed_pages": normalized.get("failed_pages") or [],
            "content_source": "block_index_evidence" if evidence_chunks else "mineru_normalized_pages",
            "block_index_version": block_index.get("version", ""),
            "block_index_hash": block_index_hash,
            "evidence_schema_version": rag_source.get("evidence_schema_version", 0),
        }
        mineru_index_kwargs = {
            "pages": pages,
            "structured_table_bundles": structured_table_bundles,
            "summary_api_key": summary_api_key,
            "index_source": MINERU_RAG_INDEX_SOURCE,
            "index_meta": mineru_index_meta,
            "build_semantic_groups": False,
        }
        if evidence_chunks and _callable_accepts_keyword(create_index, "evidence_chunks"):
            mineru_index_kwargs["evidence_chunks"] = evidence_chunks
        _call_with_optional_keyword(
            create_index,
            "embedding_provider",
            _normalize_optional_provider_id(embedding_provider),
            doc_id,
            full_text,
            str(temp_dir),
            embedding_model,
            embedding_api_key,
            embedding_api_host,
            **mineru_index_kwargs,
        )
        _set_document_index_status(
            doc_id,
            "running",
            stage="validating_vector_index",
            parse_generation=expected_parse_generation,
            document_source_hash=expected_document_source_hash,
        )
        temp_meta = _read_vector_index_meta(doc_id, temp_dir)
        if temp_meta.get("index_source") != MINERU_RAG_INDEX_SOURCE:
            raise RuntimeError("临时索引缺少 MinerU 来源标记")
        temp_ok, temp_failures = _validate_temp_vector_index(
            doc_id,
            temp_dir,
            expected_source=MINERU_RAG_INDEX_SOURCE,
            expected_parse_generation=expected_parse_generation,
            expected_document_source_hash=expected_document_source_hash,
            expected_block_index_hash=block_index_hash,
            expected_content_source=mineru_index_meta["content_source"],
            expected_evidence_schema_version=int(mineru_index_meta.get("evidence_schema_version") or 0),
        )
        if not temp_ok:
            raise RuntimeError(f"MinerU 问答索引质量门失败，已保留原索引: {', '.join(temp_failures)}")
        _set_document_index_status(
            doc_id,
            "running",
            stage="preparing_semantic_index",
            parse_generation=expected_parse_generation,
            document_source_hash=expected_document_source_hash,
        )
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
            expected_source=MINERU_RAG_INDEX_SOURCE,
            expected_parse_generation=expected_parse_generation,
            expected_document_source_hash=expected_document_source_hash,
            expected_block_index_hash=block_index_hash,
            embedding_provider=embedding_provider,
        )
        _set_document_index_status(
            doc_id,
            "running",
            stage="building_semantic_index",
            parse_generation=expected_parse_generation,
            document_source_hash=expected_document_source_hash,
        )
        semantic_result = _build_semantic_group_index(
            doc_id, semantic_rebuild["chunks"], pages, semantic_rebuild["embed_fn"], semantic_rebuild["api_key"],
            chunk_pages=semantic_rebuild["chunk_pages"],
            chunk_types=semantic_rebuild["chunk_types"],
            chunk_metadata=semantic_rebuild["chunk_metadata"],
            model=semantic_rebuild["model"], provider=semantic_rebuild["provider"], endpoint=semantic_rebuild["endpoint"],
            output_dir=str(temp_semantic_dir), raise_on_error=True,
            semantic_identity=semantic_rebuild["semantic_identity"],
        )
        _set_document_index_status(
            doc_id,
            "running",
            stage="validating_semantic_index",
            parse_generation=expected_parse_generation,
            document_source_hash=expected_document_source_hash,
        )
        semantic_validation = _validate_temp_semantic_groups(
            doc_id,
            temp_semantic_dir,
            semantic_result,
            expected_identity=semantic_rebuild["semantic_identity"],
            expected_vector_dimension=int(semantic_rebuild["semantic_identity"].get("vector_dimension") or 0),
        )
        # The expensive temp build above intentionally runs without holding
        # the upload path. The irreversible swap below is short and guarded
        # by both the publication lock and the active parse identity.
        _set_document_index_status(
            doc_id,
            "running",
            stage="publishing_rag_index",
            parse_generation=expected_parse_generation,
            document_source_hash=expected_document_source_hash,
        )
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
                # A MinerU-first upload has no valid vector pair to back up.
                # Keep a document snapshot above and explicitly clean any
                # newly published artifacts if a later publication stage fails.
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
                semantic_identity=semantic_rebuild["semantic_identity"],
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
            _set_document_index_status(doc_id, ready_status, stage=ready_status)
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
                elif not had_previous_rag and replaced_current_index:
                    fresh_cleanup = _cleanup_fresh_mineru_rag_publication(
                        doc_id,
                        fresh_document_snapshot,
                        semantic_publication=semantic_cleanup,
                    )
                    logger.warning(
                        "[RagIndex] cleaned failed first-time MinerU publication for %s: %s",
                        doc_id,
                        fresh_cleanup,
                    )
                failure_message = (
                    "MinerU 问答索引重建失败，已清理未完成的首次发布"
                    if not had_previous_rag and replaced_current_index
                    else "MinerU 问答索引重建失败，已保留原索引"
                )
                _set_document_index_status(doc_id, "failed", stage="rebuilding_rag_index_failed", error=failure_message)
            else:
                logger.info("[RagIndex] skip rollback for superseded parse generation doc=%s", doc_id)
        raise exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(temp_semantic_dir, ignore_errors=True)

    return {
        "status": ready_status,
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
            "quality_status": normalized.get("quality_status", "success"),
            "expected_page_count": normalized.get("expected_page_count", 0),
            "coverage": normalized.get("coverage", 0.0),
            "failed_pages": normalized.get("failed_pages") or [],
            "page_ledger": normalized.get("page_ledger") or [],
        },
        "backup": {
            **backup,
            "manifest": backup_manifest if 'backup_manifest' in locals() else {},
            "transaction": transaction_journal if 'transaction_journal' in locals() else {},
            "document": doc_backup,
            "semantic_groups": semantic_backup,
            "semantic_group_cleanup": semantic_cleanup,
            "fresh_publication_cleanup": fresh_cleanup,
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
        "ocr_status": "not_started",
        "ocr_execution_status": "not_started",
        "ocr_adoption_status": "not_started",
        "ocr_failed_pages": [],
        "ocr_execution_successful_pages": [],
        "ocr_applied_pages": [],
        "ocr_unapplied_pages": [],
        "ocr_warning": "",
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
    embedding_provider: Optional[str] = None,
) -> dict:
    """Persist a pending MinerU-first document and queue its atomic publication."""
    mineru_config_error = _mineru_configuration_error()
    if mineru_config_error:
        raise HTTPException(
            status_code=400,
            detail=f"已选择 MinerU 全程解析，但{mineru_config_error}",
        )

    embedding_identity = _require_explicit_rag_embedding_identity_or_400(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
        embedding_api_key=embedding_api_key,
        operation="MinerU 问答索引发布",
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
            # Deliberately persist only the non-secret identity.  It makes an
            # interrupted or mismatched index diagnosable without ever writing
            # an API key into the document manifest.
            "rag_embedding_identity": {
                "model": embedding_identity["model"],
                "provider": embedding_identity["provider"],
                "api_host": embedding_identity["api_host"],
            },
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
        previous_document = documents_store.get(doc_id)
        documents_store[doc_id] = {
            "filename": filename,
            "upload_time": datetime.now().isoformat(),
            "data": pending_data,
            "pdf_url": f"/uploads/{pdf_filename}",
        }
        _normalize_page_keys(documents_store[doc_id])
        _store_new_document_or_raise(
            doc_id,
            documents_store[doc_id],
            previous=previous_document,
            message="MinerU 解析任务的文档记录写入失败",
        )
        _set_document_index_status(doc_id, "queued", stage="waiting_for_mineru")
    deep_status = _queue_mineru_deep_parse(
        doc_id,
        full_route_options={
            "embedding_model": embedding_identity["model"],
            "embedding_api_key": embedding_identity["api_key"],
            "embedding_api_host": embedding_identity["api_host"],
            "embedding_provider": embedding_identity["provider"],
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
        "ocr_status": "not_started",
        "ocr_execution_status": "not_started",
        "ocr_adoption_status": "not_started",
        "ocr_failed_pages": [],
        "ocr_execution_successful_pages": [],
        "ocr_applied_pages": [],
        "ocr_unapplied_pages": [],
        "ocr_warning": "",
        "extraction_quality": "pending_mineru",
        "extraction_method": "mineru_pending",
        "parse_manifest": pending_data["parse_manifest"],
        "deep_parse": deep_status,
        "indexing_status": "waiting_for_mineru",
    }


def _ocr_result_has_success(ocr_result) -> bool:
    pages = getattr(ocr_result, "pages", None)
    return any(getattr(page, "success", False) for page in pages or [])


def _display_ocr_page_numbers(page_numbers) -> list:
    display_pages = []
    for page in page_numbers or []:
        try:
            page_index = int(page)
        except (TypeError, ValueError):
            continue
        if page_index < 0:
            continue
        display_pages.append(page_index + 1)
    return sorted(set(display_pages))


def _append_ocr_warning(result: dict, message: Optional[str]) -> None:
    warning = str(message or "").strip()
    if not warning:
        return
    current = str(result.get("ocr_warning") or "").strip()
    if not current:
        result["ocr_warning"] = warning
        return
    parts = [part.strip() for part in current.split("；") if part.strip()]
    if warning not in parts:
        parts.append(warning)
    result["ocr_warning"] = "；".join(parts)


def _finalize_ocr_status(result: dict) -> None:
    current = str(result.get("ocr_status") or "").strip().lower()
    if current in {"disabled", "not_needed", "unavailable"}:
        result["ocr_execution_status"] = current
        result["ocr_adoption_status"] = current
        return

    def _page_set(key: str) -> set[int]:
        pages: set[int] = set()
        for value in result.get(key) or []:
            try:
                page = int(value)
            except (TypeError, ValueError):
                continue
            if page > 0:
                pages.add(page)
        return pages

    target_pages = _page_set("ocr_target_pages")
    failed_pages = _page_set("ocr_failed_pages")
    execution_successful_pages = _page_set("ocr_execution_successful_pages")
    applied_pages = (
        _page_set("ocr_applied_pages")
        or _page_set("ocr_pages")
        or _page_set("ocr_successful_pages")
    )
    unapplied_pages = execution_successful_pages - applied_pages

    result["ocr_target_pages"] = sorted(target_pages)
    result["ocr_failed_pages"] = sorted(failed_pages)
    result["ocr_execution_successful_pages"] = sorted(execution_successful_pages)
    result["ocr_applied_pages"] = sorted(applied_pages)
    result["ocr_successful_pages"] = sorted(applied_pages)
    result["ocr_unapplied_pages"] = sorted(unapplied_pages)
    result["ocr_used"] = bool(applied_pages)

    if result.get("ocr_attempted"):
        if failed_pages:
            execution_status = "partial_success" if execution_successful_pages else "failed"
        else:
            execution_status = "success" if execution_successful_pages else "failed"
    elif failed_pages or result.get("ocr_error"):
        execution_status = "failed"
    else:
        execution_status = "not_started"

    if applied_pages:
        adoption_incomplete = bool(
            failed_pages
            or unapplied_pages
            or (target_pages and not target_pages.issubset(applied_pages))
        )
        adoption_status = "partial_success" if adoption_incomplete else "success"
    elif execution_successful_pages:
        adoption_status = "not_applied"
    elif execution_status == "failed":
        adoption_status = "failed"
    else:
        adoption_status = "not_started"

    result["ocr_execution_status"] = execution_status
    result["ocr_adoption_status"] = adoption_status
    # ocr_status is the user-facing outcome: whether OCR changed the document.
    result["ocr_status"] = adoption_status

    if unapplied_pages:
        page_info = "、".join(str(page) for page in sorted(unapplied_pages))
        if applied_pages:
            _append_ocr_warning(
                result,
                f"部分 OCR 结果未达到正文替换阈值，未采用这些页面的 OCR 文本（页码: {page_info}）",
            )
        else:
            _append_ocr_warning(
                result,
                "OCR 执行成功，但识别文本未达到正文替换阈值，未采用 OCR 文本，"
                f"已保留原始提取文本（页码: {page_info}）",
            )


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
        ocr_backend: 页级 OCR 后端 - "auto"、"tesseract" 或 "paddleocr"
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
        "ocr_backends": [],
        "ocr_status": "not_started",
        "ocr_execution_status": "not_started",
        "ocr_adoption_status": "not_started",
        "ocr_attempted": False,
        "ocr_warning": "",
        "ocr_failed_pages": [],
        "ocr_execution_successful_pages": [],
        "ocr_applied_pages": [],
        "ocr_unapplied_pages": [],
        "ocr_successful_pages": [],
        "ocr_target_pages": [],
        "ocr_pages": [],
        "extraction_quality": "good" if avg_quality >= 80 else ("acceptable" if avg_quality >= 60 else "poor"),
        "extraction_method": extraction_method,
        "avg_quality_score": round(avg_quality, 1),
        "pages_needing_ocr": pages_needing_ocr,
        "structured_table_bundles": structured_table_bundles,
        "structured_table_count": len(structured_table_bundles),
    }
    ocr_mode = (enable_ocr or "auto").strip().lower()
    ocr_target_pages = select_ocr_target_pages(enable_ocr, total_pages, pages_needing_ocr)

    if ocr_mode == "never":
        result["ocr_status"] = "disabled"
        _finalize_ocr_status(result)
        return result

    if not ocr_target_pages:
        logger.debug("[PDF] 无需执行 OCR 或 OCR 已禁用 (mode=%s, avg_quality=%.1f)", enable_ocr, avg_quality)
        result["ocr_status"] = "not_needed"
        _finalize_ocr_status(result)
        return result
    
    # 通过注册表获取 OCR 适配器。优先使用本次上传请求的设置，缺省时回退到后端全局配置。
    selected_ocr_backend = (ocr_backend or settings.ocr_backend or "auto").strip().lower()
    legacy_structured_ocr_warning = ""
    if selected_ocr_backend in {"mineru", "mistral", "doc2x"}:
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
        ocr_error = format_local_ocr_unavailable_message(selected_ocr_backend)
        logger.warning(
            "[PDF] 需要对 %s 页执行 OCR，但后端不可用: %s",
            len(ocr_target_pages),
            ocr_error,
        )
        result["ocr_status"] = "unavailable"
        result["ocr_failed_pages"] = _display_ocr_page_numbers(ocr_target_pages)
        result["ocr_error"] = ocr_error
        _append_ocr_warning(result, legacy_structured_ocr_warning)
        _append_ocr_warning(result, ocr_error)
        _finalize_ocr_status(result)
        return result
    
    if pdf_bytes is None:
        logger.warning("[PDF] 需要 OCR 但未提供 pdf_bytes")
        result["ocr_status"] = "unavailable"
        result["ocr_failed_pages"] = _display_ocr_page_numbers(ocr_target_pages)
        result["ocr_error"] = "无法执行 OCR：缺少 PDF 原始数据"
        _append_ocr_warning(result, "无法执行 OCR：缺少 PDF 原始数据")
        _finalize_ocr_status(result)
        return result
    
    # 使用适配器系统执行逐页 OCR
    logger.info("[PDF] 开始逐页 OCR，共 %s 页，后端: %s", len(ocr_target_pages), adapter.name)
    primary_outcome_recorded = False
    try:
        # 调用适配器的 ocr_pages()，仅传入需要 OCR 的页码列表
        result["ocr_attempted"] = True
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
        _append_ocr_warning(result, legacy_structured_ocr_warning)
        
        # 处理部分页面 OCR 失败的警告信息
        if ocr_result.failed_pages:
            failed_info = ", ".join(str(p) for p in ocr_result.failed_pages)
            logger.warning("[PDF] OCR 警告: 部分页面 OCR 失败（页码: %s）", failed_info)

            fallback_adapter = _ocr_registry.get_local_adapter(exclude=[adapter.name])
            if fallback_adapter is not None and fallback_adapter.name != adapter.name:
                fallback_outcome_recorded = False
                fallback_target_pages = [
                    int(page) - 1
                    for page in (result.get("ocr_failed_pages") or [])
                    if str(page).isdigit() and int(page) > 0
                ]
                if fallback_target_pages:
                    try:
                        logger.info(
                            "[PDF] OCR 补跑失败页: %s -> %s，页码: %s",
                            adapter.name,
                            fallback_adapter.name,
                            result.get("ocr_failed_pages") or [],
                        )
                        fallback_result = fallback_adapter.ocr_pages(
                            pdf_bytes=pdf_bytes,
                            page_numbers=fallback_target_pages,
                            dpi=ocr_dpi,
                        )
                        if not _ocr_result_has_success(fallback_result):
                            record_ocr_provider_use(
                                fallback_adapter.name,
                                outcome="failure",
                                operation="page_ocr",
                                fallback=True,
                            )
                            fallback_outcome_recorded = True
                            raise RuntimeError("次级 OCR 补跑未返回任何成功页面")
                        record_ocr_provider_use(
                            fallback_adapter.name,
                            outcome="success",
                            operation="page_ocr",
                            fallback=True,
                        )
                        fallback_outcome_recorded = True
                        apply_ocr_result_to_pages(
                            result,
                            pages,
                            fallback_result,
                            fallback_target_pages,
                            heuristic_rebuild,
                            is_cjk=is_cjk,
                        )
                        if result.get("ocr_failed_pages"):
                            failed_info = ", ".join(str(p) for p in (result.get("ocr_failed_pages") or []))
                            result["ocr_warning"] = (
                                f"首选 OCR ({adapter.name}) 部分页失败，已由 {fallback_adapter.name} 补跑；"
                                f"仍有失败页: {failed_info}"
                            )
                        else:
                            result["ocr_warning"] = (
                                f"首选 OCR ({adapter.name}) 部分页失败，已由 {fallback_adapter.name} 补跑完成"
                            )
                    except Exception as fallback_err:
                        if not fallback_outcome_recorded:
                            record_ocr_provider_use(
                                fallback_adapter.name,
                                outcome="failure",
                                operation="page_ocr",
                                fallback=True,
                            )
                        result["ocr_error"] = str(fallback_err)
                        if not result.get("ocr_failed_pages"):
                            result["ocr_failed_pages"] = _display_ocr_page_numbers(fallback_target_pages)
                        failed_info = ", ".join(str(p) for p in (result.get("ocr_failed_pages") or []))
                        result["ocr_warning"] = (
                            f"首选 OCR ({adapter.name}) 部分页失败，且次级 OCR ({fallback_adapter.name}) 补跑失败"
                            + (f"（剩余失败页: {failed_info}）" if failed_info else "")
                        )
                        logger.warning(
                            "[PDF] OCR 失败页补跑也失败: %s -> %s (%s)",
                            adapter.name,
                            fallback_adapter.name,
                            fallback_err,
                        )
        
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
        result["ocr_failed_pages"] = _display_ocr_page_numbers(ocr_target_pages)
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
                    result["ocr_error"] = str(fallback_err)
                    if not result.get("ocr_failed_pages"):
                        result["ocr_failed_pages"] = _display_ocr_page_numbers(ocr_target_pages)
                    result["ocr_warning"] = (
                        f"在线 OCR ({adapter.name}) 和本地 OCR 回退均失败: {str(fallback_err)}"
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
    _append_ocr_warning(result, legacy_structured_ocr_warning)
    _finalize_ocr_status(result)
    return result


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    embedding_model: Optional[str] = Form(None),
    embedding_api_key: Optional[str] = Form(None),
    embedding_api_host: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    enable_ocr: Optional[str] = Form(None),
    ocr_backend: Optional[str] = Form(None),
    parse_route: Optional[str] = Form(None),
    embedding_provider: Optional[str] = Form(None),
):
    """
    上传并处理 PDF 文件
    
    Args:
        file: 要上传的 PDF 文件
        embedding_model: 必填的文本嵌入模型
        embedding_provider: 必填的 Embedding 提供商
        embedding_api_key: 远程 Embedding 服务必填的 API 密钥
        embedding_api_host: 远程 Embedding 服务必填的 API 地址
        api_key: 语义意群摘要使用的 LLM API 密钥（可选）
        enable_ocr: OCR 模式 - "auto"（自动检测）、"always"（始终启用）或 "never"（禁用）。
                    缺失时使用后端配置中的 ocr_default_mode 默认值。
        ocr_backend: OCR 后端。缺失时使用后端配置中的 ocr_backend 默认值。
        parse_route: 主解析路线 - PDF 缺省 mineru；auto 为 MinerU 优先、未配置时回退本地；也可显式 local 或 mineru。
    """
    filename = (file.filename or "").strip()
    filename_lower = filename.lower()
    is_pdf = filename_lower.endswith('.pdf')
    is_multi_format = is_supported_format(filename)

    if not is_pdf and not is_multi_format:
        supported = "PDF, DOCX, XLSX, TXT, MD, CSV"
        raise HTTPException(status_code=400, detail=f"不支持的文件格式，支持: {supported}")

    try:
        content = await _read_upload_with_limit(file)
        _validate_uploaded_content(
            filename=filename,
            content_type=file.content_type,
            content=content,
        )
        try:
            requested_parse_route = normalize_parse_route(
                parse_route,
                default=PARSE_ROUTE_MINERU if is_pdf else PARSE_ROUTE_LOCAL,
                strict=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        embedding_identity = _require_explicit_rag_embedding_identity_or_400(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_api_host=embedding_api_host,
            operation="文档上传索引构建",
        )
        embedding_model = embedding_identity["model"]
        embedding_provider = embedding_identity["provider"]
        embedding_api_key = embedding_identity["api_key"]
        embedding_api_host = embedding_identity["api_host"]

        # 多格式文档处理（非 PDF）
        if is_multi_format and not is_pdf:
            if requested_parse_route == PARSE_ROUTE_MINERU:
                raise HTTPException(status_code=400, detail="MinerU 全程解析当前仅支持 PDF 文件")
            import tempfile
            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                extracted_data = extract_from_file(tmp_path, filename)
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
                previous_document = documents_store.get(doc_id)
                documents_store[doc_id] = {
                    "filename": filename,
                    "upload_time": datetime.now().isoformat(),
                    "data": extracted_data,
                    "pdf_url": None,
                }
                _store_new_document_or_raise(
                    doc_id,
                    documents_store[doc_id],
                    previous=previous_document,
                    message="文档记录写入失败",
                )
                summary_api_key = (api_key or "").strip() or None
                index_status = _call_with_optional_keyword(
                    _queue_document_indexes,
                    "embedding_provider",
                    _normalize_optional_provider_id(embedding_provider),
                    doc_id,
                    embedding_model,
                    embedding_api_key,
                    embedding_api_host,
                    summary_api_key,
                )
                return {
                    "message": "文档上传成功",
                    "doc_id": doc_id,
                    "filename": filename,
                    "total_pages": extracted_data["total_pages"],
                    "total_chars": len(extracted_data["full_text"]),
                    "source_type": extracted_data.get("source_type", "unknown"),
                    "indexing_status": index_status.get("status", "queued"),
                }
            finally:
                os.unlink(tmp_path)

        pdf_file = io.BytesIO(content)

        try:
            if len(PyPDF2.PdfReader(io.BytesIO(content)).pages) > _MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF 页数超过限制（最大 {_MAX_PDF_PAGES} 页）",
                )
        except HTTPException:
            raise
        except Exception:
            # Keep the existing parser's detailed invalid-PDF response path.
            pass

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
        summary_api_key = (api_key or "").strip() or None
        # MinerU is the product default.  Historical "auto" requests keep a
        # graceful local fallback only when MinerU is not configured; once it
        # is configured, auto uses the same single MinerU publication path.
        if (
            requested_parse_route == PARSE_ROUTE_MINERU
            or (requested_parse_route == PARSE_ROUTE_AUTO and _mineru_configured())
        ):
            return _start_mineru_full_route_upload(
                doc_id=doc_id,
                filename=filename,
                pdf_bytes=content,
                requested_route=requested_parse_route,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
                embedding_api_host=embedding_api_host,
                summary_api_key=summary_api_key,
                auto_selected=requested_parse_route == PARSE_ROUTE_AUTO,
                embedding_provider=embedding_provider,
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
            previous_document = documents_store.get(doc_id)
            documents_store[doc_id] = {
                "filename": filename,
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
            try:
                from services.paper_metadata_service import ensure_paper_metadata

                ensure_paper_metadata(documents_store[doc_id])
            except Exception:
                logger.debug("[Upload] paper_metadata extraction skipped", exc_info=True)
            _store_new_document_or_raise(
                doc_id,
                documents_store[doc_id],
                previous=previous_document,
                message="上传文档记录写入失败",
            )

        summary_api_key = (api_key or "").strip() or None
        index_status = _call_with_optional_keyword(
            _queue_document_indexes,
            "embedding_provider",
            _normalize_optional_provider_id(embedding_provider),
            doc_id,
            embedding_model,
            embedding_api_key,
            embedding_api_host,
            summary_api_key,
        )

        response = {
            "message": "PDF上传成功",
            "doc_id": doc_id,
            "filename": filename,
            "total_pages": extracted_data["total_pages"],
            "total_chars": len(extracted_data["full_text"]),
            "image_count": extracted_data.get("image_count", 0),
            "pdf_url": pdf_url,
            "ocr_used": extracted_data.get("ocr_used", False),
            "ocr_backend": extracted_data.get("ocr_backend"),
            "ocr_status": extracted_data.get("ocr_status", "not_started"),
            "ocr_execution_status": extracted_data.get("ocr_execution_status", "not_started"),
            "ocr_adoption_status": extracted_data.get("ocr_adoption_status", "not_started"),
            "ocr_failed_pages": [int(page) for page in (extracted_data.get("ocr_failed_pages") or []) if str(page).isdigit()],
            "ocr_execution_successful_pages": [
                int(page)
                for page in (extracted_data.get("ocr_execution_successful_pages") or [])
                if str(page).isdigit()
            ],
            "ocr_applied_pages": [
                int(page)
                for page in (extracted_data.get("ocr_applied_pages") or [])
                if str(page).isdigit()
            ],
            "ocr_unapplied_pages": [
                int(page)
                for page in (extracted_data.get("ocr_unapplied_pages") or [])
                if str(page).isdigit()
            ],
            "ocr_warning": str(extracted_data.get("ocr_warning") or extracted_data.get("ocr_error") or "").strip(),
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
        embedding_model: 必填的文本嵌入模型
        embedding_provider: 必填的 Embedding 提供商
        embedding_api_key: 远程 Embedding 服务必填的 API 密钥
        embedding_api_host: 远程 Embedding 服务必填的 API 地址
        api_key: 语义意群摘要使用的 LLM API 密钥（可选）
    """
    try:
        body = await request.json()
        url = body.get("url", "").strip()
        embedding_model = body.get("embedding_model")
        embedding_provider = body.get("embedding_provider")
        embedding_api_key = body.get("embedding_api_key")
        embedding_api_host = body.get("embedding_api_host")
        api_key = (body.get("api_key") or "").strip() or None

        if not url:
            raise HTTPException(status_code=400, detail="URL 不能为空")

        embedding_identity = _require_explicit_rag_embedding_identity_or_400(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_api_host=embedding_api_host,
            operation="URL 导入索引构建",
        )
        embedding_model = embedding_identity["model"]
        embedding_provider = embedding_identity["provider"]
        embedding_api_key = embedding_identity["api_key"]
        embedding_api_host = embedding_identity["api_host"]

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
        previous_document = documents_store.get(doc_id)
        documents_store[doc_id] = {
            "filename": f"🌐 {title[:60]}",
            "upload_time": datetime.now().isoformat(),
            "data": extracted_data,
            "pdf_url": None,
        }

        _store_new_document_or_raise(
            doc_id,
            documents_store[doc_id],
            previous=previous_document,
            message="网页导入的文档记录写入失败",
        )

        _call_with_optional_keyword(
            create_index,
            "embedding_provider",
            embedding_provider,
            doc_id,
            content,
            str(VECTOR_STORE_DIR),
            embedding_model,
            embedding_api_key,
            embedding_api_host,
            pages=extracted_data["pages"],
            summary_api_key=api_key,
            index_source="url",
            index_meta={
                "source_hash": extracted_data["parse_manifest"]["source_hash"],
                "document_source_hash": extracted_data["parse_manifest"]["source_hash"],
                "parse_generation": extracted_data["parse_manifest"]["generation"],
                "parser_route": PARSE_ROUTE_LOCAL,
            },
            build_semantic_groups=bool(api_key),
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


def _paper_metadata_hydration_public_status(doc_id: str) -> dict:
    live = _PAPER_METADATA_HYDRATION_STATUS.get(doc_id)
    if isinstance(live, dict):
        return deepcopy(live)
    doc = documents_store.get(doc_id)
    persisted = doc.get("paper_metadata_hydration") if isinstance(doc, dict) else None
    return deepcopy(persisted) if isinstance(persisted, dict) else {
        "status": "not_started",
        "doc_id": doc_id,
    }


async def _run_paper_metadata_hydration(
    doc_id: str,
    *,
    parse_generation: str,
    document_source_hash: str,
    local_metadata: dict,
) -> None:
    _PAPER_METADATA_HYDRATION_STATUS[doc_id] = {
        "status": "running",
        "doc_id": doc_id,
        "parse_generation": parse_generation,
        "document_source_hash": document_source_hash,
    }
    try:
        result = await hydrate_paper_metadata(
            local_metadata,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
            unpaywall_email=settings.paper_metadata_unpaywall_email,
            semantic_scholar_api_key=settings.paper_metadata_semantic_scholar_api_key,
            timeout_seconds=settings.paper_metadata_hydration_timeout_seconds,
            enable_openalex=bool(getattr(settings, "enable_paper_metadata_openalex", False)),
            enable_arxiv=bool(getattr(settings, "enable_paper_metadata_arxiv", False)),
            enable_openreview=bool(getattr(settings, "enable_paper_metadata_openreview", False)),
        )
    except Exception as exc:
        logger.warning("[PaperMetadata] hydration failed open doc=%s: %s", doc_id, exc)
        result = {
            "status": "unavailable",
            "doc_id": doc_id,
            "parse_generation": parse_generation,
            "document_source_hash": document_source_hash,
            "providers": {},
            "retraction": {
                "status": "unknown",
                "evidence": [],
                "notice": "外部元数据暂不可用。",
            },
            "notice": "外部元数据补全失败，本地元数据仍可用。",
        }

    result = {**dict(result or {}), "doc_id": doc_id}
    try:
        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=document_source_hash,
            )
            current = documents_store.get(doc_id)
            if not isinstance(current, dict):
                return
            candidate = deepcopy(current)
            candidate["paper_metadata_hydration"] = result
            hydrated_metadata = result.get("metadata")
            if isinstance(hydrated_metadata, dict) and hydrated_metadata:
                merged_metadata = dict(candidate.get("paper_metadata") or {})
                for field, value in hydrated_metadata.items():
                    if value not in (None, "", [], {}):
                        merged_metadata[field] = value
                merged_metadata["parse_generation"] = parse_generation
                merged_metadata["document_source_hash"] = document_source_hash
                merged_metadata["source"] = "hydrated"
                candidate["paper_metadata"] = merged_metadata
            if not save_document(doc_id, candidate):
                raise RuntimeError("metadata persistence failed")
            documents_store[doc_id] = candidate
            _PAPER_METADATA_HYDRATION_STATUS[doc_id] = deepcopy(result)
    except _SupersededParseGeneration:
        _PAPER_METADATA_HYDRATION_STATUS[doc_id] = {
            "status": "cancelled",
            "reason": "parse_identity_changed",
            "doc_id": doc_id,
        }
    except Exception as exc:
        logger.warning("[PaperMetadata] hydration publication failed doc=%s: %s", doc_id, exc)
        _PAPER_METADATA_HYDRATION_STATUS[doc_id] = {
            "status": "unavailable",
            "reason": "persistence_failed",
            "doc_id": doc_id,
        }
    finally:
        _PAPER_METADATA_HYDRATION_TASKS.pop(doc_id, None)


def _schedule_paper_metadata_hydration(doc_id: str, *, force: bool = False) -> dict:
    doc = documents_store.get(doc_id)
    if not isinstance(doc, dict):
        raise HTTPException(status_code=404, detail="文档未找到")
    manifest = _require_document_parse_ready(doc_id, doc)
    generation = str(manifest.get("generation") or "").strip()
    source_hash = str(manifest.get("source_hash") or "").strip()
    running = _PAPER_METADATA_HYDRATION_TASKS.get(doc_id)
    if running is not None and not running.done():
        return _paper_metadata_hydration_public_status(doc_id)
    persisted = doc.get("paper_metadata_hydration")
    expected_cache_identity = hydration_cache_identity(
        parse_generation=generation,
        document_source_hash=source_hash,
        unpaywall_email=settings.paper_metadata_unpaywall_email,
        semantic_scholar_api_key=settings.paper_metadata_semantic_scholar_api_key,
        enable_openalex=bool(getattr(settings, "enable_paper_metadata_openalex", False)),
        enable_arxiv=bool(getattr(settings, "enable_paper_metadata_arxiv", False)),
        enable_openreview=bool(getattr(settings, "enable_paper_metadata_openreview", False)),
    )
    persisted_cache_identity = persisted.get("cache_identity") if isinstance(persisted, dict) else None
    if (
        not force
        and isinstance(persisted, dict)
        and str(persisted.get("parse_generation") or "") == generation
        and str(persisted.get("document_source_hash") or "") == source_hash
        and isinstance(persisted_cache_identity, dict)
        and str(persisted_cache_identity.get("key") or "") == str(expected_cache_identity.get("key") or "")
        and str(persisted.get("status") or "") in {"completed", "unavailable"}
    ):
        return deepcopy(persisted)
    try:
        from services.paper_metadata_service import ensure_paper_metadata

        local_metadata = ensure_paper_metadata(doc) or {}
    except Exception:
        local_metadata = dict(doc.get("paper_metadata") or {})
    status = {
        "status": "queued",
        "doc_id": doc_id,
        "parse_generation": generation,
        "document_source_hash": source_hash,
    }
    _PAPER_METADATA_HYDRATION_STATUS[doc_id] = status
    task = asyncio.create_task(_run_paper_metadata_hydration(
        doc_id,
        parse_generation=generation,
        document_source_hash=source_hash,
        local_metadata=deepcopy(local_metadata),
    ))
    _PAPER_METADATA_HYDRATION_TASKS[doc_id] = task
    return deepcopy(status)


@router.get("/documents/{doc_id}/paper-metadata/hydration")
async def get_paper_metadata_hydration_status(doc_id: str):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    return _paper_metadata_hydration_public_status(doc_id)


@router.post("/documents/{doc_id}/paper-metadata/hydration")
async def start_paper_metadata_hydration(doc_id: str, force: bool = False):
    return _schedule_paper_metadata_hydration(doc_id, force=force)


@router.get("/documents/recall")
async def recall_uploaded_documents(
    query: str = "",
    exclude_doc_id: str = "",
    limit: int = 8,
):
    """Return explicit cross-document candidates; never auto-select an ambiguous title."""
    normalized_limit = max(1, min(int(limit or 8), 20))
    candidates = list_recallable_documents(
        documents_store,
        query=query,
        exclude_doc_id=exclude_doc_id,
        limit=normalized_limit,
    )
    return {
        "query": str(query or "").strip(),
        "candidates": candidates,
        "ambiguous": len(candidates) > 1,
    }


@router.get("/document/{doc_id}")
async def get_document(doc_id: str, include_content: bool = True):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    parse_manifest = _read_document_parse_manifest(doc_id, doc)
    # Lazily attach academic DocDetails for older uploads. This remains a
    # read endpoint from the caller's perspective, so any persistence must
    # publish a fresh document snapshot under the same lock as parse swaps.
    paper_metadata = {}
    try:
        from services.paper_metadata_service import ensure_paper_metadata

        with _get_document_publication_lock(doc_id):
            current = documents_store.get(doc_id)
            if not isinstance(current, dict):
                raise RuntimeError("文档记录不存在")
            candidate = deepcopy(current)
            before = candidate.get("paper_metadata")
            paper_metadata = ensure_paper_metadata(candidate) or {}
            if paper_metadata and paper_metadata != before:
                # A parser publication can only begin after this lock is
                # released. Persist the copied current record, never a stale
                # object captured before a MinerU/local generation swap.
                if not save_document(doc_id, candidate):
                    logger.warning(
                        "[Document] paper_metadata refresh was not persisted doc=%s", doc_id
                    )
                else:
                    documents_store[doc_id] = candidate
                    current = candidate
            doc = current
            parse_manifest = _read_document_parse_manifest(doc_id, doc)
    except Exception:
        logger.debug("[Document] paper_metadata ensure failed doc=%s", doc_id, exc_info=True)
        paper_metadata = doc.get("paper_metadata") if isinstance(doc.get("paper_metadata"), dict) else {}

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
        "ocr_status": doc["data"].get("ocr_status", "not_started"),
        "ocr_execution_status": doc["data"].get("ocr_execution_status", "not_started"),
        "ocr_adoption_status": doc["data"].get("ocr_adoption_status", "not_started"),
        "ocr_failed_pages": doc["data"].get("ocr_failed_pages", []),
        "ocr_execution_successful_pages": doc["data"].get("ocr_execution_successful_pages", []),
        "ocr_applied_pages": doc["data"].get("ocr_applied_pages", []),
        "ocr_unapplied_pages": doc["data"].get("ocr_unapplied_pages", []),
        "ocr_warning": str(doc["data"].get("ocr_warning") or doc["data"].get("ocr_error") or "").strip(),
        "extraction_quality": doc["data"].get("extraction_quality", "unknown"),
        "extraction_method": doc["data"].get("extraction_method", "unknown"),
        "parse_manifest": parse_manifest,
        "parse_ready": is_parse_prepared(parse_manifest),
        "indexing": _get_document_index_status(doc_id),
        "paper_metadata": paper_metadata or None,
        "paper_metadata_hydration": _paper_metadata_hydration_public_status(doc_id),
    }
    if settings.enable_paper_metadata_hydration:
        try:
            response["paper_metadata_hydration"] = _schedule_paper_metadata_hydration(doc_id)
        except Exception:
            logger.debug(
                "[Document] metadata hydration scheduling skipped doc=%s",
                doc_id,
                exc_info=True,
            )
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
    current_manifest = _read_document_parse_manifest(doc_id, documents_store[doc_id])
    # Keep the public service call compatible with existing integrations. The
    # optional identity-aware call is only needed when a modern task record is
    # stale or identityless.
    status = get_table_visual_verification_status(doc_id, task_id)
    metadata = current_manifest.get("metadata") if isinstance(current_manifest.get("metadata"), dict) else {}
    current_identity = (
        str(current_manifest.get("generation") or "").strip(),
        str(current_manifest.get("source_hash") or "").strip(),
    )
    task_identity = (
        str((status or {}).get("parse_generation") or "").strip(),
        str((status or {}).get("document_source_hash") or "").strip(),
    )
    if (
        status
        and all(current_identity)
        and not metadata.get("legacy_inferred")
        and task_identity != current_identity
    ):
        status = get_table_visual_verification_status(
            doc_id,
            task_id,
            current_parse_manifest=current_manifest,
        )
    if not status:
        raise HTTPException(status_code=404, detail="表格视觉核验任务未找到")
    return status


@router.get("/documents/{doc_id}/visual-assets/{parse_generation}/{attachment_id}")
async def get_document_chat_visual_asset(
    doc_id: str,
    parse_generation: str,
    attachment_id: str,
):
    """Serve one immutable, server-selected visual attached to a chat answer.

    The URL carries the parse generation and resolves a previously materialized
    attachment. It never accepts a PDF path, page number, or bbox from the
    browser, so localStorage or model output cannot turn it into an arbitrary
    file/crop endpoint. Historical generations remain readable after a reparse.
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    try:
        image_path, manifest = load_chat_visual_attachment(
            data_dir=DATA_DIR,
            doc_id=doc_id,
            parse_generation=parse_generation,
            attachment_id=attachment_id,
        )
    except ChatVisualAttachmentError as exc:
        error_code = str(exc)
        status_code = 400 if error_code.startswith("invalid_") else 404
        raise HTTPException(status_code=status_code, detail="文献图表附件不存在或已失效") from exc

    etag = str(manifest.get("image_sha256") or "")
    headers = {
        "Cache-Control": "private, max-age=31536000, immutable",
        "X-Content-Type-Options": "nosniff",
    }
    if etag:
        headers["ETag"] = f'"{etag}"'
    return FileResponse(
        path=image_path,
        media_type="image/jpeg",
        headers=headers,
    )


def recover_pending_rag_transactions() -> list[dict]:
    """Rollback interrupted source switches after the document store has loaded."""
    pending_dir = DATA_DIR / "rag_transactions" / "pending"
    if not pending_dir.exists():
        return []
    recovered: list[dict] = []
    terminal_states = {"committed", "rolled_back", "superseded"}
    for journal_path in pending_dir.glob("*.json"):
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            doc_id = str(journal.get("doc_id") or "")
            source = str(journal.get("source") or "pdf_native")
            if not doc_id or journal.get("state") in terminal_states:
                continue
            backup_manifest = _load_complete_rag_backup_manifest(doc_id, source)
            if not backup_manifest:
                logger.error("[RagIndex] pending transaction lacks a complete backup: %s", journal_path)
                continue
            with _get_document_publication_lock(doc_id):
                current_manifest = _read_document_parse_manifest(
                    doc_id,
                    documents_store.get(doc_id),
                )
                journal_generation = str(journal.get("parse_generation") or "").strip()
                journal_source_hash = str(journal.get("document_source_hash") or "").strip()
                journal_matches_current = (
                    not journal_generation
                    or matches_parse_generation(
                        current_manifest,
                        generation=journal_generation,
                        source_hash=journal_source_hash or None,
                    )
                )
                backup_matches_current = _rag_backup_matches_current_parse(
                    doc_id,
                    backup_manifest,
                    current_manifest,
                )
                if not journal_matches_current or not backup_matches_current:
                    reason = "pending_rag_transaction_parse_generation_mismatch"
                    logger.warning(
                        "[RagIndex] skip stale startup rollback doc=%s journal_generation=%s current_generation=%s",
                        doc_id,
                        journal_generation or "legacy",
                        str(current_manifest.get("generation") or ""),
                    )
                    _write_rag_transaction_journal(
                        doc_id,
                        "superseded",
                        source=source,
                        manifest_path=str(journal.get("manifest_path") or ""),
                        error=reason,
                    )
                    recovered.append({
                        "doc_id": doc_id,
                        "state": "superseded",
                        "reason": reason,
                    })
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
        rag_normalized, quality_failures = _normalize_mineru_for_document(
            doc_id,
            payload,
            doc=doc,
        )
        if quality_failures:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "缓存中的 MinerU 结果未通过质量门，未发布阅读块",
                    "quality_failures": quality_failures,
                    "quality_report": rag_normalized.get("quality_report") or {},
                },
            )
        block_index = build_block_index_from_mineru_payload(
            doc_id=doc_id,
            doc=doc,
            payload=payload,
            pdf_path=pdf_path,
        )
        block_index = _attach_mineru_quality_to_block_index(block_index, rag_normalized)
        with _get_document_publication_lock(doc_id):
            _require_current_parse_generation(
                doc_id,
                parse_generation=str(parse_manifest.get("generation") or ""),
                document_source_hash=str(parse_manifest.get("source_hash") or ""),
            )
            if not save_block_index(DATA_DIR, doc_id, block_index):
                raise RuntimeError("MinerU 阅读块索引写入失败")
            removed = _clear_block_dependent_ai_cache(doc_id, documents_store.get(doc_id))
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新执行 MinerU 阅读块重建")
    except HTTPException:
        raise
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
            _mineru_ready_status(block_index),
            stage=_mineru_ready_status(block_index),
            block_count=block_count,
            outline_count=outline_count,
            figure_count=figure_count,
            cache_removed=removed,
            active_source=MINERU_BLOCK_INDEX_SOURCE,
            active_mineru=True,
            message="已从缓存的 MinerU 结果重建索引",
            parse_generation=str(parse_manifest.get("generation") or ""),
            document_source_hash=str(parse_manifest.get("source_hash") or ""),
            quality_status=(block_index.get("mineru_meta") or {}).get("quality_status", "success"),
            expected_page_count=(block_index.get("mineru_meta") or {}).get("expected_page_count", 0),
            coverage=(block_index.get("mineru_meta") or {}).get("coverage", 0.0),
            failed_pages=(block_index.get("mineru_meta") or {}).get("failed_pages") or [],
            page_ledger=(block_index.get("mineru_meta") or {}).get("page_ledger") or [],
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

    try:
        body = await request.json()
    except Exception:
        body = {}

    doc = documents_store[doc_id]
    parse_manifest = _read_document_parse_manifest(doc_id, doc)
    requested_source = str(body.get("source") or "").strip().lower()
    if (
        requested_source == "local"
        and parse_manifest.get("resolved_route") == PARSE_ROUTE_MINERU
    ):
        raise HTTPException(
            status_code=409,
            detail="当前文档已固定为 MinerU 全程路线，问答索引必须使用 MinerU 结构化结果发布",
        )
    mineru_payload = _load_mineru_result_for_manifest(doc_id, parse_manifest)
    use_local_source = bool(
        requested_source == "local"
        or (
            parse_manifest.get("resolved_route") != PARSE_ROUTE_MINERU
            and not (_is_legacy_parse_manifest(parse_manifest) and mineru_payload)
        )
    )

    if use_local_source:
        block_index = ensure_block_index(
            doc_id=doc_id,
            doc=doc,
            data_dir=DATA_DIR,
            pdf_path=_resolve_document_pdf_path(doc),
        )
        data = doc.get("data") or {}
        rag_source = _rag_source_from_block_index(block_index, data)
        estimate = {
            "page_count": len(rag_source["pages"]),
            "full_text_chars": len(rag_source["full_text"]),
            "estimated_embedding_tokens": _estimate_text_tokens(rag_source["full_text"]),
            "estimated_chunk_count": _estimate_chunk_count(rag_source["full_text"]),
            "structured_table_count": len(data.get("structured_table_bundles") or []),
            "narrative_block_count": rag_source["block_count"],
            "source": "local",
        }
        if body.get("estimate_only"):
            return {
                "status": "estimated",
                "can_rebuild": bool(rag_source["full_text"] and rag_source["pages"]),
                "estimate": estimate,
                "quality_failures": [],
                "rag_index": _get_rag_index_status(doc_id),
            }

        embedding_identity = _require_explicit_rag_embedding_identity_or_400(
            embedding_model=body.get("embedding_model"),
            embedding_provider=body.get("embedding_provider"),
            embedding_api_key=body.get("embedding_api_key"),
            embedding_api_host=body.get("embedding_api_host"),
            operation="本地问答索引重建",
        )
        summary_api_key = (body.get("summary_api_key") or "").strip() or None
        summary_model = str(body.get("summary_model") or body.get("model") or "gpt-4o-mini").strip() or "gpt-4o-mini"
        summary_provider = str(body.get("summary_provider") or body.get("provider") or "openai").strip() or "openai"
        summary_api_host = (body.get("summary_api_host") or body.get("api_host") or "").strip()
        try:
            return await asyncio.to_thread(
                _rebuild_local_rag_index,
                doc_id,
                embedding_model=embedding_identity["model"],
                embedding_api_key=embedding_identity["api_key"],
                embedding_api_host=embedding_identity["api_host"],
                summary_api_key=summary_api_key,
                summary_model=summary_model,
                summary_provider=summary_provider,
                summary_api_host=summary_api_host,
                embedding_provider=embedding_identity["provider"],
            )
        except _SupersededParseGeneration:
            raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新发起本地索引升级")
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("[RagIndex] local rebuild failed for %s: %s", doc_id, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    parse_manifest = _require_mineru_route_compatibility(doc_id, doc)

    payload = mineru_payload or _load_mineru_result_for_manifest(doc_id, parse_manifest)
    if not payload:
        raise HTTPException(status_code=409, detail="当前解析代际没有可用的 MinerU 解析结果，请重新执行深度解析")

    normalized, failures = _normalize_mineru_for_document(doc_id, payload, doc=doc)
    ok = not failures
    estimate = {
        "page_count": normalized.get("expected_page_count") or len(normalized.get("pages") or []),
        "text_page_count": len(normalized.get("pages") or []),
        "full_text_chars": len(normalized.get("full_text") or ""),
        "estimated_embedding_tokens": _estimate_text_tokens(normalized.get("full_text") or ""),
        "estimated_chunk_count": _estimate_chunk_count(normalized.get("full_text") or ""),
        "structured_table_count": len(normalized.get("structured_table_bundles") or []),
        "source_hash": normalized.get("source_hash", ""),
        "normalizer_version": normalized.get("normalizer_version", ""),
        "quality_status": normalized.get("quality_status", "failed"),
        "coverage": normalized.get("coverage", 0.0),
        "failed_pages": normalized.get("failed_pages") or [],
        "page_ledger": normalized.get("page_ledger") or [],
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

    embedding_identity = _require_explicit_rag_embedding_identity_or_400(
        embedding_model=body.get("embedding_model"),
        embedding_provider=body.get("embedding_provider"),
        embedding_api_key=body.get("embedding_api_key"),
        embedding_api_host=body.get("embedding_api_host"),
        operation="MinerU 问答索引重建",
    )
    summary_api_key = (body.get("summary_api_key") or "").strip() or None
    summary_model = str(body.get("summary_model") or body.get("model") or "gpt-4o-mini").strip() or "gpt-4o-mini"
    summary_provider = str(body.get("summary_provider") or body.get("provider") or "openai").strip() or "openai"
    summary_api_host = (body.get("summary_api_host") or body.get("api_host") or "").strip()

    try:
        return await asyncio.to_thread(
            _rebuild_mineru_rag_index,
            doc_id,
            embedding_model=embedding_identity["model"],
            embedding_api_key=embedding_identity["api_key"],
            embedding_api_host=embedding_identity["api_host"],
            summary_api_key=summary_api_key,
            summary_model=summary_model,
            summary_provider=summary_provider,
            summary_api_host=summary_api_host,
            expected_parse_generation=str(parse_manifest.get("generation") or ""),
            expected_document_source_hash=str(parse_manifest.get("source_hash") or ""),
            embedding_provider=embedding_identity["provider"],
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
    try:
        with _get_document_publication_lock(doc_id):
            current = documents_store.get(doc_id)
            if not isinstance(current, dict):
                raise HTTPException(status_code=404, detail="文档未找到")
            parse_manifest = _read_document_parse_manifest(doc_id, current)
            if _is_full_mineru_parse_manifest(parse_manifest):
                raise HTTPException(
                    status_code=409,
                    detail="MinerU 全程解析不支持只回退问答索引；请重新选择本地路线并重新解析整份文档",
                )
            manifest = _load_complete_rag_backup_manifest(doc_id, "pdf_native")
            if not manifest:
                raise HTTPException(status_code=409, detail="没有完整的本地 RAG 回滚快照")
            if not _rag_backup_matches_current_parse(doc_id, manifest, parse_manifest):
                raise HTTPException(
                    status_code=409,
                    detail="RAG 回滚快照不属于当前文档解析代际，已拒绝覆盖当前文档",
                )
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


@router.get("/documents/{doc_id}/blocks/inventory")
async def get_document_block_inventory(
    doc_id: str,
    kind: str,
    cursor: int = 0,
    limit: int = 100,
):
    """Return a deterministic, paginated structural inventory.

    This endpoint is intentionally separate from semantic search.  A client can
    keep following ``next_cursor`` until ``coverage_complete`` without any
    Top-K relevance sampling.
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    doc = documents_store[doc_id]
    _require_document_parse_ready(doc_id, doc)
    try:
        block_index = ensure_block_index(
            doc_id=doc_id,
            doc=doc,
            data_dir=DATA_DIR,
            pdf_path=_resolve_document_pdf_path(doc),
        )
        return enumerate_block_inventory(
            block_index,
            kind,
            cursor=cursor,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("[BlockInventory] failed for %s", doc_id)
        raise HTTPException(status_code=500, detail=f"结构清单生成失败: {exc}")


def _start_downstream_outline_task(
    *,
    purpose: str,
    doc_id: str,
    doc: dict,
    parse_manifest: dict,
    block_index: dict,
    provider: str,
    model: str,
    prompt_version: str,
    metadata: Optional[dict] = None,
) -> dict:
    """Persist an identity-bound task record for a long outline request."""
    revision = str(
        active_block_index_revision(block_index, doc)
        or block_index.get("block_index_revision")
        or block_index.get("block_index_hash")
        or ""
    ).strip()
    return create_downstream_task(
        DATA_DIR,
        purpose=purpose,
        doc_id=doc_id,
        identity=build_downstream_task_identity(
            doc_id=doc_id,
            parse_generation=str(parse_manifest.get("generation") or ""),
            document_source_hash=str(parse_manifest.get("source_hash") or ""),
            block_index_revision=revision,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
        ),
        metadata={"source": "document_route", **dict(metadata or {})},
    )


def _request_bool(value, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _reading_outline_visual_policy(
    body: dict,
    *,
    provider: str,
    model: str,
    api_key: str,
    api_host: str,
):
    visual_provider = str(body.get("visual_provider") or "").strip()
    visual_model = str(body.get("visual_model") or "").strip()
    visual_api_host = str(body.get("visual_api_host") or "").strip()
    visual_api_key = str(body.get("visual_api_key") or "").strip()
    if (visual_provider or visual_model) and not visual_api_key:
        visual_api_key = _configured_provider_api_key_for_target(
            visual_provider or provider,
            visual_api_host,
        )
    local_provider = str(body.get("local_visual_provider") or "").strip()
    local_model = str(body.get("local_visual_model") or "").strip()
    local_api_host = str(body.get("local_visual_api_host") or "").strip()
    local_api_key = str(body.get("local_visual_api_key") or "").strip()
    strategy = str(body.get("visual_strategy") or "balanced").strip().lower()
    if strategy not in {"privacy", "balanced", "quality"}:
        strategy = "balanced"
    return resolve_visual_enrichment_policy(
        strategy=strategy,
        primary_provider=provider,
        primary_model=model,
        primary_api_key=api_key,
        primary_endpoint=_get_overview_provider_endpoint(provider, api_host),
        visual_provider=visual_provider,
        visual_model=visual_model,
        visual_api_key=visual_api_key,
        visual_endpoint=_get_overview_provider_endpoint(visual_provider, visual_api_host)
        if visual_provider
        else "",
        visual_enabled=_request_bool(body.get("visual_enabled"), None),
        local_visual_provider=local_provider,
        local_visual_model=local_model,
        local_visual_api_key=local_api_key,
        local_visual_endpoint=_get_overview_provider_endpoint(local_provider, local_api_host)
        if local_provider
        else "",
    )


async def _prepare_reading_outline_visuals(
    *,
    doc_id: str,
    doc: dict,
    block_index: dict,
    body: dict,
    provider: str,
    model: str,
    api_key: str,
    api_host: str,
) -> dict:
    """Run and publish bounded summary visual enrichment, failing open."""
    outcome: dict = {
        "enabled": bool(settings.enable_reading_outline_visual_preflight),
        "diagnostics": {},
        "publication": {"published": False, "reason": "disabled"},
    }
    if not settings.enable_reading_outline_visual_preflight:
        return outcome

    try:
        visual_policy = _reading_outline_visual_policy(
            body,
            provider=provider,
            model=model,
            api_key=api_key,
            api_host=api_host,
        )
        preflight_result = await preflight_summary_visuals(
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            pdf_path=_resolve_document_pdf_path(doc),
            visual_policy=visual_policy,
            max_figures=settings.reading_outline_visual_preflight_limit,
        )
        outcome["diagnostics"] = dict(preflight_result.get("diagnostics") or {})
        items = [
            item
            for item in (preflight_result.get("items") or [])
            if isinstance(item, dict)
        ]
        if items:
            outcome["publication"] = publish_visual_supplements(
                doc_id,
                parse_generation=str(preflight_result.get("parse_generation") or ""),
                document_source_hash=str(
                    preflight_result.get("document_source_hash") or ""
                ),
                visual_model_identity=str(
                    preflight_result.get("visual_model_identity") or visual_policy.identity
                ),
                items=items,
            )
        else:
            outcome["publication"] = {
                "published": False,
                "reason": str(
                    outcome["diagnostics"].get("skipped_reason") or "no_items"
                ),
            }
    except _SupersededParseGeneration:
        raise
    except Exception as exc:
        logger.warning(
            "[ReadingOutline] visual preflight failed open doc=%s: %s",
            doc_id,
            exc,
        )
        outcome["diagnostics"] = {
            "failed_open": True,
            "error": str(exc)[:240] or type(exc).__name__,
        }
        outcome["publication"] = {
            "published": False,
            "reason": "preflight_failed_open",
        }
    return outcome


def _finish_downstream_outline_task(
    task: dict,
    *,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    """Publish the final outline state without letting persistence hide a result."""
    if not task:
        return
    status = "failed"
    if result is not None:
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        source = str(result.get("source") or "").strip().lower()
        generation_status = str(meta.get("generation_status") or "").strip().lower()
        if generation_status == "partial" or source == "ai_partial":
            status = "partial"
        elif generation_status in {"fallback", "degraded"} or meta.get("generation_error"):
            status = "degraded"
        else:
            status = "succeeded"
    shortfall: dict[str, object] = {}
    result_meta = result.get("meta") if isinstance(result, dict) and isinstance(result.get("meta"), dict) else {}
    claim_verifier = result_meta.get("claim_verifier") if isinstance(result_meta.get("claim_verifier"), dict) else {}
    claim_shortfalls = claim_verifier.get("shortfalls") if isinstance(claim_verifier.get("shortfalls"), list) else []
    partial_issues = result_meta.get("partial_quality_issues") if isinstance(result_meta.get("partial_quality_issues"), list) else []
    if claim_shortfalls:
        shortfall = {
            "kind": "claim_verifier",
            "code": "claim_support_shortfall",
            "stage": "downstream_ai",
            "count": len(claim_shortfalls),
            "unresolved_count": int(claim_verifier.get("unresolved_count") or len(claim_shortfalls)),
            "retryable": True,
        }
    elif partial_issues:
        shortfall = {
            "kind": "summary_quality",
            "code": "partial_quality",
            "stage": "downstream_ai",
            "count": len(partial_issues),
            "reasons": partial_issues,
            "retryable": True,
        }
    elif error:
        shortfall = {
            "kind": "downstream_ai",
            "code": classify_error_code(error) or "generation_failed",
            "stage": "downstream_ai",
            "retryable": True,
        }
    elif status in {"partial", "degraded"}:
        shortfall = {
            "kind": "downstream_ai",
            "code": "degraded_result",
            "stage": "downstream_ai",
            "retryable": True,
        }
    try:
        transition_downstream_task(
            DATA_DIR,
            purpose=str(task.get("purpose") or ""),
            doc_id=str(task.get("doc_id") or ""),
            task_id=str(task.get("task_id") or ""),
            status=status,
            stage="completed" if result is not None else "failed",
            error=error,
            retryable=status not in {"succeeded", "cancelled"},
            result=result,
            shortfall=shortfall,
        )
    except Exception as exc:
        logger.warning("[DownstreamTask] failed to finish %s: %s", task.get("task_id"), exc)


@router.get("/documents/{doc_id}/ai-tasks/{purpose}")
async def get_document_downstream_task_status(doc_id: str, purpose: str):
    """Expose restart-safe public status for overview and outline generation."""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    normalized_purpose = str(purpose or "").strip().lower()
    if normalized_purpose not in {"overview", "reading_outline", "section_outline"}:
        raise HTTPException(status_code=404, detail="未知下游任务类型")
    record = get_downstream_task(DATA_DIR, purpose=normalized_purpose, doc_id=doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="任务未找到")

    doc = documents_store[doc_id]
    manifest = read_parse_manifest(doc, doc_id=doc_id)
    try:
        current_index = load_block_index(DATA_DIR, doc_id)
    except Exception:
        current_index = {}
    identity = record.get("identity") if isinstance(record.get("identity"), dict) else {}
    current_identity = build_downstream_task_identity(
        doc_id=doc_id,
        parse_generation=str(manifest.get("generation") or ""),
        document_source_hash=str(manifest.get("source_hash") or ""),
        block_index_revision=str(
            active_block_index_revision(current_index, doc)
            or current_index.get("block_index_revision")
            or current_index.get("block_index_hash")
            or ""
        ),
        provider=str(identity.get("provider") or ""),
        model=str(identity.get("model") or ""),
        prompt_version=str(identity.get("prompt_version") or ""),
    )
    if not downstream_task_identity_matches(record, current_identity):
        record = transition_downstream_task(
            DATA_DIR,
            purpose=normalized_purpose,
            doc_id=doc_id,
            task_id=str(record.get("task_id") or ""),
            status="cancelled",
            stage="identity_changed",
            error="文档解析或块索引已更新，请重新生成",
            retryable=True,
        ) or record

    # The state service deliberately strips credentials and endpoints. Return
    # only this durable public envelope rather than an internal task object.
    ledger = get_downstream_task_events(
        DATA_DIR,
        task_id=str(record.get("task_id") or ""),
    )
    if ledger:
        record["events"] = ledger.get("events") or []
        if ledger.get("shortfall"):
            record["shortfall"] = ledger["shortfall"]
    return record


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
        api_key = _configured_provider_api_key_for_target(provider, api_host)

    doc = documents_store[doc_id]
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
        force_rebuild=force,
    )
    try:
        visual_preflight = await _prepare_reading_outline_visuals(
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            body=body,
            provider=provider,
            model=model,
            api_key=api_key,
            api_host=api_host,
        )
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新生成阅读总结")

    # Visual supplements rotate AI caches and publish a new block-index
    # revision. Reload both shared objects before binding the summary task so
    # it can never consume or publish against the pre-enrichment revision.
    doc = documents_store[doc_id]
    parse_manifest = _require_current_parse_generation(
        doc_id,
        parse_generation=str(parse_manifest.get("generation") or ""),
        document_source_hash=str(parse_manifest.get("source_hash") or ""),
    )
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
    )
    task = _start_downstream_outline_task(
        purpose="reading_outline",
        doc_id=doc_id,
        doc=doc,
        parse_manifest=parse_manifest,
        block_index=block_index,
        provider=provider,
        model=model,
        prompt_version=READING_OUTLINE_PROMPT_VERSION,
        metadata={"visual_preflight": visual_preflight},
    )
    transition_downstream_task(
        DATA_DIR,
        purpose="reading_outline",
        doc_id=doc_id,
        task_id=str(task.get("task_id") or ""),
        status="running",
        stage="generating",
    )

    try:
        result = await get_or_create_reading_outline(
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
        transition_downstream_task(
            DATA_DIR,
            purpose="reading_outline",
            doc_id=doc_id,
            task_id=str(task.get("task_id") or ""),
            status="verifying",
            stage="parse_identity",
        )
        _finish_downstream_outline_task(task, result=result)
        return result
    except _SupersededAICacheGeneration:
        _finish_downstream_outline_task(task, error="AI 缓存已清理，请重新生成阅读总结")
        raise HTTPException(status_code=409, detail="AI 缓存已清理，请重新生成阅读总结")
    except _SupersededParseGeneration:
        _finish_downstream_outline_task(task, error="文档解析路线已更新，请重新生成阅读总结")
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新生成阅读总结")
    except Exception as exc:
        _finish_downstream_outline_task(task, error=str(exc))
        raise


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
        api_key = _configured_provider_api_key_for_target(provider, api_host)

    doc = documents_store[doc_id]
    block_index = ensure_block_index(
        doc_id=doc_id,
        doc=doc,
        data_dir=DATA_DIR,
        pdf_path=_resolve_document_pdf_path(doc),
        force_rebuild=force,
    )
    task = _start_downstream_outline_task(
        purpose="section_outline",
        doc_id=doc_id,
        doc=doc,
        parse_manifest=parse_manifest,
        block_index=block_index,
        provider=provider,
        model=model,
        prompt_version=SECTION_OUTLINE_PROMPT_VERSION,
    )
    transition_downstream_task(
        DATA_DIR,
        purpose="section_outline",
        doc_id=doc_id,
        task_id=str(task.get("task_id") or ""),
        status="running",
        stage="generating",
    )

    try:
        result = await get_or_create_section_outline(
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
        transition_downstream_task(
            DATA_DIR,
            purpose="section_outline",
            doc_id=doc_id,
            task_id=str(task.get("task_id") or ""),
            status="verifying",
            stage="parse_identity",
        )
        _finish_downstream_outline_task(task, result=result)
        return result
    except _SupersededAICacheGeneration:
        _finish_downstream_outline_task(task, error="AI 缓存已清理，请重新生成章节大纲")
        raise HTTPException(status_code=409, detail="AI 缓存已清理，请重新生成章节大纲")
    except _SupersededParseGeneration:
        _finish_downstream_outline_task(task, error="文档解析路线已更新，请重新生成章节大纲")
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新生成章节大纲")
    except Exception as exc:
        _finish_downstream_outline_task(task, error=str(exc))
        raise


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


def _graphrag_parse_identity(manifest: dict, *, block_index_hash: str = "") -> dict:
    """Return the parser generation a GraphRAG artifact is allowed to serve."""
    return {
        "parse_generation": str(manifest.get("generation") or ""),
        "document_source_hash": str(manifest.get("source_hash") or ""),
        "block_index_hash": str(block_index_hash or "").strip(),
    }


def _block_index_content_hash(block_index: dict | None) -> str:
    if not isinstance(block_index, dict):
        return ""
    return str(
        block_index.get("block_index_hash")
        or block_index.get("block_index_revision")
        or ""
    ).strip()


def _graphrag_active_block_index_hash(doc_id: str) -> str | None:
    """Return the current parse route's published block snapshot revision."""
    try:
        index = load_block_index(DATA_DIR, doc_id)
    except Exception:
        return None
    return active_block_index_revision(index, documents_store.get(doc_id))


def _is_loopback_service_url(url: str) -> bool:
    try:
        host = (urlsplit(str(url or "").strip()).hostname or "").strip().lower()
    except ValueError:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _graphrag_build_matches_active_parse(
    doc_id: str,
    *,
    parse_generation: str,
    document_source_hash: str,
    block_index_hash: str = "",
) -> bool:
    if not matches_parse_generation(
        _read_document_parse_manifest(doc_id, documents_store.get(doc_id)),
        generation=parse_generation,
        source_hash=document_source_hash,
    ):
        return False
    expected_block_index_hash = str(block_index_hash or "").strip()
    return not expected_block_index_hash or _graphrag_active_block_index_hash(doc_id) == expected_block_index_hash


def _bind_graphrag_progress_identity(
    progress,
    *,
    parse_generation: str,
    document_source_hash: str,
    block_index_hash: str = "",
):
    """将内存构建进度绑定到发起构建时的主解析代际。"""
    progress.parse_generation = str(parse_generation or "")
    progress.document_source_hash = str(document_source_hash or "")
    progress.block_index_hash = str(block_index_hash or "")
    return progress


def _graphrag_progress_matches_parse(
    progress,
    manifest: dict,
    *,
    block_index_hash: str | None = "",
) -> bool:
    if block_index_hash is None:
        return False
    return bool(
        progress
        and str(getattr(progress, "parse_generation", "") or "")
        == str(manifest.get("generation") or "")
        and str(getattr(progress, "document_source_hash", "") or "")
        == str(manifest.get("source_hash") or "")
        and (
            not str(block_index_hash or "").strip()
            or str(getattr(progress, "block_index_hash", "") or "")
            == str(block_index_hash or "").strip()
        )
    )


def _graphrag_identity_path(working_dir: str | Path) -> Path:
    return Path(working_dir) / _GRAPHRAG_PARSE_IDENTITY_FILE


def _graphrag_index_matches_parse(
    working_dir: str | Path,
    manifest: dict,
    *,
    block_index_hash: str | None = "",
) -> bool:
    if block_index_hash is None:
        return False
    expected = _graphrag_parse_identity(manifest, block_index_hash=block_index_hash)
    if not expected["parse_generation"] or not expected["document_source_hash"]:
        return False
    try:
        stored = json.loads(_graphrag_identity_path(working_dir).read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        str(stored.get("parse_generation") or "") == expected["parse_generation"]
        and str(stored.get("document_source_hash") or "") == expected["document_source_hash"]
        and (
            not expected["block_index_hash"]
            or str(stored.get("block_index_hash") or "") == expected["block_index_hash"]
        )
    )


def _write_graphrag_parse_identity(
    working_dir: str | Path,
    manifest: dict,
    *,
    block_index_hash: str = "",
) -> None:
    path = _graphrag_identity_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = _graphrag_parse_identity(manifest, block_index_hash=block_index_hash)
    if not identity["parse_generation"] or not identity["document_source_hash"]:
        raise RuntimeError("GraphRAG 缺少文档解析代际，无法发布索引")
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(identity, ensure_ascii=False), encoding="utf-8")
    os.replace(str(temp_path), str(path))


def _new_graphrag_staging_dir(doc_id: str, parse_generation: str) -> str:
    # Staging names are never user-visible. Hash both identifiers instead of
    # interpolating them into a filesystem path; active GraphRAG directories
    # remain guarded separately by _safe_graphrag_working_dir.
    doc_namespace = hashlib.sha256(str(doc_id or "").encode("utf-8")).hexdigest()
    generation_namespace = hashlib.sha256(
        str(parse_generation or "").encode("utf-8")
    ).hexdigest()[:24]
    root = Path(settings.graphrag_working_dir).resolve() / "_staging" / doc_namespace
    candidate = (root / f"{generation_namespace}.{uuid.uuid4().hex}").resolve()
    if root not in candidate.parents:
        raise HTTPException(status_code=400, detail="无效的 GraphRAG 暂存路径")
    return str(candidate)


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


def _graphrag_table_evidence_documents(
    structured_table_bundles: list | None,
    *,
    parse_generation: str,
    document_source_hash: str,
    parser_route: str,
) -> list[dict]:
    """Turn structured tables into compact GraphRAG evidence documents.

    The vector path already indexes table bundles and row evidence separately.
    GraphRAG previously skipped that channel entirely because raw table blocks
    are deliberately excluded from generic block evidence.  Keep the input
    compact and evidence-bound instead of injecting an unbounded markdown dump.
    """
    documents: list[dict] = []
    for index, bundle in enumerate(structured_table_bundles or []):
        if not isinstance(bundle, dict):
            continue
        table_id = str(bundle.get("table_id") or bundle.get("id") or f"table-{index + 1}").strip()
        bundle_id = str(bundle.get("table_bundle_id") or table_id or f"bundle-{index + 1}").strip()
        if not bundle_id:
            continue
        raw_pages = bundle.get("pages") or bundle.get("page_numbers") or []
        if not isinstance(raw_pages, list):
            raw_pages = [raw_pages]
        pages: list[int] = []
        for value in raw_pages:
            try:
                page = int(value)
            except (TypeError, ValueError):
                continue
            if page > 0 and page not in pages:
                pages.append(page)
        if not pages:
            try:
                page = int(bundle.get("page") or bundle.get("page_num") or 0)
            except (TypeError, ValueError):
                page = 0
            if page > 0:
                pages.append(page)

        caption = str(bundle.get("caption") or bundle.get("table_caption") or "").strip()
        header = str(bundle.get("header") or bundle.get("table_header") or "").strip()
        parts = [
            f"[结构化表格 {table_id or bundle_id}]",
            f"标题: {caption}" if caption else "",
            f"表头: {header}" if header else "",
        ]
        evidence_unit_ids: list[str] = []
        for unit in (bundle.get("evidence_units") or [])[:16]:
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("evidence_unit_id") or unit.get("id") or "").strip()
            if unit_id:
                evidence_unit_ids.append(unit_id)
            row_text = str(
                unit.get("content")
                or unit.get("row_text")
                or unit.get("raw_row_text")
                or unit.get("text")
                or ""
            ).strip()
            if row_text:
                parts.append(row_text[:900])
        if len(parts) <= 3:
            fallback_body = str(
                bundle.get("table_markdown")
                or bundle.get("markdown")
                or bundle.get("text")
                or bundle.get("content")
                or ""
            ).strip()
            if fallback_body:
                parts.append(fallback_body[:3600])
        content = "\n".join(part for part in parts if part).strip()
        if len(content) < 20:
            continue
        metadata = {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "evidence_source": "structured_table_bundle",
            "structured_table_bundle": True,
            "chunk_type": "table",
            "block_type": "table",
            "table_id": table_id,
            "table_bundle_id": bundle_id,
            "table_caption": caption,
            "table_header": header,
            "evidence_unit_ids": evidence_unit_ids,
            "pages": pages,
            "page_range": [min(pages), max(pages)] if pages else [],
            "parse_generation": str(parse_generation or ""),
            "document_source_hash": str(document_source_hash or ""),
            "parser_route": str(parser_route or ""),
            "source": "structured_table_bundle",
        }
        documents.append({
            "content": content,
            "source_id": f"table:{bundle_id}",
            "metadata": metadata,
        })
    return documents


def _graphrag_input_from_block_index(
    block_index: dict | None,
    *,
    parse_generation: str,
    document_source_hash: str,
    fallback_full_text: str,
    structured_table_bundles: list | None = None,
) -> tuple[str | list[dict], str]:
    """Build GraphRAG input without flattening the selected parse route.

    The normal vector path already treats ``block_index`` as the canonical
    evidence source.  GraphRAG must receive the same blocks, otherwise its
    independent second split loses the page/section/block anchors that MinerU
    produced.  A legacy document without retrievable blocks intentionally
    keeps the established full-text behavior.
    """
    rag_source = build_rag_source_from_block_index(
        block_index,
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
    )
    evidence_documents: list[dict] = []
    for item in rag_source.get("evidence_chunks") or []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("text") or item.get("content") or "").strip()
        metadata = item.get("metadata")
        if not content or not isinstance(metadata, dict):
            continue
        source_id = str(
            metadata.get("evidence_id") or metadata.get("block_id") or ""
        ).strip()
        if not source_id:
            continue
        evidence_documents.append({
            "content": content,
            "source_id": source_id,
            "metadata": metadata,
        })
    table_documents = _graphrag_table_evidence_documents(
        structured_table_bundles,
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
        parser_route=str((block_index or {}).get("parser_route") or ""),
    )
    evidence_documents.extend(table_documents)
    evidence_chars = sum(len(item["content"]) for item in evidence_documents)
    if evidence_documents and evidence_chars >= 50:
        return (
            evidence_documents,
            "block_index_evidence_with_tables" if table_documents else "block_index_evidence",
        )
    return str(fallback_full_text or "").strip(), "document_full_text"


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
        embedding_provider = (body.get("embedding_provider") or "").strip() or None
        embedding_api_key = str(body.get("embedding_api_key", "") or "").strip()
        embedding_api_host = str(body.get("embedding_api_host", "") or "").strip()
        force_rebuild = body.get("force_rebuild", False)

        provider_lower = (provider or "").lower()
        if not model:
            raise HTTPException(status_code=400, detail="GraphRAG 构建需要 model")
        if not provider_lower:
            raise HTTPException(status_code=400, detail="GraphRAG 构建需要 api_provider")
        if provider_lower == "local":
            raise HTTPException(
                status_code=400,
                detail="GraphRAG 不支持未绑定端点的 local 对话 Provider，请使用 Ollama 或显式远程 Provider",
            )
        if not api_key and provider_lower != "ollama":
            raise HTTPException(status_code=400, detail="GraphRAG 构建需要 api_key")
        if not str(api_host or "").strip():
            raise HTTPException(status_code=400, detail="GraphRAG 构建需要显式 api_host")

        from services.graphrag import GraphRAG, GraphRAGConfig, BuildProgress

        # 解析 endpoint
        host = api_host.strip().rstrip('/')
        endpoint = f"{host}/chat/completions" if not host.endswith('/chat/completions') else host
        try:
            _url_origin(endpoint)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="GraphRAG api_host 格式无效") from exc
        if provider_lower == "ollama" and not _is_loopback_service_url(endpoint):
            raise HTTPException(
                status_code=400,
                detail="Ollama GraphRAG 仅允许使用本机回环地址",
            )

        working_dir = str(_safe_graphrag_working_dir(doc_id))
        resolved_embedding_identity = _resolve_graphrag_embedding_identity_or_400(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_api_host=embedding_api_host,
        )
        resolved_embedding_model = str(resolved_embedding_identity.get("model") or "").strip()
        resolved_embedding_provider = str(resolved_embedding_identity.get("provider") or "").strip()
        resolved_embedding_endpoint = str(resolved_embedding_identity.get("api_host") or "").strip()
        resolved_embedding_dim = int(resolved_embedding_identity["dimension"])
        resolved_embedding_api_key = str(embedding_api_key or "").strip()
        if resolved_embedding_provider == "local":
            resolved_embedding_api_key = ""
        elif resolved_embedding_provider != "ollama" and not resolved_embedding_api_key:
            raise HTTPException(
                status_code=400,
                detail="远程 GraphRAG Embedding 需要显式 embedding_api_key",
            )

        config = GraphRAGConfig(
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            embedding_api_key=resolved_embedding_api_key,
            embedding_model=resolved_embedding_model,
            embedding_provider=resolved_embedding_provider,
            embedding_endpoint=resolved_embedding_endpoint,
            embedding_dim=resolved_embedding_dim,
        )

        block_index = ensure_block_index(
            doc_id=doc_id,
            doc=doc,
            data_dir=DATA_DIR,
        )
        block_index_hash = _block_index_content_hash(block_index)

        # A GraphRAG cache is valid only for the document parse generation that
        # created it. Same-PDF re-uploads intentionally retain ``doc_id``.
        # Without this identity check a graph from a former local/MinerU route
        # could be loaded merely because its model configuration still matches.
        has_persisted_index = GraphRAG.has_persisted_index(working_dir)
        if (
            not force_rebuild
            and has_persisted_index
            and _graphrag_index_matches_parse(
                working_dir,
                parse_manifest,
                block_index_hash=block_index_hash,
            )
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
                    strict_config_hash=True,
                )
                if rag is not None:
                    with _get_document_publication_lock(doc_id):
                        _require_current_parse_generation(
                            doc_id,
                            parse_generation=parse_generation,
                            document_source_hash=parse_source_hash,
                        )
                        if not _graphrag_index_matches_parse(
                            working_dir,
                            parse_manifest,
                            block_index_hash=block_index_hash,
                        ):
                            raise _SupersededParseGeneration("GraphRAG 索引不属于当前解析代际")
                        loaded_progress = _bind_graphrag_progress_identity(
                            rag.get_build_progress(),
                            parse_generation=parse_generation,
                            document_source_hash=parse_source_hash,
                            block_index_hash=block_index_hash,
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
            block_index_hash=block_index_hash,
        )
        _GRAPHRAG_BUILD_PROGRESS[doc_id] = build_progress

        graphrag_input, graphrag_input_source = _graphrag_input_from_block_index(
            block_index,
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
            fallback_full_text=full_text,
            structured_table_bundles=(doc.get("data") or {}).get("structured_table_bundles") or [],
        )
        await rag.ainsert(graphrag_input)

        if not _graphrag_build_matches_active_parse(
            doc_id,
            parse_generation=parse_generation,
            document_source_hash=parse_source_hash,
            block_index_hash=block_index_hash,
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
            if block_index_hash:
                _write_graphrag_parse_identity(
                    staging_dir,
                    parse_manifest,
                    block_index_hash=block_index_hash,
                )
            else:
                # Compatibility for old tests and legacy indexes that predate
                # the stable block snapshot revision.
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
            strict_config_hash=True,
        )
        if published_rag is None:
            if not _graphrag_build_matches_active_parse(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
                block_index_hash=block_index_hash,
            ):
                raise HTTPException(status_code=409, detail="文档解析路线已更新，已丢弃旧 GraphRAG 构建")
            raise RuntimeError("GraphRAG 发布后无法重新加载索引")

        with _get_document_publication_lock(doc_id):
            current_manifest = _require_current_parse_generation(
                doc_id,
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
            )
            current_block_index_hash = _graphrag_active_block_index_hash(doc_id)
            if block_index_hash and current_block_index_hash != block_index_hash:
                raise _SupersededParseGeneration("GraphRAG 构建期间阅读结构已更新")
            if not _graphrag_index_matches_parse(
                working_dir,
                current_manifest,
                block_index_hash=block_index_hash,
            ):
                raise _SupersededParseGeneration("GraphRAG 发布结果不属于当前解析代际")
            stats = published_rag.stats()
            published_progress = _bind_graphrag_progress_identity(
                published_rag.get_build_progress(),
                parse_generation=parse_generation,
                document_source_hash=parse_source_hash,
                block_index_hash=block_index_hash,
            )
            # 缓存实例以便查询时复用（存到模块级 registry，跨 router 共享）
            _GRAPHRAG_INSTANCES[doc_id] = published_rag
            _GRAPHRAG_BUILD_PROGRESS[doc_id] = published_progress

        return {
            "message": "GraphRAG 索引构建完成",
            "doc_id": doc_id,
            "stats": stats,
            "loaded_from_disk": False,
            "content_source": graphrag_input_source,
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
    block_index_hash = _graphrag_active_block_index_hash(doc_id)

    from services.graphrag import INSTANCES as _GRAPHRAG_INSTANCES, BUILD_PROGRESS as _GRAPHRAG_BUILD_PROGRESS

    # 1. 内存中已有实例
    if doc_id in _GRAPHRAG_INSTANCES:
        rag = _GRAPHRAG_INSTANCES[doc_id]
        working_dir = str(_safe_graphrag_working_dir(doc_id))
        if _graphrag_index_matches_parse(
            working_dir,
            parse_manifest,
            block_index_hash=block_index_hash,
        ):
            return {"doc_id": doc_id, "stats": rag.stats()}
        _GRAPHRAG_INSTANCES.pop(doc_id, None)
        _GRAPHRAG_BUILD_PROGRESS.pop(doc_id, None)

    # 2. 尝试从磁盘加载元数据（不需要 api_key 也能读取统计）
    working_dir = str(_safe_graphrag_working_dir(doc_id))
    from services.graphrag import GraphRAG
    disk_meta = GraphRAG.load_metadata(working_dir)
    if disk_meta is not None and _graphrag_index_matches_parse(
        working_dir,
        parse_manifest,
        block_index_hash=block_index_hash,
    ):
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
    block_index_hash = _graphrag_active_block_index_hash(doc_id)

    from services.graphrag import BUILD_PROGRESS as _GRAPHRAG_BUILD_PROGRESS

    # 1. 内存中的实时进度
    if doc_id in _GRAPHRAG_BUILD_PROGRESS:
        prog = _GRAPHRAG_BUILD_PROGRESS[doc_id]
        if _graphrag_progress_matches_parse(
            prog,
            parse_manifest,
            block_index_hash=block_index_hash,
        ):
            return {"doc_id": doc_id, "progress": prog.to_dict()}
        _GRAPHRAG_BUILD_PROGRESS.pop(doc_id, None)

    # 2. 磁盘元数据
    working_dir = str(_safe_graphrag_working_dir(doc_id))
    from services.graphrag import GraphRAG
    disk_meta = GraphRAG.load_metadata(working_dir)
    if disk_meta is not None and _graphrag_index_matches_parse(
        working_dir,
        parse_manifest,
        block_index_hash=block_index_hash,
    ):
        return {"doc_id": doc_id, "progress": disk_meta.to_dict()}

    raise HTTPException(status_code=404, detail="该文档未构建 GraphRAG 索引")


@router.delete("/document/{doc_id}/graphrag")
async def delete_graphrag_index(doc_id: str):
    """删除文档的 GraphRAG 索引（内存 + 磁盘）"""
    import shutil
    from services.graphrag import (
        BUILD_PROGRESS as _GRAPHRAG_BUILD_PROGRESS,
        INSTANCES as _GRAPHRAG_INSTANCES,
        get_build_lock,
    )

    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    build_lock = get_build_lock(doc_id)
    if not build_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="GraphRAG 正在构建，无法在发布过程中删除；请稍后重试")

    try:
        # 从内存移除
        _GRAPHRAG_INSTANCES.pop(doc_id, None)
        _GRAPHRAG_BUILD_PROGRESS.pop(doc_id, None)

        # 删除磁盘数据
        working_dir = _safe_graphrag_working_dir(doc_id)
        if working_dir.exists():
            shutil.rmtree(working_dir, ignore_errors=True)
            return {"message": "GraphRAG 索引已删除", "doc_id": doc_id}

        raise HTTPException(status_code=404, detail="该文档未构建 GraphRAG 索引")
    finally:
        build_lock.release()


@router.get("/api/ocr/status")
async def get_ocr_status():
    """
    检查 OCR 可用性、后端状态和当前配置

    返回包含 OCR 后端可用性、Poppler 状态、当前配置和安装指引的完整状态信息。
    """
    status = is_ocr_available()

    # 使用 OCRRegistry 获取后端可用性（is_ocr_available 已刷新本地适配器）
    available_backends = _ocr_registry.list_available()
    available_document_parsers = _document_parser_registry.list_available()
    backends = {
        "tesseract": available_backends.get("tesseract", False),
        "paddleocr": available_backends.get("paddleocr", False),
        "mineru": available_document_parsers.get("mineru", False),  # 文档级深度解析
    }

    # 检测 Poppler 可用性
    poppler_path = _find_poppler()
    poppler_available = poppler_path is not None
    diagnostics = status.get("diagnostics") or diagnose_local_ocr()
    local_available = bool(status.get("local_available"))
    unavailable_reasons = list(status.get("unavailable_reasons") or [])

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
        if provider == "mineru":
            # MinerU 支持 Worker 代理和直连 API。
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
        "local_available": local_available,
        "backends": backends,
        "page_ocr_backends": ["paddleocr", "tesseract"],
        "deprecated_page_ocr_backends": ["mineru", "mistral", "doc2x"],
        "provider_sunset": {
            "mistral": {"deprecated": True, "replacement": "local_auto", "usage": get_ocr_provider_usage("mistral")},
            "doc2x": {"deprecated": True, "replacement": "local_auto", "usage": get_ocr_provider_usage("doc2x")},
        },
        "local_page_supplements": {
            "paddleocr": {
                "role": "page_quality_supplement",
                "deprecated": False,
                "usage": get_ocr_provider_usage("paddleocr"),
            },
        },
        "poppler_available": poppler_available,
        "recommended": recommended,
        "config": config,
        "online_services": online_services,
        "install_instructions": install_instructions,
        "diagnostics": diagnostics,
        "unavailable_reasons": unavailable_reasons,
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
_SUPPORTED_ONLINE_OCR_PROVIDERS = {"mineru"}


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


@router.get("/api/runtime/addons/local-parser/status")
async def get_local_parser_addon_runtime_status():
    """Return the on-demand local parser state without installing anything."""
    return get_local_parser_addon_status()


@router.post("/api/runtime/addons/local-parser/install")
async def install_local_parser_addon():
    """Queue the fixed local-parser install profile after an explicit user action."""
    try:
        return start_local_parser_addon_install()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/ocr/online-config")
async def save_online_ocr_config(request: Request):
    """
    保存在线 OCR 服务配置

    当前只接受 MinerU（文档级解析）的在线配置。Mistral/Doc2X 的旧配置
    仍由兼容读取逻辑保留，但不再作为运行时在线入口。
    持久化到本地配置文件，并重新注册 MinerU 文档解析适配器。

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
    if provider == "mineru":
        # Worker 代理模式配置
        existing_config = _load_online_ocr_config(provider)
        access_mode = body.get("access_mode", "worker").strip()
        if access_mode not in ("worker", "direct"):
            raise HTTPException(status_code=400, detail="access_mode 必须为 'worker' 或 'direct'")
        worker_url = body.get("worker_url", "").strip()
        auth_key = body.get("auth_key", "").strip()
        token_mode = body.get("token_mode", "frontend").strip()
        token = body.get("token", "").strip()
        existing_access_mode = str(existing_config.get("access_mode") or "worker").strip()
        existing_worker_url = str(existing_config.get("worker_url") or "").strip()
        existing_base_url = str(existing_config.get("base_url") or "https://mineru.net/api/v4").strip()

        # 校验 token_mode 参数
        if token_mode not in ("frontend", "worker"):
            raise HTTPException(status_code=400, detail="token_mode 必须为 'frontend' 或 'worker'")

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

        if access_mode == "worker":
            # A Worker receives both Auth Key and frontend MinerU token.  Never
            # carry either saved secret over to a different Worker origin.
            auth_key = _credential_for_service_origin(
                supplied=auth_key,
                saved=str(existing_config.get("auth_key") or "").strip(),
                target_url=worker_url,
                saved_url=existing_worker_url if existing_access_mode == "worker" else "",
                credential_name="Worker Auth Key",
            )
            if token_mode == "frontend":
                token = _credential_for_service_origin(
                    supplied=token,
                    saved=str(existing_config.get("token") or "").strip(),
                    target_url=worker_url,
                    saved_url=existing_worker_url if existing_access_mode == "worker" else "",
                    credential_name="MinerU Token",
                )
            else:
                # Worker-managed mode must not retain a client-side token that
                # might later be accidentally forwarded by another route.
                token = ""

        # MinerU 特有选项
        if provider == "mineru":
            base_url = body.get("base_url", "").strip() or existing_base_url or "https://mineru.net/api/v4"
            try:
                base_url = validate_mineru_direct_api_base_url(base_url).rstrip("/")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if access_mode == "direct":
                token = _credential_for_service_origin(
                    supplied=token,
                    saved=str(existing_config.get("token") or "").strip(),
                    target_url=base_url,
                    saved_url=existing_base_url if existing_access_mode == "direct" else "",
                    credential_name="MinerU Token",
                )
        # Build the persisted payload only after all credential resolution.
        # In direct mode the token can be restored from the existing config
        # when the UI intentionally leaves the secret field blank. Constructing
        # this dict before that resolution used to overwrite the valid token
        # with an empty string on the next ordinary "保存" click.
        config: dict = {
            "access_mode": access_mode,
            "worker_url": worker_url,
            "auth_key": auth_key,
            "token_mode": token_mode,
            "token": token,
        }
        if provider == "mineru":
            config["base_url"] = base_url
            config["enable_ocr"] = body.get("enable_ocr", False)
            config["enable_formula"] = body.get("enable_formula", True)
            config["enable_table"] = body.get("enable_table", True)
            model_version = str(body.get("model_version") or "vlm").strip().lower()
            if model_version not in {"vlm", "pipeline"}:
                raise HTTPException(status_code=400, detail="model_version 必须为 'vlm' 或 'pipeline'")
            config["model_version"] = model_version

            if access_mode == "direct" and not token:
                raise HTTPException(
                    status_code=400,
                    detail="MinerU 直连模式缺少 Token，请填写 Token 后再保存",
                )
            if access_mode == "worker" and token_mode == "frontend" and not token:
                raise HTTPException(
                    status_code=400,
                    detail="MinerU Worker 的前端透传模式缺少 Token，请填写 Token 后再保存",
                )
    else:
        # Mistral 等直接 API 调用模式
        api_key = body.get("api_key", "").strip()
        base_url = body.get("base_url", "").strip()
        current_config = _load_online_ocr_config(provider)
        saved_base_url = str(current_config.get("base_url") or "https://api.mistral.ai").strip()
        base_url = base_url or saved_base_url
        try:
            base_url = validate_external_ocr_service_url(
                base_url,
                service_name="Mistral OCR Base URL",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        api_key = _credential_for_service_origin(
            supplied=api_key,
            saved=str(current_config.get("api_key") or "").strip(),
            target_url=base_url,
            saved_url=saved_base_url,
            credential_name="Mistral API Key",
        )
        if not api_key:
            raise HTTPException(status_code=400, detail="缺少 api_key 参数，请先填写或保存 API Key")

        config = {"api_key": api_key, "base_url": base_url}

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
            _document_parser_registry.register(MinerUDocumentParseAdapter(new_adapter))
            logger.info(f"MinerU 文档解析适配器已重新注册，可用: {new_adapter.is_available()}")
    except Exception as e:
        # 适配器注册失败不影响配置保存结果，仅记录警告
        logger.warning(f"重新注册在线 OCR 适配器失败: {e}")

    return {"success": True, "message": "配置已保存"}


@router.get("/api/ocr/online-config")
async def get_online_ocr_config():
    """
    获取在线 OCR 服务配置（敏感信息脱敏显示）

    返回当前在线解析配置状态。MinerU 是唯一可配置的在线服务；旧版
    Mistral/Doc2X 配置不会在这里重新暴露。

    响应:
        {
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
        }
    """
    result = {}

    for provider in _SUPPORTED_ONLINE_OCR_PROVIDERS:
        config = _load_online_ocr_config(provider)

        if provider == "mineru":
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

            provider_result["enable_ocr"] = config.get("enable_ocr", False)
            provider_result["enable_formula"] = config.get("enable_formula", True)
            provider_result["enable_table"] = config.get("enable_table", True)
            provider_result["model_version"] = config.get("model_version", "vlm")

            result[provider] = provider_result
        else:
            # Defensive compatibility branch; the provider whitelist above
            # currently makes this unreachable. Legacy config is not exposed.
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
    验证 MinerU 在线解析服务的 Token / Worker 连接有效性。

    请求体（MinerU）:
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
        current_config = _load_online_ocr_config("mistral")
        saved_base_url = str(current_config.get("base_url") or "https://api.mistral.ai").strip()
        base_url = (body.get("base_url") or saved_base_url).strip().rstrip("/")
        try:
            base_url = validate_external_ocr_service_url(
                base_url,
                service_name="Mistral OCR Base URL",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        api_key = _credential_for_service_origin(
            supplied=api_key,
            saved=str(current_config.get("api_key") or "").strip(),
            target_url=base_url,
            saved_url=saved_base_url,
            credential_name="Mistral API Key",
        )
        if not api_key:
            raise HTTPException(status_code=400, detail="缺少 api_key 参数，请先填写或保存 Mistral API Key")

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

    elif provider == "mineru":
        # Worker 代理模式验证：分两步——先测试 Worker 可达性，再测试 Token 有效性
        current_config = _load_online_ocr_config(provider)
        access_mode = body.get("access_mode", current_config.get("access_mode", "worker")).strip()
        if access_mode not in {"worker", "direct"}:
            raise HTTPException(status_code=400, detail="access_mode 必须为 'worker' 或 'direct'")
        existing_access_mode = str(current_config.get("access_mode") or "worker").strip()
        existing_worker_url = str(current_config.get("worker_url") or "").strip()
        existing_base_url = str(current_config.get("base_url") or "https://mineru.net/api/v4").strip()
        worker_url = body.get("worker_url", "").strip() or existing_worker_url
        auth_key = body.get("auth_key", "").strip()
        token = body.get("token", "").strip()
        token_mode = body.get("token_mode", current_config.get("token_mode", "frontend")).strip()
        if token_mode not in {"frontend", "worker"}:
            raise HTTPException(status_code=400, detail="token_mode 必须为 'frontend' 或 'worker'")
        base_url = body.get("base_url", "").strip() or existing_base_url or "https://mineru.net/api/v4"
        provider_label = "MinerU"

        if access_mode == "direct":
            try:
                base_url_clean = validate_mineru_direct_api_base_url(base_url).rstrip("/")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            token = _credential_for_service_origin(
                supplied=token,
                saved=str(current_config.get("token") or "").strip(),
                target_url=base_url_clean,
                saved_url=existing_base_url if existing_access_mode == "direct" else "",
                credential_name="MinerU Token",
            )
            if not token:
                raise HTTPException(status_code=400, detail="直连模式下必须提供 MinerU Token")
            try:
                with create_mineru_direct_http_client(
                    timeout_seconds=15.0,
                    connect_timeout_seconds=10.0,
                ) as client:
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
        auth_key = _credential_for_service_origin(
            supplied=auth_key,
            saved=str(current_config.get("auth_key") or "").strip(),
            target_url=worker_url_clean,
            saved_url=existing_worker_url if existing_access_mode == "worker" else "",
            credential_name="Worker Auth Key",
        )
        if token_mode == "frontend":
            token = _credential_for_service_origin(
                supplied=token,
                saved=str(current_config.get("token") or "").strip(),
                target_url=worker_url_clean,
                saved_url=existing_worker_url if existing_access_mode == "worker" else "",
                credential_name="MinerU Token",
            )
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
                    token_headers["X-MinerU-Key"] = token
                    token_test_url = f"{worker_url_clean}/mineru/result/__health__"

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
    OverviewGenerationSuperseded,
    OverviewWorkInvalidated,
    OverviewTaskCapacityExceeded,
    OverviewDepth,
    invalidate_overview_work,
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


def _configured_provider_api_key_for_target(provider_id: str, api_host: str = "") -> str:
    """Return a saved provider key only for that provider's configured origin.

    Request parameters may intentionally point at a different OpenAI-compatible
    service, but that is never enough authority to redirect a credential stored
    in the local provider registry. Callers can still use such a target by
    supplying an explicit request key.
    """
    provider = str(provider_id or "").strip()
    if not provider:
        return ""
    configured = ({**PROVIDER_CONFIG, **load_dynamic_providers()}.get(provider) or {})
    saved_key = str(configured.get("api_key") or "").strip()
    if not saved_key:
        return ""
    configured_endpoint = _get_overview_provider_endpoint(provider)
    target_endpoint = _get_overview_provider_endpoint(provider, api_host)
    if _same_service_origin(configured_endpoint, target_endpoint):
        return saved_key
    logger.warning(
        "[CredentialIsolation] 拒绝将已保存的 provider key 用于不同 origin provider=%s",
        provider,
    )
    return ""


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


def _optional_bool(value: object, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _resolve_overview_visual_runtime_params(
    request: Request,
    *,
    primary_api_key: str,
    primary_model: str,
    primary_provider: str,
    primary_api_host: str,
    visual_api_key: Optional[str] = None,
    visual_model: Optional[str] = None,
    visual_provider: Optional[str] = None,
    visual_api_host: Optional[str] = None,
    visual_enabled: Optional[bool] = None,
) -> tuple[str, str, str, str, bool]:
    """Resolve only explicit VLM overrides; empty values mean follow chat."""
    header = request.headers
    provider = (visual_provider or header.get("X-ChatPDF-Visual-Provider") or "").strip()
    model = (visual_model or header.get("X-ChatPDF-Visual-Model") or "").strip()
    explicit = bool(provider or model)
    api_key = (visual_api_key or header.get("X-ChatPDF-Visual-Api-Key") or "").strip() if explicit else ""
    api_host = (visual_api_host or header.get("X-ChatPDF-Visual-Api-Host") or "").strip() if explicit else ""
    header_enabled = header.get("X-ChatPDF-Visual-Enabled")
    enabled = _optional_bool(visual_enabled if visual_enabled is not None else header_enabled, True)
    if explicit and not api_key:
        api_key = _configured_provider_api_key_for_target(provider, api_host)
    return api_key, model, provider, api_host, enabled


def _resolve_overview_visual_policy_params(
    request: Request,
    *,
    visual_strategy: Optional[str] = None,
    local_visual_api_key: Optional[str] = None,
    local_visual_model: Optional[str] = None,
    local_visual_provider: Optional[str] = None,
    local_visual_api_host: Optional[str] = None,
) -> dict:
    header = request.headers
    strategy = (
        visual_strategy
        or header.get("X-ChatPDF-Visual-Strategy")
        or "balanced"
    ).strip().lower()
    if strategy not in {"privacy", "balanced", "quality"}:
        strategy = "balanced"
    provider = (
        local_visual_provider
        or header.get("X-ChatPDF-Local-Visual-Provider")
        or ""
    ).strip()
    model = (
        local_visual_model
        or header.get("X-ChatPDF-Local-Visual-Model")
        or ""
    ).strip()
    api_key = (
        local_visual_api_key
        or header.get("X-ChatPDF-Local-Visual-Api-Key")
        or ""
    ).strip()
    api_host = (
        local_visual_api_host
        or header.get("X-ChatPDF-Local-Visual-Api-Host")
        or ""
    ).strip()
    if provider and not api_key:
        api_key = _configured_provider_api_key_for_target(provider, api_host)
    return {
        "strategy": strategy,
        "local_provider": provider,
        "local_model": model,
        "local_api_key": api_key,
        "local_endpoint": _get_overview_provider_endpoint(provider, api_host) if provider else "",
    }


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
    if len(block_ids) > _MAX_PRETRANSLATE_BLOCK_IDS:
        raise HTTPException(
            status_code=413,
            detail=f"单次预翻译块数超过限制（最大 {_MAX_PRETRANSLATE_BLOCK_IDS}）",
        )
    if len(block_ids) > MAX_BLOCKS_PER_REQUEST:
        block_ids = block_ids[:MAX_BLOCKS_PER_REQUEST]

    target_lang = body.get("target_lang") or target_lang
    force = bool(body.get("force", force))
    # 逐段要点是设置里的开关，缺省保持开启（老客户端不传时行为不变）
    with_summary = bool(body.get("with_summary", True))

    api_key, model, provider, api_host = _resolve_overview_runtime_params(
        request,
        api_key,
        model,
        provider,
        api_host,
    )
    if not api_key:
        api_key = _configured_provider_api_key_for_target(provider, api_host)

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
            with_summary=with_summary,
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
    # 逐段要点是设置里的开关，缺省保持开启（老客户端不传时行为不变）
    with_summary = bool(body.get("with_summary", True))
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
        api_key = _configured_provider_api_key_for_target(provider, api_host)

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
            with_summary=with_summary,
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


@router.post("/documents/{doc_id}/blocks/backfill-summaries")
async def backfill_document_block_summaries(
    request: Request,
    doc_id: str,
    target_lang: str = "zh",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_host: Optional[str] = None,
    concurrency: int = 8,
):
    """给已翻译但没有要点的段落补要点，复用已有译文，不重跑翻译。"""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    try:
        body = await request.json()
    except Exception:
        body = {}

    block_ids = body.get("block_ids") or []
    if isinstance(block_ids, str):
        block_ids = [item.strip() for item in block_ids.split(",") if item.strip()]
    if not isinstance(block_ids, list):
        raise HTTPException(status_code=400, detail="block_ids 必须是数组")
    block_ids = [str(item) for item in block_ids if str(item).strip()]

    target_lang = body.get("target_lang") or target_lang
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
        api_key = _configured_provider_api_key_for_target(provider, api_host)

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
        return await backfill_block_summaries(
            data_dir=DATA_DIR,
            doc_id=doc_id,
            block_index=block_index,
            target_lang=target_lang,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=_get_overview_provider_endpoint(provider, api_host),
            block_ids=block_ids or None,
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
        raise HTTPException(status_code=409, detail="AI 缓存已清理，请重新翻译后再补要点")
    except _SupersededParseGeneration:
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新翻译后再补要点")
    except Exception as exc:
        logger.exception("[BlockTranslation] summary backfill failed for %s", doc_id)
        raise HTTPException(status_code=502, detail=f"补齐要点失败: {exc}")


@router.delete("/documents/{doc_id}/ai-cache")
async def clear_document_ai_cache(doc_id: str):
    """清理当前文档的 AI 辅助缓存，不删除原始文件、向量库或对话历史。"""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    removed: list[str] = []
    visual_task_reset: dict = {}
    table_visual_reset: dict = {}
    with _get_document_publication_lock(doc_id):
        invalidate_overview_work(
            doc_id,
            reason="AI 缓存已清理，请重新生成速览",
        )
        visual_task_reset = reset_visual_document_state(doc_id)
        ai_cache_generation = rotate_ai_cache_generation(DATA_DIR, doc_id)
        table_visual_reset = clear_table_visual_verification_cache(doc_id)

        current_doc = documents_store.get(doc_id)
        if isinstance(current_doc, dict):
            cleared_doc = deepcopy(current_doc)
            cleared_data = cleared_doc.get("data")
            visual_removed = False
            if isinstance(cleared_data, dict):
                for key in (
                    "visual_supplements",
                    "visual_supplement_commit",
                    "mineru_visual_assets",
                ):
                    if key in cleared_data:
                        cleared_data.pop(key, None)
                        visual_removed = True
            if visual_removed:
                if not save_document(doc_id, cleared_doc):
                    raise HTTPException(status_code=500, detail="视觉补充缓存清理写入失败")
                documents_store[doc_id] = cleared_doc
                removed.append("visual_supplements")

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
    removed.extend(["table_visual_verification", "visual_task_state"])

    return {
        "doc_id": doc_id,
        "removed": removed,
        "ai_cache_generation": ai_cache_generation,
        "table_visual_reset": table_visual_reset,
        "visual_task_reset": visual_task_reset,
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
    visual_api_key: Optional[str] = None,
    visual_model: Optional[str] = None,
    visual_provider: Optional[str] = None,
    visual_api_host: Optional[str] = None,
    visual_enabled: Optional[bool] = None,
    visual_strategy: Optional[str] = None,
    local_visual_api_key: Optional[str] = None,
    local_visual_model: Optional[str] = None,
    local_visual_provider: Optional[str] = None,
    local_visual_api_host: Optional[str] = None,
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
        api_key = _configured_provider_api_key_for_target(provider, api_host)

    visual_api_key, visual_model, visual_provider, visual_api_host, visual_enabled = _resolve_overview_visual_runtime_params(
        request,
        primary_api_key=api_key,
        primary_model=model,
        primary_provider=provider,
        primary_api_host=api_host,
        visual_api_key=visual_api_key,
        visual_model=visual_model,
        visual_provider=visual_provider,
        visual_api_host=visual_api_host,
        visual_enabled=visual_enabled,
    )
    visual_policy_params = _resolve_overview_visual_policy_params(
        request,
        visual_strategy=visual_strategy,
        local_visual_api_key=local_visual_api_key,
        local_visual_model=local_visual_model,
        local_visual_provider=local_visual_provider,
        local_visual_api_host=local_visual_api_host,
    )

    try:
        task = await create_overview_task(
            doc_id=doc_id,
            depth=depth,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=_get_overview_provider_endpoint(provider, api_host),
            visual_api_key=visual_api_key,
            visual_model=visual_model,
            visual_provider=visual_provider,
            visual_endpoint=_get_overview_provider_endpoint(visual_provider, visual_api_host),
            visual_enabled=visual_enabled,
            visual_policy_params=visual_policy_params,
            figure_render_mode=figure_render_mode,
            task_state_data_dir=DATA_DIR,
        )
    except OverviewTaskCapacityExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

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
        status: 任务状态 (pending/processing/completed/partial/fallback/failed)
        result: 完成后返回速览数据
        error: 失败时返回错误信息
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    parse_manifest = _require_document_parse_ready(doc_id, documents_store[doc_id])

    task = await get_task_status(task_id, data_dir=DATA_DIR, doc_id=doc_id)
    
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

    # A parse generation can remain stable while the published block snapshot
    # changes (for example after a visual/table supplement is committed).  A
    # recovered task must not expose an overview generated from the previous
    # snapshot merely because its parse manifest still matches.
    task_block_revision = str(task.block_index_revision or "").strip()
    if task_block_revision:
        try:
            current_block_index = load_block_index(DATA_DIR, doc_id)
            current_block_revision = str(
                active_block_index_revision(current_block_index, documents_store[doc_id])
                or current_block_index.get("block_index_revision")
                or current_block_index.get("block_index_hash")
                or ""
            ).strip()
        except Exception:
            current_block_revision = ""
        if current_block_revision != task_block_revision:
            raise HTTPException(status_code=409, detail="该速览任务属于旧的文档结构快照，已不再可用")
    
    response = {
        "task_id": task.task_id,
        "doc_id": task.doc_id,
        "depth": task.depth,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }
    task_ledger = get_downstream_task_events(DATA_DIR, task_id=task.task_id)
    if task_ledger:
        response["events"] = task_ledger.get("events") or []
        if task_ledger.get("shortfall"):
            response["shortfall"] = task_ledger["shortfall"]
    
    if task.status in {"completed", "partial", "fallback"} and task.result:
        response["result"] = task.result.model_dump()
        if task.status != "completed":
            response["warning"] = task.error or "速览结果不完整，可重新生成"
    elif task.status in {"failed", "cancelled", "invalidated", "superseded"}:
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
    visual_api_key: Optional[str] = None,
    visual_model: Optional[str] = None,
    visual_provider: Optional[str] = None,
    visual_api_host: Optional[str] = None,
    visual_enabled: Optional[bool] = None,
    visual_strategy: Optional[str] = None,
    local_visual_api_key: Optional[str] = None,
    local_visual_model: Optional[str] = None,
    local_visual_provider: Optional[str] = None,
    local_visual_api_host: Optional[str] = None,
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
        api_key = _configured_provider_api_key_for_target(provider, api_host)

    visual_api_key, visual_model, visual_provider, visual_api_host, visual_enabled = _resolve_overview_visual_runtime_params(
        request,
        primary_api_key=api_key,
        primary_model=model,
        primary_provider=provider,
        primary_api_host=api_host,
        visual_api_key=visual_api_key,
        visual_model=visual_model,
        visual_provider=visual_provider,
        visual_api_host=visual_api_host,
        visual_enabled=visual_enabled,
    )
    visual_policy_params = _resolve_overview_visual_policy_params(
        request,
        visual_strategy=visual_strategy,
        local_visual_api_key=local_visual_api_key,
        local_visual_model=local_visual_model,
        local_visual_provider=local_visual_provider,
        local_visual_api_host=local_visual_api_host,
    )

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
            visual_api_key=visual_api_key,
            visual_model=visual_model,
            visual_provider=visual_provider,
            visual_endpoint=_get_overview_provider_endpoint(visual_provider, visual_api_host),
            visual_enabled=visual_enabled,
            visual_policy_params=visual_policy_params,
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
    except (
        _SupersededParseGeneration,
        OverviewGenerationSuperseded,
        OverviewWorkInvalidated,
    ):
        raise HTTPException(status_code=409, detail="文档解析路线已更新，请重新生成速览")
    except TimeoutError:
        raise HTTPException(status_code=408, detail="速览生成超时，请稍后重试")
    except Exception as e:
        logger.error(f"获取速览失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取速览失败: {str(e)}")
