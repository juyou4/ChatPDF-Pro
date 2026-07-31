import asyncio
import hashlib
import importlib.util
import json
import logging
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import faiss
import httpx
import numpy as np
from fastapi import HTTPException
from config import settings as _settings
from services import bm25_service as _bm25_service, chunk_expander as _chunk_expander, hybrid_search as _hybrid_search
SentenceTransformer = None
_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None
_SENTENCE_TRANSFORMER_IMPORT_LOCK = threading.Lock()

from models.api_key_selector import select_api_key
from models.model_detector import is_embedding_model, is_rerank_model, get_model_provider
from models.model_id_resolver import PROVIDER_BASE_URL_HINTS, resolve_model_id, get_available_model_ids
from models.model_registry import EMBEDDING_MODELS
from runtime_mode import runtime
from services.formula_text import build_formula_alias_text, formula_term_matches, looks_formula_like
from services.rag_config import (
    get_context_chunk_expansion,
    should_apply_numeric_table_specialization,
)
from services.rerank_service import rerank_service
from services.semantic_group_store import (
    active_manifest_path,
    publish_generation,
    semantic_group_paths,
    validate_semantic_group_artifacts,
)
from services.table_visual_metadata import build_table_visual_metadata

logger = logging.getLogger(__name__)


def _get_sentence_transformer_class():
    """延迟导入本地模型运行时，避免远程模型用户为 PyTorch 启动成本买单。"""
    global SentenceTransformer, _HAS_SENTENCE_TRANSFORMERS
    if not _HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers 未安装")
    if SentenceTransformer is None:
        with _SENTENCE_TRANSFORMER_IMPORT_LOCK:
            if SentenceTransformer is None:
                try:
                    from sentence_transformers import SentenceTransformer as sentence_transformer_class
                except (ImportError, OSError):
                    _HAS_SENTENCE_TRANSFORMERS = False
                    raise
                SentenceTransformer = sentence_transformer_class
    return SentenceTransformer

# 只要分块来源、候选隔离或 semantic-group 构建规则发生变化就必须递增。
# 该版本用于阻止旧的污染型 semantic groups 继续参与新检索链路。
RAG_INDEX_VERSION = 4
EMBEDDING_IDENTITY_VERSION = 1
EMBEDDING_IDENTITY_MISMATCH_DETAIL = "当前 Embedding 配置与文档索引不一致，请切换原配置或重建索引"
KEYLESS_EMBEDDING_PROVIDERS = frozenset({"local", "ollama"})
_PROVIDER_DEFAULT_BASE_URLS = {
    "silicon": "https://api.siliconflow.cn/v1",
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "minimax": "https://api.minimax.chat/v1",
    "openai": "https://api.openai.com/v1",
}
_EMBEDDING_PROVIDER_SYNONYMS = {
    "openai": "openai",
    "local": "local",
    "silicon": "silicon",
    "siliconflow": "silicon",
    "aliyun": "aliyun",
    "dashscope": "aliyun",
    "moonshot": "moonshot",
    "kimi": "moonshot",
    "deepseek": "deepseek",
    "gemini": "gemini",
    "google": "gemini",
    "zhipu": "zhipu",
    "bigmodel": "zhipu",
    "glm": "zhipu",
    "minimax": "minimax",
    "xiaomi": "xiaomi",
    "mimo": "xiaomi",
}


def _embedding_identity_conflict(reason: str = "") -> HTTPException:
    detail = EMBEDDING_IDENTITY_MISMATCH_DETAIL
    if reason:
        detail = f"{detail}（{reason}）"
    return HTTPException(status_code=409, detail=detail)


def _normalize_embedding_provider(provider: Optional[str]) -> str:
    raw = str(provider or "").strip()
    if not raw:
        return ""
    return _EMBEDDING_PROVIDER_SYNONYMS.get(raw.casefold(), raw.casefold())


def _infer_embedding_provider_from_base_url(base_url: Optional[str]) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    normalized = _normalize_remote_embedding_base_url(raw)
    host = (urlparse(normalized).hostname or "").strip().lower().rstrip(".")
    if not host:
        return ""
    for provider, hint in PROVIDER_BASE_URL_HINTS.items():
        if hint and hint in host:
            return _normalize_embedding_provider(provider)
    return ""


def _pick_canonical_embedding_provider(*candidates: Optional[str]) -> str:
    for candidate in candidates:
        normalized = _normalize_embedding_provider(candidate)
        if normalized:
            return normalized
    return ""


def _registered_embedding_provider(config: Optional[dict]) -> str:
    """Resolve the credential owner recorded by a model registry entry."""
    resolved = config if isinstance(config, dict) else {}
    provider_id = _normalize_embedding_provider(resolved.get("provider_id"))
    if provider_id:
        return provider_id

    protocol_provider = _normalize_embedding_provider(resolved.get("provider"))
    if protocol_provider == "local":
        return "local"

    endpoint_provider = _infer_embedding_provider_from_base_url(
        resolved.get("base_url")
    )
    if endpoint_provider:
        return endpoint_provider

    # ``provider=openai`` commonly means an OpenAI-compatible protocol. It is
    # only the credential owner when no more specific registry/host signal is
    # available.
    return protocol_provider


def _normalize_remote_embedding_base_url(api_base: Optional[str]) -> str:
    """归一化并校验 OpenAI 兼容 embedding base_url。"""
    raw = (api_base or "").strip()
    if not raw:
        return _PROVIDER_DEFAULT_BASE_URLS["openai"]

    parsed = urlparse(raw.rstrip("/"))
    scheme = str(parsed.scheme or "").strip().lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Embedding API 地址格式无效")
    if parsed.username or parsed.password:
        raise ValueError("Embedding API 地址不允许包含用户名或密码")

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("Embedding API 地址缺少主机名")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Embedding API 地址端口无效") from exc

    if (scheme, port) in {("https", 443), ("http", 80)}:
        port = None
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    if not path:
        path = "/v1"

    return urlunparse((scheme, netloc, path, "", "", ""))


def _canonicalize_embedding_identity(
    embedding_model_id: Optional[str],
    *,
    embedding_provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    raw_model = str(embedding_model_id or "").strip()
    if not raw_model:
        raise ValueError("Embedding 模型不能为空")

    registry_key, config = resolve_model_id(raw_model)
    explicit_provider = _normalize_embedding_provider(embedding_provider)
    raw_provider_prefix = ""
    if ":" in raw_model:
        provider_part, _model_part = raw_model.split(":", 1)
        raw_provider_prefix = _normalize_embedding_provider(provider_part)
    if (
        explicit_provider
        and raw_provider_prefix
        and explicit_provider != raw_provider_prefix
    ):
        raise ValueError("Embedding 模型前缀与 embedding_provider 不一致")

    requested_provider = explicit_provider or raw_provider_prefix
    if registry_key is not None:
        canonical_model = registry_key
        resolved_config = config or {}
        registered_provider = _registered_embedding_provider(resolved_config)
        if (
            requested_provider
            and registered_provider
            and requested_provider != registered_provider
        ):
            raise ValueError("Embedding 模型与 embedding_provider 不一致")
        provider = _pick_canonical_embedding_provider(
            requested_provider,
            registered_provider,
            _infer_embedding_provider_from_base_url(base_url),
            get_model_provider(canonical_model),
        )
        base_candidate = base_url or resolved_config.get("base_url") or ""
    else:
        canonical_model = raw_model
        provider_from_model = ""
        if not provider_from_model:
            provider_from_model = _normalize_embedding_provider(get_model_provider(raw_model))
        provider = _pick_canonical_embedding_provider(
            requested_provider,
            _infer_embedding_provider_from_base_url(base_url),
            provider_from_model,
            "openai",
        )
        base_candidate = base_url or _PROVIDER_DEFAULT_BASE_URLS.get(provider, "")

    if provider == "local":
        if str(base_candidate or "").strip():
            raise ValueError("本地 Embedding 模型不应配置 embedding_api_host")
        canonical_api_host = ""
    else:
        if not str(base_candidate or "").strip():
            raise ValueError("远程 Embedding 模型需要显式 embedding_api_host")
        canonical_api_host = _normalize_remote_embedding_base_url(
            base_candidate
        )
        endpoint_provider = _infer_embedding_provider_from_base_url(
            canonical_api_host
        )
        if endpoint_provider and endpoint_provider != provider:
            raise ValueError("embedding_api_host 与 embedding_provider 不一致")

    return {
        "model": canonical_model,
        "provider": provider or "openai",
        "api_host": canonical_api_host,
        "is_local": provider == "local",
    }


def _embedding_identity_matches(left: dict, right: dict) -> bool:
    return (
        str(left.get("model") or "") == str(right.get("model") or "")
        and str(left.get("provider") or "") == str(right.get("provider") or "")
        and str(left.get("api_host") or "") == str(right.get("api_host") or "")
    )


def _resolve_index_embedding_identity(data: dict) -> dict:
    try:
        identity_version = int(data.get("embedding_identity_version") or 0)
    except (TypeError, ValueError):
        identity_version = 0
    if not _is_supported_embedding_identity_version(identity_version):
        raise ValueError("索引使用了当前服务不支持的 embedding_identity_version")

    identity = _canonicalize_embedding_identity(
        data.get("embedding_model"),
        embedding_provider=data.get("embedding_provider"),
        base_url=data.get("embedding_api_host"),
    )
    identity["version"] = identity_version
    return identity


def _resolve_requested_embedding_identity(
    *,
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_api_host: Optional[str] = None,
) -> tuple[Optional[dict], dict]:
    presence = {
        "model": bool(str(embedding_model or "").strip()),
        "provider": bool(str(embedding_provider or "").strip()),
        "api_host": bool(str(embedding_api_host or "").strip()),
    }
    if not any(presence.values()):
        return None, presence
    if not presence["model"]:
        raise _embedding_identity_conflict("缺少查询 Embedding 模型标识")

    try:
        identity = _canonicalize_embedding_identity(
            embedding_model,
            embedding_provider=embedding_provider,
            base_url=embedding_api_host,
        )
    except ValueError as exc:
        raise _embedding_identity_conflict(str(exc)) from exc
    return identity, presence


def _resolve_verified_query_embedding_identity(
    data: dict,
    *,
    api_key: Optional[str],
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_api_host: Optional[str] = None,
) -> dict:
    try:
        index_identity = _resolve_index_embedding_identity(data)
    except ValueError as exc:
        raise _embedding_identity_conflict(str(exc)) from exc

    requested_identity, requested_presence = _resolve_requested_embedding_identity(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
    )
    index_is_local = bool(index_identity.get("is_local"))
    if not _is_supported_embedding_identity_version(index_identity.get("version")):
        raise _embedding_identity_conflict("文档索引的 Embedding 身份版本不受当前服务支持，请重建索引")
    index_is_new = _is_current_embedding_identity_version(index_identity.get("version"))

    if requested_identity and not _embedding_identity_matches(index_identity, requested_identity):
        raise _embedding_identity_conflict()

    if index_is_local:
        return {
            "model": index_identity["model"],
            "provider": index_identity["provider"],
            "api_host": "",
            "api_key": None,
        }

    if not all(requested_presence.values()):
        if index_is_new or requested_identity is None:
            raise _embedding_identity_conflict(
                "远程索引查询必须显式提供 embedding_model/provider/api_host"
            )
        raise _embedding_identity_conflict(
            "旧远程索引查询必须显式提供 embedding_model/provider/api_host"
        )

    if requested_identity is None or not _embedding_identity_matches(index_identity, requested_identity):
        raise _embedding_identity_conflict()

    if (
        index_identity["provider"] not in KEYLESS_EMBEDDING_PROVIDERS
        and not str(api_key or "").strip()
    ):
        raise HTTPException(
            status_code=401,
            detail=_QUERY_EMBEDDING_AUTH_ERROR_DETAIL,
        )

    return {
        "model": index_identity["model"],
        "provider": index_identity["provider"],
        "api_host": index_identity["api_host"],
        "api_key": api_key if str(api_key or "").strip() else None,
    }


def _normalize_semantic_generation_identity(raw: Optional[dict]) -> dict:
    data = raw if isinstance(raw, dict) else {}
    provider = str(data.get("embedding_provider") or "").strip()
    try:
        identity_version = int(data.get("embedding_identity_version") or 0)
    except (TypeError, ValueError):
        identity_version = 0
    try:
        vector_dimension = int(data.get("vector_dimension") or 0)
    except (TypeError, ValueError):
        vector_dimension = 0
    return {
        "parse_generation": str(data.get("parse_generation") or data.get("transaction_id") or "").strip(),
        "document_source_hash": str(data.get("document_source_hash") or data.get("source_hash") or "").strip(),
        "vector_build_id": str(data.get("vector_build_id") or "").strip(),
        "embedding_identity_version": identity_version,
        "embedding_model": str(data.get("embedding_model") or "").strip(),
        "embedding_provider": provider,
        "embedding_api_host": "" if provider == "local" else str(data.get("embedding_api_host") or "").strip(),
        "vector_dimension": vector_dimension,
    }


def _semantic_generation_identity_complete(identity: Optional[dict]) -> bool:
    normalized = _normalize_semantic_generation_identity(identity)
    return bool(
        normalized.get("parse_generation")
        and normalized.get("document_source_hash")
        and normalized.get("vector_build_id")
        and _is_current_embedding_identity_version(normalized.get("embedding_identity_version"))
        and normalized.get("embedding_model")
        and normalized.get("embedding_provider")
        and (
            normalized.get("embedding_provider") == "local"
            or normalized.get("embedding_api_host")
        )
        and int(normalized.get("vector_dimension") or 0) > 0
    )


def _semantic_generation_identity_matches(left: Optional[dict], right: Optional[dict]) -> bool:
    normalized_left = _normalize_semantic_generation_identity(left)
    normalized_right = _normalize_semantic_generation_identity(right)
    return all(
        str(normalized_left.get(key) or "") == str(normalized_right.get(key) or "")
        for key in (
            "parse_generation",
            "document_source_hash",
            "vector_build_id",
            "embedding_identity_version",
            "embedding_model",
            "embedding_provider",
            "embedding_api_host",
            "vector_dimension",
        )
    )


def _extract_vector_semantic_identity(data: Optional[dict]) -> dict:
    payload = data if isinstance(data, dict) else {}
    index_meta = payload.get("index_meta") if isinstance(payload.get("index_meta"), dict) else {}
    try:
        embedding_identity = _resolve_index_embedding_identity(payload)
    except ValueError:
        embedding_identity = {
            "version": int(payload.get("embedding_identity_version") or 0),
            "model": str(payload.get("embedding_model") or "").strip(),
            "provider": str(payload.get("embedding_provider") or "").strip(),
            "api_host": str(payload.get("embedding_api_host") or "").strip(),
        }
    return _normalize_semantic_generation_identity(
        {
            "parse_generation": index_meta.get("parse_generation"),
            "document_source_hash": index_meta.get("document_source_hash"),
            "vector_build_id": payload.get("vector_build_id"),
            "embedding_identity_version": embedding_identity.get("version"),
            "embedding_model": embedding_identity.get("model"),
            "embedding_provider": embedding_identity.get("provider"),
            "embedding_api_host": embedding_identity.get("api_host"),
            "vector_dimension": payload.get("vector_dimension"),
        }
    )


def _embedding_cache_scope(
    embedding_model_id: str,
    embedding_provider: str,
    embedding_api_host: str,
) -> str:
    raw = json.dumps(
        {
            "model": str(embedding_model_id or ""),
            "provider": str(embedding_provider or ""),
            "api_host": str(embedding_api_host or ""),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "embed-scope:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _ensure_query_vector_matches_index(vec, index) -> np.ndarray:
    query_vector = np.asarray(vec, dtype="float32")
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise _embedding_identity_conflict("查询 embedding 维度无效")
    if int(query_vector.shape[1]) != int(index.d):
        raise _embedding_identity_conflict(
            f"查询向量维度 {int(query_vector.shape[1])} 与索引维度 {int(index.d)} 不一致"
        )
    return query_vector


def _require_current_vector_index_schema(data: object, doc_id: str) -> dict:
    """Reject persisted RAG artifacts that predate the active retrieval contract."""
    if not isinstance(data, dict):
        _index_cache.invalidate(doc_id)
        raise HTTPException(
            status_code=409,
            detail="问答索引格式已过期，需要按当前解析结果重建",
        )
    try:
        index_version = int(data.get("index_version") or 0)
    except (TypeError, ValueError):
        index_version = 0
    if index_version != RAG_INDEX_VERSION:
        _index_cache.invalidate(doc_id)
        raise HTTPException(
            status_code=409,
            detail="问答索引格式已升级，需要按当前解析结果重建",
        )
    return data


def _get_runtime_data_dir() -> str:
    """获取与 app.py/document_routes.py 一致的数据目录。"""
    if runtime.is_desktop:
        return runtime.data_dir
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "data")

# Lazy-loaded caches
local_embedding_models = {}

# Visual evidence is a small, revision-bound overlay.  It deliberately stays
# outside the document FAISS index so local/MinerU parse identities remain
# unchanged while a new VLM supplement is published.
_VISUAL_EVIDENCE_VECTOR_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_VISUAL_EVIDENCE_VECTOR_CACHE_LOCK = threading.Lock()
_VISUAL_EVIDENCE_VECTOR_CACHE_MAX_SIZE = 32


def _is_current_embedding_identity_version(version: Any) -> bool:
    try:
        return int(version or 0) == EMBEDDING_IDENTITY_VERSION
    except (TypeError, ValueError):
        return False


def _is_supported_embedding_identity_version(version: Any) -> bool:
    try:
        normalized = int(version or 0)
    except (TypeError, ValueError):
        return False
    return normalized in {0, EMBEDDING_IDENTITY_VERSION}

# ---- OpenAI Client 连接池 ----
_openai_clients: dict[tuple, "OpenAI"] = {}  # (api_base, key_hash) -> OpenAI


def _emit_retrieval_progress(
    progress_callback: Optional[Callable[[dict], None]],
    phase: str,
    message: str,
    **extra,
) -> None:
    """向外部进度回调发送检索阶段事件。

    回调是可选的；发生回调异常时直接吞掉，避免影响检索主流程。
    """
    if not progress_callback:
        return
    payload = {
        "type": "retrieval_progress",
        "phase": phase,
        "message": message,
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    try:
        progress_callback(payload)
    except Exception:
        logger.debug("[RetrievalProgress] 回调上报失败", exc_info=True)


def _get_openai_client(api_key: str, api_base: str) -> "OpenAI":
    """获取或创建 OpenAI client（连接池复用）"""
    from openai import OpenAI
    key_hash = hash(api_key)
    cache_key = (api_base, key_hash)
    if cache_key in _openai_clients:
        return _openai_clients[cache_key]
    client = OpenAI(api_key=api_key, base_url=api_base)
    _openai_clients[cache_key] = client
    return client


# ---- FAISS 索引 LRU 缓存 ----
class _IndexCache:
    """FAISS 索引 + chunks 数据 + 意群索引的 LRU 内存缓存

    避免每次搜索请求都从磁盘读取 index/pkl 文件。
    通过文件 mtime 检测更新，容量满时淘汰最久未用条目。
    """

    def __init__(self, max_size: int = 20):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()

    def get_index(self, doc_id: str, index_path: str, chunks_path: str):
        """获取缓存的 FAISS index 和 chunks data，未命中返回 None"""
        with self._lock:
            if doc_id in self._store:
                entry = self._store[doc_id]
                try:
                    index_stat = os.stat(index_path)
                    chunks_stat = os.stat(chunks_path)
                    signature = (
                        index_stat.st_mtime_ns,
                        index_stat.st_size,
                        chunks_stat.st_mtime_ns,
                        chunks_stat.st_size,
                    )
                    if signature == entry.get("artifact_signature"):
                        self._store.move_to_end(doc_id)
                        return entry["index"], entry["data"]
                except OSError:
                    pass
                # mtime changed or error, invalidate
                self._store.pop(doc_id, None)
            return None

    def put_index(self, doc_id: str, index, data, index_path: str, chunks_path: str):
        """缓存 FAISS index 和 chunks data"""
        try:
            index_stat = os.stat(index_path)
            chunks_stat = os.stat(chunks_path)
            signature = (
                index_stat.st_mtime_ns,
                index_stat.st_size,
                chunks_stat.st_mtime_ns,
                chunks_stat.st_size,
            )
        except OSError:
            signature = ()
        with self._lock:
            if doc_id in self._store:
                self._store[doc_id].update({
                    "index": index,
                    "data": data,
                    "artifact_signature": signature,
                })
                self._store.move_to_end(doc_id)
            else:
                self._store[doc_id] = {
                    "index": index,
                    "data": data,
                    "artifact_signature": signature,
                }
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def get_group_index(self, doc_id: str):
        """获取缓存的意群索引数据"""
        with self._lock:
            entry = self._store.get(doc_id)
            if entry:
                return entry.get("group_index_data")
            return None

    def put_group_index(self, doc_id: str, group_index_data):
        """缓存意群索引数据"""
        with self._lock:
            if doc_id in self._store:
                self._store[doc_id]["group_index_data"] = group_index_data

    def get_group_data(self, doc_id: str):
        """获取缓存的意群 JSON 数据"""
        with self._lock:
            entry = self._store.get(doc_id)
            if entry:
                return entry.get("group_chunk_map")
            return None

    def put_group_data(self, doc_id: str, group_chunk_map):
        """缓存意群 JSON 数据"""
        with self._lock:
            if doc_id in self._store:
                self._store[doc_id]["group_chunk_map"] = group_chunk_map

    def invalidate(self, doc_id: str = ""):
        """使缓存失效"""
        with self._lock:
            if doc_id:
                self._store.pop(doc_id, None)
            else:
                self._store.clear()


_index_cache = _IndexCache(max_size=20)


class QueryVectorCache:
    """查询向量 LRU 缓存（支持磁盘持久化）
    
    使用 OrderedDict 实现 LRU 淘汰策略，缓存键为 (embedding_model_id, query_text) 元组，
    确保不同模型的查询向量不会混淆。
    
    支持通过 persist_path 启用磁盘持久化，跨会话复用查询向量。
    """

    def __init__(self, max_size: int = 256, persist_path: str = ""):
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._persist_path = persist_path
        self._dirty_count = 0  # 自上次持久化以来的写入次数
        self._persist_interval = 20  # 每 N 次写入持久化一次
        self._lock = threading.RLock()
        if persist_path:
            self._load_from_disk()

    def get(self, model_id: str, query: str) -> Optional[np.ndarray]:
        """获取缓存的查询向量
        
        如果缓存命中，将该条目移到末尾（标记为最近使用）。
        
        Args:
            model_id: embedding 模型 ID
            query: 查询文本
            
        Returns:
            缓存的查询向量，未命中时返回 None
        """
        key = (model_id, query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, model_id: str, query: str, vector: np.ndarray) -> None:
        """存入查询向量
        
        如果缓存已满，淘汰最久未使用的条目（LRU 策略）。
        
        Args:
            model_id: embedding 模型 ID
            query: 查询文本
            vector: 查询向量
        """
        key = (model_id, query)
        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            # 定期持久化。该调用持有 RLock，保存的是一致性快照。
            self._dirty_count += 1
            if self._persist_path and self._dirty_count >= self._persist_interval:
                if self._save_to_disk():
                    self._dirty_count = 0

    def _load_from_disk(self):
        """从磁盘加载缓存"""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, OrderedDict):
                # 不信任异常大的本地缓存；保留最新的 max_size 条即可。
                with self._lock:
                    self._cache = OrderedDict(list(data.items())[-self._max_size:])
                logger.info(f"[QueryVectorCache] 从磁盘加载 {len(self._cache)} 条缓存")
        except Exception as e:
            logger.warning(f"[QueryVectorCache] 磁盘缓存加载失败: {e}")

    def _save_to_disk(self) -> bool:
        """持久化一致性快照，避免并发截断同一个 pickle 文件。"""
        if not self._persist_path:
            return False
        temp_path = ""
        try:
            cache_dir = os.path.dirname(self._persist_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with self._lock:
                snapshot = OrderedDict(self._cache)
            fd, temp_path = tempfile.mkstemp(
                prefix=".query_vector_cache-",
                suffix=".tmp",
                # Must share a filesystem with the destination for os.replace().
                dir=cache_dir or ".",
            )
            with os.fdopen(fd, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self._persist_path)
            return True
        except Exception as e:
            logger.warning(f"[QueryVectorCache] 磁盘缓存保存失败: {e}")
            return False
        finally:
            if temp_path:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except OSError:
                    pass

    def flush(self):
        """立即持久化到磁盘"""
        with self._lock:
            if self._persist_path and self._dirty_count > 0 and self._save_to_disk():
                self._dirty_count = 0


# 全局查询向量缓存实例（默认容量 256，启用磁盘持久化）
_cache_persist_path = os.path.join(
    _get_runtime_data_dir(), "cache", "query_vector_cache.pkl"
)
_query_vector_cache = QueryVectorCache(persist_path=_cache_persist_path)

# Vector/document swaps and semantic-group publication must share the same
# document-scoped lock. Keeping it in this dependency layer lets route code and
# background semantic workers serialize compare-and-publish without a circular
# import.
_document_publication_locks_guard = threading.Lock()
_document_publication_locks: dict[str, threading.RLock] = {}


def get_document_publication_lock(doc_id: str) -> threading.RLock:
    normalized_doc_id = str(doc_id or "").strip()
    if not normalized_doc_id:
        raise ValueError("doc_id 不能为空")
    with _document_publication_locks_guard:
        return _document_publication_locks.setdefault(
            normalized_doc_id,
            threading.RLock(),
        )

# 记录正在生成意群的文档 ID，防止重复提交（需求 6.1）
_group_generation_in_progress: set[str] = set()
_group_generation_lock = threading.RLock()
try:
    _SEMANTIC_GROUP_BACKGROUND_MAX_PENDING = max(
        1,
        min(8, int(os.getenv("CHATPDF_SEMANTIC_GROUP_BACKGROUND_MAX_PENDING", "2"))),
    )
except ValueError:
    _SEMANTIC_GROUP_BACKGROUND_MAX_PENDING = 2
_semantic_group_background_admission = threading.BoundedSemaphore(
    _SEMANTIC_GROUP_BACKGROUND_MAX_PENDING
)

# ---- 模块级单例：避免热路径中重复实例化和重复 import ----
from services.query_rewriter import QueryRewriter as _QueryRewriter
from services.query_analyzer import (
    analyze_evidence_need as _analyze_evidence_need,
    analyze_query_type as _analyze_query_type,
    get_retrieval_strategy as _get_retrieval_strategy,
)
from services.rag_config import RAGConfig as _RAGConfig
from services.context_builder import ContextBuilder as _ContextBuilder
from services.retrieval_logger import RetrievalLogger as _RetrievalLogger, RetrievalTrace as _RetrievalTrace

_query_rewriter_singleton = _QueryRewriter()
_rag_config_singleton = _RAGConfig()
_context_builder_singleton = _ContextBuilder()
_retrieval_logger_singleton = _RetrievalLogger()


def _resolve_intent_decision(query: str, intent_decision: Optional[dict] = None) -> dict:
    """优先使用路由层冻结的判定，缺失时保持旧调用兼容。"""
    if isinstance(intent_decision, dict):
        query_type = str(intent_decision.get("query_type") or "").strip()
        evidence_need = intent_decision.get("evidence_need")
        if query_type in {"overview", "extraction", "analytical", "specific"} and isinstance(evidence_need, (list, tuple, set)):
            resolved = dict(intent_decision)
            resolved["query_type"] = query_type
            resolved["evidence_need"] = [
                str(item).strip() for item in evidence_need if str(item).strip()
            ]
            resolved.setdefault("top_k", 10)
            return resolved
    return _get_retrieval_strategy(query)


def _resolve_evidence_need(
    query: str,
    evidence_need: Optional[List[str]] = None,
) -> set[str]:
    if evidence_need is not None:
        return {str(item).strip() for item in evidence_need if str(item).strip()}
    return set(_analyze_evidence_need(query) or [])

# ---- 意群数据目录（只计算一次）----
_SEMANTIC_GROUPS_DIR: str = os.path.join(
    _get_runtime_data_dir(), "semantic_groups"
)


def preprocess_text(text: str) -> str:
    """
    Lightweight preprocessing before chunking:
    - 去掉常见版权/噪声行（如 IEEE 授权提示）
    - 合并多余空行
    - 修复连字符断行
    - 过滤图表乱码（NULL字符）
    """
    if not text:
        return ""

    lines = []
    noisy_patterns = [
        "Authorized licensed use limited to",
        "All rights reserved",
    ]

    for line in text.splitlines():
        lstrip = line.strip()
        if any(pat.lower() in lstrip.lower() for pat in noisy_patterns):
            continue
        
        # 只过滤包含大量 NULL 字符的行
        null_count = line.count('\u0000') + line.count('\x00')
        if len(line) > 5 and null_count / len(line) > 0.3:
            continue
        
        # 移除 NULL 字符
        cleaned_line = line.replace('\u0000', '').replace('\x00', '')
        # 保留段落空行。``\n\n`` 是 RecursiveCharacterTextSplitter 的首选分隔符，
        # 丢掉空行会让下面的 ``\n{3,}`` 折叠变成死代码、输出里永远不存在 ``\n\n``，
        # 于是切分器在主路径和兜底路径上都退化成按单行/空格/字符切。
        # 空白行归一为空串，连续空行交给下面的折叠规则收敛成一个段落分隔符。
        lines.append(cleaned_line if cleaned_line.strip() else "")

    cleaned = "\n".join(lines)
    # 修复连字符断行：word-\nword -> wordword
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    # 统一空白
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_embedding_model_id(embedding_model_id: Optional[str]) -> Optional[str]:
    """归一化 embedding 模型 ID，返回 Model_Registry 中的键名

    使用 Model_ID_Resolver 统一解析前端传入的模型 ID，
    支持 composite key（provider:modelId）和 plain key 两种格式。

    Args:
        embedding_model_id: 前端传入的模型 ID

    Returns:
        Model_Registry 中的键名，解析失败时返回 None 并记录警告日志（包含可用模型列表）
    """
    if not embedding_model_id:
        return None

    # 使用 Model_ID_Resolver 统一解析
    registry_key, config = resolve_model_id(embedding_model_id)
    if registry_key is not None:
        return registry_key

    # 解析失败，记录警告日志并返回 None
    available_models = get_available_model_ids()
    logger.warning(
        f"无法解析模型 ID '{embedding_model_id}'，"
        f"可用模型列表: {available_models}"
    )
    return None


def _estimate_embedding_tokens(text: str) -> int:
    """粗略估算文本 token 数（偏保守）"""
    if not text:
        return 1
    content = text.strip()
    if not content:
        return 1

    ascii_chars = sum(1 for ch in content if ord(ch) < 128)
    non_ascii_chars = len(content) - ascii_chars
    # 英文约 3.5 字符/token；中日韩字符按 ~1 token 估算并略放大
    est = int(ascii_chars / 3.5 + non_ascii_chars * 1.1)
    return max(1, est)


def _truncate_text_to_token_budget(text: str, token_budget: int) -> str:
    """将文本截断到 token 预算内（保持单条输入 -> 单条向量映射）"""
    budget = max(1, int(token_budget))
    if _estimate_embedding_tokens(text) <= budget:
        return text

    left, right = 1, len(text)
    best = text[:1]
    while left <= right:
        mid = (left + right) // 2
        candidate = text[:mid]
        if _estimate_embedding_tokens(candidate) <= budget:
            best = candidate
            left = mid + 1
        else:
            right = mid - 1
    return best


def _prepare_embedding_batches(
    texts: List[str],
    token_budget: int,
    single_text_token_budget: Optional[int] = None,
) -> List[List[str]]:
    """按总 token 预算分批；超长单条文本会自动截断.

    Some OpenAI-compatible providers enforce a stricter per-input limit than
    their advertised batch context.  Keep the persisted chunk unchanged, but use
    a safer representative prefix for vectorization.
    """
    budget = max(1, int(token_budget))
    single_budget = max(1, int(single_text_token_budget or budget))
    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_tokens = 0

    for idx, raw in enumerate(texts):
        text = raw if isinstance(raw, str) else str(raw)
        est = _estimate_embedding_tokens(text)

        if est > single_budget:
            truncated = _truncate_text_to_token_budget(text, single_budget)
            logger.warning(
                f"[EmbeddingBatch] 文本过长，已截断: idx={idx}, est_tokens={est}, "
                f"budget={single_budget}, old_chars={len(text)}, new_chars={len(truncated)}"
            )
            text = truncated
            est = _estimate_embedding_tokens(text)

        if current_batch and (current_tokens + est > budget):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += est

    if current_batch:
        batches.append(current_batch)

    return batches


def _is_token_limit_error(exc: Exception) -> bool:
    """判断是否为 embedding 输入 token 超限错误"""
    msg = str(exc).lower()
    hints = (
        "input must have less than",
        "maximum context length",
        "too many tokens",
        "token limit",
    )
    return ("token" in msg) and any(h in msg for h in hints)


def _is_retryable_embedding_input_error(exc: Exception, batch: List[str]) -> bool:
    """Detect provider-side input validation errors that can be fixed by shrinking.

    SiliconFlow may return code 20015 with a generic "parameter is invalid" for
    overlong embedding inputs, without mentioning token limits.  Only treat it
    as shrink-retryable when the current request is plausibly too large.
    """
    if _is_token_limit_error(exc):
        return True

    if not batch:
        return False

    max_est_tokens = max(_estimate_embedding_tokens(item or "") for item in batch)
    total_est_tokens = sum(_estimate_embedding_tokens(item or "") for item in batch)
    plausibly_large = len(batch) > 1 or max_est_tokens > 256 or total_est_tokens > 512
    if not plausibly_large:
        return False

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = response.json()
            if isinstance(body, dict):
                code = str(body.get("code") or "")
                message = str(body.get("message") or "").lower()
                if code == "20015" and "parameter" in message and "invalid" in message:
                    return True
                err = body.get("error")
                if isinstance(err, dict):
                    ecode = str(err.get("code") or "")
                    emsg = str(err.get("message") or "").lower()
                    if ecode == "20015" and "parameter" in emsg and "invalid" in emsg:
                        return True
        except Exception:
            pass

    msg = str(exc).lower()
    return (
        "code" in msg
        and "20015" in msg
        and "parameter" in msg
        and "invalid" in msg
    )


def _is_model_not_found_error(exc: Exception) -> bool:
    """判断是否为模型不存在/未开通类错误。"""
    # 1) 先尝试从异常对象中解析结构化错误（openai/httpx）
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = response.json()
            if isinstance(body, dict):
                # 常见格式 A: {"code":20012,"message":"..."}
                code = body.get("code")
                message = body.get("message", "")
                if str(code) == "20012":
                    return True
                # 常见格式 B: {"error":{"code":"model_not_found","message":"..."}}
                if isinstance(body.get("error"), dict):
                    e = body["error"]
                    ecode = e.get("code")
                    emsg = e.get("message", "")
                    if str(ecode) == "20012" or str(ecode).lower() in {"model_not_found", "no_such_model"}:
                        return True
                    if isinstance(emsg, str) and "model does not exist" in emsg.lower():
                        return True
                if isinstance(message, str) and "model does not exist" in message.lower():
                    return True
        except Exception:
            pass

    # 2) 回退到字符串匹配
    msg = str(exc).lower()
    if "model" not in msg:
        return False
    hints = (
        "model does not exist",
        "model not exist",
        "model_not_found",
        "no such model",
        "code: 20012",
        "code:20012",
        "'code': 20012",
        "'code':20012",
        '"code": 20012',
        '"code":20012',
    )
    return any(h in msg for h in hints)


_NON_DEGRADABLE_EMBEDDING_AUTH_CODES = {
    "access_denied",
    "api_key_not_found",
    "authentication_error",
    "forbidden",
    "incorrect_api_key",
    "insufficient_permission",
    "insufficient_permissions",
    "invalid_api_key",
    "invalid_authentication",
    "key_not_found",
    "permission_denied",
    "unauthorized",
}
_NON_DEGRADABLE_EMBEDDING_MODEL_CODES = {
    "20012",
    "deployment_not_found",
    "model_decommissioned",
    "model_disabled",
    "model_not_available",
    "model_not_exist",
    "model_not_found",
    "no_such_model",
    "unsupported_model",
}
_QUERY_EMBEDDING_AUTH_ERROR_DETAIL = "Embedding API 凭证无效或无权访问，请检查 API Key 与模型授权配置"
_QUERY_EMBEDDING_MODEL_ERROR_DETAIL = "当前 Embedding 模型不存在、未开通或不可用，请切换可用模型后重试"
_EMBEDDING_MODEL_NOT_FOUND_PATTERNS = (
    re.compile(
        r"\bmodel\s+[`'\"]?[a-z0-9][a-z0-9._:/-]{0,127}[`'\"]?\s+(?:was\s+)?not\s+found\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmodel\s+[`'\"]?[a-z0-9][a-z0-9._:/-]{0,127}[`'\"]?\s+does\s+not\s+exist\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmodel\s+[`'\"]?[a-z0-9][a-z0-9._:/-]{0,127}[`'\"]?\s+is\s+unavailable\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdeployment\s+[`'\"]?[a-z0-9][a-z0-9._:/-]{0,127}[`'\"]?\s+(?:was\s+)?not\s+found\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdeployment\s+[`'\"]?[a-z0-9][a-z0-9._:/-]{0,127}[`'\"]?\s+does\s+not\s+exist\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdeployment\s+[`'\"]?[a-z0-9][a-z0-9._:/-]{0,127}[`'\"]?\s+is\s+unavailable\b",
        re.IGNORECASE,
    ),
)


def _coerce_embedding_error_status(value: object) -> Optional[int]:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if 100 <= status <= 599 else None


def _normalize_embedding_error_code(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", raw)
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_split)
    return re.sub(r"[^a-z0-9]+", "_", camel_split.casefold()).strip("_")


def _coerce_embedding_error_payload(value: object) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except Exception:
                return None
            if isinstance(parsed, dict):
                return parsed
    return None


def _iter_embedding_exception_chain(exc: Exception, max_depth: int = 5):
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    depth = 0
    while isinstance(current, BaseException) and depth < max_depth:
        node_id = id(current)
        if node_id in seen:
            break
        seen.add(node_id)
        yield current
        next_exc = getattr(current, "__cause__", None)
        if not isinstance(next_exc, BaseException):
            next_exc = getattr(current, "__context__", None)
        current = next_exc if isinstance(next_exc, BaseException) else None
        depth += 1


def _iter_embedding_error_nodes(payload: dict):
    stack = [payload]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        yield node
        nested = node.get("error")
        if isinstance(nested, dict):
            stack.append(nested)


def _extract_embedding_error_payloads(exc: Exception) -> list[dict]:
    payloads: list[dict] = []
    seen_signatures: set[str] = set()

    def _append_payload(candidate: object):
        payload = _coerce_embedding_error_payload(candidate)
        if payload is None:
            return
        try:
            signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        except Exception:
            signature = repr(payload)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        payloads.append(payload)

    for error_node in _iter_embedding_exception_chain(exc):
        _append_payload(getattr(error_node, "body", None))
        for arg in getattr(error_node, "args", ()) or ():
            _append_payload(arg)

        response = getattr(error_node, "response", None)
        if response is None:
            continue
        for candidate in (
            getattr(response, "_json", None),
            getattr(response, "json", None),
            getattr(response, "text", None),
            getattr(response, "content", None),
        ):
            payload = None
            if callable(candidate):
                try:
                    payload = candidate()
                except Exception:
                    payload = None
            else:
                payload = candidate
            _append_payload(payload)

    return payloads


def _collect_embedding_error_codes(exc: Exception) -> list[str]:
    seen: set[str] = set()
    codes: list[str] = []
    for error_node in _iter_embedding_exception_chain(exc):
        for value in (
            getattr(error_node, "code", None),
            getattr(error_node, "type", None),
            getattr(error_node, "error_code", None),
        ):
            code = _normalize_embedding_error_code(value)
            if code and code not in seen:
                seen.add(code)
                codes.append(code)

    for payload in _extract_embedding_error_payloads(exc):
        for node in _iter_embedding_error_nodes(payload):
            for field in ("code", "type", "error_code", "error_type"):
                code = _normalize_embedding_error_code(node.get(field))
                if code and code not in seen:
                    seen.add(code)
                    codes.append(code)

    return codes


def _collect_embedding_error_messages(exc: Exception) -> list[str]:
    seen: set[str] = set()
    messages: list[str] = []
    for error_node in _iter_embedding_exception_chain(exc):
        for field in ("message", "detail", "msg"):
            value = getattr(error_node, field, None)
            if isinstance(value, str):
                text = value.strip()
                if text and text not in seen:
                    seen.add(text)
                    messages.append(text)

    for payload in _extract_embedding_error_payloads(exc):
        for node in _iter_embedding_error_nodes(payload):
            for field in ("message", "detail", "msg"):
                value = node.get(field)
                if isinstance(value, str):
                    text = value.strip()
                    if text and text not in seen:
                        seen.add(text)
                        messages.append(text)

    for error_node in _iter_embedding_exception_chain(exc):
        text = str(error_node).strip()
        if text and text not in seen:
            seen.add(text)
            messages.append(text)
    return messages


def _collect_embedding_error_statuses(exc: Exception) -> list[int]:
    seen: set[int] = set()
    statuses: list[int] = []
    for error_node in _iter_embedding_exception_chain(exc):
        for value in (
            getattr(error_node, "status_code", None),
            getattr(getattr(error_node, "response", None), "status_code", None),
        ):
            status = _coerce_embedding_error_status(value)
            if status is not None and status not in seen:
                seen.add(status)
                statuses.append(status)

    for payload in _extract_embedding_error_payloads(exc):
        for node in _iter_embedding_error_nodes(payload):
            for field in ("status", "status_code", "statusCode"):
                status = _coerce_embedding_error_status(node.get(field))
                if status is not None and status not in seen:
                    seen.add(status)
                    statuses.append(status)
    return statuses


def _looks_like_auth_error_message(message: str) -> bool:
    if not isinstance(message, str):
        return False
    raw = message.strip()
    if not raw:
        return False
    lower = raw.casefold()

    direct_hints = (
        "invalid api key",
        "incorrect api key",
        "invalid_api_key",
        "permission_denied",
        "permission denied",
        "authentication error",
        "authentication_error",
        "unauthorized",
        "forbidden",
    )
    if any(hint in lower for hint in direct_hints):
        return True

    if "api key" in lower and any(hint in lower for hint in ("invalid", "incorrect", "missing", "required")):
        return True

    if any(token in raw for token in ("认证失败", "鉴权失败", "未授权", "禁止访问")):
        return True

    if (
        any(token in lower for token in ("api key", "apikey", "token", "credential"))
        or any(token in raw for token in ("API Key", "密钥", "凭证"))
    ) and any(token in raw for token in ("无效", "错误", "缺失", "未提供", "不存在")):
        return True

    return False


def _looks_like_model_access_error_message(message: str) -> bool:
    if not isinstance(message, str):
        return False
    raw = message.strip()
    if not raw:
        return False
    lower = raw.casefold()

    direct_hints = (
        "model_not_found",
        "model does not exist",
        "model not exist",
        "no such model",
        "deployment not found",
        "unsupported model",
        "model is not available",
        "model_not_available",
    )
    if "model" in lower and any(hint in lower for hint in direct_hints):
        return True

    if "model" in lower and "access" in lower and any(hint in lower for hint in ("denied", "forbidden", "permission")):
        return True

    if any(pattern.search(raw) for pattern in _EMBEDDING_MODEL_NOT_FOUND_PATTERNS):
        return True

    if "模型" in raw and any(token in raw for token in ("不存在", "未开通", "不可用", "无权限", "无权调用", "未授权调用", "未发布")):
        return True

    return False


def _build_non_degradable_query_embedding_http_error(exc: Exception) -> Optional[HTTPException]:
    codes = _collect_embedding_error_codes(exc)
    statuses = _collect_embedding_error_statuses(exc)
    messages = _collect_embedding_error_messages(exc)

    if any(code in _NON_DEGRADABLE_EMBEDDING_MODEL_CODES for code in codes):
        return HTTPException(status_code=409, detail=_QUERY_EMBEDDING_MODEL_ERROR_DETAIL)
    if any(code in _NON_DEGRADABLE_EMBEDDING_AUTH_CODES for code in codes):
        return HTTPException(status_code=401, detail=_QUERY_EMBEDDING_AUTH_ERROR_DETAIL)

    if any(status in {401, 403} for status in statuses):
        if any(_looks_like_model_access_error_message(message) for message in messages):
            return HTTPException(status_code=409, detail=_QUERY_EMBEDDING_MODEL_ERROR_DETAIL)
        return HTTPException(status_code=401, detail=_QUERY_EMBEDDING_AUTH_ERROR_DETAIL)

    if any(_looks_like_auth_error_message(message) for message in messages):
        return HTTPException(status_code=401, detail=_QUERY_EMBEDDING_AUTH_ERROR_DETAIL)
    if any(_looks_like_model_access_error_message(message) for message in messages):
        return HTTPException(status_code=409, detail=_QUERY_EMBEDDING_MODEL_ERROR_DETAIL)

    return None


def _summarize_embedding_error(exc: Exception) -> str:
    parts = [exc.__class__.__name__]
    statuses = _collect_embedding_error_statuses(exc)
    codes = _collect_embedding_error_codes(exc)
    if statuses:
        parts.append("status=" + ",".join(str(status) for status in statuses))
    if codes:
        parts.append("code=" + ",".join(codes[:4]))
    return " ".join(parts)


def _fetch_available_model_ids(api_base: str, api_key: str) -> list[str]:
    """从提供商拉取可用模型列表（最佳努力，不抛异常）。"""
    if not api_base or not api_key:
        return []

    base = api_base.rstrip("/")
    urls = []
    if base.endswith("/v1"):
        urls.append(f"{base}/models")
    else:
        urls.append(f"{base}/v1/models")
        urls.append(f"{base}/models")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for url in urls:
        try:
            with httpx.Client(timeout=httpx.Timeout(8.0, connect=5.0)) as client:
                resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                continue
            data = resp.json()
            items = data.get("data", []) if isinstance(data, dict) else []
            model_ids = []
            for item in items:
                model_id = item.get("id") if isinstance(item, dict) else None
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())
            if model_ids:
                return model_ids
        except Exception:
            continue

    return []


def _select_fallback_embedding_model(
    available_models: list[str],
    preferred_model: str,
    excluded_models: Optional[list[str]] = None,
) -> Optional[str]:
    """从可用模型中选择 embedding 回退模型。"""
    if not available_models:
        return None

    # 去重并保序
    seen = set()
    models = []
    for model_id in available_models:
        if model_id not in seen:
            seen.add(model_id)
            models.append(model_id)

    # 排除已确认失败的模型，避免“回退”仍选回原模型
    exclude_set = {
        m.strip().lower()
        for m in (excluded_models or [])
        if isinstance(m, str) and m.strip()
    }
    if exclude_set:
        models = [m for m in models if m.lower() not in exclude_set]
        if not models:
            return None

    preferred = (preferred_model or "").strip()
    if preferred:
        # 1) 模型别名升级（优先把历史模型切到新 ID）
        alias_targets = {
            "qwen/qwen-embedding-8b": "Qwen/Qwen3-Embedding-8B",
            "text-embedding-ada-002": "text-embedding-3-small",
            "minimax-embedding-v2": "embo-01",
        }
        mapped_target = alias_targets.get(preferred.lower())
        if mapped_target:
            for model_id in models:
                if model_id.lower() == mapped_target.lower():
                    return model_id

        # 2) 后缀匹配（例如 Pro/BAAI/bge-m3 与 BAAI/bge-m3）
        for model_id in models:
            if model_id.lower().endswith(preferred.lower()):
                return model_id

    # 3) 优先常见 embedding 模型
    prefer_order = [
        "BAAI/bge-m3",
        "Qwen/Qwen3-Embedding-8B",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-v4",
        "gemini-embedding-2-preview",
        "gemini-embedding-001",
        "embo-01",
        "embedding-3",
        "Qwen/Qwen-Embedding-8B",
    ]
    for target in prefer_order:
        for model_id in models:
            if model_id.lower() == target.lower():
                return model_id

    # 4) 任意可识别的 embedding 模型
    for model_id in models:
        if is_embedding_model(model_id) and not is_rerank_model(model_id):
            return model_id

    return None


def _embed_batch_with_auto_shrink(
    client,
    model: str,
    batch: List[str],
    token_budget: int,
    depth: int = 0
) -> List[list]:
    """嵌入调用：token 超限时自动拆分 batch；单条超限时自动截断重试"""
    if depth > 12:
        raise RuntimeError("embedding 重试层级过深，已中止")

    try:
        response = client.embeddings.create(model=model, input=batch)
        data = getattr(response, "data", [])
        if len(data) != len(batch):
            raise ValueError(
                f"Embedding 返回数量不匹配: input={len(batch)}, output={len(data)}"
            )
        return [item.embedding for item in data]
    except Exception as exc:
        if not _is_retryable_embedding_input_error(exc, batch):
            raise

        # 多条场景：二分拆批，避免整批失败
        if len(batch) > 1:
            mid = len(batch) // 2
            logger.warning(
                f"[EmbeddingBatch] 触发 token 限制，自动拆分重试: size={len(batch)} -> "
                f"{mid}+{len(batch)-mid}"
            )
            left = _embed_batch_with_auto_shrink(client, model, batch[:mid], token_budget, depth + 1)
            right = _embed_batch_with_auto_shrink(client, model, batch[mid:], token_budget, depth + 1)
            return left + right

        # 单条场景：继续缩短文本并重试
        original = batch[0]
        reduced_budget = max(64, int(token_budget * 0.65))
        truncated = _truncate_text_to_token_budget(original, reduced_budget)
        if len(truncated) >= len(original):
            fallback_len = max(1, len(original) // 2)
            truncated = original[:fallback_len]
        if not truncated:
            truncated = original[:1]

        logger.warning(
            f"[EmbeddingBatch] 单条文本超限，自动缩短重试: old_chars={len(original)}, "
            f"new_chars={len(truncated)}, budget={reduced_budget}"
        )
        return _embed_batch_with_auto_shrink(
            client,
            model,
            [truncated],
            reduced_budget,
            depth + 1
        )


def get_embedding_function(
    embedding_model_id: str,
    api_key: str = None,
    base_url: str = None,
    allow_model_fallback: bool = False,
):
    """获取指定模型的 embedding 函数

    优先使用 Model_ID_Resolver 解析模型 ID 并获取完整配置；
    如果 Resolver 无法解析（未注册模型），则回退到 model_detector 推断 provider 和 base_url，
    输出警告日志并尝试继续。

    Args:
        embedding_model_id: 模型 ID，支持 composite key（provider:modelId）或 plain key
        api_key: API 密钥（非本地模型必需）
        base_url: 自定义 API 基础 URL（可选，优先于注册表中的 base_url）
        allow_model_fallback: 是否允许 model-not-found 时自动切换模型

    Returns:
        embedding 函数，接受文本列表并返回向量数组

    Raises:
        ValueError: 当模型是 rerank 模型而非 embedding 模型时
        ValueError: 当非本地模型缺少 API Key 时
    """
    # 使用 Model_ID_Resolver 统一解析模型 ID
    registry_key, config = resolve_model_id(embedding_model_id)

    if registry_key is not None:
        # Resolver 解析成功，使用注册表中的配置
        embedding_model_id = registry_key
        provider = config["provider"]
        model_name = config.get("model_name", embedding_model_id)
        api_base = base_url or config.get("base_url")
    else:
        # Resolver 解析失败，尝试从 composite key 中提取 provider 信息
        logger.warning(
            f"模型 '{embedding_model_id}' 未在注册表中找到，"
            f"尝试从 composite key 推断 provider 和 base_url"
        )
        config = None

        if ":" in embedding_model_id:
            # composite key 格式：provider:modelId
            provider_part, model_part = embedding_model_id.split(":", 1)
            provider_part = provider_part.casefold()
            embedding_model_id = model_part  # 实际调用 API 时用 modelId 部分
            model_name = model_part

            # 根据 provider 推断 base_url
            provider = _normalize_embedding_provider(provider_part) or "openai"

            # 使用 provider 对应的默认 base_url
            api_base = base_url or _PROVIDER_DEFAULT_BASE_URLS.get(
                provider_part,
                _PROVIDER_DEFAULT_BASE_URLS["openai"],
            )
            logger.info(
                f"从 composite key 推断: provider={provider_part}, "
                f"model={model_part}, base_url={api_base}"
            )
        else:
            # plain key，使用 model_detector 推断
            provider = get_model_provider(embedding_model_id)
            model_name = embedding_model_id
            api_base = _normalize_remote_embedding_base_url(
                base_url or _PROVIDER_DEFAULT_BASE_URLS["openai"]
            )

    # 验证模型类型
    if not is_embedding_model(embedding_model_id):
        if is_rerank_model(embedding_model_id):
            raise ValueError(f"模型 {embedding_model_id} 是 rerank 模型，不是 embedding 模型")
        logger.warning(
            f"模型 '{embedding_model_id}' 不匹配 embedding 模型模式，尝试继续使用"
        )

    # 本地模型：使用 SentenceTransformer
    if provider == "local":
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ValueError(
                "本地 embedding 模型不可用（sentence-transformers 未安装）。"
                "请使用远程 embedding API，或安装完整依赖: pip install -r requirements.txt"
            )
        if model_name not in local_embedding_models:
            logger.info(f"加载本地 embedding 模型: {model_name}")
            sentence_transformer_class = _get_sentence_transformer_class()
            local_embedding_models[model_name] = sentence_transformer_class(model_name)
        model = local_embedding_models[model_name]
        return lambda texts: model.encode(texts)

    # 远程模型：从 Key 池中随机选择一个有效 Key
    actual_key = select_api_key(api_key) if api_key else None
    if not actual_key:
        if provider == "ollama":
            actual_key = "ollama"
        else:
            raise ValueError(f"模型 '{embedding_model_id}' 需要 API Key")

    api_base = _normalize_remote_embedding_base_url(api_base)

    # 使用连接池复用 OpenAI client，避免每次创建新连接
    client = _get_openai_client(actual_key, api_base)

    # 远程 embedding 接口通常限制“单次请求总 token”，不是“单条文本 token”
    # 使用模型 max_tokens 的 90% 作为单请求预算，并自动分批请求。
    cfg = EMBEDDING_MODELS.get(embedding_model_id, {})
    max_tokens = int(cfg.get("max_tokens") or 8192)
    request_token_budget = max(128, int(max_tokens * 0.9))
    single_text_token_budget = max(256, min(request_token_budget, int(max_tokens * 0.25), 2048))
    # 优先使用 model_name 作为真实请求模型 ID（动态模型场景下 key != model_id）
    model_for_request = model_name or embedding_model_id
    fallback_checked = not allow_model_fallback

    def embed_texts(texts):
        nonlocal model_for_request, fallback_checked
        if texts is None:
            return np.array([])
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = [t if isinstance(t, str) else str(t) for t in texts]
        if not text_list:
            return np.array([])

        batches = _prepare_embedding_batches(
            text_list,
            request_token_budget,
            single_text_token_budget=single_text_token_budget,
        )
        if len(batches) > 1:
            logger.info(
                f"[EmbeddingBatch] 模型={embedding_model_id}, 文本数={len(text_list)}, "
                f"分批={len(batches)}, 预算={request_token_budget}"
            )

        vectors: List[list] = []
        for batch in batches:
            try:
                vectors.extend(
                    _embed_batch_with_auto_shrink(
                        client=client,
                        model=model_for_request,
                        batch=batch,
                        token_budget=request_token_budget,
                    )
                )
            except Exception as exc:
                if not _is_model_not_found_error(exc):
                    raise

                # 自动换模型是 opt-in 的：换掉用户选定的 embedding 模型会产出与已持久化
                # 索引不兼容的向量，所以每个生产调用点都显式关掉了它。
                if allow_model_fallback and not fallback_checked:
                    fallback_checked = True
                    available_models = _fetch_available_model_ids(api_base, actual_key)
                    fallback_model = _select_fallback_embedding_model(
                        available_models=available_models,
                        preferred_model=model_for_request,
                        excluded_models=[model_for_request],
                    )
                    if fallback_model and fallback_model != model_for_request:
                        logger.warning(
                            f"[EmbeddingModelFallback] 模型不可用，自动回退: "
                            f"{model_for_request} -> {fallback_model}"
                        )
                        model_for_request = fallback_model
                        vectors.extend(
                            _embed_batch_with_auto_shrink(
                                client=client,
                                model=model_for_request,
                                batch=batch,
                                token_budget=request_token_budget,
                            )
                        )
                        continue

                # 但把 provider 的原始异常翻译成可操作的提示与"是否自动换模型"无关。
                # 这条 raise 曾经也被关在 allow_model_fallback 里，于是在**所有**生产
                # 调用点上，模型没开通时用户只会看到一个裸的 provider 异常。
                raise ValueError(
                    f"Embedding模型 '{model_for_request}' 不存在或未开通。"
                    "请在「模型服务」中同步模型后重新选择可用的 Embedding 模型。"
                ) from exc

        if len(vectors) != len(text_list):
            raise ValueError(
                f"Embedding 向量数量异常: input={len(text_list)}, output={len(vectors)}"
            )

        return np.array(vectors)

    return embed_texts


def get_chunk_params(embedding_model_id: str, base_chunk_size: int = 1200, base_overlap: int = 200) -> tuple[int, int]:
    """Return (chunk_size, chunk_overlap) with model-aware clamping."""
    cfg = EMBEDDING_MODELS.get(embedding_model_id, {})
    max_ctx = cfg.get("max_tokens")

    chunk_size = base_chunk_size
    if max_ctx:
        # 小上下文模型（如 512）不能被固定下限放大；按上下文窗口动态夹紧
        safe_max = max(128, int(max_ctx * 0.6))
        dynamic_floor = min(1000, max(200, int(max_ctx * 0.2)))
        chunk_size = min(chunk_size, safe_max, 2500)
        chunk_size = max(dynamic_floor, chunk_size)
        chunk_size = min(chunk_size, safe_max)
    else:
        # 如果没有max_tokens配置，使用默认的1200
        chunk_size = base_chunk_size

    # 重叠 15-25%
    chunk_overlap = max(base_overlap, int(chunk_size * 0.15))
    chunk_overlap = min(chunk_overlap, int(chunk_size * 0.25))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(100, int(chunk_size * 0.15))

    return chunk_size, chunk_overlap


def _distance_to_similarity(distance: float, is_ip: bool = True) -> float:
    """将 FAISS 距离/分数转换为 0-1 相似度

    Args:
        distance: FAISS 返回的距离或分数
        is_ip: True=Inner Product 分数（归一化后即余弦相似度），
               False=L2 距离（旧索引兼容）

    Returns:
        0-1 范围的相似度值
    """
    try:
        if is_ip:
            # IP 分数：归一化向量的内积 = 余弦相似度，范围 [-1, 1]
            # 映射到 [0, 1]
            return float(max(0.0, min(1.0, (distance + 1.0) / 2.0)))
        else:
            # L2 距离：旧索引兼容
            safe_distance = max(distance, 0.0)
            return float(1.0 / (1.0 + safe_distance))
    except Exception:
        return 0.0


def _extract_snippet_and_highlights(text: str, query: str, window: int = 100) -> Tuple[str, List[dict]]:
    """从文本中提取包含查询关键词的片段和高亮位置

    匹配策略（按优先级）：
    1. 完整短语匹配：尝试匹配整个查询字符串
    2. 单词级匹配：将查询拆分为单词逐个匹配
    """
    if not text:
        return "", []

    normalized_text = " ".join(text.split())
    lower_text = normalized_text.lower()
    query_lower = query.lower().strip()

    matches = []

    # 策略 1：完整短语匹配
    phrase_start = lower_text.find(query_lower)
    while phrase_start != -1:
        phrase_end = phrase_start + len(query_lower)
        matches.append((phrase_start, phrase_end, normalized_text[phrase_start:phrase_end]))
        phrase_start = lower_text.find(query_lower, phrase_end)

    # 策略 2：如果完整短语未匹配，回退到单词级匹配
    if not matches:
        terms = [t for t in re.split(r"[\s,;，。；、]+", query_lower) if t]
        for term in terms:
            start = lower_text.find(term)
            while start != -1:
                end = start + len(term)
                matches.append((start, end, normalized_text[start:end]))
                start = lower_text.find(term, end)

    matches.sort(key=lambda x: x[0])

    if matches:
        snippet_start = max(0, matches[0][0] - window)
        snippet_end = min(len(normalized_text), matches[0][1] + window)
    else:
        snippet_start = 0
        snippet_end = min(len(normalized_text), window * 2)

    snippet = normalized_text[snippet_start:snippet_end]
    highlights = []
    for start, end, _ in matches:
        if end <= snippet_start or start >= snippet_end:
            continue
        local_start = max(0, start - snippet_start)
        local_end = min(snippet_end - snippet_start, end - snippet_start)
        highlights.append({
            "start": int(local_start),
            "end": int(local_end),
            "text": normalized_text[start:end]
        })

    return snippet, highlights


def _build_page_index(pages: List[dict]) -> dict:
    """构建页面内容前缀索引，用于 O(1) 查找 chunk 所在页码

    对每个页面，按 80 字符窗口滑动提取前缀片段，构建 prefix -> page_num 映射。
    """
    if not pages:
        return {}
    pages = _annotate_pages_with_provenance(pages)
    index = {}
    for page in pages:
        content = page.get("content", "")
        page_num = page.get("page", 1)
        if not content:
            continue
        # 每隔 40 字符取一个 80 字符窗口作为索引键
        step = 40
        for i in range(0, max(1, len(content) - 79), step):
            key = content[i:i + 80]
            if key not in index:
                index[key] = page_num
    return index


def _normalize_positive_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
        return value if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = int(text)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _build_page_uid(page_num: int) -> str:
    return f"page:{int(page_num)}"


def _annotate_pages_with_provenance(pages: Optional[List[dict]]) -> List[dict]:
    """Return annotated page copies without mutating canonical document data."""
    if not isinstance(pages, list):
        return []
    annotated_pages = [dict(page) if isinstance(page, dict) else page for page in pages]
    for idx, page in enumerate(annotated_pages):
        if not isinstance(page, dict):
            continue
        page_num = (
            _normalize_positive_int(page.get("page"))
            or _normalize_positive_int(page.get("page_num"))
            or idx + 1
        )
        page["page"] = page_num
        if not isinstance(page.get("page_index"), int) or page.get("page_index") < 0:
            page["page_index"] = page_num - 1
        if not str(page.get("page_uid") or "").strip():
            page["page_uid"] = _build_page_uid(page_num)
    return annotated_pages


def _extract_page_candidates_from_metadata(metadata: Optional[dict]) -> List[int]:
    if not isinstance(metadata, dict):
        return []

    pages: List[int] = []

    def _append(value) -> None:
        page_num = _normalize_positive_int(value)
        if page_num and page_num not in pages:
            pages.append(page_num)

    _append(metadata.get("page"))
    _append(metadata.get("page_start"))
    _append(metadata.get("page_end"))

    page_index = metadata.get("page_index")
    if isinstance(page_index, int) and page_index >= 0:
        _append(page_index + 1)

    for key in ("page_range", "table_pages", "pages"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple)):
            for item in value:
                _append(item)

    table_page_indices = metadata.get("table_page_indices")
    if isinstance(table_page_indices, (list, tuple)):
        for item in table_page_indices:
            if isinstance(item, int) and item >= 0:
                _append(item + 1)

    return pages


def _resolve_primary_page_from_metadata(metadata: Optional[dict], fallback: int = 0) -> int:
    page_candidates = _extract_page_candidates_from_metadata(metadata)
    if page_candidates:
        return page_candidates[0]
    fallback_page = _normalize_positive_int(fallback)
    return fallback_page or 0


def _apply_page_provenance(item: dict, metadata: Optional[dict] = None) -> None:
    if not isinstance(item, dict):
        return

    page_num = _resolve_primary_page_from_metadata(metadata, fallback=item.get("page") or 0)
    if page_num > 0:
        item["page"] = page_num

    if not isinstance(item.get("page_index"), int) or item.get("page_index") < 0:
        metadata_page_index = metadata.get("page_index") if isinstance(metadata, dict) else None
        if isinstance(metadata_page_index, int) and metadata_page_index >= 0:
            item["page_index"] = metadata_page_index
        elif page_num > 0:
            item["page_index"] = page_num - 1

    page_uid = ""
    if isinstance(metadata, dict):
        page_uid = str(metadata.get("page_uid") or "").strip()
        if not page_uid:
            page_uids = metadata.get("table_page_uids") or metadata.get("page_uids")
            if isinstance(page_uids, (list, tuple)) and page_uids:
                first_uid = str(page_uids[0] or "").strip()
                if first_uid:
                    page_uid = first_uid
    if not page_uid and page_num > 0:
        page_uid = _build_page_uid(page_num)
    if page_uid and not str(item.get("page_uid") or "").strip():
        item["page_uid"] = page_uid


def _find_page_for_chunk(chunk_text: str, pages: List[dict], page_index: dict = None) -> int:
    """查找 chunk 所在的页码

    如果提供了 page_index（预构建的哈希索引），使用 O(1) 查找；
    否则回退到线性扫描。
    """
    if not pages:
        return 1

    prefix = chunk_text[:80]

    # 快速路径：使用预构建索引
    if page_index:
        if prefix in page_index:
            return page_index[prefix]
        # 尝试在索引中查找匹配的窗口
        prefix60 = chunk_text[:60].lower()
        for key, page_num in page_index.items():
            if prefix60 in key.lower():
                return page_num

    # 慢速路径：线性扫描
    for page in pages:
        content = page.get("content", "")
        if prefix in content:
            return page.get("page", 1)
        if chunk_text[:60].lower() in content.lower():
            return page.get("page", 1)
    return pages[0].get("page", 1)


def _get_document_title(doc_id: str) -> str:
    try:
        module = sys.modules.get("routes.document_routes")
        if module is None:
            return doc_id
        documents_store = getattr(module, "documents_store", {})
        doc = documents_store.get(doc_id) or {}
        filename = (doc.get("filename") or "").strip()
        if filename:
            return os.path.splitext(filename)[0]
    except Exception:
        pass
    return doc_id


def _guess_chunk_type(text: str) -> str:
    sample = (text or "").strip()
    if not sample:
        return "text"
    if _is_likely_table(sample):
        return "table"
    first_line = sample.splitlines()[0].strip()
    if re.match(r"^(?:Figure|Fig\.?|图|Table|TABLE|表)\s*\d+[a-zA-Z]?\b", first_line, re.IGNORECASE):
        return "caption"
    if "$$" in sample or "\\[" in sample or "\\]" in sample:
        return "formula"
    math_hits = len(re.findall(r"(?:\\[a-zA-Z]+|[=<>±∑∫μσλβγδθ]|x_t|y_t|z_t)", sample))
    if math_hits >= 3 and len(sample) <= 1200:
        return "formula"
    return "text"


def _normalize_structural_metadata(item: dict) -> dict:
    chunk_text = item.get("raw_chunk_text") or item.get("child_chunk") or item.get("chunk", "")
    chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip()
    if not chunk_type:
        chunk_type = _guess_chunk_type(chunk_text)
    item["chunk_type"] = chunk_type or "text"
    item["block_type"] = (item.get("block_type") or item["chunk_type"] or "text").strip() or "text"

    section_path = (item.get("section_path") or item.get("chunk_heading") or "").strip()
    item["section_path"] = section_path
    item["chunk_heading"] = (item.get("chunk_heading") or section_path).strip()

    page = item.get("page", 0)
    try:
        page_int = int(page)
    except (TypeError, ValueError):
        page_int = 0
    item["page"] = page_int if page_int > 0 else 0
    return item


def _normalize_section_heading(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _build_chunk_idx_to_group_map(group_chunk_map: Optional[dict]) -> dict:
    mapping = {}
    if not group_chunk_map:
        return mapping
    for gid, indices in group_chunk_map.items():
        if not isinstance(indices, list):
            continue
        for idx in indices:
            if isinstance(idx, int) and idx >= 0 and idx not in mapping:
                mapping[idx] = gid
    return mapping


def _resolve_result_chunk_index(item: dict, chunk_text_to_idx: dict) -> Optional[int]:
    chunk_id = item.get("chunk_id")
    if isinstance(chunk_id, int) and chunk_id >= 0:
        return chunk_id
    child_chunk_id = item.get("child_chunk_id")
    if isinstance(child_chunk_id, int) and child_chunk_id >= 0:
        return child_chunk_id
    child_chunk = item.get("child_chunk", "")
    if child_chunk:
        child_idx = chunk_text_to_idx.get(child_chunk)
        if child_idx is not None:
            return child_idx
    chunk_text = item.get("chunk", "")
    return chunk_text_to_idx.get(chunk_text)


def _get_numeric_table_boundary_text(item: dict) -> str:
    """优先保留表行边界文本，避免宽切片把相邻行混进来。"""
    for key in (
        "table_row_boundary_text",
        "table_row_raw_text",
        "raw_chunk_text",
        "row_text",
        "chunk",
    ):
        value = item.get(key)
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        if normalized:
            return normalized
    return ""


def _build_evidence_unit_text(item: dict, chunks: List[str], parent_chunks: List[str]) -> str:
    chunk_type = (item.get("block_type") or item.get("chunk_type") or "text").strip() or "text"
    if chunk_type == "table_row":
        title = (item.get("doc_title") or item.get("doc_id") or "").strip()[:200]
        section = (item.get("section_path") or item.get("chunk_heading") or "未标注章节").strip()[:200]
        page = item.get("page") or 0
        table_caption = (item.get("table_caption") or item.get("table_id") or "").strip()[:300]
        table_header = (item.get("table_header") or "").strip()[:400]
        row_text = _get_numeric_table_boundary_text(item)[:1200]
        row_id = (item.get("row_id") or "").strip()[:120]
        return (
            f"[Title]\n{title}\n\n"
            f"[Section]\n{section}\n\n"
            f"[Table]\n{table_caption}\n\n"
            f"[Header]\n{table_header}\n\n"
            f"[Row]\n{row_text}\n\n"
            f"[Hints]\npage={page}; type=table_row; row_id={row_id}"
        )

    chunk_id = item.get("chunk_id")
    parent_id = item.get("parent_id")
    local_passage = (item.get("raw_chunk_text") or item.get("child_chunk") or item.get("chunk") or "").strip()
    if not local_passage and isinstance(chunk_id, int) and 0 <= chunk_id < len(chunks):
        local_passage = (chunks[chunk_id] or "").strip()

    context_text = ""
    if isinstance(parent_id, int) and 0 <= parent_id < len(parent_chunks):
        context_text = (parent_chunks[parent_id] or "").strip()
        if context_text == local_passage:
            context_text = ""
    elif isinstance(chunk_id, int) and 0 <= chunk_id < len(chunks):
        start = max(0, chunk_id - 1)
        end = min(len(chunks), chunk_id + 2)
        context_text = "\n\n".join(
            chunks[idx].strip()
            for idx in range(start, end)
            if idx != chunk_id and chunks[idx].strip()
        )

    title = (item.get("doc_title") or item.get("doc_id") or "").strip()[:200]
    section = (item.get("section_path") or item.get("chunk_heading") or "未标注章节").strip()[:200]
    page = item.get("page") or 0
    return (
        f"[Title]\n{title}\n\n"
        f"[Section]\n{section}\n\n"
        f"[Passage]\n{local_passage[:1200]}\n\n"
        f"[Context]\n{context_text[:1200]}\n\n"
        f"[Hints]\npage={page}; type={chunk_type}"
    )


def _should_use_multi_row_bundle_context(query: str, hints: dict[str, List[str]]) -> bool:
    if not query:
        return False
    if hints.get("comparison") or _is_numeric_table_explicit_comparator_query(query, hints):
        return True
    if re.search(r"(?:第二好(?:的)?|第二佳|第二名|次优|次佳|second[- ]best|runner[- ]up|nearest competitor)", query, re.IGNORECASE):
        return True
    target_columns = [value for value in hints.get("columns", []) if str(value or "").strip()]
    return len(target_columns) >= 3 and not _is_numeric_table_winner_style_query(query, hints)


_NUMERIC_TABLE_EXPLICIT_COMPARATOR_RE = re.compile(
    r"(?:比|高多少|差多少|提升|百分点|improv(?:e|ement|ed|ing)?|gain|delta|vs|versus|compare)",
    re.IGNORECASE,
)
_NUMERIC_TABLE_SECOND_BEST_RE = re.compile(
    r"(?:第二好(?:的)?|第二佳|第二名|次优|次佳|second[- ]best|runner[- ]up|nearest competitor)",
    re.IGNORECASE,
)


def _extract_numeric_table_bundle_row_units(item: dict, query: str) -> List[dict]:
    if not isinstance(item, dict) or not query:
        return []

    chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    if chunk_type == "table_row":
        return []

    evidence_units = item.get("evidence_units")
    row_units = [
        dict(unit)
        for unit in evidence_units
        if isinstance(unit, dict)
        and str(unit.get("evidence_unit_type") or "").strip().lower() == "table_row"
    ] if isinstance(evidence_units, list) else []
    if not row_units and item.get("structured_table_bundle"):
        row_units = _extract_structured_bundle_body_rows(item)
    if not row_units:
        row_units = _extract_markdown_table_rows((item.get("chunk") or item.get("raw_chunk_text") or "").strip())
    if not row_units:
        row_units = _extract_plain_table_rows((item.get("chunk") or item.get("raw_chunk_text") or "").strip(), _query_rewriter_singleton.extract_numeric_table_hints(query), query)
    if not row_units:
        return []

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    explicit_comparator_mode = _is_numeric_table_explicit_comparator_query(query, hints)
    selected_units: List[dict] = []
    seen_row_keys: set[str] = set()
    for unit in row_units:
        merged_unit = dict(unit)
        if not merged_unit.get("table_caption") and item.get("table_caption"):
            merged_unit["table_caption"] = item.get("table_caption")
        if not merged_unit.get("table_id") and item.get("table_id"):
            merged_unit["table_id"] = item.get("table_id")
        if not merged_unit.get("table_header") and item.get("table_header"):
            merged_unit["table_header"] = item.get("table_header")

        if _is_headerish_numeric_table_row(merged_unit):
            continue

        row_id = str(merged_unit.get("row_id") or "").strip()
        if row_id and _is_composite_numeric_row_id(row_id) and not explicit_comparator_mode:
            continue

        row_text = re.sub(
            r"\s+",
            " ",
            str(
                merged_unit.get("content")
                or merged_unit.get("row_text")
                or merged_unit.get("row_numbers")
                or "",
            ),
        ).strip()
        if not row_text or not _extract_numeric_value_tokens(row_text):
            continue
        if _is_composite_numeric_row_id(row_id) and not explicit_comparator_mode:
            continue

        row_key = _normalize_numeric_table_method_token(row_id) or row_text.lower()
        if row_key in seen_row_keys:
            continue
        seen_row_keys.add(row_key)
        selected_units.append(merged_unit)

    return selected_units


def _build_multi_row_bundle_context_text(item: dict, query: str) -> str:
    if not isinstance(item, dict) or not query:
        return ""

    chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    if chunk_type == "table_row":
        return ""

    row_units = _extract_numeric_table_bundle_row_units(item, query)
    if not row_units:
        return ""

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    if not _should_use_multi_row_bundle_context(query, hints):
        return ""

    parts: list[str] = []
    for value in (
        item.get("numeric_table_exact_context_caption", "") or item.get("table_caption", ""),
        item.get("numeric_table_exact_context_header", "") or item.get("table_header", ""),
    ):
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        if normalized and normalized not in parts:
            parts.append(normalized)

    for unit in row_units:
        row_text = re.sub(
            r"\s+",
            " ",
            str(
                unit.get("content")
                or unit.get("row_text")
                or unit.get("row_numbers")
                or ""
            ),
        ).strip()
        parts.append(row_text)

    if len(parts) <= 2:
        return ""
    return "\n".join(parts)


def _maybe_attach_numeric_table_bundle_exact_context(item: dict, query: str) -> None:
    if not should_apply_numeric_table_specialization():
        return
    if not isinstance(item, dict) or not query:
        return
    if str(item.get("numeric_table_exact_context_row_text") or "").strip():
        return

    chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    if chunk_type == "table_row":
        return

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    comparison_query = bool(hints.get("comparison"))
    target_columns = {
        _normalize_numeric_column_name(value)
        for value in hints.get("columns", [])
        if value
    }
    target_tables = {
        value.lower()
        for value in hints.get("table_labels", [])
        if value
    } | {
        table_id.lower()
        for _, _, table_id in _extract_table_mentions(query)
        if table_id
    }
    preferred_sort_column = _preferred_numeric_table_sort_column(query, hints)
    explicit_comparator_mode = _is_numeric_table_explicit_comparator_query(query, hints)
    second_best_mode = bool(_NUMERIC_TABLE_SECOND_BEST_RE.search(query or ""))
    winner_only_mode = bool(
        not explicit_comparator_mode
        and not second_best_mode
        and (preferred_sort_column or target_columns or target_tables)
    )
    if not winner_only_mode:
        return

    row_units = _extract_numeric_table_bundle_row_units(item, query)
    if not row_units:
        return

    best_unit = None
    best_focused = None
    best_score = float("-inf")
    for unit in row_units:
        merged_unit = dict(unit)
        if not merged_unit.get("table_caption") and item.get("table_caption"):
            merged_unit["table_caption"] = item.get("table_caption")
        if not merged_unit.get("table_id") and item.get("table_id"):
            merged_unit["table_id"] = item.get("table_id")
        if not merged_unit.get("table_header") and item.get("table_header"):
            merged_unit["table_header"] = item.get("table_header")
        if _is_headerish_numeric_table_row(merged_unit):
            continue
        if target_tables and not _has_explicit_numeric_table_match(merged_unit, target_tables):
            continue

        focused = _build_query_focused_table_row(merged_unit, hints)
        focused_text = str(focused.get("text") or "").strip()
        column_coverage = int(focused.get("column_coverage", 0) or 0)
        if not focused_text or (target_columns and column_coverage <= 0):
            continue

        lexical = _compute_lexical_evidence_score(
            query,
            f"{merged_unit.get('table_caption', '')} {merged_unit.get('table_header', '')} {focused_text}",
        )
        score = (
            float(_numeric_table_sort_bonus(merged_unit, query, hints) or 0.0)
            + lexical
            + min(column_coverage, 4) * 0.08
        )
        if score > best_score:
            best_unit = merged_unit
            best_focused = focused
            best_score = score

    if best_unit is None or best_focused is None:
        return

    exact_row_text = str(best_focused.get("text") or "").strip()
    if not exact_row_text:
        return

    item["numeric_table_exact_context_row_text"] = exact_row_text
    item["numeric_table_exact_context_caption"] = (
        best_unit.get("table_caption")
        or item.get("table_caption")
        or ""
    )
    item["numeric_table_exact_context_header"] = (
        best_unit.get("table_header")
        or item.get("table_header")
        or ""
    )
    item["numeric_table_force_exact_context"] = True


def _build_query_focused_numeric_table_exact_context_text(item: dict, query: str) -> str:
    """把 table_row 的 exact row 投影成 query-focused 的最终上下文文本。"""
    if not isinstance(item, dict) or not query:
        return ""

    chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    if chunk_type != "table_row":
        return ""

    if not (
        item.get("numeric_table_exact_context_row_text")
        or item.get("table_row_evidence")
        or item.get("table_row_slice_kind") == "exact"
        or item.get("cell_evidence_units")
        or item.get("table_row_raw_text")
        or item.get("row_text")
        or item.get("chunk")
        or item.get("raw_chunk_text")
    ):
        return ""

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    if not (
        _is_numeric_table_explicit_comparator_query(query, hints)
        or _is_numeric_table_bundle_query(query, hints)
        or _is_numeric_table_row_band_query(query, hints)
        or _is_numeric_table_winner_style_query(query, hints)
        or "numeric_table" in (_analyze_evidence_need(query) or [])
    ):
        return ""

    row_text = re.sub(
        r"\s+",
        " ",
        str(
            item.get("numeric_table_exact_context_row_text")
            or _get_numeric_table_boundary_text(item)
            or item.get("table_row_raw_text")
            or item.get("row_text")
            or item.get("chunk")
            or item.get("raw_chunk_text")
            or ""
        ),
    ).strip()
    if not row_text:
        return ""

    row_id = str(item.get("row_id") or "").strip()
    if not row_id:
        row_id = re.split(r"\s+", row_text, maxsplit=1)[0].strip(" ,;")

    focused = _build_query_focused_table_row(
        {
            "row_id": row_id,
            "row_text": row_text,
            "row_numbers": _strip_leading_numeric_table_row_id(row_text, row_id) or row_text,
            "table_caption": item.get("numeric_table_exact_context_caption")
            or item.get("table_caption")
            or item.get("table_id")
            or "",
            "table_id": item.get("table_id") or "",
            "table_header": item.get("numeric_table_exact_context_header") or item.get("table_header") or "",
            "table_focus_columns": list(item.get("table_focus_columns") or []),
        },
        hints,
    )
    focused_text = re.sub(r"\s+", " ", str(focused.get("text") or "")).strip()
    if not focused_text:
        return ""

    caption = re.sub(
        r"\s+",
        " ",
        str(
            item.get("numeric_table_exact_context_caption")
            or item.get("table_caption")
            or item.get("table_id")
            or ""
        ),
    ).strip()
    if caption and caption not in focused_text:
        return "\n".join([caption, focused_text])
    return focused_text


def _build_context_text_for_result(item: dict, query: str = "") -> str:
    """构造真正进入最终上下文的文本；table_row 必须带上 caption 和 header。"""
    _maybe_attach_numeric_table_bundle_exact_context(item, query)
    chunk_text = (item.get("chunk") or item.get("raw_chunk_text") or "").strip()
    expanded_text = str(item.get("expanded_chunk") or "").strip()
    chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    exact_row_text = re.sub(
        r"\s+",
        " ",
        str(item.get("numeric_table_exact_context_row_text") or "").strip(),
    ).strip()
    if not exact_row_text and chunk_type == "table_row":
        exact_row_text = re.sub(
            r"\s+",
            " ",
            str(
                _get_numeric_table_boundary_text(item)
                or item.get("table_row_raw_text")
                or item.get("row_text")
                or item.get("chunk")
                or item.get("raw_chunk_text")
                or ""
            ),
        ).strip()
    has_typed_table_evidence = bool(
        chunk_type in {"table_row", "table", "caption", "table_cell"}
        or item.get("table_row_evidence")
        or item.get("table_row_slice_kind") == "exact"
        or item.get("numeric_table_exact_context_row_text")
        or item.get("evidence_units")
        or item.get("cell_evidence_units")
    )
    if exact_row_text and has_typed_table_evidence:
        focused_exact_context_text = _build_query_focused_numeric_table_exact_context_text(item, query)
        if focused_exact_context_text:
            return focused_exact_context_text
        parts: list[str] = []
        for value in (
            item.get("numeric_table_exact_context_caption", "") or item.get("table_caption", ""),
            item.get("numeric_table_exact_context_header", "") or item.get("table_header", ""),
            exact_row_text,
        ):
            normalized = re.sub(r"\s+", " ", str(value or "")).strip()
            if not normalized or normalized in parts:
                continue
            parts.append(normalized)
        if parts:
            exact_context = "\n".join(parts)
            if item.get("numeric_table_force_exact_context"):
                return exact_context
    bundle_context_text = _build_multi_row_bundle_context_text(item, query)
    if bundle_context_text:
        return bundle_context_text
    if exact_row_text and has_typed_table_evidence:
        parts: list[str] = []
        for value in (
            item.get("numeric_table_exact_context_caption", "") or item.get("table_caption", ""),
            item.get("numeric_table_exact_context_header", "") or item.get("table_header", ""),
            exact_row_text,
        ):
            normalized = re.sub(r"\s+", " ", str(value or "")).strip()
            if not normalized or normalized in parts:
                continue
            parts.append(normalized)
        if parts:
            return "\n".join(parts)
    if chunk_type != "table_row":
        if has_typed_table_evidence:
            return chunk_text
        if chunk_type == "formula" or looks_formula_like(query) or looks_formula_like(chunk_text):
            return build_formula_alias_text(chunk_text)
        return expanded_text or chunk_text

    parts: list[str] = []
    for value in (
        item.get("table_caption", ""),
        item.get("table_header", ""),
        _get_numeric_table_boundary_text(item),
    ):
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        if not normalized:
            continue
        if normalized in parts:
            continue
        parts.append(normalized)
    context_text = "\n".join(parts) if parts else chunk_text
    if chunk_type == "formula" or looks_formula_like(query) or looks_formula_like(context_text):
        return build_formula_alias_text(context_text)
    return context_text


_RUNTIME_VISUAL_PROVENANCE_FIELDS = (
    "visual_evidence_id",
    "visual_enhancement",
    "visual_source",
    "visual_supplement_revision",
    "figure_id",
    "bbox",
    "figure_bbox",
    "visual_model",
    "runtime_visual_overlay",
)


def _runtime_visual_provenance(item: dict) -> dict:
    """Return copy-safe visual provenance for a request-local overlay result."""
    if not isinstance(item, dict) or not (
        item.get("runtime_visual_overlay") or item.get("visual_enhancement")
    ):
        return {}

    provenance: dict = {}
    for key in _RUNTIME_VISUAL_PROVENANCE_FIELDS:
        value = item.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            provenance[key] = dict(value)
        elif isinstance(value, list):
            provenance[key] = list(value)
        else:
            provenance[key] = value

    evidence_id = str(
        provenance.get("visual_evidence_id") or item.get("visual_evidence_id") or ""
    ).strip()
    if evidence_id:
        provenance["visual_evidence_id"] = evidence_id
    provenance.setdefault("runtime_visual_overlay", True)
    provenance.setdefault("visual_enhancement", True)
    provenance.setdefault("visual_source", "visual_vlm")
    return provenance


def _copy_runtime_visual_provenance(item: dict, target: dict) -> dict:
    """Copy visual provenance without changing non-visual retrieval records."""
    if isinstance(target, dict):
        target.update(_runtime_visual_provenance(item))
    return target


def _visual_overlay_group_id(item: dict, fallback: str = "") -> str:
    if not isinstance(item, dict):
        return fallback
    evidence_id = str(item.get("visual_evidence_id") or "").strip()
    if evidence_id and (item.get("runtime_visual_overlay") or item.get("visual_enhancement")):
        return f"visual-{evidence_id}"
    return str(item.get("group_id") or fallback)


def _append_runtime_visual_overlay_group_context(
    context_string: str,
    retrieval_meta: dict,
    results: List[dict],
    *,
    doc_id: str,
    query: str,
    limit: int = 1,
) -> Tuple[str, dict]:
    """Append one qualified VLM observation after semantic-group assembly.

    Semantic groups are built from the persistent FAISS generation and cannot
    own request-local VLM observations. Keeping the overlay as a standalone
    numbered context block avoids contaminating groups while allowing the LLM
    and final citation path to see the same committed evidence.
    """
    if not isinstance(retrieval_meta, dict) or not results or limit <= 0:
        return context_string, retrieval_meta

    candidates = [
        item
        for item in results
        if isinstance(item, dict) and item.get("runtime_visual_overlay")
    ]
    if not candidates:
        return context_string, retrieval_meta

    def _score(item: dict) -> float:
        for key in ("similarity", "combined_score", "rerank_score", "score"):
            try:
                return float(item.get(key))
            except (TypeError, ValueError):
                continue
        return float("-inf")

    citations = [
        dict(item)
        for item in (retrieval_meta.get("citations") or [])
        if isinstance(item, dict)
    ]
    segments = [
        dict(item)
        for item in (retrieval_meta.get("_context_segments") or [])
        if isinstance(item, dict)
    ]
    existing_visual_ids = {
        str(item.get("visual_evidence_id") or "").strip()
        for item in [*citations, *segments]
        if isinstance(item, dict) and str(item.get("visual_evidence_id") or "").strip()
    }
    refs = []
    for item in [*citations, *segments]:
        try:
            refs.append(int(item.get("ref") or 0))
        except (TypeError, ValueError):
            continue
    next_ref = max(refs, default=0) + 1

    appended = 0
    context_parts = [str(context_string or "").rstrip()] if str(context_string or "").strip() else []
    for item in sorted(candidates, key=_score, reverse=True):
        evidence_id = str(item.get("visual_evidence_id") or "").strip()
        if not evidence_id or evidence_id in existing_visual_ids:
            continue
        text = (_build_context_text_for_result(item, query=query) or "").strip()
        if not text:
            continue
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        page_range = [page, page] if page > 0 else []
        page_label = str(page) if page > 0 else "未知"
        revision = str(item.get("visual_supplement_revision") or "").strip()
        context_id = str(item.get("context_id") or f"visual:{evidence_id}").strip()
        citation = _copy_runtime_visual_provenance(item, {
            "ref": next_ref,
            "evidence_id": str(item.get("evidence_id") or f"visual:{revision or 'current'}:{evidence_id}"),
            "context_id": context_id,
            "group_id": _visual_overlay_group_id(item, f"visual-{evidence_id}"),
            "block_id": item.get("block_id") or evidence_id,
            "chunk_id": item.get("chunk_id"),
            "page_range": page_range,
            "source_text": text,
            "display_text": text,
            "highlight_text": str(item.get("snippet") or text[:200]).strip(),
            "_full_text": text,
            "chunk_type": item.get("chunk_type") or "visual_evidence",
            "block_type": item.get("block_type") or "caption",
            "retrieval_type": "visual_overlay",
            "alignment_status": "candidate",
            "score": item.get("score"),
            "similarity": item.get("similarity"),
        })
        segment = _copy_runtime_visual_provenance(item, {
            "ref": next_ref,
            "evidence_id": citation["evidence_id"],
            "context_id": context_id,
            "group_id": citation["group_id"],
            "block_id": citation["block_id"],
            "chunk_id": citation.get("chunk_id"),
            "doc_id": doc_id,
            "text": text,
            "page_range": page_range,
            "modality": "visual",
            "chunk_type": citation["chunk_type"],
            "block_type": citation["block_type"],
            "retrieval_type": "visual_overlay",
            "score": item.get("similarity", item.get("score", 0.0)),
        })
        context_parts.append(
            f"[{next_ref}]【图表视觉补充 | 页码: {page_label}】\n内容:\n{text}"
        )
        citations.append(citation)
        segments.append(segment)
        existing_visual_ids.add(evidence_id)
        appended += 1
        next_ref += 1
        if appended >= limit:
            break

    if not appended:
        return context_string, retrieval_meta

    raw_chunks = retrieval_meta.get("_chunks")
    if not isinstance(raw_chunks, list):
        raw_chunks = []
    for item in candidates:
        evidence_id = str(item.get("visual_evidence_id") or "").strip()
        if not evidence_id or evidence_id not in existing_visual_ids:
            continue
        matched = False
        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue
            if str(chunk.get("visual_evidence_id") or "").strip() == evidence_id:
                _copy_runtime_visual_provenance(item, chunk)
                matched = True
                break
        if not matched:
            raw_chunks.append(_copy_runtime_visual_provenance(item, {
                "text": item.get("chunk", ""),
                "page": item.get("page", 0),
                "group_id": _visual_overlay_group_id(item, f"visual-{evidence_id}"),
                "context_id": item.get("context_id") or f"visual:{evidence_id}",
                "evidence_id": item.get("evidence_id") or evidence_id,
                "block_id": item.get("block_id") or evidence_id,
                "chunk_id": item.get("chunk_id"),
                "doc_id": doc_id,
                "chunk_type": item.get("chunk_type") or "visual_evidence",
                "block_type": item.get("block_type") or "caption",
            }))

    retrieval_meta["citations"] = citations
    retrieval_meta["_context_segments"] = segments
    retrieval_meta["_chunks"] = raw_chunks
    retrieval_meta["visual_overlay_context_count"] = int(
        retrieval_meta.get("visual_overlay_context_count") or 0
    ) + appended
    return "\n\n".join(context_parts), retrieval_meta


def _is_delta_per_sample_metric_query(query: str, hints: Optional[dict[str, List[str]]] = None) -> bool:
    hints = hints or _query_rewriter_singleton.extract_numeric_table_hints(query)
    query_columns = {
        _normalize_numeric_column_name(value)
        for value in hints.get("columns", [])
        if value
    }
    if "ΔAcc/||D_gen||" in query_columns:
        return True
    sample = (query or "").lower()
    return any(token in sample for token in ("每样本", "per sample", "average gain", "增益", "提升最大"))


def _numeric_table_delta_column_coverage(item: dict, query: str, hints: dict[str, List[str]]) -> int:
    """判断表格证据是否真正覆盖每样本提升三列，避免混入相邻表的 Acc/All 数值。"""
    required = {"||D_gen||", "Acc", "ΔAcc/||D_gen||"}

    def _columns_from_values(values) -> set[str]:
        return {
            _normalize_numeric_column_name(value)
            for value in (values or [])
            if value
        }

    candidates: list[dict] = []
    if isinstance(item, dict):
        candidates.append(item)
        candidates.extend(_extract_numeric_table_bundle_row_units(item, query))

    best = 0
    for candidate in candidates:
        header_columns = set(_extract_table_header_columns(candidate.get("table_header", "") or ""))
        focus_columns = _columns_from_values(candidate.get("table_focus_columns"))
        focused = _build_query_focused_table_row(candidate, hints)
        resolved_columns = _columns_from_values(focused.get("resolved_columns", []))
        column_map = set((focused.get("column_map") or {}).keys())
        text = " ".join(
            str(candidate.get(key) or "")
            for key in (
                "numeric_table_exact_context_row_text",
                "table_row_boundary_text",
                "table_row_raw_text",
                "row_text",
                "content",
                "chunk",
                "raw_chunk_text",
            )
        )
        text_lower = text.lower()
        text_compact = re.sub(r"\s+", "", text_lower)
        text_columns = {
            column for column, pattern in {
                "||D_gen||": r"(?:\|\||∥)\s*d\s*[_ ]?gen\s*(?:\|\||∥)|\bd\s*[_ ]?gen\b",
                "Acc": r"\bacc\.?(?:uracy)?\b|准确率|分类准确率",
                "ΔAcc/||D_gen||": r"[δ∆Δ]\s*acc\s*/\s*(?:\|\||∥)\s*d\s*[_ ]?gen\s*(?:\|\||∥)|delta\s*acc\s*/\s*d\s*gen",
            }.items()
            if re.search(pattern, text_lower, re.IGNORECASE)
            or (column == "ΔAcc/||D_gen||" and any(token in text_compact for token in ("Δacc/||d_gen||", "∆acc/∥dgen∥")))
        }
        present = header_columns | focus_columns | resolved_columns | column_map | text_columns
        best = max(best, len(required & present))
        if best >= len(required):
            return best
    return best


def _resolve_numeric_table_context_table_id(item: dict) -> str:
    explicit_table = _extract_table_id(
        (
            item.get("table_id")
            or item.get("numeric_table_exact_context_caption")
            or item.get("table_caption")
            or ""
        ).strip()
    )
    if explicit_table:
        return explicit_table.lower()

    evidence_text = (
        _build_numeric_table_evidence_text(item)
        or item.get("raw_chunk_text")
        or item.get("chunk")
        or ""
    )
    explicit_table = _extract_table_id(evidence_text)
    return explicit_table.lower() if explicit_table else ""


def _classify_numeric_table_context_role(
    item: dict,
    query: str,
    hints: dict[str, List[str]],
    *,
    primary_table_id: str = "",
    primary_anchor_pages: Optional[set[int]] = None,
) -> str:
    primary_anchor_pages = primary_anchor_pages or set()
    chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    context_table_id = _resolve_numeric_table_context_table_id(item)
    evidence_text = (
        _build_numeric_table_evidence_text(item)
        or item.get("raw_chunk_text")
        or item.get("chunk")
        or ""
    )
    evidence_lower = evidence_text.lower()
    page = int(item.get("page") or 0)
    cost_query = _is_numeric_table_cost_query(query)
    comparator_query = bool(hints.get("comparison")) or _is_numeric_table_row_band_query(query, hints)

    has_exact_row = bool(
        item.get("numeric_table_exact_context_row_text")
        or item.get("table_row_evidence")
        or item.get("table_row_slice_kind") == "exact"
        or item.get("cell_evidence_units")
        or (
            chunk_type in {"table_row", "table_cell"}
            and (
                _get_numeric_table_boundary_text(item)
                or item.get("table_row_raw_text")
                or item.get("cell_evidence_units")
            )
        )
        or (
            item.get("structured_table_bundle")
            and _build_multi_row_bundle_context_text(item, query)
        )
    )

    if cost_query:
        if _is_numeric_table_cost_anchor_text(item):
            return "anchor"
        if primary_anchor_pages and page in primary_anchor_pages:
            if (
                item.get("numeric_table_keep_support")
                or chunk_type in {"table", "caption"}
                or any(
                    token in evidence_lower
                    for token in ("cost", "overhead", "training", "inference", "hours", "days")
                )
            ):
                return "focus"
        return "background"

    if has_exact_row:
        if primary_table_id and context_table_id and context_table_id != primary_table_id:
            return "background"
        return "anchor"

    if primary_table_id and context_table_id == primary_table_id:
        if (
            item.get("structured_table_bundle")
            or chunk_type in {"table", "caption"}
            or _looks_like_numeric_table_support(evidence_text, chunk_type)
            or (item.get("numeric_table_keep_support") and chunk_type in {"table_row", "table_cell"})
        ):
            return "focus"

    if comparator_query and primary_anchor_pages and page in primary_anchor_pages:
        if item.get("structured_table_bundle") or _looks_like_numeric_table_support(evidence_text, chunk_type):
            return "focus"

    return "background"


def _cleanup_numeric_table_context_entries(
    fallback_entries: List[tuple[dict, str]],
    query: str,
) -> List[dict]:
    if not fallback_entries:
        return []

    if "numeric_table" not in (_analyze_evidence_need(query) or []):
        return [
            {"item": item, "text": text, "context_role": "background"}
            for item, text in fallback_entries
        ]

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    cost_query = _is_numeric_table_cost_query(query)
    delta_metric_query = _is_delta_per_sample_metric_query(query, hints)
    second_best_mode = bool(_NUMERIC_TABLE_SECOND_BEST_RE.search(query or ""))
    comparator_query = bool(hints.get("comparison")) or _is_numeric_table_row_band_query(query, hints) or second_best_mode
    explicit_comparator_mode = _is_numeric_table_explicit_comparator_query(query, hints)
    winner_style_query = bool(_preferred_numeric_table_sort_column(query, hints))
    keep_row_projection = comparator_query or explicit_comparator_mode or winner_style_query
    ordered_target_method_keys = [
        _normalize_numeric_table_method_token(value)
        for value in hints.get("methods", [])
        if value and _normalize_numeric_table_method_token(value)
    ]
    target_method_keys = {
        value for value in ordered_target_method_keys if value not in _NUMERIC_TABLE_METHOD_STOPWORDS
    }

    def _chunk_type(item: dict) -> str:
        return (item.get("chunk_type") or item.get("block_type") or "").strip().lower()

    def _table_key(entry: dict) -> str:
        return entry["table_id"] or f"page:{entry['page']}"

    def _exact_row_text(item: dict) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(
                item.get("numeric_table_exact_context_row_text")
                or _get_numeric_table_boundary_text(item)
                or item.get("table_row_raw_text")
                or item.get("row_text")
                or item.get("chunk")
                or item.get("raw_chunk_text")
                or ""
            ),
        ).strip()

    def _entry_has_exact_row(item: dict) -> bool:
        chunk_type = _chunk_type(item)
        return bool(
            item.get("numeric_table_exact_context_row_text")
            or item.get("table_row_evidence")
            or item.get("table_row_slice_kind") == "exact"
            or item.get("cell_evidence_units")
            or (
                chunk_type in {"table_row", "table_cell"}
                and (_get_numeric_table_boundary_text(item) or item.get("table_row_raw_text"))
            )
        )

    def _entry_anchor_rank(entry: dict, idx: int) -> tuple[int, int, int, float, float, int]:
        item = entry["item"]
        delta_coverage = _numeric_table_delta_column_coverage(item, query, hints) if delta_metric_query else 0
        return (
            1 if delta_metric_query and delta_coverage >= 3 else 0,
            1 if _chunk_type(item) == "table_row" and _entry_has_exact_row(item) else 0,
            1 if _entry_has_exact_row(item) else 0,
            1 if _build_multi_row_bundle_context_text(item, query) else 0,
            float(item.get("numeric_table_priority", 0.0) or 0.0),
            float(item.get("similarity", 0.0) or 0.0),
            -idx,
        )

    def _synthesize_same_table_bundle_text(entries: List[tuple[int, dict]]) -> str:
        if not entries:
            return ""

        for _idx, entry in entries:
            existing_bundle_text = _build_multi_row_bundle_context_text(entry["item"], query)
            if existing_bundle_text:
                return existing_bundle_text

        best_caption = ""
        best_header = ""
        row_payloads: dict[str, tuple[tuple, str]] = {}

        def _remember_caption_and_header(item: dict) -> None:
            nonlocal best_caption, best_header
            if not best_caption:
                best_caption = (
                    item.get("numeric_table_exact_context_caption")
                    or item.get("table_caption")
                    or best_caption
                )
            if not best_header:
                best_header = (
                    item.get("numeric_table_exact_context_header")
                    or item.get("table_header")
                    or best_header
                )

        def _store_row(
            row_key: str,
            row_text: str,
            row_rank: tuple,
        ) -> None:
            if not row_key or not row_text:
                return
            current = row_payloads.get(row_key)
            if current is None or row_rank > current[0]:
                row_payloads[row_key] = (row_rank, row_text)

        for idx, entry in entries:
            item = entry["item"]
            _remember_caption_and_header(item)

            row_units = []
            if _chunk_type(item) != "table_row":
                row_units = _extract_numeric_table_bundle_row_units(item, query)
            if row_units:
                for unit in row_units:
                    row_id = str(unit.get("row_id") or "").strip()
                    row_key = _normalize_numeric_table_method_token(row_id)
                    if explicit_comparator_mode:
                        if not row_key or row_key not in target_method_keys:
                            continue
                    elif row_id and _is_composite_numeric_row_id(row_id) and not comparator_query:
                        continue
                    focused_row = _build_query_focused_table_row(unit, hints)
                    row_text = re.sub(
                        r"\s+",
                        " ",
                        str(
                            focused_row.get("text")
                            or unit.get("content")
                            or unit.get("row_text")
                            or unit.get("row_numbers")
                            or ""
                        ),
                    ).strip()
                    if not row_text:
                        continue
                    row_rank = (
                        1 if row_key in target_method_keys else 0,
                        int(focused_row.get("column_coverage", 0) or 0),
                        float(_numeric_table_sort_bonus(unit, query, hints) or 0.0),
                        float(item.get("numeric_table_priority", 0.0) or 0.0),
                        float(item.get("similarity", 0.0) or 0.0),
                        -idx,
                    )
                    _store_row(row_key or row_text.lower(), row_text, row_rank)
                continue

            if not _entry_has_exact_row(item):
                continue
            row_id = str(item.get("row_id") or "").strip()
            row_key = _normalize_numeric_table_method_token(row_id)
            if explicit_comparator_mode:
                if not row_key or row_key not in target_method_keys:
                    continue
            elif row_id and _is_composite_numeric_row_id(row_id):
                continue
            row_text = _exact_row_text(item)
            if not row_text:
                continue
            row_rank = (
                1 if row_key in target_method_keys else 0,
                len(target_method_keys) if row_key in target_method_keys else 0,
                float(item.get("numeric_table_priority", 0.0) or 0.0),
                float(item.get("similarity", 0.0) or 0.0),
                -idx,
            )
            _store_row(row_key or row_text.lower(), row_text, row_rank)

        if explicit_comparator_mode:
            ordered_row_keys = [
                key for key in ordered_target_method_keys
                if key in target_method_keys and key in row_payloads
            ]
        else:
            ordered_row_keys = [
                key for key in ordered_target_method_keys
                if key in target_method_keys and key in row_payloads
            ]
            competitor_rows = sorted(
                (
                    (key, payload)
                    for key, payload in row_payloads.items()
                    if key not in ordered_row_keys
                ),
                key=lambda entry: entry[1][0],
                reverse=True,
            )
            ordered_row_keys.extend(key for key, _payload in competitor_rows[:4])
            if not ordered_row_keys:
                ordered_row_keys = [
                    key
                    for key, _payload in sorted(
                        row_payloads.items(),
                        key=lambda entry: entry[1][0],
                        reverse=True,
                    )[:4]
                ]

        row_texts: List[str] = []
        for row_key in ordered_row_keys:
            payload = row_payloads.get(row_key)
            if payload is None:
                continue
            row_text = payload[1]
            if row_text not in row_texts:
                row_texts.append(row_text)

        if len(row_texts) < 2 and not explicit_comparator_mode:
            fallback_exact_rows: list[tuple[tuple, str]] = []
            for idx, entry in entries:
                item = entry["item"]
                if not _entry_has_exact_row(item):
                    continue
                row_id = str(item.get("row_id") or "").strip()
                if row_id and _is_composite_numeric_row_id(row_id):
                    continue
                row_text = _exact_row_text(item)
                if not row_text or row_text in row_texts:
                    continue
                row_key = _normalize_numeric_table_method_token(row_id) or row_text.lower()
                row_rank = (
                    1 if row_key in target_method_keys else 0,
                    len(target_method_keys) if row_key in target_method_keys else 0,
                    float(item.get("numeric_table_priority", 0.0) or 0.0),
                    float(item.get("similarity", 0.0) or 0.0),
                    -idx,
                )
                fallback_exact_rows.append((row_rank, row_text))
            fallback_exact_rows.sort(reverse=True)
            for _row_rank, row_text in fallback_exact_rows:
                if row_text in row_texts:
                    continue
                row_texts.append(row_text)
                if len(row_texts) >= 2:
                    break

        if len(row_texts) < 2:
            synthetic_units: list[dict] = []
            for _idx, entry in entries:
                item = entry["item"]
                if not _entry_has_exact_row(item):
                    continue
                row_id = str(item.get("row_id") or "").strip()
                if row_id and _is_composite_numeric_row_id(row_id) and not explicit_comparator_mode:
                    continue
                if explicit_comparator_mode:
                    row_key = _normalize_numeric_table_method_token(row_id)
                    if not row_key or row_key not in target_method_keys:
                        continue
                row_text = _exact_row_text(item)
                if not row_text:
                    continue
                synthetic_units.append(
                    {
                        "evidence_unit_type": "table_row",
                        "row_id": row_id,
                        "row_text": row_text,
                        "row_numbers": _strip_leading_numeric_table_row_id(row_text, row_id),
                    }
                )
            if len(synthetic_units) >= 2:
                synthetic_bundle_text = _build_multi_row_bundle_context_text(
                    {
                        "chunk_type": "table",
                        "block_type": "table",
                        "structured_table_bundle": True,
                        "table_caption": best_caption,
                        "table_header": best_header,
                        "evidence_units": synthetic_units,
                    },
                    query,
                )
                if synthetic_bundle_text:
                    return synthetic_bundle_text
            return ""

        parts: list[str] = []
        for value in (best_caption, best_header, *row_texts):
            normalized = re.sub(r"\s+", " ", str(value or "")).strip()
            if normalized and normalized not in parts:
                parts.append(normalized)
        return "\n".join(parts)

    provisional_entries: List[dict] = []
    for item, text in fallback_entries:
        provisional_entries.append(
            {
                "item": item,
                "text": text,
                "page": int(item.get("page") or 0),
                "table_id": _resolve_numeric_table_context_table_id(item),
            }
        )

    has_delta_exact_anchor = bool(
        delta_metric_query
        and any(
            _numeric_table_delta_column_coverage(entry["item"], query, hints) >= 3
            for entry in provisional_entries
        )
    )

    bundle_row_keys_by_table: dict[str, set[str]] = {}
    for entry in provisional_entries:
        bundle_text = _build_multi_row_bundle_context_text(entry["item"], query)
        if not bundle_text:
            continue
        bundle_row_units = _extract_numeric_table_bundle_row_units(entry["item"], query)
        if not bundle_row_units:
            continue
        table_key = entry["table_id"] or f"page:{entry['page']}"
        table_row_keys = bundle_row_keys_by_table.setdefault(table_key, set())
        for unit in bundle_row_units:
            row_id = str(unit.get("row_id") or "").strip()
            row_text = re.sub(
                r"\s+",
                " ",
                str(
                    unit.get("content")
                    or unit.get("row_text")
                    or unit.get("row_numbers")
                    or "",
                ),
            ).strip()
            row_key = _normalize_numeric_table_method_token(row_id) or row_text.lower()
            if row_key:
                table_row_keys.add(row_key)

    anchor_candidates = []
    for idx, entry in enumerate(provisional_entries):
        role = _classify_numeric_table_context_role(entry["item"], query, hints)
        if role == "anchor":
            anchor_candidates.append((idx, entry))

    primary_table_id = ""
    primary_table_key = ""
    primary_anchor_pages: set[int] = set()
    if anchor_candidates:
        table_scores: dict[str, dict] = {}
        for idx, entry in anchor_candidates:
            table_key = _table_key(entry)
            stats = table_scores.setdefault(
                table_key,
                {
                    "count": 0,
                    "best_rank": None,
                    "best_entry": entry,
                    "first_idx": idx,
                },
            )
            stats["count"] += 1
            rank = _entry_anchor_rank(entry, idx)
            if stats["best_rank"] is None or rank > stats["best_rank"]:
                stats["best_rank"] = rank
                stats["best_entry"] = entry
            stats["first_idx"] = min(stats["first_idx"], idx)
        primary_table_key, primary_stats = max(
            table_scores.items(),
            key=lambda item: (
                item[1]["best_rank"][0],
                item[1]["best_rank"][1],
                item[1]["count"],
                item[1]["best_rank"][2],
                item[1]["best_rank"][3],
                item[1]["best_rank"][4],
                -item[1]["first_idx"],
            ),
        )
        primary_table_id = primary_stats["best_entry"]["table_id"] or ""
        primary_anchor_pages = {
            entry["page"]
            for _idx, entry in anchor_candidates
            if entry["page"] > 0 and _table_key(entry) == primary_table_key
        }
        if not primary_anchor_pages:
            primary_anchor_pages = {
                entry["page"]
                for _idx, entry in anchor_candidates
                if entry["page"] > 0
            }

    bundle_override_by_table: dict[str, str] = {}
    bundle_projection_item_ids: dict[str, int] = {}
    if comparator_query:
        grouped_entries: dict[str, list[tuple[int, dict]]] = {}
        for idx, entry in enumerate(provisional_entries):
            if has_delta_exact_anchor and _numeric_table_delta_column_coverage(entry["item"], query, hints) < 3:
                continue
            grouped_entries.setdefault(_table_key(entry), []).append((idx, entry))
        preferred_table_keys = [primary_table_key] if primary_table_key else list(grouped_entries.keys())
        for table_key in preferred_table_keys:
            table_entries = grouped_entries.get(table_key, [])
            bundle_text = _synthesize_same_table_bundle_text(table_entries)
            if bundle_text:
                bundle_override_by_table[table_key] = bundle_text
                if table_entries:
                    projection_entry = next(
                        (
                            candidate[1]
                            for candidate in table_entries
                            if _chunk_type(candidate[1]["item"]) == "table_row"
                            and _entry_has_exact_row(candidate[1]["item"])
                        ),
                        None,
                    )
                    if projection_entry is None:
                        _projection_idx, projection_entry = max(
                            table_entries,
                            key=lambda candidate: _entry_anchor_rank(candidate[1], candidate[0]),
                        )
                    bundle_projection_item_ids[table_key] = id(projection_entry["item"])

    layered_entries: List[dict] = []
    seen_by_role: dict[str, set[str]] = {"anchor": set(), "focus": set(), "background": set()}
    seen_row_keys_by_table: dict[str, set[str]] = {}
    role_counts: Counter[str] = Counter()
    role_priority = {"anchor": 0, "focus": 1, "background": 2}
    role_limits = {
        "anchor": None,
        "focus": 1 if cost_query else 2,
        "background": 1 if anchor_candidates else 2,
    }

    def _should_keep_explanatory_background(entry: dict) -> bool:
        item = entry["item"]
        chunk_type = _chunk_type(item)
        if chunk_type in {"table", "caption", "table_row", "table_cell"}:
            return False
        if entry.get("table_id") and item.get("numeric_table_keep_support"):
            return False
        if primary_anchor_pages and entry["page"] not in primary_anchor_pages:
            return False
        text = re.sub(r"\s+", " ", str(entry.get("text") or item.get("chunk") or "")).strip()
        if len(text) < 40:
            return False
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
            return False
        return True

    for entry in provisional_entries:
        role = _classify_numeric_table_context_role(
            entry["item"],
            query,
            hints,
            primary_table_id=primary_table_id,
            primary_anchor_pages=primary_anchor_pages,
        )
        if anchor_candidates and role == "background":
            if _should_keep_explanatory_background(entry):
                role = "background"
            elif keep_row_projection and _chunk_type(entry["item"]) == "table_row":
                if _table_key(entry) in bundle_row_keys_by_table:
                    role = "anchor"
                else:
                    continue
            else:
                continue

        text = entry["text"]
        table_key = _table_key(entry)
        bundle_text = bundle_override_by_table.get(table_key)
        bundle_projection_item_id = bundle_projection_item_ids.get(table_key)
        bundle_projection_entry = (
            comparator_query
            and bundle_projection_item_id is not None
            and id(entry["item"]) == bundle_projection_item_id
        )
        chunk_type = _chunk_type(entry["item"])
        if (
            comparator_query
            and bundle_text
            and bundle_projection_item_id is not None
            and id(entry["item"]) != bundle_projection_item_id
            and chunk_type in {"table_row", "table", "caption", "table_cell"}
            and not (
                chunk_type == "table_row"
                and _entry_has_exact_row(entry["item"])
                and not explicit_comparator_mode
            )
        ):
            continue
        if bundle_projection_entry:
            role = "anchor"
        if comparator_query and bundle_text:
            if bundle_projection_entry or (role == "anchor" and chunk_type != "table_row"):
                text = bundle_text
        if (
            has_delta_exact_anchor
            and chunk_type in {"table", "caption", "table_cell", "table_row"}
            and _numeric_table_delta_column_coverage(entry["item"], query, hints) < 3
        ):
            continue
        if role != "anchor" and len(text) > 320:
            text = (
                _context_builder_singleton._extract_relevant_snippet(text, query, max_len=260)
                or text[:260]
            )

        chunk_type = _chunk_type(entry["item"])
        row_key = ""
        if chunk_type == "table_row":
            row_id = str(entry["item"].get("row_id") or "").strip()
            row_text = re.sub(r"\s+", " ", str(entry["item"].get("chunk") or entry["item"].get("raw_chunk_text") or text or "")).strip()
            row_key = _normalize_numeric_table_method_token(row_id) or _normalize_sparse_bundle_row_key(row_id, row_text)
            if row_key:
                bundle_row_keys = bundle_row_keys_by_table.get(table_key)
                if bundle_row_keys and row_key in bundle_row_keys and not keep_row_projection:
                    continue
                table_seen_row_keys = seen_row_keys_by_table.setdefault(table_key, set())
                if row_key in table_seen_row_keys:
                    continue
                table_seen_row_keys.add(row_key)
        normalized_context = " ".join(text.split()).lower()
        dedupe_key = f"{entry['page']}|{entry['table_id']}|{normalized_context}"
        if bundle_projection_entry and chunk_type == "table_row" and row_key:
            # 同文本的 bundle 摘要和投影行要分别保留，避免 comparator 主锚被去重吞掉。
            dedupe_key = f"{dedupe_key}|projection|{row_key}"
        if dedupe_key in seen_by_role[role]:
            continue

        limit = role_limits.get(role)
        if limit is not None and role_counts[role] >= limit:
            continue

        layered_entries.append(
            {
                "item": entry["item"],
                "text": text,
                "context_role": role,
            }
        )
        seen_by_role[role].add(dedupe_key)
        role_counts[role] += 1

    if not layered_entries:
        return [
            {"item": item, "text": text, "context_role": "background"}
            for item, text in fallback_entries
        ]

    layered_entries.sort(key=lambda entry: role_priority.get(entry["context_role"], 99))
    logger.info(
        "[numeric_table_context] layered cleanup: anchor=%d focus=%d background=%d primary_table=%s primary_pages=%s",
        sum(1 for entry in layered_entries if entry["context_role"] == "anchor"),
        sum(1 for entry in layered_entries if entry["context_role"] == "focus"),
        sum(1 for entry in layered_entries if entry["context_role"] == "background"),
        primary_table_id or "-",
        sorted(primary_anchor_pages),
    )
    return layered_entries


def _annotate_results_for_evidence_rerank(
    doc_id: str,
    results: List[dict],
    chunks: List[str],
    parent_chunks: List[str],
    chunk_headings: List[str],
    chunk_pages: List[int],
    chunk_types: List[str],
    chunk_metadata: List[dict],
    child_to_parent: dict,
    group_chunk_map: Optional[dict],
    include_rerank_text: bool = False,
) -> List[dict]:
    if not results:
        return results

    doc_title = _get_document_title(doc_id) if include_rerank_text else ""
    chunk_text_to_idx = {text: idx for idx, text in enumerate(chunks)} if chunks else {}
    chunk_idx_to_group = _build_chunk_idx_to_group_map(group_chunk_map)

    for item in results:
        chunk_id = _resolve_result_chunk_index(item, chunk_text_to_idx)
        if chunk_id is not None:
            item["chunk_id"] = chunk_id
            if chunk_id < len(chunk_headings) and not item.get("chunk_heading"):
                item["chunk_heading"] = chunk_headings[chunk_id]
            page = item.get("page")
            if chunk_id < len(chunk_pages) and (not isinstance(page, int) or page <= 0):
                item["page"] = chunk_pages[chunk_id]
            if chunk_id < len(chunk_types) and not item.get("chunk_type"):
                item["chunk_type"] = chunk_types[chunk_id]
            if chunk_id < len(chunk_metadata):
                _apply_chunk_metadata(item, chunk_metadata[chunk_id])
            if item.get("parent_id") is None:
                parent_id = child_to_parent.get(chunk_id)
                if parent_id is not None:
                    item["parent_id"] = parent_id
            if not item.get("group_id"):
                group_id = chunk_idx_to_group.get(chunk_id, "")
                if group_id:
                    item["group_id"] = group_id
            item["raw_chunk_text"] = item.get("child_chunk") or chunks[chunk_id]
        else:
            item["raw_chunk_text"] = item.get("child_chunk") or item.get("chunk", "")

        if not item.get("chunk_type"):
            item["chunk_type"] = _guess_chunk_type(item.get("raw_chunk_text") or item.get("chunk", ""))

        item["doc_id"] = item.get("doc_id") or doc_id
        item["semantic_group_id"] = item.get("group_id") or ""
        item["snippet_basis"] = item.get("snippet") or (item.get("raw_chunk_text") or item.get("chunk", ""))[:200]
        _normalize_structural_metadata(item)
        raw_text = item.get("raw_chunk_text") or item.get("chunk", "")
        if (
            item.get("chunk_type") != "table_row"
            and raw_text
            and _looks_like_numeric_table_support(raw_text, item.get("chunk_type", ""))
        ):
            if not item.get("table_caption"):
                table_caption = _extract_table_caption_from_text(raw_text)
                if table_caption:
                    item["table_caption"] = table_caption
            if not item.get("table_id"):
                table_id = _extract_table_id(item.get("table_caption", ""))
                if table_id:
                    item["table_id"] = table_id
            if not item.get("table_header"):
                table_header = _extract_table_header_snippet(raw_text)
                if table_header:
                    item["table_header"] = table_header
        if include_rerank_text:
            item["doc_title"] = item.get("doc_title") or doc_title
            item["rerank_text"] = _build_evidence_unit_text(item, chunks, parent_chunks)

    return results


def _extract_table_caption_from_text(text: str, preferred_labels: Optional[List[str]] = None) -> str:
    if not text:
        return ""
    candidates: List[str] = []
    for line in text.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        normalized = re.sub(r"^\[TABLE\]\s*", "", normalized, flags=re.IGNORECASE)
        if re.match(r"^(?:Table|TABLE|表)\s*\.?\s*\d+[a-zA-Z]?\b", normalized, re.IGNORECASE):
            candidates.append(normalized)
    if not candidates:
        return ""
    preferred = [label.lower() for label in (preferred_labels or []) if label]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if any(label in candidate_lower for label in preferred):
            return candidate
    return candidates[0]


def _extract_table_id(caption: str) -> str:
    if not caption:
        return ""
    match = re.search(r"(Table\s*\.?\s*\d+[a-zA-Z]?|表\s*\.?\s*\d+[a-zA-Z]?)", caption, re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1)
    if value.lower().startswith("table"):
        number = re.search(r"\d+[a-zA-Z]?", value)
        return f"Table {number.group(0)}" if number else re.sub(r"\s+", " ", value).strip()
    number = re.search(r"\d+[a-zA-Z]?", value)
    return f"Table {number.group(0)}" if number else value.strip()


def _has_explicit_numeric_table_match(item: dict, target_tables: set[str]) -> bool:
    if not target_tables:
        return False
    explicit_table = _extract_table_id(
        (item.get("table_id") or item.get("table_caption") or "").strip()
    )
    if explicit_table:
        return explicit_table.lower() in target_tables
    caption = (item.get("table_caption") or "").strip().lower()
    return bool(caption and any(value in caption for value in target_tables))


_STRICT_EXPLICIT_TABLE_COLUMNS = {"FID", "Acc", "||D_gen||", "ΔAcc/||D_gen||"}


def _should_require_explicit_table_anchor(hints: dict[str, List[str]]) -> bool:
    target_tables = {value for value in hints.get("table_labels", []) if value}
    normalized_columns = {
        _normalize_numeric_column_name(value)
        for value in hints.get("columns", [])
        if value
    }
    if not target_tables or not normalized_columns:
        return False
    if not normalized_columns.issubset(_STRICT_EXPLICIT_TABLE_COLUMNS):
        return False
    return not hints.get("comparison") and not hints.get("backbones")


def _extract_prefix_table_id(text: str) -> str:
    sample = (text or "").strip()
    if not sample:
        return ""
    match = re.match(
        r"^(?:\[TABLE\]\s*)?((?:Table|表)\s*\.?\s*\d+[a-zA-Z]?)\b",
        sample,
        re.IGNORECASE,
    )
    if not match:
        return ""
    return _extract_table_id(match.group(1))


def _has_strict_numeric_table_anchor(
    item: dict,
    target_tables: set[str],
    evidence_text: str = "",
) -> bool:
    if not target_tables:
        return False
    explicit_table = _extract_table_id(
        (item.get("table_id") or item.get("table_caption") or "").strip()
    )
    if explicit_table and explicit_table.lower() in target_tables:
        return True
    for source in (
        item.get("table_caption", ""),
        item.get("raw_chunk_text", ""),
        item.get("chunk", ""),
        evidence_text,
    ):
        prefix_table = _extract_prefix_table_id(source)
        if prefix_table and prefix_table.lower() in target_tables:
            return True
    return False


def _extract_table_mentions(text: str) -> List[tuple[int, int, str]]:
    mentions: List[tuple[int, int, str]] = []
    for match in re.finditer(r"(?:Table|表)\s*\.?\s*\d+[a-zA-Z]?\b", text or "", re.IGNORECASE):
        table_id = _extract_table_id(match.group(0))
        if not table_id:
            continue
        mentions.append((match.start(), match.end(), table_id))
    return mentions


def _is_caption_like_table_mention(text: str, start: int, end: int) -> bool:
    if not text or start < 0 or end < start:
        return False
    if re.match(r"^\s*[:：]", (text or "")[end:end + 3]):
        return True
    line_start = text.rfind("\n", 0, start)
    line_start = 0 if line_start < 0 else line_start + 1
    prefix = text[line_start:start]
    if not prefix.strip():
        return True
    normalized_prefix = re.sub(r"\s+", "", prefix).lower()
    return normalized_prefix in {"[table]", "[table]:"}


def _fallback_page_text_table_bundle_id(table_id: str) -> str:
    resolved = _extract_table_id(table_id or "")
    if not resolved:
        return ""
    return f"page-text:{resolved.lower()}"


def _slice_text_to_requested_table(text: str, preferred_labels: Optional[List[str]] = None) -> str:
    mentions = _extract_table_mentions(text)
    if not mentions and preferred_labels:
        normalized_text = re.sub(r"\b(Table)(\d+)\b", r"\1 \2", text or "", flags=re.IGNORECASE)
        normalized_text = re.sub(r"(表)(\d+)", r"\1 \2", normalized_text)
        if normalized_text != (text or ""):
            mentions = _extract_table_mentions(normalized_text)
            text = normalized_text
    if not mentions or not preferred_labels:
        return text

    target_tables = {
        _extract_table_id(label).lower()
        for label in preferred_labels
        if _extract_table_id(label)
    }
    if not target_tables:
        return text

    matched_index = next(
        (
            idx
            for idx, (start, end, table_id) in enumerate(mentions)
            if table_id.lower() in target_tables and _is_caption_like_table_mention(text, start, end)
        ),
        None,
    )
    if matched_index is None:
        return ""

    start = mentions[matched_index][0]
    full_text = text or ""
    candidate_ends = [entry[0] for entry in mentions[matched_index + 1:]] + [len(full_text)]
    row_pattern = _build_plain_table_row_pattern(2, 7)
    chosen_end = len(full_text)
    for end in candidate_ends:
        candidate = full_text[start:end].strip()
        if not candidate:
            continue
        header = _extract_table_header_snippet(candidate)
        numeric_hits = len(_extract_numeric_value_tokens(candidate))
        has_table_body = bool(
            _extract_markdown_table_rows(candidate)
            or row_pattern.search(re.sub(r"\s+", " ", candidate))
            or (
                numeric_hits >= 4
                and len(_extract_table_header_columns(header)) >= 1
                and _looks_like_numeric_table_support(candidate, "table")
            )
        )
        if has_table_body:
            chosen_end = end
            break
    return full_text[start:chosen_end].strip()


def _extract_table_header_snippet(text: str) -> str:
    if not text:
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("|") and idx + 1 < len(lines) and "---" in lines[idx + 1]:
            return line[:400]

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    header_match = re.search(
        r"((?:\b(?:ResNet|DenseNet|ViT|Swin|ConvNeXt)[-_ ]?\d+\b\s*){1,4}"
        r"(?:(?<!\w)(?:all|overall|total|many|medium|med\.?|few(?:-shot)?)(?!\w)\s*){2,8})",
        normalized,
        re.IGNORECASE,
    )
    if header_match:
        return re.sub(r"\s+", " ", header_match.group(1)).strip()[:400]
    token_positions = [
        normalized.lower().find(token)
        for token in ("resnet", "all", "many", "med.", "medium", "few", "fid", "acc", "d_gen")
        if normalized.lower().find(token) >= 0
    ]
    if token_positions:
        pos = min(token_positions)
        start = max(0, pos - 20)
        end = min(len(normalized), pos + 260)
        snippet = normalized[start:end].strip()
        row_start = re.search(
            r"\b[A-Za-z][A-Za-z0-9+/_\-.]*(?:\s+[A-Z][a-z]+\s+et\s+al\.\s*\[\d+[a-z]?\])?"
            r"\s+(?:-?\d+\.?\d*|-)(?:\s+(?:-?\d+\.?\d*|-)){3,}",
            snippet,
        )
        if row_start and row_start.start() > 0:
            snippet = snippet[:row_start.start()].strip()
        return snippet[:400]
    return normalized[:220]


def _extract_markdown_table_rows(text: str) -> List[dict]:
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines: List[str] = []
    for line in raw_lines:
        if line.startswith("|"):
            lines.append(line)
            continue
        if "|" in line:
            pipe_fragment = line[line.find("|"):].strip()
            if pipe_fragment.startswith("|") and pipe_fragment.count("|") >= 2:
                lines.append(pipe_fragment)
    if not lines:
        return []

    caption = _extract_table_caption_from_text(text)
    table_id = _extract_table_id(caption)
    rows: List[dict] = []
    header = ""
    for idx, line in enumerate(lines):
        if not line.startswith("|"):
            pipe_idx = line.find("|")
            if pipe_idx <= 0:
                continue
            line = line[pipe_idx:].strip()
        if not line.startswith("|"):
            continue
        if idx + 1 < len(lines) and lines[idx + 1].startswith("|") and "---" in lines[idx + 1]:
            header = line
            for row_line in lines[idx + 2:]:
                if not row_line.startswith("|"):
                    break
                cells = [cell.strip() for cell in row_line.strip("|").split("|")]
                row_text = " | ".join(cell for cell in cells if cell)
                if not row_text:
                    continue
                row_id = cells[0].strip() if cells else ""
                rows.append(
                    {
                        "row_id": row_id,
                        "row_text": row_text,
                        "row_numbers": " ".join(cell for cell in cells[1:] if cell),
                        "table_caption": caption,
                        "table_id": table_id,
                        "table_header": header,
                    }
                )
            break
    return rows


def _extract_serialized_structured_bundle_body_rows(
    text: str,
    *,
    table_id: str = "",
    table_caption: str = "",
    table_header: str = "",
) -> List[dict]:
    if "[Structured Table Bundle]" not in (text or "") or "[Body]" not in (text or ""):
        return []

    header_match = re.search(
        r"\[Header\]\s*(.*?)(?:\n\s*\[Body\]|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    body_match = re.search(
        r"\[Body\]\s*(.*?)(?:\n\s*\[(?:Footnote|Structured Table Bundle|Hints|Header)\]|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not body_match:
        return []

    effective_header = re.sub(
        r"\s+",
        " ",
        str(table_header or (header_match.group(1) if header_match else "") or "").strip(),
    ).strip()
    caption = table_caption or _extract_table_caption_from_text(text)
    effective_table_id = table_id or _extract_table_id(caption)

    rows: List[dict] = []
    for raw_line in body_match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("["):
            continue
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in re.split(r"\s+\|\s+", line.strip())]
        cells = [cell for cell in cells if cell]
        if len(cells) < 2:
            continue
        row_id = cells[0]
        row_numbers = " ".join(
            re.sub(r"^[^=|]{1,40}=", "", cell).strip()
            for cell in cells[1:]
            if cell
        ).strip()
        row_text = " | ".join(cells)
        if not row_id or not row_numbers:
            continue
        rows.append(
            {
                "row_id": row_id,
                "row_text": row_text,
                "row_numbers": row_numbers,
                "table_caption": caption,
                "table_id": effective_table_id,
                "table_header": effective_header,
            }
        )
    return rows


def _extract_structured_table_rows(item: dict) -> List[dict]:
    evidence_units = item.get("evidence_units")
    if not isinstance(evidence_units, list):
        return []

    rows: List[dict] = []
    for unit in evidence_units:
        if not isinstance(unit, dict):
            continue
        if (unit.get("evidence_unit_type") or "").strip().lower() != "table_row":
            continue
        if unit.get("is_header_row"):
            continue

        row_text = (unit.get("row_text") or unit.get("content") or "").strip()
        if not row_text:
            continue

        cell_units = unit.get("cell_evidence_units")
        if not isinstance(cell_units, list):
            cell_units = []

        row_id = (unit.get("row_id") or "").strip()
        if not row_id and cell_units:
            row_id = str(
                cell_units[0].get("cell_text")
                or cell_units[0].get("content")
                or ""
            ).strip()

        row_numbers = (unit.get("row_numbers") or "").strip()
        if not row_numbers and cell_units:
            row_numbers = " ".join(
                str(cell.get("cell_text") or cell.get("content") or "").strip()
                for cell in cell_units[1:]
                if isinstance(cell, dict)
            ).strip()

        rows.append({
            "evidence_unit_id": unit.get("evidence_unit_id", ""),
            "row_id": row_id,
            "row_text": row_text,
            "row_numbers": row_numbers,
            "table_bundle_id": (
                unit.get("table_bundle_id")
                or item.get("table_bundle_id", "")
                or _fallback_page_text_table_bundle_id(unit.get("table_id") or item.get("table_id", ""))
            ),
            "table_caption": unit.get("table_caption") or item.get("table_caption", ""),
            "table_id": unit.get("table_id") or item.get("table_id", ""),
            "table_header": unit.get("table_header") or item.get("table_header", ""),
            "row_number": unit.get("row_number") or unit.get("row_idx"),
            "page": unit.get("page") or item.get("page"),
            "bounding_box": unit.get("bounding_box") or unit.get("bbox") or [],
            "cell_evidence_units": cell_units,
            "cell_evidence_unit_ids": unit.get("cell_evidence_unit_ids") or [
                cell.get("evidence_unit_id")
                for cell in cell_units
                if isinstance(cell, dict) and cell.get("evidence_unit_id")
            ],
        })

    return rows


def _extract_plain_table_rows(text: str, hints: dict[str, List[str]], query: str = "") -> List[dict]:
    if not text:
        return []

    scoped_text = _slice_text_to_requested_table(text, hints.get("table_labels", []))
    if hints.get("table_labels") and not scoped_text:
        return []

    normalized = re.sub(r"\s+", " ", scoped_text).strip()
    # PDF extraction often glues compact author suffixes onto method names
    # (for example `cRTKangetal.[2019]`), which breaks generic row parsing.
    # Normalize these author citations before regex row extraction so the row id
    # stays on the method side instead of collapsing to the author token.
    normalized = re.sub(
        r"(?<=[A-Za-z0-9\)])(?=([A-Z][a-z]+etal\.\[\d+[a-z]?\]))",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"([A-Z][a-z]+)etal\.\[(\d+[a-z]?)\]",
        r"\1 et al. [\2]",
        normalized,
    )
    caption = _extract_table_caption_from_text(scoped_text, preferred_labels=hints.get("table_labels", []))
    table_id = _extract_table_id(caption)
    header = _extract_table_header_snippet(scoped_text)
    header_columns = _extract_table_header_columns(header)
    rows: List[dict] = []
    seen: set[str] = set()
    markdown_rows = _extract_markdown_table_rows(scoped_text)
    if markdown_rows:
        rows.extend(markdown_rows)
        seen.update(
            (item.get("row_text") or "").strip().lower()
            for item in markdown_rows
            if (item.get("row_text") or "").strip()
        )
    comparison_query = bool(hints.get("comparison"))
    min_numeric_tokens, max_numeric_tokens = _get_plain_table_row_numeric_span(hints)
    numeric_group = rf"(?:{_NUMERIC_VALUE_TOKEN_PATTERN})"
    method_numeric_pattern = (
        rf"{numeric_group}(?:\s+{numeric_group})"
        rf"{{{max(1, min_numeric_tokens - 1)},{max(1, max_numeric_tokens - 1)}}}"
    )
    target_methods_compact = {
        re.sub(r"\s+", "", str(value or "").lower())
        for value in hints.get("methods", [])
        if value
    }

    def _looks_like_header_fragment(row_id: str) -> bool:
        normalized = re.sub(r"\s+", " ", (row_id or "").lower()).strip()
        if not normalized:
            return True

        compact = re.sub(r"\s+", "", normalized)
        tokens = [
            token.strip("()[]{}:;,.-")
            for token in re.split(r"[\s/|]+", normalized)
            if token.strip("()[]{}:;,.-")
        ]
        header_label_tokens = {
            "baseline",
            "id",
            "ood",
            "model",
            "models",
            "method",
            "methods",
            "group",
            "groups",
            "acc",
            "accuracy",
            "fid",
            "all",
            "many",
            "med",
            "few",
            "resnet",
            "p",
            "t",
        }
        if len(_extract_table_header_columns(row_id)) >= 2:
            return True
        has_target_method = bool(
            target_methods_compact and any(value in compact for value in target_methods_compact)
        )
        column_hits = {
            value.lower()
            for value in hints.get("columns", [])
            if value and value.lower() in normalized
        }
        if len(column_hits) >= 2:
            return True
        header_token_hits = sum(1 for token in tokens if token in header_label_tokens)
        if len(tokens) >= 3 and header_token_hits >= 2:
            return True
        if "baseline" in tokens and header_token_hits >= 1:
            return True
        if normalized.count("resnet") >= 2:
            return True
        if (
            not has_target_method
            and len(normalized.split()) >= 3
            and any(
                token in normalized
                for token in (
                    "method",
                    "methods",
                    "dataset",
                    "datasets",
                    "results on",
                )
            )
        ):
            return True
        return False

    # 优先用查询中的方法名锚定局部行片段，直接抽取数字列，避免专家数或相邻行串入。
    method_patterns = []
    for method in hints.get("methods", []):
        pattern = re.escape(method.strip())
        pattern = pattern.replace(r"\ ", r"\s*")
        pattern = pattern.replace(r"\(", r"\s*\(")
        pattern = pattern.replace(r"\)", r"\s*\)")
        method_patterns.append((method, pattern))

    # Do not run a single complex method+numbers regex over the whole table
    # string. PDF table text can be highly glued, and the optional author-token
    # subpattern may backtrack catastrophically. Anchor the method first, then
    # search only a short local window for the numeric span.
    method_number_re = re.compile(method_numeric_pattern, re.IGNORECASE)
    max_method_scan_chars = 360
    max_method_number_gap = 140
    for method, method_pattern in method_patterns:
        method_anchor_re = re.compile(
            rf"(?<![A-Za-z0-9])({method_pattern})",
            re.IGNORECASE,
        )
        match = None
        number_match = None
        for candidate_match in method_anchor_re.finditer(normalized):
            if _has_composite_prefix_before_method(normalized, candidate_match.start(1)):
                continue
            tail = normalized[candidate_match.end(1): candidate_match.end(1) + max_method_scan_chars]
            candidate_number_match = method_number_re.search(tail)
            if candidate_number_match is None:
                continue
            if candidate_number_match.start() > max_method_number_gap:
                continue
            match = candidate_match
            number_match = candidate_number_match
            break
        if match is None or number_match is None:
            continue
        row_numbers = number_match.group(0).strip(" ,;")
        row_text = f"{method} {row_numbers}".strip()
        key = row_text.lower()
        if len(row_text) < 12 or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "row_id": method,
                "row_text": row_text,
                "row_numbers": row_numbers,
                "table_caption": caption,
                "table_id": table_id,
                "table_header": header,
            }
        )

    # 无方法锚点时，退化为通用行模式抽取。
    if rows and not (comparison_query or _is_numeric_table_bundle_query(query, hints)):
        return rows

    search_spaces = [normalized]
    if header_columns:
        last_header = header_columns[-1]
        header_patterns = {
            "All": r"\b(?:all|overall|total)\b",
            "Many": r"\bmany\b",
            "Med.": r"\b(?:medium|med\.?)\b",
            "Few": r"\bfew(?:-shot)?\b",
        }
        pattern = header_patterns.get(last_header)
        if pattern:
            header_match = re.search(pattern, normalized, re.IGNORECASE)
            if header_match:
                trimmed = normalized[header_match.end():].strip()
                if trimmed and trimmed != normalized:
                    search_spaces.insert(0, trimmed)

    row_pattern = _build_plain_table_row_pattern(min_numeric_tokens, max_numeric_tokens)
    normalized_query_columns = {
        _normalize_numeric_column_name(value)
        for value in hints.get("columns", [])
        if value
    }
    prefer_first_duplicate_row_id = bool(
        hints.get("table_labels")
        and normalized_query_columns == {"FID", "Acc"}
        and not target_methods_compact
    )
    row_id_scores: dict[str, tuple[float, int]] = {}

    def _score_duplicate_row(unit: dict) -> float:
        if not prefer_first_duplicate_row_id:
            return 0.0
        duplicate_sort_column = ""
        if "ΔAcc/||D_gen||" in normalized_query_columns:
            duplicate_sort_column = "ΔAcc/||D_gen||"
        elif "Few" in normalized_query_columns:
            duplicate_sort_column = "Few"
        elif "Acc" in normalized_query_columns:
            duplicate_sort_column = "Acc"
        elif "FID" in normalized_query_columns:
            duplicate_sort_column = "FID"
        if not duplicate_sort_column:
            return 0.0
        focused = _build_query_focused_table_row(unit, hints)
        column_map = focused.get("column_map") or {}
        value = _parse_numeric_table_value(column_map.get(duplicate_sort_column, ""))
        if value is None:
            return float("-inf")
        if duplicate_sort_column == "FID":
            return -value
        return value

    for search_space in search_spaces:
        for match in row_pattern.finditer(search_space):
            row_id = match.group(1).strip(" ,;")
            row_numbers = match.group(2).strip(" ,;")
            row_text = f"{row_id} {row_numbers}".strip()
            key = row_text.lower()
            if len(row_text) < 12 or key in seen or _looks_like_header_fragment(row_id):
                continue
            normalized_row_id = _normalize_numeric_table_method_token(row_id)
            if (
                {"||D_gen||", "ΔAcc/||D_gen||"} & normalized_query_columns
                and re.fullmatch(r"[-−]?\d+(?:\.\d+)?", row_id.replace(",", ""))
            ):
                continue
            candidate = {
                "row_id": row_id,
                "row_text": row_text,
                "row_numbers": row_numbers,
                "table_caption": caption,
                "table_id": table_id,
                "table_header": header,
            }
            seen.add(key)
            if prefer_first_duplicate_row_id and normalized_row_id:
                candidate_score = _score_duplicate_row(candidate)
                existing = row_id_scores.get(normalized_row_id)
                if existing is None:
                    rows.append(candidate)
                    row_id_scores[normalized_row_id] = (candidate_score, len(rows) - 1)
                elif candidate_score > existing[0]:
                    rows[existing[1]] = candidate
                    row_id_scores[normalized_row_id] = (candidate_score, existing[1])
                continue
            rows.append(candidate)
    return rows


def _normalize_numeric_column_name(value: str) -> str:
    lowered = re.sub(r"\s+", "", str(value or "").lower())
    if lowered in {"all", "overall", "total"}:
        return "All"
    if lowered == "many":
        return "Many"
    if lowered in {"medium", "med", "med.", "middle"}:
        return "Med."
    if lowered in {"few", "few-shot", "tail"}:
        return "Few"
    if lowered in {"fid"}:
        return "FID"
    if lowered in {"acc", "acc.", "accuracy", "accuracy."}:
        return "Acc"
    if "d_gen" in lowered or "dgen" in lowered:
        if "delta" in lowered or "acc" in lowered or "δ" in value or "∆" in value or "Δ" in value:
            return "ΔAcc/||D_gen||"
        return "||D_gen||"
    if lowered in {"many/medium/few", "many/med./few", "many/med/few", "many/medium/few"}:
        return "Many/Med./Few"
    return str(value or "").strip()


_NUMERIC_TABLE_COLUMN_METHOD_NOISE = {
    "all",
    "overall",
    "total",
    "many",
    "medium",
    "med",
    "med.",
    "middle",
    "few",
    "few-shot",
    "tail",
    "fid",
    "acc",
    "acc.",
    "accuracy",
    "accuracy.",
}


def _is_numeric_table_column_noise(value: str) -> bool:
    normalized = re.sub(r"\s+", "", str(value or "").lower())
    if not normalized:
        return False
    if normalized in _NUMERIC_TABLE_COLUMN_METHOD_NOISE:
        return True
    canonical = _normalize_numeric_column_name(value)
    return canonical in {
        "All",
        "Many",
        "Med.",
        "Few",
        "FID",
        "Acc",
        "||D_gen||",
        "ΔAcc/||D_gen||",
        "Many/Med./Few",
    }


def _normalize_numeric_table_focus_hints(hints: Optional[dict[str, List[str]]]) -> dict[str, List[str]]:
    normalized_hints = dict(hints or {})
    methods = list(normalized_hints.get("methods", []) or [])
    columns = list(normalized_hints.get("columns", []) or [])
    column_keys = {
        _normalize_numeric_column_name(value).lower()
        for value in columns
        if value
    }
    kept_methods = []
    for method in methods:
        if _is_numeric_table_column_noise(method):
            canonical = _normalize_numeric_column_name(method)
            if canonical and canonical.lower() not in column_keys:
                columns.append(canonical)
                column_keys.add(canonical.lower())
            continue
        kept_methods.append(method)
    normalized_hints["methods"] = kept_methods
    normalized_hints["columns"] = columns
    return normalized_hints


def _normalize_numeric_table_method_token(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _row_mentions_target_method(row_id: str, target_method_keys: set[str]) -> bool:
    compact = _normalize_numeric_table_method_token(row_id)
    if not compact or not target_method_keys:
        return False
    return any(method_key and method_key in compact for method_key in target_method_keys)


def _is_composite_numeric_row_id(row_id: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(row_id or "").lower()).strip()
    if not normalized:
        return False
    return bool(re.search(r"(?:\+|/|&|\band\b|\bwith\b)", normalized))


def _is_headerish_numeric_table_row(unit: dict) -> bool:
    row_id = re.sub(r"\s+", " ", str(unit.get("row_id") or "")).strip().lower()
    if not row_id:
        return True
    if len(_extract_table_header_columns(row_id)) >= 2:
        return True

    tokens = [
        token.strip("()[]{}:;,.-")
        for token in re.split(r"[\s/|]+", row_id)
        if token.strip("()[]{}:;,.-")
    ]
    header_label_tokens = {
        "baseline",
        "id",
        "ood",
        "model",
        "models",
        "method",
        "methods",
        "group",
        "groups",
        "acc",
        "accuracy",
        "fid",
        "all",
        "many",
        "med",
        "few",
        "resnet",
        "p",
        "t",
    }
    header_token_hits = sum(1 for token in tokens if token in header_label_tokens)
    if len(tokens) >= 3 and header_token_hits >= 2:
        return True
    if "baseline" in tokens and header_token_hits >= 1:
        return True

    headerish_tokens = ("acc", "accuracy", "fid", "all", "many", "med", "few", "d_gen")
    if len(row_id.split()) >= 3 and any(token in row_id for token in headerish_tokens):
        return True

    row_text = re.sub(
        r"\s+",
        " ",
        str(unit.get("row_text") or unit.get("chunk") or unit.get("raw_chunk_text") or ""),
    ).strip().lower()
    if row_text and row_text != row_id and len(row_id.split()) <= 2:
        if (
            not _extract_numeric_value_tokens(row_text)
            and len(row_text.split()) >= 4
            and sum(1 for token in headerish_tokens if token in row_text) >= 2
        ):
            return True
    return False


def _is_numeric_table_cost_anchor_text(item: dict) -> bool:
    chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    if chunk_type in {"table_row", "table", "caption"}:
        return False
    evidence_text = (
        _build_numeric_table_evidence_text(item)
        or item.get("raw_chunk_text")
        or item.get("chunk")
        or ""
    )
    if not evidence_text:
        return False
    return _has_numeric_table_cost_anchor(evidence_text)


_NUMERIC_TABLE_BUNDLE_QUERY_RE = re.compile(
    r"(?:最高|最好|最佳|最优|最大|largest|highest|best|top(?:[- ]performing)?|which\s+(?:method|type)|哪种|哪个方法|哪类|取得最高|提升最大|第二好(?:的)?|第二佳|第二名|次优|second[- ]best|runner[- ]up)",
    re.IGNORECASE,
)


def _is_numeric_table_bundle_query(query: str, hints: dict[str, List[str]]) -> bool:
    if hints.get("comparison"):
        return True
    if not query:
        return False
    normalized_query = re.sub(r"\s+", " ", query).strip().lower()
    if _NUMERIC_TABLE_BUNDLE_QUERY_RE.search(normalized_query):
        return True
    second_best_tokens = (
        "\u7b2c\u4e8c\u597d",
        "\u7b2c\u4e8c\u4f73",
        "\u7b2c\u4e8c\u540d",
        "\u6b21\u4f18",
        "\u6b21\u4f73",
    )
    return any(
        token in normalized_query
        for token in (
            "second best",
            "second-best",
            "runner up",
            "runner-up",
            "nearest competitor",
        )
    ) or any(token in normalized_query for token in second_best_tokens)


_NUMERIC_TABLE_WINNER_QUERY_RE = re.compile(
    r"(?:最高|最好|最佳|最优|最大|largest|highest|best|top(?:[- ]performing)?|winner|winning)",
    re.IGNORECASE,
)


def _is_numeric_table_winner_style_query(query: str, hints: dict[str, List[str]]) -> bool:
    if not query or _is_numeric_table_cost_query(query):
        return False
    if not _NUMERIC_TABLE_WINNER_QUERY_RE.search(query):
        return False
    return bool(_preferred_numeric_table_sort_column(query, hints))


def _is_numeric_table_row_band_query(query: str, hints: dict[str, List[str]]) -> bool:
    return _is_numeric_table_bundle_query(query, hints) or _is_numeric_table_winner_style_query(query, hints)


def _is_numeric_table_explicit_comparator_query(
    query: str,
    hints: dict[str, List[str]],
) -> bool:
    target_methods = _extract_numeric_table_row_method_targets(hints)
    if len(target_methods) < 2:
        return False
    if hints.get("comparison"):
        return True

    normalized_query = re.sub(r"\s+", " ", (query or "").strip())
    return bool(_NUMERIC_TABLE_EXPLICIT_COMPARATOR_RE.search(normalized_query))


_NUMERIC_VALUE_TOKEN_PATTERN = (
    r"(?:[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-−]?\d+(?:\.\d+)?)"
    r"(?:\s*(?:[×xX]|·)\s*10\s*(?:\^?\s*[-−]?\s*\d+)?)?"
    r"|-"
)
_NUMERIC_VALUE_TOKEN_RE = re.compile(_NUMERIC_VALUE_TOKEN_PATTERN)


def _extract_numeric_value_tokens(text: str) -> List[str]:
    sample = re.sub(r"\s+", " ", text or "").strip()
    if not sample:
        return []
    return [match.group(0).strip() for match in _NUMERIC_VALUE_TOKEN_RE.finditer(sample)]


def _get_plain_table_row_numeric_span(hints: dict[str, List[str]]) -> tuple[int, int]:
    override = hints.get("_numeric_span_override") if isinstance(hints, dict) else None
    if isinstance(override, (list, tuple)) and len(override) >= 2:
        min_override = _normalize_positive_int(override[0]) or 0
        max_override = _normalize_positive_int(override[1]) or 0
        if min_override > 0:
            if max_override <= 0:
                max_override = min_override
            return min_override, max(min_override, max_override)

    query_columns = _sort_numeric_columns(hints.get("columns", []))
    if not query_columns:
        return 4, 7

    query_column_keys = {
        re.sub(r"\s+", "", value).lower()
        for value in query_columns
        if value
    }
    frequency_bin_column_keys = {"all", "many", "med.", "few"}
    if query_column_keys and query_column_keys.issubset(frequency_bin_column_keys):
        if len(query_column_keys) >= 4:
            return 4, 7
        # Some frequency-bin result tables omit `All` and only expose
        # `Many / Med. / Few`. Lower the minimum span so those rows can be
        # recovered, but keep the upper bound for wide result-table rows
        # that still carry 5 numeric values.
        return 3, 7
    if frequency_bin_column_keys & query_column_keys:
        return 4, 7

    has_fid = "fid" in query_column_keys
    has_acc = bool({"acc", "accuracy"} & query_column_keys)
    has_d_gen = any("d_gen" in value or "dgen" in value for value in query_column_keys)
    has_delta = any("delta" in value or "δacc" in value or "∂acc" in value for value in query_column_keys)

    if has_fid and has_acc and not (has_d_gen or has_delta):
        return 2, 2
    if (has_d_gen or has_delta) and has_acc:
        return 3, 3
    if has_acc:
        return 2, 4
    if has_d_gen or has_delta:
        return 2, 4
    return max(2, len(query_columns)), min(7, max(2, len(query_columns)) + 2)


def _build_plain_table_row_pattern(min_numeric_tokens: int, max_numeric_tokens: int) -> re.Pattern:
    min_numeric_tokens = max(2, min_numeric_tokens)
    max_numeric_tokens = max(min_numeric_tokens, max_numeric_tokens)
    numeric_group = rf"(?:{_NUMERIC_VALUE_TOKEN_PATTERN})"
    repeated_group = rf"{numeric_group}(?:\s+{numeric_group}){{{min_numeric_tokens - 1},{max_numeric_tokens - 1}}}"
    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"([^\W\d][\w+/_\-.\[\]=]*(?:\s+[^\W\d][\w+/_\-.\[\]=]*){{0,3}}(?:\s*\([^)]{{0,32}}\))?)"
        rf"(?:\s+[A-Z][a-z]+\s+et\s+al\.\s*\[\d+[a-z]?\])?"
        rf"\s+({repeated_group})"
    )


_NUMERIC_TABLE_METHOD_STOPWORDS = {
    "fid",
    "acc",
    "accuracy",
    "in",
    "flops",
    "time",
    "cost",
    "latency",
    "runtime",
    "overhead",
    "params",
    "param",
    "parameter",
    "parameters",
}

_NUMERIC_TABLE_COST_QUERY_HINTS = (
    "flops",
    "推理时间",
    "开销",
    "latency",
    "runtime",
    "overhead",
    "cost",
    "inference time",
    "inference overhead",
    "training time",
    "训练时间",
    "耗时",
    "extra flops",
)

_NUMERIC_TABLE_COST_EVIDENCE_HINTS = (
    "24 hours",
    "24 hour",
    "24h",
    "six days",
    "6 days",
    "6天",
    "24小时",
    "no extra overhead",
    "without extra overhead",
    "no additional overhead",
    "additional overhead",
    "inference overhead",
)


def _extract_numeric_table_row_method_targets(hints: dict[str, List[str]]) -> set[str]:
    targets: set[str] = set()
    for value in hints.get("methods", []):
        key = _normalize_numeric_table_method_token(value)
        if not key or key in _NUMERIC_TABLE_METHOD_STOPWORDS:
            continue
        targets.add(key)
    return targets


def _is_numeric_table_cost_query(query: str) -> bool:
    sample = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if not sample:
        return False
    return any(token in sample for token in _NUMERIC_TABLE_COST_QUERY_HINTS)


def _has_numeric_table_cost_anchor(text: str) -> bool:
    sample = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not sample:
        return False
    compact = re.sub(r"[\s\-–—]+", "", sample)
    if any(token in sample for token in _NUMERIC_TABLE_COST_EVIDENCE_HINTS):
        return True
    has_duration_anchor = any(
        (
            re.search(r"24\s*hours?", sample),
            re.search(r"(?:six|6)\s*days?", sample),
            "24hours" in compact,
            "24hour" in compact,
            "6days" in compact,
            "sixdays" in compact,
            "24小时" in compact,
            "6天" in compact,
        )
    )
    has_overhead_anchor = any(
        (
            re.search(r"no\s*extra\s*overhead", sample),
            re.search(r"without\s*extra\s*overhead", sample),
            re.search(r"no\s*additional\s*overhead", sample),
            re.search(r"inference\s*overhead", sample),
            re.search(r"extra\s*flops", sample),
            "noextraoverhead" in compact,
            "withoutextraoverhead" in compact,
            "noadditionaloverhead" in compact,
            "inferenceoverhead" in compact,
            "extraflops" in compact,
        )
    )
    return has_duration_anchor or has_overhead_anchor


def _preferred_numeric_table_sort_column(query: str, hints: dict[str, List[str]]) -> str:
    query_lower = re.sub(r"\s+", " ", (query or "").lower()).strip()
    query_columns = _sort_numeric_columns(hints.get("columns", []))
    column_set = set(query_columns)

    if not column_set:
        return ""

    best_query = any(
        token in query_lower
        for token in ("最高", "最好", "最佳", "最优", "最大", "largest", "highest", "best", "top")
    )
    gain_query = any(
        token in query_lower
        for token in ("提升最大", "average gain", "per sample", "每样本", "增益", "gain")
    )
    few_query = "few" in query_lower or "少样本" in query_lower or "尾类" in query_lower
    accuracy_query = any(token in query_lower for token in ("acc", "accuracy", "准确率", "分类准确率"))

    if "ΔAcc/||D_gen||" in column_set and (best_query or gain_query):
        return "ΔAcc/||D_gen||"
    if gain_query and ("||D_gen||" in column_set or "Acc" in column_set):
        return "ΔAcc/||D_gen||"
    if "Few" in column_set and best_query and few_query:
        return "Few"
    if "Acc" in column_set and best_query and accuracy_query:
        return "Acc"
    if "FID" in column_set and best_query and accuracy_query:
        return "Acc"
    if "FID" in column_set and any(token in query_lower for token in ("最小", "最低", "smallest", "lowest", "minimum")):
        return "FID"
    if "All" in column_set and any(token in query_lower for token in ("all", "overall", "total", "整体", "总体")):
        return "All"
    return ""


def _resolve_numeric_table_effective_top_k(
    query: str,
    top_k: int,
    hints: Optional[dict[str, List[str]]] = None,
    results: Optional[List[dict]] = None,
) -> int:
    if top_k <= 0:
        return top_k
    if "numeric_table" not in (_analyze_evidence_need(query) or []):
        return top_k

    hints = hints or _query_rewriter_singleton.extract_numeric_table_hints(query)
    target_methods = _extract_numeric_table_row_method_targets(hints)
    explicit_comparison_methods = _is_numeric_table_explicit_comparator_query(query, hints)
    if bool(hints.get("comparison")) or _is_numeric_table_row_band_query(query, hints):
        top_k = max(top_k, 3)
    if explicit_comparison_methods:
        available_methods = {
            _normalize_numeric_table_method_token(item.get("row_id", ""))
            for item in (results or [])
            if (item.get("chunk_type") or item.get("block_type") or "").strip().lower() == "table_row"
            and _normalize_numeric_table_method_token(item.get("row_id", "")) in target_methods
        }
        if len(available_methods) > top_k:
            return max(top_k, min(len(available_methods), 6))
        return top_k
    if _is_numeric_table_row_band_query(query, hints):
        available_rows = [
            item
            for item in (results or [])
            if (item.get("chunk_type") or item.get("block_type") or "").strip().lower() == "table_row"
        ]
        if available_rows:
            return max(top_k, min(max(len(available_rows), 4), 6))
    return top_k


def _parse_numeric_table_value(value: str) -> Optional[float]:
    sample = re.sub(r"\s+", "", str(value or "").strip())
    if not sample or sample == "-":
        return None
    sample = sample.replace("−", "-")
    sample = sample.replace("×", "x").replace("·", "x")
    match = re.fullmatch(
        r"([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)(?:x10(?:\^?([-+]?\d+))?)?",
        sample,
        re.IGNORECASE,
    )
    if not match:
        return None
    base = match.group(1).replace(",", "")
    try:
        value_float = float(base)
    except ValueError:
        return None
    exponent = int(match.group(2)) if match.group(2) else 0
    return value_float * (10 ** exponent)


def _numeric_table_sort_bonus(unit: dict, query: str, hints: dict[str, List[str]]) -> float:
    if _is_headerish_numeric_table_row(unit):
        return 0.0
    column = _preferred_numeric_table_sort_column(query, hints)
    if not column:
        query_lower = re.sub(r"\s+", " ", (query or "").lower()).strip()
        focus_columns = {
            _normalize_numeric_column_name(value)
            for value in (unit.get("table_focus_columns") or [])
            if value
        }
        if any(token in query_lower for token in ("提升最大", "average gain", "per sample", "每样本", "增益", "gain")) and "ΔAcc/||D_gen||" in focus_columns:
            column = "ΔAcc/||D_gen||"
        elif ("few" in query_lower or "few-shot" in query_lower or "少样本" in query_lower or "尾类" in query_lower) and "Few" in focus_columns:
            column = "Few"
        elif any(token in query_lower for token in ("all", "overall", "total", "整体", "总体")) and "All" in focus_columns:
            column = "All"
        elif any(token in query_lower for token in ("acc", "accuracy", "准确率", "分类准确率")) and "Acc" in focus_columns:
            column = "Acc"
        elif "fid" in query_lower and "FID" in focus_columns:
            column = "FID"
    if not column:
        return 0.0
    focused = _build_query_focused_table_row(unit, hints)
    column_map = focused.get("column_map") or {}
    raw_value = column_map.get(column, "")
    if not raw_value:
        alias_map = {
            "FID": r"fid",
            "Acc": r"acc(?:uracy)?",
            "All": r"all|overall|total",
            "Many": r"many",
            "Med.": r"med\.?|medium",
            "Few": r"few(?:-shot)?",
            "||D_gen||": r"(?:\|\||∥)\s*d\s*[_ ]?gen\s*(?:\|\||∥)",
            "ΔAcc/||D_gen||": r"[Δ∆]\s*acc(?:/\s*(?:\|\||∥)\s*d\s*[_ ]?gen\s*(?:\|\||∥))?",
        }
        value_pattern = alias_map.get(column, re.escape(column))
        for source in (
            focused.get("text", ""),
            unit.get("chunk", ""),
            unit.get("raw_chunk_text", ""),
            unit.get("table_row_raw_text", ""),
        ):
            match = re.search(
                rf"(?:{value_pattern})\s*[:=]\s*({_NUMERIC_VALUE_TOKEN_PATTERN})",
                source or "",
                re.IGNORECASE,
            )
            if match:
                raw_value = match.group(1).strip()
                break
    value = _parse_numeric_table_value(raw_value)
    if value is None:
        return 0.0

    if column == "FID":
        score = -value
    elif column == "ΔAcc/||D_gen||":
        score = value * 100000.0
    else:
        score = value
    return score * 0.5


def _has_composite_prefix_before_method(text: str, match_start: int) -> bool:
    prefix = (text or "")[max(0, match_start - 32):match_start]
    return bool(
        re.search(r"(?:[A-Za-z0-9\)])\s*(?:\+|/|&|and|with)\s*$", prefix, re.IGNORECASE)
    )


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _strip_leading_numeric_table_row_id(text: str, row_id: str) -> str:
    sample = str(text or "").strip()
    label = str(row_id or "").strip()
    if not sample or not label:
        return sample

    compact_sample = re.sub(r"\s+", "", sample)
    compact_label = re.sub(r"\s+", "", label)
    if not compact_sample.lower().startswith(compact_label.lower()):
        return sample

    seen = 0
    cutoff = 0
    target = len(compact_label)
    for idx, char in enumerate(sample):
        if char.isspace():
            continue
        seen += 1
        if seen >= target:
            cutoff = idx + 1
            break

    stripped = sample[cutoff:].strip()
    return stripped or sample


def _extract_table_header_backbones(header: str) -> List[str]:
    matches = re.finditer(
        r"\b(?:ResNet|DenseNet|ViT|Swin|ConvNeXt)[-_ ]?\d+\b",
        header or "",
        re.IGNORECASE,
    )
    return _dedupe_preserve_order([match.group(0).strip() for match in matches])


def _extract_table_header_columns(header: str) -> List[str]:
    columns: List[str] = []
    for match in re.finditer(
        r"(?<!\w)(?:all|overall|total|many|medium|med\.?|few(?:-shot)?|fid|acc(?:uracy)?\.?|(?:\|\||∥)\s*d\s*[_ ]?gen\s*(?:\|\||∥)|[Δ∆]\s*acc(?:/\s*(?:\|\||∥)\s*d\s*[_ ]?gen\s*(?:\|\||∥))?)(?!\w)",
        header or "",
        re.IGNORECASE,
    ):
        normalized = _normalize_numeric_column_name(match.group(0))
        if normalized:
            columns.append(normalized)
    return columns


def _extract_rightmost_unique_columns(columns: List[str]) -> List[str]:
    reversed_unique: List[str] = []
    seen: set[str] = set()
    for column in reversed(columns):
        key = column.lower()
        if key in seen:
            continue
        seen.add(key)
        reversed_unique.append(column)
    return list(reversed(reversed_unique))


def _sort_numeric_columns(columns: List[str]) -> List[str]:
    normalized = _dedupe_preserve_order(
        [_normalize_numeric_column_name(value) for value in columns if value]
    )
    preferred_order = {
        "FID": 0,
        "||D_gen||": 1,
        "Acc": 2,
        "ΔAcc/||D_gen||": 3,
        "All": 4,
        "Many": 5,
        "Med.": 6,
        "Few": 7,
    }
    return [
        normalized[idx]
        for idx in sorted(
            range(len(normalized)),
            key=lambda idx: (preferred_order.get(normalized[idx], 99), idx),
        )
    ]


def _build_query_focused_table_row(unit: dict, hints: dict[str, List[str]]) -> dict:
    row_text = _get_numeric_table_boundary_text(unit)
    if not row_text:
        return {
            "text": "",
            "matched_backbone": "",
            "resolved_columns": [],
            "column_coverage": 0,
        }

    row_id = (unit.get("row_id") or "").strip()
    numeric_row_text = (unit.get("row_numbers") or "").strip()
    if not numeric_row_text:
        numeric_row_text = _strip_leading_numeric_table_row_id(row_text, row_id)
    if not numeric_row_text:
        numeric_row_text = row_text
    row_values = _extract_numeric_value_tokens(numeric_row_text)
    if not row_values:
        return {
            "text": "",
            "matched_backbone": "",
            "resolved_columns": [],
            "column_coverage": 0,
        }

    query_columns = _sort_numeric_columns(hints.get("columns", []))
    query_backbones = _dedupe_preserve_order(hints.get("backbones", []))
    header = unit.get("table_header", "") or ""
    header_backbones = _extract_table_header_backbones(header)
    header_columns = _extract_table_header_columns(header)
    header_column_keys = {column for column in header_columns if column}
    tail_columns = _extract_rightmost_unique_columns(header_columns)
    if tail_columns:
        tail_columns = _sort_numeric_columns(tail_columns)
    else:
        tail_columns = query_columns

    generic_metric_columns = {"FID", "Acc", "||D_gen||", "ΔAcc/||D_gen||"}
    strict_explicit_metric_query = bool(
        hints.get("table_labels")
        and query_columns
        and set(query_columns).issubset(generic_metric_columns)
    )
    allow_query_column_fallback = True
    if strict_explicit_metric_query:
        focus_column_keys = {
            _normalize_numeric_column_name(value)
            for value in (unit.get("table_focus_columns") or [])
            if value
        }
        compact_metric_match = len(row_values) in {len(query_columns), len(query_columns) + 1}
        support_metric_match = bool(
            "ΔAcc/||D_gen||" in query_columns
            and "||D_gen||" in header_column_keys
            and "Acc" in header_column_keys
            and len(row_values) >= len(query_columns) + 1
        )
        allow_query_column_fallback = (
            (bool(header_column_keys) and set(query_columns).issubset(header_column_keys))
            or (bool(focus_column_keys) and set(query_columns).issubset(focus_column_keys))
            or compact_metric_match
            or support_metric_match
        )
        if not allow_query_column_fallback:
            tail_columns = []

    matched_backbone = ""
    if query_backbones:
        for backbone in header_backbones:
            if any(backbone.lower() == target.lower() for target in query_backbones):
                matched_backbone = backbone
                break
        if not matched_backbone and not header_backbones and len(query_backbones) == 1:
            matched_backbone = query_backbones[0]
        if header_backbones and not matched_backbone:
            return {
                "text": "",
                "matched_backbone": "",
                "resolved_columns": [],
                "column_coverage": 0,
            }

    column_map: dict[str, str] = {}
    tail_column_keys = set(tail_columns)
    metric_tail_columns = (
        tail_columns
        if tail_columns and set(tail_columns).issubset(generic_metric_columns)
        else []
    )
    if (
        "ΔAcc/||D_gen||" in query_columns
        and "||D_gen||" in tail_column_keys
        and "Acc" in tail_column_keys
    ):
        metric_tail_columns = ["||D_gen||", "Acc", "ΔAcc/||D_gen||"]
    if (
        allow_query_column_fallback
        and metric_tail_columns
        and not matched_backbone
        and len(row_values) > len(metric_tail_columns)
    ):
        prefix_values = row_values[:len(metric_tail_columns)]
        column_map = {
            column: value
            for column, value in zip(metric_tail_columns, prefix_values)
        }
    should_preserve_support_tail_columns = bool(
        allow_query_column_fallback
        and not column_map
        and query_columns
        and tail_columns
        and "ΔAcc/||D_gen||" in query_columns
        and "||D_gen||" in tail_column_keys
        and set(query_columns).issubset(tail_column_keys)
        and len(row_values) >= len(tail_columns)
    )
    if should_preserve_support_tail_columns:
        mapped_values = row_values[-len(tail_columns):]
        column_map = {column: value for column, value in zip(tail_columns, mapped_values)}
    if not column_map and allow_query_column_fallback and query_columns and len(row_values) == len(query_columns):
        column_map = {column: value for column, value in zip(query_columns, row_values)}
    if (
        not column_map
        and allow_query_column_fallback
        and query_columns
        and "ΔAcc/||D_gen||" in query_columns
        and "||D_gen||" not in query_columns
        and len(row_values) == len(query_columns) + 1
    ):
        mapped_columns = ["||D_gen||", *query_columns]
        column_map = {column: value for column, value in zip(mapped_columns, row_values[-len(mapped_columns):])}
    if (
        not column_map
        and allow_query_column_fallback
        and query_columns
        and len(row_values) == len(query_columns) + 1
        and query_columns[0] != "FID"
    ):
        mapped_values = row_values[-len(query_columns):]
        column_map = {column: value for column, value in zip(query_columns, mapped_values)}
    if not column_map and tail_columns and len(row_values) >= len(tail_columns):
        mapped_values = row_values[-len(tail_columns):]
        column_map = {column: value for column, value in zip(tail_columns, mapped_values)}
    if not column_map and len(row_values) >= 4 and (
        matched_backbone or any(column in {"All", "Many", "Med.", "Few"} for column in query_columns)
    ):
        fallback_columns = ["All", "Many", "Med.", "Few"]
        mapped_values = row_values[-len(fallback_columns):]
        column_map = {column: value for column, value in zip(fallback_columns, mapped_values)}
    if not column_map and len(row_values) == 2 and matched_backbone:
        column_map = {"All": row_values[-1]}
    if (
        allow_query_column_fallback
        and len(query_columns) == 1
        and query_columns[0] in {"All", "Many", "Med.", "Few"}
    ):
        requested_column = query_columns[0]
        if requested_column == "All" and len(row_values) >= 4:
            column_map = {"All": row_values[-4]}
        elif requested_column == "Many" and len(row_values) >= 3:
            column_map = {"Many": row_values[-3]}
        elif requested_column == "Med." and len(row_values) >= 3:
            column_map = {"Med.": row_values[-2]}
        elif requested_column == "Few" and len(row_values) >= 3:
            column_map = {"Few": row_values[-1]}

    if not column_map:
        return {
            "text": "",
            "matched_backbone": matched_backbone,
            "resolved_columns": [],
            "column_coverage": 0,
        }

    resolved_columns = query_columns or tail_columns
    if (
        tail_columns
        and query_columns
        and set(query_columns).issubset(set(tail_columns))
        and len(tail_columns) > len(query_columns)
        and (
            strict_explicit_metric_query
            or set(tail_columns).issubset(generic_metric_columns)
        )
    ):
        resolved_columns = metric_tail_columns or tail_columns
    elif (
        metric_tail_columns
        and query_columns
        and set(query_columns).issubset(set(metric_tail_columns))
        and len(metric_tail_columns) > len(query_columns)
    ):
        resolved_columns = metric_tail_columns
    if (
        query_columns
        and "ΔAcc/||D_gen||" in query_columns
        and "||D_gen||" in column_map
        and "||D_gen||" not in resolved_columns
    ):
        resolved_columns = ["||D_gen||", *resolved_columns]
    pairs = [(column, column_map[column]) for column in resolved_columns if column in column_map]
    if query_columns and not pairs:
        return {
            "text": "",
            "matched_backbone": matched_backbone,
            "resolved_columns": [],
            "column_coverage": 0,
        }

    focus_parts: List[str] = []
    if row_id:
        focus_parts.append(row_id)
    if matched_backbone:
        focus_parts.append(matched_backbone)
    focus_parts.extend(f"{column}={value}" for column, value in pairs)

    return {
        "text": " | ".join(focus_parts).strip(),
        "matched_backbone": matched_backbone,
        "resolved_columns": [column for column, _ in pairs],
        "column_coverage": len(pairs),
        "column_map": column_map,
    }


def _expand_numeric_table_evidence_units(
    results: List[dict],
    query: str,
    *,
    include_rerank_text: bool = False,
    doc_title: str = "",
) -> List[dict]:
    if not should_apply_numeric_table_specialization():
        return results
    needs = set(_analyze_evidence_need(query) or [])
    if not results or not ({"numeric_table", "comparison_multi_aspect"} & needs):
        return results

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    comparison_query = bool(hints.get("comparison"))
    bundle_query = _is_numeric_table_bundle_query(query, hints)
    hinted_columns = list(hints.get("columns", []) or [])
    hinted_methods = list(hints.get("methods", []) or [])
    method_column_noise = [
        value for value in hinted_methods if _is_numeric_table_column_noise(value)
    ]
    if method_column_noise:
        hints = dict(hints)
        hints["methods"] = [
            value for value in hinted_methods if not _is_numeric_table_column_noise(value)
        ]
        existing_columns = list(hinted_columns)
        existing_column_keys = {
            _normalize_numeric_column_name(value).lower()
            for value in existing_columns
            if value
        }
        for value in method_column_noise:
            canonical = _normalize_numeric_column_name(value)
            key = canonical.lower()
            if canonical and key not in existing_column_keys:
                existing_columns.append(canonical)
                existing_column_keys.add(key)
        frequency_columns = {
            _normalize_numeric_column_name(value)
            for value in existing_columns
            if _normalize_numeric_column_name(value) in {"All", "Many", "Med.", "Few"}
        }
        if frequency_columns:
            existing_columns = [
                value
                for value in existing_columns
                if _normalize_numeric_column_name(value) != "Acc"
            ]
        hints["columns"] = existing_columns
    target_methods = _extract_numeric_table_row_method_targets(hints)
    explicit_comparison_methods = comparison_query and len(target_methods) >= 2
    target_datasets = _extract_numeric_table_dataset_mentions(" ".join(hints.get("datasets", [])))
    target_backbones = {value.lower() for value in hints.get("backbones", []) if value}
    target_columns = {
        _normalize_numeric_column_name(value).lower()
        for value in hints.get("columns", [])
        if value
    }
    target_tables = {
        value.lower()
        for value in hints.get("table_labels", [])
        if value
    } | {
        table_id.lower()
        for _, _, table_id in _extract_table_mentions(query)
        if table_id
    }
    binary_factor_query = bool(
        target_datasets
        and re.search(
            r"(?:\bIL\b|\bTL\b|image\s+list|text\s+list|combination|both|simultaneously|benefit|effect|同时|组合|都|影响|作用)",
            query,
            re.IGNORECASE,
        )
    )
    require_explicit_table_anchor = _should_require_explicit_table_anchor(hints)
    preferred_sort_column = _preferred_numeric_table_sort_column(query, hints)
    min_row_number_hits, _ = _get_plain_table_row_numeric_span(hints)
    if bundle_query:
        row_limit = 4 if len(target_methods) <= 1 else min(5, max(len(target_methods), 4))
    elif comparison_query:
        row_limit = 5 if len(target_methods) <= 1 else min(6, max(len(target_methods) + 2, 5))
    else:
        row_limit = 1 if len(target_methods) <= 1 else min(3, len(target_methods))
        if not target_methods:
            row_limit = 2
    if binary_factor_query:
        row_limit = max(row_limit, 4)
    expanded: List[dict] = list(results)
    seen_keys = {
        (
            (item.get("chunk") or item.get("raw_chunk_text") or "").strip().lower(),
            (item.get("page") or 0),
        )
        for item in results
    }

    max_items_to_expand = max(8, min(len(results), row_limit * 6))
    expand_started_at = time.perf_counter()
    max_expand_seconds = 1.5
    max_plain_parse_chars = 2500
    for item in results[:max_items_to_expand]:
        if time.perf_counter() - expand_started_at > max_expand_seconds:
            logger.warning(
                "[numeric_table] evidence-unit 扩展超过 %.1fs，提前停止（已处理部分候选）",
                max_expand_seconds,
            )
            break
        chunk_text = (item.get("raw_chunk_text") or item.get("chunk") or "").strip()
        chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
        structured_row_units = _extract_structured_table_rows(item)
        has_structured_rows = bool(structured_row_units)
        if not chunk_text and not structured_row_units:
            continue
        if not structured_row_units and not _looks_like_numeric_table_support(chunk_text, chunk_type):
            continue

        row_units = structured_row_units
        if not row_units and item.get("structured_table_bundle"):
            # Stored bundle chunks may carry the full table body in metadata even when
            # the serialized chunk text is only a sparse wrapper. Prefer body rows first
            # so second-best / winner queries keep comparator rows from the same bundle.
            row_units = _extract_structured_bundle_body_rows(item)
        if not row_units:
            row_units = _extract_markdown_table_rows(chunk_text)
        if not row_units and len(chunk_text) <= max_plain_parse_chars:
            row_units = _extract_plain_table_rows(chunk_text, hints, query)

        if row_units:
            normalized_units: List[dict] = []
            for unit in row_units:
                merged_unit = dict(unit)
                if not merged_unit.get("table_caption") and item.get("table_caption"):
                    merged_unit["table_caption"] = item.get("table_caption")
                if not merged_unit.get("table_id") and item.get("table_id"):
                    merged_unit["table_id"] = item.get("table_id")
                if not merged_unit.get("table_header") and item.get("table_header"):
                    merged_unit["table_header"] = item.get("table_header")
                normalized_units.append(merged_unit)
            row_units = normalized_units

        if target_tables:
            explicit_match_units = [
                unit for unit in row_units if _has_explicit_numeric_table_match(unit, target_tables)
            ]
            if explicit_match_units:
                row_units = explicit_match_units
            else:
                # 显式表号题优先拒绝 mixed-table markdown 误标；如果首轮行抽取未能
                # 产出 exact table 命中，则退回到 scoped plain/markdown 抽取再试一次。
                fallback_units = []
                if not has_structured_rows and len(chunk_text) <= max_plain_parse_chars:
                    fallback_units = _extract_plain_table_rows(chunk_text, hints, query)
                if fallback_units:
                    normalized_fallback_units: List[dict] = []
                    for unit in fallback_units:
                        merged_unit = dict(unit)
                        if not merged_unit.get("table_caption") and item.get("table_caption"):
                            merged_unit["table_caption"] = item.get("table_caption")
                        if not merged_unit.get("table_id") and item.get("table_id"):
                            merged_unit["table_id"] = item.get("table_id")
                        if not merged_unit.get("table_header") and item.get("table_header"):
                            merged_unit["table_header"] = item.get("table_header")
                        normalized_fallback_units.append(merged_unit)
                    row_units = [
                        unit for unit in normalized_fallback_units
                        if _has_explicit_numeric_table_match(unit, target_tables)
                    ]
                else:
                    # 显式表号题只允许 exact caption/table_id 命中的 generic rows 升格，
                    # 避免 side-by-side / mixed-page 文本把相邻表的数字伪装成目标表行。
                    row_units = []

        ranked_units: List[tuple[float, dict]] = []
        for unit in row_units:
            row_text = unit["row_text"].strip()
            row_lower = row_text.lower()
            row_numbers = (unit.get("row_numbers") or row_text).strip()
            typed_row_unit = bool(unit.get("evidence_unit_id"))
            focused_row = _build_query_focused_table_row(unit, hints)
            focused_text = focused_row.get("text", "")
            focus_backbone_hit = bool(focused_row.get("matched_backbone"))
            focus_column_coverage = int(focused_row.get("column_coverage", 0) or 0)
            row_method_scope = " ".join(
                part.lower()
                for part in (
                    unit.get("row_id", ""),
                    focused_text,
                    row_text,
                )
                if part
            )
            combined_lower = " ".join(
                part.lower()
                for part in (
                    unit.get("table_caption", ""),
                    unit.get("table_header", ""),
                    focused_text,
                    row_text,
                )
                if part
            )
            normalized_row = re.sub(r"\s+", "", combined_lower)
            row_method_key = _normalize_numeric_table_method_token(unit.get("row_id", ""))
            row_method_hit = bool(target_methods and row_method_key in target_methods)
            composite_target_noise = (
                bundle_query
                and bool(target_methods)
                and not explicit_comparison_methods
                and not row_method_hit
                and _is_composite_numeric_row_id(unit.get("row_id", ""))
                and _row_mentions_target_method(unit.get("row_id", ""), target_methods)
            )
            dataset_mentions = _extract_numeric_table_dataset_mentions(combined_lower)
            row_column_hits = sum(1 for value in target_columns if value in combined_lower)
            row_number_hits = len(_extract_numeric_value_tokens(row_numbers))
            table_caption = unit.get("table_caption", "") or unit.get("table_id", "")
            table_scope_backbone = ""
            if target_backbones and typed_row_unit:
                table_scope = " ".join(
                    str(unit.get(key) or "")
                    for key in ("table_caption", "table_header", "table_id")
                ).lower()
                for backbone in sorted(target_backbones, key=len, reverse=True):
                    if backbone and backbone in table_scope:
                        table_scope_backbone = backbone
                        break
            effective_backbone_hit = focus_backbone_hit or bool(table_scope_backbone)
            lexical = _compute_lexical_evidence_score(
                query,
                f"{table_caption} {unit.get('table_header', '')} {focused_text or row_text}",
            )
            strict_table_match = bool(target_tables and any(value in table_caption.lower() for value in target_tables))
            strict_dataset_match = bool(target_datasets and dataset_mentions & target_datasets)
            sort_bonus = _numeric_table_sort_bonus(unit, query, hints)
            allow_competitor_row = (
                bundle_query
                and not explicit_comparison_methods
                and row_column_hits >= max(1, min(len(target_columns), 2))
                and (not target_backbones or effective_backbone_hit)
                and (not target_columns or focus_column_coverage >= min(len(target_columns), 1))
                and not composite_target_noise
            )
            comparison_competitor_row = bool(
                comparison_query
                and len(target_methods) >= 1
                and not row_method_hit
                and (row_column_hits >= max(1, min(len(target_columns), 1)) if target_columns else True)
            )
            if target_datasets and not strict_dataset_match and not strict_table_match:
                continue
            if composite_target_noise:
                continue
            if explicit_comparison_methods and target_methods and not row_method_hit:
                continue
            # 显式 comparator 题只保留 query 中明确点名的方法，避免把 CE / 复合方法行混进 bundle。
            if target_methods and not row_method_hit and not allow_competitor_row and not comparison_competitor_row:
                continue
            if target_backbones and row_method_hit and not effective_backbone_hit:
                continue
            if target_columns and row_column_hits == 0 and not row_method_hit:
                continue
            if target_columns and len(target_columns) >= 2 and row_column_hits < 2 and not row_method_hit:
                continue
            if target_columns and row_method_hit and focus_column_coverage == 0:
                continue
            min_required_row_numbers = 1 if typed_row_unit else min_row_number_hits
            if row_number_hits < min_required_row_numbers:
                continue
            lexical_threshold = 0.18 if target_methods else 0.12
            if strict_table_match:
                lexical_threshold -= 0.03
            if strict_dataset_match:
                lexical_threshold -= 0.02
            exact_row_override = bool(
                strict_table_match
                and sort_bonus > 0
                and (
                    not target_columns
                    or focus_column_coverage >= max(1, min(len(target_columns), 1))
                )
            )
            if lexical < lexical_threshold and not exact_row_override:
                continue

            strength = lexical
            if row_method_hit:
                strength += 0.2
            elif allow_competitor_row:
                strength += 0.06
            elif comparison_competitor_row:
                strength += 0.05
            strength += min(row_column_hits, 3) * 0.04
            if strict_table_match:
                strength += 0.08
            if strict_dataset_match:
                strength += 0.09
            if focused_text:
                strength += 0.08 + min(focus_column_coverage, 4) * 0.03
            strength += sort_bonus
            ranked_units.append((strength, unit))

        seen_ranked_row_keys: set[str] = set()
        for strength, unit in sorted(ranked_units, key=lambda item: item[0], reverse=True):
            row_text = unit["row_text"].strip()
            focused_row = _build_query_focused_table_row(unit, hints)
            focused_text = (focused_row.get("text") or row_text).strip()
            normalized_row_key = _normalize_numeric_table_method_token(unit.get("row_id", ""))
            dedupe_key = normalized_row_key or re.sub(
                r"\s+",
                " ",
                (unit.get("row_id") or focused_text),
            ).strip().lower()
            if dedupe_key and dedupe_key in seen_ranked_row_keys:
                continue
            lexical = _compute_lexical_evidence_score(
                query,
                f"{unit.get('table_caption', '')} {unit.get('table_header', '')} {focused_text}",
            )
            key = (focused_text.lower(), int(item.get("page") or 0))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if dedupe_key:
                seen_ranked_row_keys.add(dedupe_key)

            similarity = float(item.get("similarity", 0.0) or 0.0) + min(0.22, lexical * 0.30) + min(0.18, strength * 0.18)
            similarity_percent = min(99.99, round(similarity * 100, 2))
            candidate = {
                **item,
                "chunk": focused_text,
                "raw_chunk_text": row_text,
                "highlight_text": focused_text[:240],
                "snippet": focused_text[:160],
                "chunk_type": "table_row",
                "block_type": "table_row",
                "table_row_evidence": True,
                "table_caption": unit.get("table_caption", ""),
                "table_header": unit.get("table_header", ""),
                "table_id": unit.get("table_id", ""),
                "table_bundle_id": (
                    unit.get("table_bundle_id")
                    or item.get("table_bundle_id", "")
                    or _fallback_page_text_table_bundle_id(unit.get("table_id") or item.get("table_id", ""))
                ),
                "evidence_unit_id": unit.get("evidence_unit_id", ""),
                "row_id": unit.get("row_id", ""),
                "table_row_number": unit.get("row_number"),
                "table_focus_backbone": focused_row.get("matched_backbone", "") or table_scope_backbone,
                "table_focus_columns": focused_row.get("resolved_columns", []),
                "table_row_raw_text": row_text,
                "table_row_boundary_text": row_text,
                "table_row_bbox": unit.get("bounding_box") or unit.get("bbox") or [],
                "cell_evidence_units": unit.get("cell_evidence_units", []),
                "cell_evidence_unit_ids": unit.get("cell_evidence_unit_ids", []),
                "table_row_slice_kind": "exact",
                "similarity": similarity,
                "similarity_percent": similarity_percent,
                "table_row_score": round(lexical, 4),
                "table_row_strength": round(strength, 4),
                "doc_title": item.get("doc_title") or doc_title,
            }
            if include_rerank_text:
                candidate["rerank_text"] = _build_evidence_unit_text(candidate, [], [])
            expanded.append(candidate)
            if len(seen_ranked_row_keys) >= row_limit:
                break

    return expanded


def _resolve_numeric_table_signature(item: dict) -> str:
    page = item.get("page") or 0
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0

    table_id = (item.get("table_id") or "").strip()
    if not table_id:
        raw_text = (item.get("raw_chunk_text") or item.get("chunk") or "").strip()
        caption = _extract_table_caption_from_text(raw_text)
        table_id = _extract_table_id(caption)
    if table_id:
        return f"{table_id.lower()}|p{page}"

    group_id = (item.get("group_id") or "").strip()
    if group_id:
        return f"group:{group_id.lower()}|p{page}"
    return ""


def _dedupe_numeric_table_evidence_units(results: List[dict], query: str) -> List[dict]:
    if not should_apply_numeric_table_specialization():
        return results
    needs = set(_analyze_evidence_need(query) or [])
    if not results or not ({"numeric_table", "comparison_multi_aspect"} & needs):
        return results

    row_signatures = {
        _resolve_numeric_table_signature(item)
        for item in results
        if (item.get("chunk_type") or item.get("block_type") or "").strip().lower() == "table_row"
    }
    row_signatures.discard("")
    if not row_signatures:
        return results

    deduped: List[dict] = []
    removed = 0
    for item in results:
        chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
        if chunk_type == "table_row":
            deduped.append(item)
            continue

        signature = _resolve_numeric_table_signature(item)
        raw_text = (item.get("raw_chunk_text") or item.get("chunk") or "").strip()
        if (
            signature
            and signature in row_signatures
            and not item.get("numeric_table_keep_support")
            and (
                chunk_type in {"table", "caption"}
                or _looks_like_numeric_table_support(raw_text, chunk_type)
            )
        ):
            removed += 1
            continue
        deduped.append(item)

    if removed:
        logger.info(f"[NumericTableDedup] 移除与 table_row 重复的原始表格证据 {removed} 条")
    return deduped


def _mark_numeric_table_support_chunks(results: List[dict], query: str) -> List[dict]:
    if not should_apply_numeric_table_specialization():
        return results
    needs = set(_analyze_evidence_need(query) or [])
    if not results or not ({"numeric_table", "comparison_multi_aspect"} & needs):
        return results

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    comparison_query = bool(hints.get("comparison"))
    preferred_sort_column = _preferred_numeric_table_sort_column(query, hints)
    if not comparison_query and not preferred_sort_column:
        return results

    target_methods = _extract_numeric_table_row_method_targets(hints)
    target_datasets = _extract_numeric_table_dataset_mentions(" ".join(hints.get("datasets", [])))
    target_backbones = {
        str(value or "").strip().lower()
        for value in hints.get("backbones", [])
        if value
    }
    target_tables = {
        value.lower()
        for value in hints.get("table_labels", [])
        if value
    } | {
        table_id.lower()
        for _, _, table_id in _extract_table_mentions(query)
        if table_id
    }
    preferred_column_key = _normalize_numeric_column_name(preferred_sort_column) if preferred_sort_column else ""
    anchor_table_ids: set[str] = set()

    def _chunk_type(item: dict) -> str:
        return (item.get("chunk_type") or item.get("block_type") or "").strip().lower()

    def _table_id(item: dict) -> str:
        explicit_table = _extract_table_id(
            (item.get("table_id") or item.get("table_caption") or "").strip()
        )
        if explicit_table:
            return explicit_table.lower()
        evidence_text = _build_numeric_table_evidence_text(item) or item.get("raw_chunk_text") or item.get("chunk") or ""
        explicit_table = _extract_table_id(evidence_text)
        return explicit_table.lower() if explicit_table else ""

    def _support_matches_target_profile(item: dict) -> bool:
        if _chunk_type(item) not in {"table", "caption"}:
            return False
        table_id = _table_id(item)
        if target_tables and (not table_id or table_id not in target_tables):
            return False

        evidence_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        )
        if not evidence_text:
            return False

        if target_datasets:
            dataset_mentions = _extract_numeric_table_dataset_mentions(evidence_text)
            if not (dataset_mentions & target_datasets):
                return False

        if target_backbones:
            candidate_backbones = {
                backbone.strip().lower()
                for backbone in _extract_table_header_backbones(item.get("table_header", "") or evidence_text)
                if backbone
            }
            if not candidate_backbones or not (candidate_backbones & target_backbones):
                return False

        if preferred_column_key:
            candidate_columns = {
                _normalize_numeric_column_name(value)
                for value in (
                    list(item.get("table_focus_columns") or [])
                    + _extract_table_header_columns(item.get("table_header", "") or evidence_text)
                )
                if value
            }
            if preferred_column_key not in candidate_columns:
                return False

        return bool(table_id or target_datasets or target_backbones or preferred_column_key)

    for item in results:
        if _chunk_type(item) != "table_row":
            continue
        row_key = _normalize_numeric_table_method_token(item.get("row_id", ""))
        if comparison_query and len(target_methods) >= 2 and target_methods and row_key not in target_methods:
            continue
        table_id = _table_id(item)
        if table_id:
            anchor_table_ids.add(table_id)

    if not anchor_table_ids:
        for item in results:
            if _chunk_type(item) == "table_row":
                table_id = _table_id(item)
                if table_id:
                    anchor_table_ids.add(table_id)
                    break

    for item in results:
        chunk_type = _chunk_type(item)
        if chunk_type not in {"table", "caption"}:
            continue
        table_id = _table_id(item)
        if (table_id and table_id in anchor_table_ids) or _support_matches_target_profile(item):
            item["numeric_table_keep_support"] = True

    return results



def _attach_numeric_table_exact_context(item: dict, anchor_row: dict) -> None:
    """把 exact row 绑到同表 support 上，避免 live context_segments 丢掉关键数值。"""
    if not item or not anchor_row:
        return
    exact_row_text = _get_numeric_table_boundary_text(anchor_row)
    if not exact_row_text:
        return
    item["numeric_table_exact_context_row_text"] = exact_row_text
    item["numeric_table_exact_context_caption"] = (
        anchor_row.get("table_caption")
        or item.get("table_caption")
        or item.get("numeric_table_exact_context_caption")
        or ""
    )
    item["numeric_table_exact_context_header"] = (
        anchor_row.get("table_header")
        or item.get("table_header")
        or item.get("numeric_table_exact_context_header")
        or ""
    )


def _apply_numeric_table_same_bundle_hard_gate(results: List[dict], query: str) -> List[dict]:
    if not should_apply_numeric_table_specialization():
        return results
    needs = set(_analyze_evidence_need(query) or [])
    if not results or not ({"numeric_table", "comparison_multi_aspect"} & needs):
        return results

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    comparison_query = bool(hints.get("comparison"))
    cost_query = _is_numeric_table_cost_query(query)
    target_methods = _extract_numeric_table_row_method_targets(hints)
    target_datasets = _extract_numeric_table_dataset_mentions(" ".join(hints.get("datasets", [])))
    target_backbones = {
        str(value or "").strip().lower()
        for value in hints.get("backbones", [])
        if value
    }
    target_columns = {
        _normalize_numeric_column_name(value).lower()
        for value in hints.get("columns", [])
        if value
    }
    target_tables = {
        value.lower()
        for value in hints.get("table_labels", [])
        if value
    } | {
        table_id.lower()
        for _, _, table_id in _extract_table_mentions(query)
        if table_id
    }
    preferred_sort_column = _preferred_numeric_table_sort_column(query, hints)
    bundle_query = _is_numeric_table_bundle_query(query, hints)
    explicit_comparator_mode = _is_numeric_table_explicit_comparator_query(query, hints)
    bundle_query = bundle_query or explicit_comparator_mode
    winner_only_mode = bool(
        preferred_sort_column
        or (target_tables and target_columns and not comparison_query)
    )

    def _chunk_type(item: dict) -> str:
        return (item.get("chunk_type") or item.get("block_type") or "").strip().lower()

    def _bundle_table_id(item: dict) -> str:
        explicit_table = _extract_table_id(
            (item.get("table_id") or item.get("table_caption") or "").strip()
        )
        if explicit_table:
            return explicit_table.lower()
        evidence_text = _build_numeric_table_evidence_text(item) or item.get("raw_chunk_text") or item.get("chunk") or ""
        explicit_table = _extract_table_id(evidence_text)
        return explicit_table.lower() if explicit_table else ""

    def _bundle_key(item: dict) -> str:
        evidence_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        )
        table_id = _bundle_table_id(item) or "-"
        dataset_mentions = _extract_numeric_table_dataset_mentions(evidence_text)
        dataset_key = ",".join(sorted(dataset_mentions)) if dataset_mentions else "-"
        backbone_key = (item.get("table_focus_backbone") or "").strip().lower()
        if not backbone_key:
            header_backbones = _extract_table_header_backbones(item.get("table_header", "") or evidence_text)
            if header_backbones:
                backbone_key = header_backbones[0].strip().lower()
        focus_columns = [
            _normalize_numeric_column_name(value)
            for value in (item.get("table_focus_columns") or [])
            if value
        ]
        if not focus_columns:
            focus_columns = _sort_numeric_columns(
                _extract_table_header_columns(item.get("table_header", "") or evidence_text)
            )
        column_key = ",".join(
            sorted(
                value.lower()
                for value in focus_columns
                if value
            )
        ) if focus_columns else "-"
        return "|".join((table_id, dataset_key, backbone_key or "-", column_key or "-"))

    def _is_composite_target_noise(item: dict) -> bool:
        if _chunk_type(item) != "table_row":
            return False
        if not bundle_query or not target_methods or explicit_comparator_mode:
            return False
        row_id = item.get("row_id", "")
        return (
            _is_composite_numeric_row_id(row_id)
            and _row_mentions_target_method(row_id, target_methods)
            and _normalize_numeric_table_method_token(row_id) not in target_methods
        )

    def _looks_like_headerish_anchor(item: dict) -> bool:
        return _is_headerish_numeric_table_row(item)

    def _is_cost_anchor_text(item: dict) -> bool:
        if not cost_query:
            return False
        return _is_numeric_table_cost_anchor_text(item)

    def _build_focused_row(item: dict) -> dict:
        row_text = re.sub(
            r"\s+",
            " ",
            str(
                item.get("numeric_table_exact_context_row_text")
                or _get_numeric_table_boundary_text(item)
                or item.get("table_row_raw_text")
                or item.get("row_text")
                or item.get("chunk")
                or item.get("raw_chunk_text")
                or ""
            ),
        ).strip()
        if not row_text:
            return {}
        return _build_query_focused_table_row(
            {
                "row_id": item.get("row_id") or "",
                "row_text": row_text,
                "row_numbers": _strip_leading_numeric_table_row_id(row_text, item.get("row_id") or "") or row_text,
                "table_caption": item.get("numeric_table_exact_context_caption")
                or item.get("table_caption")
                or item.get("table_id")
                or "",
                "table_id": item.get("table_id") or "",
                "table_header": item.get("numeric_table_exact_context_header")
                or item.get("table_header")
                or "",
                "table_focus_columns": list(item.get("table_focus_columns") or []),
            },
            hints,
        )

    def _matches_target_dataset(item: dict) -> bool:
        if not target_datasets or cost_query:
            return True
        evidence_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        )
        dataset_mentions = _extract_numeric_table_dataset_mentions(evidence_text)
        return bool(dataset_mentions & target_datasets)

    def _matches_target_backbone(item: dict) -> bool:
        if not target_backbones or cost_query:
            return True
        candidate_backbones = {
            str(item.get("table_focus_backbone") or "").strip().lower()
        } if item.get("table_focus_backbone") else set()
        evidence_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        )
        candidate_backbones.update(
            value.strip().lower()
            for value in _extract_table_header_backbones(item.get("table_header", "") or evidence_text)
            if value
        )
        focused_row = _build_focused_row(item)
        matched_backbone = str(focused_row.get("matched_backbone") or "").strip().lower()
        if matched_backbone:
            candidate_backbones.add(matched_backbone)
        return bool(candidate_backbones & target_backbones)

    def _matches_target_columns(item: dict) -> bool:
        if not target_columns or cost_query:
            return True
        evidence_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        )
        candidate_columns = {
            _normalize_numeric_column_name(value).lower()
            for value in (item.get("table_focus_columns") or [])
            if value
        }
        candidate_columns.update(
            _normalize_numeric_column_name(value).lower()
            for value in _extract_table_header_columns(item.get("table_header", "") or evidence_text)
            if value
        )
        focused_row = _build_focused_row(item)
        candidate_columns.update(
            _normalize_numeric_column_name(value).lower()
            for value in (focused_row.get("resolved_columns") or [])
            if value
        )
        return bool(candidate_columns & target_columns)

    def _matches_target_hard_gate(item: dict) -> bool:
        return (
            _matches_target_columns(item)
            and _matches_target_dataset(item)
            and _matches_target_backbone(item)
        )

    def _matches_target_table(item: dict) -> bool:
        if not target_tables:
            return True
        item_table_id = _bundle_table_id(item)
        if item_table_id and item_table_id in target_tables:
            return True
        evidence_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        )
        explicit_table = _extract_table_id(evidence_text)
        return bool(explicit_table and explicit_table.lower() in target_tables)

    def _fallback_without_anchor() -> List[dict]:
        fallback: List[dict] = []
        seen_ids: set[int] = set()

        def _append(item: dict) -> None:
            item_id = id(item)
            if item_id in seen_ids:
                return
            fallback.append(item)
            seen_ids.add(item_id)

        if cost_query:
            for item in results:
                if _is_cost_anchor_text(item):
                    _append(item)
            if fallback:
                return fallback

        row_candidates = [
            item
            for item in results
            if _chunk_type(item) == "table_row"
            and not _is_composite_target_noise(item)
            and (not target_methods or _normalize_numeric_table_method_token(item.get("row_id", "")) in target_methods)
            and _matches_target_hard_gate(item)
            and _matches_target_table(item)
        ]
        for item in row_candidates:
            _append(item)

        table_support_candidates = [
            item
            for item in results
            if _chunk_type(item) in {"table", "caption"}
            and _matches_target_table(item)
            and _matches_target_hard_gate(item)
        ]
        for item in table_support_candidates:
            _append(item)

        return fallback

    row_items = [item for item in results if _chunk_type(item) == "table_row"]
    if not row_items:
        fallback = _fallback_without_anchor()
        return fallback or results

    def _select_anchor_row(items: List[dict]) -> Optional[dict]:
        candidates: List[tuple[int, dict]] = []
        for idx, item in enumerate(items):
            if _is_composite_target_noise(item):
                continue
            if target_methods:
                row_key = _normalize_numeric_table_method_token(item.get("row_id", ""))
                if row_key not in target_methods:
                    continue
            if (winner_only_mode or explicit_comparator_mode or comparison_query) and not _matches_target_hard_gate(item):
                continue
            candidates.append((idx, item))
        if not candidates:
            return None

        def _is_exact_row_support(item: dict) -> bool:
            return bool(
                item.get("table_row_evidence")
                or item.get("table_row_slice_kind") == "exact"
                or item.get("numeric_table_exact_context_row_text")
                or item.get("evidence_units")
                or item.get("cell_evidence_units")
            )

        bundle_stats: dict[str, tuple[int, int]] = {}
        for _idx, item in candidates:
            bundle_key = _bundle_key(item)
            if not bundle_key:
                continue
            bundle_count, bundle_exact_count = bundle_stats.get(bundle_key, (0, 0))
            bundle_count += 1
            if _is_exact_row_support(item):
                bundle_exact_count += 1
            bundle_stats[bundle_key] = (bundle_count, bundle_exact_count)

        if winner_only_mode or bundle_query or explicit_comparator_mode:
            ranked: List[tuple[int, int, int, int, float, float, float, int, dict]] = []
            for idx, item in candidates:
                bundle_key = _bundle_key(item)
                bundle_count, bundle_exact_count = bundle_stats.get(bundle_key, (0, 0))
                ranked.append(
                    (
                        bundle_exact_count,
                        bundle_count,
                        1 if _is_exact_row_support(item) else 0,
                        0 if _looks_like_headerish_anchor(item) else 1,
                        float(_numeric_table_sort_bonus(item, query, hints) or 0.0),
                        float(item.get("numeric_table_priority", 0.0) or 0.0),
                        float(item.get("similarity", 0.0) or 0.0),
                        -idx,
                        item,
                    )
                )
            ranked.sort(reverse=True)
            if ranked:
                return ranked[0][8]
        return candidates[0][1]

    allowed_bundle_keys: set[str] = set()
    allowed_table_ids: set[str] = set()
    allowed_row_ids: set[str] = set()
    anchor_rows: List[dict] = []

    if explicit_comparator_mode:
        for item in row_items:
            row_key = _normalize_numeric_table_method_token(item.get("row_id", ""))
            if row_key in target_methods and not _is_composite_target_noise(item):
                if not _matches_target_hard_gate(item):
                    continue
                anchor_rows.append(item)
        if not anchor_rows:
            fallback = _fallback_without_anchor()
            return fallback or results
        for item in anchor_rows:
            bundle_key = _bundle_key(item)
            if bundle_key:
                allowed_bundle_keys.add(bundle_key)
            table_id = _bundle_table_id(item)
            if table_id:
                allowed_table_ids.add(table_id)
            row_key = _normalize_numeric_table_method_token(item.get("row_id", ""))
            if row_key:
                allowed_row_ids.add(row_key)
    else:
        anchor_row = _select_anchor_row(row_items)
        if anchor_row is None:
            fallback = _fallback_without_anchor()
            return fallback or results
        anchor_rows.append(anchor_row)
        bundle_key = _bundle_key(anchor_row)
        if bundle_key:
            allowed_bundle_keys.add(bundle_key)
        table_id = _bundle_table_id(anchor_row)
        if table_id:
            allowed_table_ids.add(table_id)
        if winner_only_mode:
            row_key = _normalize_numeric_table_method_token(anchor_row.get("row_id", ""))
            if row_key:
                allowed_row_ids.add(row_key)

    filtered: List[dict] = []
    for item in results:
        chunk_type = _chunk_type(item)
        if chunk_type not in {"table_row", "table", "caption"}:
            if _is_cost_anchor_text(item):
                filtered.append(item)
            continue
        if _is_composite_target_noise(item):
            continue

        item_table_id = _bundle_table_id(item)
        item_bundle_key = _bundle_key(item)
        exact_context_anchor = None
        if item_table_id:
            exact_context_anchor = next(
                (
                    anchor
                    for anchor in anchor_rows
                    if _bundle_table_id(anchor) == item_table_id
                ),
                None,
            )
        if exact_context_anchor is None and len(anchor_rows) == 1:
            exact_context_anchor = anchor_rows[0]
        if chunk_type == "table_row":
            row_key = _normalize_numeric_table_method_token(item.get("row_id", ""))
            if explicit_comparator_mode:
                if row_key not in allowed_row_ids:
                    continue
                if not _matches_target_hard_gate(item):
                    continue
            elif winner_only_mode:
                if not _matches_target_hard_gate(item):
                    continue
                if target_methods:
                    if allowed_row_ids and row_key not in allowed_row_ids:
                        continue
                elif not target_tables:
                    if allowed_bundle_keys and item_bundle_key not in allowed_bundle_keys:
                        continue
                elif allowed_row_ids and row_key not in allowed_row_ids:
                    continue
            elif allowed_bundle_keys and item_bundle_key not in allowed_bundle_keys:
                continue

        if item_table_id and allowed_table_ids and item_table_id not in allowed_table_ids:
            if chunk_type == "table_row":
                continue
            if chunk_type not in {"table", "caption"}:
                continue

        if chunk_type in {"table", "caption"} and not item_table_id:
            continue

        if chunk_type in {"table", "caption"} and allowed_table_ids and item_table_id not in allowed_table_ids:
            continue

        if chunk_type in {"table", "caption"} and exact_context_anchor is not None:
            _attach_numeric_table_exact_context(item, exact_context_anchor)

        if chunk_type == "table_row" and not explicit_comparator_mode and not winner_only_mode:
            if allowed_bundle_keys and item_bundle_key not in allowed_bundle_keys:
                continue

        filtered.append(item)

    if not any(_chunk_type(item) == "table_row" for item in filtered):
        if filtered and all(
            _is_cost_anchor_text(item) or _chunk_type(item) in {"table", "caption"}
            for item in filtered
        ):
            return filtered
        recovered: List[dict] = []
        seen_ids: set[int] = set()
        for item in anchor_rows + filtered:
            item_id = id(item)
            if item_id in seen_ids:
                continue
            recovered.append(item)
            seen_ids.add(item_id)
        if any(_chunk_type(item) == "table_row" for item in recovered):
            return recovered
        return results
    return filtered



def _apply_group_pre_cap(results: List[dict], per_group_limit: int = 4) -> Tuple[List[dict], dict]:
    if not results or per_group_limit <= 0:
        return results, {}

    kept = []
    group_counts = {}
    removed = 0

    for item in results:
        group_id = (item.get("group_id") or "").strip()
        if group_id:
            count = group_counts.get(group_id, 0)
            if count >= per_group_limit:
                removed += 1
                continue
            group_counts[group_id] = count + 1
        kept.append(item)

    stats = {
        "input": len(results),
        "output": len(kept),
        "removed": removed,
        "group_limit": per_group_limit,
        "unique_groups": len(group_counts),
    }
    if removed > 0:
        logger.info(f"[RerankPreCap] 候选 {len(results)} → {len(kept)}，移除同组冗余 {removed} 条")
    return kept, stats


def _apply_page_pre_cap(results: List[dict], per_page_limit: int = 2) -> Tuple[List[dict], dict]:
    if not results or per_page_limit <= 0:
        return results, {}

    kept = []
    page_counts = {}
    removed = 0
    for item in results:
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            count = page_counts.get(page, 0)
            if count >= per_page_limit:
                removed += 1
                continue
            page_counts[page] = count + 1
        kept.append(item)

    stats = {
        "input": len(results),
        "output": len(kept),
        "removed": removed,
        "page_limit": per_page_limit,
        "unique_pages": len(page_counts),
    }
    if removed > 0:
        logger.info(f"[RerankPagePreCap] 候选 {len(results)} → {len(kept)}，移除同页冗余 {removed} 条")
    return kept, stats


def _apply_group_post_cap(
    results: List[dict],
    top_k: int,
    per_group_limit: int = 2,
    per_section_limit: int = 2,
) -> Tuple[List[dict], dict]:
    if not results or top_k <= 0:
        return [], {}

    selected = []
    selected_ids = set()
    group_counts = {}
    section_counts = {}
    group_blocked = 0
    section_blocked = 0

    def _can_take(item: dict, enforce_section: bool) -> Tuple[bool, str]:
        group_id = (item.get("group_id") or "").strip()
        if group_id and group_counts.get(group_id, 0) >= per_group_limit:
            return False, "group"
        section = _normalize_section_heading(item.get("chunk_heading", ""))
        if enforce_section and section and section_counts.get(section, 0) >= per_section_limit:
            return False, "section"
        return True, ""

    def _take(item: dict, track_section: bool) -> None:
        group_id = (item.get("group_id") or "").strip()
        if group_id:
            group_counts[group_id] = group_counts.get(group_id, 0) + 1
        section = _normalize_section_heading(item.get("chunk_heading", ""))
        if track_section and section:
            section_counts[section] = section_counts.get(section, 0) + 1
        selected.append(item)
        selected_ids.add(id(item))

    for item in results:
        ok, reason = _can_take(item, True)
        if not ok:
            if reason == "group":
                group_blocked += 1
            elif reason == "section":
                section_blocked += 1
            continue
        _take(item, True)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for item in results:
            if id(item) in selected_ids:
                continue
            ok, reason = _can_take(item, False)
            if not ok:
                if reason == "group":
                    group_blocked += 1
                continue
            _take(item, False)
            if len(selected) >= top_k:
                break

    stats = {
        "input": len(results),
        "output": len(selected),
        "group_limit": per_group_limit,
        "section_limit": per_section_limit,
        "group_blocked": group_blocked,
        "section_blocked": section_blocked,
        "unique_groups": len(group_counts),
        "unique_sections": len(section_counts),
    }
    return selected[:top_k], stats


def _focus_mode_compress(
    results: List[dict],
    query: str,
    window_size: int = 2,
    max_sentences: int = 4,
    min_chars: int = 80,
) -> List[dict]:
    """对 rerank 后的候选列表做句级 Focus Mode 压缩。

    策略：
    1. 将 chunk 文本按句子切分
    2. 用 query 关键词为每句打分（命中词数）
    3. 选出 top max_sentences 句，并各自扩展 window_size 个上下文句形成窗口
    4. 合并去重窗口后拼接为 focus_text，替换 chunk 用于 LLM 上下文
    5. 保留 focus_original_chars / focus_sentences_count / focus_compression_ratio 供诊断
    6. citation / highlight 追溯字段（chunk_id, parent_id, page, group_id）原样保留

    文本低于 min_chars 或切不出多句时直接跳过，保留原始 chunk。
    """
    from services.sentence_window_splitter import split_sentences

    if not results or not query:
        return results

    # 提取 query 关键词（≥2字符）
    import re as _re
    query_terms = [
        t.lower() for t in _re.split(r'[\s,;，。；、？！?!：:""\'\'\"\"]+', query)
        if len(t) >= 2
    ]
    numeric_table_query = "numeric_table" in (_analyze_evidence_need(query) or [])
    cost_query = numeric_table_query and _is_numeric_table_cost_query(query)

    for item in results:
        chunk_text = (item.get("chunk") or "").strip()
        if not chunk_text or len(chunk_text) < min_chars:
            continue
        chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
        source_text = (item.get("raw_chunk_text") or chunk_text).strip()
        if numeric_table_query and (
            chunk_type in {"table_row", "table", "caption"}
            or _looks_like_numeric_table_support(source_text, chunk_type)
            or (cost_query and _has_numeric_table_cost_anchor(source_text))
        ):
            continue

        sentences = split_sentences(chunk_text)
        if len(sentences) <= 2:
            continue

        # 为每句打分
        def _score(sent: str) -> float:
            sl = sent.lower()
            return sum(1 for t in query_terms if t in sl)

        scored = [(i, _score(s), s) for i, s in enumerate(sentences)]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 选 top max_sentences 个支持句（保持原顺序）
        top_indices = sorted({i for i, sc, _ in scored[:max_sentences]})

        # 展开窗口
        window_indices: set = set()
        for idx in top_indices:
            for wi in range(max(0, idx - window_size), min(len(sentences), idx + window_size + 1)):
                window_indices.add(wi)
        window_indices_sorted = sorted(window_indices)

        focus_text = " ".join(sentences[i] for i in window_indices_sorted).strip()
        if not focus_text:
            continue

        original_chars = len(chunk_text)
        focus_chars = len(focus_text)
        compression_ratio = round(focus_chars / max(original_chars, 1), 4)

        # 只有压缩率低于 0.95 才覆盖（避免无意义替换）
        if compression_ratio >= 0.95:
            continue

        item["focus_original_chunk"] = chunk_text
        item["focus_original_chars"] = original_chars
        item["focus_sentences_count"] = len(window_indices_sorted)
        item["focus_compression_ratio"] = compression_ratio
        item["chunk"] = focus_text

    return results


def _ensure_numeric_table_evidence_slots(
    results: List[dict],
    query: str,
    top_k: int,
) -> List[dict]:
    if not should_apply_numeric_table_specialization():
        return results[:top_k] if results else []
    if not results or top_k <= 0:
        return []
    if "numeric_table" not in (_analyze_evidence_need(query) or []):
        return results[:top_k]

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    effective_top_k = _resolve_numeric_table_effective_top_k(query, top_k, hints, results)
    target_tables = {
        value.lower()
        for value in hints.get("table_labels", [])
        if value
    } | {
        table_id.lower()
        for _, _, table_id in _extract_table_mentions(query)
        if table_id
    }
    require_explicit_table_anchor = _should_require_explicit_table_anchor(hints)
    ordered_target_method_keys = [
        _normalize_numeric_table_method_token(value)
        for value in hints.get("methods", [])
        if value and _normalize_numeric_table_method_token(value)
    ]
    ordered_target_method_keys = [
        value for value in ordered_target_method_keys if value not in _NUMERIC_TABLE_METHOD_STOPWORDS
    ]
    target_method_keys = set(ordered_target_method_keys)
    comparison_query = bool(hints.get("comparison"))
    preferred_sort_column = _preferred_numeric_table_sort_column(query, hints)
    bundle_query = _is_numeric_table_bundle_query(query, hints)
    second_best_mode = bool(_NUMERIC_TABLE_SECOND_BEST_RE.search(query or ""))
    explicit_comparison_methods = _is_numeric_table_explicit_comparator_query(query, hints)
    target_datasets = _extract_numeric_table_dataset_mentions(" ".join(hints.get("datasets", [])))
    target_backbones = {
        str(value or "").lower()
        for value in hints.get("backbones", [])
        if value
    }
    target_columns = {
        _normalize_numeric_column_name(value)
        for value in hints.get("columns", [])
        if value
    }
    cost_query = _is_numeric_table_cost_query(query)
    winner_bundle_query = bool(
        preferred_sort_column
        and not target_tables
        and not target_method_keys
        and not cost_query
    )
    bundle_query = bundle_query or explicit_comparison_methods or winner_bundle_query
    min_support = 1
    if bundle_query:
        min_support = min(effective_top_k, 4 if len(hints.get("methods", [])) >= 2 else 3)
    elif len(hints.get("columns", [])) >= 3:
        min_support = min(effective_top_k, 2)

    def _chunk_type(item: dict) -> str:
        return (item.get("chunk_type") or item.get("block_type") or "").strip().lower()

    def _row_method_key(item: dict) -> str:
        return _normalize_numeric_table_method_token(item.get("row_id", ""))

    def _bundle_group_key(item: dict) -> str:
        if _chunk_type(item) != "table_row":
            return ""
        evidence_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        )
        page = item.get("page") or 0
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = 0
        table_key = (item.get("table_id") or "").strip().lower() or f"page:{page}"
        dataset_mentions = _extract_numeric_table_dataset_mentions(evidence_text)
        dataset_key = ",".join(sorted(dataset_mentions & target_datasets)) if target_datasets else ""
        backbone_key = (item.get("table_focus_backbone") or "").strip().lower()
        if target_backbones and backbone_key and backbone_key not in target_backbones:
            return ""
        resolved_columns = {
            _normalize_numeric_column_name(value)
            for value in (item.get("table_focus_columns") or [])
            if value
        }
        column_values = sorted(
            value.lower()
            for value in resolved_columns
            if value and (not target_columns or value in target_columns)
        )
        column_key = ",".join(column_values)
        return "|".join((table_key, dataset_key or "-", backbone_key or "-", column_key or "-"))

    def _is_composite_target_noise(item: dict) -> bool:
        if _chunk_type(item) != "table_row":
            return False
        if not bundle_query or not target_method_keys or explicit_comparison_methods:
            return False
        row_id = item.get("row_id", "")
        return (
            _is_composite_numeric_row_id(row_id)
            and _row_mentions_target_method(row_id, target_method_keys)
            and _row_method_key(item) not in target_method_keys
        )

    def _matches_target_table(item: dict) -> bool:
        if not target_tables:
            return False
        evidence_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        )
        if require_explicit_table_anchor:
            return _has_strict_numeric_table_anchor(item, target_tables, evidence_text)
        if _has_explicit_numeric_table_match(item, target_tables):
            return True
        if _chunk_type(item) == "table_row":
            return False
        return any(value in evidence_text.lower() for value in target_tables)

    def _is_hard_support(item: dict) -> bool:
        chunk_type = _chunk_type(item)
        evidence_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        ).strip()
        if not evidence_text:
            return False
        if chunk_type == "table_row":
            return True
        if chunk_type in {"table", "caption"}:
            if _looks_like_numeric_table_support(evidence_text, chunk_type):
                return True
            return bool(target_tables and _matches_target_table(item))
        if cost_query and _is_numeric_table_cost_anchor_text(item):
            return True
        return False

    def _support_score(item: dict) -> float:
        chunk_type = _chunk_type(item)
        raw_text = (
            _build_numeric_table_evidence_text(item)
            or item.get("raw_chunk_text")
            or item.get("chunk")
            or ""
        ).strip()
        score = float(item.get("numeric_table_priority", 0.0) or 0.0)
        if chunk_type == "table_row":
            score += 5.0
            if item.get("table_row_slice_kind") == "exact":
                score += 1.2
            elif item.get("table_row_slice_kind") == "broad":
                score -= 0.4
        elif chunk_type == "table":
            score += 3.0
        elif chunk_type == "caption":
            score += 1.5
        if cost_query and _is_numeric_table_cost_anchor_text(item):
            score += 4.5
        if _looks_like_numeric_table_support(raw_text, chunk_type):
            score += 2.0
        if item.get("row_id"):
            score += 0.8
        if item.get("table_id"):
            score += 0.6
        score += min(len(item.get("numeric_table_anchor_hits") or []), 5) * 0.4
        score += float(item.get("similarity", 0.0) or 0.0) * 0.5
        score += _numeric_table_sort_bonus(item, query, hints)
        return score

    ordered_scores = [(_support_score(item), idx, item) for idx, item in enumerate(results)]
    support_candidates = [
        (score, idx, item)
        for score, idx, item in ordered_scores
        if score >= 3.0 and _is_hard_support(item) and not _is_composite_target_noise(item)
    ]
    if not support_candidates:
        return results[:top_k]

    bundle_group_counts: dict[str, int] = {}
    bundle_group_exact_counts: dict[str, int] = {}
    for _score, _idx, item in ordered_scores:
        if _chunk_type(item) != "table_row":
            continue
        bundle_key = _bundle_group_key(item)
        if not bundle_key:
            continue
        bundle_group_counts[bundle_key] = bundle_group_counts.get(bundle_key, 0) + 1
        if (
            item.get("table_row_evidence")
            or item.get("table_row_slice_kind") == "exact"
            or item.get("numeric_table_exact_context_row_text")
        ):
            bundle_group_exact_counts[bundle_key] = bundle_group_exact_counts.get(bundle_key, 0) + 1

    final = list(results[:effective_top_k])
    protected_ids = {
        id(item)
        for item in final
        if (
            (target_tables and _matches_target_table(item))
            or (explicit_comparison_methods and _row_method_key(item) in target_method_keys)
        )
    }

    def _is_support(item: dict) -> bool:
        return _is_hard_support(item) and _support_score(item) >= 3.0

    def _prefer_exact_slice_support(items: List[dict]) -> List[dict]:
        if not items:
            return items

        cost_support: List[dict] = []
        exact_support: List[dict] = []
        fallback: List[dict] = []
        for item in items:
            chunk_type = _chunk_type(item)
            if cost_query and _is_numeric_table_cost_anchor_text(item):
                cost_support.append(item)
                continue
            if (
                chunk_type == "table_row"
                or item.get("table_row_evidence")
                or item.get("table_row_slice_kind") == "exact"
            ):
                exact_support.append(item)
                continue
            if (
                chunk_type in {"table", "caption"}
                and item.get("table_augmented_scope") != "page_content"
                and not item.get("numeric_table_mixed_table")
                ):
                exact_support.append(item)
                continue
            fallback.append(item)

        if cost_query:
            merged = cost_support + exact_support + fallback
        else:
            merged = exact_support + fallback
        return merged if merged else items

    def _dedupe_explicit_comparator_rows(items: List[dict]) -> List[dict]:
        if not explicit_comparison_methods:
            return items

        deduped: List[dict] = []
        seen_method_keys: set[str] = set()
        for item in items:
            if _chunk_type(item) != "table_row":
                deduped.append(item)
                continue

            row_key = _row_method_key(item)
            if row_key and row_key in target_method_keys:
                if row_key in seen_method_keys:
                    continue
                seen_method_keys.add(row_key)
            deduped.append(item)
        return deduped

    def _limit_second_best_bundle_rows(items: List[dict]) -> List[dict]:
        if not second_best_mode or explicit_comparison_methods:
            return items

        row_items = [item for item in items if _chunk_type(item) == "table_row"]
        max_second_best_rows = max(len(target_method_keys), 1) + 1
        if len(row_items) <= max_second_best_rows:
            return items

        preferred_bundle_keys: list[str] = []
        for item in row_items:
            row_key = _row_method_key(item)
            if target_method_keys and row_key not in target_method_keys:
                continue
            bundle_key = _bundle_group_key(item)
            if bundle_key and bundle_key not in preferred_bundle_keys:
                preferred_bundle_keys.append(bundle_key)
        if not preferred_bundle_keys:
            fallback_bundle_key = next(
                (_bundle_group_key(item) for item in row_items if _bundle_group_key(item)),
                "",
            )
            if fallback_bundle_key:
                preferred_bundle_keys.append(fallback_bundle_key)
        if not preferred_bundle_keys:
            return items

        def _second_best_rank_score(item: dict) -> float:
            focused_row = _build_query_focused_table_row(item, hints)
            column_map = focused_row.get("column_map") or {}
            all_value = _parse_numeric_table_value(column_map.get("All", ""))
            if all_value is not None:
                return all_value

            rank_values = [
                _parse_numeric_table_value(value)
                for column, value in column_map.items()
                if column in {"Many", "Med.", "Few", "Acc"}
            ]
            rank_values = [value for value in rank_values if value is not None]
            if rank_values:
                return sum(rank_values) / len(rank_values)
            return float("-inf")

        kept_row_ids: set[int] = set()
        kept_target_keys: set[str] = set()
        for item in items:
            if _chunk_type(item) != "table_row":
                continue
            if preferred_bundle_keys and _bundle_group_key(item) not in preferred_bundle_keys:
                continue
            row_key = _row_method_key(item)
            if target_method_keys:
                if row_key not in target_method_keys or row_key in kept_target_keys:
                    continue
                kept_target_keys.add(row_key)
            elif kept_row_ids:
                continue
            kept_row_ids.add(id(item))

        competitor_candidates: list[tuple[float, float, float, int, dict]] = []
        for idx, item in enumerate(items):
            if _chunk_type(item) != "table_row":
                continue
            if preferred_bundle_keys and _bundle_group_key(item) not in preferred_bundle_keys:
                continue
            row_key = _row_method_key(item)
            if target_method_keys and row_key in target_method_keys:
                continue
            competitor_candidates.append(
                (
                    _second_best_rank_score(item),
                    _support_score(item),
                    float(_numeric_table_sort_bonus(item, query, hints) or 0.0),
                    float(item.get("numeric_table_priority", 0.0) or 0.0),
                    -idx,
                    item,
                )
            )
        if competitor_candidates:
            best_competitor = max(competitor_candidates)[5]
            kept_row_ids.add(id(best_competitor))

        limited: List[dict] = []
        for item in items:
            chunk_type = _chunk_type(item)
            if chunk_type == "table_row":
                if id(item) in kept_row_ids:
                    limited.append(item)
                continue
            if (
                chunk_type in {"table", "caption"}
                or item.get("table_augmented_scope") == "page_content"
                or _looks_like_numeric_table_support(
                    _build_numeric_table_evidence_text(item)
                    or item.get("raw_chunk_text")
                    or item.get("chunk")
                    or "",
                    chunk_type,
                )
            ):
                continue
            limited.append(item)
        return limited

    def _support_candidate_priority(entry: tuple[float, int, dict]) -> tuple[int, int, int, float, int]:
        score, idx, item = entry
        chunk_type = _chunk_type(item)
        bundle_key = _bundle_group_key(item)
        bundle_exact_count = bundle_group_exact_counts.get(bundle_key, 0) if bundle_key else 0
        bundle_count = bundle_group_counts.get(bundle_key, 0) if bundle_key else 0
        if (
            chunk_type == "table_row"
            or item.get("table_row_evidence")
            or item.get("table_row_slice_kind") == "exact"
        ):
            support_rank = 0
        elif (
            chunk_type in {"table", "caption"}
            and item.get("table_augmented_scope") != "page_content"
            and not item.get("numeric_table_mixed_table")
        ):
            support_rank = 1
        else:
            support_rank = 2
        return (support_rank, -bundle_exact_count, -bundle_count, -score, idx)

    ordered_support_candidates = sorted(support_candidates, key=_support_candidate_priority)
    hard_support_pool = _limit_second_best_bundle_rows(
        _dedupe_explicit_comparator_rows(
            _prefer_exact_slice_support([item for _score, _idx, item in ordered_support_candidates])
        )
    )
    support_target_k = effective_top_k if explicit_comparison_methods else top_k
    if (
        not cost_query
        and support_target_k > 0
        and len(hard_support_pool) >= support_target_k
        and any(_chunk_type(item) == "table_row" for item in hard_support_pool)
    ):
        return hard_support_pool[:support_target_k]
    ordered_cost_anchor_candidates = [
        item
        for _score, _idx, item in sorted(ordered_scores, key=lambda entry: (-entry[0], entry[1]))
        if cost_query and _is_numeric_table_cost_anchor_text(item)
    ]

    required_supports: List[dict] = []
    seen_required_ids: set[int] = set()

    def _append_required(predicate) -> None:
        for _, _, item in ordered_support_candidates:
            item_id = id(item)
            if item_id in seen_required_ids:
                continue
            if predicate(item):
                required_supports.append(item)
                seen_required_ids.add(item_id)
                break

    if target_tables and not any(
        _matches_target_table(item) and _chunk_type(item) == "table_row" and _is_support(item)
        for item in final
    ):
        _append_required(
            lambda item: _matches_target_table(item) and _chunk_type(item) == "table_row"
        )
    if target_tables and not any(_matches_target_table(item) and _is_support(item) for item in final):
        _append_required(lambda item: _matches_target_table(item))

    if explicit_comparison_methods:
        for method_key in ordered_target_method_keys:
            if any(
                _row_method_key(item) == method_key
                and _is_support(item)
                and (not target_tables or _matches_target_table(item))
                for item in final
            ):
                continue
            _append_required(
                lambda item, method_key=method_key: (
                    _chunk_type(item) == "table_row"
                    and _row_method_key(item) == method_key
                    and (not target_tables or _matches_target_table(item))
                )
            )

    if bundle_query and target_method_keys and not any(
        _row_method_key(item) in target_method_keys
        and _is_support(item)
        and (not target_tables or _matches_target_table(item))
        for item in final
    ):
        _append_required(
            lambda item: (
                _chunk_type(item) == "table_row"
                and _row_method_key(item) in target_method_keys
                and (not target_tables or _matches_target_table(item))
            )
        )

    if cost_query and not any(_is_numeric_table_cost_anchor_text(item) for item in final):
        cost_anchor_candidates = [
            (float(item.get("similarity", 0.0) or 0.0), idx, item)
            for idx, item in enumerate(results)
            if _is_numeric_table_cost_anchor_text(item)
        ]
        if cost_anchor_candidates:
            _, _, cost_anchor_item = max(cost_anchor_candidates, key=lambda entry: (entry[0], -entry[1]))
            _append_required(lambda item, cost_anchor_item=cost_anchor_item: item is cost_anchor_item)

    if bundle_query and not explicit_comparison_methods:
        bundle_groups: dict[str, list[dict]] = {}
        bundle_group_scores: dict[str, float] = {}
        for score, _, item in ordered_support_candidates:
            if _chunk_type(item) != "table_row":
                continue
            if target_tables and not _matches_target_table(item):
                continue
            bundle_key = _bundle_group_key(item)
            if not bundle_key:
                continue
            bundle_groups.setdefault(bundle_key, []).append(item)
            bundle_group_scores[bundle_key] = max(bundle_group_scores.get(bundle_key, 0.0), score)
        if bundle_groups:
            preferred_bundle_key = max(
                bundle_groups,
                key=lambda key: (len(bundle_groups[key]), bundle_group_scores.get(key, 0.0)),
            )
            required_bundle_rows = min(effective_top_k, 4 if target_method_keys else 3)
            bundle_selected_ids = {
                id(item)
                for item in final
                if _chunk_type(item) == "table_row"
                and _bundle_group_key(item) == preferred_bundle_key
                and _is_support(item)
            }
            bundle_selected_ids.update(
                id(item)
                for item in required_supports
                if _chunk_type(item) == "table_row"
                and _bundle_group_key(item) == preferred_bundle_key
                and _is_support(item)
            )
            for item in bundle_groups[preferred_bundle_key]:
                item_id = id(item)
                if len(bundle_selected_ids) >= required_bundle_rows:
                    break
                if item_id in bundle_selected_ids or item_id in seen_required_ids:
                    continue
                required_supports.append(item)
                seen_required_ids.add(item_id)
                bundle_selected_ids.add(item_id)

    if cost_query and ordered_cost_anchor_candidates and not any(
        _is_numeric_table_cost_anchor_text(item) for item in final
    ):
        cost_anchor_item = ordered_cost_anchor_candidates[0]
        cost_anchor_id = id(cost_anchor_item)
        if cost_anchor_id not in seen_required_ids:
            required_supports.append(cost_anchor_item)
            seen_required_ids.add(cost_anchor_id)

    selected_ids = {id(item) for item in final}

    def _inject_required(item: dict) -> None:
        item_id = id(item)
        if item_id in selected_ids:
            protected_ids.add(item_id)
            return
        if len(final) < effective_top_k:
            final.append(item)
            selected_ids.add(item_id)
            protected_ids.add(item_id)
            return
        replacement_indices = sorted(
            range(len(final)),
            key=lambda idx: (
                id(final[idx]) in protected_ids,
                _row_method_key(final[idx]) in target_method_keys,
                _matches_target_table(final[idx]),
                _is_support(final[idx]),
                _support_score(final[idx]),
                -idx,
            ),
        )
        if not replacement_indices:
            return
        replace_idx = replacement_indices[0]
        selected_ids.discard(id(final[replace_idx]))
        final[replace_idx] = item
        selected_ids.add(item_id)
        protected_ids.add(item_id)

    def _finalize_support_selection(items: List[dict]) -> List[dict]:
        if (
            not cost_query
            and support_target_k > 0
            and len(hard_support_pool) >= support_target_k
        ):
            return hard_support_pool[:support_target_k]
        finalized = _limit_second_best_bundle_rows(
            _dedupe_explicit_comparator_rows(
                _prefer_exact_slice_support(_prioritize_numeric_table_results(items, query))
            )
        )
        if not cost_query and any(_chunk_type(item) == "table_row" and _is_support(item) for item in finalized):
            finalized = [
                item
                for item in finalized
                if not (
                    item.get("table_augmented_scope") == "page_content"
                    and not _is_support(item)
                )
            ]
        return finalized[:effective_top_k]

    for item in required_supports:
        _inject_required(item)

    existing_support = sum(1 for item in final if _is_support(item))
    needed = max(0, min_support - existing_support)
    if needed <= 0:
        return _finalize_support_selection(final)

    additions = [
        item
        for _, _, item in ordered_support_candidates
        if id(item) not in selected_ids
    ][:needed]
    if not additions:
        return _finalize_support_selection(final)

    replacement_indices = [
        idx for idx in range(len(final) - 1, -1, -1)
        if not _is_support(final[idx])
    ]
    for item in additions:
        if replacement_indices:
            final[replacement_indices.pop(0)] = item
        elif len(final) < effective_top_k:
            final.append(item)

    return _finalize_support_selection(final)


def _finalize_without_rerank(
    results: List[dict],
    query: str,
    top_k: int,
    config,
) -> List[dict]:
    ordered = _prioritize_numeric_table_results(results, query)
    ordered = _apply_numeric_table_same_bundle_hard_gate(ordered, query)
    final = _ensure_numeric_table_evidence_slots(ordered, query, top_k)
    final = _apply_numeric_table_same_bundle_hard_gate(final, query)
    if config.enable_focus_mode:
        final = _focus_mode_compress(
            final,
            query,
            window_size=config.focus_mode_window_size,
            max_sentences=config.focus_mode_max_sentences,
            min_chars=config.focus_mode_min_chars,
        )
    return final


def _should_bypass_conditional_rerank_for_numeric_table(
    results: List[dict],
    query: str,
    top_k: int,
) -> bool:
    if not should_apply_numeric_table_specialization():
        return False
    if not results or top_k <= 0:
        return False
    if "numeric_table" not in (_analyze_evidence_need(query) or []):
        return False

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    effective_top_k = _resolve_numeric_table_effective_top_k(query, top_k, hints, results)
    target_methods = _extract_numeric_table_row_method_targets(hints)
    target_datasets = _extract_numeric_table_dataset_mentions(" ".join(hints.get("datasets", [])))
    target_columns = {
        _normalize_numeric_column_name(value)
        for value in hints.get("columns", [])
        if value
    }
    target_backbones = {
        str(value or "").lower()
        for value in hints.get("backbones", [])
        if value
    }
    comparison_query = _is_numeric_table_explicit_comparator_query(query, hints)

    rows_by_group: dict[tuple[int, str, str, str, str], set[str]] = {}
    best_column_coverage = 0
    primary_row_found = False

    for item in results:
        chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
        if chunk_type != "table_row":
            continue

        row_id = (item.get("row_id") or "").strip()
        if not row_id:
            continue

        compact_row_id = _normalize_numeric_table_method_token(row_id)
        matched_methods = {
            method for method in target_methods if method and method == compact_row_id
        }
        if target_methods and not matched_methods:
            continue

        matched_backbone = (item.get("table_focus_backbone") or "").strip().lower()
        if target_backbones and matched_backbone and matched_backbone not in target_backbones:
            continue

        resolved_columns = {
            _normalize_numeric_column_name(value)
            for value in (item.get("table_focus_columns") or [])
            if value
        }
        column_coverage = len(resolved_columns & target_columns) if target_columns else 0
        if target_columns:
            best_column_coverage = max(best_column_coverage, column_coverage)

        if comparison_query:
            page = item.get("page") or 0
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = 0
            table_id = (item.get("table_id") or "").strip().lower()
            dataset_mentions = _extract_numeric_table_dataset_mentions(
                " ".join(
                    str(value or "")
                    for value in (
                        item.get("table_caption", ""),
                        item.get("table_header", ""),
                        item.get("chunk", ""),
                        item.get("raw_chunk_text", ""),
                    )
                )
            )
            dataset_key = ",".join(sorted(dataset_mentions & target_datasets)) if target_datasets else ""
            column_key = ",".join(sorted(value.lower() for value in resolved_columns))
            group_key = (
                page,
                table_id or f"page:{page}",
                dataset_key or "-",
                matched_backbone or "-",
                column_key or "-",
            )
            rows_by_group.setdefault(group_key, set()).update(matched_methods)
        else:
            primary_row_found = True

    if comparison_query:
        required_rows = min(
            effective_top_k,
            4 if len(target_methods) >= 3 else max(len(target_methods), 2),
        )
        return any(len(methods) >= required_rows for methods in rows_by_group.values())

    if not target_columns:
        return primary_row_found
    return primary_row_found and best_column_coverage >= min(len(target_columns), 3)


def _build_retrieval_diagnostics(results: List[dict], query: str) -> dict:
    if not results:
        return {}

    total = len(results)
    raw_chunks = [
        (item.get("raw_chunk_text") or item.get("chunk") or "").strip()
        for item in results
        if (item.get("raw_chunk_text") or item.get("chunk") or "").strip()
    ]
    unique_chunks = len(set(raw_chunks)) if raw_chunks else 0
    duplicate_ratio = 0.0 if not raw_chunks else max(0.0, 1.0 - unique_chunks / max(len(raw_chunks), 1))
    unique_groups = {
        (item.get("group_id") or "").strip()
        for item in results
        if (item.get("group_id") or "").strip()
    }
    unique_sections = {
        _normalize_section_heading(item.get("chunk_heading", ""))
        for item in results
        if _normalize_section_heading(item.get("chunk_heading", ""))
    }
    reference_hits = sum(
        1 for item in results
        if _is_reference_like_text(item.get("raw_chunk_text") or item.get("chunk", ""))
    )
    table_hits = sum(1 for item in results if item.get("chunk_type") == "table")
    formula_hits = sum(1 for item in results if item.get("chunk_type") == "formula")
    numeric_hits = sum(
        1 for item in results
        if any(ch.isdigit() for ch in (item.get("raw_chunk_text") or item.get("chunk", ""))[:400])
    )

    numeric_table_query = "numeric_table" in (_analyze_evidence_need(query) or [])
    source_counts = Counter()
    multi_source_count = 0
    for item in results:
        sources = _infer_retrieval_sources(item)
        if len(sources) > 1:
            multi_source_count += 1
        for source in sources:
            source_counts[source] += 1
    source_total = sum(source_counts.values())
    source_entropy = 0.0
    if source_total:
        for count in source_counts.values():
            p = count / source_total
            source_entropy -= p * math.log2(p)

    diagnostics = {
        "duplicate_chunk_ratio": round(duplicate_ratio, 4),
        "dedup_removed": max(0, len(raw_chunks) - unique_chunks),
        "dedup_ratio": round(duplicate_ratio, 4),
        "source_mix": dict(source_counts),
        "source_mix_entropy": round(source_entropy, 4),
        "multi_source_result_count": multi_source_count,
        "rerank_applied": any(
            item.get("reranked") or item.get("rerank_score") is not None or item.get("combined_score") is not None
            for item in results
        ),
        "unique_group_count": len(unique_groups),
        "unique_group_coverage": round(len(unique_groups) / max(total, 1), 4),
        "unique_section_count": len(unique_sections),
        "section_diversity_ratio": round(len(unique_sections) / max(total, 1), 4),
        "reference_pollution_count": reference_hits,
        "reference_pollution_ratio": round(reference_hits / max(total, 1), 4),
        "table_chunk_hits": table_hits,
        "formula_chunk_hits": formula_hits,
        "numeric_chunk_hits": numeric_hits,
        "numeric_table_query": numeric_table_query,
    }
    if numeric_table_query:
        support_hits = sum(
            1 for item in results
            if item.get("chunk_type") in {"table", "formula", "caption"}
            or any(ch.isdigit() for ch in (item.get("raw_chunk_text") or item.get("chunk", ""))[:400])
        )
        diagnostics["numeric_table_hit_quality"] = round(support_hits / max(total, 1), 4)

    rerank_scores = []
    for item in results:
        score = item.get("rerank_score")
        if score is None:
            score = item.get("combined_score")
        if score is None:
            continue
        try:
            rerank_scores.append(float(score))
        except (TypeError, ValueError):
            continue
    if rerank_scores:
        ordered_scores = sorted(rerank_scores)
        diagnostics["rerank_score_distribution"] = {
            "count": len(rerank_scores),
            "min": round(ordered_scores[0], 4),
            "max": round(ordered_scores[-1], 4),
            "avg": round(sum(ordered_scores) / len(ordered_scores), 4),
            "p25": round(ordered_scores[int((len(ordered_scores) - 1) * 0.25)], 4),
            "p50": round(ordered_scores[int((len(ordered_scores) - 1) * 0.50)], 4),
            "p75": round(ordered_scores[int((len(ordered_scores) - 1) * 0.75)], 4),
        }
    chunk_refs = []
    for idx, item in enumerate(results[:20]):
        ref = {
            "rank": idx + 1,
            "chunk_id": item.get("chunk_id"),
            "parent_id": item.get("parent_id"),
            "doc_id": item.get("doc_id"),
            "page": item.get("page"),
            "group_id": item.get("group_id"),
        }
        if any(value not in (None, "") for value in ref.values()):
            chunk_refs.append(ref)
    if chunk_refs:
        diagnostics["ranked_chunk_refs"] = chunk_refs

    pool_pages: list[int] = []
    pool_ids: list[str] = []
    pool_group_ids: list[str] = []
    pool_chunk_ids: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        page = item.get("page")
        try:
            page_int = int(page) if page not in (None, "") else 0
        except (TypeError, ValueError):
            page_int = 0
        if page_int and page_int not in pool_pages:
            pool_pages.append(page_int)
        for key in ("context_id", "evidence_id", "chunk_id", "child_chunk_id", "parent_id", "doc_id"):
            value = str(item.get(key) or "").strip()
            if value and value not in pool_ids:
                pool_ids.append(value)
            if key in {"chunk_id", "child_chunk_id"} and value and value not in pool_chunk_ids:
                pool_chunk_ids.append(value)
        group_id = str(item.get("group_id") or "").strip()
        if group_id:
            if group_id not in pool_group_ids:
                pool_group_ids.append(group_id)
            if group_id not in pool_ids:
                pool_ids.append(group_id)
        chunk_idx = item.get("chunk_idx")
        if chunk_idx not in (None, ""):
            label = f"chunk:{chunk_idx}"
            if label not in pool_ids:
                pool_ids.append(label)
        if page_int:
            label = f"page:{page_int}"
            if label not in pool_ids:
                pool_ids.append(label)
    if pool_pages or pool_ids or pool_group_ids:
        diagnostics["candidate_pool"] = {
            "pages": pool_pages[:80],
            "ids": pool_ids[:160],
            "group_ids": pool_group_ids[:80],
            "chunk_ids": pool_chunk_ids[:80],
            "selected_pages": pool_pages[:80],
            "selected_ids": pool_ids[:160],
            "selected_group_ids": pool_group_ids[:80],
            "selected_chunk_ids": pool_chunk_ids[:80],
            "selected_count": total,
            "candidate_count": total,
            "by_tool": [
                {
                    "tool": "non_agent_retrieval",
                    "round": 1,
                    "pages": pool_pages[:80],
                    "ids": pool_ids[:160],
                    "group_ids": pool_group_ids[:80],
                    "chunk_ids": pool_chunk_ids[:80],
                    "selected_pages": pool_pages[:80],
                    "selected_ids": pool_ids[:160],
                    "selected_group_ids": pool_group_ids[:80],
                    "selected_chunk_ids": pool_chunk_ids[:80],
                    "selected_count": total,
                    "candidate_count": total,
                }
            ],
        }
    precap_stats = next((item.get("_rerank_precap_stats") for item in results if item.get("_rerank_precap_stats")), None)
    if isinstance(precap_stats, dict):
        diagnostics["rerank_precap"] = precap_stats

    # Focus Mode 压缩统计（仅当有条目被压缩时计算）
    compressed_items = [item for item in results if "focus_compression_ratio" in item]
    if compressed_items:
        avg_ratio = sum(item["focus_compression_ratio"] for item in compressed_items) / len(compressed_items)
        diagnostics["focus_mode_compressed_count"] = len(compressed_items)
        diagnostics["focus_mode_avg_compression_ratio"] = round(avg_ratio, 4)
        diagnostics["focus_mode_total_chars_saved"] = sum(
            item.get("focus_original_chars", 0) - len(item.get("chunk", ""))
            for item in compressed_items
        )

    # section/path 路径多样性
    path_keys = [
        f"{_normalize_section_heading(item.get('chunk_heading',''))}|{item.get('group_id','') or item.get('parent_id','')}"
        for item in results
    ]
    unique_paths = len(set(pk for pk in path_keys if pk.strip("|")))
    singleton_paths = sum(
        1 for pk in set(pk for pk in path_keys if pk.strip("|"))
        if path_keys.count(pk) == 1
    )
    diagnostics["unique_path_count"] = unique_paths
    diagnostics["singleton_path_count"] = singleton_paths
    diagnostics["path_diversity_ratio"] = round(unique_paths / max(total, 1), 4)

    return diagnostics


def _append_retrieval_source(item: dict, source: str) -> dict:
    if not isinstance(item, dict):
        return item
    candidates = [v.strip() for v in str(source or "").replace(",", "+").split("+") if v.strip()]
    sources = []
    existing = item.get("retrieval_sources")
    if isinstance(existing, list):
        sources = [str(v).strip() for v in existing if str(v).strip()]
    elif isinstance(existing, str) and existing.strip():
        sources = [v.strip() for v in existing.replace(",", "+").split("+") if v.strip()]
    for normalized in candidates:
        if normalized not in sources:
            sources.append(normalized)
    if sources:
        item["retrieval_sources"] = sources
        item["retrieval_source"] = "+".join(sources)
    return item


def _mark_retrieval_source(results: List[dict], source: str) -> List[dict]:
    for item in results or []:
        _append_retrieval_source(item, source)
    return results


def _infer_retrieval_sources(item: dict) -> list[str]:
    if not isinstance(item, dict):
        return ["unknown"]
    sources = []
    existing = item.get("retrieval_sources")
    if isinstance(existing, list):
        sources.extend(str(v).strip() for v in existing if str(v).strip())
    elif isinstance(existing, str) and existing.strip():
        sources.extend(v.strip() for v in existing.replace(",", "+").split("+") if v.strip())
    if item.get("bm25") and "bm25" not in sources:
        sources.append("bm25")
    if item.get("table_augmented") and "table_augment" not in sources:
        sources.append("table_augment")
    if item.get("hybrid") and not sources:
        sources.append("hybrid")
    if item.get("semantic_group_id") or item.get("group_id"):
        if "semantic_group" not in sources and item.get("rrf_score"):
            sources.append("semantic_group")
    if not sources:
        sources.append("vector")
    return sources


def _build_context_assembly_diagnostics(
    results: List[dict],
    context_text: str,
    *,
    token_budget: int = 0,
    hierarchical_stats: dict = None,
) -> dict:
    retrieval_diag = _build_retrieval_diagnostics(results, "")
    token_used = _estimate_embedding_tokens(context_text or "")
    diag = {
        "source_mix": retrieval_diag.get("source_mix", {}),
        "source_mix_entropy": retrieval_diag.get("source_mix_entropy", 0.0),
        "dedup_removed": retrieval_diag.get("dedup_removed", 0),
        "dedup_ratio": retrieval_diag.get("dedup_ratio", 0.0),
        "rerank_applied": retrieval_diag.get("rerank_applied", False),
        "token_budget_used": token_used,
        "token_budget_limit": max(0, int(token_budget or 0)),
        "token_budget_ratio": round(token_used / token_budget, 4) if token_budget else 0.0,
    }
    if hierarchical_stats:
        diag["hierarchical"] = hierarchical_stats
    return diag


def _apply_rerank(
    query: str,
    candidates: List[dict],
    reranker_model: Optional[str] = None,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None
) -> List[dict]:
    """对候选结果应用重排序

    注意：此函数为同步调用，在 async 上下文中应通过
    asyncio.to_thread() 调用以避免阻塞事件循环。
    """
    model_name = reranker_model or "BAAI/bge-reranker-base"
    provider = (rerank_provider or "local").lower()
    logger.info(f"[Rerank] 开始重排序: provider={provider}, model={model_name}, 候选数={len(candidates)}")

    try:
        result = rerank_service.rerank(
            query,
            candidates,
            model_name=model_name,
            provider=provider,
            api_key=rerank_api_key,
            endpoint=rerank_endpoint
        )
        logger.info(f"[Rerank] 重排序完成，返回 {len(result)} 条结果")
        return result
    except Exception as e:
        logger.error(f"[Rerank] 重排序失败: {e}", exc_info=True)
        # 回退到相似度排序，不静默吞掉错误
        return sorted(candidates, key=lambda x: x.get("similarity", 0), reverse=True)


# ---- Wide-net retrieval: expand candidate pool when reranking is enabled ----
_WIDE_RETRIEVAL_MULTIPLIER = 5


def _apply_mmr(
    candidates: List[dict],
    top_k: int,
    mmr_lambda: float = 0.5,
) -> List[dict]:
    """Maximal Marginal Relevance: balance relevance with diversity.

    Uses word-set Jaccard overlap as inter-document similarity approximation.
    Greedy: select candidate maximizing mmr_lambda * relevance - (1-mmr_lambda) * max_sim_to_selected.
    """
    if len(candidates) <= top_k:
        return candidates

    selected: List[dict] = []
    selected_word_sets: List[set] = []
    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        if not selected:
            best = remaining[0]
        else:
            best = None
            best_score = float('-inf')
            for cand in remaining:
                rel = cand.get('similarity', 0.0)
                cand_words = set(cand.get('chunk', '')[:600].split())
                if cand_words:
                    sims = [
                        len(cand_words & sw) / max(len(cand_words | sw), 1)
                        for sw in selected_word_sets
                    ]
                    max_sim = max(sims) if sims else 0.0
                else:
                    max_sim = 0.0
                score = mmr_lambda * rel - (1 - mmr_lambda) * max_sim
                if score > best_score:
                    best_score = score
                    best = cand
        selected.append(best)
        selected_word_sets.append(set(best.get('chunk', '')[:600].split()))
        remaining.remove(best)

    return selected


def _is_likely_table(text: str) -> bool:
    """Heuristic to detect table-like text chunks (markdown/ascii/tsv tables)."""
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    if text.count('|') >= 4:
        return True
    if text.count('\t') >= 4:
        return True
    numeric_lines = sum(1 for ln in lines if sum(1 for ch in ln if ch.isdigit()) >= 3)
    if len(lines) >= 3 and numeric_lines >= len(lines) * 0.5:
        return True
    return False


def _augment_with_table_chunks(
    results: List[dict],
    chunks: List[str],
    pages: List[dict],
    page_index: dict,
    query: str = "",
    evidence_need: Optional[List[str]] = None,
    max_augment: int = 3,
    chunk_pages: Optional[List[int]] = None,
    chunk_metadata: Optional[List[dict]] = None,
) -> List[dict]:
    """Augment results with table-like chunks from the same pages as hit chunks.

    Embedding models often fail to retrieve tables via semantic search; this
    secondary pass adds table chunks that share a page with an already-retrieved
    chunk, improving numeric/tabular question coverage.
    """
    if not results or (not chunks and not pages):
        return results

    pages = _annotate_pages_with_provenance(pages)
    hit_pages: set[int] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        hit_pages.update(_extract_page_candidates_from_metadata(item))
    if not hit_pages:
        return results
    numeric_table_query = "numeric_table" in (evidence_need or _analyze_evidence_need(query) or [])
    candidate_pages = set(hit_pages)
    if numeric_table_query:
        for page in list(hit_pages):
            if isinstance(page, int) and page > 0:
                candidate_pages.add(page - 1)
                candidate_pages.add(page + 1)
        numeric_hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
        cost_query = _is_numeric_table_cost_query(query)
        target_methods = _extract_numeric_table_row_method_targets(numeric_hints)
        comparison_query = bool(numeric_hints.get("comparison"))
        preferred_sort_column = _preferred_numeric_table_sort_column(query, numeric_hints)
        allow_sparse_page_fallback = bool(preferred_sort_column) or (
            comparison_query and len(target_methods) <= 1
        )
    else:
        numeric_hints = {}
        cost_query = False
        allow_sparse_page_fallback = False

    pages_by_number: dict[int, dict] = {}
    for page_payload in pages or []:
        if not isinstance(page_payload, dict):
            continue
        try:
            page_num = int(page_payload.get("page") or page_payload.get("page_num") or 0)
        except (TypeError, ValueError):
            page_num = 0
        if page_num > 0 and page_num not in pages_by_number:
            pages_by_number[page_num] = page_payload

    def _has_nonempty_upgrade_value(value) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value)
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) > 0
        return True

    upgraded_results: List[dict] = []
    existing_texts: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            upgraded_results.append(item)
            continue
        original_texts = {
            str(item.get("chunk") or ""),
            str(item.get("raw_chunk_text") or ""),
        }
        upgraded_item = item
        if numeric_table_query and item.get("structured_table_bundle"):
            page = _resolve_primary_page_from_metadata(item)
            if page <= 0:
                try:
                    page = int(item.get("page") or 0)
                except (TypeError, ValueError):
                    page = 0
            chunk_text = str(item.get("chunk") or item.get("raw_chunk_text") or "").strip()
            if chunk_text and page > 0:
                upgraded_chunk, upgraded_metadata = _maybe_upgrade_sparse_structured_bundle(
                    chunk_text,
                    item,
                    pages_by_number.get(page),
                    query=query,
                )
                if upgraded_chunk != chunk_text and isinstance(upgraded_metadata, dict):
                    upgraded_item = dict(item)
                    upgraded_item["chunk"] = upgraded_chunk
                    upgraded_item["raw_chunk_text"] = upgraded_chunk
                    for key, value in upgraded_metadata.items():
                        if _has_nonempty_upgrade_value(value):
                            upgraded_item[key] = value
                    _apply_page_provenance(upgraded_item, upgraded_metadata)
        upgraded_results.append(upgraded_item)
        existing_texts.update(
            text
            for text in {
                *original_texts,
                str(upgraded_item.get("chunk") or ""),
                str(upgraded_item.get("raw_chunk_text") or ""),
            }
            if text
        )
    results = upgraded_results
    augmented_candidates: dict[tuple[str, int], tuple[float, dict, bool]] = {}
    structured_anchor_pages: set[int] = set()

    def _is_cost_support_text(text: str) -> bool:
        if not cost_query:
            return False
        sample = re.sub(r"\s+", " ", (text or "").lower()).strip()
        if not sample:
            return False
        if _has_numeric_table_cost_anchor(sample):
            return True
        return any(
            token in sample
            for token in ("discussion", "limitation", "limitations", "future work", "conclusion", "cost profile")
        )

    def _score_numeric_candidate(chunk_text: str, page: int) -> tuple[float, list[str], bool]:
        boost_score = 1.0
        anchor_hits: list[str] = []
        anchor_groups = {
            "table_labels": 0,
            "datasets": 0,
            "backbones": 0,
            "methods": 0,
            "columns": 0,
        }
        chunk_lower = chunk_text.lower()
        for group_name, weight in (
            ("table_labels", 0.8),
            ("datasets", 1.1),
            ("backbones", 1.0),
            ("methods", 1.2),
            ("columns", 0.9),
        ):
            hits = [value for value in numeric_hints.get(group_name, []) if value.lower() in chunk_lower]
            if not hits:
                continue
            anchor_groups[group_name] = len(hits)
            boost_score += min(len(hits), 3) * weight
            anchor_hits.extend(hits[:3])
        if page in hit_pages:
            boost_score += 0.8
        if len(set(re.findall(r'\btable\s*\d+\b', chunk_text.lower()))) >= 2:
            boost_score -= 0.9

        if anchor_groups["methods"] >= 1 and anchor_groups["backbones"] >= 1:
            boost_score += 0.8
        if anchor_groups["columns"] >= max(1, min(len(numeric_hints.get("columns", [])), 2)):
            boost_score += 0.5

        cost_anchor_match = _is_cost_support_text(chunk_text)
        if cost_anchor_match:
            boost_score += 6.0 if _has_numeric_table_cost_anchor(chunk_text) else 2.0
            anchor_hits.append("cost_anchor")
        elif cost_query and (
            _looks_like_numeric_table_support(chunk_text, "table")
            or bool(re.search(r"\btable\s*\d+\b", chunk_lower))
        ):
            boost_score -= 3.5

        strong_anchor_match = (
            anchor_groups["table_labels"] >= 1
            or (anchor_groups["methods"] >= 1 and anchor_groups["backbones"] >= 1)
            or (
                anchor_groups["methods"] >= 1
                and anchor_groups["columns"] >= 1
                and len({value.lower() for value in anchor_hits}) >= 3
            )
            or cost_anchor_match
        )
        return boost_score, list(dict.fromkeys(anchor_hits)), strong_anchor_match

    def _record_candidate(
        chunk_text: str,
        page: int,
        boost_score: float,
        anchor_hits: list[str],
        scope: str,
        strong_anchor_match: bool,
        metadata: Optional[dict] = None,
    ) -> None:
        key = (chunk_text, int(page or 0))
        structured_bundle = isinstance(metadata, dict) and metadata.get("structured_table_bundle")
        inferred_chunk_type = ""
        if structured_bundle or _is_likely_table(chunk_text):
            inferred_chunk_type = "table"
        elif _is_cost_support_text(chunk_text):
            inferred_chunk_type = "text"
        candidate = {
            "chunk": chunk_text,
            "raw_chunk_text": chunk_text,
            "page": page,
            "similarity": 0.45 + min(boost_score * 0.04, 0.28),
            "similarity_percent": 45.0 + min(boost_score * 4.0, 28.0),
            "score": 0.0,
            "snippet": chunk_text[:200],
            "highlights": [],
            "reranked": False,
            "table_augmented": True,
            "table_augmented_scope": scope,
            "numeric_table_anchor_hits": anchor_hits,
        }
        if inferred_chunk_type:
            candidate["chunk_type"] = inferred_chunk_type
            candidate["block_type"] = inferred_chunk_type
        if metadata:
            _apply_chunk_metadata(candidate, metadata)
        _apply_page_provenance(candidate, metadata)
        previous = augmented_candidates.get(key)
        if previous is None or boost_score > previous[0]:
            augmented_candidates[key] = (boost_score, candidate, strong_anchor_match)

    for idx, chunk_text in enumerate(chunks):
        if chunk_text in existing_texts:
            continue
        metadata = chunk_metadata[idx] if isinstance(chunk_metadata, list) and idx < len(chunk_metadata) else {}
        is_structured_bundle = isinstance(metadata, dict) and metadata.get("structured_table_bundle")
        is_cost_like = _is_cost_support_text(chunk_text)
        is_table_like = is_structured_bundle or _is_likely_table(chunk_text) or (
            numeric_table_query and _looks_like_numeric_table_support(chunk_text)
        )
        if not is_table_like and not is_cost_like:
            continue
        page = _resolve_primary_page_from_metadata(metadata)
        if page <= 0:
            page = chunk_pages[idx] if isinstance(chunk_pages, list) and idx < len(chunk_pages) else 0
        if (not isinstance(page, int) or page <= 0) and pages:
            page = _find_page_for_chunk(chunk_text, pages, page_index=page_index)
        effective_chunk_text = chunk_text
        effective_metadata = metadata if isinstance(metadata, dict) else None
        if numeric_table_query and is_structured_bundle and page > 0:
            effective_chunk_text, effective_metadata = _maybe_upgrade_sparse_structured_bundle(
                chunk_text,
                effective_metadata,
                pages_by_number.get(page),
                query=query,
            )
        if page not in candidate_pages:
            continue
        if numeric_table_query:
            boost_score, anchor_hits, strong_anchor_match = _score_numeric_candidate(effective_chunk_text, page)
            if isinstance(effective_metadata, dict) and effective_metadata.get("structured_table_bundle"):
                boost_score += 0.45
                table_ref = " ".join(
                    str(effective_metadata.get(key) or "")
                    for key in ("table_id", "table_caption", "table_header")
                ).lower()
                if any(
                    label.lower() in table_ref
                    for label in numeric_hints.get("table_labels", [])
                    if label
                ):
                    boost_score += 0.6
                    strong_anchor_match = True
        else:
            boost_score, anchor_hits, strong_anchor_match = 1.0, [], False
        structured_bundle_has_rows = bool(
            isinstance(effective_metadata, dict)
            and effective_metadata.get("structured_table_bundle")
            and effective_metadata.get("evidence_units")
        )
        _record_candidate(
            effective_chunk_text,
            page,
            boost_score,
            anchor_hits,
            scope=(
                "structured_bundle"
                if isinstance(effective_metadata, dict) and effective_metadata.get("structured_table_bundle")
                else ("nearby_page" if numeric_table_query else "same_page")
            ),
            strong_anchor_match=strong_anchor_match,
            metadata=effective_metadata if isinstance(effective_metadata, dict) else None,
        )
        # winner / implicit second-best 题即使已有少量 row evidence，
        # 也要允许同页 page_content 回补，避免只剩单行候选时直接收口。
        if strong_anchor_match and structured_bundle_has_rows and not allow_sparse_page_fallback:
            structured_anchor_pages.add(page)

    if numeric_table_query:
        for page_payload in pages:
            try:
                page = int(page_payload.get("page") or 0)
            except (TypeError, ValueError):
                continue
            page_content = (page_payload.get("content") or page_payload.get("text") or "").strip()
            if not page_content or page_content in existing_texts or page not in candidate_pages:
                continue
            if page in structured_anchor_pages:
                continue
            is_cost_like = _is_cost_support_text(page_content)
            is_table_like = _is_likely_table(page_content) or _looks_like_numeric_table_support(
                page_content,
                "table",
            )
            if not is_table_like and not is_cost_like:
                continue
            boost_score, anchor_hits, strong_anchor_match = _score_numeric_candidate(page_content, page)
            if not strong_anchor_match and page not in hit_pages:
                continue
            boost_score += 0.25
            _record_candidate(
                page_content,
                page,
                boost_score,
                anchor_hits,
                scope="page_content",
                strong_anchor_match=strong_anchor_match,
                metadata=page_payload,
            )

    local_has_strong_anchor = any(item[2] for item in augmented_candidates.values())
    local_has_strong_cost_anchor = cost_query and any(
        _is_numeric_table_cost_anchor_text(candidate)
        for _, candidate, _ in augmented_candidates.values()
    )
    should_scan_global_anchor = (
        numeric_table_query
        and (
            (cost_query and not local_has_strong_cost_anchor)
            or
            bool(numeric_hints.get("table_labels"))
            or (
                not local_has_strong_anchor
                and (
                    numeric_hints.get("table_labels")
                    or numeric_hints.get("backbones")
                    or numeric_hints.get("methods")
                )
            )
        )
    )
    if should_scan_global_anchor:
        for idx, chunk_text in enumerate(chunks):
            if chunk_text in existing_texts:
                continue
            metadata = chunk_metadata[idx] if isinstance(chunk_metadata, list) and idx < len(chunk_metadata) else {}
            is_structured_bundle = isinstance(metadata, dict) and metadata.get("structured_table_bundle")
            is_cost_like = _is_cost_support_text(chunk_text)
            is_table_like = is_structured_bundle or _is_likely_table(chunk_text) or _looks_like_numeric_table_support(chunk_text)
            if not is_table_like and not is_cost_like:
                continue
            page = _resolve_primary_page_from_metadata(metadata)
            if page <= 0:
                page = chunk_pages[idx] if isinstance(chunk_pages, list) and idx < len(chunk_pages) else 0
            if (not isinstance(page, int) or page <= 0) and pages:
                page = _find_page_for_chunk(chunk_text, pages, page_index=page_index)
            effective_chunk_text = chunk_text
            effective_metadata = metadata if isinstance(metadata, dict) else None
            if numeric_table_query and is_structured_bundle and page > 0:
                effective_chunk_text, effective_metadata = _maybe_upgrade_sparse_structured_bundle(
                    chunk_text,
                    effective_metadata,
                    pages_by_number.get(page),
                    query=query,
                )
            if page in candidate_pages:
                continue
            boost_score, anchor_hits, strong_anchor_match = _score_numeric_candidate(effective_chunk_text, page)
            if isinstance(effective_metadata, dict) and effective_metadata.get("structured_table_bundle"):
                boost_score += 0.45
                table_ref = " ".join(
                    str(effective_metadata.get(key) or "")
                    for key in ("table_id", "table_caption", "table_header")
                ).lower()
                if any(
                    label.lower() in table_ref
                    for label in numeric_hints.get("table_labels", [])
                    if label
                ):
                    boost_score += 0.6
                    strong_anchor_match = True
            if not strong_anchor_match:
                continue
            boost_score += 0.35
            _record_candidate(
                effective_chunk_text,
                page,
                boost_score,
                anchor_hits,
                scope=(
                    "structured_bundle_global_anchor"
                    if isinstance(effective_metadata, dict) and effective_metadata.get("structured_table_bundle")
                    else "global_anchor"
                ),
                strong_anchor_match=strong_anchor_match,
                metadata=effective_metadata if isinstance(effective_metadata, dict) else None,
            )
            if isinstance(effective_metadata, dict) and effective_metadata.get("structured_table_bundle"):
                structured_anchor_pages.add(page)
        for page_payload in pages:
            try:
                page = int(page_payload.get("page") or 0)
            except (TypeError, ValueError):
                continue
            page_content = (page_payload.get("content") or page_payload.get("text") or "").strip()
            if not page_content or page_content in existing_texts or page in candidate_pages:
                continue
            if page in structured_anchor_pages:
                continue
            is_cost_like = _is_cost_support_text(page_content)
            is_table_like = _is_likely_table(page_content) or _looks_like_numeric_table_support(
                page_content,
                "table",
            )
            if not is_table_like and not is_cost_like:
                continue
            boost_score, anchor_hits, strong_anchor_match = _score_numeric_candidate(page_content, page)
            if not strong_anchor_match:
                continue
            boost_score += 0.55
            _record_candidate(
                page_content,
                page,
                boost_score,
                anchor_hits,
                scope="page_global_anchor",
                strong_anchor_match=strong_anchor_match,
            )

    ranked_candidates = sorted(
        augmented_candidates.values(),
        key=lambda item: item[0],
        reverse=True,
    )
    selected_candidates = list(ranked_candidates[:max_augment])
    if cost_query and max_augment > 0:
        def _candidate_is_cost_anchor(candidate_item: dict) -> bool:
            return _is_numeric_table_cost_anchor_text(candidate_item)

        best_cost_candidate = next(
            (candidate for candidate in ranked_candidates if _candidate_is_cost_anchor(candidate[1])),
            None,
        )
        if best_cost_candidate and not any(best_cost_candidate[1] is existing[1] for existing in selected_candidates):
            replacement_idx = next(
                (
                    idx
                    for idx in range(len(selected_candidates) - 1, -1, -1)
                    if not _candidate_is_cost_anchor(selected_candidates[idx][1])
                ),
                len(selected_candidates) - 1,
            )
            if selected_candidates:
                selected_candidates[replacement_idx] = best_cost_candidate
            else:
                selected_candidates.append(best_cost_candidate)
            selected_candidates.sort(key=lambda item: item[0], reverse=True)
    augmented = [item for _, item, _ in selected_candidates]

    if augmented:
        if numeric_table_query and any(item.get("table_augmented_scope") == "global_anchor" for item in augmented):
            scope = "同页/相邻页 + 全局锚点"
        else:
            scope = "同页/相邻页" if numeric_table_query else "同页"
        logger.info(f"[TableAugment] 补充 {len(augmented)} 个{scope}表格 chunk")
    return results + augmented


def _ensure_structured_table_row_shard_results(
    results: List[dict],
    chunks: List[str],
    chunk_pages: List[int],
    chunk_types: List[str],
    chunk_metadata: List[dict],
    query: str,
    top_k: int,
    max_shards: int = 2,
) -> List[dict]:
    """Promote precise structured-table row shards into final numeric contexts.

    Large MinerU table bundles are kept as whole-table chunks for broad table
    semantics, but their embeddings can be dominated by the caption/header. This
    pass uses the already-persisted row shards as a same-table evidence sidecar
    so table-specific questions are not answered from a truncated/generic table
    context only.
    """
    if not should_apply_numeric_table_specialization():
        return results[:top_k] if top_k > 0 else []
    if not results or top_k <= 0 or max_shards <= 0:
        return results[:top_k] if top_k > 0 else []
    if "numeric_table" not in (_analyze_evidence_need(query) or []):
        return results[:top_k]
    if not chunks or not isinstance(chunk_metadata, list):
        return results[:top_k]

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    target_tables = {
        value.lower()
        for value in hints.get("table_labels", [])
        if value
    } | {
        table_id.lower()
        for _, _, table_id in _extract_table_mentions(query)
        if table_id
    }
    if not target_tables:
        return results[:top_k]

    def _norm(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    selected_table_ids: set[str] = set()
    selected_bundle_ids: set[str] = set()
    existing_chunk_ids: set[int] = set()
    existing_texts: set[str] = set()
    selected_has_row_shard = False
    for item in results:
        if not isinstance(item, dict):
            continue
        try:
            existing_chunk_ids.add(int(item.get("chunk_id")))
        except (TypeError, ValueError):
            pass
        text = str(item.get("chunk") or item.get("raw_chunk_text") or "")
        if text:
            existing_texts.add(text)
        if item.get("table_row_shard") or (item.get("chunk_type") or "").lower() == "table_row":
            selected_has_row_shard = True
        for key in ("table_id", "table_caption", "numeric_table_exact_context_caption"):
            table_id = _extract_table_id(str(item.get(key) or ""))
            if table_id:
                selected_table_ids.add(table_id.lower())
        for key in ("table_bundle_id", "parent_table_bundle_id"):
            value = str(item.get(key) or "").strip()
            if value:
                selected_bundle_ids.add(value)

    # If the final context already contains a row shard, keep it. The helper is
    # intentionally conservative and only fills the common "whole table only"
    # gap observed in MinerU numeric-table evaluations.
    if selected_has_row_shard:
        return results[:top_k]

    query_lower = _norm(query)
    group_values: dict[str, list[str]] = {
        "datasets": [_norm(v) for v in hints.get("datasets", []) if _norm(v)],
        "backbones": [_norm(v) for v in hints.get("backbones", []) if _norm(v)],
        "methods": [_norm(v) for v in hints.get("methods", []) if _norm(v)],
        "columns": [_norm(v) for v in hints.get("columns", []) if _norm(v)],
    }
    # Single-letter/very short method hints such as "T", "L", "I", "O" are too
    # noisy unless the table label already matched, so they get a tiny weight.
    short_method_values = {value for value in group_values["methods"] if len(value) <= 2}
    short_column_values = {value for value in group_values["columns"] if len(value) <= 2}

    def _has_target_table(ref_text: str) -> bool:
        ref_lower = _norm(ref_text)
        return any(table and table in ref_lower for table in target_tables)

    def _score_row_shard(text: str, metadata: dict) -> tuple[float, list[str]]:
        ref_text = " ".join(
            str(metadata.get(key) or "")
            for key in ("table_id", "table_caption", "table_header", "parent_table_bundle_id")
        )
        combined = _norm(f"{ref_text}\n{text}")
        table_hit = _has_target_table(combined)
        parent_bundle = str(metadata.get("parent_table_bundle_id") or metadata.get("table_bundle_id") or "").strip()
        table_id = _extract_table_id(str(metadata.get("table_id") or metadata.get("table_caption") or ""))
        selected_bundle_hit = bool(parent_bundle and parent_bundle in selected_bundle_ids)
        selected_table_hit = bool(table_id and table_id.lower() in selected_table_ids)
        if not (table_hit or selected_bundle_hit or selected_table_hit):
            return 0.0, []

        score = 5.0 if table_hit else 2.5
        if selected_bundle_hit:
            score += 1.2
        if selected_table_hit:
            score += 1.0
        hits: list[str] = []
        for group_name, weight in (
            ("datasets", 1.0),
            ("backbones", 1.0),
            ("methods", 0.9),
            ("columns", 0.8),
        ):
            for value in group_values.get(group_name, []):
                if not value:
                    continue
                if value in combined:
                    adjusted = weight
                    if value in short_method_values or value in short_column_values:
                        adjusted = min(adjusted, 0.2)
                    score += adjusted
                    hits.append(value)
        if metadata.get("row_start") is not None and metadata.get("row_end") is not None:
            score += 0.25
        if re.search(r"\d", text or ""):
            score += 0.25
        if any(token and token in combined for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", query_lower)[:12]):
            score += 0.2
        return score, list(dict.fromkeys(hits))

    scored: list[tuple[float, int, dict]] = []
    for idx, chunk_text in enumerate(chunks):
        if idx in existing_chunk_ids or chunk_text in existing_texts:
            continue
        metadata = chunk_metadata[idx] if idx < len(chunk_metadata) and isinstance(chunk_metadata[idx], dict) else {}
        chunk_type = (chunk_types[idx] if idx < len(chunk_types) else metadata.get("chunk_type") or "").strip().lower()
        if not (
            metadata.get("table_row_shard")
            or metadata.get("table_row_slice_kind") == "shard"
            or chunk_type == "table_row"
            or "[structured table row shard]" in str(chunk_text).lower()
        ):
            continue
        score, hits = _score_row_shard(str(chunk_text), metadata)
        if score < 5.2:
            continue
        page = _resolve_primary_page_from_metadata(metadata)
        if page <= 0 and idx < len(chunk_pages):
            page = chunk_pages[idx]
        candidate = {
            "chunk": chunk_text,
            "raw_chunk_text": chunk_text,
            "page": page,
            "score": 0.0,
            "similarity": 0.78 + min(score * 0.01, 0.12),
            "similarity_percent": round((0.78 + min(score * 0.01, 0.12)) * 100, 2),
            "snippet": str(chunk_text)[:200],
            "highlights": [],
            "reranked": False,
            "chunk_id": idx,
            "chunk_type": "table_row",
            "block_type": "table_row",
            "table_augmented": True,
            "table_augmented_scope": "structured_row_shard",
            "numeric_table_anchor_hits": hits,
            "numeric_table_priority": score,
        }
        _apply_chunk_metadata(candidate, metadata)
        _apply_page_provenance(candidate, metadata)
        scored.append((score, idx, candidate))

    if not scored:
        return results[:top_k]

    scored.sort(key=lambda row: (-row[0], row[1]))
    final = list(results[:top_k])
    max_shards = _resolve_structured_table_row_shard_limit(query, max_shards)

    def _result_chunk_type(item: dict) -> str:
        return (item.get("chunk_type") or item.get("block_type") or "").strip().lower()

    def _is_row_candidate(item: dict) -> bool:
        return bool(item.get("table_row_shard") or _result_chunk_type(item) == "table_row")

    def _is_table_support(item: dict) -> bool:
        return _result_chunk_type(item) in {"table", "caption", "table_cell"}

    def _same_table_support(item: dict, candidate: dict) -> bool:
        item_table = _extract_table_id(
            str(
                item.get("table_id")
                or item.get("table_caption")
                or item.get("numeric_table_exact_context_caption")
                or ""
            )
        ).lower()
        candidate_table = _extract_table_id(
            str(
                candidate.get("table_id")
                or candidate.get("table_caption")
                or candidate.get("numeric_table_exact_context_caption")
                or ""
            )
        ).lower()
        if item_table and candidate_table and item_table == candidate_table:
            return True
        item_bundle = str(item.get("table_bundle_id") or item.get("parent_table_bundle_id") or "").strip()
        candidate_bundle = str(candidate.get("table_bundle_id") or candidate.get("parent_table_bundle_id") or "").strip()
        return bool(item_bundle and candidate_bundle and item_bundle == candidate_bundle)

    def _replacement_index(candidate: dict) -> Optional[int]:
        # Prefer replacing redundant table supports from the same table, but keep
        # at least one caption/bundle context around row shards for faithfulness.
        table_support_indices = [
            idx for idx, item in enumerate(final)
            if isinstance(item, dict) and _is_table_support(item) and _same_table_support(item, candidate)
        ]
        if len(table_support_indices) > 1:
            return table_support_indices[-1]

        non_table_indices = [
            idx for idx, item in enumerate(final)
            if isinstance(item, dict) and not (_is_row_candidate(item) or _is_table_support(item))
        ]
        if len(non_table_indices) > 1:
            return non_table_indices[-1]

        row_indices = [
            idx for idx, item in enumerate(final)
            if isinstance(item, dict) and _is_row_candidate(item)
        ]
        if len(row_indices) > max(1, max_shards):
            return row_indices[-1]
        return None

    for _score, _idx, candidate in scored[:max_shards]:
        key = str(candidate.get("chunk") or "")
        if not key or any(str(item.get("chunk") or "") == key for item in final if isinstance(item, dict)):
            continue
        if len(final) < top_k:
            final.append(candidate)
            continue
        replace_idx = _replacement_index(candidate)
        if replace_idx is None:
            continue
        final[replace_idx] = candidate
    return final[:top_k]


def _resolve_structured_table_row_shard_limit(query: str, default_limit: int = 2) -> int:
    """Use more row shards for explicit numeric comparisons, but keep a cap."""
    if default_limit <= 0:
        return default_limit
    try:
        hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    except Exception:
        hints = {}
    methods = {
        _normalize_numeric_table_method_token(value)
        for value in hints.get("methods", [])
        if _normalize_numeric_table_method_token(value)
    }
    backbones = {str(value or "").strip().lower() for value in hints.get("backbones", []) if str(value or "").strip()}
    columns = {_normalize_numeric_column_name(value) for value in hints.get("columns", []) if value}
    comparison_query = bool(hints.get("comparison")) or _is_numeric_table_row_band_query(query, hints)
    if not comparison_query and len(methods) <= 1:
        return default_limit
    desired = max(default_limit, len(methods), min(len(columns), 3), min(len(backbones), 2))
    if comparison_query:
        desired = max(desired, 3)
    return min(max(desired, 1), 4)


def structure_aware_split(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[str]:
    """结构感知分块

    优先级：
    1. 识别受保护区域（表格、LaTeX 公式块），标记为不可切分
    2. 按段落边界（双换行）切分文本
    3. 合并连续段落到 chunk_size 以内
    4. 受保护区域保持完整，超过 chunk_size 时单独成块
    5. 检测失败时回退到 RecursiveCharacterTextSplitter

    Args:
        text: 待分块的文本
        chunk_size: 最大分块字符数（默认 1200）
        chunk_overlap: 分块重叠字符数（默认 200）

    Returns:
        分块后的文本列表
    """
    result = structure_aware_split_with_context(text, chunk_size, chunk_overlap)
    return [chunk_text for chunk_text, _ in result]


def structure_aware_split_with_context(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[tuple[str, str]]:
    """结构感知分块（带章节上下文）

    与 structure_aware_split 相同的分块逻辑，但额外返回每个 chunk
    所属的章节标题上下文，用于 Contextual Chunking。

    Args:
        text: 待分块的文本
        chunk_size: 最大分块字符数（默认 1200）
        chunk_overlap: 分块重叠字符数（默认 200）

    Returns:
        (chunk_text, heading_context) 元组列表
    """
    if not text or not text.strip():
        return []

    try:
        # 步骤 1：识别受保护区域（表格和公式块）
        protected_regions = _find_protected_regions(text)

        # 步骤 2：按段落边界切分，同时保护受保护区域
        segments = _split_by_paragraphs_with_protection(text, protected_regions)

        if not segments:
            raise ValueError("段落切分结果为空")

        # 步骤 3：合并段落为分块，尊重 chunk_size 限制
        chunks_with_ctx = _merge_segments_into_chunks(segments, chunk_size, chunk_overlap)

        if not chunks_with_ctx:
            raise ValueError("合并分块结果为空")

        return chunks_with_ctx

    except Exception as e:
        # 检测失败时回退到 RecursiveCharacterTextSplitter
        logger.warning(f"结构感知分块失败，回退到 RecursiveCharacterTextSplitter: {e}")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return [(c, "") for c in text_splitter.split_text(text)]


def _find_protected_regions(text: str) -> list[tuple[int, int]]:
    """识别文本中的受保护区域（表格和公式块）

    受保护区域类型：
    - 表格：连续的以 | 开头且包含 | 分隔符的行
    - 显示公式：$$...$$ 或 \\[...\\] 包裹的区域

    Args:
        text: 原始文本

    Returns:
        受保护区域的 (start, end) 位置列表，按 start 排序
    """
    regions = []

    # 检测表格区域：连续的 markdown 表格行（以 | 开头或包含 | 分隔符）
    table_pattern = re.compile(
        r'(?:^[ \t]*\|.+\|[ \t]*$\n?){2,}',
        re.MULTILINE
    )
    for m in table_pattern.finditer(text):
        regions.append((m.start(), m.end()))

    # 检测显示公式：$$...$$ 块（跨行）
    display_math_pattern = re.compile(r'\$\$[\s\S]+?\$\$')
    for m in display_math_pattern.finditer(text):
        regions.append((m.start(), m.end()))

    # 检测显示公式：\[...\] 块（跨行）
    bracket_math_pattern = re.compile(r'\\\[[\s\S]+?\\\]')
    for m in bracket_math_pattern.finditer(text):
        regions.append((m.start(), m.end()))

    # 按起始位置排序并合并重叠区域
    regions.sort(key=lambda r: r[0])
    merged = []
    for start, end in regions:
        if merged and start <= merged[-1][1]:
            # 与上一个区域重叠，合并
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def _split_by_paragraphs_with_protection(
    text: str,
    protected_regions: list[tuple[int, int]],
) -> list[dict]:
    """按段落和标题边界切分文本，同时保护受保护区域不被切割

    将文本分为两类段：
    - 普通段落：可以被进一步合并或切分
    - 受保护段：表格或公式块，必须保持完整

    切分策略（优先级从高到低）：
    1. 受保护区域边界（表格、公式块）
    2. Markdown 标题边界（# ## ### 等）
    3. 编号标题边界（1. 1.1 2.3.4 等）
    4. 段落边界（双换行）

    Args:
        text: 原始文本
        protected_regions: 受保护区域的 (start, end) 列表

    Returns:
        段列表，每个元素为 {"text": str, "protected": bool, "heading": str|None}
    """
    if not protected_regions:
        # 没有受保护区域，按段落+标题边界切分
        return _split_normal_text_with_headings(text)

    segments = []
    pos = 0

    for region_start, region_end in protected_regions:
        # 处理受保护区域之前的普通文本
        if pos < region_start:
            normal_text = text[pos:region_start]
            segments.extend(_split_normal_text_with_headings(normal_text))

        # 添加受保护区域
        protected_text = text[region_start:region_end].strip()
        if protected_text:
            segments.append({"text": protected_text, "protected": True, "heading": None})

        pos = region_end

    # 处理最后一个受保护区域之后的普通文本
    if pos < len(text):
        remaining_text = text[pos:]
        segments.extend(_split_normal_text_with_headings(remaining_text))

    return segments


# 标题检测正则（用于结构感知分段）
_RE_HEADING_LINE = re.compile(
    r'^(?:'
    r'\s*#{1,6}\s+\S'       # Markdown 标题：# ## ###
    r'|\s*\d+(\.\d+)*\.?\s+\S'  # 编号标题：1. 1.1 2.3.4
    r')',
    re.MULTILINE,
)


def _split_normal_text_with_headings(text: str) -> list[dict]:
    """按标题边界和段落边界切分普通文本

    优先在标题行前切分，其次在双换行处切分。

    Args:
        text: 普通文本（不含受保护区域）

    Returns:
        段列表，每个元素为 {"text": str, "protected": False, "heading": str|None}
    """
    if not text or not text.strip():
        return []

    # 按换行拆分为行，然后识别标题行并在标题前切分
    lines = text.split('\n')
    segments = []
    current_lines = []
    current_heading = None

    for line in lines:
        stripped = line.strip()

        # 检测是否为标题行
        is_heading = False
        if stripped:
            if re.match(r'^\s*#{1,6}\s+\S', line):
                is_heading = True
            elif re.match(r'^\s*\d+(\.\d+)*\.?\s+\S', stripped):
                is_heading = True
            else:
                # 全大写行（英文标题）
                alpha_chars = re.sub(r'[^a-zA-Z]', '', stripped)
                if len(alpha_chars) >= 2 and alpha_chars.isupper() and len(stripped) < 100:
                    is_heading = True

        if is_heading and current_lines:
            # 遇到标题：先保存之前积累的段落
            seg_text = '\n'.join(current_lines).strip()
            if seg_text:
                segments.append({
                    "text": seg_text,
                    "protected": False,
                    "heading": current_heading,
                })
            current_lines = [line]
            current_heading = stripped
        elif not stripped and current_lines:
            # 空行：检查是否是段落分隔（连续空行）
            # 保留单个空行在当前段落中
            current_lines.append(line)
        else:
            current_lines.append(line)
            if is_heading and not current_heading:
                current_heading = stripped

    # 保存最后一个段落
    if current_lines:
        seg_text = '\n'.join(current_lines).strip()
        if seg_text:
            segments.append({
                "text": seg_text,
                "protected": False,
                "heading": current_heading,
            })

    # 如果没有找到任何标题，回退到按双换行切分
    if len(segments) <= 1 and text.strip():
        paragraphs = re.split(r'\n\n+', text)
        return [{"text": p.strip(), "protected": False, "heading": None}
                for p in paragraphs if p.strip()]

    return segments


def _merge_segments_into_chunks(
    segments: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, str]]:
    """将段合并为分块，尊重 chunk_size 限制和受保护区域完整性

    合并策略：
    - 连续的普通段落合并到 chunk_size 以内
    - 受保护段独立或与相邻普通段落合并（不超过 chunk_size）
    - 受保护段本身超过 chunk_size 时单独成块
    - 通过重叠实现分块间的上下文连续性
    - 追踪每个 chunk 所属的章节标题上下文

    Args:
        segments: 段列表（来自 _split_by_paragraphs_with_protection）
        chunk_size: 最大分块字符数
        chunk_overlap: 分块重叠字符数

    Returns:
        (chunk_text, heading_context) 元组列表
        - chunk_text: 分块文本
        - heading_context: 该 chunk 所属的章节标题（可为空字符串）
    """
    chunks = []  # [(text, heading)]
    current_parts = []  # 当前分块中的文本片段
    current_len = 0
    active_heading = ""  # 当前活跃的章节标题

    def _commit_chunk():
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append(("\n\n".join(current_parts), active_heading))
            current_parts = []
            current_len = 0

    for seg in segments:
        seg_text = seg["text"]
        seg_len = len(seg_text)
        is_protected = seg["protected"]

        # 更新活跃标题（segments 带有 heading 字段）
        seg_heading = seg.get("heading")
        if seg_heading:
            active_heading = seg_heading

        if is_protected:
            if seg_len > chunk_size:
                _commit_chunk()
                chunks.append((seg_text, active_heading))
            elif current_len + seg_len + 2 > chunk_size:
                _commit_chunk()
                current_parts.append(seg_text)
                current_len = seg_len
            else:
                current_parts.append(seg_text)
                current_len += seg_len + (2 if current_parts else 0)
        else:
            # 普通段落
            if seg_len > chunk_size:
                # 超大段落：先提交当前缓冲，再拆分超大段落
                _commit_chunk()
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                _oversized_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
                )
                for sub in _oversized_splitter.split_text(seg_text):
                    chunks.append((sub, active_heading))
                continue

            if current_len + seg_len + 2 > chunk_size and current_parts:
                _commit_chunk()

                # 实现重叠：从当前分块末尾取 overlap 部分作为新分块的开头
                overlap_parts = _get_overlap_parts(current_parts, chunk_overlap)
                current_parts = overlap_parts
                current_len = sum(len(p) for p in current_parts) + max(0, (len(current_parts) - 1) * 2)

            current_parts.append(seg_text)
            current_len += seg_len + (2 if len(current_parts) > 1 else 0)

    # 提交最后一个分块
    _commit_chunk()

    return [(c, h) for c, h in chunks if c.strip()]


def _get_overlap_parts(parts: list[str], overlap_size: int) -> list[str]:
    """从分块末尾提取重叠部分

    从 parts 列表的末尾向前取，直到累计字符数达到 overlap_size。

    Args:
        parts: 当前分块的文本片段列表
        overlap_size: 目标重叠字符数

    Returns:
        用于重叠的文本片段列表
    """
    if not parts or overlap_size <= 0:
        return []

    overlap_parts = []
    total = 0
    for p in reversed(parts):
        if total + len(p) > overlap_size and overlap_parts:
            break
        overlap_parts.insert(0, p)
        total += len(p)

    return overlap_parts


def _sanitize_structured_table_bundle(bundle: dict) -> dict:
    """清洗结构化表格 bundle，确保可安全写入索引元数据。"""
    if not isinstance(bundle, dict):
        return {}
    cleaned = {}
    for key in (
        "bundle_id",
        "evidence_unit_id",
        "table_id",
        "table_caption",
        "table_header",
        "table_body_markdown",
        "table_markdown",
        "bundle_text",
        "html_table",
        "table_footnote",
        "page_start",
        "page_end",
        "page",
        "pages",
        "page_index",
        "page_uid",
        "page_uids",
        "bounding_box",
        "bbox",
        "table_bbox",
        "bounding_boxes",
        # Parser-owned bboxes above are NOT in canonical page points (MinerU emits
        # ``normalized_0_1000``, ODL emits ``pdf_bottom_left_points``). The additive
        # geometry below carries the converted box plus the label that tells every
        # downstream consumer how to read the raw one; dropping it here is what let
        # citation anchors publish parser coordinates as ``pdf_top_left_points``.
        "raw_bbox",
        "bbox_coordinate_space",
        "page_size",
        "visual_bbox",
        "visual_coordinate_space",
        "source_ids",
        "source_id",
        "table_bundle_id",
        "previous_table_ids",
        "next_table_ids",
        "source",
        "selected_source",
        "selection_reason",
        "table_selector_score",
        "table_selector_version",
        "table_selector_candidates",
        "evidence_units",
        "table_instance_id",
        "table_source_hash",
    ):
        value = bundle.get(key)
        if value in (None, "", [], {}):
            continue
        if key == "evidence_units":
            cleaned[key] = _sanitize_nested_value(value)
            continue
        cleaned[key] = value
    cleaned.update(build_table_visual_metadata(cleaned))
    return cleaned


def _sanitize_control_text(text: str) -> str:
    if not text:
        return ""
    return "".join(ch for ch in text if ord(ch) >= 32 or ch in "\t\n\r")


def _sanitize_nested_value(value):
    if isinstance(value, str):
        return _sanitize_control_text(value)
    if isinstance(value, dict):
        return {key: _sanitize_nested_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_nested_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_nested_value(item) for item in value)
    return value


def _build_structured_table_bundle_chunk(bundle: dict) -> str:
    """把结构化 table bundle 转成适合向量索引的 typed chunk 文本。"""
    sanitized = _sanitize_structured_table_bundle(bundle)
    if not sanitized:
        return ""

    caption = (sanitized.get("table_caption") or sanitized.get("table_id") or "").strip()
    table_id = (sanitized.get("table_id") or "").strip()
    header = (sanitized.get("table_header") or "").strip()
    body = (sanitized.get("table_body_markdown") or "").strip()
    footnote = (sanitized.get("table_footnote") or "").strip()
    page_start = sanitized.get("page_start")
    page_end = sanitized.get("page_end")

    hint_parts = []
    if table_id:
        hint_parts.append(f"table_id={table_id}")
    if isinstance(page_start, int) and page_start > 0:
        if isinstance(page_end, int) and page_end > page_start:
            hint_parts.append(f"pages={page_start}-{page_end}")
        else:
            hint_parts.append(f"page={page_start}")
    bundle_id = (sanitized.get("bundle_id") or "").strip()
    if bundle_id:
        hint_parts.append(f"bundle_id={bundle_id}")

    lines = ["[Structured Table Bundle]"]
    if caption:
        lines.extend(["", caption])
    if hint_parts:
        lines.extend(["", "[Hints]", "; ".join(hint_parts)])
    if header:
        lines.extend(["", "[Header]", header])
    if body:
        lines.extend(["", "[Body]", body])
    if footnote:
        lines.extend(["", "[Footnote]", footnote])
    return "\n".join(lines).strip()


_STRUCTURED_TABLE_ROW_SHARD_SIZE = 10


def _split_structured_table_cells(text: str) -> List[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if "|" in value:
        return [
            re.sub(r"\s+", " ", part.replace("\\|", "|")).strip()
            for part in value.strip("|").split("|")
        ]
    if ";" in value and ":" in value:
        return [re.sub(r"\s+", " ", part).strip() for part in value.split(";")]
    return [re.sub(r"\s+", " ", value).strip()]


def _structured_table_header_cells(unit: dict) -> List[str]:
    header = str(unit.get("table_header") or unit.get("numeric_table_exact_context_header") or "").strip()
    cells = [cell for cell in _split_structured_table_cells(header) if cell]
    return cells


def _structured_table_cell_values(unit: dict) -> List[str]:
    cells = unit.get("cell_evidence_units")
    if isinstance(cells, list):
        values = [
            re.sub(r"\s+", " ", str(cell.get("content") or cell.get("text") or cell.get("cell_text") or "")).strip()
            for cell in cells
            if isinstance(cell, dict)
        ]
        values = [value for value in values if value]
        if values:
            return values
    for key in ("raw_row_text", "row_text", "content", "row_numbers", "table_row_boundary_text"):
        value = re.sub(r"\s+", " ", str(unit.get(key) or "")).strip()
        if value:
            return [part for part in _split_structured_table_cells(value) if part]
    return []


def _looks_header_bound_row_text(text: str) -> bool:
    sample = str(text or "")
    if ":" not in sample:
        return False
    pairs = [part for part in re.split(r";|\n", sample) if ":" in part]
    return len(pairs) >= 2 or bool(re.search(r"\b(?:AP|AP50|Acc|All|Many|Med|Few|Backbone|Method|Pre-?train|Dataset)\b\s*:", sample, re.I))


def _structured_table_header_bound_row_text(unit: dict) -> str:
    headers = _structured_table_header_cells(unit)
    values = _structured_table_cell_values(unit)
    if not values:
        return ""
    if not headers:
        return ""
    parts: List[str] = []
    for idx, value in enumerate(values):
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if not value:
            continue
        header = re.sub(r"\s+", " ", str(headers[idx] if idx < len(headers) else "")).strip()
        if not header:
            header = f"Column {idx + 1}"
        if header == value:
            parts.append(value)
        else:
            parts.append(f"{header}: {value}")
    return "; ".join(parts).strip()


def _structured_table_row_text(unit: dict) -> str:
    if not isinstance(unit, dict):
        return ""
    for key in ("row_text", "content", "row_numbers", "table_row_boundary_text"):
        value = unit.get(key)
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text:
            if _looks_header_bound_row_text(text):
                return text
            bound = _structured_table_header_bound_row_text(unit)
            return bound or text
    cells = unit.get("cell_evidence_units")
    if isinstance(cells, list):
        parts = [
            re.sub(r"\s+", " ", str(cell.get("content") or cell.get("text") or cell.get("cell_text") or "")).strip()
            for cell in cells
            if isinstance(cell, dict)
        ]
        parts = [part for part in parts if part]
        if parts:
            bound = _structured_table_header_bound_row_text(unit)
            return bound or " | ".join(parts)
    return ""


def _extract_structured_table_row_shard_units(bundle: dict) -> List[dict]:
    evidence_units = bundle.get("evidence_units")
    if not isinstance(evidence_units, list):
        return []
    rows: List[dict] = []
    for index, unit in enumerate(evidence_units):
        if not isinstance(unit, dict):
            continue
        unit_type = str(unit.get("evidence_unit_type") or "").strip().lower()
        if unit_type != "table_row":
            continue
        if unit.get("is_header_row"):
            continue
        row = dict(unit)
        row.setdefault("table_header", bundle.get("table_header", ""))
        row.setdefault("table_caption", bundle.get("table_caption", ""))
        row.setdefault("table_id", bundle.get("table_id", ""))
        row.setdefault("table_bundle_id", bundle.get("bundle_id", "") or bundle.get("table_bundle_id", ""))
        row.setdefault("table_instance_id", bundle.get("table_instance_id", ""))
        row.setdefault("table_source_hash", bundle.get("table_source_hash", ""))
        row_text = _structured_table_row_text(row)
        if not row_text:
            continue
        row.setdefault("row_index", index)
        row["row_text"] = row_text
        row["content"] = row_text
        rows.append(row)
    return rows


def _build_structured_table_row_shard_chunk(
    bundle: dict,
    rows: List[dict],
    shard_index: int,
    row_start: int,
    row_end: int,
) -> str:
    caption = (bundle.get("table_caption") or bundle.get("table_id") or "").strip()
    table_id = (bundle.get("table_id") or "").strip()
    header = (bundle.get("table_header") or "").strip()
    bundle_id = (bundle.get("bundle_id") or "").strip()
    page_start = bundle.get("page_start")
    page_end = bundle.get("page_end")

    hint_parts = []
    if table_id:
        hint_parts.append(f"table_id={table_id}")
    if bundle_id:
        hint_parts.append(f"bundle_id={bundle_id}")
    hint_parts.append(f"shard={shard_index + 1}")
    hint_parts.append(f"rows={row_start}-{row_end}")
    if isinstance(page_start, int) and page_start > 0:
        if isinstance(page_end, int) and page_end > page_start:
            hint_parts.append(f"pages={page_start}-{page_end}")
        else:
            hint_parts.append(f"page={page_start}")

    row_lines = [_structured_table_row_text(row) for row in rows]
    row_lines = [line for line in row_lines if line]

    lines = ["[Structured Table Row Shard]"]
    if caption:
        lines.extend(["", caption])
    if hint_parts:
        lines.extend(["", "[Hints]", "; ".join(hint_parts)])
    if header:
        lines.extend(["", "[Header]", header])
    if row_lines:
        lines.extend(["", "[Rows]", *row_lines])
    return "\n".join(lines).strip()


def _structured_table_exact_row_pages(row: dict, fallback_page: int) -> List[int]:
    pages: List[int] = []
    for key in ("page", "page_start", "page_index"):
        value = row.get(key)
        if isinstance(value, int):
            page = value + 1 if key == "page_index" and value >= 0 else value
            if page > 0:
                pages.append(page)
                break
    if not pages:
        pages = [fallback_page]
    return sorted(set(pages))


_CANONICAL_CHUNK_BBOX_SPACE = "pdf_top_left_points"


def _canonical_bbox_of(record: Any) -> list:
    """Return a record's bbox in PDF top-left points, or [] when it has none.

    Table records keep their parser-owned bbox verbatim — MinerU emits
    ``normalized_0_1000`` and ODL emits ``pdf_bottom_left_points`` — and carry
    the converted box alongside it as ``visual_bbox`` (see
    ``services.document_geometry.visual_geometry``). Only the converted box may
    leave for a citation anchor.
    """
    if not isinstance(record, dict):
        return []
    candidates = [record.get("visual_bbox")]
    space = str(
        record.get("bbox_coordinate_space") or record.get("coordinate_space") or ""
    ).strip().lower()
    if space == _CANONICAL_CHUNK_BBOX_SPACE:
        # Records already authored in canonical points may carry only the raw key.
        candidates.extend([record.get("bounding_box"), record.get("bbox"), record.get("table_bbox")])
    for value in candidates:
        if isinstance(value, list) and len(value) >= 4:
            try:
                return [float(part) for part in value[:4]]
            except (TypeError, ValueError):
                continue
    return []


def _structured_table_chunk_geometry(bundle: dict, row: Optional[dict] = None) -> dict:
    """Build the bbox fields of a table chunk's metadata.

    ``_attach_block_index_citation_anchors`` treats any chunk that already has a
    bbox as authoritative: it stamps ``coordinate_space="pdf_top_left_points"``
    and skips the block-index backfill. Publishing a parser-space box there
    therefore both mislabels it and suppresses the correct fallback, so the
    highlight is drawn in the wrong place. Emit only converted boxes, with the
    space spelled out; when nothing is convertible the chunk carries no bbox and
    the block-index backfill takes over.
    """
    bboxes: list = []
    seen: set = set()

    def _add(bbox: list) -> None:
        if not bbox:
            return
        key = tuple(bbox)
        if key not in seen:
            seen.add(key)
            bboxes.append(bbox)

    if isinstance(row, dict):
        _add(_canonical_bbox_of(row))
        for cell in row.get("cell_evidence_units") or []:
            _add(_canonical_bbox_of(cell))
    if not bboxes:
        _add(_canonical_bbox_of(bundle))
    if not bboxes:
        return {"table_bbox": [], "table_bboxes": []}
    return {
        "table_bbox": bboxes[0],
        "table_bboxes": bboxes,
        "coordinate_space": _CANONICAL_CHUNK_BBOX_SPACE,
    }


def _build_structured_table_exact_row_chunk(bundle: dict, row: dict, row_number: int) -> str:
    caption = (bundle.get("table_caption") or bundle.get("table_id") or "").strip()
    table_id = (bundle.get("table_id") or "").strip()
    header = (bundle.get("table_header") or "").strip()
    bundle_id = (bundle.get("bundle_id") or "").strip()
    row_id = re.sub(r"\s+", " ", str(row.get("row_id") or f"row {row_number}")).strip()
    row_text = _structured_table_row_text(row)
    if not row_text:
        return ""

    hint_parts = []
    if table_id:
        hint_parts.append(f"table_id={table_id}")
    if bundle_id:
        hint_parts.append(f"bundle_id={bundle_id}")
    if row_id:
        hint_parts.append(f"row_id={row_id}")
    hint_parts.append(f"row={row_number}")

    lines = ["[Structured Table Exact Row]"]
    if caption:
        lines.extend(["", caption])
    if hint_parts:
        lines.extend(["", "[Hints]", "; ".join(hint_parts)])
    if header:
        lines.extend(["", "[Header]", header])
    lines.extend(["", "[Row]", row_text])
    return "\n".join(lines).strip()


def _structured_table_row_shard_pages(rows: List[dict], fallback_page: int) -> List[int]:
    pages: List[int] = []
    for row in rows:
        for key in ("page", "page_start", "page_index"):
            value = row.get(key)
            if isinstance(value, int):
                page = value + 1 if key == "page_index" and value >= 0 else value
                if page > 0:
                    pages.append(page)
                    break
    if not pages:
        pages = [fallback_page]
    return sorted(set(pages))


def _structured_table_selector_metadata(sanitized: dict) -> dict:
    """Metadata added by offline table-source selection, if present."""
    return {
        key: sanitized.get(key)
        for key in (
            "selected_source",
            "selection_reason",
            "table_selector_score",
            "table_selector_version",
            "table_selector_candidates",
        )
        if sanitized.get(key) not in (None, "", [], {})
    }


def _append_structured_table_bundle_chunks(
    doc_id: str,
    chunks: List[str],
    chunk_headings: List[str],
    chunk_pages: List[int],
    chunk_types: List[str],
    chunk_metadata: List[dict],
    structured_table_bundles: Optional[List[dict]],
) -> None:
    """将结构化表格 bundle 追加为 typed table chunks。"""
    if not structured_table_bundles:
        return

    appended = 0
    appended_exact_rows = 0
    appended_row_shards = 0
    seen_bundle_ids = set()
    for bundle in structured_table_bundles:
        sanitized = _sanitize_structured_table_bundle(bundle)
        if not sanitized:
            continue
        bundle_id = (sanitized.get("bundle_id") or "").strip()
        if bundle_id and bundle_id in seen_bundle_ids:
            continue
        chunk_text = _build_structured_table_bundle_chunk(sanitized)
        if not chunk_text:
            continue

        pages = sanitized.get("pages") or []
        if isinstance(pages, list):
            page_candidates = [int(page) for page in pages if isinstance(page, int) and page > 0]
        else:
            page_candidates = []
        primary_page = sanitized.get("page_start")
        if not isinstance(primary_page, int) or primary_page <= 0:
            primary_page = page_candidates[0] if page_candidates else 1
        primary_page_index = primary_page - 1
        primary_page_uid = _build_page_uid(primary_page)
        table_page_indices = [page - 1 for page in (page_candidates or [primary_page])]
        table_page_uids = [_build_page_uid(page) for page in (page_candidates or [primary_page])]

        chunk_metadata.append({
            "structured_table_bundle": True,
            "table_bundle_id": bundle_id,
            "evidence_unit_id": sanitized.get("evidence_unit_id", ""),
            "table_id": sanitized.get("table_id", ""),
            "table_caption": sanitized.get("table_caption", ""),
            "table_header": sanitized.get("table_header", ""),
            "table_body_markdown": sanitized.get("table_body_markdown", ""),
            "html_table": sanitized.get("html_table", ""),
            "table_footnote": sanitized.get("table_footnote", ""),
            "page_range": [sanitized.get("page_start", primary_page), sanitized.get("page_end", primary_page)],
            "table_pages": page_candidates or [primary_page],
            "page_index": primary_page_index,
            "page_uid": primary_page_uid,
            "table_page_indices": table_page_indices,
            "table_page_uids": table_page_uids,
            **_structured_table_chunk_geometry(sanitized),
            "table_source_ids": sanitized.get("source_ids", []),
            "table_instance_id": sanitized.get("table_instance_id", ""),
            "table_source_hash": sanitized.get("table_source_hash", ""),
            "evidence_units": sanitized.get("evidence_units", []),
            "source": sanitized.get("source", "odl"),
            **_structured_table_selector_metadata(sanitized),
        })
        chunks.append(chunk_text)
        chunk_headings.append(sanitized.get("table_caption") or sanitized.get("table_id") or "Structured Table Bundle")
        chunk_pages.append(primary_page)
        chunk_types.append("table")
        if bundle_id:
            seen_bundle_ids.add(bundle_id)
        appended += 1

        row_units = _extract_structured_table_row_shard_units(sanitized)
        if not row_units:
            continue

        heading_base = sanitized.get("table_caption") or sanitized.get("table_id") or "Structured Table"
        for row_offset, row in enumerate(row_units):
            row_number = row_offset + 1
            exact_row_text = _structured_table_row_text(row)
            exact_row_chunk = _build_structured_table_exact_row_chunk(sanitized, row, row_number)
            if not exact_row_text or not exact_row_chunk:
                continue
            row_pages = _structured_table_exact_row_pages(row, primary_page)
            row_primary_page = row_pages[0]
            row_page_indices = [page - 1 for page in row_pages]
            row_page_uids = [_build_page_uid(page) for page in row_pages]
            row_id = re.sub(r"\s+", " ", str(row.get("row_id") or f"row {row_number}")).strip()
            exact_id = f"{bundle_id or sanitized.get('table_id') or 'table'}:row:{row_number}"

            chunk_metadata.append({
                "structured_table_bundle": True,
                "table_row_evidence": True,
                "table_row_slice_kind": "exact",
                "parent_table_bundle_id": bundle_id,
                "table_bundle_id": bundle_id,
                "evidence_unit_id": exact_id,
                "row_id": row_id,
                "row_start": row_number,
                "row_end": row_number,
                "row_count": 1,
                "row_text": exact_row_text,
                "table_row_boundary_text": exact_row_text,
                "numeric_table_exact_context_row_text": exact_row_text,
                "numeric_table_exact_context_caption": sanitized.get("table_caption", ""),
                "numeric_table_exact_context_header": sanitized.get("table_header", ""),
                "table_id": sanitized.get("table_id", ""),
                "table_caption": sanitized.get("table_caption", ""),
                "table_header": sanitized.get("table_header", ""),
                "table_body_markdown": exact_row_text,
                "html_table": "",
                "table_footnote": sanitized.get("table_footnote", ""),
                "page_range": [row_pages[0], row_pages[-1]],
                "table_pages": row_pages,
                "page_index": row_primary_page - 1,
                "page_uid": _build_page_uid(row_primary_page),
                "table_page_indices": row_page_indices,
                "table_page_uids": row_page_uids,
                **_structured_table_chunk_geometry(sanitized, row=row),
                "table_source_ids": sanitized.get("source_ids", []),
                "table_instance_id": sanitized.get("table_instance_id", ""),
                "table_source_hash": sanitized.get("table_source_hash", ""),
                "evidence_units": [row],
                "cell_evidence_units": row.get("cell_evidence_units", []),
                "source": sanitized.get("source", "odl"),
                **_structured_table_selector_metadata(sanitized),
            })
            chunks.append(exact_row_chunk)
            chunk_headings.append(f"{heading_base} row {row_number}")
            chunk_pages.append(row_primary_page)
            chunk_types.append("table_row")
            appended_exact_rows += 1

        for shard_index, offset in enumerate(range(0, len(row_units), _STRUCTURED_TABLE_ROW_SHARD_SIZE)):
            shard_rows = row_units[offset:offset + _STRUCTURED_TABLE_ROW_SHARD_SIZE]
            row_start = offset + 1
            row_end = offset + len(shard_rows)
            shard_text = _build_structured_table_row_shard_chunk(
                sanitized,
                shard_rows,
                shard_index,
                row_start,
                row_end,
            )
            if not shard_text:
                continue
            shard_pages = _structured_table_row_shard_pages(shard_rows, primary_page)
            shard_primary_page = shard_pages[0]
            shard_page_indices = [page - 1 for page in shard_pages]
            shard_page_uids = [_build_page_uid(page) for page in shard_pages]
            shard_body = "\n".join(_structured_table_row_text(row) for row in shard_rows if _structured_table_row_text(row))
            shard_row_text = "\n".join(_structured_table_row_text(row) for row in shard_rows if _structured_table_row_text(row))
            shard_id = f"{bundle_id or sanitized.get('table_id') or 'table'}:rows:{row_start}-{row_end}"

            chunk_metadata.append({
                "structured_table_bundle": True,
                "table_row_shard": True,
                "table_row_slice_kind": "shard",
                "parent_table_bundle_id": bundle_id,
                "table_bundle_id": bundle_id,
                "evidence_unit_id": shard_id,
                "row_id": f"rows {row_start}-{row_end}",
                "row_start": row_start,
                "row_end": row_end,
                "row_count": len(shard_rows),
                "row_text": shard_row_text,
                "table_row_boundary_text": shard_row_text,
                "table_id": sanitized.get("table_id", ""),
                "table_caption": sanitized.get("table_caption", ""),
                "table_header": sanitized.get("table_header", ""),
                "table_body_markdown": shard_body,
                "html_table": "",
                "table_footnote": sanitized.get("table_footnote", ""),
                "page_range": [shard_pages[0], shard_pages[-1]],
                "table_pages": shard_pages,
                "page_index": shard_primary_page - 1,
                "page_uid": _build_page_uid(shard_primary_page),
                "table_page_indices": shard_page_indices,
                "table_page_uids": shard_page_uids,
                **_structured_table_chunk_geometry(sanitized),
                "table_source_ids": sanitized.get("source_ids", []),
                "table_instance_id": sanitized.get("table_instance_id", ""),
                "table_source_hash": sanitized.get("table_source_hash", ""),
                "evidence_units": shard_rows,
                "source": sanitized.get("source", "odl"),
                **_structured_table_selector_metadata(sanitized),
            })
            chunks.append(shard_text)
            chunk_headings.append(f"{heading_base} rows {row_start}-{row_end}")
            chunk_pages.append(shard_primary_page)
            chunk_types.append("table_row")
            appended_row_shards += 1

    if appended > 0:
        logger.info(
            f"[{doc_id}] 追加 {appended} 个 structured table bundle chunks，"
            f"{appended_exact_rows} 个 exact table rows，{appended_row_shards} 个 table row shards"
        )


def _maybe_append_runtime_structured_table_bundle_chunks(
    doc_id: str,
    chunks: List[str],
    chunk_headings: List[str],
    chunk_pages: List[int],
    chunk_types: List[str],
    chunk_metadata: List[dict],
    pages: Optional[List[dict]],
) -> None:
    """旧索引缺少 structured bundle 元数据时，基于 pages 运行时回补 bundle chunks。"""
    if not pages:
        return
    has_structured_bundle = any(
        isinstance(metadata, dict)
        and (
            metadata.get("structured_table_bundle")
            or metadata.get("table_bundle_id")
            or metadata.get("evidence_units")
        )
        for metadata in chunk_metadata
    )
    if has_structured_bundle:
        return
    if any("[Structured Table Bundle]" in str(chunk or "") for chunk in chunks):
        return

    runtime_bundles = _extract_page_text_table_bundles(pages)
    if not runtime_bundles:
        return

    before_count = len(chunks)
    _append_structured_table_bundle_chunks(
        doc_id,
        chunks,
        chunk_headings,
        chunk_pages,
        chunk_types,
        chunk_metadata,
        runtime_bundles,
    )
    appended = len(chunks) - before_count
    if appended > 0:
        logger.info(
            f"[{doc_id}] 运行时回补 {appended} 个 structured table bundle chunks "
            f"(旧索引缺少 chunk_metadata / structured bundles)"
        )


def _normalize_chunk_metadata_list(chunk_metadata: Optional[list], length: int) -> List[dict]:
    normalized: List[dict] = []
    source = chunk_metadata if isinstance(chunk_metadata, list) else []
    for idx in range(length):
        value = source[idx] if idx < len(source) and isinstance(source[idx], dict) else {}
        normalized.append(dict(value))
    return normalized


def _apply_chunk_metadata(item: dict, metadata: Optional[dict]) -> None:
    if not isinstance(metadata, dict):
        return
    for key, value in metadata.items():
        if value in (None, "", [], {}):
            continue
        if key == "page_range":
            if not item.get("page_range"):
                item["page_range"] = value
            continue
        if key == "table_pages":
            if not item.get("table_pages"):
                item["table_pages"] = value
            continue
        if key == "table_bundle_id":
            if not item.get("table_bundle_id"):
                item["table_bundle_id"] = value
            continue
        if key == "source" and item.get("source"):
            continue
        if not item.get(key):
            item[key] = value


_PAGE_TEXT_TABLE_CAPTION_RE = re.compile(r"Table\s*\d+\s*:", re.IGNORECASE)
_YOLO_FALLBACK_TABLE_MARKER = "[TABLE_YOLO_FALLBACK]"


def _trim_page_text_table_segment(segment: str, max_lines: int = 28) -> str:
    lines: List[str] = []
    narrative_run = 0
    for raw_line in (segment or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(lines) >= max_lines:
            break
        table_like = bool(
            re.search(
                r"\d|[%±×∆τ]|method|model|group|all|overall|many|few|med\.|acc|fid|auc|f1|map|score",
                line,
                re.IGNORECASE,
            )
        )
        if lines and not table_like:
            narrative_run += 1
            if narrative_run >= 2 and len(lines) >= 6:
                break
        else:
            narrative_run = 0
        lines.append(line)
    return "\n".join(lines).strip()


def _build_runtime_page_text_table_hints(table_id: str, header: str) -> dict[str, List[str]]:
    header_columns = _sort_numeric_columns(_extract_table_header_columns(header))
    return {
        "table_labels": [table_id] if table_id else [],
        "datasets": [],
        "backbones": _extract_table_header_backbones(header),
        "methods": [],
        "columns": header_columns,
        "comparison": [],
    }


def _estimate_runtime_table_row_numeric_span(segment: str) -> tuple[int, int]:
    row_pattern = _build_plain_table_row_pattern(3, 12)
    counts: List[int] = []
    for raw_line in (segment or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(?:Table|TABLE|表)\s*\.?\s*\d+", line, re.IGNORECASE):
            continue
        match = row_pattern.search(line)
        if not match:
            continue
        row_id = match.group(1).strip()
        normalized_row_id = re.sub(r"\s+", " ", row_id.lower()).strip()
        if (
            not normalized_row_id
            or normalized_row_id in {"method", "model", "group", "statistics"}
            or bool(re.search(r"(?:^|[-_])(?:lt|dataset|data)$", normalized_row_id, re.IGNORECASE))
            or bool(re.search(r"(?:19|20)\d{2}$", normalized_row_id))
        ):
            continue
        row_numbers = match.group(2).strip()
        numeric_tokens = len(_extract_numeric_value_tokens(row_numbers))
        if numeric_tokens >= 3:
            counts.append(numeric_tokens)
    if counts:
        best_count = max(Counter(counts).items(), key=lambda item: (item[1], item[0]))[0]
        return best_count, best_count
    return 0, 0


def _build_runtime_page_text_table_header(table_id: str, header: str) -> str:
    header_columns = _sort_numeric_columns(_extract_table_header_columns(header))
    if not header_columns:
        return header

    header_keys = set(header_columns)
    if "ΔAcc/||D_gen||" in header_keys or "||D_gen||" in header_keys:
        lead_column = "Group"
    elif {"All", "Many", "Med.", "Few"} & header_keys:
        lead_column = "Method"
    elif {"FID", "Acc"} & header_keys:
        lead_column = "Model"
    else:
        lead_column = "Row"
    return " | ".join([lead_column, *header_columns])


def _build_runtime_page_text_evidence_units(
    segment: str,
    table_id: str,
    caption: str,
    header: str,
    page_num: int,
    focus_hints: Optional[dict[str, List[str]]] = None,
) -> List[dict]:
    bundle_id = f"page-text:{table_id.lower()}" if table_id else ""
    parse_hints = _build_runtime_page_text_table_hints(table_id, header)
    min_span, max_span = _estimate_runtime_table_row_numeric_span(segment)
    if min_span > 0:
        parse_hints["_numeric_span_override"] = [min_span, max_span]
    row_units = _extract_plain_table_rows(segment, parse_hints)
    focus_hints = _normalize_numeric_table_focus_hints(focus_hints or parse_hints)
    if focus_hints is not parse_hints:
        focus_row_units = _extract_plain_table_rows(segment, focus_hints)
        if focus_row_units:
            merged_rows: dict[str, dict] = {}
            ordered_keys: List[str] = []

            def _store_row(row: dict) -> None:
                row_key = _normalize_sparse_bundle_row_key(
                    str(row.get("row_id") or ""),
                    str(row.get("row_text") or row.get("content") or ""),
                )
                if not row_key:
                    return
                if row_key in merged_rows:
                    return
                ordered_keys.append(row_key)
                merged_rows[row_key] = row

            for row in row_units:
                _store_row(row)
            for row in focus_row_units:
                _store_row(row)
            row_units = [merged_rows[key] for key in ordered_keys]
    evidence_units: List[dict] = []
    seen_rows: set[str] = set()

    for row_number, row in enumerate(row_units, start=1):
        if _is_headerish_numeric_table_row(row):
            continue
        focused_row = _build_query_focused_table_row(row, focus_hints)
        if not (focused_row.get("text") or "").strip() and focus_hints is not parse_hints:
            focused_row = _build_query_focused_table_row(row, parse_hints)
        row_text = (focused_row.get("text") or row.get("row_text") or "").strip()
        if not row_text:
            continue
        dedupe_key = re.sub(r"\s+", " ", row_text).strip().lower()
        if dedupe_key in seen_rows:
            continue
        seen_rows.add(dedupe_key)
        evidence_units.append(
            {
                "evidence_unit_id": f"{bundle_id}:row:{row_number}",
                "evidence_unit_type": "table_row",
                "table_bundle_id": bundle_id,
                "table_id": table_id,
                "table_caption": caption,
                "table_header": header,
                "table_focus_columns": focused_row.get("resolved_columns", parse_hints.get("columns", [])),
                "row_id": row.get("row_id", ""),
                "row_text": row_text,
                "row_numbers": row.get("row_numbers", ""),
                "content": row_text,
                "row_number": row_number,
                "page": page_num,
            }
        )

    return evidence_units


def _build_runtime_page_text_table_body(segment: str, evidence_units: List[dict]) -> str:
    if not evidence_units:
        return segment
    body_lines = [
        (unit.get("row_text") or unit.get("content") or "").strip()
        for unit in evidence_units
        if (unit.get("row_text") or unit.get("content") or "").strip()
    ]
    if body_lines:
        return "\n".join(body_lines)
    return segment


def _split_yolo_fallback_table_segments(page_text: str) -> List[str]:
    if _YOLO_FALLBACK_TABLE_MARKER not in (page_text or ""):
        return []
    segments: List[str] = []
    parts = page_text.split(_YOLO_FALLBACK_TABLE_MARKER)
    for raw_part in parts[1:]:
        lines: List[str] = [_YOLO_FALLBACK_TABLE_MARKER]
        for raw_line in raw_part.splitlines():
            line = raw_line.strip()
            if not line:
                if len(lines) > 1:
                    break
                continue
            if line == _YOLO_FALLBACK_TABLE_MARKER:
                break
            if line.startswith("[") and line.endswith("]") and line != _YOLO_FALLBACK_TABLE_MARKER:
                break
            lines.append(line)
            if len(lines) >= 18:
                break
        segment = "\n".join(lines).strip()
        if len(segment) >= 40:
            segments.append(segment)
    return segments


def _build_yolo_fallback_table_id(segment: str, page_num: int, index: int) -> tuple[str, str]:
    caption = _extract_table_caption_from_text(segment)
    table_id = _extract_table_id(caption)
    if table_id:
        return table_id, caption or table_id
    return f"YOLO Table p{page_num}-{index}", f"YOLO fallback table on page {page_num}"


def _extract_yolo_fallback_table_bundles(page_text: str, page_num: int, seen_table_ids: set[str]) -> List[dict]:
    bundles: List[dict] = []
    for idx, raw_segment in enumerate(_split_yolo_fallback_table_segments(page_text), start=1):
        segment = raw_segment.replace(_YOLO_FALLBACK_TABLE_MARKER, "").strip()
        if len(segment) < 30:
            continue
        numeric_count = len(_extract_numeric_value_tokens(segment))
        if numeric_count < 2 and not re.search(r"\b(table|method|model|dataset|acc|all|many|few|med)\b", segment, re.IGNORECASE):
            continue
        table_id, caption = _build_yolo_fallback_table_id(segment, page_num, idx)
        table_key = table_id.lower()
        if table_key in seen_table_ids:
            continue
        raw_header = _extract_table_header_snippet(segment)
        normalized_header = _build_runtime_page_text_table_header(table_id, raw_header)
        evidence_units = _build_runtime_page_text_evidence_units(
            segment,
            table_id,
            caption,
            normalized_header or raw_header,
            page_num,
        )
        table_body = _build_runtime_page_text_table_body(segment, evidence_units)
        bundles.append({
            "bundle_id": f"yolo-page-text:p{page_num}:table:{idx}",
            "table_id": table_id,
            "table_caption": caption,
            "table_header": normalized_header or raw_header,
            "table_body_markdown": table_body,
            "html_table": "",
            "table_footnote": "",
            "page_start": page_num,
            "page_end": page_num,
            "pages": [page_num],
            "page_index": page_num - 1,
            "page_uid": _build_page_uid(page_num),
            "page_uids": [_build_page_uid(page_num)],
            "source_ids": [],
            "evidence_units": evidence_units,
            "source": "doclayout_yolo_fallback",
        })
        seen_table_ids.add(table_key)
    return bundles


def _normalize_sparse_bundle_row_key(row_id: str, row_text: str) -> str:
    normalized_row_id = _normalize_numeric_table_method_token(row_id or "")
    if normalized_row_id:
        return normalized_row_id
    sample = re.sub(r"\s+", " ", (row_text or "")).strip().lower()
    sample = re.sub(r"[^0-9a-z\u4e00-\u9fffτ∆×%().+\-/|= ]+", "", sample)
    return sample[:160]


def _extract_structured_bundle_body_rows(bundle: dict) -> List[dict]:
    body = (bundle.get("table_body_markdown") or "").strip()
    if not body:
        chunk_text = (bundle.get("chunk") or bundle.get("raw_chunk_text") or "").strip()
        if chunk_text:
            return _extract_serialized_structured_bundle_body_rows(
                chunk_text,
                table_id=(bundle.get("table_id") or "").strip(),
                table_caption=(bundle.get("table_caption") or "").strip(),
                table_header=(bundle.get("table_header") or "").strip(),
            )
        return []
    rows = _extract_markdown_table_rows(body)
    if rows:
        return rows
    table_id = (bundle.get("table_id") or "").strip()
    header = (bundle.get("table_header") or "").strip()
    hints = _build_runtime_page_text_table_hints(table_id, header)
    return _extract_plain_table_rows(body, hints, query="")


def _merge_sparse_bundle_evidence_units(
    bundle: dict,
    recovered_units: List[dict],
    *,
    table_id: str,
    caption: str,
    header: str,
    page_num: int,
) -> List[dict]:
    existing_units = bundle.get("evidence_units")
    existing_row_units: dict[str, dict] = {}
    passthrough_units: List[dict] = []
    if isinstance(existing_units, list):
        for unit in existing_units:
            if not isinstance(unit, dict):
                continue
            if (unit.get("evidence_unit_type") or "").strip().lower() != "table_row":
                passthrough_units.append(dict(unit))
                continue
            if unit.get("is_header_row"):
                continue
            row_key = _normalize_sparse_bundle_row_key(
                str(unit.get("row_id") or ""),
                str(unit.get("row_text") or unit.get("content") or ""),
            )
            if row_key and row_key not in existing_row_units:
                existing_row_units[row_key] = dict(unit)

    merged_rows: List[dict] = []
    seen_keys: set[str] = set()
    for recovered_unit in recovered_units:
        row_key = _normalize_sparse_bundle_row_key(
            str(recovered_unit.get("row_id") or ""),
            str(recovered_unit.get("row_text") or recovered_unit.get("content") or ""),
        )
        if not row_key or row_key in seen_keys:
            continue
        seen_keys.add(row_key)
        merged = dict(recovered_unit)
        existing = existing_row_units.pop(row_key, None)
        if existing:
            for key, value in existing.items():
                if value not in (None, "", [], {}):
                    if key in {"row_text", "content", "row_numbers", "table_caption", "table_header", "table_focus_columns"} and merged.get(key):
                        continue
                    merged[key] = value
        merged["table_bundle_id"] = (
            merged.get("table_bundle_id")
            or bundle.get("table_bundle_id")
            or bundle.get("bundle_id", "")
        )
        merged["table_id"] = merged.get("table_id") or table_id
        merged["table_caption"] = merged.get("table_caption") or caption
        merged["table_header"] = merged.get("table_header") or header
        merged["page"] = merged.get("page") or page_num
        merged_rows.append(merged)

    for row_key, unit in existing_row_units.items():
        if row_key not in seen_keys:
            merged_rows.append(unit)

    return merged_rows + passthrough_units


def _prefer_sparse_bundle_text(existing_text: str, candidate_text: str, *, header_like: bool) -> str:
    existing = re.sub(r"\s+", " ", str(existing_text or "")).strip()
    candidate = re.sub(r"\s+", " ", str(candidate_text or "")).strip()
    if not candidate:
        return existing
    if not existing:
        return candidate
    if header_like:
        existing_columns = len(_extract_table_header_columns(existing))
        candidate_columns = len(_extract_table_header_columns(candidate))
        if existing_columns > candidate_columns:
            return existing
        if existing_columns == candidate_columns and existing.count(" ") >= candidate.count(" "):
            return existing
        return candidate
    if existing.count(" ") >= candidate.count(" "):
        return existing
    return candidate


def _maybe_upgrade_sparse_structured_bundle(
    chunk_text: str,
    metadata: Optional[dict],
    page_payload: Optional[dict],
    *,
    query: str = "",
) -> tuple[str, Optional[dict]]:
    if not should_apply_numeric_table_specialization():
        return chunk_text, metadata
    if not isinstance(metadata, dict) or not metadata.get("structured_table_bundle"):
        return chunk_text, metadata
    if not isinstance(page_payload, dict):
        return chunk_text, metadata

    table_id = str(metadata.get("table_id") or "").strip()
    if not table_id:
        return chunk_text, metadata

    existing_row_units = [
        unit for unit in (metadata.get("evidence_units") or [])
        if isinstance(unit, dict)
        and (unit.get("evidence_unit_type") or "").strip().lower() == "table_row"
        and (unit.get("row_text") or unit.get("content") or "").strip()
    ]
    existing_body = str(metadata.get("table_body_markdown") or "").strip()
    query_hints = _query_rewriter_singleton.extract_numeric_table_hints(query) if query else {}
    focus_tokens = [
        re.sub(r"\s+", "", str(value or "")).lower()
        for value in [
            *(query_hints.get("columns") or []),
            *(query_hints.get("methods") or []),
            *(query_hints.get("comparison") or []),
        ]
        if value
    ]
    existing_search_text = " ".join(
        [
            existing_body,
            *[
                str(unit.get("row_text") or unit.get("content") or unit.get("row_numbers") or "")
                for unit in existing_row_units
            ],
        ]
    )
    existing_search_norm = re.sub(r"\s+", "", existing_search_text).lower()
    existing_has_focus = bool(
        focus_tokens and all(token in existing_search_norm for token in focus_tokens)
    )
    if len(existing_row_units) >= 2 and existing_body and (not focus_tokens or existing_has_focus):
        return chunk_text, metadata

    page_text = (page_payload.get("content") or page_payload.get("text") or "").strip()
    if not page_text:
        return chunk_text, metadata

    try:
        page_num = int(page_payload.get("page") or 0)
    except (TypeError, ValueError):
        page_num = 0
    if page_num <= 0:
        page_num = _resolve_primary_page_from_metadata(metadata)
    if page_num <= 0:
        return chunk_text, metadata

    scoped_segment = _slice_text_to_requested_table(page_text, [table_id])
    if not scoped_segment:
        return chunk_text, metadata
    segment = _trim_page_text_table_segment(scoped_segment)
    if len(segment) < 40:
        return chunk_text, metadata

    current_rows = _extract_structured_table_rows(metadata)
    current_body_rows = _extract_structured_bundle_body_rows(metadata)
    current_keys = {
        _normalize_sparse_bundle_row_key(row.get("row_id", ""), row.get("row_text", ""))
        for row in [*current_rows, *current_body_rows]
        if _normalize_sparse_bundle_row_key(row.get("row_id", ""), row.get("row_text", ""))
    }

    raw_header = _extract_table_header_snippet(segment)
    recovered_header_candidate = _build_runtime_page_text_table_header(
        table_id,
        raw_header,
    )
    recovered_header = _prefer_sparse_bundle_text(
        str(metadata.get("table_header") or "").strip(),
        recovered_header_candidate,
        header_like=True,
    ) or recovered_header_candidate or str(metadata.get("table_header") or "").strip()
    recovered_caption_candidate = _extract_table_caption_from_text(segment) or table_id
    recovered_caption = _prefer_sparse_bundle_text(
        str(metadata.get("table_caption") or "").strip(),
        recovered_caption_candidate,
        header_like=False,
    ) or recovered_caption_candidate
    recovered_units = _build_runtime_page_text_evidence_units(
        segment,
        table_id,
        recovered_caption,
        recovered_header,
        page_num,
        focus_hints=query_hints,
    )
    validation_units = recovered_units
    existing_row_fallback_units: List[dict] = []
    for row in [*current_rows, *current_body_rows]:
        if not isinstance(row, dict):
            continue
        row_unit = dict(row)
        row_unit.setdefault("evidence_unit_type", "table_row")
        row_unit.setdefault("table_id", table_id)
        row_unit.setdefault("table_caption", recovered_caption)
        row_unit.setdefault("table_header", recovered_header)
        row_unit.setdefault("page", page_num)
        existing_row_fallback_units.append(row_unit)
    if existing_row_fallback_units:
        validation_units = _merge_sparse_bundle_evidence_units(
            {
                "evidence_units": existing_row_fallback_units,
                "table_bundle_id": str(metadata.get("table_bundle_id") or ""),
                "bundle_id": str(metadata.get("bundle_id") or ""),
            },
            recovered_units,
            table_id=table_id,
            caption=recovered_caption,
            header=recovered_header,
            page_num=page_num,
        )
    recovered_keys = {
        _normalize_sparse_bundle_row_key(unit.get("row_id", ""), unit.get("row_text", ""))
        for unit in validation_units
        if _normalize_sparse_bundle_row_key(unit.get("row_id", ""), unit.get("row_text", ""))
    }
    current_row_count = len(current_keys)
    recovered_row_count = len(recovered_keys)
    if recovered_row_count < max(2, current_row_count + 1):
        return chunk_text, metadata
    if current_keys and not current_keys.issubset(recovered_keys):
        return chunk_text, metadata

    upgraded = dict(metadata)
    upgraded["evidence_units"] = _merge_sparse_bundle_evidence_units(
        upgraded,
        recovered_units,
        table_id=table_id,
        caption=recovered_caption,
        header=recovered_header,
        page_num=page_num,
    )
    existing_header_columns = len(
        _extract_table_header_columns(str(metadata.get("table_header") or ""))
    )
    recovered_header_columns = len(_extract_table_header_columns(recovered_header))
    if recovered_header and recovered_header_columns >= existing_header_columns:
        upgraded["table_header"] = recovered_header
    if recovered_caption:
        upgraded["table_caption"] = recovered_caption
    recovered_body = _build_runtime_page_text_table_body(segment, recovered_units)
    existing_body = str(metadata.get("table_body_markdown") or "").strip()
    if recovered_body:
        focus_tokens = [
            re.sub(r"\s+", "", str(value or "")).lower()
            for value in (query_hints.get("columns") or query_hints.get("methods") or [])
            if value
        ]
        recovered_body_norm = re.sub(r"\s+", "", recovered_body).lower()
        existing_body_norm = re.sub(r"\s+", "", existing_body).lower()
        recovered_has_focus = bool(focus_tokens and any(token in recovered_body_norm for token in focus_tokens))
        existing_has_focus = bool(focus_tokens and any(token in existing_body_norm for token in focus_tokens))
        if len(recovered_body) > len(existing_body) or (recovered_has_focus and not existing_has_focus):
            upgraded["table_body_markdown"] = recovered_body
    upgraded["sparse_table_bundle"] = True
    upgraded["sparse_table_bundle_source"] = "page_text_row_recovery"
    upgraded["sparse_table_bundle_original_rows"] = current_row_count
    upgraded["sparse_table_bundle_recovered_rows"] = recovered_row_count

    upgraded_chunk = _build_structured_table_bundle_chunk(upgraded)
    return upgraded_chunk or chunk_text, upgraded


def _extract_page_text_table_bundles(pages: Optional[List[dict]]) -> List[dict]:
    """在 ODL 不可用时，从现有页面文本中提取 Table X 结构化 bundle。"""
    bundles: List[dict] = []
    seen_table_ids: set[str] = set()

    for page in pages or []:
        if not isinstance(page, dict):
            continue
        page_num = page.get("page")
        if not isinstance(page_num, int) or page_num <= 0:
            page_num = page.get("page_num") if isinstance(page.get("page_num"), int) else 1
        page_text = (page.get("content") or page.get("text") or "").strip()
        if not page_text:
            continue

        yolo_bundles = _extract_yolo_fallback_table_bundles(page_text, page_num, seen_table_ids)
        if yolo_bundles:
            bundles.extend(yolo_bundles)

        matches = list(_PAGE_TEXT_TABLE_CAPTION_RE.finditer(page_text))
        if not matches:
            continue

        for idx, match in enumerate(matches):
            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(page_text)
            raw_segment = page_text[match.start():next_start]
            matched_table_id = _extract_table_id(match.group(0))
            if matched_table_id:
                scoped_segment = _slice_text_to_requested_table(
                    page_text[match.start():],
                    [matched_table_id],
                )
                if scoped_segment:
                    raw_segment = scoped_segment
            segment = _trim_page_text_table_segment(raw_segment)
            if len(segment) < 40 and next_start > match.start():
                segment = _trim_page_text_table_segment(page_text[match.start():next_start])
            if len(segment) < 40:
                continue

            caption = _extract_table_caption_from_text(segment) or segment.splitlines()[0].strip()
            table_id = _extract_table_id(caption) or _extract_table_id(match.group(0))
            if not table_id or table_id.lower() in seen_table_ids:
                continue
            raw_header = _extract_table_header_snippet(segment)
            normalized_header = _build_runtime_page_text_table_header(table_id, raw_header)
            evidence_units = _build_runtime_page_text_evidence_units(
                segment,
                table_id,
                caption,
                normalized_header or raw_header,
                page_num,
            )
            table_body = _build_runtime_page_text_table_body(segment, evidence_units)

            bundles.append({
                "bundle_id": f"page-text:{table_id.lower()}",
                "table_id": table_id,
                "table_caption": caption,
                "table_header": normalized_header or raw_header,
                "table_body_markdown": table_body,
                "html_table": "",
                "table_footnote": "",
                "page_start": page_num,
                "page_end": page_num,
                "pages": [page_num],
                "page_index": page_num - 1,
                "page_uid": _build_page_uid(page_num),
                "page_uids": [_build_page_uid(page_num)],
                "source_ids": [],
                "evidence_units": evidence_units,
                "source": "page_text_bundle",
            })
            seen_table_ids.add(table_id.lower())

    return bundles


# 这些块的结构本身就是语义，不能过长度切分器。表格被按字符切开后表头只留在片 0，
# 后续片以 ``| cell48 | value48 |`` 这样的无表头残片进入索引；公式和代码同理会被腰斩。
# 参考 kotaemon 的做法：按 metadata type 分桶，只有文本类过 splitter。
# 表格的可检索粒度由 ``_append_structured_table_bundle_chunks`` 的整表 / exact-row /
# shard 三级负责，这里的 block 级副本只是粗粒度兜底，切碎它有害无益。
_UNSPLITTABLE_EVIDENCE_BLOCK_TYPES = frozenset({"table", "formula", "code"})


def _split_evidence_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    allow_split: bool = True,
) -> List[str]:
    """Split one source block without discarding its evidence identity."""
    value = preprocess_text(text)
    if not value:
        return []
    if not allow_split or len(value) <= chunk_size:
        return [value]
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        pieces = splitter.split_text(value)
    except Exception:
        step = max(1, chunk_size - max(0, min(chunk_overlap, chunk_size - 1)))
        pieces = [value[index:index + chunk_size] for index in range(0, len(value), step)]
    return [piece for piece in pieces if str(piece or "").strip()]


def _build_structured_evidence_chunks(
    evidence_chunks: Optional[List[dict]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[List[str], List[str], List[int], List[str], List[dict]]:
    """Normalize block-index evidence into aligned vector-index inputs.

    The vector index may split a long source block, but every resulting
    fragment keeps the original ``block_id``/bbox/section identity so citations
    never need a post-retrieval fuzzy match for current structured documents.
    """
    chunks: List[str] = []
    headings: List[str] = []
    pages: List[int] = []
    chunk_types: List[str] = []
    metadata_items: List[dict] = []

    for evidence_index, item in enumerate(evidence_chunks or []):
        if not isinstance(item, dict):
            continue
        raw_text = str(item.get("text") or item.get("content") or "").strip()
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        metadata = dict(metadata)
        heading = str(item.get("heading") or metadata.get("section_path") or "").strip()
        try:
            page = int(item.get("page") or metadata.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if page <= 0:
            page = 1
        chunk_type = str(
            item.get("block_type")
            or metadata.get("block_type")
            or metadata.get("chunk_type")
            or "text"
        ).strip().lower() or "text"
        pieces = _split_evidence_text(
            raw_text,
            chunk_size,
            chunk_overlap,
            allow_split=chunk_type not in _UNSPLITTABLE_EVIDENCE_BLOCK_TYPES,
        )
        if not pieces:
            continue
        block_id = str(metadata.get("block_id") or "").strip()
        if block_id and not metadata.get("block_ids"):
            metadata["block_ids"] = [block_id]
        metadata.setdefault("block_type", chunk_type)
        metadata.setdefault("chunk_type", chunk_type)
        metadata.setdefault("page", page)
        metadata.setdefault("page_range", [page, page])
        if heading:
            metadata.setdefault("section_path", heading)
        if not metadata.get("evidence_id"):
            metadata["evidence_id"] = f"evidence:{evidence_index}"

        for fragment_index, piece in enumerate(pieces):
            fragment_metadata = dict(metadata)
            if len(pieces) > 1:
                fragment_metadata["evidence_fragment_index"] = fragment_index
                fragment_metadata["evidence_fragment_count"] = len(pieces)
                fragment_metadata["evidence_id"] = (
                    f"{metadata['evidence_id']}:f{fragment_index + 1}"
                )
                # 每个分片都原样继承整块 bbox（浅拷贝），高亮框会覆盖远多于该片的
                # 文字。这里没有片级坐标可用，但至少要把精度说清楚，别让引用层
                # 把它当成片级精确框。
                if fragment_metadata.get("bbox") or fragment_metadata.get("rects"):
                    fragment_metadata["bbox_precision"] = "block"
            chunks.append(piece)
            headings.append(heading)
            pages.append(page)
            chunk_types.append(chunk_type)
            metadata_items.append(fragment_metadata)

    return chunks, headings, pages, chunk_types, metadata_items


def _bind_chunk_parse_identity(chunk_metadata: List[dict], index_meta: Optional[dict]) -> None:
    """Persist the active parse identity on every evidence-bearing chunk."""
    meta = index_meta if isinstance(index_meta, dict) else {}
    parse_generation = str(meta.get("parse_generation") or "").strip()
    document_source_hash = str(
        meta.get("document_source_hash") or meta.get("source_hash") or ""
    ).strip()
    parser_route = str(meta.get("parser_route") or "").strip()
    for item in chunk_metadata:
        if not isinstance(item, dict):
            continue
        if parse_generation:
            item.setdefault("parse_generation", parse_generation)
        if document_source_hash:
            item.setdefault("document_source_hash", document_source_hash)
        if parser_route:
            item.setdefault("parser_route", parser_route)


def build_vector_index(
    doc_id: str,
    text: str,
    vector_store_dir: str,
    embedding_model_id: str = "local-minilm",
    api_key: str = None,
    api_host: str = None,
    pages: List[dict] = None,
    evidence_chunks: Optional[List[dict]] = None,
    structured_table_bundles: Optional[List[dict]] = None,
    summary_api_key: str = None,
    summary_model: str = "gpt-4o-mini",
    summary_provider: str = "openai",
    summary_api_host: str = "",
    index_source: str = "pdf_native",
    index_meta: Optional[dict] = None,
    build_semantic_groups: bool = True,
    embedding_provider: Optional[str] = None,
):
    try:
        logger.info(f"[{doc_id}] Building vector index...")
        # 使用 Model_ID_Resolver 统一解析模型 ID
        registry_key, config = resolve_model_id(embedding_model_id)
        if registry_key is not None:
            embedding_model_id = registry_key
        else:
            available_models = get_available_model_ids()
            raise ValueError(
                f"Embedding 模型 '{embedding_model_id}' 未配置或不受支持，"
                f"可用模型列表: {available_models}"
            )
        embedding_identity = _canonicalize_embedding_identity(
            embedding_model_id,
            embedding_provider=embedding_provider,
            base_url=api_host,
        )
        embedding_model_id = embedding_identity["model"]
        embedding_provider = embedding_identity["provider"]
        verified_api_host = embedding_identity["api_host"]

        # 分块策略：按模型最大上下文自适应，默认 1200 / 200（约 15-20% 重叠），限制在 1000-2500
        chunk_size, chunk_overlap = get_chunk_params(embedding_model_id, base_chunk_size=1200, base_overlap=200)
        effective_index_meta = dict(index_meta or {})
        effective_index_source = str(
            index_source or effective_index_meta.get("index_source") or "pdf_native"
        ).strip() or "pdf_native"
        pages = _annotate_pages_with_provenance(pages)
        chunk_page_index = _build_page_index(pages or [])

        chunks, chunk_headings, chunk_pages, chunk_types, chunk_metadata = _build_structured_evidence_chunks(
            evidence_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if chunks:
            evidence_schema_versions: list[int] = []
            for item in chunk_metadata:
                if not isinstance(item, dict):
                    continue
                try:
                    evidence_schema_versions.append(
                        int(item.get("evidence_schema_version") or 0)
                    )
                except (TypeError, ValueError):
                    continue
            effective_index_meta["content_source"] = "block_index_evidence"
            effective_index_meta["evidence_schema_version"] = max(evidence_schema_versions, default=0)
            logger.info(
                "[%s] 使用 block-index evidence 分块，生成 %s 个带来源身份的分块",
                doc_id,
                len(chunks),
            )
        else:
            # Do not trust caller-provided provenance: a missing evidence_chunks
            # argument means this artifact was actually rebuilt from flattened text.
            effective_index_meta["content_source"] = "document_full_text"
            effective_index_meta["evidence_schema_version"] = 0
            preprocessed_text = preprocess_text(text)
            # 优先使用结构感知分块，保护表格和公式完整性（需求 4.1, 4.2, 4.3, 4.4）
            try:
                chunks_with_ctx = structure_aware_split_with_context(
                    preprocessed_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                if chunks_with_ctx:
                    chunks = [c for c, _ in chunks_with_ctx]
                    chunk_headings = [h for _, h in chunks_with_ctx]
                    logger.info(f"[{doc_id}] 使用结构感知分块，生成 {len(chunks)} 个分块")
                else:
                    raise ValueError("结构感知分块返回空结果")
            except Exception as e:
                # 回退到 RecursiveCharacterTextSplitter（需求 4.4 安全降级）
                logger.warning(f"结构感知分块失败，回退到 RecursiveCharacterTextSplitter: {e}")
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
                chunks = text_splitter.split_text(preprocessed_text)
                chunk_headings = [""] * len(chunks)
            chunk_pages = [
                _find_page_for_chunk(chunk_text, pages or [], page_index=chunk_page_index) if pages else 1
                for chunk_text in chunks
            ]
            chunk_types = [_guess_chunk_type(chunk_text) for chunk_text in chunks]
            chunk_metadata = [{} for _ in chunks]

        # Contextual chunking uses the same section paths regardless of whether
        # the chunks came from raw page text or the structured block index.
        from services.rag_config import RAGConfig as _BuildRAGConfig
        _build_rag_config = _BuildRAGConfig()
        effective_table_bundles = structured_table_bundles or _extract_page_text_table_bundles(pages)

        _append_structured_table_bundle_chunks(
            doc_id,
            chunks,
            chunk_headings,
            chunk_pages,
            chunk_types,
            chunk_metadata,
            effective_table_bundles,
        )
        _bind_chunk_parse_identity(chunk_metadata, effective_index_meta)

        logger.info(f"[{doc_id}] Split into {len(chunks)} chunks")
        if not chunks:
            raise ValueError("向量索引构建失败：文档没有生成任何可索引分块")
        invalid_chunk_indices = [
            index
            for index, chunk in enumerate(chunks)
            if not isinstance(chunk, str) or not chunk.strip()
        ]
        if invalid_chunk_indices:
            preview = ", ".join(str(index) for index in invalid_chunk_indices[:8])
            raise ValueError(f"向量索引构建失败：存在空白分块（chunk={preview}）")

        aligned_fields = {
            "chunk_headings": chunk_headings,
            "chunk_pages": chunk_pages,
            "chunk_types": chunk_types,
            "chunk_metadata": chunk_metadata,
        }
        misaligned_fields = [
            name for name, values in aligned_fields.items()
            if len(values) != len(chunks)
        ]
        if misaligned_fields:
            raise ValueError(
                "向量索引构建失败：分块元数据长度不一致（"
                + ", ".join(misaligned_fields)
                + "）"
            )

        embed_fn = get_embedding_function(
            embedding_model_id,
            api_key,
            verified_api_host,
            False,
        )

        # Contextual Chunking：用带章节前缀的文本做 embedding，提升语义区分度
        if _build_rag_config.enable_contextual_chunking:
            embed_texts = []
            ctx_count = 0
            for chunk_text, heading in zip(chunks, chunk_headings):
                if heading:
                    embed_texts.append(f"[章节: {heading}]\n{chunk_text}")
                    ctx_count += 1
                else:
                    embed_texts.append(chunk_text)
            if ctx_count > 0:
                logger.info(f"[{doc_id}] Contextual Chunking: {ctx_count}/{len(chunks)} 个 chunk 注入章节上下文")
            embeddings = embed_fn(embed_texts)
        else:
            embeddings = embed_fn(chunks)

        try:
            embeddings_f32 = np.asarray(embeddings, dtype="float32")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"向量索引构建失败：嵌入结果无法转换为数值矩阵: {exc}") from exc
        if embeddings_f32.ndim != 2:
            raise ValueError(
                "向量索引构建失败：嵌入结果必须是二维矩阵，"
                f"实际 ndim={embeddings_f32.ndim}"
            )
        if embeddings_f32.shape[0] != len(chunks):
            raise ValueError(
                "向量索引构建失败：嵌入向量数量与分块数量不一致，"
                f"vectors={embeddings_f32.shape[0]}, chunks={len(chunks)}"
            )
        if embeddings_f32.shape[1] <= 0:
            raise ValueError("向量索引构建失败：嵌入向量维度为空")
        if not np.all(np.isfinite(embeddings_f32)):
            raise ValueError("向量索引构建失败：嵌入结果包含 NaN 或 Infinity")
        vector_norms = np.linalg.norm(embeddings_f32, axis=1)
        if not np.all(np.isfinite(vector_norms)) or np.any(vector_norms <= 0):
            raise ValueError("向量索引构建失败：嵌入结果包含零向量或无效向量")

        embeddings_f32 = np.ascontiguousarray(embeddings_f32, dtype="float32")
        # 归一化向量，使 Inner Product = 余弦相似度
        faiss.normalize_L2(embeddings_f32)
        if not np.all(np.isfinite(embeddings_f32)):
            raise ValueError("向量索引构建失败：归一化后的嵌入结果包含非有限值")

        dimension = embeddings_f32.shape[1]
        n_vectors = embeddings_f32.shape[0]

        if n_vectors > 2000:
            # 大文档：使用 IVF 索引加速检索
            n_clusters = min(64, n_vectors // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings_f32)
            index.nprobe = min(8, n_clusters)
            logger.info(f"[{doc_id}] 使用 IndexIVFFlat: {n_vectors} 向量, {n_clusters} 簇")
        else:
            index = faiss.IndexFlatIP(dimension)

        index.add(embeddings_f32)
        if int(index.ntotal) != len(chunks) or int(index.ntotal) <= 0:
            raise RuntimeError(
                "向量索引构建失败：FAISS 向量数量异常，"
                f"vectors={int(index.ntotal)}, chunks={len(chunks)}"
            )

        os.makedirs(vector_store_dir, exist_ok=True)
        index_path = os.path.join(vector_store_dir, f"{doc_id}.index")
        chunks_path = os.path.join(vector_store_dir, f"{doc_id}.pkl")

        faiss.write_index(index, index_path)
        persisted_index = faiss.read_index(index_path)
        persisted_vector_count = int(persisted_index.ntotal)
        persisted_dimension = int(persisted_index.d)
        if persisted_vector_count != len(chunks) or persisted_vector_count <= 0:
            raise RuntimeError(
                "向量索引落盘校验失败：向量数量异常，"
                f"vectors={persisted_vector_count}, chunks={len(chunks)}"
            )
        if persisted_dimension != dimension or persisted_dimension <= 0:
            raise RuntimeError(
                "向量索引落盘校验失败：向量维度异常，"
                f"dimension={persisted_dimension}, expected={dimension}"
            )

        # Parent-Child 分块：生成 parent chunks 并保存映射
        parent_chunks = []
        child_to_parent = {}  # child_index -> parent_index
        parent_chunk_size = chunk_size * 3  # parent ~3600 字符
        i = 0
        while i < len(chunks):
            # 合并连续 child chunks 为一个 parent
            parent_parts = []
            parent_len = 0
            parent_idx = len(parent_chunks)
            start_i = i
            while i < len(chunks) and parent_len + len(chunks[i]) + 2 <= parent_chunk_size:
                parent_parts.append(chunks[i])
                parent_len += len(chunks[i]) + 2
                child_to_parent[i] = parent_idx
                i += 1
            if not parent_parts:
                # 单个 chunk 超过 parent_chunk_size
                parent_parts.append(chunks[i])
                child_to_parent[i] = parent_idx
                i += 1
            parent_chunks.append("\n\n".join(parent_parts))

        vector_build_id = uuid.uuid4().hex
        save_data = {
            "index_version": RAG_INDEX_VERSION,
            "embedding_identity_version": EMBEDDING_IDENTITY_VERSION,
            "vector_build_id": vector_build_id,
            "chunks": chunks,
            "embedding_model": embedding_model_id,
            "embedding_provider": embedding_provider,
            "embedding_api_host": verified_api_host,
            "chunk_headings": chunk_headings,
            "chunk_pages": chunk_pages,
            "chunk_types": chunk_types,
            "chunk_metadata": chunk_metadata,
            "parent_chunks": parent_chunks,
            "child_to_parent": child_to_parent,
            "index_source": effective_index_source,
            "source_hash": effective_index_meta.get("source_hash", ""),
            "rebuilt_at": effective_index_meta.get("rebuilt_at", ""),
            "previous_index_source": effective_index_meta.get("previous_index_source", ""),
            "normalizer_version": effective_index_meta.get("normalizer_version", ""),
            "index_meta": effective_index_meta,
            "vector_count": persisted_vector_count,
            "vector_dimension": persisted_dimension,
            "build_validation": {
                "valid": True,
                "chunk_count": len(chunks),
                "vector_count": persisted_vector_count,
                "vector_dimension": persisted_dimension,
            },
        }
        with open(chunks_path, "wb") as f:
            pickle.dump(save_data, f)

        semantic_identity = _extract_vector_semantic_identity(save_data)

        logger.info(f"[{doc_id}] Vector index saved to {index_path}")

        # ---- 语义意群异步生成与意群级别向量索引构建（需求 6.1）----
        if build_semantic_groups:
            _build_semantic_group_index_async(
                doc_id=doc_id,
                chunks=chunks,
                pages=pages,
                chunk_pages=chunk_pages,
                chunk_types=chunk_types,
                chunk_metadata=chunk_metadata,
                embed_fn=embed_fn,
                api_key=summary_api_key,
                model=summary_model,
                provider=summary_provider,
                endpoint=summary_api_host,
                source_hash=str(
                    effective_index_meta.get("document_source_hash")
                    or effective_index_meta.get("source_hash")
                    or ""
                ),
                transaction_id=str(effective_index_meta.get("parse_generation") or ""),
                vector_store_dir=vector_store_dir,
                semantic_identity=semantic_identity,
            )

        return {
            "status": "ready",
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "vector_count": persisted_vector_count,
            "vector_dimension": persisted_dimension,
            "index_path": index_path,
            "chunks_path": chunks_path,
            "index_source": effective_index_source,
            "parse_generation": str(effective_index_meta.get("parse_generation") or ""),
            "document_source_hash": str(effective_index_meta.get("document_source_hash") or ""),
            "vector_build_id": vector_build_id,
        }

    except Exception as e:
        logger.error(f"[{doc_id}] Error building vector index: {e}")
        raise


def _build_semantic_group_index_async(
    doc_id: str,
    chunks: list[str],
    pages: list[dict],
    embed_fn,
    api_key: str = None,
    chunk_pages: Optional[List[int]] = None,
    chunk_types: Optional[List[str]] = None,
    chunk_metadata: Optional[List[dict]] = None,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    endpoint: str = "",
    output_dir: str | None = None,
    raise_on_error: bool = False,
    source_hash: str = "",
    transaction_id: str = "",
    vector_store_dir: str = "",
    semantic_identity: Optional[dict] = None,
):
    """异步启动意群生成任务（需求 6.1, 6.4）

    使用受限后台任务执行意群生成，不阻塞文档上传流程。
    同一解析代际只允许一个任务，全局额度满时跳过可选构建，避免上传高峰
    创建无限线程。主向量索引和问答不会受影响，后续可重新触发意群构建。

    Args:
        doc_id: 文档唯一标识
        chunks: 文本分块列表
        pages: 文档页面数据列表
        embed_fn: 嵌入函数
        api_key: LLM API 密钥（用于意群摘要生成）
    """
    normalized_semantic_identity = _normalize_semantic_generation_identity(semantic_identity)
    task_key = f"{doc_id}:{normalized_semantic_identity.get('vector_build_id') or transaction_id or 'legacy'}"
    with _group_generation_lock:
        if task_key in _group_generation_in_progress:
            logger.info(f"[{doc_id}] 意群生成任务已在进行中，跳过")
            return {"status": "disabled", "group_count": 0, "paths": []}
        if not _semantic_group_background_admission.acquire(blocking=False):
            logger.warning(f"[{doc_id}] 意群生成任务队列已满，跳过本次可选构建")
            return {
                "status": "deferred",
                "reason": "queue_full",
                "group_count": 0,
                "paths": [],
            }
        _group_generation_in_progress.add(task_key)

    def _task():
        try:
            _build_semantic_group_index(
                doc_id,
                chunks,
                pages,
                embed_fn,
                chunk_pages=chunk_pages,
                chunk_types=chunk_types,
                chunk_metadata=chunk_metadata,
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                output_dir=output_dir,
                raise_on_error=raise_on_error,
                source_hash=source_hash,
                transaction_id=transaction_id,
                vector_store_dir=vector_store_dir,
                semantic_identity=normalized_semantic_identity,
            )
        except Exception as e:
            # 任务失败时记录日志（需求 6.4），不影响主流程
            logger.error(f"[{doc_id}] 意群生成后台任务失败: {e}", exc_info=True)
        finally:
            with _group_generation_lock:
                _group_generation_in_progress.discard(task_key)
            _semantic_group_background_admission.release()

    try:
        threading.Thread(
            target=_task,
            name=f"chatpdf-semantic-group-{doc_id[:12]}",
            daemon=True,
        ).start()
    except Exception:
        with _group_generation_lock:
            _group_generation_in_progress.discard(task_key)
        _semantic_group_background_admission.release()
        raise
    logger.info(f"[{doc_id}] 意群生成后台任务已启动")
    return {"status": "queued", "group_count": 0, "paths": []}


def _build_semantic_group_index(
    doc_id: str,
    chunks: List[str],
    pages: List[dict],
    embed_fn,
    api_key: str = None,
    chunk_pages: Optional[List[int]] = None,
    chunk_types: Optional[List[str]] = None,
    chunk_metadata: Optional[List[dict]] = None,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    endpoint: str = "",
    output_dir: str | None = None,
    raise_on_error: bool = False,
    source_hash: str = "",
    transaction_id: str = "",
    vector_store_dir: str = "",
    semantic_identity: Optional[dict] = None,
):
    """在分块索引构建完成后，生成语义意群并构建意群级别向量索引

    流程：
    1. 检查 RAGConfig.enable_semantic_groups 是否启用
    2. 从 pages 数据推导每个分块对应的页码（chunk_pages）
    3. 调用 SemanticGroupService.generate_groups 生成意群
    4. 为意群的 digest 文本构建 FAISS 向量索引
    5. 保存意群数据（JSON）和意群向量索引（FAISS + pkl）

    Args:
        doc_id: 文档唯一标识
        chunks: 文本分块列表
        pages: 文档页面数据列表（每个元素包含 page 和 content 字段），可为 None
        embed_fn: 嵌入函数
        api_key: LLM API 密钥（用于意群摘要生成）
    """
    from services.rag_config import RAGConfig
    from services.semantic_group_service import SemanticGroupService

    config = RAGConfig()

    # 检查是否启用语义意群功能
    if not config.enable_semantic_groups:
        logger.info(f"[{doc_id}] 语义意群功能已禁用，跳过意群生成")
        return {"status": "disabled", "group_count": 0, "paths": []}

    staged_dir = ""
    normalized_semantic_identity = _normalize_semantic_generation_identity(semantic_identity)
    try:
        logger.info(f"[{doc_id}] 开始生成语义意群...")

        narrative_chunks, narrative_pages, original_indices = _prepare_narrative_semantic_inputs(
            chunks,
            pages,
            chunk_pages=chunk_pages,
            chunk_types=chunk_types,
            chunk_metadata=chunk_metadata,
        )
        if not narrative_chunks:
            logger.warning(f"[{doc_id}] 没有可用于语义意群的正文分块，跳过意群索引构建")
            return {"status": "empty", "group_count": 0, "paths": []}

        # 创建 SemanticGroupService 实例
        group_service = SemanticGroupService(
            api_key=api_key or "",
            model=model or "gpt-4o-mini",
            provider=provider or "openai",
            endpoint=endpoint or "",
        )

        # 调用 generate_groups 生成语义意群（异步方法需要在同步上下文中运行）
        groups = _run_async(group_service.generate_groups(
            chunks=narrative_chunks,
            chunk_pages=narrative_pages,
            target_chars=config.target_group_chars,
            min_chars=config.min_group_chars,
            max_chars=config.max_group_chars,
        ))

        if not groups:
            logger.warning(f"[{doc_id}] 语义意群生成结果为空，跳过意群索引构建")
            return {"status": "empty", "group_count": 0, "paths": []}

        # SemanticGroupService 对过滤后的正文数组编号。检索侧仍用完整向量
        # chunk 数组反查，因此发布前必须恢复为原始 chunk id。
        for group in groups:
            group.chunk_indices = [
                original_indices[index]
                for index in group.chunk_indices
                if isinstance(index, int) and 0 <= index < len(original_indices)
            ]

        logger.info(f"[{doc_id}] 生成了 {len(groups)} 个语义意群")

        # Transactional callers supply their own staging directory. Ordinary
        # background writes also stage first, then atomically switch active.
        publish_active_generation = output_dir is None
        semantic_root = _get_semantic_groups_dir()
        if publish_active_generation:
            staging_root = os.path.join(semantic_root, "_tmp")
            os.makedirs(staging_root, exist_ok=True)
            staged_dir = tempfile.mkdtemp(prefix=f"{doc_id}.", dir=staging_root)
            groups_store_dir = staged_dir
        else:
            groups_store_dir = output_dir
        os.makedirs(groups_store_dir, exist_ok=True)

        # 保存意群数据为 JSON
        group_service.save_groups(doc_id, groups, groups_store_dir)
        logger.info(f"[{doc_id}] 意群数据已保存到 {groups_store_dir}")

        groups_json_path = os.path.join(groups_store_dir, f"{doc_id}.json")
        if normalized_semantic_identity:
            try:
                with open(groups_json_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    payload.update({
                        "parse_generation": normalized_semantic_identity.get("parse_generation") or "",
                        "document_source_hash": normalized_semantic_identity.get("document_source_hash") or "",
                        "vector_build_id": normalized_semantic_identity.get("vector_build_id") or "",
                        "embedding_identity_version": int(normalized_semantic_identity.get("embedding_identity_version") or 0),
                        "embedding_model": normalized_semantic_identity.get("embedding_model") or "",
                        "embedding_provider": normalized_semantic_identity.get("embedding_provider") or "",
                        "embedding_api_host": normalized_semantic_identity.get("embedding_api_host") or "",
                        "vector_dimension": int(normalized_semantic_identity.get("vector_dimension") or 0),
                    })
                    with open(groups_json_path, "w", encoding="utf-8") as handle:
                        json.dump(payload, handle, ensure_ascii=False, indent=2)
            except Exception as exc:
                raise RuntimeError(f"写入 semantic groups 身份元数据失败: {exc}") from exc

        # 为意群的 digest 文本构建 FAISS 向量索引
        digest_texts = [g.digest for g in groups]
        group_ids = [g.group_id for g in groups]

        paths = [os.path.join(groups_store_dir, f"{doc_id}.json")]
        if digest_texts:
            group_embeddings = embed_fn(digest_texts)
            dimension = group_embeddings.shape[1]
            group_index = faiss.IndexFlatL2(dimension)
            group_index.add(np.array(group_embeddings).astype('float32'))

            # 保存意群 FAISS 索引
            group_index_path = os.path.join(groups_store_dir, f"{doc_id}_groups.index")
            faiss.write_index(group_index, group_index_path)

            # 保存意群元数据（digest 文本列表和 group_id 映射）
            group_meta_path = os.path.join(groups_store_dir, f"{doc_id}_groups.pkl")
            with open(group_meta_path, "wb") as f:
                pickle.dump({
                    "digest_texts": digest_texts,
                    "group_ids": group_ids,
                    "parse_generation": normalized_semantic_identity.get("parse_generation") or "",
                    "document_source_hash": normalized_semantic_identity.get("document_source_hash") or "",
                    "vector_build_id": normalized_semantic_identity.get("vector_build_id") or "",
                    "embedding_identity_version": int(normalized_semantic_identity.get("embedding_identity_version") or 0),
                    "embedding_model": normalized_semantic_identity.get("embedding_model") or "",
                    "embedding_provider": normalized_semantic_identity.get("embedding_provider") or "",
                    "embedding_api_host": normalized_semantic_identity.get("embedding_api_host") or "",
                    "vector_dimension": int(normalized_semantic_identity.get("vector_dimension") or dimension),
                }, f)

            paths.extend([group_index_path, group_meta_path])
            logger.info(
                f"[{doc_id}] 意群向量索引已保存: "
                f"index={group_index_path}, meta={group_meta_path}, "
                f"共 {len(groups)} 个意群"
            )
        if publish_active_generation:
            if not _semantic_generation_identity_complete(normalized_semantic_identity):
                raise RuntimeError("semantic groups 发布必须绑定完整向量身份")
            artifact_paths = {
                "json": Path(groups_store_dir) / f"{doc_id}.json",
                "index": Path(groups_store_dir) / f"{doc_id}_groups.index",
                "pkl": Path(groups_store_dir) / f"{doc_id}_groups.pkl",
            }
            validation = validate_semantic_group_artifacts(
                artifact_paths,
                doc_id,
                expected_identity=normalized_semantic_identity,
                expected_vector_dimension=int(normalized_semantic_identity.get("vector_dimension") or 0),
            )
            if not validation["valid"]:
                raise RuntimeError("semantic groups validation failed: " + ", ".join(validation["errors"]))
            with get_document_publication_lock(doc_id):
                if not _semantic_generation_matches_vector_index(
                    doc_id,
                    vector_store_dir,
                    semantic_identity=normalized_semantic_identity,
                ):
                    logger.info(
                        "[%s] 丢弃已过期的语义意群后台结果: generation=%s build=%s",
                        doc_id,
                        normalized_semantic_identity.get("parse_generation") or transaction_id,
                        normalized_semantic_identity.get("vector_build_id") or "",
                    )
                    return {"status": "stale", "group_count": 0, "paths": []}
                published = publish_generation(
                    semantic_root,
                    doc_id,
                    groups_store_dir,
                    source_hash=source_hash,
                    transaction_id=transaction_id,
                    semantic_identity=normalized_semantic_identity,
                )
            _index_cache.invalidate(doc_id)
            paths = list(published["paths"].values())
            staged_dir = ""
        return {"status": "ready", "group_count": len(groups), "paths": paths}

    except Exception as e:
        # 意群生成失败不影响普通异步主流程；staged 路径可显式要求失败上抛。
        logger.warning(f"[{doc_id}] 语义意群生成失败，继续使用分块级别索引: {e}")
        if raise_on_error:
            raise
        return {"status": "failed", "group_count": 0, "paths": [], "error": str(e)}
    finally:
        if staged_dir:
            shutil.rmtree(staged_dir, ignore_errors=True)


def _derive_chunk_pages(chunks: List[str], pages: List[dict]) -> List[int]:
    """从 pages 数据推导每个分块对应的页码

    使用 _find_page_for_chunk 函数将每个分块映射到对应的页码。
    如果 pages 数据不可用，则所有分块默认分配到第 1 页。

    Args:
        chunks: 文本分块列表
        pages: 文档页面数据列表，可为 None

    Returns:
        每个分块对应的页码列表
    """
    if not pages:
        # 没有页面数据时，所有分块默认分配到第 1 页
        return [1] * len(chunks)

    return [_find_page_for_chunk(chunk, pages) for chunk in chunks]


def _prepare_narrative_semantic_inputs(
    chunks: List[str],
    pages: Optional[List[dict]],
    *,
    chunk_pages: Optional[List[int]] = None,
    chunk_types: Optional[List[str]] = None,
    chunk_metadata: Optional[List[dict]] = None,
) -> tuple[List[str], List[int], List[int]]:
    """只让正文进入 narrative semantic groups，并保留原始 chunk id。"""
    derived_pages = _derive_chunk_pages(chunks, pages or [])
    aligned_pages = list(chunk_pages or [])
    aligned_types = list(chunk_types or [])
    aligned_metadata = list(chunk_metadata or [])
    narrative_chunks: List[str] = []
    narrative_pages: List[int] = []
    original_indices: List[int] = []

    for index, chunk in enumerate(chunks):
        text = str(chunk or "").strip()
        if not text:
            continue
        metadata = aligned_metadata[index] if index < len(aligned_metadata) and isinstance(aligned_metadata[index], dict) else {}
        chunk_type = str(
            aligned_types[index] if index < len(aligned_types) else ""
        ).strip().lower() or _guess_chunk_type(text)
        table_metadata = bool(
            metadata.get("structured_table_bundle")
            or metadata.get("table_row_evidence")
            or metadata.get("table_row_shard")
            or metadata.get("table_bundle_id")
        )
        if chunk_type in {"table", "table_row", "table_cell"} or table_metadata:
            continue
        if _is_table_fragment(text):
            continue

        page = aligned_pages[index] if index < len(aligned_pages) else 0
        if not isinstance(page, int) or page <= 0:
            page = derived_pages[index] if index < len(derived_pages) else 1
        narrative_chunks.append(text)
        narrative_pages.append(page)
        original_indices.append(index)

    return narrative_chunks, narrative_pages, original_indices


def _run_async(coro):
    """在同步上下文中运行异步协程

    如果当前已有事件循环在运行，则使用 nest_asyncio 或创建新线程；
    否则直接使用 asyncio.run()。

    Args:
        coro: 异步协程对象

    Returns:
        协程的返回值
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 当前已有事件循环在运行（如在 FastAPI 请求处理中）
        # 使用新线程运行异步任务
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def _load_group_index(doc_id: str) -> Optional[dict]:
    """加载意群级别 FAISS 索引和元数据

    从运行时 semantic_groups 目录加载意群的 FAISS 索引文件和 pkl 元数据文件。
    如果文件不存在或加载失败，返回 None。

    Args:
        doc_id: 文档唯一标识

    Returns:
        包含 index、digest_texts、group_ids 的字典，加载失败时返回 None
    """
    # 确定意群数据存储目录
    groups_store_dir = _get_semantic_groups_dir(doc_id)

    group_index_path = os.path.join(groups_store_dir, f"{doc_id}_groups.index")
    group_meta_path = os.path.join(groups_store_dir, f"{doc_id}_groups.pkl")

    if not os.path.exists(group_index_path) or not os.path.exists(group_meta_path):
        logger.info(f"[{doc_id}] 意群级别索引不存在，回退到仅分块级别检索")
        return None

    # 优先从缓存读取
    cached_gid = _index_cache.get_group_index(doc_id)
    if cached_gid is not None:
        return cached_gid

    try:
        group_index = faiss.read_index(group_index_path)
        with open(group_meta_path, "rb") as f:
            group_meta = pickle.load(f)

        digest_texts = group_meta.get("digest_texts", [])
        group_ids = group_meta.get("group_ids", [])

        if not digest_texts or not group_ids:
            logger.warning(f"[{doc_id}] 意群元数据为空，回退到仅分块级别检索")
            return None

        result = {
            "index": group_index,
            "digest_texts": digest_texts,
            "group_ids": group_ids,
        }
        _index_cache.put_group_index(doc_id, result)
        logger.info(f"[{doc_id}] 已加载意群级别索引，共 {len(group_ids)} 个意群")
        return result
    except Exception as e:
        logger.warning(f"[{doc_id}] 加载意群级别索引失败，回退到仅分块级别检索: {e}")
        return None


def _load_group_data(doc_id: str) -> Optional[dict]:
    """加载意群 JSON 数据，获取每个意群包含的 chunk_indices 映射

    用于在 RRF 融合后进行同组 chunk 去重。

    Args:
        doc_id: 文档唯一标识

    Returns:
        group_id -> chunk_indices 的映射字典，加载失败时返回 None
    """
    groups_json_path = os.path.join(_get_semantic_groups_dir(doc_id), f"{doc_id}.json")

    if not os.path.exists(groups_json_path):
        return None

    # 优先从缓存读取
    cached_gcm = _index_cache.get_group_data(doc_id)
    if cached_gcm is not None:
        return cached_gcm

    try:
        import json
        with open(groups_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        groups = data.get("groups", [])
        # 构建 group_id -> chunk_indices 映射
        group_chunk_map = {}
        for g in groups:
            group_chunk_map[g["group_id"]] = g.get("chunk_indices", [])

        _index_cache.put_group_data(doc_id, group_chunk_map)
        return group_chunk_map
    except Exception as e:
        logger.warning(f"[{doc_id}] 加载意群 JSON 数据失败: {e}")
        return None


def _search_group_index(
    group_index_data: dict,
    query_vector: np.ndarray,
    search_k: int,
) -> List[dict]:
    """在意群级别 FAISS 索引中搜索

    Args:
        group_index_data: _load_group_index 返回的字典
        query_vector: 查询向量
        search_k: 搜索返回的最大结果数

    Returns:
        意群级别搜索结果列表，每个元素包含 group_id、distance 等信息
    """
    group_index = group_index_data["index"]
    group_ids = group_index_data["group_ids"]

    # 限制搜索数量不超过索引中的向量数
    actual_k = min(search_k, group_index.ntotal)
    if actual_k <= 0:
        return []

    D, I = group_index.search(np.array(query_vector).astype('float32'), actual_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(group_ids):
            results.append({
                "group_id": group_ids[idx],
                "distance": float(dist),
                "group_rank": len(results),  # 在意群搜索中的排名
            })

    return results


def _rrf_merge_chunk_and_group(
    chunk_results: List[dict],
    group_results: List[dict],
    group_chunk_map: Optional[dict],
    chunks: List[str],
    pages: List[dict],
    query: str,
    top_k: int = 10,
    k: int = 60,
) -> List[dict]:
    """使用 RRF 算法融合分块级别和意群级别检索结果

    RRF 公式: score = sum(1 / (k + rank_i)) 对每个排名列表

    融合策略：
    1. 分块级别结果直接参与 RRF 排名
    2. 意群级别结果展开为其包含的所有 chunk，每个 chunk 继承意群的排名
    3. 同一 chunk 在两路结果中的 RRF 分数累加
    4. 同组 chunk 去重：属于同一意群的多个 chunk 只保留 RRF 分数最高的

    Args:
        chunk_results: 分块级别检索结果列表
        group_results: 意群级别检索结果列表
        group_chunk_map: group_id -> chunk_indices 映射，可为 None
        chunks: 所有文本分块列表
        pages: 文档页面数据
        query: 用户查询文本
        top_k: 返回结果数量
        k: RRF 常数（默认 60）

    Returns:
        融合后的结果列表，按 RRF 分数降序排列
    """
    # 步骤 1：计算分块级别的 RRF 分数
    # chunk_text -> rrf_score
    rrf_scores = {}
    # chunk_text -> 原始结果数据
    chunk_data = {}
    # chunk_text -> 所属 group_id（用于去重）
    chunk_group_map = {}

    for rank, item in enumerate(chunk_results):
        chunk_text = item.get("chunk", "")
        if not chunk_text:
            continue
        rrf_score = 1.0 / (k + rank + 1)
        rrf_scores[chunk_text] = rrf_scores.get(chunk_text, 0.0) + rrf_score
        if chunk_text not in chunk_data:
            chunk_data[chunk_text] = item.copy()
        _append_retrieval_source(chunk_data[chunk_text], item.get("retrieval_source") or "vector")

    # 步骤 2：将意群级别结果展开为 chunk 级别，计算 RRF 分数
    if group_results and group_chunk_map:
        for rank, group_item in enumerate(group_results):
            group_id = group_item["group_id"]
            chunk_indices = group_chunk_map.get(group_id, [])
            group_rrf_score = 1.0 / (k + rank + 1)

            for chunk_idx in chunk_indices:
                if 0 <= chunk_idx < len(chunks):
                    chunk_text = chunks[chunk_idx]
                    # 累加意群级别的 RRF 分数
                    rrf_scores[chunk_text] = rrf_scores.get(chunk_text, 0.0) + group_rrf_score

                    # 记录 chunk 所属的 group_id
                    if chunk_text not in chunk_group_map:
                        chunk_group_map[chunk_text] = group_id

                    # 如果该 chunk 还没有结果数据，创建一个
                    if chunk_text not in chunk_data:
                        page_num = _find_page_for_chunk(chunk_text, pages)
                        snippet, highlights = _extract_snippet_and_highlights(chunk_text, query)
                        chunk_data[chunk_text] = {
                            "chunk": chunk_text,
                            "page": page_num,
                            "score": 0.0,
                            "similarity": 0.5,
                            "similarity_percent": 50.0,
                            "snippet": snippet,
                            "highlights": highlights,
                            "reranked": False,
                        }
                    _append_retrieval_source(chunk_data[chunk_text], "semantic_group")

    # 步骤 3：同组 chunk 去重 —— 属于同一意群的多个 chunk 只保留 RRF 分数最高的 2 个
    if chunk_group_map:
        # 构建反向映射：chunk_index -> group_id（基于 group_chunk_map）
        chunk_idx_to_group = {}
        if group_chunk_map:
            for gid, indices in group_chunk_map.items():
                for idx in indices:
                    if 0 <= idx < len(chunks):
                        chunk_idx_to_group[chunks[idx]] = gid

        # 按 group_id 分组，每组只保留 RRF 分数最高的 2 个 chunk
        # group_id -> [(chunk_text, rrf_score), ...]
        group_chunks = {}
        chunks_to_remove = set()

        for chunk_text, rrf_score in rrf_scores.items():
            gid = chunk_idx_to_group.get(chunk_text)
            if gid is None:
                # 不属于任何意群的 chunk，保留
                continue

            if gid not in group_chunks:
                group_chunks[gid] = [(chunk_text, rrf_score)]
            else:
                group_chunks[gid].append((chunk_text, rrf_score))

        # 每组保留 top-2
        for gid, chunk_list in group_chunks.items():
            if len(chunk_list) <= 2:
                continue
            # 按 RRF 分数降序排列，移除第 3 个及之后的
            chunk_list.sort(key=lambda x: x[1], reverse=True)
            for chunk_text, _ in chunk_list[2:]:
                chunks_to_remove.add(chunk_text)

        # 移除被去重的 chunk
        for ct in chunks_to_remove:
            rrf_scores.pop(ct, None)

    # 步骤 4：按 RRF 分数排序并返回 top_k 结果
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_text, rrf_score in sorted_chunks[:top_k]:
        item = chunk_data.get(chunk_text, {})
        if not item:
            continue
        item = item.copy()
        item["rrf_score"] = rrf_score
        item["hybrid"] = True
        results.append(item)

    return results


def _is_table_fragment(text: str) -> bool:
    """检测文本是否为表格/数据碎片

    表格碎片特征：
    - 大量孤立的数字（被空格分隔的短数字序列）
    - 缺少完整句子（没有句号结尾的长句）
    - 高比例的数字 token vs 文字 token
    """
    if not text or len(text) < 20:
        return False

    # 按空格拆分为 token
    tokens = text.split()
    if len(tokens) < 3:
        return False

    # 统计数字 token（纯数字或小数）和文字 token
    num_tokens = 0
    for t in tokens:
        cleaned = t.strip('(),%↑↓·-')
        if not cleaned:
            continue
        # 纯数字、小数、百分比、带单位的数字（如 2.0m, 1.5m）
        if re.match(r'^-?\d+\.?\d*[a-zA-Z]?$', cleaned):
            num_tokens += 1

    num_ratio = num_tokens / len(tokens) if tokens else 0

    # 检查是否有完整句子（至少一个 10+ 字符的句子以句号结尾）
    sentences = re.split(r'[.!?。！？]', text)
    has_real_sentence = any(len(s.strip()) > 30 for s in sentences)

    # 数字 token 占比 > 25% 且没有完整句子 → 表格碎片（从 35% 降至 25%，减少误判）
    if num_ratio > 0.25 and not has_real_sentence:
        return True

    # 数字 token 占比 > 50% → 几乎肯定是表格
    if num_ratio > 0.5:
        return True

    return False


_REFERENCE_QUERY_HINTS = (
    "参考文献",
    "文献",
    "引用",
    "引文",
    "相关工作",
    "references",
    "reference",
    "bibliography",
    "citation",
    "citations",
    "related work",
)


def _is_reference_query(query: str) -> bool:
    """判断用户问题是否在询问文献/引用信息。"""
    if not query:
        return False
    q = query.lower()
    return any(hint in q for hint in _REFERENCE_QUERY_HINTS)


def _is_reference_like_text(text: str) -> bool:
    """检测文本是否呈现“参考文献列表”风格。"""
    if not text:
        return False

    sample = text[:1200]
    sample_lower = sample.lower()

    lines = [ln.strip() for ln in sample.splitlines() if ln.strip()]
    first_line = lines[0].lower() if lines else ""
    if re.match(
        r"^(?:\d+(?:\.\d+)*[.)]?\s*)?(?:references|bibliography|参考文献)\b",
        first_line,
        re.IGNORECASE,
    ):
        return True

    citation_markers = len(re.findall(r"\[[0-9]{1,3}\]", sample))
    year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", sample))
    et_al_hits = sample_lower.count("et al")
    doi_hits = len(re.findall(r"\b(?:doi|arxiv)\b", sample_lower))
    author_hits = len(re.findall(r"\b[A-Z][a-z]+,\s*(?:[A-Z]\.|[A-Z][a-z]+)", sample))

    numbered_lines = sum(1 for ln in lines if re.match(r"^\[?\d{1,3}\]?[.)]?\s", ln))
    if (
        re.match(r"^\[\d{1,3}\]", sample)
        and citation_markers >= 2
        and year_hits >= 2
    ):
        return True
    reference_entry_lines = sum(
        1
        for line in lines
        if re.search(r"\b(?:19|20)\d{2}\b", line)
        and (
            re.match(r"^\[?\d{1,3}\]?[.)]?\s", line)
            or re.search(r"\bet\s+al\.?\b|\bdoi\b|\barxiv\b", line, re.IGNORECASE)
            or re.search(r"\b[A-Z][a-z]+,\s*(?:[A-Z]\.|[A-Z][a-z]+)", line)
        )
    )
    if reference_entry_lines >= 2:
        return True

    # 双栏 PDF 的旧 full_text 可能把整页参考文献压成单行。此时不能要求
    # 换行结构，但高密度的年份 + 书目信号仍足以区分文献页和方法正文。
    venue_hits = len(re.findall(
        r"\b(?:proceedings|conference|journal|transactions|advances|springer|ieee|acm)\b|\bpp\.\s*\d+",
        sample_lower,
    ))
    if year_hits >= 5 and (
        doi_hits >= 2
        or author_hits >= 3
        or venue_hits >= 3
        or (et_al_hits >= 3 and venue_hits >= 2)
    ):
        return True

    signal = 0
    if citation_markers >= 2:
        signal += 1
    if year_hits >= 2:
        signal += 1
    if et_al_hits >= 1 or author_hits >= 2:
        signal += 1
    if doi_hits >= 1:
        signal += 1
    if lines and (numbered_lines / len(lines)) >= 0.35 and year_hits >= 1:
        signal += 1

    # 正文的 related-work 句子经常同时包含多个年份和 ``et al.``。只有
    # 三种以上信号，并且文本还呈现多行/多编号条目结构时，才当成文献列表。
    return signal >= 3 and (len(lines) >= 2 or citation_markers >= 3)


def filter_reference_trap_results(
    results: List[dict],
    query: str,
    evidence_need: Optional[List[str]] = None,
) -> List[dict]:
    """统一过滤参考文献型污染结果，供主检索链路和其他调用方复用。"""
    if not results:
        return results
    if _is_reference_query(query) or "reference_meta" in (evidence_need or []):
        normalized = []
        for item in results:
            normalized_item = dict(item)
            _normalize_structural_metadata(normalized_item)
            normalized.append(normalized_item)
        return normalized

    filtered_results = []
    removed = 0
    numeric_table_query = "numeric_table" in (evidence_need or [])
    for item in results:
        normalized_item = dict(item)
        _normalize_structural_metadata(normalized_item)
        chunk_text = normalized_item.get("raw_chunk_text") or normalized_item.get("chunk", "")
        structural_path = " ".join(
            str(normalized_item.get(key) or "")
            for key in ("chunk_heading", "section_path", "section_title")
        ).strip()
        reference_section = bool(
            structural_path
            and re.search(r"(?:^|\b)(?:references|bibliography|参考文献)(?:\b|$)", structural_path, re.I)
        )
        if chunk_text and (reference_section or _is_reference_like_text(chunk_text)):
            chunk_type = normalized_item.get("chunk_type") or normalized_item.get("block_type") or ""
            if numeric_table_query and _looks_like_numeric_table_support(chunk_text, chunk_type):
                filtered_results.append(normalized_item)
                continue
            normalized_item["reference_like"] = True
            removed += 1
            continue
        filtered_results.append(normalized_item)

    if filtered_results:
        if removed > 0:
            logger.info(f"[检索净化] 过滤参考文献型结果 {removed} 条")
        return filtered_results
    if removed > 0:
        logger.info("[检索净化] 候选全部为参考文献型结果，返回空结果触发正文降级")
    return []


def filter_reference_trap_texts(
    texts: List[str],
    query: str,
    evidence_need: Optional[List[str]] = None,
) -> List[str]:
    wrapped = [{"chunk": text} for text in texts if isinstance(text, str) and text.strip()]
    filtered = filter_reference_trap_results(wrapped, query, evidence_need=evidence_need)
    return [item.get("chunk", "") for item in filtered if item.get("chunk", "").strip()]


def _phrase_boost(results: List[dict], query: str, boost_factor: float = 1.2) -> List[dict]:
    """对包含完整查询短语的 chunk 进行相似度加权提升

    向量检索是语义匹配，可能把只包含部分关键词的碎片排在前面。
    此函数检查每个 chunk 是否包含完整的查询短语（忽略大小写），
    如果包含则提升其 similarity 和 similarity_percent。

    同时对"表格碎片"进行降权。

    Args:
        results: 搜索结果列表
        query: 用户查询文本
        boost_factor: 提升倍数（默认 1.5）

    Returns:
        重新排序后的结果列表
    """
    if not results or not query or len(query.strip()) < 2:
        return results

    query_lower = query.lower().strip()
    reference_query = _is_reference_query(query)
    # 将查询拆分为单词，用于计算覆盖率
    query_terms = [t for t in re.split(r"[\s,;，。；、]+", query_lower) if len(t) > 1]

    for item in results:
        chunk_text = item.get("chunk", "")
        chunk_lower = chunk_text.lower()
        if not chunk_lower:
            continue

        # 检测表格碎片：降权到 0.5x
        if _is_table_fragment(chunk_text):
            item["similarity"] = item.get("similarity", 0) * 0.5
            item["similarity_percent"] = round(item.get("similarity_percent", 0) * 0.5, 2)
            item["table_fragment"] = True
            continue

        # 非“文献查询”场景下，对参考文献型文本降权，避免其占据高位引用
        if not reference_query and _is_reference_like_text(chunk_text):
            item["similarity"] = item.get("similarity", 0) * 0.65
            item["similarity_percent"] = round(item.get("similarity_percent", 0) * 0.65, 2)
            item["reference_like"] = True
            continue

        # 完整短语匹配：最大提升
        if query_lower in chunk_lower:
            item["similarity"] = min(item.get("similarity", 0) * boost_factor, 1.0)
            item["similarity_percent"] = min(round(item.get("similarity_percent", 0) * boost_factor, 2), 99.99)
            item["phrase_match"] = True
            continue

        # 部分词覆盖率加权：覆盖越多提升越大
        if query_terms:
            matched = sum(1 for t in query_terms if t in chunk_lower)
            coverage = matched / len(query_terms)
            if coverage >= 0.8:
                factor = 1.0 + (boost_factor - 1.0) * coverage * 0.5
                item["similarity"] = min(item.get("similarity", 0) * factor, 1.0)
                item["similarity_percent"] = min(round(item.get("similarity_percent", 0) * factor, 2), 99.99)

    # 按调整后的 similarity 重新排序
    return sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)


def _apply_query_intent_boost(results: List[dict], query: str) -> List[dict]:
    if not results or not query:
        return results

    query_lower = query.lower().strip()
    wants_intro = any(hint in query_lower for hint in (
        "motivation", "动机", "背景", "why", "为什么", "不足", "不够充分", "区别", "不同",
    ))
    wants_experiment = any(hint in query_lower for hint in (
        "dataset", "datasets", "数据集", "baseline", "baselines", "基线", "实验", "评估",
    ))
    wants_limitation = any(hint in query_lower for hint in (
        "limitation", "limitations", "future work", "局限", "限制", "未来", "改进",
    ))
    wants_cost = _is_numeric_table_cost_query(query)

    if not any((wants_intro, wants_experiment, wants_limitation, wants_cost)):
        return results

    boosted = False
    for item in results:
        chunk_text = item.get("chunk", "")
        if not chunk_text:
            continue

        sample = chunk_text[:1500].lower()
        importance = item.get("importance_weight", 1.0)
        factor = 1.0

        if wants_intro:
            if importance >= 1.3 or any(token in sample for token in ("abstract", "introduction", "motivation", "background", "摘要", "引言", "背景")):
                factor *= 1.25
            elif _is_reference_like_text(chunk_text):
                factor *= 0.6

        if wants_experiment:
            if importance >= 1.3 or any(token in sample for token in ("dataset", "datasets", "baseline", "baselines", "experiment", "evaluation", "数据集", "基线", "实验", "评估")):
                factor *= 1.2
            if _is_reference_like_text(chunk_text):
                factor *= 0.35

        if wants_limitation:
            if any(token in sample for token in ("limitation", "limitations", "future work", "discussion", "conclusion", "局限", "限制", "未来", "改进")):
                factor *= 1.2
            elif importance < 1.0:
                factor *= 0.8

        if wants_cost:
            has_cost_anchor = _has_numeric_table_cost_anchor(sample) or _has_numeric_table_cost_anchor(chunk_text)
            is_table_like = _looks_like_numeric_table_support(chunk_text, item.get("chunk_type", "") or item.get("block_type", ""))
            if has_cost_anchor:
                factor *= 1.35 if importance >= 1.0 else 1.25
            elif is_table_like:
                factor *= 0.62
            elif any(token in sample for token in ("discussion", "limitation", "limitations", "future work", "conclusion")):
                factor *= 1.15

        if factor != 1.0:
            item["similarity"] = min(item.get("similarity", 0) * factor, 1.0)
            item["similarity_percent"] = min(round(item.get("similarity_percent", 0) * factor, 2), 99.99)
            item["query_intent_boost"] = round(factor, 3)
            boosted = True

    if boosted:
        return sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)
    return results


def _looks_like_numeric_table_support(chunk_text: str, chunk_type: str = "") -> bool:
    if not chunk_text:
        return False
    sample = chunk_text[:1600]
    sample_lower = sample.lower()
    if chunk_type in {"table", "table_row"} or _is_likely_table(sample):
        return True
    column_hit = bool(
        re.search(
            r'\b(all|overall|many|medium|med\.?|few|fid|acc(?:uracy)?|auc|f1|map|score|'
            r'precision|recall|bleu|rouge|em)\b',
            sample_lower,
        )
    )
    number_hit = len(re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[%±x×])?', sample_lower)) >= 4
    row_anchor_hit = bool(
        re.search(
            r'(?:^|\n|\|)\s*[A-Za-z][A-Za-z0-9.+/_+() -]{1,48}\s*(?:\||\s{2,})?\s*'
            r'[-+]?\d+(?:\.\d+)?',
            sample,
        )
    )
    header_hit = bool(re.search(r'\b(method|model|approach|group|dataset|metric|score)\b', sample_lower))
    separator_hit = sample.count("|") >= 4 or len(re.findall(r"\s{2,}", sample)) >= 4
    return column_hit and number_hit and (row_anchor_hit or header_hit or separator_hit)


def _build_numeric_table_evidence_text(item: dict) -> str:
    parts: List[str] = []
    seen: set[str] = set()
    for value in (
        item.get("table_id", ""),
        item.get("table_caption", ""),
        item.get("table_header", ""),
        item.get("chunk", ""),
        item.get("raw_chunk_text", ""),
    ):
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(normalized)
    return " ".join(parts).strip()


def _extract_numeric_table_dataset_mentions(text: str) -> set[str]:
    mentions: set[str] = set()
    sample = re.sub(r"\s+", " ", text or "").strip()
    if not sample:
        return mentions
    token_pattern = re.compile(
        r'\b(?:[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+|[A-Za-z]*[A-Z][A-Za-z0-9.+/_-]*)(?:19|20)?\d{0,2}\b'
    )
    for match in token_pattern.finditer(sample):
        token = match.group(0).strip(" ,.;:[]{}")
        if not token:
            continue
        if (
            re.search(r"(?:^|[-_])(?:LT|Dataset|Data)$", token, re.IGNORECASE)
            or re.search(r"(?:19|20)\d{2}$", token)
            or re.search(r"(?:^|[-_])(?:INat|Nat|Bench|Corpus|Set)(?:[-_]|$)", token, re.IGNORECASE)
        ):
            mentions.add(re.sub(r"\s+", "-", token).lower())
    return mentions


def _apply_numeric_table_boost(
    results: List[dict],
    query: str,
    evidence_need: Optional[List[str]] = None,
) -> List[dict]:
    if not results or not query:
        return results
    if "numeric_table" not in _resolve_evidence_need(query, evidence_need):
        return results

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    target_tables = {
        value.lower()
        for value in hints.get("table_labels", [])
        if value
    } | {
        table_id.lower()
        for _, _, table_id in _extract_table_mentions(query)
        if table_id
    }
    require_explicit_table_anchor = _should_require_explicit_table_anchor(hints)
    cost_query = _is_numeric_table_cost_query(query)
    boosted = False
    adjusted: List[dict] = []

    for item in results:
        chunk_text = (item.get("chunk") or item.get("raw_chunk_text") or "").strip()
        evidence_text = _build_numeric_table_evidence_text(item) or chunk_text
        if not evidence_text:
            adjusted.append(item)
            continue

        chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
        chunk_lower = evidence_text.lower()
        factor = 1.0
        anchor_hits: list[str] = []
        is_table_like = _looks_like_numeric_table_support(evidence_text, chunk_type)
        group_hit_counts: dict[str, int] = {}
        present_table_labels = set(re.findall(r'\btable\s*\d+\b', chunk_lower))
        target_table_labels = target_tables
        table_match = (
            _has_strict_numeric_table_anchor(item, target_table_labels, evidence_text)
            if require_explicit_table_anchor and target_table_labels
            else (
                not target_table_labels
                or bool(any(value in chunk_lower for value in target_table_labels))
            )
        )

        if is_table_like:
            factor *= 1.12

        for group_name, weight in (
            ("table_labels", 0.06),
            ("datasets", 0.08),
            ("backbones", 0.08),
            ("methods", 0.10),
            ("columns", 0.08),
            ("comparison", 0.05),
        ):
            group_hits = [
                value
                for value in hints.get(group_name, [])
                if value and value.lower() in chunk_lower
            ]
            if not group_hits:
                continue
            group_hit_counts[group_name] = len(group_hits)
            factor *= 1.0 + min(len(group_hits), 3) * weight
            anchor_hits.extend(group_hits[:3])

        column_hits = group_hit_counts.get("columns", 0)
        method_hits = group_hit_counts.get("methods", 0)
        if hints.get("columns") and column_hits == 0:
            factor *= 0.88 if is_table_like else 0.58
        elif len(hints.get("columns", [])) >= 3 and column_hits < 2:
            factor *= 0.72 if is_table_like else 0.62
        if hints.get("methods") and method_hits == 0:
            factor *= 0.55 if column_hits >= 2 else 0.40
        if not is_table_like and column_hits < 2:
            factor *= 0.72
        if target_table_labels and not table_match:
            if require_explicit_table_anchor:
                factor *= 0.18 if is_table_like else 0.10
            else:
                factor *= 0.82 if is_table_like else 0.68
        if target_table_labels and present_table_labels and not (target_table_labels & present_table_labels):
            factor *= 0.35 if column_hits >= 2 else 0.20
        if column_hits >= 3 and group_hit_counts.get("methods", 0) >= 1:
            factor *= 2.00

        if cost_query:
            has_cost_anchor = _has_numeric_table_cost_anchor(evidence_text)
            if has_cost_anchor:
                factor *= 1.30
            elif is_table_like:
                factor *= 0.72
            elif chunk_type not in {"table_row", "table", "caption"} and any(
                token in chunk_lower for token in ("discussion", "limitation", "future work", "conclusion")
            ):
                factor *= 1.15

        mixed_tables = present_table_labels
        if len(mixed_tables) >= 2:
            factor *= 0.78
        if item.get("table_augmented_scope") == "page_content":
            factor *= 0.86
            if len(mixed_tables) >= 2:
                factor *= 0.32
            elif target_table_labels and present_table_labels and not (target_table_labels & present_table_labels):
                factor *= 0.22

        if chunk_type == "caption" and not anchor_hits:
            factor *= 0.85

        if factor != 1.0:
            boosted = True
            updated = dict(item)
            updated["similarity"] = float(item.get("similarity", 0.0) or 0.0) * factor
            updated["similarity_percent"] = min(
                round(float(item.get("similarity_percent", 0.0) or 0.0) * factor, 2),
                99.99,
            )
            updated["numeric_table_boost"] = round(factor, 4)
            updated["numeric_table_anchor_hits"] = list(dict.fromkeys(anchor_hits))
            if len(mixed_tables) >= 2:
                updated["numeric_table_mixed_table"] = True
            adjusted.append(updated)
        else:
            adjusted.append(item)

    if boosted:
        adjusted.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return adjusted
    return results


def _normalize_numeric_exact_term(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _augment_with_numeric_exact_row_search(
    results: List[dict],
    chunks: List[str],
    chunk_pages: List[int],
    chunk_types: List[str],
    chunk_metadata: List[dict],
    pages: Optional[List[dict]],
    page_index,
    query: str,
    evidence_need: Optional[List[str]] = None,
    max_rows: int = 8,
) -> List[dict]:
    """Add number-aware exact row candidates for numeric table questions.

    Vector/BM25 retrieval can miss a short exact row when a table caption or
    surrounding paragraph dominates the query.  This local lane scans typed
    table-row chunks only, scores them with table/method/column/number anchors,
    and appends a small set of high-confidence rows to the candidate pool.
    """
    if not query or "numeric_table" not in (evidence_need or _analyze_evidence_need(query) or []):
        return results
    if not chunks or not isinstance(chunk_metadata, list):
        return results

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    target_tables = {
        _normalize_numeric_exact_term(value)
        for value in hints.get("table_labels", [])
        if value
    } | {
        _normalize_numeric_exact_term(table_id)
        for _, _, table_id in _extract_table_mentions(query)
        if table_id
    }
    grouped_terms: dict[str, list[str]] = {
        "methods": [
            _normalize_numeric_exact_term(value)
            for value in hints.get("methods", [])
            if _normalize_numeric_exact_term(value) and not _is_numeric_table_column_noise(str(value))
        ],
        "columns": [
            _normalize_numeric_exact_term(_normalize_numeric_column_name(value))
            for value in hints.get("columns", [])
            if _normalize_numeric_exact_term(value)
        ],
        "datasets": [
            _normalize_numeric_exact_term(value)
            for value in hints.get("datasets", [])
            if _normalize_numeric_exact_term(value)
        ],
        "backbones": [
            _normalize_numeric_exact_term(value)
            for value in hints.get("backbones", [])
            if _normalize_numeric_exact_term(value)
        ],
        "comparison": [
            _normalize_numeric_exact_term(value)
            for value in hints.get("comparison", [])
            if _normalize_numeric_exact_term(value)
        ],
    }
    query_numbers = {
        _normalize_numeric_exact_term(value)
        for value in re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", query)
        if value
    }
    query_lower = str(query or "").lower()
    winner_style_regex_query = bool(
        grouped_terms["columns"]
        and re.search(
            r"(?:最高|最低|最大|最小|最好|最佳|最优|highest|lowest|largest|smallest|best|maximum|minimum)",
            query_lower,
            re.IGNORECASE,
        )
    )
    has_non_table_anchor = bool(
        grouped_terms["methods"]
        or grouped_terms["datasets"]
        or grouped_terms["backbones"]
        or query_numbers
        or winner_style_regex_query
    )
    if not target_tables and not has_non_table_anchor:
        return results

    existing_chunk_ids: set[int] = set()
    existing_texts: set[str] = set()
    for item in results or []:
        if not isinstance(item, dict):
            continue
        try:
            existing_chunk_ids.add(int(item.get("chunk_id")))
        except (TypeError, ValueError):
            pass
        text = str(item.get("chunk") or item.get("raw_chunk_text") or "")
        if text:
            existing_texts.add(text)

    def _term_hit(term: str, text: str) -> bool:
        if not term:
            return False
        if term in text:
            return True
        compact_term = re.sub(r"[\s_\-]+", "", term)
        compact_text = re.sub(r"[\s_\-]+", "", text)
        return bool(compact_term and compact_term in compact_text)

    def _score_candidate(chunk_text: str, metadata: dict, chunk_type: str) -> tuple[float, list[str]]:
        row_text = str(
            metadata.get("numeric_table_exact_context_row_text")
            or metadata.get("row_text")
            or metadata.get("table_row_boundary_text")
            or chunk_text
            or ""
        )
        ref_text = " ".join(
            str(metadata.get(key) or "")
            for key in (
                "table_id",
                "table_caption",
                "numeric_table_exact_context_caption",
                "table_header",
                "numeric_table_exact_context_header",
                "row_id",
            )
        )
        combined = _normalize_numeric_exact_term(f"{ref_text}\n{row_text}\n{chunk_text}")
        ref_norm = _normalize_numeric_exact_term(ref_text)

        table_hit = False
        if target_tables:
            table_hit = any(_term_hit(table, ref_norm) or _term_hit(table, combined) for table in target_tables)
            if not table_hit:
                return 0.0, []

        score = 2.0
        if chunk_type == "table_row":
            score += 1.0
        if metadata.get("table_row_slice_kind") == "exact" or metadata.get("numeric_table_exact_context_row_text"):
            score += 2.0
        if table_hit:
            score += 2.5

        hits: list[str] = []
        weighted_groups = (
            ("methods", 1.8),
            ("datasets", 1.0),
            ("backbones", 1.0),
            ("columns", 1.1),
            ("comparison", 0.8),
        )
        method_hits = 0
        column_hits = 0
        for group_name, weight in weighted_groups:
            seen_in_group = 0
            for term in grouped_terms.get(group_name, []):
                if not term:
                    continue
                if _term_hit(term, combined):
                    seen_in_group += 1
                    hits.append(term)
            if seen_in_group:
                score += min(seen_in_group, 4) * weight
                if group_name == "methods":
                    method_hits = seen_in_group
                if group_name == "columns":
                    column_hits = seen_in_group

        number_hits = 0
        for number in query_numbers:
            if _term_hit(number, combined):
                number_hits += 1
                hits.append(number)
        if number_hits:
            score += min(number_hits, 3) * 1.0

        score += min(_compute_lexical_evidence_score(query, f"{ref_text}\n{row_text}") * 2.0, 2.0)

        if not target_tables:
            if method_hits <= 0 and number_hits <= 0 and not (winner_style_regex_query and column_hits > 0):
                return 0.0, []
            if grouped_terms["columns"] and column_hits <= 0:
                score -= 1.0
        if not re.search(r"\d", row_text):
            score -= 1.0
        elif winner_style_regex_query and column_hits > 0:
            score += 1.5
            hits.append("regex:column_numeric_row")
        if metadata.get("table_row_shard") and metadata.get("table_row_slice_kind") != "exact":
            score -= 1.2

        return score, list(dict.fromkeys(hits))

    scored: list[tuple[float, int, dict]] = []
    for idx, chunk_text in enumerate(chunks):
        if idx in existing_chunk_ids or chunk_text in existing_texts:
            continue
        metadata = chunk_metadata[idx] if idx < len(chunk_metadata) and isinstance(chunk_metadata[idx], dict) else {}
        chunk_type = (chunk_types[idx] if idx < len(chunk_types) else metadata.get("chunk_type") or "").strip().lower()
        if chunk_type != "table_row" and metadata.get("table_row_slice_kind") != "exact":
            continue
        score, hits = _score_candidate(str(chunk_text), metadata, chunk_type)
        if score < (5.8 if target_tables else 5.0):
            continue

        page_num = chunk_pages[idx] if idx < len(chunk_pages) else 0
        page_num = _resolve_primary_page_from_metadata(metadata, fallback=page_num)
        if (not isinstance(page_num, int) or page_num <= 0) and pages:
            page_num = _find_page_for_chunk(str(chunk_text), pages, page_index=page_index)
        snippet, highlights = _extract_snippet_and_highlights(str(chunk_text), query)
        candidate = {
            "chunk": chunk_text,
            "raw_chunk_text": chunk_text,
            "page": page_num,
            "score": score,
            "similarity": min(0.86 + min(score * 0.01, 0.10), 0.97),
            "similarity_percent": round(min(0.86 + min(score * 0.01, 0.10), 0.97) * 100, 2),
            "snippet": snippet,
            "highlights": highlights,
            "reranked": False,
            "chunk_id": idx,
            "chunk_heading": metadata.get("table_caption") or metadata.get("table_id") or "",
            "section_path": metadata.get("table_caption") or metadata.get("table_id") or "",
            "chunk_type": "table_row",
            "block_type": "table_row",
            "numeric_exact_search": True,
            "retrieval_source": "numeric_exact_row",
            "numeric_table_priority": score,
            "numeric_table_anchor_hits": hits,
            "numeric_regex_locator": any(str(hit).startswith("regex:") for hit in hits),
            "numeric_regex_locator_hits": [hit for hit in hits if str(hit).startswith("regex:")],
        }
        _apply_chunk_metadata(candidate, metadata)
        _apply_page_provenance(candidate, metadata)
        scored.append((score, idx, candidate))

    if not scored:
        return results

    scored.sort(key=lambda row: (-row[0], row[1]))
    merged = list(results or [])
    for _score, _idx, candidate in scored[: max(1, int(max_rows or 1))]:
        chunk_text = str(candidate.get("chunk") or "")
        if any(str(item.get("chunk") or "") == chunk_text for item in merged if isinstance(item, dict)):
            continue
        merged.append(candidate)
    logger.info("[NumericExactSearch] 补充 %s 个 exact table row 候选", len(merged) - len(results or []))
    return merged


def _prioritize_numeric_table_results(results: List[dict], query: str) -> List[dict]:
    """在最终结果排序中前置 numeric_table 的强表格证据。"""
    if not results or not query:
        return results
    if "numeric_table" not in (_analyze_evidence_need(query) or []):
        return results

    hints = _query_rewriter_singleton.extract_numeric_table_hints(query)
    target_method_keys = _extract_numeric_table_row_method_targets(hints)
    comparison_query = bool(hints.get("comparison"))
    bundle_query = _is_numeric_table_bundle_query(query, hints)
    explicit_comparison_methods = comparison_query and len(target_method_keys) >= 2
    cost_query = _is_numeric_table_cost_query(query)
    has_table_row_result = any(
        (item.get("chunk_type") or item.get("block_type") or "").strip().lower() == "table_row"
        for item in results
    )
    target_datasets = _extract_numeric_table_dataset_mentions(" ".join(hints.get("datasets", [])))
    target_backbones = {value.lower() for value in hints.get("backbones", []) if value}
    target_columns = {
        _normalize_numeric_column_name(value).lower()
        for value in hints.get("columns", [])
        if value
    }
    target_tables = {
        value.lower()
        for value in hints.get("table_labels", [])
        if value
    } | {
        table_id.lower()
        for _, _, table_id in _extract_table_mentions(query)
        if table_id
    }
    require_explicit_table_anchor = _should_require_explicit_table_anchor(hints)
    preferred_sort_column = _preferred_numeric_table_sort_column(query, hints)
    winner_style_query = bool(preferred_sort_column)
    row_band_query = _is_numeric_table_row_band_query(query, hints)
    ranked: List[tuple[float, int, dict]] = []
    strong_support_count = 0

    for original_rank, item in enumerate(results):
        updated = dict(item)
        chunk_text = (updated.get("chunk") or updated.get("raw_chunk_text") or "").strip()
        raw_chunk_text = (updated.get("raw_chunk_text") or "").strip()
        table_caption = (updated.get("table_caption") or updated.get("table_id") or "").strip()
        table_header = (updated.get("table_header") or "").strip()
        evidence_text = "\n".join(
            part
            for part in (table_caption, table_header, chunk_text, raw_chunk_text)
            if part
        ).strip()
        evidence_lower = evidence_text.lower()
        chunk_type = (updated.get("chunk_type") or updated.get("block_type") or "").strip().lower()
        headerish_anchor = chunk_type == "table_row" and _is_headerish_numeric_table_row(updated)
        is_table_like = _looks_like_numeric_table_support(evidence_text or chunk_text, chunk_type)
        number_hits = len(re.findall(r'\d+\.?\d*', evidence_lower))
        present_table_labels = set(re.findall(r'\btable\s*\d+\b', evidence_lower))
        target_table_labels = target_tables
        method_hits = [
            value for value in hints.get("methods", [])
            if value and re.sub(r"\s+", "", value.lower()) in re.sub(r"\s+", "", evidence_lower)
        ]
        backbone_hits = {
            value.lower()
            for value in hints.get("backbones", [])
            if value and value.lower() in evidence_lower
        }
        column_hits = {
            _normalize_numeric_column_name(value).lower()
            for value in hints.get("columns", [])
            if value and value.lower() in evidence_lower
        }
        row_key = _normalize_numeric_table_method_token(updated.get("row_id", ""))
        row_method_exact = bool(row_key and row_key in target_method_keys)
        explicit_table_match = _has_explicit_numeric_table_match(updated, target_table_labels)
        strict_table_match = _has_strict_numeric_table_anchor(updated, target_table_labels, evidence_text)
        table_match = not target_table_labels or (
            strict_table_match
            if require_explicit_table_anchor
            else (
                explicit_table_match
                or (
                    chunk_type != "table_row"
                    and bool(any(value in evidence_lower for value in target_table_labels))
                )
            )
        )
        dataset_mentions = _extract_numeric_table_dataset_mentions(evidence_text)
        dataset_match = not target_datasets or bool(dataset_mentions & target_datasets)

        resolved_column_hits: set[str] = set()
        resolved_backbone_hits: set[str] = set()
        if chunk_type == "table_row" and (target_columns or target_backbones or preferred_sort_column):
            row_text = re.sub(
                r"\s+",
                " ",
                str(
                    updated.get("numeric_table_exact_context_row_text")
                    or _get_numeric_table_boundary_text(updated)
                    or updated.get("table_row_raw_text")
                    or updated.get("row_text")
                    or updated.get("chunk")
                    or updated.get("raw_chunk_text")
                    or ""
                ),
            ).strip()
            if row_text:
                focused_row = _build_query_focused_table_row(
                    {
                        "row_id": updated.get("row_id") or "",
                        "row_text": row_text,
                        "row_numbers": _strip_leading_numeric_table_row_id(row_text, updated.get("row_id") or "") or row_text,
                        "table_caption": updated.get("numeric_table_exact_context_caption")
                        or updated.get("table_caption")
                        or updated.get("table_id")
                        or "",
                        "table_id": updated.get("table_id") or "",
                        "table_header": updated.get("numeric_table_exact_context_header")
                        or updated.get("table_header")
                        or "",
                        "table_focus_columns": list(updated.get("table_focus_columns") or []),
                    },
                    hints,
                )
                resolved_column_hits = {
                    _normalize_numeric_column_name(value).lower()
                    for value in (focused_row.get("resolved_columns") or [])
                    if value
                }
                matched_backbone = str(focused_row.get("matched_backbone") or "").strip().lower()
                if matched_backbone:
                    resolved_backbone_hits.add(matched_backbone)
        if updated.get("table_focus_backbone"):
            resolved_backbone_hits.add(str(updated.get("table_focus_backbone") or "").strip().lower())
        resolved_backbone_hits.update(
            value.strip().lower()
            for value in _extract_table_header_backbones(updated.get("table_header", "") or evidence_text)
            if value
        )

        effective_column_hits = column_hits | resolved_column_hits
        effective_backbone_hits = backbone_hits | resolved_backbone_hits
        backbone_match = not target_backbones or bool(effective_backbone_hits & target_backbones)
        column_match = not target_columns or bool(effective_column_hits & target_columns)
        same_bundle_match = table_match and dataset_match and backbone_match and column_match
        bundle_sensitive_query = bool(
            target_table_labels or target_columns or target_backbones or target_datasets or bundle_query
        )
        bundle_row_match = chunk_type == "table_row" and (
            row_method_exact
            or (target_table_labels and table_match)
            or (target_columns and len(effective_column_hits & target_columns) >= 1 and same_bundle_match)
        )
        composite_target_noise = (
            bundle_query
            and bool(target_method_keys)
            and not explicit_comparison_methods
            and not row_method_exact
            and _is_composite_numeric_row_id(updated.get("row_id", ""))
            and _row_mentions_target_method(updated.get("row_id", ""), target_method_keys)
        )

        priority = 0.0
        anchor_hits: list[str] = []

        if is_table_like:
            priority += 4.0
        if chunk_type == "table_row":
            priority += 2.2
            if row_method_exact:
                priority += 1.4
            if updated.get("table_row_slice_kind") == "exact":
                priority += 0.8
        elif chunk_type == "table":
            priority += 1.6
        elif chunk_type == "caption":
            priority += 0.8
        if row_band_query and not cost_query:
            if chunk_type == "table_row":
                priority += 3.0
                if updated.get("table_row_slice_kind") == "exact":
                    priority += 0.6
            elif chunk_type in {"table", "caption"}:
                priority += 1.0
            else:
                priority -= 4.0
        if target_table_labels:
            if table_match:
                priority += 3.5
            else:
                priority -= 8.5 if require_explicit_table_anchor else 6.0
        if number_hits >= 4:
            priority += min(number_hits, 12) * 0.12

        for group_name, weight in (
            ("table_labels", 1.0),
            ("datasets", 1.3),
            ("backbones", 1.2),
            ("methods", 1.5),
            ("columns", 1.2),
            ("comparison", 0.7),
        ):
            group_hits = [
                value
                for value in hints.get(group_name, [])
                if value and value.lower() in evidence_lower
            ]
            if not group_hits:
                continue
            priority += min(len(group_hits), 3) * weight
            anchor_hits.extend(group_hits[:3])

        if len(hints.get("columns", [])) >= 3 and len(effective_column_hits & target_columns) >= 3:
            priority += 1.5
        if hints.get("methods") and method_hits:
            priority += 0.9
        if chunk_type == "table_row" and method_hits:
            priority += 1.1
        if chunk_type == "table_row" and effective_backbone_hits & target_backbones:
            priority += 0.9
        if chunk_type == "table_row" and target_columns and same_bundle_match and len(effective_column_hits & target_columns) >= 1:
            priority += 0.9
        if chunk_type == "table_row" and len(effective_column_hits & target_columns) >= 2:
            priority += 1.0
        if preferred_sort_column and chunk_type == "table_row":
            preferred_sort_key = _normalize_numeric_column_name(preferred_sort_column).lower()
            if preferred_sort_key in effective_column_hits:
                priority += 1.6
            elif winner_style_query and not cost_query:
                priority -= 5.8
        if chunk_type == "table_row" and target_table_labels and present_table_labels & target_table_labels:
            priority += 0.7
        if target_table_labels and present_table_labels and not (target_table_labels & present_table_labels):
            priority -= 3.0
        if headerish_anchor:
            priority -= 3.5
        if chunk_type == "table_row" and target_table_labels and not (
            strict_table_match if require_explicit_table_anchor else explicit_table_match
        ):
            priority -= 3.2
        if explicit_comparison_methods and chunk_type == "table_row" and not row_method_exact:
            priority -= 4.5
        if composite_target_noise:
            priority -= 4.8
        if has_table_row_result and bundle_sensitive_query and chunk_type not in {"table_row", "table", "caption"}:
            if method_hits or effective_backbone_hits or len(effective_column_hits) >= 1 or present_table_labels:
                priority -= 3.4 if bundle_query else 2.4
        if has_table_row_result and updated.get("table_augmented_scope") == "page_content":
            priority -= 2.8 if bundle_query else 1.6
            if len(present_table_labels) >= 2:
                priority -= 1.8
            elif target_table_labels and present_table_labels and not (target_table_labels & present_table_labels):
                priority -= 2.2
        if chunk_type == "table_row" and target_table_labels and not table_match:
            priority -= 4.6 if require_explicit_table_anchor else 2.8
        if chunk_type == "table_row" and not same_bundle_match and target_table_labels:
            priority -= 1.5 if require_explicit_table_anchor else 1.0
        if chunk_type == "table_row" and not bundle_row_match and comparison_query and len(target_method_keys) <= 1:
            priority -= 0.5
        if (winner_style_query or explicit_comparison_methods or comparison_query) and not cost_query:
            if chunk_type == "table_row":
                if target_columns and not column_match:
                    priority -= 6.0
                if target_backbones and not backbone_match:
                    priority -= 4.5
                if target_datasets and not dataset_match:
                    priority -= 4.5
            elif has_table_row_result and chunk_type in {"table", "caption"}:
                if target_columns and not column_match:
                    priority -= 4.8
                if target_backbones and not backbone_match:
                    priority -= 3.6
                if target_datasets and not dataset_match:
                    priority -= 3.6
        priority += _numeric_table_sort_bonus(updated, query, hints)
        if preferred_sort_column and has_table_row_result and chunk_type not in {"table_row", "table", "caption"} and not cost_query:
            priority -= 3.2
        if cost_query:
            has_cost_anchor = _has_numeric_table_cost_anchor(evidence_text)
            if has_cost_anchor:
                priority += 6.0 if chunk_type not in {"table_row", "table", "caption"} else 3.5
            elif is_table_like:
                priority -= 8.0
            elif chunk_type not in {"table_row", "table", "caption"} and any(
                token in evidence_lower for token in ("discussion", "limitation", "future work", "conclusion")
            ):
                priority += 2.0
        if not is_table_like and not anchor_hits:
            priority -= 1.4
        if chunk_type not in {"table_row", "table", "caption"} and method_hits and effective_backbone_hits and len(effective_column_hits) >= 2:
            priority -= 1.2
        if len(chunk_text) < 120 and number_hits < 2:
            priority -= 0.8

        if priority >= 4.0:
            strong_support_count += 1

        updated["numeric_table_priority"] = round(priority, 4)
        updated["numeric_table_anchor_hits"] = list(dict.fromkeys(anchor_hits))
        ranked.append((priority, original_rank, updated))

    if strong_support_count == 0 and not has_table_row_result:
        return results

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item for _, _, item in ranked]


def _filter_reference_pollution(
    results: List[dict],
    query: str,
    evidence_need: Optional[List[str]] = None,
) -> List[dict]:
    return filter_reference_trap_results(results, query, evidence_need=evidence_need)


def _csv_to_set(raw_value: str) -> set[str]:
    return {
        item.strip()
        for item in (raw_value or "").split(",")
        if item and item.strip()
    }


def _should_enable_hyde(query_type: str, evidence_need: List[str], config) -> bool:
    if not config.enable_hyde:
        return False
    blocked = _csv_to_set(getattr(config, "hyde_evidence_blocklist", ""))
    allowed_types = _csv_to_set(getattr(config, "hyde_query_types", ""))
    allowed_evidence = _csv_to_set(getattr(config, "hyde_evidence_allowlist", ""))
    evidence_set = set(evidence_need or [])
    if evidence_set & blocked:
        return False
    return query_type in allowed_types or bool(evidence_set & allowed_evidence)


def _should_enable_query_expansion(query_type: str, evidence_need: List[str], config) -> bool:
    if not config.enable_query_expansion:
        return False
    blocked = _csv_to_set(getattr(config, "query_expansion_evidence_blocklist", ""))
    allowed_types = _csv_to_set(getattr(config, "query_expansion_query_types", ""))
    allowed_evidence = _csv_to_set(getattr(config, "query_expansion_evidence_allowlist", ""))
    evidence_set = set(evidence_need or [])
    if evidence_set & blocked:
        return False
    return query_type in allowed_types or bool(evidence_set & allowed_evidence)


def _select_multi_query_merge_mode(query_type: str, evidence_need: List[str], config) -> str:
    """P3.3c 选择多查询合并策略。

    - numeric_table / extraction 类查询走 intersection（高 precision）
    - 其他查询走 rrf（鲁棒性强）
    """
    intersection_types = _csv_to_set(getattr(config, "query_expansion_intersection_types", ""))
    evidence_set = set(evidence_need or [])
    if query_type in intersection_types or bool(evidence_set & intersection_types):
        return "intersection"
    default_mode = (getattr(config, "query_expansion_merge_mode", "rrf") or "rrf").strip().lower()
    if default_mode not in {"rrf", "intersection", "weighted_avg", "union"}:
        return "rrf"
    return default_mode


def _should_force_conditional_rerank(
    query_type: str,
    evidence_need: List[str],
    reranker_model: Optional[str],
    config,
) -> bool:
    if not config.enable_conditional_rerank:
        return False
    cond_types = _csv_to_set(config.conditional_rerank_types or "")
    cond_evidence = _csv_to_set(getattr(config, "conditional_rerank_evidence_needs", ""))
    evidence_set = set(evidence_need or [])
    return query_type in cond_types or bool(evidence_set & cond_evidence)


_STRUCTURAL_SECTION_BLACKLIST = (
    "references",
    "bibliography",
    "appendix",
    "broader impacts",
    "broader impact",
    "safeguards",
    "acknowledgments",
    "acknowledgements",
    "funding",
    "declarations",
    "conflicts of interest",
    "author contributions",
    "supplementary",
    "附录",
    "参考文献",
    "致谢",
    "资金",
    "作者贡献",
)


_EXPLICIT_TABLE_STRUCTURE_QUERY_RE = re.compile(
    r"(?:表\s*\d+|表格|表中|表里|该表|这张表|table\s*\d*|table\s+of|columns?|rows?|headers?)",
    re.IGNORECASE,
)


def _is_explicit_table_structure_query(query: str) -> bool:
    """Keep table rows for structural questions even when no metric is named."""
    return bool(_EXPLICIT_TABLE_STRUCTURE_QUERY_RE.search(str(query or "")))


def _is_structural_noise(text: str) -> bool:
    """检测 chunk 是否为结构性噪声章节（附录/参考/声明/致谢等）。
    仅看前 600 字符的第一行，避免误伤正文中提到这些词的段落。
    """
    if not text:
        return False
    sample = text[:600]
    first_line = sample.split("\n")[0].strip().lower()
    for word in _STRUCTURAL_SECTION_BLACKLIST:
        if first_line.startswith(word) or (len(first_line) < 40 and word in first_line):
            return True
    return False


def _unified_post_clean(
    results: List[dict],
    query: str,
    top_k: int,
    evidence_need: Optional[List[str]] = None,
) -> List[dict]:
    """统一末端清洗器（参考 openclaw candidateMultiplier + minScore 策略）

    在检索管线最末端执行，保证最终上下文干净：
    1. 结构性噪声降权（附录/声明/致谢等章节标题开头的 chunk）
    2. 碎片惩罚（过短 chunk）
    3. 最小相似度过滤（但至少保留 min_keep 条）
    4. 保序截断到 top_k
    """
    _rc = _rag_config_singleton
    if not results or not _rc.enable_post_clean:
        return results[:top_k]

    min_score = _rc.post_clean_min_score
    min_keep = min(max(_rc.post_clean_min_keep, 1), max(top_k, 1))
    cost_query = _is_numeric_table_cost_query(query)
    numeric_table_query = "numeric_table" in _resolve_evidence_need(query, evidence_need)
    table_query = numeric_table_query or _is_explicit_table_structure_query(query)

    scored = []
    for item in results:
        chunk_text = item.get("chunk", "")
        chunk_type = str(item.get("chunk_type") or item.get("block_type") or "").strip().lower()
        sim = item.get("similarity", 0.0)
        penalty = 1.0
        keep_cost_anchor = bool(cost_query and chunk_text and _has_numeric_table_cost_anchor(chunk_text))

        if chunk_text and _is_structural_noise(chunk_text) and not keep_cost_anchor:
            penalty *= 0.15
        if chunk_text and len(chunk_text.strip()) < 80 and not keep_cost_anchor:
            penalty *= 0.5
        if not table_query and not keep_cost_anchor:
            if chunk_type in {"table_row", "table_cell"}:
                penalty *= 0.15
            elif chunk_type == "table" or (chunk_text and _is_table_fragment(chunk_text)):
                penalty *= 0.30

        scored.append((item, sim * penalty if penalty < 1.0 else sim, keep_cost_anchor))

    scored.sort(key=lambda x: (x[2], x[1]), reverse=True)

    kept = []
    for i, (item, adj_score, keep_cost_anchor) in enumerate(scored):
        if keep_cost_anchor or adj_score >= min_score or i < min_keep:
            kept.append(item)

    removed = len(results) - len(kept)
    if removed > 0:
        logger.info(f"[统一清洗] 过滤低质量/结构噪声结果 {removed} 条 (min_score={min_score})")

    return kept[:top_k]



# ── P0-C: 候选前置过滤 —————————————————————————————————————————————————————
# formula/caption/title chunk 对非公式类问题贡献极低但极易被向量检索命中；
# 在 annotation（chunk_type 已填充）之后、rerank 之前做轻量得分折扣。
_FORMULA_QUERY_TOKENS = frozenset({
    "公式", "formula", "equation", "算法", "algorithm", "derive",
    "推导", "proof", "表达式", "expression",
})
_NUMERIC_QUERY_TOKENS = frozenset({
    "准确率", "accuracy", "数值", "score", "metric", "table", "表格",
    "性能", "performance", "结果", "result", "f1", "bleu", "rouge",
})


def _sanitize_by_chunk_type(
    results: List[dict],
    query: str,
    intent_decision: Optional[dict] = None,
) -> List[dict]:
    """对特定 chunk_type 做得分折扣（不完全排除，避免误伤）

    折扣策略（考虑题目意图）：
    - formula  : 非公式问题 ×0.25；公式/算法问题 ×1.0
    - caption  : 非数值问题 ×0.50；数值/表格问题 ×0.80
    - 极短 chunk (< 40 chars) 且非表格/数字：额外 ×0.40
    """
    if not results:
        return results

    query_lower = (query or "").lower()
    modalities = {
        str(item).strip().lower()
        for item in ((intent_decision or {}).get("modalities") or [])
        if str(item).strip()
    } if isinstance(intent_decision, dict) else set()
    frozen_evidence_need = (
        list((intent_decision or {}).get("evidence_need") or [])
        if isinstance(intent_decision, dict)
        else None
    )
    is_formula_query = "formula" in modalities or any(t in query_lower for t in _FORMULA_QUERY_TOKENS)
    is_numeric_query = any(t in query_lower for t in _NUMERIC_QUERY_TOKENS)
    is_numeric_table_query = "numeric_table" in _resolve_evidence_need(
        query,
        frozen_evidence_need,
    )
    is_table_query = (
        "table" in modalities
        or is_numeric_table_query
        or _is_explicit_table_structure_query(query)
    )
    is_visual_query = bool(
        modalities & {"figure", "layout", "image"}
    ) or bool(re.search(
        r"\b(?:figure|fig\.?|chart|diagram|image)\b|图\s*\d*|图表|插图|示意图",
        query or "",
        re.IGNORECASE,
    ))

    adjusted = []
    penalized = 0
    excluded = 0
    for item in results:
        chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
        sim = item.get("similarity", 0.0)
        penalty = 1.0
        chunk_text = (item.get("chunk") or "").strip()

        if not is_table_query:
            if chunk_type in {"table_row", "table_cell"}:
                excluded += 1
                continue
            if _is_table_fragment(chunk_text) and not is_visual_query:
                excluded += 1
                continue

        if chunk_type == "formula" and not is_formula_query:
            penalty *= 0.25
        elif chunk_type == "caption":
            if not is_table_query and not is_visual_query:
                penalty *= 0.50
            elif not is_table_query and not is_numeric_query:
                penalty *= 0.80
        elif chunk_type == "table" and not is_table_query:
            penalty *= 0.70 if is_visual_query else 0.35

        if len(chunk_text) < 40 and chunk_type not in ("table",) and not any(ch.isdigit() for ch in chunk_text):
            penalty *= 0.40

        if penalty < 1.0:
            penalized += 1
            item = dict(item)  # 浅拷贝，避免改动原对象
            item["similarity"] = sim * penalty
            item["_type_penalty"] = round(penalty, 4)

        adjusted.append(item)

    if penalized:
        logger.debug(f"[TypeSanitizer] 对 {penalized}/{len(results)} 个候选应用类型折扣")
    if excluded:
        logger.info(f"[TypeSanitizer] 非表格问题隔离表格行/数字碎片 {excluded} 条")

    adjusted.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
    return adjusted


def _compute_lexical_evidence_score(query: str, chunk_text: str) -> float:
    """计算 chunk 与 query 之间的词法证据得分（不需要模型，纯字符串匹配）

    评分维度：
    1. 查询词命中率：query 中 ≥2 字符的词在 chunk 中出现的比例
    2. 数字匹配奖励：query 含数字且 chunk 也含对应数字
    3. 长度惩罚：过短 chunk 得分减半

    返回 [0, 1] 的标量，0 = 无词法重叠，1 = 完全匹配
    """
    if not query or not chunk_text:
        return 0.0
    import re as _re
    query_terms = [
        t.lower() for t in _re.split(r'[\s,;，。；、？！?!：:""\'\'\"\"]+', query)
        if len(t) >= 2
    ]
    if not query_terms:
        return 0.0
    chunk_lower = chunk_text.lower()
    hit_count = sum(1 for t in query_terms if t in chunk_lower)
    base_score = hit_count / len(query_terms)
    if looks_formula_like(query) or looks_formula_like(chunk_text):
        formula_hits = sum(1 for term in query_terms if formula_term_matches(term, chunk_text))
        if formula_hits:
            base_score = min(1.0, base_score + min(0.35, formula_hits / max(len(query_terms), 1) * 0.5))

    # 数字匹配奖励
    query_digits = set(_re.findall(r'\d+\.?\d*', query))
    if query_digits:
        chunk_digits = set(_re.findall(r'\d+\.?\d*', chunk_text))
        if query_digits & chunk_digits:
            base_score = min(1.0, base_score + 0.2)

    # 过短 chunk 惩罚
    if len(chunk_text.strip()) < 60:
        base_score *= 0.5

    return min(1.0, base_score)


def _apply_evidence_gate(
    results: List[dict],
    query: str,
    relevance_weight: float = 0.7,
    evidence_weight: float = 0.3,
) -> List[dict]:
    """对 rerank 结果施加词法证据门控，融合 topic-relevance 与 evidence-usefulness

    final_score = relevance_weight * rerank_score + evidence_weight * lexical_evidence_score

    只对"有明确 rerank_score"的结果生效（即已开启 rerank 的情况）。
    """
    if not results:
        return results
    gated = []
    for item in results:
        rerank_score = item.get("rerank_score")
        if rerank_score is None:
            gated.append(item)
            continue
        chunk_text = (item.get("chunk") or "").strip()
        ev_score = _compute_lexical_evidence_score(query, chunk_text)
        relevance_score = float(item.get("similarity", rerank_score))
        type_penalty = float(item.get("_type_penalty", 1.0) or 1.0)
        combined = (relevance_weight * relevance_score + evidence_weight * ev_score) * type_penalty
        item = dict(item)
        item["evidence_score"] = round(ev_score, 4)
        item["applied_type_penalty"] = round(type_penalty, 4)
        item["combined_score"] = round(combined, 6)
        gated.append(item)
    gated.sort(key=lambda x: x.get("combined_score", x.get("rerank_score", 0.0)), reverse=True)
    return gated


def _finalize_with_optional_rerank(
    query: str,
    results: List[dict],
    top_k: int,
    use_rerank: bool,
    reranker_model: Optional[str],
    rerank_provider: Optional[str],
    rerank_api_key: Optional[str],
    rerank_endpoint: Optional[str],
    timings: dict,
    progress_callback=None,
    conditional_rerank_active: bool = False,
) -> List[dict]:
    """在所有候选扩展完成后执行最终 rerank，确保重排序是末端裁决。"""
    _fm_config = _rag_config_singleton
    effective_top_k = _resolve_numeric_table_effective_top_k(query, top_k, results=results)

    if not use_rerank:
        return _finalize_without_rerank(results, query, top_k, _fm_config)

    if (
        conditional_rerank_active
        and _should_bypass_conditional_rerank_for_numeric_table(results, query, effective_top_k)
    ):
        logger.info("[ConditionalRerank] numeric_table 已有稳定 table_row bundle，跳过最终 rerank")
        ordered = _apply_numeric_table_same_bundle_hard_gate(results, query)
        final = ordered[:effective_top_k]
        if not all(
            (item.get("chunk_type") or item.get("block_type") or "").strip().lower() == "table_row"
            for item in final
        ):
            final = _ensure_numeric_table_evidence_slots(ordered, query, effective_top_k)
        final = _apply_numeric_table_same_bundle_hard_gate(final, query)
        if _fm_config.enable_focus_mode:
            t1 = time.perf_counter()
            final = _focus_mode_compress(
                final,
                query,
                window_size=_fm_config.focus_mode_window_size,
                max_sentences=_fm_config.focus_mode_max_sentences,
                min_chars=_fm_config.focus_mode_min_chars,
            )
            timings["focus_mode_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        return final

    t0 = time.perf_counter()
    _emit_retrieval_progress(progress_callback, "rerank_start", "正在重排序候选片段...")
    reranked_results = _apply_rerank(
        query,
        results,
        reranker_model,
        rerank_provider,
        rerank_api_key,
        rerank_endpoint,
    )
    reranked_results = _apply_evidence_gate(reranked_results, query)
    effective_top_k = _resolve_numeric_table_effective_top_k(
        query,
        top_k,
        results=reranked_results,
    )
    reranked_results, post_cap_stats = _apply_group_post_cap(reranked_results, top_k=effective_top_k)
    if post_cap_stats:
        logger.info(f"[RerankPostCap] 输出 {post_cap_stats['output']} 条，group_blocked={post_cap_stats['group_blocked']}, section_blocked={post_cap_stats['section_blocked']}")
    min_score = float(getattr(_fm_config, "rerank_score_min", 0.0) or 0.0)
    min_keep = min(
        max(int(getattr(_fm_config, "rerank_score_min_keep", 1) or 1), 1),
        max(effective_top_k, 1),
    )
    if min_score > 0:
        kept = []
        removed = 0
        for idx, item in enumerate(reranked_results):
            score = item.get("combined_score")
            if score is None:
                score = item.get("rerank_score")
            if score is None:
                score = item.get("similarity", 0.0)
            if float(score or 0.0) >= min_score or idx < min_keep:
                kept.append(item)
            else:
                removed += 1
        if removed > 0:
            logger.info(
                f"[RerankFloor] 过滤低分候选 {removed} 条 "
                f"(min_score={min_score}, min_keep={min_keep})"
            )
        reranked_results = kept
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _emit_retrieval_progress(progress_callback, "rerank_done", "重排序完成。")

    final = _ensure_numeric_table_evidence_slots(reranked_results, query, effective_top_k)
    final = _apply_numeric_table_same_bundle_hard_gate(final, query)
    if _fm_config.enable_focus_mode:
        t1 = time.perf_counter()
        final = _focus_mode_compress(
            final, query,
            window_size=_fm_config.focus_mode_window_size,
            max_sentences=_fm_config.focus_mode_max_sentences,
            min_chars=_fm_config.focus_mode_min_chars,
        )
        timings["focus_mode_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        compressed = sum(1 for item in final if "focus_compression_ratio" in item)
        if compressed:
            logger.info(f"[FocusMode] 压缩 {compressed}/{len(final)} 个候选")

    return final


def _normalize_runtime_visual_evidence(visual_evidence: Optional[List[dict]], limit: int = 8) -> List[dict]:
    """Normalize the small, committed VLM evidence overlay for one search."""
    normalized: List[dict] = []
    seen: set[str] = set()
    for item in visual_evidence or []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        text = " ".join(str(item.get("text") or item.get("analysis") or "").split())
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if not item_id or not text or page <= 0 or item_id in seen:
            continue
        seen.add(item_id)
        normalized.append({
            "id": item_id,
            "page": page,
            "text": text[:1600],
            "caption": " ".join(str(item.get("caption") or "").split())[:400],
            "figure_id": str(item.get("figure_id") or "").strip()[:160],
            "bbox": list(item.get("bbox") or [])[:4],
            "revision": str(item.get("visual_supplement_revision") or "").strip()[:80],
            "visual_model": dict(item.get("visual_model") or {}),
        })
        if len(normalized) >= max(0, int(limit)):
            break
    return normalized


def _visual_overlay_cache_key(
    embedding_model_id: str,
    embedding_provider: str,
    embedding_api_host: str,
    evidence: List[dict],
) -> str:
    payload = {
        "embedding_model": str(embedding_model_id or ""),
        "embedding_provider": str(embedding_provider or ""),
        "embedding_api_host": str(embedding_api_host or ""),
        "items": [
            {
                "id": item["id"],
                "revision": item["revision"],
                "text": item["text"],
                "caption": item["caption"],
            }
            for item in evidence
        ],
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _get_visual_overlay_vectors(
    evidence: List[dict],
    *,
    embedding_model_id: str,
    embedding_provider: str,
    embedding_api_host: str,
    embed_fn: Callable,
    is_ip_index: bool,
) -> np.ndarray:
    cache_key = _visual_overlay_cache_key(
        embedding_model_id,
        embedding_provider,
        embedding_api_host,
        evidence,
    )
    with _VISUAL_EVIDENCE_VECTOR_CACHE_LOCK:
        cached = _VISUAL_EVIDENCE_VECTOR_CACHE.get(cache_key)
        if cached is not None:
            _VISUAL_EVIDENCE_VECTOR_CACHE.move_to_end(cache_key)
            return cached.copy()

    texts = [
        "\n".join(part for part in ("[图表视觉补充]", item.get("caption") or "", item["text"]) if part)
        for item in evidence
    ]
    vectors = np.asarray(embed_fn(texts), dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[0] != len(evidence):
        raise ValueError("视觉补充 embedding 维度无效")
    if is_ip_index:
        faiss.normalize_L2(vectors)

    with _VISUAL_EVIDENCE_VECTOR_CACHE_LOCK:
        _VISUAL_EVIDENCE_VECTOR_CACHE[cache_key] = vectors.copy()
        _VISUAL_EVIDENCE_VECTOR_CACHE.move_to_end(cache_key)
        while len(_VISUAL_EVIDENCE_VECTOR_CACHE) > _VISUAL_EVIDENCE_VECTOR_CACHE_MAX_SIZE:
            _VISUAL_EVIDENCE_VECTOR_CACHE.popitem(last=False)
    return vectors


def _append_runtime_visual_evidence_chunks(
    chunks: List[str],
    chunk_headings: List[str],
    chunk_pages: List[int],
    chunk_types: List[str],
    chunk_metadata: List[dict],
    visual_evidence: Optional[List[dict]],
) -> List[int]:
    """Append committed visual evidence as non-persistent runtime chunks."""
    evidence = _normalize_runtime_visual_evidence(visual_evidence)
    if not evidence:
        return []

    indices: List[int] = []
    for item in evidence:
        caption = item.get("caption") or f"图表 {item['figure_id'] or item['id']}"
        chunk = "\n".join(part for part in ("[图表视觉补充]", caption, item["text"]) if part)
        index = len(chunks)
        chunks.append(chunk)
        chunk_headings.append(f"图表视觉补充 · {caption}"[:480])
        chunk_pages.append(item["page"])
        chunk_types.append("visual_evidence")
        chunk_metadata.append({
            "source": "visual_vlm",
            "visual_enhancement": True,
            "visual_source": "visual_vlm",
            # Keep the durable supplement id alongside the request-local
            # chunk index so citations and diagnostics can trace the evidence
            # back to the exact VLM figure reading after the request ends.
            "visual_evidence_id": item["id"],
            "context_id": f"visual:{item['id']}",
            "evidence_id": item["id"],
            "block_id": item["id"],
            "visual_supplement_revision": item.get("revision") or "",
            "figure_id": item.get("figure_id") or "",
            "bbox": item.get("bbox") or [],
            "figure_bbox": item.get("bbox") or [],
            "visual_model": item.get("visual_model") or {},
            "runtime_visual_overlay": True,
        })
        indices.append(index)
    return indices


def _build_runtime_visual_overlay_results(
    *,
    chunks: List[str],
    chunk_headings: List[str],
    chunk_pages: List[int],
    chunk_types: List[str],
    chunk_metadata: List[dict],
    indices: List[int],
    query: str,
    query_vector: np.ndarray,
    embedding_model_id: str,
    embedding_provider: str,
    embedding_api_host: str,
    embed_fn: Callable,
    is_ip_index: bool,
) -> List[dict]:
    if not indices or query_vector is None:
        return []
    evidence = []
    for index in indices:
        if index >= len(chunks) or index >= len(chunk_metadata):
            continue
        metadata = chunk_metadata[index]
        evidence.append({
            "id": str(metadata.get("visual_evidence_id") or index),
            "page": chunk_pages[index] if index < len(chunk_pages) else 0,
            "text": chunks[index],
            "caption": chunk_headings[index] if index < len(chunk_headings) else "",
            "revision": str(metadata.get("visual_supplement_revision") or ""),
            "visual_model": dict(metadata.get("visual_model") or {}),
        })
    if not evidence:
        return []

    vectors = _get_visual_overlay_vectors(
        evidence,
        embedding_model_id=embedding_model_id,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
        embed_fn=embed_fn,
        is_ip_index=is_ip_index,
    )
    query_row = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    if vectors.shape[1] != query_row.shape[1]:
        raise ValueError("视觉补充与查询 embedding 维度不一致")

    if is_ip_index:
        raw_scores = vectors @ query_row[0]
    else:
        raw_scores = np.sum((vectors - query_row[0]) ** 2, axis=1)

    results: List[dict] = []
    for index, raw_score in zip(indices, raw_scores):
        similarity = _distance_to_similarity(float(raw_score), is_ip=is_ip_index)
        # VLM output is supporting evidence.  It can enrich a chart answer,
        # but it should not displace stronger original text/table evidence.
        adjusted_similarity = min(0.72, similarity * 0.78)
        if adjusted_similarity < 0.28:
            continue
        chunk_text = chunks[index]
        metadata = chunk_metadata[index] if index < len(chunk_metadata) else {}
        page_num = chunk_pages[index] if index < len(chunk_pages) else 0
        snippet, highlights = _extract_snippet_and_highlights(chunk_text, query)
        result = {
            "chunk": chunk_text,
            "page": page_num,
            "score": float(raw_score),
            "similarity": adjusted_similarity,
            "similarity_percent": round(adjusted_similarity * 100, 2),
            "visual_overlay_score": round(similarity, 6),
            "snippet": snippet,
            "highlights": highlights,
            "reranked": False,
            "chunk_id": index,
            "chunk_heading": chunk_headings[index] if index < len(chunk_headings) else "图表视觉补充",
            "section_path": chunk_headings[index] if index < len(chunk_headings) else "图表视觉补充",
            "chunk_type": chunk_types[index] if index < len(chunk_types) else "visual_evidence",
            "block_type": "caption",
        }
        _apply_chunk_metadata(result, metadata)
        _append_retrieval_source(result, "visual_overlay")
        results.append(result)
    return sorted(results, key=lambda item: item.get("similarity", 0.0), reverse=True)[:2]


def _merge_runtime_visual_overlay_results(results: List[dict], overlay_results: List[dict]) -> List[dict]:
    if not overlay_results:
        return results
    existing_chunks = {str(item.get("chunk") or "") for item in results}
    merged = [*results]
    for item in overlay_results:
        if str(item.get("chunk") or "") not in existing_chunks:
            merged.append(item)
    return sorted(merged, key=lambda item: item.get("similarity", 0.0), reverse=True)


def _result_similarity_for_visual_slot(item: dict) -> float | None:
    if not isinstance(item, dict):
        return None
    for key in ("similarity", "combined_score", "rerank_score", "score"):
        try:
            value = float(item.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _is_protected_from_visual_overlay_replacement(item: dict) -> bool:
    """Never evict precise table evidence to make room for VLM commentary."""
    if not isinstance(item, dict):
        return True
    chunk_type = str(item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    return bool(
        chunk_type in {"table", "table_row", "table_cell"}
        or item.get("table_id")
        or item.get("table_bundle_id")
        or item.get("numeric_table_exact_context_row_text")
    )


def _retain_runtime_visual_overlay_result(
    results: List[dict],
    overlay_results: List[dict],
    *,
    top_k: int,
) -> List[dict]:
    """Keep at most one qualified visual observation through final retrieval cuts.

    VLM evidence is deliberately not part of the persistent index, so later
    group fusion and reranking can legitimately omit it.  Once it has passed
    the overlay relevance threshold, retain one only when it can replace a
    weaker, non-structured result.  It cannot displace table/row evidence.
    """
    if top_k <= 0:
        return []
    final = list(results[:top_k])
    if any(item.get("runtime_visual_overlay") for item in final if isinstance(item, dict)):
        return final

    candidates = [
        item for item in overlay_results
        if isinstance(item, dict) and item.get("runtime_visual_overlay")
    ]
    if not candidates:
        return final
    candidate = max(candidates, key=lambda item: _result_similarity_for_visual_slot(item) or float("-inf"))
    candidate_score = _result_similarity_for_visual_slot(candidate)
    if candidate_score is None:
        return final

    existing_chunks = {str(item.get("chunk") or "") for item in final if isinstance(item, dict)}
    if str(candidate.get("chunk") or "") in existing_chunks:
        return final
    if len(final) < top_k:
        return [*final, candidate]

    replaceable = [
        (index, item, _result_similarity_for_visual_slot(item))
        for index, item in enumerate(final)
        if isinstance(item, dict) and not _is_protected_from_visual_overlay_replacement(item)
    ]
    replaceable = [row for row in replaceable if row[2] is not None]
    if not replaceable:
        return final
    replace_index, _incumbent, incumbent_score = min(replaceable, key=lambda row: row[2])
    if candidate_score < incumbent_score:
        return final

    final[replace_index] = candidate
    return final


_CITATION_ANCHOR_BBOX_KEYS = (
    "bbox",
    "page_bbox",
    "block_bbox",
    "table_bbox",
    "bounding_box",
    "full_bbox_page_pts",
    "body_bbox_page_pts",
)

_CITATION_ANCHOR_INDEX_CACHE: OrderedDict[str, tuple[tuple[int, int], dict]] = OrderedDict()
_CITATION_ANCHOR_INDEX_CACHE_LOCK = threading.Lock()
_CITATION_ANCHOR_INDEX_CACHE_MAX_SIZE = 24


def _load_cached_citation_block_index(path: Path) -> dict | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    stamp = (int(stat.st_mtime_ns), int(stat.st_size))
    cache_key = str(path.resolve())

    with _CITATION_ANCHOR_INDEX_CACHE_LOCK:
        cached = _CITATION_ANCHOR_INDEX_CACHE.get(cache_key)
        if cached and cached[0] == stamp:
            _CITATION_ANCHOR_INDEX_CACHE.move_to_end(cache_key)
            return cached[1]

    try:
        block_index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(block_index, dict):
        return None

    with _CITATION_ANCHOR_INDEX_CACHE_LOCK:
        _CITATION_ANCHOR_INDEX_CACHE[cache_key] = (stamp, block_index)
        _CITATION_ANCHOR_INDEX_CACHE.move_to_end(cache_key)
        while len(_CITATION_ANCHOR_INDEX_CACHE) > _CITATION_ANCHOR_INDEX_CACHE_MAX_SIZE:
            _CITATION_ANCHOR_INDEX_CACHE.popitem(last=False)
    return block_index


def _valid_citation_bbox(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        bbox = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(item) for item in bbox):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _citation_anchor_value(item: dict, *keys):
    containers = [item]
    metadata = item.get("metadata") if isinstance(item, dict) else None
    if isinstance(metadata, dict):
        containers.append(metadata)
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _citation_anchor_metadata_from_result(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}

    anchor: dict = {}
    page_range = _citation_anchor_value(item, "page_range")
    page = _citation_anchor_value(item, "page")
    if page in (None, "") and isinstance(page_range, (list, tuple)) and page_range:
        page = page_range[0]
    try:
        page = int(page or 0)
    except (TypeError, ValueError):
        page = 0
    if page > 0:
        anchor["page"] = page
        anchor["page_range"] = [page, page]
    elif isinstance(page_range, (list, tuple)) and page_range:
        try:
            start = int(page_range[0])
            end = int(page_range[1] if len(page_range) > 1 else start)
        except (TypeError, ValueError):
            start = end = 0
        if start > 0 and end >= start:
            anchor["page_range"] = [start, end]

    for key in ("block_id", "chunk_id"):
        value = _citation_anchor_value(item, key)
        if value not in (None, ""):
            anchor[key] = value

    for key in _CITATION_ANCHOR_BBOX_KEYS:
        bbox = _valid_citation_bbox(_citation_anchor_value(item, key))
        if bbox:
            anchor["bbox"] = bbox
            break

    raw_rects = _citation_anchor_value(item, "rects", "line_rects")
    if isinstance(raw_rects, list):
        rects = [_valid_citation_bbox(value) for value in raw_rects]
        rects = [value for value in rects if value]
        if rects:
            anchor["rects"] = rects[:64]

    page_size = _citation_anchor_value(item, "page_size")
    if not isinstance(page_size, (list, tuple)) or len(page_size) < 2:
        width = _citation_anchor_value(item, "page_width", "width_pts")
        height = _citation_anchor_value(item, "page_height", "height_pts")
        page_size = [width, height] if width and height else None
    if isinstance(page_size, (list, tuple)) and len(page_size) >= 2:
        try:
            width = float(page_size[0])
            height = float(page_size[1])
        except (TypeError, ValueError):
            width = height = 0.0
        if np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0:
            anchor["page_size"] = [width, height]

    coordinate_space = str(
        _citation_anchor_value(item, "coordinate_space") or ""
    ).strip()
    if coordinate_space:
        anchor["coordinate_space"] = coordinate_space[:80]
    parse_generation = str(
        _citation_anchor_value(item, "parse_generation") or ""
    ).strip()
    if parse_generation:
        anchor["parse_generation"] = parse_generation[:160]
    return anchor


def _citation_anchor_normalize_text(value) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _citation_anchor_bigrams(value) -> set[str]:
    text = _citation_anchor_normalize_text(value)
    if len(text) < 2:
        return {text} if text else set()
    return {text[index:index + 2] for index in range(len(text) - 1)}


def _citation_anchor_text_similarity(left, right) -> float:
    left_set = _citation_anchor_bigrams(left)
    right_set = _citation_anchor_bigrams(right)
    if not left_set or not right_set:
        return 0.0
    return (2.0 * len(left_set & right_set)) / (len(left_set) + len(right_set))


def _citation_anchor_block_score(block: dict, result: dict, query: str) -> float:
    block_text = str(block.get("text") or "")
    normalized_block = _citation_anchor_normalize_text(block_text)
    if not normalized_block:
        return -1.0

    focus_text = str(
        result.get("highlight_text")
        or result.get("snippet")
        or query
        or ""
    )
    chunk_text = str(result.get("chunk") or result.get("text") or "")
    normalized_focus = _citation_anchor_normalize_text(focus_text)
    normalized_chunk = _citation_anchor_normalize_text(chunk_text)
    score = 0.0
    if normalized_focus:
        if normalized_focus in normalized_block:
            score += 180.0 + min(len(normalized_focus), 240) / 8.0
        elif len(normalized_block) >= 8 and normalized_block in normalized_focus:
            score += 145.0 + min(len(normalized_block), 240) / 10.0
        score += _citation_anchor_text_similarity(normalized_block, normalized_focus) * 70.0
    if normalized_chunk:
        if normalized_block in normalized_chunk:
            score += 12.0 + min(len(normalized_block), 400) / 100.0
        elif len(normalized_chunk) >= 12 and normalized_chunk in normalized_block:
            score += 8.0
    if str(block.get("type") or "").lower() == "artifact":
        score -= 100.0
    return score


def _select_block_line_rects(block: dict, result: dict, query: str) -> list[list[float]]:
    anchors = [
        anchor
        for anchor in (block.get("line_anchors") or [])
        if isinstance(anchor, dict)
        and anchor.get("text")
        and _valid_citation_bbox(anchor.get("bbox"))
    ]
    if not anchors:
        return []

    focus_text = str(
        result.get("highlight_text")
        or result.get("snippet")
        or query
        or ""
    )
    normalized_focus = _citation_anchor_normalize_text(focus_text)
    if len(normalized_focus) < 4:
        return []

    best: tuple[float, int, int] | None = None
    max_window = min(5, len(anchors))
    for start in range(len(anchors)):
        for size in range(1, max_window + 1):
            end = min(len(anchors), start + size)
            if end <= start:
                continue
            window_text = " ".join(
                str(anchor.get("text") or "")
                for anchor in anchors[start:end]
            )
            normalized_window = _citation_anchor_normalize_text(window_text)
            if not normalized_window:
                continue
            score = _citation_anchor_text_similarity(
                normalized_window,
                normalized_focus,
            ) * 100.0
            if normalized_focus in normalized_window:
                score += 120.0
            elif normalized_window in normalized_focus:
                score += 90.0
            score -= max(0, size - 1) * 1.5
            if best is None or score > best[0]:
                best = (score, start, end)

    if best is None or best[0] < 45.0:
        return []
    return [
        list(anchors[index]["bbox"])[:4]
        for index in range(best[1], best[2])
    ]


def _load_block_index_for_citation_anchors(
    doc_id: str,
    vector_store_dir: str,
    index_meta: dict,
) -> dict | None:
    if not doc_id or not vector_store_dir:
        return None
    path = Path(vector_store_dir).parent / "block_indexes" / f"{doc_id}.json"
    block_index = _load_cached_citation_block_index(path)
    if not block_index:
        return None

    metadata = index_meta if isinstance(index_meta, dict) else {}
    expected_generation = str(metadata.get("parse_generation") or "").strip()
    expected_source_hash = str(
        metadata.get("document_source_hash") or metadata.get("source_hash") or ""
    ).strip()
    expected_block_index_hash = str(
        metadata.get("block_index_hash") or metadata.get("block_index_revision") or ""
    ).strip()
    if expected_generation and str(block_index.get("parse_generation") or "") != expected_generation:
        return None
    if expected_source_hash and str(block_index.get("document_source_hash") or "") != expected_source_hash:
        return None
    if expected_block_index_hash and str(
        block_index.get("block_index_hash") or block_index.get("block_index_revision") or ""
    ).strip() != expected_block_index_hash:
        return None
    return block_index


def _attach_block_index_citation_anchors(
    doc_id: str,
    vector_store_dir: str,
    results: List[dict],
    *,
    query: str = "",
    index_meta: Optional[dict] = None,
) -> List[dict]:
    block_index = _load_block_index_for_citation_anchors(
        doc_id,
        vector_store_dir,
        index_meta or {},
    )
    if not block_index or not results:
        return results

    pages_by_number = {
        int(page.get("page") or 0): page
        for page in (block_index.get("pages") or [])
        if isinstance(page, dict) and int(page.get("page") or 0) > 0
    }
    parse_generation = str(block_index.get("parse_generation") or "")
    attached = 0

    for result in results:
        if not isinstance(result, dict):
            continue
        try:
            page_number = int(result.get("page") or (result.get("page_range") or [0])[0] or 0)
        except (TypeError, ValueError, IndexError):
            page_number = 0
        page = pages_by_number.get(page_number)
        if not page:
            continue

        page_size = [
            float(page.get("width_pts") or 612.0),
            float(page.get("height_pts") or 792.0),
        ]
        current_anchor = _citation_anchor_metadata_from_result(result)
        if current_anchor.get("bbox"):
            result.setdefault("coordinate_space", "pdf_top_left_points")
            result.setdefault("page_size", page_size)
            if parse_generation:
                result.setdefault("parse_generation", parse_generation)
            continue

        blocks = [
            block
            for block in (page.get("blocks") or [])
            if isinstance(block, dict) and _valid_citation_bbox(block.get("bbox"))
        ]
        if not blocks:
            continue

        matched_block = None
        requested_block_id = str(result.get("block_id") or "")
        if requested_block_id:
            matched_block = next(
                (
                    block for block in blocks
                    if str(block.get("block_id") or "") == requested_block_id
                ),
                None,
            )
        if matched_block is None:
            scored = [
                (_citation_anchor_block_score(block, result, query), block)
                for block in blocks
            ]
            score, candidate = max(scored, key=lambda item: item[0])
            if score >= 28.0:
                matched_block = candidate
        if matched_block is None:
            continue

        result["block_id"] = matched_block.get("block_id") or result.get("block_id") or ""
        result["bbox"] = list(matched_block.get("bbox") or [])[:4]
        line_rects = _select_block_line_rects(matched_block, result, query)
        if line_rects:
            result["rects"] = line_rects
        result["coordinate_space"] = "pdf_top_left_points"
        result["page_size"] = page_size
        result["page_range"] = [page_number, page_number]
        if parse_generation:
            result["parse_generation"] = parse_generation
        attached += 1

    if attached:
        logger.debug("[%s] attached block citation anchors to %s retrieval results", doc_id, attached)
    return results


def _normalize_intent_page_ranges(intent_decision: Optional[dict]) -> tuple[tuple[int, int], ...]:
    """Read page constraints from the frozen route without expanding them."""
    raw_ranges = intent_decision.get("page_ranges") if isinstance(intent_decision, dict) else []
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_range in raw_ranges or []:
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) < 2:
            continue
        try:
            start = int(raw_range[0])
            end = int(raw_range[1])
        except (TypeError, ValueError):
            continue
        if start <= 0 or end <= 0:
            continue
        value = (min(start, end), max(start, end))
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _result_overlaps_page_scope(
    result: dict,
    page_ranges: tuple[tuple[int, int], ...],
) -> bool:
    if not page_ranges:
        return True
    if not isinstance(result, dict):
        return False

    page_intervals: list[tuple[int, int]] = []
    raw_range = result.get("page_range")
    if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
        try:
            range_start = int(raw_range[0])
            range_end = int(raw_range[1])
        except (TypeError, ValueError):
            range_start = range_end = 0
        if range_start > 0 and range_end > 0:
            page_intervals.append((min(range_start, range_end), max(range_start, range_end)))

    for page in _extract_page_candidates_from_metadata(result):
        page_intervals.append((page, page))
    if not page_intervals:
        return False
    return any(
        start <= interval_end and interval_start <= end
        for interval_start, interval_end in page_intervals
        for start, end in page_ranges
    )


def _filter_results_to_intent_page_scope(
    results: List[dict],
    page_ranges: tuple[tuple[int, int], ...],
) -> List[dict]:
    if not page_ranges:
        return results
    return [
        result for result in results
        if _result_overlaps_page_scope(result, page_ranges)
    ]

def search_document_chunks(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: str = None,
    top_k: int = 10,
    candidate_k: int = 20,
    use_rerank: bool = False,
    reranker_model: Optional[str] = None,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None,
    use_hybrid: bool = True,
    selected_text: Optional[str] = None,  # 新增：用于查询改写中的指示代词解析
    progress_callback: Optional[Callable[[dict], None]] = None,
    enable_query_expansion_override: Optional[bool] = None,
    query_expansion_api_key: Optional[str] = None,
    query_expansion_model: str = "",
    query_expansion_provider: str = "",
    query_expansion_endpoint: str = "",
    visual_evidence: Optional[List[dict]] = None,
    intent_decision: Optional[dict] = None,
    query_is_canonical: bool = False,
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_api_host: Optional[str] = None,
) -> Tuple[List[dict], dict]:
    """检索文档 chunk，返回检索结果和各阶段耗时。

    Returns:
        (results, timings) 元组
        - results: 检索结果列表，每项包含 chunk、page、score 等字段
        - timings: 各阶段耗时字典（毫秒），如 {"vector_search_ms": 12.3, "total_ms": 29.3}
          未执行的阶段不包含对应字段
    """
    original_query = query

    decision = _resolve_intent_decision(original_query, intent_decision)
    page_scope_ranges = _normalize_intent_page_ranges(decision)
    query_is_canonical = bool(
        query_is_canonical
        or (
            isinstance(intent_decision, dict)
            and str(intent_decision.get("intent_id") or "").strip()
        )
    )

    # Query rewriting is a legacy convenience for direct callers. Route-level
    # intent orchestration passes a frozen retrieval query and must not let the
    # lower layer mutate it a second time.
    pre_rewrite_evidence_need = list(decision.get("evidence_need") or [])
    try:
        if not query_is_canonical:
            rewritten_query = _query_rewriter_singleton.rewrite(
                original_query,
                selected_text=selected_text,
                evidence_need=pre_rewrite_evidence_need,
            )
            if rewritten_query != original_query:
                logger.info(f"[{doc_id}] 查询改写: '{original_query}' → '{rewritten_query}'")
                query = rewritten_query
                _emit_retrieval_progress(
                    progress_callback,
                    "query_rewrite",
                    "检索问题已改写，正在继续检索...",
                    query=query,
                )
    except Exception as e:
        logger.warning(f"[{doc_id}] 查询改写失败，使用原始查询: {e}")

    # 查询类型分析 + 动态 candidate_k（提升召回率）
    query_type = str(decision.get("query_type") or "specific")
    evidence_need = pre_rewrite_evidence_need
    analysis_query = original_query
    _rc_ref = _rag_config_singleton
    _candidate_k_map = {
        'extraction': max(50, int(candidate_k * _rc_ref.extraction_candidate_multiplier)),
        'overview':   max(30, int(candidate_k * _rc_ref.overview_candidate_multiplier)),
        'analytical': max(25, int(candidate_k * _rc_ref.analytical_candidate_multiplier)),
        'specific':   max(20, candidate_k),
    }
    candidate_k = _candidate_k_map.get(query_type, max(candidate_k, 20))
    if 'numeric_table' in evidence_need:
        candidate_k = max(candidate_k, 48)
    if 'section_explanation' in evidence_need:
        candidate_k = max(candidate_k, 30)
    logger.info(
        f"[{doc_id}] 查询类型: {query_type}, evidence_need={evidence_need}, 动态 candidate_k: {candidate_k}"
    )
    _emit_retrieval_progress(
        progress_callback,
        "analysis",
        f"正在分析查询并确定检索策略（{query_type}）...",
        query_type=query_type,
        evidence_need=evidence_need,
        candidate_k=candidate_k,
    )

    # Wide-net: expand candidate pool when reranking, giving reranker more choices
    if use_rerank:
        wide_k = max(candidate_k, top_k * _WIDE_RETRIEVAL_MULTIPLIER)
        if wide_k > candidate_k:
            logger.info(f"[{doc_id}] Wide-net rerank: candidate_k {candidate_k} → {wide_k}")
            candidate_k = wide_k

    # 检索耗时记录（需求 10.1）
    timings = {}
    t_total = time.perf_counter()

    index_path = os.path.join(vector_store_dir, f"{doc_id}.index")
    chunks_path = os.path.join(vector_store_dir, f"{doc_id}.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise HTTPException(status_code=404, detail="向量索引未找到,请重新上传PDF")

    # 优先从 LRU 缓存读取，避免每次磁盘 I/O
    cached = _index_cache.get_index(doc_id, index_path, chunks_path)
    if cached is not None:
        index, data = cached
    else:
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        _index_cache.put_index(doc_id, index, data, index_path, chunks_path)

    data = _require_current_vector_index_schema(data, doc_id)
    verified_embedding = _resolve_verified_query_embedding_identity(
        data,
        api_key=api_key,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
    )

    pages = _annotate_pages_with_provenance(pages)

    chunks = data["chunks"]
    embedding_model_id = verified_embedding["model"]
    verified_embedding_provider = verified_embedding["provider"]
    verified_embedding_api_host = verified_embedding["api_host"]
    verified_embedding_api_key = verified_embedding["api_key"]
    embedding_cache_scope = _embedding_cache_scope(
        embedding_model_id,
        verified_embedding_provider,
        verified_embedding_api_host,
    )
    parent_chunks = data.get("parent_chunks", [])
    child_to_parent = data.get("child_to_parent", {})
    chunk_headings = data.get("chunk_headings") or [""] * len(chunks)
    chunk_pages = data.get("chunk_pages") or [0] * len(chunks)
    chunk_types = data.get("chunk_types") or [_guess_chunk_type(c) for c in chunks]
    chunk_metadata = _normalize_chunk_metadata_list(data.get("chunk_metadata"), len(chunks))

    if len(chunk_headings) != len(chunks):
        chunk_headings = (chunk_headings[:len(chunks)] + [""] * len(chunks))[:len(chunks)]
    if len(chunk_pages) != len(chunks):
        chunk_pages = (chunk_pages[:len(chunks)] + [0] * len(chunks))[:len(chunks)]
    if len(chunk_types) != len(chunks):
        chunk_types = (chunk_types[:len(chunks)] + [""] * len(chunks))[:len(chunks)]
    if len(chunk_metadata) != len(chunks):
        chunk_metadata = _normalize_chunk_metadata_list(chunk_metadata, len(chunks))

    visual_overlay_indices: List[int] = []
    if visual_evidence and "numeric_table" not in evidence_need:
        # The index cache owns these lists.  Copy before adding a request-local
        # overlay so one query cannot leak visual chunks into another request.
        chunks = list(chunks)
        chunk_headings = list(chunk_headings)
        chunk_pages = list(chunk_pages)
        chunk_types = list(chunk_types)
        chunk_metadata = [dict(item) for item in chunk_metadata]
        visual_overlay_indices = _append_runtime_visual_evidence_chunks(
            chunks,
            chunk_headings,
            chunk_pages,
            chunk_types,
            chunk_metadata,
            visual_evidence,
        )

    _maybe_append_runtime_structured_table_bundle_chunks(
        doc_id,
        chunks,
        chunk_headings,
        chunk_pages,
        chunk_types,
        chunk_metadata,
        pages,
    )

    semantic_groups_current = (
        _semantic_groups_match_vector_index(doc_id, vector_store_dir)
        and not page_scope_ranges
    )
    group_chunk_map = _load_group_data(doc_id) or {} if semantic_groups_current else {}

    embed_fn = get_embedding_function(
        embedding_model_id,
        verified_embedding_api_key,
        verified_embedding_api_host,
        False,
    )

    # 预构建页面前缀索引，加速 chunk → 页码映射
    _page_index = _build_page_index(pages)

    if pages and any((not isinstance(page, int) or page <= 0) for page in chunk_pages):
        chunk_pages = [
            page if isinstance(page, int) and page > 0 else _find_page_for_chunk(chunks[idx], pages, page_index=_page_index)
            for idx, page in enumerate(chunk_pages)
        ]

    # 检测索引类型：IP（新索引）还是 L2（旧索引）
    is_ip_index = (index.metric_type == faiss.METRIC_INNER_PRODUCT)

    def _normalize_query_vector(vec):
        """归一化查询向量（仅 IP 索引需要）"""
        v = np.array(vec).astype('float32')
        if is_ip_index:
            faiss.normalize_L2(v)
        return v

    # ---- RAG 优化：HyDE + 多查询扩展 ----
    _search_rag_config = _rag_config_singleton

    # HyDE：用假设文档的 embedding 替代原始查询 embedding
    hyde_passage = None
    hyde_enabled = (
        False
        if enable_query_expansion_override is False
        else _should_enable_hyde(query_type, evidence_need, _search_rag_config)
    )
    if hyde_enabled and query_expansion_api_key:
        try:
            _emit_retrieval_progress(progress_callback, "hyde_start", "正在生成语义扩展（HyDE）...")
            from services.query_expander import generate_hyde_passage
            hyde_passage = _run_async(generate_hyde_passage(
                query,
                query_expansion_api_key,
                model=query_expansion_model,
                provider=query_expansion_provider,
                endpoint=query_expansion_endpoint,
            ))
            if hyde_passage:
                logger.info(f"[{doc_id}] HyDE 启用，假设文档 {len(hyde_passage)} 字符")
                _emit_retrieval_progress(progress_callback, "hyde_done", "HyDE 扩展完成，正在召回相关内容...")
        except Exception as e:
            logger.warning(f"[{doc_id}] HyDE 生成失败，降级为原始查询: {e}")
    elif hyde_enabled:
        logger.info(f"[{doc_id}] 跳过 HyDE：未提供与辅助模型匹配的专用 API Key")

    search_k = max(candidate_k, top_k)

    # 向量检索计时开始（需求 10.1）
    t0 = time.perf_counter()
    query_vector = None
    hyde_vector = None
    D = I = D_orig = I_orig = None
    vector_error = None

    try:
        # 查询向量 LRU 缓存（需求 5.1, 5.2, 5.3）
        # HyDE 模式下：同时缓存原始查询向量和 HyDE 向量
        cached_vector = _query_vector_cache.get(embedding_cache_scope, query)
        if cached_vector is not None:
            query_vector = _ensure_query_vector_matches_index(cached_vector, index)
            logger.info(f"[{doc_id}] 查询向量缓存命中: model={embedding_model_id}")
            _emit_retrieval_progress(progress_callback, "embedding_query", "查询向量缓存命中，正在进行召回...")
        else:
            _emit_retrieval_progress(progress_callback, "embedding_query", "正在计算查询向量...")
            query_vector = _ensure_query_vector_matches_index(
                _normalize_query_vector(embed_fn([query])),
                index,
            )
            _query_vector_cache.put(embedding_cache_scope, query, query_vector)

        # HyDE：额外生成假设文档的 embedding 用于检索
        if hyde_passage:
            hyde_cache_key = f"hyde:{query}"
            cached_hyde = _query_vector_cache.get(embedding_cache_scope, hyde_cache_key)
            if cached_hyde is not None:
                hyde_vector = _ensure_query_vector_matches_index(cached_hyde, index)
                _emit_retrieval_progress(progress_callback, "hyde_cache_hit", "HyDE 向量缓存命中，正在继续召回...")
            else:
                _emit_retrieval_progress(progress_callback, "hyde_embedding", "正在计算 HyDE 向量...")
                hyde_vector = _ensure_query_vector_matches_index(
                    _normalize_query_vector(embed_fn([hyde_passage])),
                    index,
                )
                _query_vector_cache.put(embedding_cache_scope, hyde_cache_key, hyde_vector)

        # 主查询检索（使用 HyDE 向量或原始查询向量）
        primary_vector = hyde_vector if hyde_vector is not None else query_vector
        _emit_retrieval_progress(progress_callback, "vector_search", "正在进行向量召回...")
        D, I = index.search(np.asarray(primary_vector, dtype="float32"), search_k)

        # 如果启用了 HyDE，同时用原始查询向量检索并合并（双路 RRF）
        if hyde_vector is not None:
            D_orig, I_orig = index.search(np.asarray(query_vector, dtype="float32"), search_k)
    except HTTPException:
        raise
    except Exception as e:
        non_degradable_error = _build_non_degradable_query_embedding_http_error(e)
        if non_degradable_error is not None:
            logger.warning(
                f"[{doc_id}] 查询 embedding 不可降级失败: {_summarize_embedding_error(e)}"
            )
            raise non_degradable_error from e
        vector_error = e
        if not use_hybrid:
            raise
        logger.warning(f"[{doc_id}] 向量召回失败，降级为 BM25-only/hybrid: {e}")
        _emit_retrieval_progress(
            progress_callback,
            "vector_search_fallback",
            "向量召回失败，正在降级为关键词检索...",
        )
    finally:
        timings["vector_search_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    vector_results = []
    vector_chunk_set = set()  # 记录向量搜索已返回的 chunk

    def _build_results_from_faiss(D_arr, I_arr):
        """从 FAISS 搜索结果构建结果列表"""
        results = []
        for dist, idx in zip(D_arr[0], I_arr[0]):
            # FAISS pads an undersized result set with ``-1``.  Python would
            # otherwise treat that as the last request-local overlay chunk.
            if 0 <= idx < len(chunks):
                chunk_text = chunks[idx]
                metadata = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
                page_num = chunk_pages[idx] if idx < len(chunk_pages) else 0
                page_num = _resolve_primary_page_from_metadata(metadata, fallback=page_num)
                if (not isinstance(page_num, int) or page_num <= 0) and pages:
                    page_num = _find_page_for_chunk(chunk_text, pages, page_index=_page_index)
                similarity = _distance_to_similarity(float(dist), is_ip=is_ip_index)
                snippet, highlights = _extract_snippet_and_highlights(chunk_text, query)
                chunk_heading = chunk_headings[idx] if idx < len(chunk_headings) else ""
                chunk_type = chunk_types[idx] if idx < len(chunk_types) else _guess_chunk_type(chunk_text)
                result = {
                    "chunk": chunk_text,
                    "page": page_num,
                    "score": float(dist),
                    "similarity": similarity,
                    "similarity_percent": round(similarity * 100, 2),
                    "snippet": snippet,
                    "highlights": highlights,
                    "reranked": False,
                    "chunk_id": idx,
                    "chunk_heading": chunk_heading,
                    "section_path": chunk_heading,
                    "chunk_type": chunk_type,
                    "block_type": chunk_type,
                }
                if idx < len(chunk_metadata):
                    _apply_chunk_metadata(result, metadata)
                _apply_page_provenance(result, metadata)
                results.append(result)
        return results

    def _build_results_from_bm25(bm25_items: List[dict]) -> List[dict]:
        """将 BM25 结果转换为统一的候选格式。"""
        if not bm25_items:
            return []

        max_score = max(float(item.get("score", 0.0) or 0.0) for item in bm25_items)
        max_score = max(max_score, 1.0)
        results = []

        for item in bm25_items:
            idx = item.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(chunks):
                continue

            chunk_text = chunks[idx]
            metadata = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
            page_num = chunk_pages[idx] if idx < len(chunk_pages) else 0
            page_num = _resolve_primary_page_from_metadata(metadata, fallback=page_num)
            if (not isinstance(page_num, int) or page_num <= 0) and pages:
                page_num = _find_page_for_chunk(chunk_text, pages, page_index=_page_index)

            score = float(item.get("score", 0.0) or 0.0)
            similarity = min(score / max_score, 0.995)
            snippet, highlights = _extract_snippet_and_highlights(chunk_text, query)
            chunk_heading = chunk_headings[idx] if idx < len(chunk_headings) else ""
            chunk_type = chunk_types[idx] if idx < len(chunk_types) else _guess_chunk_type(chunk_text)

            result = {
                "chunk": chunk_text,
                "page": page_num,
                "score": score,
                "bm25_score": score,
                "similarity": similarity,
                "similarity_percent": round(similarity * 100, 2),
                "snippet": snippet,
                "highlights": highlights,
                "reranked": False,
                "bm25": True,
                "chunk_id": idx,
                "chunk_heading": chunk_heading,
                "section_path": chunk_heading,
                "chunk_type": chunk_type,
                "block_type": chunk_type,
            }
            if idx < len(chunk_metadata):
                _apply_chunk_metadata(result, metadata)
            _apply_page_provenance(result, metadata)
            results.append(result)

        return results

    def _expand_to_parent_chunks(results_list, top_n):
        """将 child chunk 结果扩展为 parent chunk，去重同 parent 的命中

        保留每个 parent 中最高分的 child 的元数据，
        但将 chunk 文本替换为 parent chunk 文本。

        Args:
            results_list: child 级别的检索结果列表
            top_n: 返回的最大结果数

        Returns:
            parent 级别的结果列表
        """
        if not parent_chunks or not child_to_parent:
            return results_list

        # child_text -> child_index 映射
        child_text_to_idx = {chunks[i]: i for i in range(len(chunks))}

        seen_parents = {}  # parent_idx -> best result item
        expanded = []

        for item in results_list:
            child_text = item.get("chunk", "")
            child_idx = child_text_to_idx.get(child_text)
            if child_idx is None:
                # 非标准 chunk（如精确短语注入），保留原样
                expanded.append(item)
                continue

            p_idx = child_to_parent.get(child_idx)
            if p_idx is None or p_idx >= len(parent_chunks):
                expanded.append(item)
                continue

            if p_idx in seen_parents:
                # 同一 parent 的多个 child，跳过（保留最高分的）
                continue

            # 替换 chunk 为 parent chunk
            new_item = item.copy()
            new_item["chunk"] = parent_chunks[p_idx]
            new_item["child_chunk"] = child_text  # 保留原始 child 用于高亮
            new_item["child_chunk_id"] = child_idx
            new_item["parent_expanded"] = True
            new_item["parent_id"] = p_idx
            seen_parents[p_idx] = True
            expanded.append(new_item)

        return expanded[:top_n]

    primary_results = filter_reference_trap_results(
        _build_results_from_faiss(D, I) if D is not None and I is not None else [],
        query,
        evidence_need=evidence_need,
    )
    _mark_retrieval_source(primary_results, "hyde" if hyde_vector is not None else "vector")

    # HyDE 双路 RRF 融合：合并 HyDE 路和原始查询路的结果
    if D_orig is not None and I_orig is not None:
        orig_results = filter_reference_trap_results(
            _build_results_from_faiss(D_orig, I_orig),
            query,
            evidence_need=evidence_need,
        )
        _mark_retrieval_source(orig_results, "vector")
        vector_results = _hybrid_search.reciprocal_rank_fusion(
            primary_results, orig_results,
            k=60, top_k=search_k, chunk_key='chunk'
        )
        logger.info(
            f"[{doc_id}] HyDE 双路 RRF 融合: "
            f"HyDE路={len(primary_results)}, 原始路={len(orig_results)}, "
            f"融合后={len(vector_results)}"
        )
    else:
        vector_results = primary_results
    vector_results = filter_reference_trap_results(vector_results, query, evidence_need=evidence_need)

    visual_overlay_results: List[dict] = []
    if visual_overlay_indices and query_vector is not None:
        try:
            t_visual = time.perf_counter()
            visual_overlay_results = _build_runtime_visual_overlay_results(
                chunks=chunks,
                chunk_headings=chunk_headings,
                chunk_pages=chunk_pages,
                chunk_types=chunk_types,
                chunk_metadata=chunk_metadata,
                indices=visual_overlay_indices,
                query=query,
                query_vector=query_vector,
                embedding_model_id=embedding_model_id,
                embedding_provider=verified_embedding_provider,
                embedding_api_host=verified_embedding_api_host,
                embed_fn=embed_fn,
                is_ip_index=is_ip_index,
            )
            vector_results = _merge_runtime_visual_overlay_results(vector_results, visual_overlay_results)
            if visual_overlay_results:
                timings["visual_overlay_ms"] = round((time.perf_counter() - t_visual) * 1000, 1)
                logger.info("[%s] merged %s committed visual evidence candidates", doc_id, len(visual_overlay_results))
        except Exception as exc:
            logger.warning("[%s] visual evidence overlay skipped: %s", doc_id, exc)

    for item in vector_results:
        vector_chunk_set.add(item.get("chunk", ""))

    # --- 多查询扩展融合（P3.3a-c：放宽 gate + simplify + intersection mode） ---
    query_expansion_enabled = (
        bool(enable_query_expansion_override)
        if enable_query_expansion_override is not None
        else _should_enable_query_expansion(query_type, evidence_need, _search_rag_config)
    )
    if query_expansion_enabled and query_vector is not None and not query_expansion_api_key:
        logger.info(
            f"[{doc_id}] 跳过多查询扩展：未提供专用 LLM 参数，避免复用 embedding key"
        )
        query_expansion_enabled = False

    if query_expansion_enabled and query_expansion_api_key and query_vector is not None:
        try:
            _emit_retrieval_progress(progress_callback, "query_expansion_start", "正在扩展检索问题，补充同义查询...")
            from services.query_expander import expand_query
            # P3.3b 查询简化：原查询 > 50 字符时走简化版本（移除冗余前缀/填充词）
            simplified_query = query
            if getattr(_search_rag_config, "enable_query_simplify", True):
                try:
                    from services.query_simplifier import simplify_query_local
                    min_chars = int(getattr(_search_rag_config, "query_simplify_min_chars", 50))
                    if len(query) >= min_chars:
                        simp = simplify_query_local(query)
                        if simp and simp != query and len(simp) < len(query):
                            simplified_query = simp
                            logger.info(f"[{doc_id}] 查询简化: '{query[:40]}' → '{simp[:40]}'")
                except Exception as _e_simp:
                    logger.warning(f"[{doc_id}] 查询简化失败: {_e_simp}")

            expanded_queries = _run_async(
                expand_query(
                    simplified_query,
                    query_expansion_api_key,
                    n=_search_rag_config.query_expansion_n,
                    model=query_expansion_model,
                    provider=query_expansion_provider,
                    endpoint=query_expansion_endpoint,
                )
            )
            if expanded_queries:
                expansion_result_lists = [vector_results]
                for eq in expanded_queries:
                    eq_vector = _normalize_query_vector(embed_fn([eq]))
                    D_eq, I_eq = index.search(np.array(eq_vector).astype('float32'), search_k)
                    expanded_results = filter_reference_trap_results(
                        _build_results_from_faiss(D_eq, I_eq),
                        query,
                        evidence_need=evidence_need,
                    )
                    _mark_retrieval_source(expanded_results, "multi_query")
                    expansion_result_lists.append(expanded_results)
                # P3.3c 按 query_type 选择合并策略：numeric_table/extraction → intersection，其他 → rrf
                merge_mode = _select_multi_query_merge_mode(query_type, evidence_need, _search_rag_config)
                vector_results = _hybrid_search.merge_multi_query_results(
                    expansion_result_lists,
                    mode=merge_mode,
                    top_k=search_k,
                    chunk_key='chunk',
                    rrf_k=60,
                )
                logger.info(
                    f"[{doc_id}] 多查询扩展启用 (mode={merge_mode}): "
                    f"{len(expanded_queries)} 个扩展查询, 合并后 {len(vector_results)} 条"
                )
                _emit_retrieval_progress(
                    progress_callback,
                    "query_expansion_done",
                    f"多查询扩展完成，生成 {len(expanded_queries)} 个扩展查询（合并 {merge_mode}）。",
                )
        except Exception as e:
            logger.warning(f"[{doc_id}] 多查询扩展失败，跳过: {e}")

    bm25_results: List[dict] = []
    if use_hybrid:
        t_bm25 = time.perf_counter()
        try:
            _emit_retrieval_progress(progress_callback, "bm25_search", "正在进行关键词召回...")
            bm25_hits = _bm25_service.bm25_search(doc_id, query, chunks, top_k=search_k)
            bm25_results = filter_reference_trap_results(
                _build_results_from_bm25(bm25_hits),
                query,
                evidence_need=evidence_need,
            )
            _mark_retrieval_source(bm25_results, "bm25")
        except Exception as e:
            logger.warning(f"[{doc_id}] BM25 检索失败，跳过混合检索: {e}")
            bm25_results = []
        timings["bm25_search_ms"] = round((time.perf_counter() - t_bm25) * 1000, 1)

    # --- 混合检索 / BM25 回退 / 纯向量路径 ---
    # 条件 rerank gate（参考 ragflow 按题型启用）：
    # 当 enable_conditional_rerank=True 且当前 query_type 在配置列表中时强制开启 rerank
    requested_use_rerank = use_rerank
    if not use_rerank and _should_force_conditional_rerank(
        query_type,
        evidence_need,
        reranker_model,
        _search_rag_config,
    ):
        use_rerank = True
        logger.info(
            f"[{doc_id}] 条件 rerank gate 触发: query_type={query_type}, evidence_need={evidence_need}"
        )
    conditional_rerank_active = bool(use_rerank and not requested_use_rerank)
    pre_rerank_top_k = search_k if use_rerank else top_k
    if use_hybrid and bm25_results:
        if vector_results:
            results = _hybrid_search.hybrid_search_merge(
                vector_results,
                bm25_results,
                top_k=search_k,
                query_type=query_type,
            )
        else:
            results = bm25_results[:search_k]
            if vector_error is not None:
                logger.info(f"[{doc_id}] 向量召回不可用，使用 BM25-only 候选 {len(results)} 条")
    else:
        results = sorted(vector_results, key=lambda x: x.get("similarity", 0), reverse=True)

    # --- 意群级别检索 + RRF 融合（在纯向量/rerank 检索之后） ---
    # 意群检索计时开始
    t0 = time.perf_counter()
    _emit_retrieval_progress(progress_callback, "group_search_start", "正在融合语义意群结果...")
    if query_vector is not None:
        results = _merge_with_group_search(
            doc_id=doc_id,
            chunk_results=results,
            query_vector=query_vector,
            chunks=chunks,
            pages=pages,
            query=query,
            top_k=pre_rerank_top_k,
            vector_store_dir=vector_store_dir,
        )
        results = _apply_query_intent_boost(results, analysis_query)
        results = _apply_numeric_table_boost(results, analysis_query, evidence_need)
        results = _filter_reference_pollution(results, analysis_query, evidence_need=evidence_need)
    else:
        logger.info(f"[{doc_id}] 跳过意群级融合：查询向量不可用")
    # 意群检索计时结束（仅在实际执行时记录）
    group_search_elapsed = round((time.perf_counter() - t0) * 1000, 1)
    if group_search_elapsed > 0.1:
        timings["group_search_ms"] = group_search_elapsed
    _emit_retrieval_progress(progress_callback, "group_search_done", "语义意群融合完成。")

    # 邻居 chunk 上下文扩展
    try:
        _expand_n = get_context_chunk_expansion()
        if _expand_n > 0:
            results = _chunk_expander.expand_context_chunks(results, chunks, expand_n=_expand_n)
    except Exception as _expand_err:
        logger.debug(f"[{doc_id}] chunk 扩展跳过: {_expand_err}")

    # 同页表格补充检索
    results = _augment_with_table_chunks(
        results, chunks, pages, _page_index,
        query=analysis_query, evidence_need=evidence_need,
        chunk_pages=chunk_pages,
        chunk_metadata=chunk_metadata,
    )
    results = _augment_with_numeric_exact_row_search(
        results,
        chunks=chunks,
        chunk_pages=chunk_pages,
        chunk_types=chunk_types,
        chunk_metadata=chunk_metadata,
        pages=pages,
        page_index=_page_index,
        query=analysis_query,
        evidence_need=evidence_need,
    )
    results = _apply_query_intent_boost(results, analysis_query)
    results = _apply_numeric_table_boost(results, analysis_query, evidence_need)
    results = _filter_reference_pollution(results, analysis_query, evidence_need=evidence_need)
    post_clean_top_k = pre_rerank_top_k
    if "numeric_table" in evidence_need:
        post_clean_top_k = max(
            post_clean_top_k,
            min(len(results), max(top_k * 4, 24)),
        )
    results = _unified_post_clean(
        results,
        analysis_query,
        post_clean_top_k,
        evidence_need,
    )
    results = _annotate_results_for_evidence_rerank(
        doc_id=doc_id,
        results=results,
        chunks=chunks,
        parent_chunks=parent_chunks,
        chunk_headings=chunk_headings,
        chunk_pages=chunk_pages,
        chunk_types=chunk_types,
        chunk_metadata=chunk_metadata,
        child_to_parent=child_to_parent,
        group_chunk_map=group_chunk_map,
        include_rerank_text=use_rerank,
    )
    results = _expand_numeric_table_evidence_units(
        results,
        analysis_query,
        include_rerank_text=use_rerank,
        doc_title=_get_document_title(doc_id) if use_rerank else "",
    )
    results = _mark_numeric_table_support_chunks(results, analysis_query)
    results = _dedupe_numeric_table_evidence_units(results, analysis_query)
    results = _sanitize_by_chunk_type(results, analysis_query, decision)
    if use_rerank:
        results, pre_cap_stats = _apply_group_pre_cap(results)
        page_capped_results, page_pre_cap_stats = _apply_page_pre_cap(results)
        if len(page_capped_results) >= max(top_k, min(len(results), top_k * 2)):
            results = page_capped_results
        else:
            page_pre_cap_stats["recovered_due_to_small_pool"] = True
        if results and (pre_cap_stats or page_pre_cap_stats):
            results[0]["_rerank_precap_stats"] = {
                "group": pre_cap_stats,
                "page": page_pre_cap_stats,
            }
    results = _finalize_with_optional_rerank(
        query=analysis_query,
        results=results,
        top_k=top_k,
        use_rerank=use_rerank,
        reranker_model=reranker_model,
        rerank_provider=rerank_provider,
        rerank_api_key=rerank_api_key,
        rerank_endpoint=rerank_endpoint,
        timings=timings,
        progress_callback=progress_callback,
        conditional_rerank_active=conditional_rerank_active,
    )
    results = _prioritize_numeric_table_results(results, analysis_query)
    results = _ensure_structured_table_row_shard_results(
        results,
        chunks=chunks,
        chunk_pages=chunk_pages,
        chunk_types=chunk_types,
        chunk_metadata=chunk_metadata,
        query=analysis_query,
        top_k=_resolve_numeric_table_effective_top_k(analysis_query, top_k, results=results),
    )
    results = _prioritize_numeric_table_results(results, analysis_query)
    if visual_overlay_results and "numeric_table" not in evidence_need:
        before_visual_slot = len(results)
        results = _retain_runtime_visual_overlay_result(
            results,
            visual_overlay_results,
            top_k=top_k,
        )
        if any(item.get("runtime_visual_overlay") for item in results if isinstance(item, dict)):
            timings["visual_overlay_retained"] = 1
            if before_visual_slot:
                logger.info("[%s] retained one qualified visual evidence result through final ranking", doc_id)


    results = _attach_block_index_citation_anchors(
        doc_id,
        vector_store_dir,
        results,
        query=analysis_query,
        index_meta=data.get("index_meta", {}) if isinstance(data, dict) else {},
    )
    if page_scope_ranges:
        page_scope_candidates = len(results)
        results = _filter_results_to_intent_page_scope(results, page_scope_ranges)
        timings["page_scope_candidate_count"] = page_scope_candidates
        timings["page_scope_result_count"] = len(results)
    # 总耗时记录（需求 10.1）
    timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
    logger.info(f"[{doc_id}] 检索耗时: {timings}")
    _emit_retrieval_progress(
        progress_callback,
        "complete",
        f"检索完成，共耗时 {timings['total_ms']}ms。",
        timings=timings,
    )

    return results, timings


def _merge_with_group_search(
    doc_id: str,
    chunk_results: List[dict],
    query_vector: np.ndarray,
    chunks: List[str],
    pages: List[dict],
    query: str,
    top_k: int = 10,
    vector_store_dir: str = "",
) -> List[dict]:
    """尝试加载意群级别索引并与分块结果进行 RRF 融合

    如果意群索引不存在或加载失败，直接返回原始分块结果（需求 6.3 降级回退）。

    Args:
        doc_id: 文档唯一标识
        chunk_results: 分块级别检索结果
        query_vector: 查询向量
        chunks: 所有文本分块列表
        pages: 文档页面数据
        query: 用户查询文本
        top_k: 返回结果数量

    Returns:
        融合后的结果列表，或原始分块结果（降级时）
    """
    config = _rag_config_singleton

    if not _semantic_groups_match_vector_index(doc_id, vector_store_dir):
        logger.info(f"[{doc_id}] 意群索引不属于当前向量代际，降级到分块检索")
        return chunk_results

    # 检查是否启用语义意群功能
    if not config.enable_semantic_groups:
        logger.info(f"[{doc_id}] 语义意群功能已禁用，使用分块级别检索结果")
        return chunk_results

    # 小文档跳过意群检索（需求 10.3）
    if len(chunks) < config.small_doc_chunk_threshold:
        logger.info(f"[{doc_id}] 小文档（{len(chunks)} 分块），跳过意群级别检索")
        return chunk_results

    try:
        # 加载意群级别索引
        group_index_data = _load_group_index(doc_id)
        if group_index_data is None:
            # 意群索引不存在，回退到仅分块级别检索（需求 6.3）
            return chunk_results

        # 在意群级别索引中搜索（search_k 设为 top_k * 2，提高召回率）
        group_results = _search_group_index(
            group_index_data=group_index_data,
            query_vector=query_vector,
            search_k=top_k * 2,
        )

        if not group_results:
            logger.info(f"[{doc_id}] 意群级别检索无结果，使用分块级别检索结果")
            return chunk_results

        # 加载意群 JSON 数据获取 chunk_indices 映射
        group_chunk_map = _load_group_data(doc_id)

        # 使用 RRF 融合分块和意群两路结果
        merged_results = _rrf_merge_chunk_and_group(
            chunk_results=chunk_results,
            group_results=group_results,
            group_chunk_map=group_chunk_map,
            chunks=chunks,
            pages=pages,
            query=query,
            top_k=top_k,
            k=60,  # 标准 RRF 常数
        )

        logger.info(
            f"[{doc_id}] RRF 融合完成: "
            f"分块结果={len(chunk_results)}条, "
            f"意群结果={len(group_results)}条, "
            f"融合后={len(merged_results)}条"
        )

        return merged_results

    except Exception as e:
        # 意群检索失败不影响主流程，回退到分块级别检索
        logger.warning(f"[{doc_id}] 意群级别检索失败，回退到分块级别检索: {e}")
        return chunk_results


def _get_semantic_groups_dir(doc_id: str = "") -> str:
    """Return the active generation directory, with a legacy-layout fallback."""
    if not doc_id:
        return _SEMANTIC_GROUPS_DIR
    paths = semantic_group_paths(_SEMANTIC_GROUPS_DIR, doc_id)
    return str(next(iter(paths.values())).parent)


def _semantic_generation_matches_vector_index(
    doc_id: str,
    vector_store_dir: str,
    *,
    parse_generation: str = "",
    document_source_hash: str = "",
    semantic_identity: Optional[dict] = None,
) -> bool:
    if not vector_store_dir:
        return False
    chunks_path = Path(vector_store_dir) / f"{doc_id}.pkl"
    try:
        with open(chunks_path, "rb") as handle:
            data = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return False
    if not isinstance(data, dict):
        return False
    current_identity = _extract_vector_semantic_identity(data)
    expected_identity = _normalize_semantic_generation_identity(semantic_identity)
    if expected_identity:
        if not _semantic_generation_identity_complete(expected_identity):
            return False
        return _semantic_generation_identity_matches(current_identity, expected_identity)
    return (
        str(current_identity.get("parse_generation") or "") == str(parse_generation)
        and str(current_identity.get("document_source_hash") or "") == str(document_source_hash)
    )


def _semantic_groups_match_vector_index(doc_id: str, vector_store_dir: str) -> bool:
    """仅让与当前向量索引同代际的 semantic generation 参与检索。"""
    chunks_path = Path(vector_store_dir or "") / f"{doc_id}.pkl"
    try:
        with open(chunks_path, "rb") as handle:
            data = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return False
    if not isinstance(data, dict):
        return False
    try:
        index_version = int(data.get("index_version") or 0)
    except (TypeError, ValueError):
        index_version = 0
    if index_version != RAG_INDEX_VERSION:
        # 旧 semantic groups 可能含有 table_row，不能与新正文检索链混用。
        return False
    vector_identity = _extract_vector_semantic_identity(data)
    if not _semantic_generation_identity_complete(vector_identity):
        # Legacy semantic groups cannot prove which parser generation or
        # embedding build produced them. Keep chunk-level retrieval available,
        # but require regeneration before group-level evidence can participate.
        return False
    try:
        active = _normalize_semantic_generation_identity(json.loads(
            active_manifest_path(_SEMANTIC_GROUPS_DIR, doc_id).read_text(encoding="utf-8")
        ))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if not _semantic_generation_identity_complete(active):
        return False
    if not _semantic_generation_identity_matches(vector_identity, active):
        return False
    semantic_paths = semantic_group_paths(_SEMANTIC_GROUPS_DIR, doc_id)
    meta_path = semantic_paths.get("pkl")
    if meta_path is None or not meta_path.exists():
        return False
    try:
        with open(meta_path, "rb") as handle:
            group_meta = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return False
    group_identity = _normalize_semantic_generation_identity(group_meta if isinstance(group_meta, dict) else {})
    if not _semantic_generation_identity_complete(group_identity):
        return False
    return _semantic_generation_identity_matches(vector_identity, group_identity)


def get_relevant_context(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: str = None,
    top_k: int = 10,  # 增加到10
    use_rerank: bool = False,
    reranker_model: Optional[str] = None,
    candidate_k: int = 20,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None,
    selected_text: Optional[str] = None,  # 新增：用于查询改写中的指示代词解析
    model_context_window: int = 0,  # 动态 Token 预算：LLM 模型的上下文窗口大小
    answer_max_tokens: int = 0,  # 期望的输出 Token 数，用于上下文预算感知
    progress_callback: Optional[Callable[[dict], None]] = None,
    query_expansion_api_key: Optional[str] = None,
    query_expansion_model: str = "",
    query_expansion_provider: str = "",
    query_expansion_endpoint: str = "",
    visual_evidence: Optional[List[dict]] = None,
    intent_decision: Optional[dict] = None,
    query_is_canonical: bool = False,
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_api_host: Optional[str] = None,
) -> Tuple[str, dict]:
    """获取与查询相关的上下文文本和检索元数据

    集成 GranularitySelector、TokenBudgetManager、ContextBuilder 和 RetrievalLogger，
    实现混合粒度检索策略。当语义意群可用时，使用智能粒度选择和 Token 预算管理；
    否则回退到原有的简单拼接逻辑。

    Args:
        doc_id: 文档唯一标识
        query: 用户查询文本
        vector_store_dir: 向量索引存储目录
        pages: 文档页面数据列表
        api_key: API 密钥
        top_k: 返回结果数量
        use_rerank: 是否使用重排序
        reranker_model: 重排序模型
        candidate_k: 候选结果数量
        rerank_provider: 重排序提供商
        rerank_api_key: 重排序 API 密钥
        rerank_endpoint: 重排序端点

    Returns:
        (context_string, retrieval_meta) 元组
        - context_string: 格式化的上下文字符串
        - retrieval_meta: 检索元数据字典，包含 query_type、granularities、
          token_used、fallback、citations 等信息
    """
    decision = _resolve_intent_decision(query, intent_decision)
    query_type = str(decision.get("query_type") or "specific")
    evidence_need = list(decision.get("evidence_need") or [])
    page_scope_ranges = _normalize_intent_page_ranges(decision)

    # 延迟导入（仅首次触发模块加载，后续为字典查找）
    from services.semantic_group_service import SemanticGroupService
    from services.granularity_selector import GranularitySelector
    from services.token_budget import TokenBudgetManager

    # 获取搜索结果，解构返回的 (results, timings) 元组
    results, timings = search_document_chunks(
        doc_id,
        query,
        vector_store_dir=vector_store_dir,
        pages=pages,
        api_key=api_key,
        top_k=top_k,
        candidate_k=candidate_k,
        use_rerank=use_rerank,
        reranker_model=reranker_model,
        rerank_provider=rerank_provider,
        rerank_api_key=rerank_api_key,
        rerank_endpoint=rerank_endpoint,
        selected_text=selected_text,  # 传递 selected_text 用于查询改写
        progress_callback=progress_callback,
        query_expansion_api_key=query_expansion_api_key,
        query_expansion_model=query_expansion_model,
        query_expansion_provider=query_expansion_provider,
        query_expansion_endpoint=query_expansion_endpoint,
        visual_evidence=visual_evidence,
        intent_decision=decision,
        query_is_canonical=query_is_canonical,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_api_host=embedding_api_host,
    )

    config = _rag_config_singleton
    prefer_raw_chunk_context = "numeric_table" in evidence_need
    semantic_groups_current = (
        _semantic_groups_match_vector_index(doc_id, vector_store_dir)
        and not page_scope_ranges
    )

    # 动态 Token 预算：根据模型上下文窗口动态调整，同时扣除输出预留
    if config.token_budget_ratio > 0 and model_context_window > 0:
        raw_budget = int(model_context_window * config.token_budget_ratio)
        # 若已知期望输出长度，从总预算中扣除，为输出留足空间
        output_reserve = answer_max_tokens if answer_max_tokens > 0 else config.reserve_for_answer
        dynamic_budget = max(raw_budget - output_reserve, 2000)  # 最少 2000
        if dynamic_budget != config.max_token_budget:
            logger.info(
                f"[{doc_id}] 动态 Token 预算: {config.max_token_budget} → {dynamic_budget} "
                f"(模型窗口={model_context_window}, 比例={config.token_budget_ratio}, "
                f"输出预留={output_reserve})"
            )
            config.max_token_budget = dynamic_budget

    # 尝试使用语义意群增强检索
    if config.enable_semantic_groups and semantic_groups_current and not prefer_raw_chunk_context:
        try:
            context_str, retrieval_meta = _build_context_with_groups(
                doc_id=doc_id,
                query=query,
                results=results,
                config=config,
                vector_store_dir=vector_store_dir,
                timings=timings,
                intent_decision=decision,
            )
            if context_str is not None:
                context_str, retrieval_meta = _append_runtime_visual_overlay_group_context(
                    context_str,
                    retrieval_meta,
                    results,
                    doc_id=doc_id,
                    query=query,
                )
                retrieval_meta["query_type"] = retrieval_meta.get("query_type") or query_type
                retrieval_meta["evidence_need"] = list(evidence_need)
                retrieval_meta["search_query"] = query
                return context_str, retrieval_meta
        except Exception as e:
            # 意群增强失败，回退到简单拼接
            logger.warning(f"[{doc_id}] 意群增强检索失败，回退到简单拼接: {e}")
    elif config.enable_semantic_groups and not semantic_groups_current:
        logger.info(f"[{doc_id}] 当前向量代际没有匹配的语义意群，使用分块上下文")
    elif prefer_raw_chunk_context:
        logger.info(f"[{doc_id}] numeric_table 查询跳过语义意群摘要，直接使用原始 chunk 上下文")

    # 回退逻辑：numeric_table 走分层上下文投影，其他题型保留简单拼接去重。
    fallback_entries: List[tuple[dict, str]] = []
    for item in results:
        context_text = (_build_context_text_for_result(item, query=query) or "").strip()
        if not context_text:
            continue
        fallback_entries.append((item, context_text))

    if prefer_raw_chunk_context:
        layered_entries = _cleanup_numeric_table_context_entries(fallback_entries, query)
    else:
        seen_fallback_contexts: set[str] = set()
        layered_entries = []
        for item, context_text in fallback_entries:
            normalized_context = " ".join(context_text.split()).lower()
            dedupe_key = f"{int(item.get('page') or 0)}|{normalized_context}"
            if dedupe_key in seen_fallback_contexts:
                continue
            seen_fallback_contexts.add(dedupe_key)
            layered_entries.append(
                {
                    "item": item,
                    "text": context_text,
                    "context_role": "background",
                }
            )

    # P3.2 Hierarchical retrieval: 在 fallback 路径中用所属 semantic group 的 digest/full_text
    # 增强 layered_entries（小 chunk 命中 → 拼 parent group 上下文）
    _hier_stats = None
    # P3.2 升级：
    #   - 按 query_type 选择粒度：overview/analytical → full_text，其他 → digest
    #   - 单 group ≤ 6000 字符（防止单意群占满预算）
    #   - 总 hierarchical 升级体量 ≤ 18000 字符（提前截断）
    #   - 同 group 第二次出现仍保留原 chunk_text（避免重复）
    if not prefer_raw_chunk_context and semantic_groups_current:
        try:
            _hier_groups_dir = _get_semantic_groups_dir(doc_id)
            from services.semantic_group_service import SemanticGroupService as _SGS_h
            _hier_groups = _SGS_h().load_groups(doc_id, _hier_groups_dir)
            if _hier_groups:
                _hier_group_map = {g.group_id: g for g in _hier_groups}
                _hier_first_seen_groups: set = set()
                _hier_enriched_count = 0
                _hier_total_chars = 0
                _HIER_PER_GROUP_CAP = 6000
                _HIER_TOTAL_CAP = 18000
                # 按 query_type 决定升级粒度
                _prefer_full_text = query_type in ("overview", "analytical")
                # 第一遍：按出现顺序，为每个 group 第一次出现的 chunk 升级为 parent text
                for entry in layered_entries:
                    item = entry.get("item") or {}
                    group_id = (item.get("group_id") or "").strip()
                    if not group_id:
                        continue
                    g = _hier_group_map.get(group_id)
                    if not g:
                        continue
                    if _prefer_full_text:
                        parent_text = (g.full_text or g.digest or "").strip()
                        granularity_label = "full_text" if g.full_text else "digest"
                    else:
                        parent_text = (g.digest or g.full_text or "").strip()
                        granularity_label = "digest" if g.digest else "full_text"
                    if not parent_text:
                        continue
                    # 单 group 字符上限
                    if len(parent_text) > _HIER_PER_GROUP_CAP:
                        parent_text = parent_text[:_HIER_PER_GROUP_CAP] + "...(单意群截断)"
                    if group_id not in _hier_first_seen_groups:
                        # 总体预算检查
                        if _hier_total_chars + len(parent_text) > _HIER_TOTAL_CAP:
                            logger.info(
                                f"[{doc_id}] hierarchical fallback: 达到总预算上限 {_HIER_TOTAL_CAP} 字符，停止升级"
                            )
                            _hier_first_seen_groups.add(group_id)
                            continue
                        # 该 group 第一次出现：升级为 parent_text
                        if len(parent_text) > len(entry.get("text", "")):
                            entry["text"] = parent_text
                            entry["_parent_group_text"] = parent_text
                            entry["_parent_group_id"] = group_id
                            entry["_hierarchical_granularity"] = granularity_label
                            _hier_enriched_count += 1
                            _hier_total_chars += len(parent_text)
                        _hier_first_seen_groups.add(group_id)
                    # 同 group 的后续 chunks 保持 chunk_text 不变，避免冗余
                logger.info(
                    f"[{doc_id}] hierarchical fallback (P3.2): enriched {_hier_enriched_count} entries "
                    f"({len(_hier_first_seen_groups)} unique groups, {_hier_total_chars} chars, "
                    f"prefer_full_text={_prefer_full_text}, query_type={query_type})"
                )
                _hier_stats = {
                    "enriched_count": _hier_enriched_count,
                    "unique_groups": len(_hier_first_seen_groups),
                    "total_chars": _hier_total_chars,
                    "prefer_full_text": _prefer_full_text,
                    "per_group_cap": _HIER_PER_GROUP_CAP,
                    "total_cap": _HIER_TOTAL_CAP,
                }
        except Exception as e:
            logger.warning(f"[{doc_id}] hierarchical fallback 增强失败: {e}")

    relevant_chunks = [entry["text"] for entry in layered_entries]
    context_string = "\n\n...\n\n".join(relevant_chunks)

    # 回退路径也生成基本的 citations（基于 chunk 的页码信息）
    # numeric_table 查询优先把 exact row / typed cell 证据传给答案端和 context_segments。
    fallback_citations = []
    for idx, entry in enumerate(layered_entries):
        item = entry["item"]
        context_text = entry["text"]
        citation = _build_fallback_citation_from_result(
            item,
            idx + 1,
            query,
            context_text=context_text,
            context_role=entry.get("context_role", ""),
        )
        if citation:
            # P2.1 hierarchical：附加 context_segment_text 用 parent group 全文
            # _build_citation_context_text 优先使用 context_segment_text
            parent_text = entry.get("_parent_group_text")
            if parent_text:
                citation["context_segment_text"] = parent_text
                citation["source_text"] = parent_text
                citation["_full_text"] = parent_text
            fallback_citations.append(citation)

    # 质量阈值检查（需求 8.1, 8.4）
    low_relevance = False
    if results:
        max_similarity = max(r.get("similarity", 0.0) for r in results)
        if max_similarity < config.relevance_threshold:
            low_relevance = True
            low_relevance_hint = (
                "\n\n⚠️ 注意：以上检索结果与用户问题的相关度较低，"
                "文档中可能不包含与该问题直接相关的内容。"
                "请基于已有信息谨慎回答，并明确告知用户信息可能不够充分。"
            )
            context_string += low_relevance_hint
            logger.info(
                f"[{doc_id}] 回退路径检索结果质量低于阈值 "
                f"(max_similarity={max_similarity:.3f} < threshold={config.relevance_threshold})"
            )

    # 构建回退情况下的 retrieval_meta
    # 如果触发低质量阈值，优先记录 low_relevance（需求 8.3）
    if low_relevance:
        fallback_type = "low_relevance"
        fallback_detail = "所有检索结果相似度低于质量阈值"
    else:
        fallback_type = "groups_disabled" if not config.enable_semantic_groups else "index_missing"
        fallback_detail = f"回退到简单拼接逻辑，原因: {fallback_type}"
    trace = _RetrievalTrace(
        query=query,
        query_type=query_type,
        query_confidence=0.0,
        chunk_hits=len(results),
        group_hits=0,
        token_budget=config.max_token_budget,
        token_reserved=config.reserve_for_answer,
        token_used=0,
        fallback_type=fallback_type,
        fallback_detail=fallback_detail,
        citations=fallback_citations,
        max_relevance_score=max((r.get("similarity", 0.0) for r in results), default=-1.0),
    )
    _retrieval_logger_singleton.log_trace(trace)
    retrieval_meta = _retrieval_logger_singleton.to_retrieval_meta(trace)
    retrieval_meta["diagnostics"] = {
        "retrieval": _build_retrieval_diagnostics(results, query),
        "context_assembly": _build_context_assembly_diagnostics(
            results,
            context_string,
            token_budget=config.max_token_budget,
            hierarchical_stats=_hier_stats,
        ),
    }
    retrieval_meta["query_type"] = query_type
    retrieval_meta["evidence_need"] = list(evidence_need)
    retrieval_meta["search_query"] = query

    # 将检索耗时数据合并到 retrieval_meta（需求 1.2）
    retrieval_meta["timings"] = {k: v for k, v in timings.items() if not k.startswith("_")}

    # 传递原始 chunks 用于结构化引文匹配
    retrieval_meta["_chunks"] = [
        _copy_runtime_visual_provenance(entry["item"], {
            "text": entry["text"],
            "raw_text": entry["item"].get("chunk", ""),
            "page": entry["item"].get("page", 0),
            "group_id": _visual_overlay_group_id(entry["item"], f"chunk-{i}"),
            "context_id": entry["item"].get("context_id", ""),
            "evidence_id": entry["item"].get("evidence_id", ""),
            "block_id": entry["item"].get("block_id", ""),
            "chunk_id": entry["item"].get("chunk_id"),
            "parent_id": entry["item"].get("parent_id"),
            "doc_id": entry["item"].get("doc_id", doc_id),
            "chunk_type": entry["item"].get("chunk_type", ""),
            "block_type": entry["item"].get("block_type", entry["item"].get("chunk_type", "")),
            "chunk_heading": entry["item"].get("chunk_heading", ""),
            "section_path": entry["item"].get("section_path", entry["item"].get("chunk_heading", "")),
            "table_id": entry["item"].get("table_id", ""),
            "table_instance_id": entry["item"].get("table_instance_id", ""),
            "table_source_hash": entry["item"].get("table_source_hash", ""),
            "table_caption": entry["item"].get("numeric_table_exact_context_caption") or entry["item"].get("table_caption", ""),
            "table_header": entry["item"].get("numeric_table_exact_context_header") or entry["item"].get("table_header", ""),
            "numeric_table_exact_context_row_text": entry["item"].get("numeric_table_exact_context_row_text", ""),
            "numeric_table_exact_context_caption": entry["item"].get("numeric_table_exact_context_caption", ""),
            "numeric_table_exact_context_header": entry["item"].get("numeric_table_exact_context_header", ""),
            "evidence_units": entry["item"].get("evidence_units", []),
            "cell_evidence_units": entry["item"].get("cell_evidence_units", []),
            "context_role": entry.get("context_role", ""),
            **_citation_anchor_metadata_from_result(entry["item"]),
        })
        for i, entry in enumerate(layered_entries)
    ]

    # 回退路径：chunk 即为 LLM 看到的上下文段，直接复用
    retrieval_meta["_context_segments"] = [
        _copy_runtime_visual_provenance(entry["item"], {
            "ref": idx + 1,
            "evidence_id": entry["item"].get("evidence_id") or f"{doc_id}:fallback:{idx + 1}",
            "doc_id": doc_id,
            "context_id": entry["item"].get("context_id", ""),
            "block_id": entry["item"].get("block_id", ""),
            "chunk_id": entry["item"].get("chunk_id"),
            "text": entry["text"],
            "page_range": entry["item"].get("page_range") or [entry["item"].get("page", 0), entry["item"].get("page", 0)],
            "group_id": _visual_overlay_group_id(entry["item"], f"chunk-{idx}"),
            "modality": entry["item"].get("modality") or entry["item"].get("chunk_type") or "text",
            "chunk_type": entry["item"].get("chunk_type", ""),
            "block_type": entry["item"].get("block_type", entry["item"].get("chunk_type", "")),
            "table_id": entry["item"].get("table_id", ""),
            "table_bundle_id": entry["item"].get("table_bundle_id", ""),
            "table_instance_id": entry["item"].get("table_instance_id", ""),
            "table_source_hash": entry["item"].get("table_source_hash", ""),
            "evidence_unit_id": entry["item"].get("evidence_unit_id", ""),
            "table_caption": entry["item"].get("numeric_table_exact_context_caption") or entry["item"].get("table_caption", ""),
            "table_header": entry["item"].get("numeric_table_exact_context_header") or entry["item"].get("table_header", ""),
            "numeric_table_exact_context_row_text": entry["item"].get("numeric_table_exact_context_row_text", ""),
            "numeric_table_exact_context_caption": entry["item"].get("numeric_table_exact_context_caption", ""),
            "numeric_table_exact_context_header": entry["item"].get("numeric_table_exact_context_header", ""),
            "table_row_evidence": entry["item"].get("table_row_evidence", False),
            "table_row_slice_kind": entry["item"].get("table_row_slice_kind", ""),
            "score": entry["item"].get("similarity", entry["item"].get("score", 0.0)),
            "context_role": entry.get("context_role", ""),
            **_citation_anchor_metadata_from_result(entry["item"]),
        })
        for idx, entry in enumerate(layered_entries)
    ]

    if page_scope_ranges:
        retrieval_meta["page_scope"] = {
            "ranges": [list(item) for item in page_scope_ranges],
            "enforced": True,
            "result_count": len(results),
        }
    return context_string, retrieval_meta


def _extract_primary_cell_evidence_units(item: dict) -> List[dict]:
    cell_evidence_units = item.get("cell_evidence_units")
    if isinstance(cell_evidence_units, list) and cell_evidence_units:
        return [dict(unit) for unit in cell_evidence_units if isinstance(unit, dict)]

    evidence_units = item.get("evidence_units")
    if not isinstance(evidence_units, list):
        return []

    for unit in evidence_units:
        if not isinstance(unit, dict):
            continue
        if (unit.get("evidence_unit_type") or "").strip().lower() != "table_row":
            continue
        row_cells = unit.get("cell_evidence_units")
        if isinstance(row_cells, list) and row_cells:
            return [dict(cell) for cell in row_cells if isinstance(cell, dict)]
    return []


def _build_fallback_citation_from_result(
    item: dict,
    ref: int,
    query: str,
    context_text: Optional[str] = None,
    context_role: str = "",
) -> Optional[dict]:
    if not isinstance(item, dict):
        return None

    chunk_text = (context_text or _build_context_text_for_result(item, query=query) or "").strip()
    if not chunk_text:
        return None

    page = item.get("page", 0)
    chunk_type = (item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    page_range = item.get("page_range") or [page, page]

    exact_row_text = re.sub(
        r"\s+",
        " ",
        str(
            item.get("numeric_table_exact_context_row_text")
            or _get_numeric_table_boundary_text(item)
            or item.get("table_row_raw_text")
            or ""
        ),
    ).strip()
    if not exact_row_text and chunk_type == "table_row":
        exact_row_text = re.sub(
            r"\s+",
            " ",
            str(item.get("raw_chunk_text") or item.get("chunk") or ""),
        ).strip()

    has_typed_table_evidence = bool(
        chunk_type in {"table_row", "table", "caption", "table_cell"}
        or item.get("table_row_evidence")
        or item.get("table_row_slice_kind") == "exact"
        or item.get("numeric_table_exact_context_row_text")
        or item.get("evidence_units")
        or item.get("cell_evidence_units")
    )
    display_text = exact_row_text if has_typed_table_evidence and exact_row_text else chunk_text
    highlight_text = (
        display_text
        if has_typed_table_evidence and display_text
        else _context_builder_singleton._extract_relevant_snippet(chunk_text, query, max_len=200)
    )
    if not highlight_text:
        highlight_text = display_text[:200]

    citation = _copy_runtime_visual_provenance(item, {
        "ref": ref,
        "evidence_id": item.get("evidence_unit_id") or item.get("evidence_id") or f"chunk-{ref - 1}:{page}",
        "context_id": item.get("context_id", ""),
        "block_id": item.get("block_id", ""),
        "chunk_id": item.get("chunk_id"),
        "group_id": _visual_overlay_group_id(item, f"chunk-{ref - 1}"),
        "page_range": page_range,
        "source_text": chunk_text,
        "display_text": display_text,
        "highlight_text": highlight_text,
        "_full_text": chunk_text,
        "alignment_status": "candidate",
        "retrieval_type": "vector",
        "chunk_type": item.get("chunk_type", ""),
        "block_type": item.get("block_type", item.get("chunk_type", "")),
        "table_id": item.get("table_id", ""),
        "table_instance_id": item.get("table_instance_id", ""),
        "table_source_hash": item.get("table_source_hash", ""),
        "table_caption": item.get("numeric_table_exact_context_caption") or item.get("table_caption", ""),
        "table_header": item.get("numeric_table_exact_context_header") or item.get("table_header", ""),
        "numeric_table_exact_context_row_text": item.get("numeric_table_exact_context_row_text", ""),
        "numeric_table_exact_context_caption": item.get("numeric_table_exact_context_caption", ""),
        "numeric_table_exact_context_header": item.get("numeric_table_exact_context_header", ""),
        "evidence_units": item.get("evidence_units", []),
        "cell_evidence_units": _extract_primary_cell_evidence_units(item),
        "context_role": context_role,
    })
    citation.update(_citation_anchor_metadata_from_result(item))
    if has_typed_table_evidence:
        citation["context_segment_text"] = chunk_text
    return citation


def _format_layered_context(
    context_string: str,
    fitted_selections: list,
    raw_chunks: List[dict],
    query_type: str,
) -> Tuple[str, dict]:
    """P3.4 三层上下文格式化（借鉴 paper-burner-x streaming-multi-hop）

    将 ContextBuilder 输出的 context_string 重新组织为：
    - 【🎯 命中片段】top-3 raw chunk（精确证据）
    - 【📖 重点意群】granularity=full 的意群（完整章节）
    - 【📋 背景意群】granularity=digest/summary 的意群（关联摘要）

    仅在 query_type ∈ {overview, analytical} 且存在多种粒度时生效；
    保留原有 [N] 引用编号，不破坏 citation 链路。

    Returns:
        (formatted_context, layer_stats)
        layer_stats: {"layered": bool, "n_focus": int, "n_background": int, "n_chunks": int}
    """
    layer_stats = {"layered": False, "n_focus": 0, "n_background": 0, "n_chunks": 0}
    if query_type not in ("overview", "analytical"):
        return context_string, layer_stats
    if not fitted_selections or not context_string:
        return context_string, layer_stats

    pattern = re.compile(r"^\[(\d+)\]【.*?】", re.MULTILINE)
    matches = list(pattern.finditer(context_string))
    if not matches:
        return context_string, layer_stats

    blocks = []
    for i, m in enumerate(matches):
        ref = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context_string)
        blocks.append({"ref": ref, "text": context_string[start:end].rstrip()})

    gran_by_ref = {idx + 1: sel.get("granularity", "full") for idx, sel in enumerate(fitted_selections)}
    focus_blocks = [b for b in blocks if gran_by_ref.get(b["ref"]) in ("full",)]
    background_blocks = [b for b in blocks if gran_by_ref.get(b["ref"]) in ("digest", "summary")]

    # 至少需要两种粒度才分层；否则保留原结构
    if not focus_blocks or not background_blocks:
        return context_string, layer_stats

    parts: list[str] = []
    # Layer 1: 命中片段（top-3 raw chunks）
    if raw_chunks:
        top_chunks = []
        seen_chunks: set = set()
        for r in raw_chunks[:8]:
            ct = (r.get("chunk") or "").strip() if isinstance(r, dict) else ""
            if not ct:
                continue
            key = ct[:120]
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            page = r.get("page", 0) if isinstance(r, dict) else 0
            preview = ct[:400] + ("..." if len(ct) > 400 else "")
            top_chunks.append(f"(p.{page}) {preview}")
            if len(top_chunks) >= 3:
                break
        if top_chunks:
            parts.append(
                "【🎯 命中片段】(检索 top-3 chunk，作为精确证据)\n\n"
                + "\n\n".join(top_chunks)
            )
            layer_stats["n_chunks"] = len(top_chunks)

    parts.append(
        "【📖 重点意群】(query 命中的完整章节内容)\n\n"
        + "\n\n".join(b["text"] for b in focus_blocks)
    )
    parts.append(
        "【📋 背景意群】(关联章节摘要，提供全局视角)\n\n"
        + "\n\n".join(b["text"] for b in background_blocks)
    )

    layer_stats.update({
        "layered": True,
        "n_focus": len(focus_blocks),
        "n_background": len(background_blocks),
    })
    return "\n\n---\n\n".join(parts), layer_stats

_STRUCTURAL_EVIDENCE_QUERY_RE = re.compile(
    r"架构|结构|拓扑|交互|机制|流程|网络|模块|"
    r"architecture|structure|topology|interaction|mechanism|pipeline|network|module",
    re.IGNORECASE,
)


def _build_direct_hit_evidence_by_group(
    fitted_selections: list[dict],
    group_best_chunk_meta: dict,
    *,
    query: str,
    evidence_need: list[str],
) -> dict[str, str]:
    """为结构性问题保留少量原始命中，避免意群摘要压掉关键拓扑句。"""
    needs_direct_evidence = bool(
        {str(item or "").strip() for item in evidence_need}
        & {"section_explanation", "analysis_explanation"}
    ) or bool(_STRUCTURAL_EVIDENCE_QUERY_RE.search(str(query or "")))
    if not needs_direct_evidence or not isinstance(group_best_chunk_meta, dict):
        return {}

    direct_evidence: dict[str, str] = {}
    used_chars = 0
    max_groups = 8
    max_per_group = 720
    max_total = 5000
    for selection in fitted_selections:
        if len(direct_evidence) >= max_groups or used_chars >= max_total:
            break
        group = selection.get("group") if isinstance(selection, dict) else None
        group_id = str(getattr(group, "group_id", "") or "").strip()
        if not group_id:
            continue
        metadata = group_best_chunk_meta.get(group_id) or {}
        direct_text = str(metadata.get("_direct_context_text") or "").strip()
        if not direct_text:
            continue

        granularity = str(selection.get("granularity") or "full")
        text_attr = {"full": "full_text", "digest": "digest", "summary": "summary"}.get(
            granularity, "full_text"
        )
        group_text = str(getattr(group, text_attr, "") or "")
        if direct_text in group_text:
            continue
        if len(direct_text) > max_per_group:
            direct_text = (
                _context_builder_singleton._extract_relevant_snippet(
                    direct_text, query, max_len=max_per_group
                )
                or direct_text[:max_per_group]
            )
        remaining = max_total - used_chars
        if remaining < 120:
            break
        direct_text = direct_text[:remaining].strip()
        if not direct_text:
            continue
        direct_evidence[group_id] = direct_text
        used_chars += len(direct_text)

    return direct_evidence

def _build_context_with_groups(
    doc_id: str,
    query: str,
    results: List[dict],
    config,
    vector_store_dir: str = None,
    timings: dict = None,
    intent_decision: Optional[dict] = None,
) -> Tuple[Optional[str], dict]:
    """使用语义意群构建增强上下文

    流程：
    1. 加载语义意群数据
    2. 加载分块数据（用于 chunk_indices 精确映射）
    3. 使用 GranularitySelector.select_mixed 分配混合粒度
    4. 使用 TokenBudgetManager.fit_within_budget 调整 Token 预算
    5. 使用 ContextBuilder.build_context 构建格式化上下文
    6. 使用 RetrievalLogger 记录检索追踪

    Args:
        doc_id: 文档唯一标识
        query: 用户查询文本
        results: search_document_chunks 返回的搜索结果
        config: RAGConfig 配置对象
        vector_store_dir: 向量索引存储目录（可选），用于加载分块数据以支持 chunk_indices 精确映射
        timings: search_document_chunks 返回的各阶段耗时字典（可选），将合并到 retrieval_meta 中

    Returns:
        (context_string, retrieval_meta) 元组，如果意群不可用返回 (None, {})
    """
    from services.semantic_group_service import SemanticGroupService
    from services.granularity_selector import GranularitySelector
    from services.token_budget import TokenBudgetManager
    from services.context_builder import ContextBuilder
    from services.retrieval_logger import RetrievalLogger, RetrievalTrace

    # 步骤 1：加载语义意群数据
    groups_store_dir = _get_semantic_groups_dir(doc_id)
    group_service = SemanticGroupService()
    groups = group_service.load_groups(doc_id, groups_store_dir)

    if not groups:
        logger.info(f"[{doc_id}] 语义意群数据不可用，回退到简单拼接")
        return None, {}

    logger.info(f"[{doc_id}] 已加载 {len(groups)} 个语义意群，开始构建增强上下文")

    # 步骤 1.5：加载分块数据，用于 chunk_indices 精确映射
    chunks = None
    if vector_store_dir:
        try:
            chunks_path = os.path.join(vector_store_dir, f"{doc_id}.pkl")
            if os.path.exists(chunks_path):
                with open(chunks_path, "rb") as f:
                    chunks_data = pickle.load(f)
                if isinstance(chunks_data, dict):
                    chunks = chunks_data.get("chunks", None)
                else:
                    chunks = chunks_data
                if chunks:
                    logger.info(f"[{doc_id}] 已加载 {len(chunks)} 个分块，用于 chunk_indices 精确映射")
        except Exception as e:
            logger.warning(f"[{doc_id}] 加载分块数据失败，回退到子串匹配: {e}")
            chunks = None

    # 步骤 2：根据搜索结果对意群进行排序
    # 将搜索结果中的 chunk 映射回对应的意群，按 RRF/相关性排序
    group_best_chunk_meta: dict = {}
    ranked_groups, group_best_chunks = _rank_groups_by_results(
        groups,
        results,
        chunks=chunks,
        best_chunk_meta_out=group_best_chunk_meta,
    )

    if not ranked_groups:
        logger.info(f"[{doc_id}] 无法将搜索结果映射到意群，回退到简单拼接")
        return None, {}

    # 步骤 3：使用 GranularitySelector 分配混合粒度
    selector = GranularitySelector()

    # 先获取查询类型对应的最大意群数限制
    selection_info = selector.select(
        query=query,
        groups=groups,
        max_tokens=config.max_token_budget,
        intent_decision=intent_decision,
    )
    # P0-A fix: extraction 严格遵守 max_groups 上限，不允许 len(results) 撑大;
    # 其他题型保留原有 max() 逻辑（允许 results 数量推高上限）
    if selection_info.query_type == "extraction":
        max_groups = selection_info.max_groups
    else:
        max_groups = max(selection_info.max_groups, len(results))

    # 截断排序后的意群列表，避免引入过多低相关性意群
    ranked_groups_limited = ranked_groups[:max_groups]

    mixed_selections = selector.select_mixed(
        query=query,
        ranked_groups=ranked_groups_limited,
        max_tokens=config.max_token_budget,
        intent_decision=intent_decision,
    )

    # 步骤 4：使用 TokenBudgetManager 调整 Token 预算
    # P0-B: 按题型设差异化 Token 上限，避免 extraction 塞进 5000+ token
    _TYPE_TOKEN_CAP = {"extraction": 2000, "specific": 3500, "analytical": 5000, "overview": 7000}
    effective_budget = min(
        config.max_token_budget,
        _TYPE_TOKEN_CAP.get(selection_info.query_type, config.max_token_budget),
    )
    budget_manager = TokenBudgetManager(
        max_tokens=effective_budget,
        reserve_for_answer=config.reserve_for_answer,
    )
    fitted_selections = budget_manager.fit_within_budget(mixed_selections)
    evidence_need_for_context = list((intent_decision or {}).get("evidence_need") or [])
    if not isinstance(intent_decision, dict):
        evidence_need_for_context = _analyze_evidence_need(query)
    direct_hit_evidence_by_group = _build_direct_hit_evidence_by_group(
        fitted_selections,
        group_best_chunk_meta,
        query=query,
        evidence_need=evidence_need_for_context,
    )

    # 步骤 5：使用 ContextBuilder 构建格式化上下文
    context_builder = ContextBuilder()
    context_string, citations = context_builder.build_context(
        fitted_selections,
        group_best_chunks=group_best_chunks,
        group_best_chunk_meta=group_best_chunk_meta,
        direct_evidence_by_group=direct_hit_evidence_by_group,
        query=query,
    )

    # P3.4 三层上下文格式化（仅 overview/analytical 启用，多粒度时生效）
    # 通过 settings.enable_p34_layered_context 总开关控制（默认 ON，仅 ablation 时关闭）
    layered_context_stats = {"layered": False}
    try:
        from config import settings as _global_settings
        _p34_enabled = getattr(_global_settings, "enable_p34_layered_context", True)
    except Exception:
        _p34_enabled = True
    if _p34_enabled:
        try:
            formatted_context, layered_context_stats = _format_layered_context(
                context_string=context_string,
                fitted_selections=fitted_selections,
                raw_chunks=results,
                query_type=selection_info.query_type,
            )
            if layered_context_stats.get("layered"):
                context_string = formatted_context
                logger.info(
                    f"[{doc_id}] P3.4 三层上下文格式化生效: "
                    f"focus={layered_context_stats.get('n_focus', 0)}, "
                    f"background={layered_context_stats.get('n_background', 0)}, "
                    f"chunks={layered_context_stats.get('n_chunks', 0)}"
                )
        except Exception as _e_layered:
            logger.warning(f"[{doc_id}] P3.4 三层上下文格式化失败: {_e_layered}")
    else:
        logger.debug(f"[{doc_id}] P3.4 layered context 已 ablation 关闭")

    # 步骤 6：计算实际使用的 Token 数
    token_used = sum(item.get("tokens", 0) for item in fitted_selections)

    # 步骤 6.5：检索结果质量阈值检查（需求 8.1, 8.4）
    low_relevance = False
    if results:
        max_similarity = max(r.get("similarity", 0.0) for r in results)
        if max_similarity < config.relevance_threshold:
            low_relevance = True
            low_relevance_hint = (
                "\n\n⚠️ 注意：以上检索结果与用户问题的相关度较低，"
                "文档中可能不包含与该问题直接相关的内容。"
                "请基于已有信息谨慎回答，并明确告知用户信息可能不够充分。"
            )
            context_string += low_relevance_hint
            logger.info(
                f"[{doc_id}] 检索结果质量低于阈值 "
                f"(max_similarity={max_similarity:.3f} < threshold={config.relevance_threshold})"
            )

    # 步骤 7：使用 RetrievalLogger 记录检索追踪
    # 查询类型已在步骤 3 中获取（selection_info）

    evidence_need = list((intent_decision or {}).get("evidence_need") or [])
    if not isinstance(intent_decision, dict):
        evidence_need = _analyze_evidence_need(query)
    retrieval_logger = RetrievalLogger()
    trace = RetrievalTrace(
        query=query,
        query_type=selection_info.query_type,
        query_confidence=1.0,
        chunk_hits=len(results),
        group_hits=len(ranked_groups),
        rrf_top_k=[
            {"group_id": g.group_id, "rank": i, "source": "rrf"}
            for i, g in enumerate(ranked_groups[:10])
        ],
        token_budget=config.max_token_budget,
        token_reserved=config.reserve_for_answer,
        token_used=token_used,
        granularity_assignments=[
            {"group_id": item["group"].group_id, "granularity": item["granularity"]}
            for item in fitted_selections
        ],
        fallback_type="low_relevance" if low_relevance else None,
        fallback_detail="所有检索结果相似度低于质量阈值" if low_relevance else None,
        citations=citations,
        max_relevance_score=max((r.get("similarity", 0.0) for r in results), default=-1.0),
    )
    retrieval_logger.log_trace(trace)
    retrieval_meta = retrieval_logger.to_retrieval_meta(trace)
    retrieval_meta["diagnostics"] = {
        "retrieval": _build_retrieval_diagnostics(results, query),
        "context_assembly": _build_context_assembly_diagnostics(
            results,
            context_string,
            token_budget=config.max_token_budget,
        ),
    }
    if layered_context_stats.get("layered"):
        retrieval_meta["diagnostics"]["layered_context"] = layered_context_stats
    retrieval_meta["evidence_need"] = list(evidence_need)
    retrieval_meta["search_query"] = query

    # 将检索耗时数据合并到 retrieval_meta（需求 1.2）
    if timings is not None:
        retrieval_meta["timings"] = {k: v for k, v in timings.items() if not k.startswith("_")}

    # 传递原始 chunks 用于结构化引文匹配
    retrieval_meta["_chunks"] = [
        _copy_runtime_visual_provenance(item, {
            "text": item.get("chunk", ""),
            "page": item.get("page", 0),
            "group_id": _visual_overlay_group_id(item, ""),
            "context_id": item.get("context_id", ""),
            "evidence_id": item.get("evidence_id", ""),
            "block_id": item.get("block_id", ""),
            "chunk_id": item.get("chunk_id"),
            "parent_id": item.get("parent_id"),
            "doc_id": item.get("doc_id", doc_id),
            "chunk_type": item.get("chunk_type", ""),
            "block_type": item.get("block_type", item.get("chunk_type", "")),
            "chunk_heading": item.get("chunk_heading", ""),
            "section_path": item.get("section_path", item.get("chunk_heading", "")),
            **_citation_anchor_metadata_from_result(item),
        })
        for item in results
    ]

    # 传递意群级上下文段（LLM 实际看到的文本），并保留最佳 chunk 的定位锚点。
    citation_by_group = {
        citation.get("group_id"): citation
        for citation in citations
        if isinstance(citation, dict) and citation.get("group_id")
    }
    _context_segments = []
    for idx, selection in enumerate(fitted_selections):
        group = selection["group"]
        granularity = selection.get("granularity", "full")
        text_attr = {"full": "full_text", "digest": "digest", "summary": "summary"}.get(granularity, "full_text")
        citation = citation_by_group.get(group.group_id, {})
        text = str(citation.get("source_text") or getattr(group, text_attr, "") or "")
        segment = {
            "ref": idx + 1,
            "evidence_id": f"{doc_id}:group:{idx + 1}",
            "doc_id": doc_id,
            "chunk_id": None,
            "text": text,
            "page_range": [group.page_range[0], group.page_range[1]],
            "group_id": group.group_id,
            "modality": "text",
            "score": selection.get("score", 0.0),
        }
        segment.update(_citation_anchor_metadata_from_result(citation))
        _context_segments.append(segment)
    retrieval_meta["_context_segments"] = _context_segments

    logger.info(
        f"[{doc_id}] 增强上下文构建完成: "
        f"意群数={len(fitted_selections)}, "
        f"Token 使用={token_used}/{budget_manager.available_tokens}, "
        f"查询类型={selection_info.query_type}"
    )

    return context_string, retrieval_meta


def _rank_groups_by_results(
    groups: list,
    results: List[dict],
    chunks: List[str] = None,
    best_chunk_meta_out: Optional[dict] = None,
) -> tuple:
    """根据搜索结果对语义意群进行排序

    优先使用 chunk_indices 反向映射进行精确匹配（O(1) 查找），
    当 chunks 参数不可用或匹配失败时，回退到子串匹配作为兜底策略。

    Args:
        groups: 语义意群列表
        results: search_document_chunks 返回的搜索结果
        chunks: 文档的所有文本分块列表（可选），用于构建 chunk_text -> chunk_index 映射
        best_chunk_meta_out: 可选输出映射，保留 group_id 对应最佳检索结果的定位元数据

    Returns:
        (ranked_groups, group_best_chunks) 元组
        - ranked_groups: 按相关性排序的语义意群列表（最相关的在前）
        - group_best_chunks: dict，group_id -> 最佳匹配的 chunk 文本（用于精确引用高亮）
    """
    if not groups or not results:
        if isinstance(best_chunk_meta_out, dict):
            best_chunk_meta_out.clear()
        return [], {}

    # 构建 chunk_index → group 的反向映射（基于意群的 chunk_indices 字段）
    chunk_idx_to_group = {}
    for group in groups:
        for idx in group.chunk_indices:
            chunk_idx_to_group[idx] = group

    # 构建 chunk_text → chunk_index 的映射（用于从搜索结果定位 chunk 索引）
    chunk_text_to_idx = {}
    if chunks:
        for i, text in enumerate(chunks):
            chunk_text_to_idx[text] = i

    group_scores = {}  # group_id -> 最佳排名（越小越好）
    group_similarity = {}  # group_id -> 最佳相似度分数
    group_best_chunks = {}  # group_id -> 最佳匹配的 chunk 文本（用于精确引用高亮）
    group_best_results = {}  # group_id -> 最佳匹配的原始检索结果

    for rank, result in enumerate(results):
        chunk_text = result.get("chunk", "")
        if not chunk_text:
            continue

        # 获取该 chunk 的相似度分数
        similarity = result.get("similarity", 0.0)

        matched_group = None

        # 优先通过 chunk_index 精确匹配（O(1) 查找）
        chunk_idx = chunk_text_to_idx.get(chunk_text)
        if chunk_idx is not None:
            matched_group = chunk_idx_to_group.get(chunk_idx)

        # 回退到子串匹配作为兜底策略
        if matched_group is None:
            for group in groups:
                if chunk_text in group.full_text:
                    matched_group = group
                    break

        if matched_group:
            gid = matched_group.group_id
            if gid not in group_scores:
                group_scores[gid] = rank
                group_similarity[gid] = similarity
                # 记录该意群最佳匹配的 chunk 文本（相似度最高的那个）
                group_best_chunks[gid] = chunk_text
                group_best_results[gid] = result
            else:
                # 保留最高排名（最小的 rank 值）
                if rank < group_scores[gid]:
                    group_scores[gid] = rank
                # 保留最高相似度，同时更新最佳 chunk 文本
                if similarity > group_similarity[gid]:
                    group_similarity[gid] = similarity
                    group_best_chunks[gid] = chunk_text
                    group_best_results[gid] = result

    # 过滤掉相关性过低的意群
    # 策略：如果最佳意群的相似度 > 0.5，则过滤掉相似度低于最佳值 50% 的意群
    # （从 30% 提升至 50%，避免引入不相关的意群）
    if group_similarity:
        best_similarity = max(group_similarity.values())
        if best_similarity > 0.5:
            threshold = best_similarity * 0.5
            filtered_ids = {
                gid for gid, sim in group_similarity.items()
                if sim >= threshold
            }
            removed = set(group_scores.keys()) - filtered_ids
            if removed:
                logger.info(
                    f"相关性过滤：移除 {len(removed)} 个低相关意群 "
                    f"(阈值={threshold:.3f}, 最佳={best_similarity:.3f})"
                )
            group_scores = {gid: r for gid, r in group_scores.items() if gid in filtered_ids}
            # 同步清理 group_best_chunks
            group_best_chunks = {gid: t for gid, t in group_best_chunks.items() if gid in filtered_ids}
            group_best_results = {
                gid: item for gid, item in group_best_results.items()
                if gid in filtered_ids
            }

    # 按排名排序意群
    sorted_group_ids = sorted(group_scores.keys(), key=lambda gid: group_scores[gid])

    # 构建 group_id -> group 对象的映射
    group_map = {g.group_id: g for g in groups}

    ranked_groups = [group_map[gid] for gid in sorted_group_ids if gid in group_map]
    if isinstance(best_chunk_meta_out, dict):
        best_chunk_meta_out.clear()
        for group_id, item in group_best_results.items():
            metadata = _citation_anchor_metadata_from_result(item)
            direct_context_text = (_build_context_text_for_result(item) or "").strip()
            if direct_context_text:
                metadata["_direct_context_text"] = direct_context_text
            best_chunk_meta_out[group_id] = metadata

    return ranked_groups, group_best_chunks
