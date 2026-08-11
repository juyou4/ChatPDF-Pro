"""GraphRAG 主类

适配 Chatpdf 后端架构的 GraphRAG 实现。
使用 Chatpdf 现有的 chat_service.call_ai_api 进行 LLM 调用，
支持通过配置项灵活切换 LLM 提供商和模型。
"""

from __future__ import annotations

import asyncio
import os
import json
import logging
import hashlib
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Type, cast, List, Union, Optional

from ._op import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
    local_query,
    global_query,
    hybrid_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    wrap_embedding_func_with_attrs,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

logger = logging.getLogger(__name__)
_GRAPHRAG_EMBEDDING_IDENTITY_VERSION = 1
# GraphRAG originally accepted a flattened document string only.  Version 2
# makes the graph keep block-index provenance on every generated text unit.
# It is part of the persisted config hash so legacy full-text graphs rebuild
# instead of silently surviving a structured-input upgrade.
GRAPHRAG_EVIDENCE_INPUT_VERSION = 2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


# ── 构建进度追踪 ──

@dataclass
class BuildProgress:
    """GraphRAG 构建进度与元数据"""
    # 绑定主解析代际，防止同一 doc_id 的旧任务污染新文档状态
    parse_generation: str = ""
    document_source_hash: str = ""
    # 构建状态: pending / building / done / failed
    status: str = "pending"
    # 当前阶段: chunking / extracting / clustering / reporting / persisting
    stage: str = ""
    # 进度百分比（0-100）
    progress: int = 0
    # 最近错误信息
    last_error: str = ""
    # 构建开始时间
    build_start: str = ""
    # 构建完成时间
    build_end: str = ""
    # 构建/查询使用的模型
    model: str = ""
    # LLM provider 与 endpoint（不含密钥）
    provider: str = ""
    endpoint: str = ""
    # embedding 模型
    embedding_model: str = ""
    # embedding provider 与 endpoint（不含密钥）
    embedding_provider: str = ""
    embedding_endpoint: str = ""
    embedding_identity_version: int = 0
    # embedding 维度
    embedding_dim: int = 0
    # 配置哈希（用于检测配置变更后是否需要重建）
    config_hash: str = ""
    # 实体数
    num_nodes: int = 0
    # 关系数
    num_edges: int = 0
    # 文档数
    num_docs: int = 0
    # 块数
    num_chunks: int = 0
    # 输入合同：用于诊断图谱是否保留了 block-index 证据身份。
    evidence_input_version: int = 0
    input_source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BuildProgress":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _compute_config_hash(config: GraphRAGConfig, chunk_token_size: int, max_gleaning: int) -> str:
    """计算配置哈希，用于检测配置变更后是否需要重建"""
    hash_input = json.dumps({
        "model": config.model,
        "provider": config.provider,
        "endpoint": config.endpoint,
        "embedding_model": config.embedding_model,
        "embedding_provider": config.embedding_provider,
        "embedding_endpoint": config.embedding_endpoint,
        "embedding_identity_version": _GRAPHRAG_EMBEDDING_IDENTITY_VERSION,
        "evidence_input_version": GRAPHRAG_EVIDENCE_INPUT_VERSION,
        "embedding_dim": config.embedding_dim,
        "chunk_token_size": chunk_token_size,
        "max_gleaning": max_gleaning,
    }, sort_keys=True)
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """获取或创建事件循环"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        logger.info("[GraphRAG] 在子线程中创建新的事件循环")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


@dataclass
class GraphRAGConfig:
    """GraphRAG 配置（独立于 AppSettings，便于按文档实例化）"""
    # LLM 配置（通过 Chatpdf 的 call_ai_api 调用）
    api_key: str = ""
    model: str = ""
    provider: str = ""
    endpoint: str = ""
    # 可选：使用独立的廉价模型做实体摘要
    cheap_model: str = ""
    cheap_provider: str = ""
    cheap_endpoint: str = ""
    # Embedding 配置
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_provider: str = ""
    embedding_endpoint: str = ""
    # Callers must bind the dimension resolved for the selected embedding
    # identity. A zero default deliberately fails in ``GraphRAG.__post_init__``
    # instead of guessing a historical OpenAI-sized vector.
    embedding_dim: int = 0

    def safe_hash_dict(self) -> dict:
        """返回不含 api_key 的配置字典，用于哈希和展示"""
        return {
            "model": self.model,
            "provider": self.provider,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
        }


def _compose_provider_scoped_embedding_model(model: str, provider: str) -> str:
    raw_model = str(model or "").strip()
    raw_provider = str(provider or "").strip()
    if not raw_model or not raw_provider or ":" in raw_model:
        return raw_model
    return f"{raw_provider}:{raw_model}"


def _require_positive_embedding_dim(value: object, *, context: str) -> int:
    """Reject missing or malformed dimensions before any vector storage is created."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{context} 必须是正整数")
    try:
        dimension = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} 必须是正整数") from exc
    if dimension <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{context} 必须是正整数")
    return dimension


def _has_complete_embedding_identity(progress: BuildProgress | None) -> bool:
    if not isinstance(progress, BuildProgress):
        return False
    if int(getattr(progress, "embedding_identity_version", 0) or 0) != _GRAPHRAG_EMBEDDING_IDENTITY_VERSION:
        return False
    provider = str(getattr(progress, "embedding_provider", "") or "").strip().lower()
    model = str(getattr(progress, "embedding_model", "") or "").strip()
    endpoint = str(getattr(progress, "embedding_endpoint", "") or "").strip()
    try:
        embedding_dim = int(getattr(progress, "embedding_dim", 0) or 0)
    except (TypeError, ValueError):
        embedding_dim = 0
    if not provider or not model:
        return False
    if provider != "local" and not endpoint:
        return False
    return embedding_dim > 0


async def _chatpdf_llm_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    api_key: str = "",
    model: str = "",
    provider: str = "",
    endpoint: str = "",
    hashing_kv: BaseKVStorage = None,
    **kwargs,
) -> str:
    """通过 Chatpdf 的 call_ai_api 调用 LLM"""
    from services.chat_service import call_ai_api
    from services.completion_outcome import resolve_completion_outcome
    from services.llm_cache_service import get_llm_cache

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 检查 LLM 缓存
    if hashing_kv is not None:
        from ._utils import compute_args_hash
        # v2 invalidates legacy entries that did not record/validate finish_reason
        # and may therefore contain a token-truncated extraction.
        args_hash = compute_args_hash(f"{model}:completion-v2", messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    max_tokens = kwargs.pop("max_tokens", 4096)
    temperature = kwargs.pop("temperature", 0.0)

    response = await call_ai_api(
        messages=messages,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    outcome = resolve_completion_outcome(
        response,
        transport_complete=not bool(
            isinstance(response, dict) and response.get("error")
        ),
    )
    if not outcome.publishable:
        logger.warning(
            "[GraphRAG] 拒绝缓存未完整生成的结果: status=%s finish_reason=%s",
            outcome.status.value,
            outcome.finish_reason or "unknown",
        )
        return ""

    content = ""
    if isinstance(response, dict) and not response.get("error"):
        choices = response.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

    if not content:
        error_msg = response.get("error", "未知错误") if isinstance(response, dict) else str(response)
        logger.warning(f"[GraphRAG] LLM 调用返回空内容: {error_msg}")
        content = ""

    # 写入缓存
    if hashing_kv is not None and content:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": content,
                    "model": model,
                    "completion_status": outcome.status.value,
                    "finish_reason": outcome.finish_reason,
                }
            }
        )

    return content


async def _chatpdf_embedding_func(
    texts: list[str],
    api_key: str = "",
    model: str = "",
    provider: str = "",
    endpoint: str = "",
    expected_dim: int | None = None,
) -> "np.ndarray":
    """通过统一 embedding 适配器获取 embedding。"""
    import numpy as np
    from services.embedding_service import get_embedding_function

    expected_dim = _require_positive_embedding_dim(
        expected_dim,
        context="GraphRAG Embedding expected_dim",
    )
    scoped_model = _compose_provider_scoped_embedding_model(model, provider)
    embed_fn = get_embedding_function(
        scoped_model or model,
        api_key=api_key or None,
        base_url=endpoint or None,
        allow_model_fallback=False,
    )
    embeddings = await asyncio.to_thread(embed_fn, texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"GraphRAG Embedding 返回格式异常，ndim={embeddings.ndim}")
    actual_dim = int(embeddings.shape[1]) if embeddings.shape[0] else expected_dim
    if actual_dim != expected_dim:
        raise ValueError(
            f"GraphRAG Embedding 维度不匹配：模型 {model} 预期 {expected_dim}，接口返回 {actual_dim}"
        )
    return embeddings


@dataclass
class GraphRAG:
    """GraphRAG 主类 - 适配 Chatpdf 后端"""

    working_dir: str = field(
        default_factory=lambda: f"data/graphrag_cache_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    config: GraphRAGConfig = field(default_factory=GraphRAGConfig)

    # 文本分块
    chunk_token_size: int = 2000
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4"

    # 实体提取
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # 图聚类
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # 社区报告
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # Embedding
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM 并发
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # 存储类
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # 扩展
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # 查询上下文缓存（避免重复组装上下文，尤其 hybrid 模式检索开销大）
    _query_context_cache: dict = field(default_factory=dict)
    _query_cache_max_size: int = 128

    def __post_init__(self):
        logger.info(f"[GraphRAG] 初始化，working_dir={self.working_dir}")

        if not os.path.exists(self.working_dir):
            logger.info(f"[GraphRAG] 创建工作目录 {self.working_dir}")
            os.makedirs(self.working_dir)

        global_config = asdict(self)
        # 移除不可序列化的字段
        global_config.pop("config", None)
        global_config.pop("key_string_value_json_storage_cls", None)
        global_config.pop("vector_db_storage_cls", None)
        global_config.pop("graph_storage_cls", None)
        global_config.pop("convert_response_to_json_func", None)
        self._global_config = global_config

        # 构建 embedding 函数
        cfg = self.config
        if not str(cfg.embedding_model or "").strip():
            raise ValueError("GraphRAG Embedding model 未配置")
        if not str(cfg.embedding_provider or "").strip():
            raise ValueError("GraphRAG Embedding provider 未配置")
        if str(cfg.embedding_provider or "").strip().lower() != "local" and not str(cfg.embedding_endpoint or "").strip():
            raise ValueError("GraphRAG 远程 Embedding endpoint 未配置")
        if (
            str(cfg.embedding_provider or "").strip().lower()
            not in {"local", "ollama"}
            and not str(cfg.embedding_api_key or "").strip()
        ):
            raise ValueError("GraphRAG 远程 Embedding API Key 未配置")
        embedding_dim = _require_positive_embedding_dim(
            cfg.embedding_dim,
            context="GraphRAG Embedding dimension",
        )
        embedding_func = wrap_embedding_func_with_attrs(
            embedding_dim=embedding_dim,
            max_token_size=8192,
        )(partial(
            _chatpdf_embedding_func,
            # Callers must explicitly bind an embedding key to its endpoint.
            # Falling back to the LLM key here would bypass that boundary when
            # persisted GraphRAG metadata points at another service.
            api_key=cfg.embedding_api_key,
            model=cfg.embedding_model,
            provider=cfg.embedding_provider,
            endpoint=cfg.embedding_endpoint,
            expected_dim=embedding_dim,
        ))

        # 构建 LLM 函数
        best_model_func = partial(
            _chatpdf_llm_complete,
            api_key=cfg.api_key,
            model=cfg.model,
            provider=cfg.provider,
            endpoint=cfg.endpoint,
        )
        cheap_model_func = partial(
            _chatpdf_llm_complete,
            api_key=cfg.api_key,
            model=cfg.cheap_model or cfg.model,
            provider=cfg.cheap_provider or cfg.provider,
            endpoint=cfg.cheap_endpoint or cfg.endpoint,
        )

        # 初始化存储
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=self._global_config
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=self._global_config
        )
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=self._global_config
            )
            if self.enable_llm_cache
            else None
        )
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=self._global_config
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=self._global_config
        )
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=self._global_config,
            embedding_func=embedding_func,
            meta_fields={"entity_name"},
        )
        # 关系向量库：用于 hybrid/global 查询时增强关系召回
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=self._global_config,
            embedding_func=embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )

        # 构建进度追踪
        self._build_progress = BuildProgress()
        self._config_hash = _compute_config_hash(
            self.config, self.chunk_token_size, self.entity_extract_max_gleaning
        )
        self._build_progress.config_hash = self._config_hash
        self._build_progress.model = self.config.model
        self._build_progress.provider = self.config.provider
        self._build_progress.endpoint = self.config.endpoint
        self._build_progress.embedding_model = self.config.embedding_model
        self._build_progress.embedding_provider = self.config.embedding_provider
        self._build_progress.embedding_endpoint = self.config.embedding_endpoint
        self._build_progress.embedding_identity_version = _GRAPHRAG_EMBEDDING_IDENTITY_VERSION
        self._build_progress.embedding_dim = embedding_dim
        self._build_progress.evidence_input_version = GRAPHRAG_EVIDENCE_INPUT_VERSION

        # 应用并发限流
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            embedding_func
        )
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(cheap_model_func, hashing_kv=self.llm_response_cache)
        )

        # 写入 global_config 供 _op 模块使用
        self._global_config["best_model_func"] = self.best_model_func
        self._global_config["cheap_model_func"] = self.cheap_model_func
        self._global_config["embedding_func"] = self.embedding_func
        self._global_config["convert_response_to_json_func"] = self.convert_response_to_json_func

    def insert(self, string_or_strings: Union[str, List[str]]):
        """同步插入文档"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        """同步查询"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()) -> str:
        """异步查询，支持 local / global / hybrid 三种模式。

        如果 only_output_context=True（aquery_context 调用），启用查询上下文缓存，
        避免同一查询反复组装上下文（尤其 hybrid 模式检索开销大）。
        """
        mode = getattr(param, "mode", "local")

        # 上下文缓存：仅缓存 only_output_context=True 的组装结果
        if param.only_output_context:
            cache_key = f"ctx_{mode}_{hashlib.md5(query.encode()).hexdigest()[:12]}"
            cached = self._query_context_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"[GraphRAG] 命中查询上下文缓存: {cache_key}")
                return cached

        if mode == "global":
            response = await global_query(
                query,
                self.community_reports,
                param,
                self._global_config,
            )
        elif mode == "hybrid":
            response = await hybrid_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self._global_config,
            )
        else:
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self._global_config,
            )

        # 写入上下文缓存（控制大小避免内存无限膨胀）
        if param.only_output_context and response:
            if len(self._query_context_cache) >= self._query_cache_max_size:
                # LRU 简单清理：清空一半
                keys = list(self._query_context_cache.keys())
                for k in keys[: len(keys) // 2]:
                    del self._query_context_cache[k]
            cache_key = f"ctx_{mode}_{hashlib.md5(query.encode()).hexdigest()[:12]}"
            self._query_context_cache[cache_key] = response

        await self._query_done()
        return response

    async def aquery_context(self, query: str, param: QueryParam = None) -> str:
        """仅返回 GraphRAG 上下文（不调用 LLM），用于融合到 RAG 管道。支持所有模式。"""
        if param is None:
            param = QueryParam(only_output_context=True)
        else:
            param.only_output_context = True
        context = await self.aquery(query, param)
        # aquery already calls _query_done; avoid double call
        return context

    async def ainsert(self, string_or_strings: Union[str, dict, List[Union[str, dict]]]):
        """异步插入文档（带进度追踪）"""
        self._build_progress.status = "building"
        self._build_progress.build_start = datetime.now().isoformat()
        self._build_progress.last_error = ""
        self._build_progress.evidence_input_version = GRAPHRAG_EVIDENCE_INPUT_VERSION
        raw_items = [string_or_strings] if isinstance(string_or_strings, (str, dict)) else list(string_or_strings or [])
        self._build_progress.input_source = (
            "block_index_evidence"
            if any(isinstance(item, dict) for item in raw_items)
            else "document_full_text"
        )
        self._save_metadata()
        try:
            string_or_strings = raw_items
            # 去重检测
            self._build_progress.stage = "chunking"
            self._build_progress.progress = 10
            new_docs = await self._prepare_new_docs(string_or_strings)
            if not new_docs:
                self._build_progress.status = "done"
                self._build_progress.stage = "skipped"
                self._build_progress.progress = 100
                self._build_progress.build_end = datetime.now().isoformat()
                self._save_metadata()
                return
            logger.info(f"[GraphRAG] 插入 {len(new_docs)} 个新文档")

            # 分块
            inserting_chunks = await self._prepare_inserting_chunks(new_docs)
            if not inserting_chunks:
                self._build_progress.status = "done"
                self._build_progress.stage = "skipped"
                self._build_progress.progress = 100
                self._build_progress.build_end = datetime.now().isoformat()
                self._save_metadata()
                return
            logger.info(f"[GraphRAG] 插入 {len(inserting_chunks)} 个新块")

            # 实体提取
            self._build_progress.stage = "extracting"
            self._build_progress.progress = 30
            self._save_metadata()
            await self._process_entities_and_clusters(inserting_chunks)

            # 持久化
            self._build_progress.stage = "persisting"
            self._build_progress.progress = 90
            self._save_metadata()
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)

            self._build_progress.status = "done"
            self._build_progress.stage = "done"
            self._build_progress.progress = 100
            self._build_progress.build_end = datetime.now().isoformat()
            self._update_progress_stats()
            self._save_metadata()
        except Exception as e:
            self._build_progress.status = "failed"
            self._build_progress.last_error = str(e)
            self._build_progress.build_end = datetime.now().isoformat()
            self._save_metadata()
            raise
        finally:
            await self._insert_done()

    @staticmethod
    def _normalize_insert_item(item: Any, ordinal: int) -> tuple[str, dict, str]:
        """Normalize a legacy string or a block-index evidence document.

        Evidence metadata is intentionally whitelisted.  GraphRAG persists
        text chunks to JSON, so allowing arbitrary caller fields here would
        make its storage format a second, uncontrolled API surface.
        """
        if not isinstance(item, dict):
            return str(item or "").strip(), {}, ""

        content = str(item.get("content") or item.get("text") or "").strip()
        raw_metadata = item.get("metadata")
        metadata_source = raw_metadata if isinstance(raw_metadata, dict) else {}
        allowed_metadata = (
            "evidence_schema_version",
            "evidence_source",
            "evidence_id",
            "block_id",
            "block_ids",
            "section_id",
            "section_path",
            "page",
            "page_range",
            "bbox",
            "rects",
            "block_type",
            "chunk_type",
            "reading_order",
            "parse_generation",
            "document_source_hash",
            "parser_route",
            "source",
            "page_size",
            "coordinate_space",
        )
        metadata = {
            key: metadata_source[key]
            for key in allowed_metadata
            if metadata_source.get(key) not in (None, "", [], {})
        }
        source_id = str(
            item.get("source_id")
            or metadata.get("evidence_id")
            or metadata.get("block_id")
            or f"structured:{ordinal}"
        ).strip()
        return content, metadata, source_id

    async def _prepare_new_docs(self, string_or_strings):
        new_docs = {}
        for ordinal, item in enumerate(string_or_strings or []):
            content, metadata, source_id = self._normalize_insert_item(item, ordinal)
            if not content:
                continue
            # Preserve legacy text-hash keys for string callers.  Structured
            # sources need their block identity in the key so repeated text on
            # different pages cannot collapse into one evidence document.
            key_material = f"{source_id}\n{content}" if source_id else content
            doc_key = compute_mdhash_id(key_material, prefix="doc-")
            new_docs[doc_key] = {
                "content": content,
                "metadata": metadata,
                "source_id": source_id,
            }
        _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not new_docs:
            logger.warning("[GraphRAG] 所有文档已存在于存储中")
        return new_docs

    async def _prepare_inserting_chunks(self, new_docs):
        inserting_chunks = {}
        for doc_key, doc in new_docs.items():
            source_id = str(doc.get("source_id") or "").strip()
            metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            split_chunks = chunking_by_token_size(
                doc["content"],
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
                tiktoken_model=self.tiktoken_model_name,
            )
            chunks = {}
            for fragment_index, dp in enumerate(split_chunks):
                chunk_content = str(dp.get("content") or "").strip()
                if not chunk_content:
                    continue
                key_material = chunk_content
                if source_id:
                    key_material = f"{doc_key}\n{fragment_index}\n{chunk_content}"
                chunk_key = compute_mdhash_id(key_material, prefix="chunk-")
                record = {
                    **dp,
                    "full_doc_id": doc_key,
                    **metadata,
                }
                if source_id:
                    record["graphrag_source_id"] = source_id
                    record["graphrag_fragment_index"] = fragment_index
                    record["graphrag_fragment_count"] = len(split_chunks)
                chunks[chunk_key] = record
            inserting_chunks.update(chunks)
        _add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
        inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
        if not inserting_chunks:
            logger.warning("[GraphRAG] 所有块已存在于存储中")
        return inserting_chunks

    async def _process_entities_and_clusters(self, inserting_chunks):
        logger.info("[GraphRAG] 实体提取中...")
        maybe_new_kg = await extract_entities(
            inserting_chunks,
            knwoledge_graph_inst=self.chunk_entity_relation_graph,
            entity_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            global_config=self._global_config,
        )
        if maybe_new_kg is None:
            logger.warning("[GraphRAG] 未发现新实体")
            return
        self.chunk_entity_relation_graph = maybe_new_kg

        # 分阶段持久化：实体提取完成后先持久化图和实体向量库
        await asyncio.gather(
            self.chunk_entity_relation_graph.index_done_callback(),
            self.entities_vdb.index_done_callback(),
            self.relationships_vdb.index_done_callback(),
        )
        logger.info("[GraphRAG] 实体/关系已持久化")

        # 聚类
        self._build_progress.stage = "clustering"
        self._build_progress.progress = 60
        self._save_metadata()
        logger.info("[GraphRAG] 社区聚类中...")
        await self.chunk_entity_relation_graph.clustering(self.graph_cluster_algorithm)

        # 聚类完成后持久化图（社区信息已写入节点）
        await self.chunk_entity_relation_graph.index_done_callback()

        # 社区报告
        self._build_progress.stage = "reporting"
        self._build_progress.progress = 75
        self._save_metadata()
        logger.info("[GraphRAG] 社区报告生成中...")
        await self.community_reports.drop()
        await generate_community_report(self.community_reports, self.chunk_entity_relation_graph, self._global_config)

        # 报告生成完成后持久化社区报告
        await self.community_reports.index_done_callback()

    async def _insert_done(self):
        tasks = [cast(StorageNameSpace, storage_inst).index_done_callback() for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ] if storage_inst is not None]
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = [cast(StorageNameSpace, storage_inst).index_done_callback() for storage_inst in [
            self.llm_response_cache
        ] if storage_inst is not None]
        await asyncio.gather(*tasks)

    def stats(self) -> dict:
        """返回 GraphRAG 索引统计信息（含构建元数据）"""
        graph = self.chunk_entity_relation_graph._graph
        result = {
            "working_dir": self.working_dir,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_docs": len(self.full_docs._data) if hasattr(self.full_docs, '_data') else 0,
            "num_chunks": len(self.text_chunks._data) if hasattr(self.text_chunks, '_data') else 0,
        }
        # 附加构建元数据
        result["build_meta"] = self._build_progress.to_dict()
        return result

    def get_build_progress(self) -> BuildProgress:
        """获取当前构建进度"""
        return self._build_progress

    def _update_progress_stats(self):
        """从实际存储更新进度中的统计数字"""
        graph = self.chunk_entity_relation_graph._graph
        self._build_progress.num_nodes = graph.number_of_nodes()
        self._build_progress.num_edges = graph.number_of_edges()
        self._build_progress.num_docs = len(self.full_docs._data) if hasattr(self.full_docs, '_data') else 0
        self._build_progress.num_chunks = len(self.text_chunks._data) if hasattr(self.text_chunks, '_data') else 0

    def _save_metadata(self):
        """持久化构建元数据到 working_dir"""
        meta_path = os.path.join(self.working_dir, "build_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._build_progress.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[GraphRAG] 保存构建元数据失败: {e}")

    @staticmethod
    def load_metadata(working_dir: str) -> Optional[BuildProgress]:
        """从磁盘加载构建元数据"""
        meta_path = os.path.join(working_dir, "build_meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return BuildProgress.from_dict(data)
        except Exception as e:
            logger.warning(f"[GraphRAG] 加载构建元数据失败: {e}")
            return None

    @staticmethod
    def has_persisted_index(working_dir: str) -> bool:
        """检查 working_dir 是否存在已持久化的 GraphRAG 索引"""
        graph_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
        return os.path.exists(graph_file)

    @staticmethod
    async def load_from_disk(working_dir: str, config: GraphRAGConfig,
                             chunk_token_size: int = 2000,
                             entity_extract_max_gleaning: int = 1,
                             best_model_max_async: int = 16,
                             cheap_model_max_async: int = 16,
                             strict_config_hash: bool = False) -> Optional["GraphRAG"]:
        """从磁盘加载已持久化的 GraphRAG 实例（不重新构建）

        如果 working_dir 不存在或索引不完整，返回 None。
        非严格模式下，config_hash 不一致只告警；严格模式要求
        config_hash 与当前 schema/配置完全一致，并且 embedding 身份元数据完整。
        """
        if not GraphRAG.has_persisted_index(working_dir):
            return None

        # 检查配置哈希
        disk_meta = GraphRAG.load_metadata(working_dir)
        current_hash = _compute_config_hash(config, chunk_token_size, entity_extract_max_gleaning)
        if strict_config_hash:
            if disk_meta is None:
                logger.warning("[GraphRAG] 缺少构建元数据，拒绝严格加载")
                return None
            if not str(disk_meta.config_hash or "").strip():
                logger.warning("[GraphRAG] 缺少 config_hash，拒绝严格加载")
                return None
            if disk_meta.config_hash != current_hash:
                logger.warning(
                    "[GraphRAG] 配置哈希不匹配，拒绝严格加载: 磁盘=%s 当前=%s",
                    disk_meta.config_hash,
                    current_hash,
                )
                return None
            if not _has_complete_embedding_identity(disk_meta):
                logger.warning("[GraphRAG] 缺少完整 embedding 身份元数据，拒绝严格加载")
                return None
        elif disk_meta and disk_meta.config_hash and disk_meta.config_hash != current_hash:
            logger.warning(
                f"[GraphRAG] 配置哈希不匹配: 磁盘={disk_meta.config_hash}, 当前={current_hash}，"
                f"可能需要重建索引"
            )

        rag = GraphRAG(
            working_dir=working_dir,
            config=config,
            chunk_token_size=chunk_token_size,
            entity_extract_max_gleaning=entity_extract_max_gleaning,
            best_model_max_async=best_model_max_async,
            cheap_model_max_async=cheap_model_max_async,
        )        # 从磁盘元数据恢复进度状态
        if disk_meta:
            rag._build_progress = disk_meta
        else:
            # 无元数据但有索引文件，标记为 done
            rag._build_progress.status = "done"
            rag._build_progress.stage = "done"
            rag._build_progress.progress = 100
            rag._build_progress.config_hash = current_hash
            rag._update_progress_stats()
            rag._save_metadata()
        logger.info(
            f"[GraphRAG] 从磁盘加载: {rag._build_progress.num_nodes} 节点, "
            f"{rag._build_progress.num_edges} 边, 状态={rag._build_progress.status}"
        )
        return rag
