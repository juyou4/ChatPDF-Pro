"""GraphRAG 主类

适配 Chatpdf 后端架构的 GraphRAG 实现。
使用 Chatpdf 现有的 chat_service.call_ai_api 进行 LLM 调用，
支持通过配置项灵活切换 LLM 提供商和模型。
"""

import asyncio
import os
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, List, Union

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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


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
    embedding_dim: int = 1536


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
        args_hash = compute_args_hash(model, messages)
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
            {args_hash: {"return": content, "model": model}}
        )

    return content


async def _chatpdf_embedding_func(
    texts: list[str],
    api_key: str = "",
    model: str = "",
    provider: str = "",
    endpoint: str = "",
) -> "np.ndarray":
    """通过 OpenAI 兼容 API 获取 embedding"""
    import numpy as np
    import httpx
    from models.api_key_selector import select_api_key

    sanitized_key = select_api_key(api_key) or (api_key.strip() if api_key else "")

    if not endpoint:
        logger.error("[GraphRAG] Embedding endpoint 未配置")
        return np.zeros((len(texts), 1536), dtype=np.float32)

    # 确保 endpoint 以 /embeddings 结尾
    embed_url = endpoint.rstrip("/")
    if not embed_url.endswith("/embeddings"):
        embed_url = embed_url + "/embeddings"

    headers = {
        "Authorization": f"Bearer {sanitized_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(embed_url, headers=headers, json=body)
        if resp.status_code != 200:
            logger.error(f"[GraphRAG] Embedding API 错误: {resp.status_code} {resp.text[:200]}")
            return np.zeros((len(texts), 1536), dtype=np.float32)
        result = resp.json()
        return np.array([dp["embedding"] for dp in result["data"]], dtype=np.float32)


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
        embedding_func = wrap_embedding_func_with_attrs(
            embedding_dim=cfg.embedding_dim,
            max_token_size=8192,
        )(partial(
            _chatpdf_embedding_func,
            api_key=cfg.embedding_api_key or cfg.api_key,
            model=cfg.embedding_model,
            provider=cfg.embedding_provider or cfg.provider,
            endpoint=cfg.embedding_endpoint or cfg.endpoint,
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
        """异步查询，支持 local / global / hybrid 三种模式。"""
        mode = getattr(param, "mode", "local")
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

    async def ainsert(self, string_or_strings: Union[str, List[str]]):
        """异步插入文档"""
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # 去重检测
            new_docs = await self._prepare_new_docs(string_or_strings)
            if not new_docs:
                return
            logger.info(f"[GraphRAG] 插入 {len(new_docs)} 个新文档")

            # 分块
            inserting_chunks = await self._prepare_inserting_chunks(new_docs)
            if not inserting_chunks:
                return
            logger.info(f"[GraphRAG] 插入 {len(inserting_chunks)} 个新块")

            # 实体提取 + 社区聚类 + 社区报告
            await self._process_entities_and_clusters(inserting_chunks)

            # 持久化
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _prepare_new_docs(self, string_or_strings):
        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in string_or_strings
        }
        _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not new_docs:
            logger.warning("[GraphRAG] 所有文档已存在于存储中")
        return new_docs

    async def _prepare_inserting_chunks(self, new_docs):
        inserting_chunks = {}
        for doc_key, doc in new_docs.items():
            chunks = {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                    **dp,
                    "full_doc_id": doc_key,
                }
                for dp in chunking_by_token_size(
                    doc["content"],
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size,
                    tiktoken_model=self.tiktoken_model_name,
                )
            }
            inserting_chunks.update(chunks)
        _add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
        inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
        if not inserting_chunks:
            logger.warning("[GraphRAG] 所有块已存在于存储中")
        return inserting_chunks

    async def _process_entities_and_clusters(self, inserting_chunks):
        await self.community_reports.drop()
        logger.info("[GraphRAG] 实体提取中...")
        maybe_new_kg = await extract_entities(
            inserting_chunks,
            knwoledge_graph_inst=self.chunk_entity_relation_graph,
            entity_vdb=self.entities_vdb,
            global_config=self._global_config,
        )
        if maybe_new_kg is None:
            logger.warning("[GraphRAG] 未发现新实体")
            return
        self.chunk_entity_relation_graph = maybe_new_kg
        logger.info("[GraphRAG] 社区聚类中...")
        await self.chunk_entity_relation_graph.clustering(self.graph_cluster_algorithm)
        logger.info("[GraphRAG] 社区报告生成中...")
        await generate_community_report(self.community_reports, self.chunk_entity_relation_graph, self._global_config)

    async def _insert_done(self):
        tasks = [cast(StorageNameSpace, storage_inst).index_done_callback() for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunk_entity_relation_graph,
        ] if storage_inst is not None]
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = [cast(StorageNameSpace, storage_inst).index_done_callback() for storage_inst in [
            self.llm_response_cache
        ] if storage_inst is not None]
        await asyncio.gather(*tasks)

    def stats(self) -> dict:
        """返回 GraphRAG 索引统计信息"""
        graph = self.chunk_entity_relation_graph._graph
        return {
            "working_dir": self.working_dir,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_docs": len(self.full_docs._data) if hasattr(self.full_docs, '_data') else 0,
            "num_chunks": len(self.text_chunks._data) if hasattr(self.text_chunks, '_data') else 0,
        }
