"""
记忆向量索引管理模块

使用 FAISS 存储记忆条目的向量表示，支持语义检索。
复用 embedding_service.get_embedding_function 生成向量。
"""

import hashlib
import logging
import os
import pickle
import threading
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import faiss
import numpy as np

from services.bm25_service import (
    BM25Index,
    BM25_TOKENIZER_VERSION,
    get_bm25_tokenizer_signature,
)

logger = logging.getLogger(__name__)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """归一化向量，使 Inner Product = 余弦相似度"""
    v = vectors.astype(np.float32).copy()
    faiss.normalize_L2(v)
    return v


class MemoryIndex:
    """记忆向量索引管理"""

    def __init__(
        self,
        index_dir: str,
        embedding_model_id: str = "local-minilm",
        *,
        embedding_provider: str = "",
        embedding_api_host: str = "",
        vector_search_enabled: Optional[bool] = None,
    ):
        """
        初始化记忆向量索引

        Args:
            index_dir: 索引存储目录，如 "data/memory/memory_index/"
            embedding_model_id: embedding 模型 ID，默认使用本地 MiniLM
            embedding_provider: 模型提供商身份，不包含凭证
            embedding_api_host: 归一化后的 API 基础地址，不包含凭证
            vector_search_enabled: 是否启用向量检索。None 时根据当前运行时
                自动判断；Windows 下 FAISS 与本地 PyTorch 冲突时使用 BM25-only。
        """
        self.index_dir = index_dir
        self.embedding_model_id = embedding_model_id
        self.embedding_provider = str(embedding_provider or "").strip().casefold()
        self.embedding_api_host = str(embedding_api_host or "").strip().rstrip("/")
        if vector_search_enabled is None:
            vector_search_enabled = self._default_vector_search_enabled(
                embedding_model_id
            )
        self.vector_search_enabled = bool(vector_search_enabled)
        self.vector_disabled_reason = (
            "windows_faiss_torch_openmp_conflict"
            if not self.vector_search_enabled
            else ""
        )
        self.index: Optional[faiss.IndexFlatIP] = None
        # 元数据：与 FAISS 索引行一一对应
        self.entry_ids: list[str] = []
        self.texts: list[str] = []
        # 内容 hash 映射：用于变更检测 + 增量索引
        self._content_hashes: dict[str, str] = {}  # entry_id -> content_hash
        # 嵌入缓存：content_hash -> embedding，避免重复计算
        self._embedding_cache: dict[str, np.ndarray] = {}  # content_hash -> embedding
        # 持久化 BM25 索引
        self._bm25: Optional[BM25Index] = None
        self._bm25_variants: dict[str, BM25Index] = {}
        # FAISS、entry_ids/texts 与 BM25 必须作为同一代状态提交和读取。
        self._state_lock = threading.RLock()
        self._state_generation: int = 0
        self._synced_generation: int = 0
        # dirty-sync 状态
        self.dirty: bool = False
        self.last_sync_at: str = ""
        self.last_reindex_at: str = ""
        self.last_reindex_reason: str = ""
        self.index_version: int = 1
        self.stored_embedding_model: str = embedding_model_id
        self.stored_embedding_provider: str = self.embedding_provider
        self.stored_embedding_api_host: str = self.embedding_api_host
        self.rebuild_required: bool = False
        self.rebuild_reason: str = ""
        self._sync_debounce_seconds: float = 5.0
        self._sync_timer: Optional[threading.Timer] = None
        self._save_lock = threading.RLock()
        self._pending_reason: str = ""
        self.on_change: Optional[Callable[[str], None]] = None

    @staticmethod
    def _default_vector_search_enabled(embedding_model_id: str) -> bool:
        """Keep remote/test models enabled; fence only unsafe local models."""
        try:
            from models.model_id_resolver import resolve_model_id
            from models.model_detector import get_model_provider
            from services.embedding_service import is_local_embedding_runtime_supported

            registry_key, config = resolve_model_id(embedding_model_id)
            provider = (
                str((config or {}).get("provider") or "").casefold()
                if registry_key is not None
                else str(get_model_provider(embedding_model_id) or "").casefold()
            )
            if provider == "local":
                return is_local_embedding_runtime_supported()
        except Exception as exc:
            logger.debug("无法判断记忆向量运行时，保留既有模式: %s", exc)
        return True

    @staticmethod
    def _hash_content(text: str) -> str:
        """计算内容 hash（MD5，仅用于变更检测）"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _mark_state_changed_locked(self) -> int:
        """推进内存状态代际；调用方必须持有 _state_lock。"""
        self._state_generation += 1
        self.dirty = self._state_generation != self._synced_generation
        return self._state_generation

    def _mark_snapshot_synced(self, snapshot_generation: int) -> None:
        """只确认快照实际写出的代际，保留其后提交的 dirty 状态。"""
        with self._state_lock:
            self._synced_generation = snapshot_generation
            self.dirty = self._state_generation != self._synced_generation

    def _notify_change(self, reason: str) -> None:
        callback = self.on_change
        if callback is None:
            return
        try:
            callback(str(reason or "mutation"))
        except Exception as exc:
            logger.warning("记忆索引变更回调失败: %s", exc)

    def _get_embed_fn(self, api_key: str = None):
        """获取 embedding 函数"""
        if not self.vector_search_enabled:
            raise RuntimeError(
                "记忆向量检索已因 Windows FAISS/PyTorch OpenMP 冲突停用"
            )
        from services.embedding_service import get_embedding_function
        return get_embedding_function(
            self.embedding_model_id,
            api_key=api_key,
            base_url=self.embedding_api_host or None,
            allow_model_fallback=False,
        )

    def embedding_identity(self) -> dict[str, str]:
        """Return the credential-free vector-space identity for this snapshot."""
        return {
            "model": str(self.embedding_model_id or "").strip(),
            "provider": str(self.embedding_provider or "").strip().casefold(),
            "api_host": str(self.embedding_api_host or "").strip().rstrip("/"),
        }

    def _embedding_cache_namespace(self) -> str:
        identity = self.embedding_identity()
        return "|".join(
            (identity["provider"], identity["model"], identity["api_host"])
        )

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取缓存的 embedding
        
        Args:
            text: 文本内容
            
        Returns:
            缓存的 embedding，如果不存在则返回 None
        """
        content_hash = self._hash_content(text)
        return self._embedding_cache.get(content_hash)
    
    def _cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """缓存 embedding
        
        Args:
            text: 文本内容
            embedding: 对应的向量
        """
        content_hash = self._hash_content(text)
        self._embedding_cache[content_hash] = embedding
    
    def _embed_texts(self, texts: list[str], api_key: str = None, use_cache: bool = True) -> np.ndarray:
        """将文本列表转为向量数组，支持缓存
        
        Args:
            texts: 文本列表
            api_key: API 密钥
            use_cache: 是否使用缓存，默认 True
            
        Returns:
            向量数组
        """
        if not texts:
            return np.array([], dtype=np.float32)
        
        # 检查缓存
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cached = self._get_cached_embedding(text)
                if cached is not None:
                    cached_embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # 对未缓存的文本进行 embedding
        new_embeddings = None
        if uncached_texts:
            embed_fn = self._get_embed_fn(api_key)
            new_embeddings_list = embed_fn(uncached_texts)
            new_embeddings = np.array(new_embeddings_list, dtype=np.float32)
            
            # 缓存新的 embedding
            for text, emb in zip(uncached_texts, new_embeddings):
                self._cache_embedding(text, emb)
        
        # 合并缓存和新的 embedding
        if not cached_embeddings:
            return new_embeddings if new_embeddings is not None else np.array([], dtype=np.float32)
        
        if new_embeddings is None:
            # 全部来自缓存
            result = np.zeros((len(texts), cached_embeddings[0][1].shape[0]), dtype=np.float32)
            for i, emb in cached_embeddings:
                result[i] = emb
            return result
        
        # 合并缓存和新计算的
        result = np.zeros((len(texts), new_embeddings.shape[1]), dtype=np.float32)
        for i, emb in cached_embeddings:
            result[i] = emb
        for idx, emb in zip(uncached_indices, new_embeddings):
            result[idx] = emb
        
        return result

    def add_entry(
        self,
        entry_id: str,
        text: str,
        api_key: str = None,
        *,
        should_commit: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """为记忆条目生成向量并添加到 FAISS 索引

        如果 entry_id 已存在且内容未变化（hash 相同），跳过重复 embedding。

        Args:
            entry_id: 记忆条目唯一标识
            text: 记忆内容文本
            api_key: API 密钥（远程模型需要）
        """
        # 变更检测：hash 相同则跳过
        if should_commit is not None and not should_commit():
            return False
        new_hash = self._hash_content(text)
        with self._state_lock:
            if entry_id in self._content_hashes and self._content_hashes[entry_id] == new_hash:
                logger.debug(f"记忆条目内容未变化，跳过 embedding: {entry_id}")
                return False

        try:
            if not self.vector_search_enabled:
                with self._state_lock:
                    if should_commit is not None and not should_commit():
                        return False
                    if entry_id in self.entry_ids:
                        idx = self.entry_ids.index(entry_id)
                        self.texts[idx] = text
                    else:
                        self.entry_ids.append(entry_id)
                        self.texts.append(text)
                    self._content_hashes[entry_id] = new_hash
                    self._rebuild_bm25()
                    self._mark_state_changed_locked()
                self.schedule_sync(reason="add")
                self._notify_change("add")
                logger.debug("记忆条目已写入 BM25-only 索引: %s", entry_id)
                return True

            # 使用缓存机制进行 embedding
            embeddings = self._embed_texts([text], api_key, use_cache=True)
            dimension = embeddings.shape[1]

            # 归一化向量，使 IP = 余弦相似度
            embeddings = _normalize_vectors(embeddings)

            # A memory clear can happen while the embedding call is in
            # progress. Check the caller's generation fence before any state
            # becomes visible or durable again.
            if should_commit is not None and not should_commit():
                return False

            with self._state_lock:
                if should_commit is not None and not should_commit():
                    return False
                # embedding 计算期间可能有另一个写入先完成，提交前再次检查。
                if self._content_hashes.get(entry_id) == new_hash:
                    return False
                # 首次添加时创建索引
                if self.index is None:
                    self.index = faiss.IndexFlatIP(dimension)

                self.index.add(embeddings)
                self.entry_ids.append(entry_id)
                self.texts.append(text)
                self._content_hashes[entry_id] = new_hash

                # 增量更新 BM25 索引
                self._rebuild_bm25()
                self._mark_state_changed_locked()

            # 异步 dirty-sync 持久化
            self.schedule_sync(reason="add")
            self._notify_change("add")
            logger.info(f"记忆条目已添加到向量索引: {entry_id}")
            return True
        except Exception as e:
            logger.error(f"添加记忆条目到向量索引失败: {e}")
            raise

    def remove_entry(self, entry_id: str) -> None:
        """从索引中移除指定条目

        由于 FAISS IndexFlatL2 不支持单条删除，
        采用重建索引的方式移除条目。

        Args:
            entry_id: 要移除的记忆条目 ID
        """
        with self._state_lock:
            if entry_id not in self.entry_ids:
                logger.warning(f"记忆条目不在向量索引中: {entry_id}")
                # The authoritative store may already have removed the row even
                # when this baseline index was stale. Downstream snapshots still
                # need the deletion fence for privacy correctness.
                self._notify_change("remove_missing")
                return

            idx = self.entry_ids.index(entry_id)
            self.entry_ids.pop(idx)
            self.texts.pop(idx)
            self._content_hashes.pop(entry_id, None)

            if self.index is not None and self.index.ntotal > 0:
                # 从 FAISS 索引中提取所有向量
                all_vectors = faiss.rev_swig_ptr(
                    self.index.get_xb(), self.index.ntotal * self.index.d
                ).reshape(self.index.ntotal, self.index.d).copy()

                # 删除对应行并重建索引
                remaining_vectors = np.delete(all_vectors, idx, axis=0)
                dimension = self.index.d
                self.index = faiss.IndexFlatIP(dimension)
                if len(remaining_vectors) > 0:
                    remaining_vectors = _normalize_vectors(remaining_vectors)
                    self.index.add(remaining_vectors)

            # 如果没有条目了，清空索引
            if len(self.entry_ids) == 0:
                self.index = None

            # 重建 BM25 索引
            self._rebuild_bm25()
            self._mark_state_changed_locked()

        self.schedule_sync(reason="remove")
        self._notify_change("remove")
        logger.info(f"记忆条目已从向量索引移除: {entry_id}")

    def search(self, query: str, top_k: int = 3, api_key: str = None) -> list[dict]:
        """向量检索最相关的记忆条目

        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            api_key: API 密钥（远程模型需要）

        Returns:
            [{"entry_id": str, "similarity": float, "text": str}, ...]
        """
        if not self.vector_search_enabled:
            return []
        with self._state_lock:
            if self.index is None or self.index.ntotal == 0:
                return []

        try:
            # 查询向量缓存：避免重复 embedding 计算
            from services.embedding_service import _query_vector_cache
            cache_key = f"memory:{query}"
            cache_namespace = self._embedding_cache_namespace()
            cached = _query_vector_cache.get(cache_namespace, cache_key)
            if cached is not None:
                query_embedding = cached
            else:
                query_embedding = self._embed_texts([query], api_key)
                # 归一化查询向量
                query_embedding = _normalize_vectors(query_embedding)
                _query_vector_cache.put(cache_namespace, cache_key, query_embedding)
            with self._state_lock:
                if self.index is None or self.index.ntotal == 0:
                    return []
                # FAISS search 与 add/remove/write_index 互斥，结果和元数据取同一快照。
                actual_k = min(top_k, self.index.ntotal)
                distances, indices = self.index.search(query_embedding, actual_k)
                is_ip = (self.index.metric_type == faiss.METRIC_INNER_PRODUCT)
                entry_ids_snapshot = list(self.entry_ids)
                texts_snapshot = list(self.texts)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(entry_ids_snapshot):
                    continue
                if is_ip:
                    # IP 归一化后分数即余弦相似度 [0, 1]
                    similarity = max(0.0, min(float(dist), 1.0))
                else:
                    # 旧 L2 索引兼容
                    similarity = 1.0 / (1.0 + float(dist))
                results.append({
                    "entry_id": entry_ids_snapshot[idx],
                    "similarity": similarity,
                    "text": texts_snapshot[idx],
                })

            return results
        except Exception as e:
            logger.error(f"记忆向量检索失败: {e}")
            return []

    def rebuild(self, entries: list, api_key: str = None) -> None:
        """重建整个索引（用于清空后重建或修复）

        Args:
            entries: MemoryEntry 对象列表
            api_key: API 密钥（远程模型需要）
        """
        if not entries:
            self._embedding_cache.clear()
            self._apply_reindex_state(
                {
                    "index": None,
                    "entry_ids": [],
                    "texts": [],
                    "content_hashes": {},
                    "bm25": None,
                },
                reason="rebuild",
            )
            logger.info("记忆向量索引已清空")
            return

        self.safe_reindex(entries, api_key=api_key, reason="rebuild")

    def _rebuild_bm25(self) -> None:
        """从 self.texts 重建 BM25 索引"""
        with self._state_lock:
            texts_snapshot = list(self.texts)
            if not texts_snapshot:
                self._set_primary_bm25(None)
                return
            bm25 = BM25Index()
            bm25.build(texts_snapshot)
            self._set_primary_bm25(bm25)

    def _set_primary_bm25(self, bm25: Optional[BM25Index]) -> None:
        """更新持久化主索引，并使所有请求级只读变体失效。"""
        with self._state_lock:
            self._bm25 = bm25
            self._bm25_variants = {}
            if bm25 is not None:
                signature = str(getattr(bm25, "tokenizer_signature", "") or "")
                if signature:
                    self._bm25_variants[signature] = bm25

    def _mark_rebuild_required(self, reason: str, *, stored_model: str = "") -> None:
        """记录需要重建索引的原因。"""
        self.rebuild_required = True
        self.rebuild_reason = reason
        if stored_model:
            self.stored_embedding_model = stored_model

    def _clear_rebuild_required(self) -> None:
        """清除待重建标记。"""
        self.rebuild_required = False
        self.rebuild_reason = ""
        self.stored_embedding_model = self.embedding_model_id
        self.stored_embedding_provider = self.embedding_provider
        self.stored_embedding_api_host = self.embedding_api_host

    def matches_entries(self, entries: list) -> bool:
        """Whether the snapshot exactly matches the current retrievable store."""
        expected_ids = [str(entry.id) for entry in entries]
        expected_hashes = {
            str(entry.id): self._hash_content(str(entry.content or ""))
            for entry in entries
        }
        with self._state_lock:
            return bool(
                self.entry_ids == expected_ids
                and self._content_hashes == expected_hashes
                and (
                    not self.vector_search_enabled
                    or not expected_ids
                    or (
                        self.index is not None
                        and self.index.ntotal == len(expected_ids)
                    )
                )
            )

    def prune_stale_entries(self, entries: list, *, reason: str = "store_mutation") -> int:
        """Remove deleted or content-changed rows without calculating embeddings.

        New rows are intentionally not added here because a remote credential is
        unavailable on most background memory writes. The next authenticated
        retrieval rebuilds the complete snapshot; this method only guarantees
        that stale or deleted text cannot remain queryable or durable meanwhile.
        """
        expected_hashes = {
            str(entry.id): self._hash_content(str(entry.content or ""))
            for entry in entries
        }
        with self._state_lock:
            keep_positions = [
                position
                for position, entry_id in enumerate(self.entry_ids)
                if expected_hashes.get(str(entry_id))
                == self._content_hashes.get(str(entry_id))
            ]
            removed = len(self.entry_ids) - len(keep_positions)
            if removed <= 0:
                return 0

            old_ids = list(self.entry_ids)
            old_texts = list(self.texts)
            self.entry_ids = [old_ids[position] for position in keep_positions]
            self.texts = [old_texts[position] for position in keep_positions]
            self._content_hashes = {
                entry_id: self._content_hashes[entry_id]
                for entry_id in self.entry_ids
            }

            if self.index is not None:
                dimension = self.index.d
                if keep_positions:
                    all_vectors = faiss.rev_swig_ptr(
                        self.index.get_xb(), self.index.ntotal * dimension
                    ).reshape(self.index.ntotal, dimension).copy()
                    kept_vectors = all_vectors[keep_positions]
                    self.index = faiss.IndexFlatIP(dimension)
                    self.index.add(_normalize_vectors(kept_vectors))
                else:
                    self.index = None

            self._rebuild_bm25()
            self._mark_state_changed_locked()

        self.schedule_sync(reason=reason)
        return removed

    def _build_reindex_state(self, entries: list, api_key: str = None) -> dict:
        """先在内存中构建新索引状态，成功后再整体切换。"""
        if not entries:
            return {
                "index": None,
                "entry_ids": [],
                "texts": [],
                "content_hashes": {},
                "bm25": None,
            }

        texts = [str(e.content or "") for e in entries]
        entry_ids = [str(e.id) for e in entries]
        content_hashes = {
            entry_id: self._hash_content(text)
            for entry_id, text in zip(entry_ids, texts)
        }
        bm25 = BM25Index()
        bm25.build(texts)

        index = None
        if self.vector_search_enabled:
            reusable_vectors: dict[str, np.ndarray] = {}
            with self._state_lock:
                if (
                    self.index is not None
                    and self.index.ntotal == len(self.entry_ids)
                ):
                    dimension = self.index.d
                    existing_vectors = faiss.rev_swig_ptr(
                        self.index.get_xb(), self.index.ntotal * dimension
                    ).reshape(self.index.ntotal, dimension).copy()
                    for position, existing_id in enumerate(self.entry_ids):
                        if (
                            content_hashes.get(str(existing_id))
                            == self._content_hashes.get(str(existing_id))
                        ):
                            reusable_vectors[str(existing_id)] = existing_vectors[position]

            missing_positions = [
                position
                for position, entry_id in enumerate(entry_ids)
                if entry_id not in reusable_vectors
            ]
            missing_vectors: dict[int, np.ndarray] = {}
            if missing_positions:
                embedded = self._embed_texts(
                    [texts[position] for position in missing_positions],
                    api_key,
                    use_cache=True,
                )
                missing_vectors = {
                    position: vector
                    for position, vector in zip(missing_positions, embedded)
                }

            embeddings = np.asarray(
                [
                    reusable_vectors.get(entry_id, missing_vectors.get(position))
                    for position, entry_id in enumerate(entry_ids)
                ],
                dtype=np.float32,
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(entry_ids):
                raise ValueError("记忆向量增量重建结果不完整")
            dimension = embeddings.shape[1]
            embeddings = _normalize_vectors(embeddings)
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

        return {
            "index": index,
            "entry_ids": entry_ids,
            "texts": texts,
            "content_hashes": content_hashes,
            "bm25": bm25,
        }

    def _apply_reindex_state(
        self,
        state: dict,
        *,
        reason: str = "rebuild",
        expected_generation: Optional[int] = None,
    ) -> bool:
        """提交新的索引状态。"""
        with self._state_lock:
            if (
                expected_generation is not None
                and self._state_generation != expected_generation
            ):
                return False
            self.index = state["index"]
            self.entry_ids = state["entry_ids"]
            self.texts = state["texts"]
            self._content_hashes = state["content_hashes"]
            self._set_primary_bm25(state["bm25"])
            self._mark_state_changed_locked()
        now_iso = datetime.now(timezone.utc).isoformat()
        self.last_reindex_at = now_iso
        self.last_reindex_reason = reason
        self._clear_rebuild_required()
        self.schedule_sync(reason="rebuild")
        self._notify_change(reason)
        return True

    def safe_reindex(self, entries: list, api_key: str = None, reason: str = "rebuild") -> bool:
        """安全重建索引，失败时保留旧索引。"""
        with self._state_lock:
            start_generation = self._state_generation
        try:
            state = self._build_reindex_state(entries, api_key=api_key)
        except Exception as e:
            logger.error(f"安全重建记忆向量索引失败，保留旧索引: {e}")
            self._mark_rebuild_required(f"{reason}_failed", stored_model=self.stored_embedding_model or self.embedding_model_id)
            raise

        applied = self._apply_reindex_state(
            state,
            reason=reason,
            expected_generation=start_generation,
        )
        if not applied:
            logger.info(
                "记忆索引重建结果已过期，保留较新的内存状态: reason=%s start=%s current=%s",
                reason,
                start_generation,
                self._state_generation,
            )
            return False
        logger.info(f"记忆向量索引已安全重建，共 {len(entries)} 条")
        return True

    def bm25_search(self, query: str, top_k: int = 3) -> list[dict]:
        """使用持久化的 BM25 索引检索记忆

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [{"chunk": str, "entry_id": str, "score": float}, ...]
        """
        signature = get_bm25_tokenizer_signature()
        with self._state_lock:
            texts_snapshot = list(self.texts)
            entry_ids_snapshot = list(self.entry_ids)
            if not texts_snapshot:
                return []
            bm25 = self._bm25_variants.get(signature)
            if bm25 is None or bm25.chunks != texts_snapshot:
                bm25 = BM25Index(use_jieba=signature.endswith(":jieba"))
                bm25.build(texts_snapshot)
                self._bm25_variants[signature] = bm25

        # 变体构建后不可变；使用同一时刻的 entry id 快照保证结果映射一致。
        raw_results = bm25.search(query, top_k=top_k)
        results = []
        for item in raw_results:
            idx = item["index"]
            if 0 <= idx < len(entry_ids_snapshot):
                results.append({
                    "chunk": item["chunk"],
                    "entry_id": entry_ids_snapshot[idx],
                    "score": item["score"],
                })
        return results

    def _write_snapshot(self) -> int:
        """在状态锁内持久化同一代 FAISS 与元数据。"""
        with self._state_lock:
            self._write_snapshot_locked()
            return self._state_generation

    def _write_snapshot_locked(self) -> None:
        """持久化 FAISS 索引和元数据到磁盘"""
        os.makedirs(self.index_dir, exist_ok=True)

        index_path = os.path.join(self.index_dir, "memory.index")
        meta_path = os.path.join(self.index_dir, "memory.pkl")
        temp_suffix = f".{uuid.uuid4().hex}.tmp"

        # 保存 FAISS 索引
        if self.index is not None:
            temp_index_path = index_path + temp_suffix
            faiss.write_index(self.index, temp_index_path)
            os.replace(temp_index_path, index_path)
        elif os.path.exists(index_path):
            # 索引为空时删除旧文件
            os.remove(index_path)

        # 保存元数据（含 BM25 索引 + 内容 hash）
        # 注意：embedding_cache 不持久化（内存缓存），每次启动重建
        bm25_tokenizer_version = getattr(
            self._bm25,
            "tokenizer_version",
            BM25_TOKENIZER_VERSION,
        )
        bm25_tokenizer_signature = getattr(
            self._bm25,
            "tokenizer_signature",
            get_bm25_tokenizer_signature(),
        )
        meta = {
            "entry_ids": self.entry_ids,
            "texts": self.texts,
            "embedding_model": self.embedding_model_id,
            "embedding_provider": self.embedding_provider,
            "embedding_api_host": self.embedding_api_host,
            "embedding_identity_version": 1,
            "vector_search_enabled": self.vector_search_enabled,
            "bm25": self._bm25,
            "bm25_tokenizer_version": bm25_tokenizer_version,
            "bm25_tokenizer_signature": bm25_tokenizer_signature,
            "content_hashes": self._content_hashes,
            "last_reindex_reason": self.last_reindex_reason,
            "state_generation": self._state_generation,
        }
        temp_meta_path = meta_path + temp_suffix
        with open(temp_meta_path, "wb") as f:
            pickle.dump(meta, f)
        os.replace(temp_meta_path, meta_path)

        logger.debug(f"记忆向量索引已保存到 {self.index_dir}")

    def save(self) -> None:
        """立即持久化当前索引快照。"""
        with self._save_lock:
            self._cancel_pending_sync()
            snapshot_generation = self._write_snapshot()
            self._mark_snapshot_synced(snapshot_generation)
            self.last_sync_at = datetime.now(timezone.utc).isoformat()

    def _cancel_pending_sync(self) -> None:
        if self._sync_timer is not None:
            self._sync_timer.cancel()
            self._sync_timer = None

    def schedule_sync(self, reason: str = "") -> None:
        """标记 dirty，并在 debounce 窗口后统一落盘。"""
        with self._save_lock:
            with self._state_lock:
                self.dirty = self._state_generation != self._synced_generation
                needs_sync = self.dirty
            self._pending_reason = reason or self._pending_reason
            self._cancel_pending_sync()
            if not needs_sync:
                self._pending_reason = ""
                return
            delay = self._sync_debounce_seconds
            if delay <= 0:
                self.flush_sync(reason=reason)
                return
            self._sync_timer = threading.Timer(delay, self.flush_sync, kwargs={"reason": reason})
            self._sync_timer.daemon = True
            self._sync_timer.start()

    def flush_sync(self, reason: str = "") -> None:
        """刷新 dirty 索引到磁盘。"""
        with self._save_lock:
            self._cancel_pending_sync()
            with self._state_lock:
                needs_sync = self._state_generation != self._synced_generation
                self.dirty = needs_sync
            if not needs_sync and reason != "manual":
                return
            snapshot_generation = self._write_snapshot()
            now_iso = datetime.now(timezone.utc).isoformat()
            self._mark_snapshot_synced(snapshot_generation)
            self.last_sync_at = now_iso
            if (reason or self._pending_reason) == "rebuild":
                self.last_reindex_at = now_iso
            self._pending_reason = ""

    def get_status(self) -> dict:
        """返回索引 dirty-sync 状态快照。"""
        with self._save_lock:
            with self._state_lock:
                return {
                    "dirty": self._state_generation != self._synced_generation,
                    "last_sync_at": self.last_sync_at,
                    "last_reindex_at": self.last_reindex_at,
                    "last_reindex_reason": self.last_reindex_reason,
                    "index_version": self.index_version,
                    "state_generation": self._state_generation,
                    "synced_generation": self._synced_generation,
                    "pending_sync": self._sync_timer is not None,
                    "stored_embedding_model": self.stored_embedding_model or self.embedding_model_id,
                    "embedding_provider": self.stored_embedding_provider or self.embedding_provider,
                    "embedding_api_host": self.stored_embedding_api_host or self.embedding_api_host,
                    "vector_search_enabled": self.vector_search_enabled,
                    "vector_index_size": (
                        self.index.ntotal if self.index is not None else 0
                    ),
                    "vector_disabled_reason": self.vector_disabled_reason,
                    "lexical_index_size": len(self.entry_ids),
                    "retrieval_mode": (
                        "hybrid" if self.vector_search_enabled else "bm25_only"
                    ),
                    "rebuild_required": self.rebuild_required,
                    "rebuild_reason": self.rebuild_reason,
                }

    def load(self) -> bool:
        """从磁盘加载索引

        Returns:
            是否成功加载
        """
        index_path = os.path.join(self.index_dir, "memory.index")
        meta_path = os.path.join(self.index_dir, "memory.pkl")

        if not os.path.exists(meta_path):
            logger.info("记忆向量索引元数据不存在，跳过加载")
            return False

        try:
            # 加载元数据
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            try:
                stored_state_generation = max(0, int(meta.get("state_generation", 0) or 0))
            except (TypeError, ValueError):
                stored_state_generation = 0
            with self._state_lock:
                self._state_generation = stored_state_generation
                self._synced_generation = stored_state_generation
                self.dirty = False

            self.entry_ids = meta.get("entry_ids", [])
            self.texts = meta.get("texts", [])
            stored_model = meta.get("embedding_model", "")
            self.stored_embedding_model = stored_model or self.embedding_model_id
            stored_provider = str(meta.get("embedding_provider") or "").strip().casefold()
            stored_api_host = str(meta.get("embedding_api_host") or "").strip().rstrip("/")
            self.stored_embedding_provider = stored_provider or self.embedding_provider
            self.stored_embedding_api_host = stored_api_host or self.embedding_api_host
            self.last_reindex_reason = meta.get("last_reindex_reason", "")

            # 恢复持久化的 BM25 索引
            stored_bm25 = meta.get("bm25")
            try:
                stored_bm25_tokenizer_version = int(meta.get("bm25_tokenizer_version", 0) or 0)
            except (TypeError, ValueError):
                stored_bm25_tokenizer_version = 0
            stored_bm25_tokenizer_signature = str(
                meta.get("bm25_tokenizer_signature") or ""
            )
            current_bm25_tokenizer_signature = get_bm25_tokenizer_signature()
            bm25_snapshot_required = False
            if (
                isinstance(stored_bm25, BM25Index)
                and stored_bm25.doc_count > 0
                and stored_bm25_tokenizer_version == BM25_TOKENIZER_VERSION
                and stored_bm25_tokenizer_signature == current_bm25_tokenizer_signature
                and getattr(stored_bm25, "tokenizer_version", 0) == BM25_TOKENIZER_VERSION
                and getattr(stored_bm25, "tokenizer_signature", "") == current_bm25_tokenizer_signature
            ):
                self._set_primary_bm25(stored_bm25)
            elif self.texts:
                # 旧版 BM25 或旧 tokenizer 不能复用；从已持久化文本轻量重建，
                # 无需重新计算 embedding。
                self._rebuild_bm25()
                bm25_snapshot_required = True
            else:
                self._set_primary_bm25(None)

            # 恢复内容 hash 映射
            stored_hashes = meta.get("content_hashes", {})
            if isinstance(stored_hashes, dict):
                self._content_hashes = stored_hashes
            else:
                # 旧版元数据没有 hash，从 texts 重建
                self._content_hashes = {
                    eid: self._hash_content(txt)
                    for eid, txt in zip(self.entry_ids, self.texts)
                }

            # 检查 embedding 模型是否一致
            identity_mismatch = bool(
                (stored_model and stored_model != self.embedding_model_id)
                or (stored_provider and stored_provider != self.embedding_provider)
                or (stored_api_host and stored_api_host != self.embedding_api_host)
            )
            if identity_mismatch:
                logger.warning(
                    "索引 embedding 身份不一致: 存储=%s/%s/%s 当前=%s/%s/%s，需要重建索引",
                    stored_provider,
                    stored_model,
                    stored_api_host,
                    self.embedding_provider,
                    self.embedding_model_id,
                    self.embedding_api_host,
                )
                self._mark_rebuild_required(
                    (
                        "embedding_model_changed"
                        if stored_model != self.embedding_model_id
                        and not (
                            stored_provider != self.embedding_provider
                            or stored_api_host != self.embedding_api_host
                        )
                        else "embedding_identity_changed"
                    ),
                    stored_model=stored_model,
                )
                self.entry_ids = []
                self.texts = []
                self.index = None
                self._set_primary_bm25(None)
                return False

            # BM25-only 模式保留 entry_ids/texts 与持久化词法索引，绝不因
            # 当前机器不能安全加载 Torch 而清空历史。FAISS 只是可重建缓存。
            if not self.vector_search_enabled:
                self.index = None
            elif os.path.exists(index_path):
                self.index = faiss.read_index(index_path)

                # 验证索引与元数据一致性
                if self.index.ntotal != len(self.entry_ids):
                    logger.warning(
                        f"FAISS 索引条目数({self.index.ntotal})与元数据"
                        f"({len(self.entry_ids)})不一致，需要重建索引"
                    )
                    self._mark_rebuild_required("metadata_count_mismatch", stored_model=stored_model)
                    self.entry_ids = []
                    self.texts = []
                    self.index = None
                    self._set_primary_bm25(None)
                    return False
            else:
                # 元数据存在但索引文件不存在
                if self.entry_ids:
                    logger.warning("FAISS 索引文件缺失但元数据存在，需要重建索引")
                    self._mark_rebuild_required("missing_index_file", stored_model=stored_model)
                    self.entry_ids = []
                    self.texts = []
                    self._set_primary_bm25(None)
                    return False

            # 自动迁移旧 L2 索引到 IP 索引
            if (
                self.vector_search_enabled
                and self.index is not None
                and self.index.metric_type != faiss.METRIC_INNER_PRODUCT
            ):
                logger.info("检测到旧 L2 索引，自动迁移为 IP 索引...")
                n = self.index.ntotal
                d = self.index.d
                if n > 0:
                    all_vectors = faiss.rev_swig_ptr(
                        self.index.get_xb(), n * d
                    ).reshape(n, d).copy()
                    all_vectors = _normalize_vectors(all_vectors)
                    self.index = faiss.IndexFlatIP(d)
                    self.index.add(all_vectors)
                else:
                    self.index = faiss.IndexFlatIP(d)
                self.save()
                bm25_snapshot_required = False
                logger.info(f"L2 → IP 迁移完成，共 {n} 条向量")

            if bm25_snapshot_required:
                self.save()
                logger.info(
                    "记忆 BM25 tokenizer 已迁移到版本 %s",
                    BM25_TOKENIZER_VERSION,
                )

            with self._state_lock:
                self._synced_generation = self._state_generation
                self.dirty = False
            self._clear_rebuild_required()

            logger.info(
                "记忆%s索引已加载，共 %d 条",
                "混合" if self.vector_search_enabled else " BM25-only ",
                len(self.entry_ids),
            )
            return True
        except Exception as e:
            logger.error(f"加载记忆向量索引失败: {e}")
            self._mark_rebuild_required("load_failed", stored_model=self.stored_embedding_model or self.embedding_model_id)
            with self._state_lock:
                self.index = None
                self.entry_ids = []
                self.texts = []
                self._set_primary_bm25(None)
            return False
