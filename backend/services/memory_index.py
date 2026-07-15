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
from typing import Optional

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

    def __init__(self, index_dir: str, embedding_model_id: str = "local-minilm"):
        """
        初始化记忆向量索引

        Args:
            index_dir: 索引存储目录，如 "data/memory/memory_index/"
            embedding_model_id: embedding 模型 ID，默认使用本地 MiniLM
        """
        self.index_dir = index_dir
        self.embedding_model_id = embedding_model_id
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
        self.rebuild_required: bool = False
        self.rebuild_reason: str = ""
        self._sync_debounce_seconds: float = 5.0
        self._sync_timer: Optional[threading.Timer] = None
        self._save_lock = threading.RLock()
        self._pending_reason: str = ""

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

    def _get_embed_fn(self, api_key: str = None):
        """获取 embedding 函数"""
        from services.embedding_service import get_embedding_function
        return get_embedding_function(self.embedding_model_id, api_key=api_key)

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

    def add_entry(self, entry_id: str, text: str, api_key: str = None) -> None:
        """为记忆条目生成向量并添加到 FAISS 索引

        如果 entry_id 已存在且内容未变化（hash 相同），跳过重复 embedding。

        Args:
            entry_id: 记忆条目唯一标识
            text: 记忆内容文本
            api_key: API 密钥（远程模型需要）
        """
        # 变更检测：hash 相同则跳过
        new_hash = self._hash_content(text)
        with self._state_lock:
            if entry_id in self._content_hashes and self._content_hashes[entry_id] == new_hash:
                logger.debug(f"记忆条目内容未变化，跳过 embedding: {entry_id}")
                return

        try:
            # 使用缓存机制进行 embedding
            embeddings = self._embed_texts([text], api_key, use_cache=True)
            dimension = embeddings.shape[1]

            # 归一化向量，使 IP = 余弦相似度
            embeddings = _normalize_vectors(embeddings)

            with self._state_lock:
                # embedding 计算期间可能有另一个写入先完成，提交前再次检查。
                if self._content_hashes.get(entry_id) == new_hash:
                    return
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
            logger.info(f"记忆条目已添加到向量索引: {entry_id}")
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
        with self._state_lock:
            if self.index is None or self.index.ntotal == 0:
                return []

        try:
            # 查询向量缓存：避免重复 embedding 计算
            from services.embedding_service import _query_vector_cache
            cache_key = f"memory:{query}"
            cached = _query_vector_cache.get(self.embedding_model_id, cache_key)
            if cached is not None:
                query_embedding = cached
            else:
                query_embedding = self._embed_texts([query], api_key)
                # 归一化查询向量
                query_embedding = _normalize_vectors(query_embedding)
                _query_vector_cache.put(self.embedding_model_id, cache_key, query_embedding)
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

        texts = [e.content for e in entries]
        entry_ids = [e.id for e in entries]
        embeddings = self._embed_texts(texts, api_key, use_cache=True)
        dimension = embeddings.shape[1]
        embeddings = _normalize_vectors(embeddings)

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        bm25 = BM25Index()
        bm25.build(texts)

        return {
            "index": index,
            "entry_ids": entry_ids,
            "texts": texts,
            "content_hashes": {
                entry_id: self._hash_content(text)
                for entry_id, text in zip(entry_ids, texts)
            },
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
            if stored_model and stored_model != self.embedding_model_id:
                logger.warning(
                    f"索引 embedding 模型不一致: 存储={stored_model}, "
                    f"当前={self.embedding_model_id}，需要重建索引"
                )
                self._mark_rebuild_required("embedding_model_changed", stored_model=stored_model)
                self.entry_ids = []
                self.texts = []
                self.index = None
                self._set_primary_bm25(None)
                return False

            # 加载 FAISS 索引
            if os.path.exists(index_path):
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
            if self.index is not None and self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
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

            logger.info(f"记忆向量索引已加载，共 {len(self.entry_ids)} 条")
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
