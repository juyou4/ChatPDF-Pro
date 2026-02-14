"""
记忆向量索引管理模块

使用 FAISS 存储记忆条目的向量表示，支持语义检索。
复用 embedding_service.get_embedding_function 生成向量。
"""

import logging
import os
import pickle
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


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
        self.index: Optional[faiss.IndexFlatL2] = None
        # 元数据：与 FAISS 索引行一一对应
        self.entry_ids: list[str] = []
        self.texts: list[str] = []

    def _get_embed_fn(self, api_key: str = None):
        """获取 embedding 函数"""
        from services.embedding_service import get_embedding_function
        return get_embedding_function(self.embedding_model_id, api_key=api_key)

    def _embed_texts(self, texts: list[str], api_key: str = None) -> np.ndarray:
        """将文本列表转为向量数组"""
        embed_fn = self._get_embed_fn(api_key)
        embeddings = embed_fn(texts)
        return np.array(embeddings, dtype=np.float32)

    def add_entry(self, entry_id: str, text: str, api_key: str = None) -> None:
        """为记忆条目生成向量并添加到 FAISS 索引

        Args:
            entry_id: 记忆条目唯一标识
            text: 记忆内容文本
            api_key: API 密钥（远程模型需要）
        """
        try:
            embeddings = self._embed_texts([text], api_key)
            dimension = embeddings.shape[1]

            # 首次添加时创建索引
            if self.index is None:
                self.index = faiss.IndexFlatL2(dimension)

            self.index.add(embeddings)
            self.entry_ids.append(entry_id)
            self.texts.append(text)

            # 自动持久化
            self.save()
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
        if entry_id not in self.entry_ids:
            logger.warning(f"记忆条目不在向量索引中: {entry_id}")
            return

        idx = self.entry_ids.index(entry_id)
        self.entry_ids.pop(idx)
        self.texts.pop(idx)

        if self.index is not None and self.index.ntotal > 0:
            # 从 FAISS 索引中提取所有向量
            all_vectors = faiss.rev_swig_ptr(
                self.index.get_xb(), self.index.ntotal * self.index.d
            ).reshape(self.index.ntotal, self.index.d).copy()

            # 删除对应行并重建索引
            remaining_vectors = np.delete(all_vectors, idx, axis=0)
            dimension = self.index.d
            self.index = faiss.IndexFlatL2(dimension)
            if len(remaining_vectors) > 0:
                self.index.add(remaining_vectors.astype(np.float32))

        # 如果没有条目了，清空索引
        if len(self.entry_ids) == 0:
            self.index = None

        self.save()
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
        if self.index is None or self.index.ntotal == 0:
            return []

        try:
            query_embedding = self._embed_texts([query], api_key)
            # 实际搜索数量不超过索引中的条目数
            actual_k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, actual_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.entry_ids):
                    continue
                # L2 距离转相似度：使用 1 / (1 + distance) 映射到 (0, 1]
                similarity = 1.0 / (1.0 + float(dist))
                results.append({
                    "entry_id": self.entry_ids[idx],
                    "similarity": similarity,
                    "text": self.texts[idx],
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
        # 清空现有索引和元数据
        self.index = None
        self.entry_ids = []
        self.texts = []

        if not entries:
            self.save()
            logger.info("记忆向量索引已清空")
            return

        try:
            texts = [e.content for e in entries]
            ids = [e.id for e in entries]

            embeddings = self._embed_texts(texts, api_key)
            dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            self.entry_ids = ids
            self.texts = texts

            self.save()
            logger.info(f"记忆向量索引已重建，共 {len(entries)} 条")
        except Exception as e:
            logger.error(f"重建记忆向量索引失败: {e}")
            raise

    def save(self) -> None:
        """持久化 FAISS 索引和元数据到磁盘"""
        os.makedirs(self.index_dir, exist_ok=True)

        index_path = os.path.join(self.index_dir, "memory.index")
        meta_path = os.path.join(self.index_dir, "memory.pkl")

        # 保存 FAISS 索引
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        elif os.path.exists(index_path):
            # 索引为空时删除旧文件
            os.remove(index_path)

        # 保存元数据
        meta = {
            "entry_ids": self.entry_ids,
            "texts": self.texts,
            "embedding_model": self.embedding_model_id,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        logger.debug(f"记忆向量索引已保存到 {self.index_dir}")

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

            self.entry_ids = meta.get("entry_ids", [])
            self.texts = meta.get("texts", [])
            stored_model = meta.get("embedding_model", "")

            # 检查 embedding 模型是否一致
            if stored_model and stored_model != self.embedding_model_id:
                logger.warning(
                    f"索引 embedding 模型不一致: 存储={stored_model}, "
                    f"当前={self.embedding_model_id}，需要重建索引"
                )
                self.entry_ids = []
                self.texts = []
                self.index = None
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
                    self.entry_ids = []
                    self.texts = []
                    self.index = None
                    return False
            else:
                # 元数据存在但索引文件不存在
                if self.entry_ids:
                    logger.warning("FAISS 索引文件缺失但元数据存在，需要重建索引")
                    self.entry_ids = []
                    self.texts = []
                    return False

            logger.info(f"记忆向量索引已加载，共 {len(self.entry_ids)} 条")
            return True
        except Exception as e:
            logger.error(f"加载记忆向量索引失败: {e}")
            self.index = None
            self.entry_ids = []
            self.texts = []
            return False
