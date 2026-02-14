"""
记忆混合检索器模块

复用 hybrid_search_merge 融合向量检索和 BM25 检索结果，
实现记忆条目的混合检索和上下文格式化。
"""

import logging
from typing import Optional

from services.bm25_service import BM25Index
from services.hybrid_search import hybrid_search_merge
from services.memory_index import MemoryIndex
from services.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """记忆混合检索器"""

    def __init__(self, memory_store: MemoryStore, memory_index: MemoryIndex):
        """
        初始化记忆检索器

        Args:
            memory_store: 记忆持久化存储实例
            memory_index: 记忆向量索引实例
        """
        self.memory_store = memory_store
        self.memory_index = memory_index

    def retrieve(self, query: str, top_k: int = 3, api_key: str = None) -> list[dict]:
        """混合检索相关记忆

        1. 向量检索：通过 memory_index.search 获取语义相关记忆
        2. BM25 检索：对所有记忆文本构建临时 BM25 索引进行关键词匹配
        3. RRF 融合：通过 hybrid_search_merge 合并两路结果

        Args:
            query: 用户查询文本
            top_k: 返回的最大结果数，默认 3
            api_key: API 密钥（远程 embedding 模型需要）

        Returns:
            top_k 条最相关记忆，每项包含 entry_id, text, source_type 等字段
        """
        if not query or not query.strip():
            return []

        # 获取所有记忆条目，用于 BM25 检索
        all_entries = self.memory_store.get_all_entries()
        if not all_entries:
            return []

        # 构建 entry_id -> entry 的映射，方便后续查找
        entry_map = {entry.id: entry for entry in all_entries}

        # 1. 向量检索
        vector_results = self._vector_search(query, top_k=top_k, api_key=api_key)

        # 2. BM25 检索
        bm25_results = self._bm25_search(query, all_entries, top_k=top_k)

        # 3. RRF 融合
        merged = hybrid_search_merge(
            vector_results, bm25_results, top_k=top_k
        )

        # 将融合结果转换为统一的输出格式
        results = []
        for item in merged:
            entry_id = item.get("entry_id", "")
            entry = entry_map.get(entry_id)
            results.append({
                "entry_id": entry_id,
                "text": item.get("chunk", ""),
                "source_type": entry.source_type if entry else "unknown",
                "doc_id": entry.doc_id if entry else None,
                "rrf_score": item.get("rrf_score", 0.0),
            })

        return results[:top_k]

    def _vector_search(self, query: str, top_k: int = 3, api_key: str = None) -> list[dict]:
        """向量检索，将结果转换为 hybrid_search_merge 期望的格式

        Args:
            query: 查询文本
            top_k: 返回数量
            api_key: API 密钥

        Returns:
            格式化的向量检索结果列表，每项包含 chunk 和 entry_id 字段
        """
        try:
            raw_results = self.memory_index.search(query, top_k=top_k, api_key=api_key)
            # 转换为 hybrid_search_merge 期望的格式（需要 chunk 字段）
            return [
                {
                    "chunk": item["text"],
                    "entry_id": item["entry_id"],
                    "similarity": item["similarity"],
                }
                for item in raw_results
            ]
        except Exception as e:
            logger.error(f"记忆向量检索失败: {e}")
            return []

    def _bm25_search(self, query: str, entries: list, top_k: int = 3) -> list[dict]:
        """BM25 检索，对记忆文本构建临时索引进行关键词匹配

        Args:
            query: 查询文本
            entries: MemoryEntry 对象列表
            top_k: 返回数量

        Returns:
            格式化的 BM25 检索结果列表，每项包含 chunk 和 entry_id 字段
        """
        if not entries:
            return []

        try:
            # 构建临时 BM25 索引
            texts = [entry.content for entry in entries]
            bm25_index = BM25Index()
            bm25_index.build(texts)

            # 执行检索
            raw_results = bm25_index.search(query, top_k=top_k)

            # 转换为 hybrid_search_merge 期望的格式
            results = []
            for item in raw_results:
                idx = item["index"]
                if 0 <= idx < len(entries):
                    results.append({
                        "chunk": item["chunk"],
                        "entry_id": entries[idx].id,
                        "score": item["score"],
                    })
            return results
        except Exception as e:
            logger.error(f"记忆 BM25 检索失败: {e}")
            return []

    def build_memory_context(self, memories: list[dict]) -> str:
        """将检索到的记忆条目格式化为上下文字符串

        格式：
        用户历史记忆：
        - [来源类型] 内容摘要
        - [来源类型] 内容摘要

        Args:
            memories: retrieve 方法返回的记忆列表

        Returns:
            格式化的记忆上下文字符串，无记忆时返回空字符串
        """
        if not memories:
            return ""

        lines = ["用户历史记忆："]
        for mem in memories:
            source_type = mem.get("source_type", "unknown")
            text = mem.get("text", "")
            lines.append(f"- [{source_type}] {text}")

        return "\n".join(lines)
