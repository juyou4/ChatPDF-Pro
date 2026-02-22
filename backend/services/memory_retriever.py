"""
记忆混合检索器模块

复用 hybrid_search_merge 融合向量检索和 BM25 检索结果，
实现记忆条目的混合检索和上下文格式化。
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from services.hybrid_search import hybrid_search_merge
from services.memory_index import MemoryIndex
from services.memory_store import MemoryStore

# 时间衰减半衰期（天）
_DECAY_HALF_LIFE_DAYS = 30.0

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """记忆混合检索器"""

    def __init__(self, memory_store: MemoryStore, memory_index: MemoryIndex, active_pool=None):
        """
        初始化记忆检索器

        Args:
            memory_store: 记忆持久化存储实例
            memory_index: 记忆向量索引实例
            active_pool: 活跃记忆池实例（可选，用于优先检索）
        """
        self.memory_store = memory_store
        self.memory_index = memory_index
        self.active_pool = active_pool

    def retrieve(self, query: str, top_k: int = 3, api_key: str = None, 
                 doc_id: Optional[str] = None, filter_by_doc: bool = False,
                 filter_tags: Optional[list[str]] = None) -> list[dict]:
        """混合检索相关记忆

        1. 优先从 Active_Pool 检索（如果可用）
        2. 向量检索：通过 memory_index.search 获取语义相关记忆
        3. BM25 检索：对所有记忆文本构建临时 BM25 索引进行关键词匹配
        4. RRF 融合：通过 hybrid_search_merge 合并两路结果
        5. 文档相关性：同一文档的记忆优先（如果提供 doc_id）
        6. 标签过滤：仅返回包含指定标签的记忆（如果提供 filter_tags）

        Args:
            query: 用户查询文本
            top_k: 返回的最大结果数，默认 3
            api_key: API 密钥（远程 embedding 模型需要）
            doc_id: 当前文档 ID，用于文档相关性加权（可选）
            filter_by_doc: 是否只返回当前文档的记忆，默认 False（仅加权）
            filter_tags: 标签过滤列表，仅返回包含指定标签的记忆（可选）

        Returns:
            top_k 条最相关记忆，每项包含 entry_id, text, source_type 等字段
        """
        if not query or not query.strip():
            return []

        # 获取所有记忆条目，用于 BM25 检索
        all_entries = self.memory_store.get_all_entries()
        if not all_entries:
            return []

        # 如果启用文档过滤，只保留当前文档的记忆
        if filter_by_doc and doc_id:
            all_entries = [e for e in all_entries if e.doc_id == doc_id]
            if not all_entries:
                return []

        # 标签过滤：仅保留包含指定标签的记忆
        if filter_tags:
            filter_set = set(filter_tags)
            all_entries = [e for e in all_entries if filter_set.intersection(e.tags)]
            if not all_entries:
                return []

        # 构建 entry_id -> entry 的映射，方便后续查找
        entry_map = {entry.id: entry for entry in all_entries}
        # 构建 entry_id -> doc_id 索引，用于优化 _record_hits
        self._entry_doc_index = {entry.id: entry.doc_id for entry in all_entries}

        # 1. 向量检索
        vector_results = self._vector_search(query, top_k=top_k, api_key=api_key)

        # 2. BM25 检索
        bm25_results = self._bm25_search(query, all_entries, top_k=top_k)

        # 3. RRF 融合
        merged = hybrid_search_merge(
            vector_results, bm25_results, top_k=top_k
        )

        # 将融合结果转换为统一的输出格式，并应用时间衰减 + 动态重要性
        now = datetime.now(timezone.utc)
        results = []
        hit_entry_ids = []  # 记录命中的 entry_id，用于更新命中统计

        for item in merged:
            entry_id = item.get("entry_id", "")
            entry = entry_map.get(entry_id)
            rrf_score = item.get("rrf_score", 0.0)

            # 动态评分：基础重要性 × 时间衰减 × 命中次数加成 × 文档相关性
            dynamic_score = rrf_score
            if entry:
                dynamic_score = self._compute_dynamic_score(
                    rrf_score, entry, now, doc_id=doc_id
                )
                hit_entry_ids.append(entry_id)

            results.append({
                "entry_id": entry_id,
                "text": item.get("chunk", ""),
                "source_type": entry.source_type if entry else "unknown",
                "doc_id": entry.doc_id if entry else None,
                "rrf_score": dynamic_score,
            })

        # 按动态评分重新排序
        results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        # 异步更新命中统计（不阻塞检索）
        if hit_entry_ids:
            self._record_hits(hit_entry_ids, entry_map, now)

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
        """BM25 检索，使用 MemoryIndex 中持久化的 BM25 索引

        Args:
            query: 查询文本
            entries: MemoryEntry 对象列表（保留参数兼容性）
            top_k: 返回数量

        Returns:
            格式化的 BM25 检索结果列表，每项包含 chunk 和 entry_id 字段
        """
        try:
            return self.memory_index.bm25_search(query, top_k=top_k)
        except Exception as e:
            logger.error(f"记忆 BM25 检索失败: {e}")
            return []

    @staticmethod
    def _compute_dynamic_score(rrf_score: float, entry, now: datetime, 
                               doc_id: Optional[str] = None) -> float:
        """计算动态评分：RRF 分数 × 重要性权重 × 时间衰减 × 命中加成 × 文档相关性

        公式：score = rrf_score × importance_w × recency_decay × hit_boost × doc_relevance
        - importance_w: importance 归一化到 [0.5, 1.5] 区间
        - recency_decay: exp(-days / half_life)，基于 last_hit_at 或 created_at
        - hit_boost: 1 + log2(1 + hit_count) × 0.1
        - doc_relevance: 文档相关性权重，同一文档的记忆权重 1.2，其他 1.0

        Args:
            rrf_score: RRF 融合后的基础分数
            entry: MemoryEntry 对象
            now: 当前 UTC 时间
            doc_id: 当前文档 ID，用于文档相关性加权（可选）

        Returns:
            调整后的动态分数
        """
        # 重要性权重：[0.5, 1.5]
        importance_w = 0.5 + entry.importance

        # 时间衰减：基于最后命中时间（无命中则用创建时间）
        # 优化：使用更平滑的衰减曲线
        ref_time_str = entry.last_hit_at or entry.created_at
        days_since = 0.0
        if ref_time_str:
            try:
                ref_time = datetime.fromisoformat(ref_time_str)
                if ref_time.tzinfo is None:
                    ref_time = ref_time.replace(tzinfo=timezone.utc)
                delta = (now - ref_time).total_seconds() / 86400.0
                days_since = max(0.0, delta)
            except (ValueError, TypeError):
                pass
        
        # 优化时间衰减：使用更平滑的指数衰减，半衰期 30 天
        recency_decay = math.exp(-days_since * math.log(2) / _DECAY_HALF_LIFE_DAYS)
        # 确保衰减值在 [0.1, 1.0] 范围内，避免过度衰减
        recency_decay = max(0.1, min(1.0, recency_decay))

        # 命中次数加成：1 + log2(1 + hit_count) × 0.1
        # 限制最大加成，避免过度偏向高频记忆
        hit_boost = 1.0 + min(0.3, math.log2(1 + entry.hit_count) * 0.1)

        # 文档相关性权重：同一文档的记忆优先
        doc_relevance = 1.2 if (doc_id and entry.doc_id == doc_id) else 1.0

        return rrf_score * importance_w * recency_decay * hit_boost * doc_relevance

    def _record_hits(self, entry_ids: list[str], entry_map: dict, now: datetime) -> None:
        """记录检索命中：更新 hit_count 和 last_hit_at

        使用 entry_id → doc_id 索引优化，仅加载命中条目所在的 session，
        避免遍历所有 session 文件。

        Args:
            entry_ids: 本次命中的 entry_id 列表
            entry_map: entry_id -> MemoryEntry 映射
            now: 当前 UTC 时间
        """
        hit_set = set(entry_ids)
        now_iso = now.isoformat()

        # 使用 entry_id → doc_id 索引（在 retrieve 中构建）
        entry_doc_index = getattr(self, "_entry_doc_index", {})

        try:
            # 按 doc_id 分组命中的 entry_id，减少文件 I/O
            profile_hits = set()
            session_hits: dict[str, set[str]] = {}  # doc_id -> set of entry_ids

            for eid in hit_set:
                doc_id = entry_doc_index.get(eid)
                if doc_id is None:
                    # 无 doc_id 的条目在 profile 中
                    profile_hits.add(eid)
                else:
                    if doc_id not in session_hits:
                        session_hits[doc_id] = set()
                    session_hits[doc_id].add(eid)

            # 更新 profile 中的 entries
            if profile_hits:
                profile = self.memory_store.load_profile()
                profile_changed = False
                for e_data in profile.get("entries", []):
                    if e_data.get("id") in profile_hits:
                        e_data["hit_count"] = e_data.get("hit_count", 0) + 1
                        e_data["last_hit_at"] = now_iso
                        profile_changed = True
                if profile_changed:
                    self.memory_store.save_profile(profile)

            # 仅加载包含命中条目的 session
            for doc_id, eids in session_hits.items():
                session = self.memory_store.load_session(doc_id)
                session_changed = False

                for item in session.get("qa_summaries", []):
                    if item.get("id") in eids:
                        item["hit_count"] = item.get("hit_count", 0) + 1
                        item["last_hit_at"] = now_iso
                        session_changed = True

                for item in session.get("important_memories", []):
                    if item.get("id") in eids:
                        item["hit_count"] = item.get("hit_count", 0) + 1
                        item["last_hit_at"] = now_iso
                        session_changed = True

                if session_changed:
                    self.memory_store.save_session(doc_id, session)
        except Exception as e:
            logger.warning(f"记录记忆命中统计失败: {e}")

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
