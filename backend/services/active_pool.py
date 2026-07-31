"""
类 OS 活跃记忆池（Active Pool）

使用 LRU（最近最少使用）策略管理内存中的活跃记忆条目。
类似操作系统的 RAM 管理：活跃记忆保持在内存中，不活跃的记忆
通过 Page-Out 卸载到磁盘（Passive Store）。
"""

import logging
import re
from collections import OrderedDict
from typing import Optional

from services.memory_store import MemoryEntry

logger = logging.getLogger(__name__)


class ActivePool:
    """类 OS 活跃记忆池（LRU 策略）

    使用 OrderedDict 实现 LRU：
    - 最近使用的条目在末尾
    - 最久未使用的条目在头部
    - 达到容量上限时淘汰头部条目
    """

    def __init__(self, capacity: int = 100):
        """初始化活跃记忆池

        Args:
            capacity: 池容量上限，默认 100 条
        """
        self.capacity = max(1, capacity)  # 至少容纳 1 条
        self._pool: OrderedDict[str, MemoryEntry] = OrderedDict()

    @staticmethod
    def is_cacheable(entry: MemoryEntry) -> bool:
        """仅缓存高价值长期记忆，避免普通 QA 摘要挤占热缓存。"""
        if not entry or getattr(entry, "status", "active") == "archived_raw":
            return False
        return getattr(entry, "memory_kind", "") in {"profile", "doc_fact", "consolidated"}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()) if token]

    def entries(self) -> list[MemoryEntry]:
        """返回当前池中的条目快照，按最近使用优先。"""
        return list(reversed(list(self._pool.values())))

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """获取记忆条目（命中时移到最近使用位置）

        实现 Page-In 语义：被访问的记忆移到 LRU 末尾，
        表示最近使用过。

        Args:
            entry_id: 记忆条目 ID

        Returns:
            命中的 MemoryEntry，未命中返回 None
        """
        if entry_id not in self._pool:
            return None
        # 移到末尾（最近使用位置）
        self._pool.move_to_end(entry_id)
        return self._pool[entry_id]

    def put(self, entry: MemoryEntry) -> Optional[MemoryEntry]:
        """放入记忆条目，返回被淘汰的记忆（如果有）

        如果条目已存在，更新并移到末尾。
        如果池已满，淘汰最久未使用的条目（头部）。

        Args:
            entry: 要放入的记忆条目

        Returns:
            被淘汰的 MemoryEntry，如果没有淘汰则返回 None
        """
        if not self.is_cacheable(entry):
            return None
        evicted = None

        # 如果已存在，先移除旧条目（不算淘汰）
        if entry.id in self._pool:
            self._pool.move_to_end(entry.id)
            self._pool[entry.id] = entry
            return None

        # 池已满，淘汰最久未使用的条目（头部）
        if len(self._pool) >= self.capacity:
            # popitem(last=False) 弹出头部（最久未使用）
            evicted_id, evicted = self._pool.popitem(last=False)
            logger.debug(f"[ActivePool] 淘汰记忆: {evicted_id}")

        # 放入新条目（末尾，最近使用位置）
        self._pool[entry.id] = entry
        return evicted

    def search(self, entry_ids: list[str]) -> list[MemoryEntry]:
        """批量查找，返回在池中的记忆条目

        不改变 LRU 顺序（仅查找，不算访问）。

        Args:
            entry_ids: 要查找的记忆条目 ID 列表

        Returns:
            在池中找到的 MemoryEntry 列表
        """
        results = []
        for eid in entry_ids:
            if eid in self._pool:
                results.append(self._pool[eid])
        return results

    def query(
        self,
        query: str,
        *,
        top_k: int = 3,
        doc_id: str | None = None,
        filter_by_doc: bool = False,
        filter_tags: list[str] | None = None,
        allowed_ids: set[str] | None = None,
    ) -> list[dict]:
        """在热缓存中执行轻量关键词检索，并把命中条目标记为最近使用。"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        filter_set = set(filter_tags or [])
        scored: list[tuple[float, str, MemoryEntry]] = []

        for entry_id, entry in list(self._pool.items()):
            if allowed_ids is not None and entry_id not in allowed_ids:
                continue
            if filter_by_doc and doc_id and entry.doc_id not in (None, doc_id):
                continue
            if filter_set and not filter_set.intersection(entry.tags or []):
                continue

            haystack_parts = [
                entry.title or "",
                entry.summary or "",
                entry.content or "",
                " ".join(entry.tags or []),
            ]
            haystack = " ".join(haystack_parts).lower()
            overlap = 0.0
            exact_boost = 0.0
            for token in query_tokens:
                if token not in haystack:
                    continue
                overlap += min(len(token), 8)
                if (entry.title or "").lower().find(token) >= 0:
                    exact_boost += 1.0
            if overlap <= 0:
                continue

            score = overlap + exact_boost + min(0.5, entry.importance) + min(1.0, entry.hit_count * 0.05)
            scored.append((score, entry_id, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict] = []
        for score, entry_id, entry in scored[:top_k]:
            self.get(entry_id)
            results.append({
                "chunk": entry.content,
                "entry_id": entry.id,
                "similarity": score,
                "from_active_pool": True,
            })
        return results

    def preload(self, entries: list[MemoryEntry]) -> None:
        """预加载记忆条目（服务启动时调用）

        按顺序加载，超出容量的部分会被忽略。
        已存在的条目会被更新。

        Args:
            entries: 要预加载的记忆条目列表（应按 last_hit_at 降序排列）
        """
        for entry in entries:
            if not self.is_cacheable(entry):
                continue
            if len(self._pool) >= self.capacity and entry.id not in self._pool:
                # 池已满且不是更新已有条目，跳过
                break
            self._pool[entry.id] = entry
        logger.info(f"[ActivePool] 预加载完成，当前池大小: {len(self._pool)}/{self.capacity}")

    def size(self) -> int:
        """当前池中记忆数量

        Returns:
            池中条目数量
        """
        return len(self._pool)

    def clear(self) -> None:
        """Drop all in-memory copies after a user requests a full erase."""
        self._pool.clear()

    def remove_entry(self, entry_id: str) -> None:
        """Remove one document-owned entry without flushing unrelated hot data."""
        self._pool.pop(entry_id, None)
