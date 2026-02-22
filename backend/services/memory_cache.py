"""记忆数据内存缓存模块

为高频访问的记忆数据提供内存级缓存，避免每次调用都扫描所有 JSON 文件。
缓存在记忆写入操作后自动失效，使用 threading.Lock 保证线程安全。

Requirements: 5.1
"""

import threading
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.memory_store import MemoryEntry


class MemoryCache:
    """记忆数据内存缓存

    为 get_all_entries() 提供内存缓存层，写入操作后调用 invalidate() 使缓存失效。
    使用 threading.Lock 保证多线程环境下的数据一致性。
    """

    def __init__(self):
        self._entries_cache: Optional[list["MemoryEntry"]] = None
        self._cache_time: float = 0
        self._lock = threading.Lock()

    def get_all_entries(self) -> Optional[list["MemoryEntry"]]:
        """获取缓存的所有条目，缓存未命中返回 None

        Returns:
            缓存命中时返回记忆条目列表的浅拷贝，未命中返回 None
        """
        with self._lock:
            if self._entries_cache is None:
                return None
            # 返回浅拷贝，防止外部修改影响缓存
            return list(self._entries_cache)

    def set_all_entries(self, entries: list["MemoryEntry"]) -> None:
        """设置缓存

        Args:
            entries: 要缓存的记忆条目列表
        """
        with self._lock:
            # 存储浅拷贝，防止外部修改影响缓存
            self._entries_cache = list(entries)
            self._cache_time = time.monotonic()

    def invalidate(self) -> None:
        """使缓存失效（写入操作后调用）"""
        with self._lock:
            self._entries_cache = None
            self._cache_time = 0
