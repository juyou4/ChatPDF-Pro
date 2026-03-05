"""LLM 响应缓存服务

基于请求参数的 hash 做缓存，避免重复 LLM API 调用。
主要用于 GraphRAG 构建（实体提取、社区报告等）中大量重复性 LLM 调用。

缓存存储方式：JSON 文件（每个 working_dir 一个缓存文件）。
"""

import json
import logging
import os
from hashlib import md5
from typing import Optional, Union

logger = logging.getLogger(__name__)


class LLMCache:
    """基于文件的 LLM 响应缓存"""

    def __init__(self, cache_dir: str = "data/llm_cache"):
        self._cache_dir = cache_dir
        self._data: dict[str, dict] = {}
        self._file_path = os.path.join(cache_dir, "llm_cache.json")
        self._dirty = False
        self._load()

    def _load(self):
        """从磁盘加载缓存"""
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(f"[LLMCache] 加载 {len(self._data)} 条缓存")
            except Exception as e:
                logger.warning(f"[LLMCache] 加载缓存失败: {e}")
                self._data = {}

    def save(self):
        """持久化缓存到磁盘"""
        if not self._dirty:
            return
        os.makedirs(self._cache_dir, exist_ok=True)
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            self._dirty = False
            logger.debug(f"[LLMCache] 保存 {len(self._data)} 条缓存")
        except Exception as e:
            logger.warning(f"[LLMCache] 保存缓存失败: {e}")

    @staticmethod
    def compute_hash(model: str, messages: list[dict], **kwargs) -> str:
        """计算请求参数的哈希值"""
        key_parts = {
            "model": model,
            "messages": messages,
        }
        if kwargs:
            key_parts.update({k: v for k, v in kwargs.items() if v is not None})
        return md5(json.dumps(key_parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[str]:
        """查询缓存"""
        entry = self._data.get(cache_key)
        if entry is not None:
            return entry.get("response")
        return None

    def put(self, cache_key: str, response: str, model: str = ""):
        """写入缓存"""
        self._data[cache_key] = {
            "response": response,
            "model": model,
        }
        self._dirty = True

    def clear(self):
        """清空缓存"""
        self._data = {}
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._data)


# 模块级单例
_cache_instances: dict[str, LLMCache] = {}


def get_llm_cache(cache_dir: str = "data/llm_cache") -> LLMCache:
    """获取或创建 LLM 缓存实例（按 cache_dir 区分）"""
    if cache_dir not in _cache_instances:
        _cache_instances[cache_dir] = LLMCache(cache_dir)
    return _cache_instances[cache_dir]
