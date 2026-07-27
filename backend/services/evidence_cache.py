"""证据评分缓存（paper-qa 风格 RCS 的复用层）

`evidence_scorer` 每条证据都要一次 LLM 调用来产出 summary + relevance_score。
同一份证据在这些场景里会被反复重打分：

- 用户点「重新生成」
- 流式中断后重试
- Agent 多轮迭代里同一 chunk 再次进入候选池
- 用户把同一个问题再问一遍

这里按 (解析身份, 问题, 证据) 缓存打分结果，命中即跳过 LLM。

**刻意不做"相似问题复用"**：summary 是 *面向该问题* 生成的压缩，
把为"表2的F1是多少"生成的摘要拿去回答"表2用了什么数据集"会给出误导性证据。
paper-qa 自己也是把 question 揉进 Context.id 的。想让摘要跨问题起作用，
正确做法是把它作为一条带 question 标注的记忆注入，让模型看得见它的原始问题，
而不是在评分层静默复用。
"""
import hashlib
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_CAPACITY = 2000


def normalize_question(question: str) -> str:
    """问题归一：忽略空白与大小写差异，但不做任何语义等价判断。"""
    return re.sub(r"\s+", " ", str(question or "")).strip().casefold()


def make_cache_key(
    *,
    question: str,
    evidence_id: str,
    doc_id: str = "",
    parse_generation: str = "",
) -> str:
    """缓存键含解析身份：文档重新解析后旧摘要必须失效。"""
    normalized = normalize_question(question)
    if not normalized or not evidence_id:
        return ""
    raw = " ".join([
        str(doc_id or ""),
        str(parse_generation or ""),
        normalized,
        str(evidence_id),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class EvidenceScoreCache:
    """(解析身份, 问题, 证据) → (summary, score) 的 LRU 缓存。

    只放在内存里：证据摘要是可再生的派生数据，重启后重算即可，
    没必要为它引入落盘格式与迁移负担。
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self.capacity = max(1, int(capacity))
        self._store: "OrderedDict[str, tuple[str, int]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[tuple[str, int]]:
        if not key:
            return None
        with self._lock:
            value = self._store.get(key)
            if value is None:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, summary: str, score: int) -> None:
        if not key:
            return
        with self._lock:
            self._store[key] = (str(summary or ""), int(score or 0))
            self._store.move_to_end(key)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def invalidate_prefix_free(self) -> None:
        """整体清空（换文档解析代际时由调用方触发）。"""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "capacity": self.capacity,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }


# 进程级单例：评分发生在多个请求链路里，共享一份缓存才有意义
_GLOBAL_CACHE: Optional[EvidenceScoreCache] = None
_GLOBAL_LOCK = threading.Lock()


def get_evidence_cache() -> EvidenceScoreCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_CACHE is None:
                capacity = DEFAULT_CAPACITY
                try:
                    from config import settings
                    capacity = int(settings.evidence_score_cache_size)
                except Exception:
                    pass
                _GLOBAL_CACHE = EvidenceScoreCache(capacity)
    return _GLOBAL_CACHE


def reset_evidence_cache() -> None:
    """仅供测试与「清空记忆」路径使用。"""
    global _GLOBAL_CACHE
    with _GLOBAL_LOCK:
        _GLOBAL_CACHE = None


def is_cache_enabled() -> bool:
    try:
        from config import settings
        return bool(settings.evidence_score_cache_enabled)
    except Exception:
        return True
