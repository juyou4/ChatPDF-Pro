"""
记忆管理核心服务

整合 MemoryStore、MemoryIndex、MemoryRetriever、KeywordExtractor，
提供统一的记忆管理业务接口。

核心功能：
- 记忆检索：检索相关记忆并返回格式化上下文
- 记忆写入：保存 QA 摘要、重要记忆、关键词更新
- CRUD 操作：增删改查记忆条目
- 摘要上限控制：超过上限时移除最早的非重要摘要
"""
import hashlib
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Optional

from services.keyword_extractor import KeywordExtractor
from services.memory_index import MemoryIndex
from services.memory_llm import (
    MemoryLLMBudget,
    call_llm_sync,
    consume_budget,
    parse_bullet_list,
)
from services.memory_quality import (
    is_unusable_automatic_answer,
    sanitize_automatic_memory_content,
    is_unsafe_automatic_document_answer,
    is_unscoped_document_absence_claim,
)
from services.memory_retriever import MemoryRetriever
from services.memory_store import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)

# 延迟导入新模块的辅助函数，避免循环依赖
def _import_new_modules():
    """延迟导入新增模块，返回 (MemoryTagger, MemoryCompressor, ContextInjector, ActivePool)"""
    from services.memory_tagger import MemoryTagger
    from services.memory_compressor import MemoryCompressor
    from services.context_injector import ContextInjector
    from services.active_pool import ActivePool
    return MemoryTagger, MemoryCompressor, ContextInjector, ActivePool

# 默认配置
DEFAULT_MAX_SUMMARIES = 50  # QA 摘要数量上限
DEFAULT_KEYWORD_THRESHOLD = 3  # 关键词频率阈值
DEFAULT_RETRIEVAL_TOP_K = 3  # 记忆检索返回条数
QUESTION_MAX_LEN = 100  # 问题截取最大长度
ANSWER_MAX_LEN = 200  # 回答截取最大长度

# 文档内容会随着重新解析切换 generation。自动生成的文档记忆必须绑定
# 当时的解析身份；用户主动保存/点赞的记忆则是用户意图，不应随之丢失。
_AUTOMATIC_DOCUMENT_MEMORY_SOURCE_TYPES = {
    "auto_qa",
    "llm_distilled",
    "compressed",
    # A rolling summary contains model-derived conclusions from a particular
    # parse generation. It must not survive a local <-> MinerU route switch.
    "session_summary",
}

# 提炼模型表示"本轮没有值得记住的内容"时使用的哨兵回复。
_NO_FACT_SENTINELS = {"无", "none", "no important fact", "n/a", "null", "nothing"}

# 用户显式的记忆指令。命中后该轮记忆直接按用户主动保存对待，
# 不必再等命中次数累积到晋升阈值。
_EXPLICIT_MEMORY_REQUEST_RE = re.compile(
    r"记住|记一下|记下来|别忘|不要忘|以后都|以后请|从现在起|今后|一直用|"
    r"我的偏好|我偏好|默认就|下次也|不要再|别再|"
    r"remember (?:that|this)|keep in mind|from now on|always |never ",
    re.IGNORECASE,
)


def _is_no_fact_sentinel(text: str) -> bool:
    """判断提炼结果是否为"无事实"哨兵，容忍标点与大小写差异。"""
    normalized = str(text or "").strip().strip("。.!！,，:：\"'`*")
    if not normalized:
        return True
    return normalized.lower() in _NO_FACT_SENTINELS

_DOCUMENT_ABSENCE_RE = re.compile(
    r"未(?:给出|说明|提供|公开)|没有(?:给出|说明|提供)|未披露|不(?:清楚|明确)|无法(?:确认|得知)|"
    r"\b(?:does not|doesn't|did not|has not|have not)\s+(?:give|provide|describe|specify|disclose)\b|"
    r"\bnot\s+(?:given|provided|described|specified|disclosed)\b|\b(?:unclear|unknown)\b",
    re.IGNORECASE,
)
_ARCHITECTURE_BROAD_RE = re.compile(
    r"架构|结构|拓扑|机制|交互|流程|网络|检测头|"
    r"architecture|structure|topology|mechanism|interaction|pipeline|network|detector",
    re.IGNORECASE,
)
_ARCHITECTURE_DETAIL_SCOPE_RE = re.compile(
    r"逐层|层数|通道|维度|张量|投影|归一化|配置|超参|具体(?:模块|字段|实现)|实现细节|"
    r"\b(?:layer(?:s)?|channel(?:s)?|dimension(?:s)?|tensor(?:s)?|projection|normalization|"
    r"configuration|hyperparameter(?:s)?|implementation(?:\s+details?)?)\b",
    re.IGNORECASE,
)


def _is_unscoped_architecture_absence_fact(text: str) -> bool:
    """拒绝把“没有结构”这类无边界否定写成文档事实。"""
    normalized = str(text or "").strip()
    return bool(
        normalized
        and _DOCUMENT_ABSENCE_RE.search(normalized)
        and _ARCHITECTURE_BROAD_RE.search(normalized)
        and not _ARCHITECTURE_DETAIL_SCOPE_RE.search(normalized)
    )


def has_explicit_memory_request(text: str) -> bool:
    """用户是否在本轮明确要求记住某件事。"""
    if not text:
        return False
    return bool(_EXPLICIT_MEMORY_REQUEST_RE.search(str(text)))


def _serialized_memory_mutation(method):
    """Serialize a service-level mutation with delayed memory publishers."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._store_mutation_lock:
            return method(self, *args, **kwargs)

    return wrapped


class MemoryService:
    """记忆管理核心服务（单例）"""

    def __init__(self, data_dir: str, embedding_model_id: str = "local-minilm", use_sqlite: bool = False):
        """
        初始化记忆管理服务

        Args:
            data_dir: 记忆数据根目录，如 "data/memory/"
            embedding_model_id: embedding 模型 ID
            use_sqlite: 是否使用 SQLite 存储（可选增强）
        """
        self.data_dir = data_dir
        # Automatic memory work may outlive the originating chat request. A
        # full clear advances this generation so delayed work cannot recreate
        # sessions, daily Markdown, event records, or vector entries.
        self._write_generation_lock = threading.RLock()
        self._store_mutation_lock = threading.RLock()
        self._write_generation = 0
        # A document-level clear must only fence delayed writers for that
        # document. The global generation remains for ``clear_all``.
        self._document_write_generations: dict[str, int] = {}
        
        # 根据配置选择存储后端
        if use_sqlite:
            try:
                from services.memory_store_sqlite import MemoryStoreSQLite
                self.store = MemoryStoreSQLite(data_dir, use_sqlite=True)
                logger.info("启用 SQLite FTS 镜像；JSON/事件快照仍是生命周期权威源")
            except Exception as e:
                logger.warning(f"SQLite 存储初始化失败，回退到 JSON: {e}")
                from services.memory_store import MemoryStore
                self.store = MemoryStore(data_dir)
        else:
            from services.memory_store import MemoryStore
            self.store = MemoryStore(data_dir)
        
        self.index = MemoryIndex(
            os.path.join(data_dir, "memory_index"), embedding_model_id
        )
        # 审计日志与主存储解耦：记忆被删掉后，它的演化历史仍然查得到。
        try:
            from services.memory_audit_log import MemoryAuditLog
            self.audit_log = MemoryAuditLog(data_dir)
        except Exception as exc:
            logger.warning(f"[MemoryService] 审计日志初始化失败，跳过审计: {exc}")
            self.audit_log = None
        self.keyword_extractor = KeywordExtractor()
        # doc_id -> {signature, fact_count, summary}；图谱抽取的结果缓存，
        # 让检索热路径不必也不会触发 LLM 调用。
        self._graph_cache: dict[str, dict[str, Any]] = {}
        self.max_summaries = DEFAULT_MAX_SUMMARIES
        self.keyword_threshold = DEFAULT_KEYWORD_THRESHOLD

        # 初始化新增模块（使用 _safe_execute 确保优雅降级）
        self.tagger = None
        self.compressor = None
        self.context_injector = None
        self.active_pool = None
        try:
            from config import settings as app_settings
        except Exception:
            app_settings = None

        try:
            MemoryTagger, MemoryCompressor, ContextInjector, ActivePool = _import_new_modules()
            self.tagger = MemoryTagger()
            compression_threshold = getattr(app_settings, "memory_compression_threshold", 20) if app_settings else 20
            self.compressor = MemoryCompressor(
                compression_threshold=compression_threshold,
                max_compressed=getattr(app_settings, "memory_compression_max_items", 5) if app_settings else 5,
                oversized_chars=getattr(app_settings, "memory_compression_oversized_chars", 2000) if app_settings else 2000,
            )
            token_budget = getattr(app_settings, "memory_injection_token_budget", 800) if app_settings else 800
            kind_budgets = getattr(app_settings, "memory_injection_kind_budgets", None) if app_settings else None
            self.context_injector = ContextInjector(
                token_budget=token_budget,
                kind_budgets=dict(kind_budgets) if kind_budgets else None,
            )
            pool_size = getattr(app_settings, "memory_active_pool_size", 100) if app_settings else 100
            self.active_pool = ActivePool(capacity=pool_size)
        except Exception as e:
            logger.warning(f"[MemoryService] 新增模块初始化失败，降级为基础功能: {e}")

        # 写入裁决器：新事实与既有记忆比对后决定 ADD/UPDATE/DELETE/NONE
        self.arbiter = None
        try:
            from services.memory_arbiter import MemoryArbiter
            self.arbiter = MemoryArbiter()
        except Exception as exc:
            logger.warning(f"[MemoryService] 裁决器初始化失败，写入退回纯追加: {exc}")

        # 初始化检索器（传入 active_pool）
        self.retriever = MemoryRetriever(self.store, self.index, active_pool=self.active_pool)

        # 尝试加载已有的向量索引
        loaded = self.index.load()
        if not loaded:
            self._safe_execute("MemoryIndex.recover", self._recover_index_from_store)

        # 预加载活跃记忆池
        self._safe_execute(
            "MemoryGuard.quarantine",
            self._quarantine_unscoped_architecture_absence_memories,
        )
        self._safe_execute("ActivePool.preload", self._preload_active_pool)

    def capture_write_generation(self, doc_id: str | None = None) -> int | tuple[int, str, int]:
        """Capture the global/document fence for a delayed memory writer."""
        with self._write_generation_lock:
            if doc_id:
                return (
                    self._write_generation,
                    str(doc_id),
                    self._document_write_generations.get(str(doc_id), 0),
                )
            return self._write_generation

    def is_write_generation_current(
        self,
        expected_generation: int | tuple[int, str, int] | None,
        *,
        doc_id: str | None = None,
    ) -> bool:
        """Return whether a delayed writer still belongs to the active store.

        Older callers may still pass a plain global integer. New document
        writers carry ``(global_generation, doc_id, doc_generation)`` so a
        document-local clear cannot be undone by a late background task.
        """
        if expected_generation is None:
            return True
        with self._write_generation_lock:
            if isinstance(expected_generation, tuple) and len(expected_generation) == 3:
                global_generation, expected_doc_id, document_generation = expected_generation
                if doc_id is not None and str(doc_id) != str(expected_doc_id):
                    return False
                return (
                    int(global_generation) == self._write_generation
                    and self._document_write_generations.get(str(expected_doc_id), 0)
                    == int(document_generation)
                )
            try:
                return int(expected_generation) == self._write_generation
            except (TypeError, ValueError):
                return False

    def _global_generation_from_fence(
        self,
        expected_generation: int | tuple[int, str, int] | None,
    ) -> int | None:
        """Extract the global component for profile-scoped writes."""
        if expected_generation is None:
            return None
        if isinstance(expected_generation, tuple) and len(expected_generation) == 3:
            return int(expected_generation[0])
        try:
            return int(expected_generation)
        except (TypeError, ValueError):
            return None

    # ==================== 安全执行与预加载 ====================

    def _safe_execute(self, component_name: str, func, *args, **kwargs):
        """安全执行组件方法，异常时降级

        Args:
            component_name: 组件名称（用于日志）
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值，异常时返回 None
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[{component_name}] 执行失败，降级处理: {e}")
            return None

    @staticmethod
    def _is_unscoped_architecture_absence_automatic_memory(
        source_type: str,
        content: str,
    ) -> bool:
        return bool(
            str(source_type or "").strip().lower()
            in _AUTOMATIC_DOCUMENT_MEMORY_SOURCE_TYPES
            and is_unsafe_automatic_document_answer(content)
        )

    def _quarantine_unscoped_architecture_absence_memories(self) -> int:
        """Invalidate pre-guard automatic document facts without deleting them."""
        candidates = [
            entry
            for entry in self.store.get_all_entries()
            if entry.doc_id
            and entry.is_retrievable
            and self._is_unscoped_architecture_absence_automatic_memory(
                entry.source_type,
                entry.content,
            )
        ]
        invalidated = 0
        for entry in candidates:
            if self.invalidate_entry(
                entry.id,
                reason="unsafe_document_absence_guard",
                actor="system_guard",
            ):
                invalidated += 1
        if invalidated:
            logger.warning(
                "[MemoryGuard] invalidated %d pre-existing unsafe document claims",
                invalidated,
            )
        return invalidated

    def _preload_active_pool(self) -> None:
        """预加载活跃记忆池（服务启动时调用）

        从存储中加载最近使用的记忆条目到 Active_Pool。
        """
        if not self.active_pool:
            return
        try:
            all_entries = self.store.get_all_entries()
            all_entries = [
                entry
                for entry in all_entries
                if entry.is_retrievable
                and not self._is_unscoped_architecture_absence_automatic_memory(
                    entry.source_type,
                    entry.content,
                )
            ]
            # 按 last_hit_at 降序排列，取前 N 条
            sorted_entries = sorted(
                all_entries,
                key=lambda e: e.last_hit_at or "",
                reverse=True,
            )
            self.active_pool.preload(sorted_entries[:self.active_pool.capacity])
        except Exception as e:
            logger.warning(f"[ActivePool] 预加载失败: {e}")

    def _recover_index_from_store(self) -> None:
        """当索引缺失或损坏时，从存储快照/事件回放结果重建索引。"""
        entries = [
            entry for entry in self.store.get_all_entries()
            if entry.status != "archived_raw"
            and entry.is_retrievable
            and not self._is_unscoped_architecture_absence_automatic_memory(
                entry.source_type,
                entry.content,
            )
        ]
        if not entries:
            logger.info("[MemoryIndex] 无可恢复条目，跳过索引重建")
            return
        self.index.safe_reindex(entries, reason="recover")
        self.index.flush_sync(reason="manual")
        logger.info(f"[MemoryIndex] 已从存储恢复并重建索引，共 {len(entries)} 条")

    def _page_in_active_pool(self, entry: MemoryEntry | None) -> None:
        """将可缓存记忆放入热池，供后续检索快路径复用。"""
        if not self.active_pool or entry is None:
            return
        try:
            self.active_pool.put(entry)
        except Exception as exc:
            logger.debug(f"Active_Pool Page-In 失败: {exc}")

    @staticmethod
    def _truncate_text(text: str, limit: int = 180) -> str:
        normalized = " ".join((text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit] + "..."

    def _build_memory_title(self, entry: MemoryEntry) -> str:
        if entry.title:
            return entry.title
        if entry.memory_kind == "consolidated":
            return "压缩记忆"
        if entry.memory_kind == "doc_fact":
            return "文档事实"
        if entry.memory_scope == "profile":
            return "用户画像"
        first_line = (entry.content or "").splitlines()[0].strip()
        return first_line[:60] if first_line else "记忆条目"

    def _build_memory_hit(
        self,
        entry: MemoryEntry,
        *,
        score: float = 0.0,
        query: str = "",
        content_override: str = "",
    ) -> dict[str, Any]:
        trace = dict(entry.trace or {})
        if query and "query" not in trace:
            trace["query"] = query
        summary = entry.summary or self._truncate_text(content_override or entry.content)
        return {
            "id": entry.id,
            "entry_id": entry.id,
            "content": content_override or entry.content,
            "source_type": entry.source_type,
            "doc_id": entry.doc_id,
            "score": score,
            "rrf_score": score,
            "importance": entry.importance,
            "memory_tier": entry.memory_tier,
            "memory_kind": entry.memory_kind,
            "memory_scope": entry.memory_scope,
            "status": entry.status,
            "title": self._build_memory_title(entry),
            "summary": summary,
            "created_at": entry.created_at,
            "tags": list(entry.tags or []),
            "source_ref": dict(entry.source_ref or {}),
            "derived_from": list(entry.derived_from or []),
            "trace": trace,
            "last_used_query": entry.last_used_query or query or "",
            "valid_at": entry.valid_at or entry.created_at,
            "invalid_at": entry.invalid_at,
            "disabled_at": entry.disabled_at,
            "retrievable": entry.is_retrievable,
        }

    def _serialize_entry(self, entry: MemoryEntry, include_content: bool = True) -> dict[str, Any]:
        payload = self._build_memory_hit(entry, score=0.0, query=entry.last_used_query or "")
        if not include_content:
            payload.pop("content", None)
        return payload

    def _entry_sort_key(self, entry: MemoryEntry) -> tuple:
        return (entry.created_at or "", entry.id)

    def _get_entry_map(self) -> dict[str, MemoryEntry]:
        return {entry.id: entry for entry in self.store.get_all_entries()}

    def _find_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._get_entry_map().get(entry_id)

    @staticmethod
    def _normalize_parse_identity(parse_identity: dict | None) -> dict[str, str] | None:
        """Normalize the parse identity accepted from document-facing callers.

        ``generation``/``source_hash`` are the manifest names; the persisted
        source reference uses the artifact-wide ``parse_generation`` /
        ``document_source_hash`` pair. Accept both to keep service callers
        small and legacy JSON untouched.
        """
        if not isinstance(parse_identity, dict):
            return None
        generation = str(
            parse_identity.get("parse_generation")
            or parse_identity.get("generation")
            or ""
        ).strip()
        source_hash = str(
            parse_identity.get("document_source_hash")
            or parse_identity.get("source_hash")
            or ""
        ).strip()
        if not generation or not source_hash:
            return None
        return {
            "parse_generation": generation,
            "document_source_hash": source_hash,
        }

    @classmethod
    def _entry_matches_parse_identity(
        cls,
        entry: MemoryEntry | None,
        *,
        doc_id: str | None,
        parse_identity: dict | None,
    ) -> bool:
        """Keep user-curated memories while fencing automatic document facts."""
        identity = cls._normalize_parse_identity(parse_identity)
        if identity is None or entry is None or not doc_id or entry.doc_id != doc_id:
            return True
        if entry.source_type not in _AUTOMATIC_DOCUMENT_MEMORY_SOURCE_TYPES:
            return True
        source_ref = entry.source_ref if isinstance(entry.source_ref, dict) else {}
        return (
            str(source_ref.get("parse_generation") or "") == identity["parse_generation"]
            and str(
                source_ref.get("document_source_hash")
                or source_ref.get("source_hash")
                or ""
            ) == identity["document_source_hash"]
        )

    @classmethod
    def _filter_retrieved_memories_for_parse_identity(
        cls,
        memories: list[dict],
        *,
        entry_map: dict[str, MemoryEntry],
        doc_id: str | None,
        parse_identity: dict | None,
    ) -> list[dict]:
        """Drop stale automatic hits returned by a shared memory index."""
        identity = cls._normalize_parse_identity(parse_identity)

        filtered: list[dict] = []
        for memory in memories or []:
            entry = entry_map.get(memory.get("entry_id", "")) if isinstance(memory, dict) else None
            source_type = str(
                entry.source_type
                if entry is not None
                else (memory.get("source_type") if isinstance(memory, dict) else "")
                or ""
            )
            content = str(
                entry.content
                if entry is not None
                else (
                    memory.get("text") or memory.get("content") or ""
                    if isinstance(memory, dict)
                    else ""
                )
            )
            if source_type in _AUTOMATIC_DOCUMENT_MEMORY_SOURCE_TYPES:
                if is_unsafe_automatic_document_answer(content):
                    logger.info("[MemoryGuard] suppressing unsafe automatic memory hit")
                    continue
                content = sanitize_automatic_memory_content(content, source_type)
                if not content:
                    continue
                if isinstance(memory, dict):
                    memory = dict(memory)
                    memory["text"] = content
                    memory["content"] = content
            if identity is None or not doc_id:
                filtered.append(memory)
                continue
            if entry is not None:
                if cls._entry_matches_parse_identity(
                    entry,
                    doc_id=doc_id,
                    parse_identity=identity,
                ):
                    filtered.append(memory)
                continue

            # An orphaned index hit cannot prove the generation. Only reject
            # automatic document memories; manual/liked hits remain durable.
            if (
                isinstance(memory, dict)
                and memory.get("doc_id") == doc_id
                and memory.get("source_type") in _AUTOMATIC_DOCUMENT_MEMORY_SOURCE_TYPES
            ):
                continue
            filtered.append(memory)
        return filtered

    def _build_working_memory_hits(self, chat_history: list[dict] | None, doc_id: str | None = None) -> list[dict[str, Any]]:
        """将最近若干轮对话转成工作记忆命中。

        窗口大小取 ``memory_working_window_size``。命中按"越近分越高"排序，
        这样 ContextInjector 在 working 配额装不下全部轮次时保留最近的对话，
        而不是被稳定排序留下最旧的几轮。
        """
        if not chat_history:
            return []
        working_messages = self.get_working_memory(chat_history)
        if not working_messages:
            return []

        rounds: list[tuple[dict, dict | None]] = []
        idx = 0
        while idx < len(working_messages):
            user_msg = working_messages[idx]
            assistant_msg = working_messages[idx + 1] if idx + 1 < len(working_messages) else None
            if user_msg.get("role") == "user":
                rounds.append((user_msg, assistant_msg if assistant_msg and assistant_msg.get("role") == "assistant" else None))
            idx += 2

        hits: list[dict[str, Any]] = []
        for round_idx, (user_msg, assistant_msg) in enumerate(reversed(rounds)):
            question = (user_msg or {}).get("content", "").strip()
            answer = (assistant_msg or {}).get("content", "").strip()
            if not question or is_unusable_automatic_answer(answer):
                continue
            if is_unsafe_automatic_document_answer(answer):
                continue
            content = f"Q: {question}\nA: {answer}".strip()
            if not content:
                continue
            # round_idx=0 是最近一轮；分数随距离递减但保持在 profile 之上。
            recency_score = max(0.5, 1.0 - round_idx * 0.05)
            synthetic = MemoryEntry(
                id=f"working-{round_idx}-{abs(hash(content))}",
                content=content,
                source_type="working_memory",
                doc_id=doc_id,
                importance=1.0,
                memory_tier="working",
                memory_kind="working",
                memory_scope="document" if doc_id else "profile",
                title=question[:60] or "最近对话",
                summary=self._truncate_text(content),
                source_ref={"question": question[:200]} if question else {},
                trace={"kind": "working_memory", "round": round_idx},
            )
            hits.append(self._build_memory_hit(synthetic, score=recency_score, query=question))
        return hits

    @staticmethod
    def _detect_graph_memory_needed(query: str) -> bool:
        lowered = (query or "").lower()
        keywords = (
            "图", "figure", "fig", "表", "table", "方法", "framework", "pipeline",
            "dataset", "数据集", "metric", "指标", "结论", "compare", "对比",
        )
        return any(keyword in lowered for keyword in keywords)

    def _build_graph_memory_hits(
        self,
        doc_id: str | None,
        query: str,
        parse_identity: dict | None = None,
    ) -> list[dict[str, Any]]:
        """根据论文图谱摘要生成轻量 graph memory 命中。"""
        if not doc_id or not self._detect_graph_memory_needed(query):
            return []

        graph = self.get_graph_summary(doc_id, parse_identity=parse_identity)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        if not nodes:
            return []

        lowered_query = (query or "").lower()
        matched_nodes = [
            node for node in nodes
            if node.get("label", "").lower() in lowered_query or any(token and token in lowered_query for token in node.get("label", "").lower().split())
        ]
        if not matched_nodes:
            # 关键词门很宽（"表/方法/指标/结论"都算命中），但没有实体对得上就说明
            # 图谱帮不上这个问题。此时注入任意节点只是噪声，直接放弃这一路。
            return []

        matched_ids = {node["id"] for node in matched_nodes}
        matched_edges = [
            edge for edge in edges
            if edge.get("source") in matched_ids or edge.get("target") in matched_ids
        ][:4]

        summary_lines = [
            "图谱节点：" + "；".join(f"{node['type']}={node['label']}" for node in matched_nodes[:4]),
        ]
        if matched_edges:
            summary_lines.append(
                "图谱关系：" + "；".join(
                    f"{edge['source']} -{edge['type']}-> {edge['target']}" for edge in matched_edges
                )
            )
        content = "\n".join(summary_lines)
        synthetic = MemoryEntry(
            id=f"graph-{doc_id}-{abs(hash(content))}",
            content=content,
            source_type="graph_summary",
            doc_id=doc_id,
            importance=0.8,
            memory_tier="long_term",
            memory_kind="graph",
            memory_scope="document",
            title="论文图谱摘要",
            summary=self._truncate_text(content),
            trace={"kind": "graph_summary", "doc_id": doc_id},
        )
        hit = self._build_memory_hit(synthetic, score=0.75, query=query)
        hit["graph"] = {
            "matched_nodes": matched_nodes,
            "matched_edges": matched_edges,
        }
        return [hit]

    @staticmethod
    def _dedupe_memory_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            hit_id = hit.get("id") or hit.get("entry_id")
            if not hit_id or hit_id in seen:
                continue
            seen.add(hit_id)
            ordered.append(hit)
        return ordered

    # ==================== 记忆检索 ====================

    def retrieve_memories(
        self, query: str, top_k: int = DEFAULT_RETRIEVAL_TOP_K, api_key: str = None,
        doc_id: str = None, filter_by_doc: bool = False,
        parse_identity: dict | None = None,
    ) -> str:
        """检索相关记忆并返回格式化的上下文字符串

        Args:
            query: 用户查询文本
            top_k: 返回的最大结果数
            api_key: API 密钥（远程模型需要）
            doc_id: 当前文档 ID，用于文档相关性加权（可选）
            filter_by_doc: 是否只返回当前文档的记忆，默认 False（仅加权）
            parse_identity: 当前文档解析身份；提供后自动文档记忆必须匹配

        Returns:
            格式化的记忆上下文字符串，无记忆时返回空字符串
        """
        try:
            # 定期评估记忆重要性（每 10 次检索评估一次，避免频繁计算）
            import random
            if random.random() < 0.1:  # 10% 概率触发评估
                try:
                    self.evaluate_and_update_importance()
                except Exception as e:
                    logger.debug(f"定期重要性评估失败（不影响检索）: {e}")
            
            entry_map = {e.id: e for e in self.store.get_all_entries()}
            identity = self._normalize_parse_identity(parse_identity)
            retrieval_top_k = max(top_k * 4, top_k + 8) if identity and doc_id else top_k
            memories = self.retriever.retrieve(
                query, top_k=retrieval_top_k, api_key=api_key,
                doc_id=doc_id, filter_by_doc=filter_by_doc
            )
            memories = self._filter_retrieved_memories_for_parse_identity(
                memories,
                entry_map=entry_map,
                doc_id=doc_id,
                parse_identity=identity,
            )[:top_k]
            
            # 检索后触发晋升检查
            for mem in memories:
                try:
                    entry = entry_map.get(mem.get("entry_id", "")) if isinstance(mem, dict) else mem
                    if entry:
                        self.check_and_promote(entry)
                except Exception as e:
                    logger.debug(f"晋升检查失败（不影响检索）: {e}")

            # 检索命中的记忆加入 Active_Pool（Page-In）
            if self.active_pool and memories:
                for mem in memories:
                    try:
                        entry = entry_map.get(mem.get("entry_id", "")) if isinstance(mem, dict) else mem
                        if entry:
                            self.active_pool.put(entry)
                    except Exception as e:
                        logger.debug(f"Active_Pool Page-In 失败: {e}")
            
            return self.retriever.build_memory_context(memories)
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return ""

    def retrieve_memories_raw(
        self, query: str, top_k: int = DEFAULT_RETRIEVAL_TOP_K, api_key: str = None,
        doc_id: str = None, filter_by_doc: bool = False, chat_history: list[dict] | None = None,
        parse_identity: dict | None = None,
    ) -> list[dict]:
        """检索相关记忆并返回原始记忆列表（供 ContextInjector 使用）

        Args:
            query: 用户查询文本
            top_k: 返回的最大结果数
            api_key: API 密钥
            doc_id: 当前文档 ID
            filter_by_doc: 是否只返回当前文档的记忆
            parse_identity: 当前文档解析身份；提供后自动文档记忆必须匹配

        Returns:
            记忆字典列表，每条包含 content, memory_tier, importance 等字段
        """
        try:
            working_hits = self._build_working_memory_hits(chat_history, doc_id=doc_id)
            all_entries = self.store.get_all_entries()
            entry_map = {e.id: e for e in all_entries}
            identity = self._normalize_parse_identity(parse_identity)
            retrieval_top_k = max(top_k * 4, top_k + 8) if identity and doc_id else top_k
            memories = self.retriever.retrieve(
                query, top_k=retrieval_top_k, api_key=api_key,
                doc_id=doc_id, filter_by_doc=filter_by_doc
            )
            memories = self._filter_retrieved_memories_for_parse_identity(
                memories,
                entry_map=entry_map,
                doc_id=doc_id,
                parse_identity=identity,
            )[:top_k]
            enriched = []
            for mem in memories:
                entry_id = mem.get("entry_id", "")
                entry = entry_map.get(entry_id)
                if entry:
                    enriched_mem = self._build_memory_hit(
                        entry,
                        score=mem.get("rrf_score", 0.0),
                        query=query,
                        content_override=mem.get("text", entry.content),
                    )
                else:
                    fallback_entry = MemoryEntry(
                        id=entry_id or str(uuid.uuid4()),
                        content=mem.get("text", ""),
                        source_type=mem.get("source_type", "manual"),
                        doc_id=mem.get("doc_id"),
                    )
                    enriched_mem = self._build_memory_hit(
                        fallback_entry,
                        score=mem.get("rrf_score", 0.0),
                        query=query,
                        content_override=mem.get("text", ""),
                    )
                if mem.get("from_archive"):
                    # 让用户看得出这条是压缩归档里回捞出来的兜底证据
                    enriched_mem["from_archive"] = True
                    enriched_mem["title"] = f"（归档回捞）{enriched_mem.get('title', '')}".strip()
                enriched.append(enriched_mem)
            graph_hits = self._build_graph_memory_hits(doc_id, query, parse_identity=identity)
            summary_hits = self._build_session_summary_hits(doc_id, query, parse_identity=identity)
            return self._dedupe_memory_hits(
                [*working_hits, *summary_hits, *graph_hits, *enriched]
            )
        except Exception as e:
            logger.error(f"记忆原始检索失败: {e}")
            return []

    # ==================== 分层记忆架构 ====================

    @_serialized_memory_mutation
    def check_and_promote(self, entry: MemoryEntry) -> None:
        """检查并执行记忆晋升

        晋升条件：
        - memory_tier 为 "short_term"
        - hit_count >= 晋升阈值（默认 5）
        - last_hit_at 在最近 7 天内

        Args:
            entry: 待检查的记忆条目
        """
        if entry.memory_tier != "short_term":
            return

        # 读取配置的晋升阈值
        try:
            from config import settings
            promotion_threshold = settings.memory_promotion_threshold
        except Exception:
            promotion_threshold = 5

        if entry.hit_count < promotion_threshold:
            return

        # 检查 last_hit_at 是否在最近 7 天内
        if not entry.last_hit_at:
            return

        try:
            last_hit_time = datetime.fromisoformat(entry.last_hit_at)
            if last_hit_time.tzinfo is None:
                last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return

        now = datetime.now(timezone.utc)
        if (now - last_hit_time) > timedelta(days=7):
            return

        # 满足晋升条件，更新 memory_tier
        entry.memory_tier = "long_term"
        self._persist_tier_change(entry)
        logger.info(f"记忆晋升: {entry.id} short_term -> long_term (hit_count={entry.hit_count})")

    def check_and_demote(self) -> None:
        """检查并执行记忆降级

        降级条件：
        - memory_tier 为 "long_term"
        - 距离 last_hit_at 超过降级天数（默认 90 天）
        - hit_count < 3
        """
        try:
            from config import settings
            demotion_days = settings.memory_demotion_days
        except Exception:
            demotion_days = 90

        now = datetime.now(timezone.utc)
        all_entries = self.store.get_all_entries()
        demoted_count = 0

        for entry in all_entries:
            if entry.memory_tier != "long_term":
                continue
            if entry.hit_count >= 3:
                continue

            # 计算距离 last_hit_at 的天数
            last_hit_time = None
            if entry.last_hit_at:
                try:
                    last_hit_time = datetime.fromisoformat(entry.last_hit_at)
                    if last_hit_time.tzinfo is None:
                        last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            if not last_hit_time:
                # 没有命中记录，使用创建时间
                try:
                    last_hit_time = datetime.fromisoformat(entry.created_at)
                    if last_hit_time.tzinfo is None:
                        last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

            days_since_hit = (now - last_hit_time).total_seconds() / 86400.0
            if days_since_hit > demotion_days:
                entry.memory_tier = "archived"
                self._persist_tier_change(entry)
                demoted_count += 1

        if demoted_count > 0:
            logger.info(f"记忆降级: {demoted_count} 条记忆 long_term -> archived")

    def _persist_tier_change(self, entry: MemoryEntry) -> None:
        """持久化记忆层级变更到存储

        在 profile 的 entries 和 session 的 important_memories 中查找并更新 memory_tier。

        Args:
            entry: 已更新 memory_tier 的记忆条目
        """
        # 在 profile 中查找
        profile = self.store.load_profile()
        for item in profile.get("entries", []):
            if item.get("id") == entry.id:
                item["memory_tier"] = entry.memory_tier
                self.store.save_profile(profile)
                return

        # 在 session 中查找
        if entry.doc_id:
            session = self.store.load_session(entry.doc_id)
            for item in session.get("important_memories", []):
                if item.get("id") == entry.id:
                    item["memory_tier"] = entry.memory_tier
                    self.store.save_session(entry.doc_id, session)
                    return

    def get_working_memory(self, chat_history: list[dict], window_size: int = None) -> list[dict]:
        """获取工作记忆（滑动窗口）

        从对话历史中提取最近 N 轮对话（user + assistant 配对）。

        Args:
            chat_history: 完整对话历史列表，每项包含 role 和 content
            window_size: 窗口大小（保留的轮数），默认从配置读取

        Returns:
            最近 N 轮对话的消息列表
        """
        if window_size is None:
            try:
                from config import settings
                window_size = settings.memory_working_window_size
            except Exception:
                window_size = 10

        if not chat_history:
            return []

        # 提取 user+assistant 配对的轮次
        rounds: list[list[dict]] = []
        i = 0
        while i < len(chat_history):
            msg = chat_history[i]
            if msg.get("role") == "user":
                # 尝试配对下一条 assistant 消息
                round_msgs = [msg]
                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "assistant":
                    round_msgs.append(chat_history[i + 1])
                    i += 2
                else:
                    i += 1
                rounds.append(round_msgs)
            else:
                i += 1

        # 取最后 N 轮
        recent_rounds = rounds[-window_size:] if len(rounds) > window_size else rounds

        # 展平为消息列表
        result = []
        for round_msgs in recent_rounds:
            result.extend(round_msgs)
        return result

    # ==================== 写入去重闸门 ====================

    @staticmethod
    def _normalize_for_dedupe(text: str) -> str:
        return " ".join(str(text or "").split()).strip().lower()

    @classmethod
    def _content_fingerprint(cls, text: str) -> str:
        normalized = cls._normalize_for_dedupe(text)
        if not normalized:
            return ""
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _explicit_promotion_enabled() -> bool:
        try:
            from config import settings
            return bool(settings.memory_explicit_promotion_enabled)
        except Exception:
            return True

    @staticmethod
    def _dedupe_similarity_threshold() -> float:
        try:
            from config import settings
            return float(settings.memory_dedup_similarity_threshold)
        except Exception:
            return 0.85

    def _find_duplicate_memory(
        self,
        content: str,
        *,
        doc_id: str | None,
        candidates: list[MemoryEntry],
    ) -> Optional[MemoryEntry]:
        """在同作用域的活跃记忆中找出与 content 重复的条目。

        先做归一化内容指纹的精确查重，再用向量近邻拦截近义重复。
        只有当前文档的同代际记忆参与比对。用户画像不是文档事实的
        去重基线，不能让自动提炼把用户手工偏好当作可覆盖的旧事实。
        任何一步失败都放行写入——去重是优化，不是正确性前提。
        """
        fingerprint = self._content_fingerprint(content)
        if not fingerprint:
            return None

        scoped: dict[str, MemoryEntry] = {}
        for entry in candidates:
            if getattr(entry, "status", "active") != "active":
                continue
            if entry.doc_id != doc_id:
                continue
            scoped[entry.id] = entry
        if not scoped:
            return None

        for entry in scoped.values():
            if self._content_fingerprint(entry.content) == fingerprint:
                return entry

        threshold = self._dedupe_similarity_threshold()
        if threshold >= 1.0:
            return None
        try:
            # 迭代也放进 try：索引可能返回非列表（旧实现或被替换的后端），
            # 去重失败必须放行写入，不能让它把整条写入链路带崩。
            for hit in self.index.search(content, top_k=5) or []:
                try:
                    similarity = float(hit.get("similarity", 0.0))
                except (TypeError, ValueError, AttributeError):
                    continue
                if similarity < threshold:
                    continue
                entry = scoped.get(hit.get("entry_id", ""))
                if entry is not None:
                    return entry
        except Exception as exc:
            logger.debug(f"[MemoryDedupe] 相似度检索失败，放行写入: {exc}")
        return None

    def _select_non_duplicate(
        self,
        items: list[tuple[Any, str]],
        *,
        doc_id: str | None,
        parse_identity: dict | None = None,
    ) -> tuple[list[Any], int]:
        """从 (标识, 内容) 列表中筛掉重复项，返回 (保留的标识, 丢弃数量)。"""
        if not items:
            return [], 0
        try:
            candidates = [
                entry
                for entry in self.store.get_all_entries()
                if entry.doc_id == doc_id
                and self._entry_matches_parse_identity(
                    entry,
                    doc_id=doc_id,
                    parse_identity=parse_identity,
                )
            ]
        except Exception as exc:
            logger.debug(f"[MemoryDedupe] 读取既有记忆失败，放行本批写入: {exc}")
            return [key for key, _ in items], 0

        kept: list[Any] = []
        dropped = 0
        seen_fingerprints: set[str] = set()
        for key, content in items:
            fingerprint = self._content_fingerprint(content)
            if fingerprint and fingerprint in seen_fingerprints:
                dropped += 1
                continue
            duplicate = self._find_duplicate_memory(
                content, doc_id=doc_id, candidates=candidates
            )
            if duplicate is not None:
                logger.debug(
                    "[MemoryDedupe] 跳过与 %s 重复的候选记忆", duplicate.id
                )
                dropped += 1
                continue
            if fingerprint:
                seen_fingerprints.add(fingerprint)
            kept.append(key)
        return kept, dropped

    # ==================== 写入裁决与失效 ====================

    @staticmethod
    def _arbitration_enabled() -> bool:
        try:
            from config import settings
            return bool(settings.memory_arbitration_enabled)
        except Exception:
            return True

    def _record_audit(self, memory_id: str, event: str, **kwargs) -> None:
        """写审计日志；失败只记 debug，绝不影响记忆写入本身。"""
        if not getattr(self, "audit_log", None):
            return
        try:
            self.audit_log.record(memory_id, event, **kwargs)
        except Exception as exc:
            logger.debug(f"[MemoryAudit] 记录 {event} 失败: {exc}")

    def _arbitrate_facts(
        self,
        facts: list[str],
        *,
        doc_id: str | None,
        api_key: str | None,
        model: str | None,
        api_provider: str | None,
        parse_identity: dict | None = None,
        budget: MemoryLLMBudget | None = None,
    ) -> tuple[list[str], list[tuple[str, str, str]], list[tuple[str, str]]]:
        """对提炼出的事实做写入裁决。

        Returns:
            (要新增的事实, [(target_id, 新内容, 旧内容)], [(target_id, 旧内容)])
            分别对应 ADD、UPDATE、DELETE(失效)。
        """
        if not facts or not self.arbiter or not self._arbitration_enabled():
            return list(facts), [], []
        if not (api_key and model and api_provider):
            return list(facts), [], []

        try:
            # LLM arbitration is allowed to evolve system-derived facts in the
            # current document revision only. It must never update/delete a
            # profile or a manual/liked record merely because wording happens
            # to be similar.
            entry_map = {
                entry.id: entry
                for entry in self.store.get_all_entries()
                if entry.doc_id == doc_id
                and entry.source_type in {"auto_qa", "llm_distilled", "compressed"}
                and self._entry_matches_parse_identity(
                    entry,
                    doc_id=doc_id,
                    parse_identity=parse_identity,
                )
            }
            candidates = self.arbiter.collect_candidates(
                facts, index=self.index, entry_map=entry_map, doc_id=doc_id
            )
            if not candidates:
                return list(facts), [], []
            if not consume_budget(budget, "arbitrate"):
                logger.info("[MemoryLLMBudget] 预算已尽，跳过写入裁决（本轮退回纯追加）")
                return list(facts), [], []
            decisions = self.arbiter.arbitrate(
                facts,
                candidates,
                api_key=api_key,
                model=model,
                provider=api_provider,
            )
        except Exception as exc:
            logger.warning(f"[MemoryArbiter] 裁决过程异常，降级为全部新增: {exc}")
            return list(facts), [], []

        from services.memory_arbiter import (
            ACTION_ADD,
            ACTION_DELETE,
            ACTION_UPDATE,
        )

        to_add: list[str] = []
        to_update: list[tuple[str, str, str]] = []
        to_invalidate: list[tuple[str, str]] = []
        for decision in decisions:
            if decision.action == ACTION_ADD:
                to_add.append(decision.text)
            elif decision.action == ACTION_UPDATE:
                to_update.append((decision.target_id, decision.text, decision.old_content))
            elif decision.action == ACTION_DELETE:
                to_invalidate.append((decision.target_id, decision.old_content))
                # 矛盾的旧记忆失效后，新事实仍要进来，否则这轮信息就丢了。
                to_add.append(decision.text)
            # NONE：语义等价，什么都不做

        if to_update or to_invalidate:
            logger.info(
                "[MemoryArbiter] doc_id=%s 裁决结果: 新增 %d, 更新 %d, 失效 %d",
                doc_id,
                len(to_add),
                len(to_update),
                len(to_invalidate),
            )
        return to_add, to_update, to_invalidate

    @_serialized_memory_mutation
    def invalidate_entry(
        self,
        entry_id: str,
        *,
        reason: str = "superseded",
        actor: str = "system",
    ) -> bool:
        """把一条记忆标记为已失效：不再参与检索，但保留可追溯且可恢复。"""
        entry = self._find_entry(entry_id)
        if entry is None:
            return False
        if entry.invalid_at:
            return True

        now = datetime.now(timezone.utc).isoformat()
        trace = dict(entry.trace or {})
        trace["invalidated_reason"] = reason
        ok = self.store.update_entry_fields(
            entry_id, {"invalid_at": now, "trace": trace}
        )
        if not ok:
            return False
        try:
            self.index.remove_entry(entry_id)
        except Exception as exc:
            logger.debug(f"[Memory] 失效条目移出索引失败 {entry_id}: {exc}")
        self._record_audit(
            entry_id,
            "invalidate",
            old_content=entry.content,
            reason=reason,
            actor=actor,
            doc_id=entry.doc_id,
        )
        return True

    @_serialized_memory_mutation
    def revalidate_entry(self, entry_id: str, *, actor: str = "user") -> bool:
        """撤销失效，把记忆放回检索池。"""
        entry = self._find_entry(entry_id)
        if entry is None:
            return False
        if not entry.invalid_at:
            return True

        trace = dict(entry.trace or {})
        trace.pop("invalidated_reason", None)
        ok = self.store.update_entry_fields(entry_id, {"invalid_at": "", "trace": trace})
        if not ok:
            return False
        try:
            self.index.add_entry(entry_id, entry.content)
        except Exception as exc:
            logger.debug(f"[Memory] 恢复条目重新索引失败 {entry_id}: {exc}")
        self._record_audit(
            entry_id,
            "revalidate",
            new_content=entry.content,
            actor=actor,
            doc_id=entry.doc_id,
        )
        return True

    @_serialized_memory_mutation
    def set_entry_disabled(
        self,
        entry_id: str,
        disabled: bool,
        *,
        actor: str = "user",
    ) -> bool:
        """用户手动停用/启用一条记忆（负向控制，可逆）。"""
        entry = self._find_entry(entry_id)
        if entry is None:
            return False

        if disabled:
            if entry.disabled_at:
                return True
            now = datetime.now(timezone.utc).isoformat()
            if not self.store.update_entry_fields(entry_id, {"disabled_at": now}):
                return False
            try:
                self.index.remove_entry(entry_id)
            except Exception as exc:
                logger.debug(f"[Memory] 停用条目移出索引失败 {entry_id}: {exc}")
            self._record_audit(
                entry_id, "disable", old_content=entry.content,
                actor=actor, doc_id=entry.doc_id,
            )
            return True

        if not entry.disabled_at:
            return True
        if not self.store.update_entry_fields(entry_id, {"disabled_at": ""}):
            return False
        try:
            self.index.add_entry(entry_id, entry.content)
        except Exception as exc:
            logger.debug(f"[Memory] 启用条目重新索引失败 {entry_id}: {exc}")
        self._record_audit(
            entry_id, "enable", new_content=entry.content,
            actor=actor, doc_id=entry.doc_id,
        )
        return True

    # ==================== 记忆 LLM 预算 ====================

    @staticmethod
    def _llm_calls_per_turn() -> int:
        try:
            from config import settings
            return max(0, int(settings.memory_llm_calls_per_turn))
        except Exception:
            return 3

    def _new_llm_budget(self) -> MemoryLLMBudget:
        return MemoryLLMBudget(self._llm_calls_per_turn())

    # ==================== 滚动会话摘要 ====================

    @staticmethod
    def _session_summary_enabled() -> bool:
        try:
            from config import settings
            return bool(settings.memory_session_summary_enabled)
        except Exception:
            return True

    @staticmethod
    def _session_summary_interval() -> int:
        try:
            from config import settings
            return max(2, int(settings.memory_session_summary_interval))
        except Exception:
            return 6

    def _find_session_summary_entry(
        self,
        doc_id: str,
        *,
        parse_identity: dict | None = None,
    ) -> Optional[MemoryEntry]:
        from services.memory_summary import SESSION_SUMMARY_KIND

        for entry in self.store.get_all_entries():
            if (
                entry.doc_id == doc_id
                and entry.memory_kind == SESSION_SUMMARY_KIND
                and self._entry_matches_parse_identity(
                    entry,
                    doc_id=doc_id,
                    parse_identity=parse_identity,
                )
            ):
                return entry
        return None

    def update_session_summary(
        self,
        doc_id: str,
        chat_history: list[dict] | None,
        *,
        api_key: str | None,
        model: str | None,
        api_provider: str | None,
        parse_identity: dict | None = None,
        expected_generation: int | tuple[int, str, int] | None = None,
        budget: MemoryLLMBudget | None = None,
    ) -> Optional[str]:
        """维护该文档的滚动会话摘要，返回新摘要文本；未更新返回 None。

        只在后台写入线程里调用——它会发起一次 LLM 调用。
        """
        if not self._session_summary_enabled() or not doc_id:
            return None
        if not self.is_write_generation_current(expected_generation, doc_id=doc_id):
            return None
        if not (api_key and model and api_provider):
            return None
        if not chat_history:
            return None

        usable = []
        for message in chat_history:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"} or not message.get("content"):
                continue
            if message.get("role") == "assistant" and is_unsafe_automatic_document_answer(
                str(message.get("content") or "")
            ):
                continue
            usable.append(message)
        if len(usable) < 2:
            return None

        from services.memory_summary import (
            SESSION_SUMMARY_KIND,
            SESSION_SUMMARY_SOURCE_TYPE,
            build_rolling_summary,
        )

        existing = self._find_session_summary_entry(
            doc_id,
            parse_identity=parse_identity,
        )
        covered = int((existing.source_ref or {}).get("covered_messages", 0)) if existing else 0

        # 攒够增量才更新，把 LLM 成本摊薄到多轮对话上
        if len(usable) - covered < self._session_summary_interval():
            return None

        if not consume_budget(budget, "session_summary"):
            logger.info("[MemoryLLMBudget] 预算已尽，跳过会话摘要更新")
            return None

        new_messages = usable[covered:] if covered < len(usable) else usable
        summary_text = build_rolling_summary(
            existing.content if existing else "",
            new_messages,
            api_key=api_key,
            model=model,
            provider=api_provider,
        )
        if not summary_text:
            return None
        if is_unsafe_automatic_document_answer(summary_text):
            logger.warning("[MemoryGuard] rejected unsafe session summary")
            return None

        identity = self._normalize_parse_identity(parse_identity) or {}
        source_ref = {**identity, "covered_messages": len(usable)}
        with self._write_generation_lock, self._store_mutation_lock:
            if not self.is_write_generation_current(expected_generation, doc_id=doc_id):
                logger.info("[Memory] clear 后拒绝过期会话摘要: doc_id=%s", doc_id)
                return None
            if existing is not None:
                self.store.update_entry_fields(existing.id, {"source_ref": source_ref})
                self.update_entry(
                    existing.id, summary_text, actor="system", reason="session_summary_roll"
                )
            else:
                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=summary_text,
                    source_type=SESSION_SUMMARY_SOURCE_TYPE,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    doc_id=doc_id,
                    importance=0.8,
                    memory_tier="short_term",
                    memory_kind=SESSION_SUMMARY_KIND,
                    memory_scope="document",
                    title="会话摘要",
                    summary=self._truncate_text(summary_text),
                    source_ref=source_ref,
                    trace={"kind": "session_summary"},
                )
                self.store.add_entry(entry)
                self._record_audit(
                    entry.id, "add", new_content=summary_text,
                    reason="session_summary", actor="system", doc_id=doc_id,
                )
        logger.info("[SessionSummary] doc_id=%s 摘要已更新（覆盖 %d 条消息）", doc_id, len(usable))
        return summary_text

    def _build_session_summary_hits(
        self,
        doc_id: str | None,
        query: str,
        parse_identity: dict | None = None,
    ) -> list[dict[str, Any]]:
        """滚动摘要总是注入（不依赖检索命中），它是会话连续性而非相关性。"""
        if not doc_id or not self._session_summary_enabled():
            return []
        entry = self._find_session_summary_entry(
            doc_id,
            parse_identity=parse_identity,
        )
        if entry is None or not entry.is_retrievable:
            return []
        if is_unsafe_automatic_document_answer(entry.content):
            return []
        if not self._entry_matches_parse_identity(
            entry, doc_id=doc_id, parse_identity=parse_identity
        ):
            return []
        return [self._build_memory_hit(entry, score=0.9, query=query)]

    # ==================== 存储配额 ====================

    @staticmethod
    def _quota_chars_per_doc() -> int:
        try:
            from config import settings
            return max(0, int(settings.memory_quota_chars_per_doc))
        except Exception:
            return 200_000

    def _doc_entries_for_quota(self, doc_id: str) -> list[MemoryEntry]:
        """从单个文档的 session 里取出记忆条目，供配额统计与回收使用。"""
        try:
            session = self.store.load_session(doc_id)
        except Exception as exc:
            logger.debug(f"[MemoryQuota] 读取 session 失败 doc_id={doc_id}: {exc}")
            return []

        entries: list[MemoryEntry] = []
        for item in session.get("qa_summaries", []) or []:
            source_type = item.get("source_type", "auto_qa")
            if source_type in {"llm_distilled", "compressed"} and not item.get("answer"):
                content = item.get("question", "")
            else:
                content = "Q: {}\nA: {}".format(
                    item.get("question", ""), item.get("answer", "")
                )
            entries.append(MemoryEntry.from_dict({**item, "content": content, "doc_id": doc_id}))
        for item in session.get("important_memories", []) or []:
            entries.append(MemoryEntry.from_dict({**item, "doc_id": doc_id}))
        return entries

    def enforce_quota(self, doc_id: str | None) -> int:
        """把单文档记忆的总字符数压回配额内，返回清理条数。

        条数阈值（压缩用）对长短记忆一视同仁——20 条长表格记忆和 20 条一句话
        记忆占用差一个数量级，只有配额能给出可预测的存储上界。

        清理顺序刻意保守：
        1. 先回收已停用的条目（用户已经表示不想要）
        2. 再回收已失效的条目（已被后续对话推翻）
        3. 再回收压缩归档的原始条目（信息已进 consolidated）
        4. 仍超额才动活跃条目，且按"重要度低、最久未命中"排序，
           **永不删除 importance>=1.0 的用户主动保存记忆**
        """
        quota = self._quota_chars_per_doc()
        if quota <= 0 or not doc_id:
            return 0

        # 只读该文档的 session，不做全量 get_all_entries()：
        # 文档级配额本就不需要全量数据，而全量读会触发存储层的快照重放，
        # 那是写入路径上不该顺带发生的副作用。
        entries = self._doc_entries_for_quota(doc_id)
        used = sum(len(e.content or "") for e in entries)
        if used <= quota:
            return 0

        def _sort_key(entry: MemoryEntry) -> tuple:
            return (entry.importance, entry.last_hit_at or "", entry.created_at or "")

        tiers = [
            sorted([e for e in entries if e.disabled_at], key=_sort_key),
            sorted([e for e in entries if e.invalid_at and not e.disabled_at], key=_sort_key),
            sorted(
                [e for e in entries if e.status == "archived_raw" and not e.invalid_at and not e.disabled_at],
                key=_sort_key,
            ),
            sorted(
                [e for e in entries if e.is_retrievable and e.importance < 1.0],
                key=_sort_key,
            ),
        ]

        removed = 0
        for tier in tiers:
            for entry in tier:
                if used <= quota:
                    break
                if self.delete_entry(entry.id):
                    used -= len(entry.content or "")
                    removed += 1
            if used <= quota:
                break

        if removed:
            logger.info(
                "[MemoryQuota] doc_id=%s 超出配额，已回收 %d 条记忆（剩余 %d/%d 字符）",
                doc_id, removed, used, quota,
            )
        elif used > quota:
            logger.warning(
                "[MemoryQuota] doc_id=%s 仍超出配额但无可回收条目（%d/%d 字符），"
                "剩余全部是用户主动保存的记忆", doc_id, used, quota,
            )
        return removed

    def get_quota_status(self, doc_id: str | None) -> dict[str, Any]:
        """返回配额占用情况，供面板展示。"""
        quota = self._quota_chars_per_doc()
        entries = [
            e for e in self.store.get_all_entries()
            if doc_id is None or e.doc_id == doc_id
        ]
        used = sum(len(e.content or "") for e in entries)
        return {
            "doc_id": doc_id,
            "used_chars": used,
            "quota_chars": quota,
            "entry_count": len(entries),
            "over_quota": bool(quota and used > quota),
        }

    @_serialized_memory_mutation
    def restore_archived_entry(self, entry_id: str, *, actor: str = "user") -> bool:
        """把压缩归档的原始记忆恢复为活跃状态。

        压缩是有损的——LLM 可能摘丢了某个数值。用户在归档里找回那条时，
        应当能一键让它重新参与检索，而不是只能只读查看。
        """
        entry = self._find_entry(entry_id)
        if entry is None:
            return False
        if entry.status != "archived_raw":
            return True

        trace = dict(entry.trace or {})
        trace["restored_from_archive"] = True
        trace.pop("archived_reason", None)
        ok = self.store.update_entry_fields(
            entry_id,
            {"status": "active", "memory_tier": "short_term", "trace": trace},
        )
        if not ok:
            return False
        try:
            self.index.add_entry(entry_id, entry.content)
        except Exception as exc:
            logger.debug(f"[Memory] 归档恢复重新索引失败 {entry_id}: {exc}")
        self._page_in_active_pool(entry)
        self._record_audit(
            entry_id,
            "revalidate",
            new_content=entry.content,
            reason="restored_from_archive",
            actor=actor,
            doc_id=entry.doc_id,
        )
        return True

    def get_entry_history(self, entry_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """返回单条记忆的演化链。"""
        if not getattr(self, "audit_log", None):
            return []
        return self.audit_log.history(entry_id, limit=limit)

    def get_recent_audit(
        self,
        limit: int = 50,
        doc_id: str | None = None,
        event: str | None = None,
    ) -> list[dict[str, Any]]:
        """返回最近的记忆变更，用于面板总览。"""
        if not getattr(self, "audit_log", None):
            return []
        return self.audit_log.recent(limit=limit, doc_id=doc_id, event=event)

    # ==================== 记忆写入 ====================

    def save_qa_summary(
        self,
        doc_id: str,
        chat_history: list[dict],
        n: int = 3,
        api_key: str = None,
        model: str = None,
        api_provider: str = None,
        parse_identity: dict | None = None,
        expected_generation: int | None = None,
    ) -> bool:
        """从对话历史中提取最后 N 轮 QA 摘要并保存

        优先使用 LLM 提炼持久性事实（借鉴 OpenClaw），
        LLM 不可用时降级为截断摘要。

        Args:
            doc_id: 文档标识
            chat_history: 对话历史列表，每项包含 role 和 content
            n: 提取最后 N 轮 QA 对，默认 3
            api_key: LLM API 密钥（用于记忆提炼）
            model: LLM 模型名称
            api_provider: LLM 提供商
            parse_identity: 生成该对话答案时的文档解析身份
        """
        if not chat_history or not doc_id:
            return False

        if expected_generation is None:
            expected_generation = self.capture_write_generation(doc_id)

        # 提取 QA 对：从对话历史中配对 user/assistant 消息
        qa_pairs = self._extract_qa_pairs(chat_history)
        if not qa_pairs:
            return False

        # 取最后 N 轮
        recent_pairs = qa_pairs[-n:]
        safe_recent_pairs = [
            (question, answer)
            for question, answer in recent_pairs
            if not is_unsafe_automatic_document_answer(answer)
        ]
        if len(safe_recent_pairs) != len(recent_pairs):
            logger.info(
                "[MemoryGuard] skipped %d unsafe automatic QA summaries",
                len(recent_pairs) - len(safe_recent_pairs),
            )
        if not safe_recent_pairs:
            return True
        recent_pairs = safe_recent_pairs

        identity = self._normalize_parse_identity(parse_identity)
        source_ref = dict(identity or {})

        # 一轮写入的 LLM 调用总预算：提炼/裁决/压缩/会话摘要/图谱按优先级消费，
        # 预算耗尽时低优先级的那些直接跳过（它们各自的触发阈值仍在，下轮会重试）。
        llm_budget = self._new_llm_budget()

        # 尝试 LLM 提炼
        distilled_facts = None
        if api_key and model and api_provider:
            distilled_facts = self._distill_facts(
                recent_pairs, api_key, model, api_provider, budget=llm_budget
            )

        # 裁决产出的改写与失效，在下面的写入事务里一并落地。
        pending_updates: list[tuple[str, str, str]] = []
        pending_invalidations: list[tuple[str, str]] = []

        # 用户明说"记住…"时，本批记忆直接按用户主动保存对待。
        explicit_request = bool(
            self._explicit_promotion_enabled()
            and any(has_explicit_memory_request(question) for question, _ in recent_pairs)
        )

        # 去重闸门放在锁外：向量检索不应压进存储临界区。
        if distilled_facts:
            candidate_facts = [
                text for text in (str(fact or "").strip() for fact in distilled_facts) if text
            ]
            rejected_absence_facts = [
                fact for fact in candidate_facts
                if is_unscoped_document_absence_claim(fact)
            ]
            normalized_facts = [
                fact for fact in candidate_facts
                if not is_unscoped_document_absence_claim(fact)
            ]
            if rejected_absence_facts:
                logger.info(
                    "[MemoryDistill] doc_id=%s 跳过 %d 条无边界结构缺失断言",
                    doc_id,
                    len(rejected_absence_facts),
                )
            distilled_facts, dropped = self._select_non_duplicate(
                [(fact, fact) for fact in normalized_facts],
                doc_id=doc_id,
                parse_identity=identity,
            )
            if dropped:
                logger.info(
                    "[MemoryDedupe] doc_id=%s 跳过 %d 条重复提炼事实", doc_id, dropped
                )
            if not distilled_facts:
                # 全部重复：清空候选让写入块自然写不出东西。
                # 这里**不能直接 return**——会话摘要、配额、图谱这些后台维护
                # 与"这轮有没有新事实"无关，短路掉它们会让长会话里的摘要停更。
                # 也不能落回 auto_qa 分支，否则低质截断摘要会绕过去重再进来一次。
                recent_pairs = []
            # 去重之后才做裁决：能省掉一次 LLM 调用。
            distilled_facts, pending_updates, pending_invalidations = self._arbitrate_facts(
                distilled_facts,
                doc_id=doc_id,
                api_key=api_key,
                model=model,
                api_provider=api_provider,
                parse_identity=identity,
                budget=llm_budget,
            )
            if not distilled_facts and not pending_updates and not pending_invalidations:
                recent_pairs = []
        else:
            recent_pairs, dropped = self._select_non_duplicate(
                [
                    (
                        (question, answer),
                        f"Q: {question[:QUESTION_MAX_LEN]}\nA: {answer[:ANSWER_MAX_LEN]}",
                    )
                    for question, answer in recent_pairs
                ],
                doc_id=doc_id,
                parse_identity=identity,
            )
            if dropped:
                logger.info(
                    "[MemoryDedupe] doc_id=%s 跳过 %d 条重复 QA 摘要", doc_id, dropped
                )

        # Keep snapshot/session/event writes in one short critical section. The
        # expensive LLM and embedding calls stay outside it, but their final
        # publication is fenced below.
        with self._write_generation_lock, self._store_mutation_lock:
            if not self.is_write_generation_current(expected_generation, doc_id=doc_id):
                logger.info("[Memory] clear 后拒绝过期 QA 摘要写入: doc_id=%s", doc_id)
                return False

            # 先落地裁决对既有记忆的改写与失效，再写新条目，
            # 这样"旧事实失效 + 新事实入库"在同一个临界区内完成。
            for target_id, new_content, _old_content in pending_updates:
                try:
                    self.update_entry(
                        target_id,
                        new_content,
                        actor="arbiter",
                        reason="arbitration_merge",
                    )
                except Exception as exc:
                    logger.warning(f"[MemoryArbiter] 应用更新失败 {target_id}: {exc}")
            for target_id, _old_content in pending_invalidations:
                try:
                    self.invalidate_entry(
                        target_id, reason="contradicted_by_new_fact", actor="arbiter"
                    )
                except Exception as exc:
                    logger.warning(f"[MemoryArbiter] 应用失效失败 {target_id}: {exc}")

            session = self.store.load_session(doc_id)
            created_entries: list[MemoryEntry] = []
            existing_auto_qa = {
                (
                    str(item.get("question") or "").strip(),
                    str(item.get("answer") or "").strip(),
                )
                for item in session.get("qa_summaries", [])
                if item.get("source_type", "auto_qa") == "auto_qa"
            }
            existing_distilled = {
                str(item.get("question") or "").strip()
                for item in session.get("qa_summaries", [])
                if item.get("source_type") == "llm_distilled"
            }

            if distilled_facts:
                for fact in distilled_facts:
                    fact = str(fact or "").strip()
                    if not fact or fact in existing_distilled:
                        continue
                    existing_distilled.add(fact)
                    entry_id = str(uuid.uuid4())
                    # 本批含显式记忆指令时无法精确归因到某条事实，整批一起晋升。
                    fact_importance = 1.0 if explicit_request else 0.7
                    fact_trace = {"kind": "distilled_fact", "source": "qa_history"}
                    if explicit_request:
                        fact_trace["promoted_by"] = "explicit_request"
                    summary = {
                        "id": entry_id,
                        "question": fact,
                        "answer": "",
                        "source_type": "llm_distilled",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "importance": fact_importance,
                        "memory_kind": "doc_fact",
                        "memory_scope": "document",
                        "status": "active",
                        "title": "文档事实",
                        "summary": self._truncate_text(fact),
                        "source_ref": dict(source_ref),
                        "trace": dict(fact_trace),
                    }
                    if explicit_request:
                        summary["memory_tier"] = "long_term"
                    session["qa_summaries"].append(summary)
                    created_entries.append(MemoryEntry(
                        id=entry_id,
                        content=fact,
                        source_type="llm_distilled",
                        created_at=summary["created_at"],
                        doc_id=doc_id,
                        importance=fact_importance,
                        memory_tier="long_term" if explicit_request else "short_term",
                        memory_kind="doc_fact",
                        memory_scope="document",
                        title="文档事实",
                        summary=summary["summary"],
                        source_ref=dict(source_ref),
                        trace=dict(summary["trace"]),
                    ))
                logger.info(f"LLM 记忆提炼: {len(distilled_facts)} 条事实")
            else:
                for question, answer in recent_pairs:
                    truncated_q = question[:QUESTION_MAX_LEN]
                    truncated_a = answer[:ANSWER_MAX_LEN]
                    qa_key = (truncated_q.strip(), truncated_a.strip())
                    if not all(qa_key) or qa_key in existing_auto_qa:
                        continue
                    existing_auto_qa.add(qa_key)
                    entry_id = str(uuid.uuid4())
                    # 摘要能精确归因到提问，逐条判断是否为显式记忆指令。
                    pair_explicit = bool(
                        self._explicit_promotion_enabled()
                        and has_explicit_memory_request(question)
                    )
                    pair_importance = 1.0 if pair_explicit else 0.5
                    pair_trace = {"kind": "qa_summary", "source": "chat_history"}
                    if pair_explicit:
                        pair_trace["promoted_by"] = "explicit_request"
                    summary = {
                        "id": entry_id,
                        "question": truncated_q,
                        "answer": truncated_a,
                        "source_type": "auto_qa",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "importance": pair_importance,
                        "memory_kind": "episodic",
                        "memory_scope": "document",
                        "status": "active",
                        "title": truncated_q[:60] or "对话摘要",
                        "summary": self._truncate_text(f"Q: {truncated_q}\nA: {truncated_a}"),
                        "source_ref": dict(source_ref),
                        "trace": dict(pair_trace),
                    }
                    if pair_explicit:
                        summary["memory_tier"] = "long_term"
                    session["qa_summaries"].append(summary)
                    created_entries.append(MemoryEntry(
                        id=entry_id,
                        content=f"Q: {truncated_q}\nA: {truncated_a}",
                        source_type="auto_qa",
                        created_at=summary["created_at"],
                        doc_id=doc_id,
                        importance=pair_importance,
                        memory_tier="long_term" if pair_explicit else "short_term",
                        memory_kind="episodic",
                        memory_scope="document",
                        title=summary["title"],
                        summary=summary["summary"],
                        source_ref=dict(source_ref),
                        trace=dict(summary["trace"]),
                    ))

            self._enforce_summary_limit(session)
            retained_ids = {item.get("id") for item in session.get("qa_summaries", [])}
            created_entries = [entry for entry in created_entries if entry.id in retained_ids]
            session["last_accessed"] = datetime.now(timezone.utc).isoformat()
            self.store.save_session(doc_id, session)
            try:
                for entry in created_entries:
                    self.store._write_memory_markdown(entry, is_long_term=False)
                    self.store._append_event("add", {"entry": entry.to_dict(), "scope": "qa_summary"})
                    self._record_audit(
                        entry.id,
                        "add",
                        new_content=entry.content,
                        reason=entry.source_type,
                        actor="system",
                        doc_id=doc_id,
                    )
            except Exception as exc:
                logger.warning(f"同步写入 Markdown 失败: {exc}")

        for entry in created_entries:
            try:
                self.index.add_entry(
                    entry.id,
                    entry.content,
                    should_commit=lambda: self.is_write_generation_current(
                        expected_generation, doc_id=doc_id
                    ),
                )
            except Exception as exc:
                logger.error(f"添加 QA 摘要到向量索引失败: {exc}")

        # 保存完成后检查是否需要压缩（安全执行）
        if self.is_write_generation_current(expected_generation, doc_id=doc_id):
            self._safe_execute(
                "MemoryCompressor.check",
                self._check_and_compress,
                doc_id,
                api_key,
                model,
                api_provider,
                identity,
                expected_generation,
                llm_budget,
            )
        # QA 摘要直接更新 session，不会经过 MemoryStore.add_entry。
        # 在压缩收敛后才失效缓存，避免压缩阶段读取到半更新会话。
        if self.is_write_generation_current(expected_generation, doc_id=doc_id):
            self.store.cache.invalidate()

        # 滚动会话摘要：补上 working 层之外的中期叙事连续性。
        # 同样跑在后台，按消息增量阈值触发。
        if self.is_write_generation_current(expected_generation, doc_id=doc_id):
            self._safe_execute(
                "SessionSummary.update",
                self.update_session_summary,
                doc_id,
                chat_history,
                api_key=api_key,
                model=model,
                api_provider=api_provider,
                parse_identity=identity,
                expected_generation=expected_generation,
                budget=llm_budget,
            )

        # 配额回收：压缩用的条数阈值管不住"20 条长表格记忆"，
        # 这里给存储一个可预测的上界。
        if self.is_write_generation_current(expected_generation, doc_id=doc_id):
            self._safe_execute("MemoryQuota.enforce", self.enforce_quota, doc_id)

        # 图谱重建放在这条后台写入链路的最后：凭证现成、不占响应路径，
        # 且由增量阈值把 LLM 成本摊薄到多轮对话上。
        if self.is_write_generation_current(expected_generation, doc_id=doc_id):
            self._safe_execute(
                "MemoryGraph.rebuild",
                self.rebuild_graph,
                doc_id,
                api_key=api_key,
                model=model,
                api_provider=api_provider,
                parse_identity=identity,
                expected_generation=expected_generation,
                budget=llm_budget,
            )

        llm_budget.log_summary(doc_id)
        return True

    def _check_and_compress(
        self,
        doc_id: str,
        api_key: str = None,
        model: str = None,
        api_provider: str = None,
        parse_identity: dict | None = None,
        expected_generation: int | None = None,
        budget: MemoryLLMBudget | None = None,
    ) -> None:
        """检查并执行记忆压缩

        当同一文档的记忆条目数量超过压缩阈值时，触发压缩流程。

        Args:
            doc_id: 文档 ID
            api_key: LLM API 密钥
            model: LLM 模型名称
            api_provider: LLM 提供商
        """
        if not self.compressor or not doc_id:
            return
        if not self.is_write_generation_current(expected_generation, doc_id=doc_id):
            return
        all_entries = self.store.get_all_entries()
        doc_entries = [
            e for e in all_entries
            if e.doc_id == doc_id
            and e.status == "active"
            and e.source_type in {"auto_qa", "llm_distilled"}
            and not is_unsafe_automatic_document_answer(e.content)
            and e.memory_kind != "consolidated"
            and self._entry_matches_parse_identity(
                e,
                doc_id=doc_id,
                parse_identity=parse_identity,
            )
        ]
        if not self.compressor.should_compress(doc_id, doc_entries):
            return
        if not consume_budget(budget, "compress"):
            logger.info("[MemoryLLMBudget] 预算已尽，跳过本轮压缩（阈值仍在，下轮会重试）")
            return
        compressed = self.compressor.compress(
            doc_entries, api_key=api_key, model=model, api_provider=api_provider
        )
        if not compressed:
            return
        compressed = [
            entry
            for entry in compressed
            if not is_unsafe_automatic_document_answer(entry.content)
        ]
        if not compressed:
            logger.warning("[MemoryGuard] rejected unsafe compressed memory")
            return

        with self._write_generation_lock, self._store_mutation_lock:
            if not self.is_write_generation_current(expected_generation, doc_id=doc_id):
                logger.info("[Memory] clear 后拒绝过期压缩结果: doc_id=%s", doc_id)
                return
            identity = self._normalize_parse_identity(parse_identity)
            if identity:
                for entry in compressed:
                    entry.source_ref = dict(identity)
            compressed_ids = [c.id for c in compressed]
            for e in doc_entries:
                archived_trace = dict(e.trace or {})
                archived_trace["archived_into"] = compressed_ids
                archived_trace["archived_reason"] = "compressed"
                self.store.update_entry_fields(
                    e.id,
                    {
                        "status": "archived_raw",
                        "memory_tier": "archived",
                        "trace": archived_trace,
                    },
                )
                try:
                    self.index.remove_entry(e.id)
                except Exception as exc:
                    logger.debug(f"[MemoryCompressor] 移除原始记忆索引失败 {e.id}: {exc}")
            self.store.batch_add_entries(compressed)

        for entry in compressed:
            try:
                committed = self.index.add_entry(
                    entry.id,
                    entry.content,
                    should_commit=lambda: self.is_write_generation_current(
                        expected_generation, doc_id=doc_id
                    ),
                )
                if committed:
                    self._page_in_active_pool(entry)
            except Exception as exc:
                logger.debug(f"[MemoryCompressor] 添加压缩记忆索引失败 {entry.id}: {exc}")
        logger.info(f"[MemoryCompressor] 文档 {doc_id} 压缩完成: {len(doc_entries)} -> {len(compressed)}")

    def _distill_facts(
        self,
        qa_pairs: list[tuple[str, str]],
        api_key: str,
        model: str,
        api_provider: str,
        budget: MemoryLLMBudget | None = None,
    ) -> Optional[list[str]]:
        """使用 LLM 从 QA 对中提炼持久性事实

        借鉴 OpenClaw 记忆策略：让 LLM 决定哪些信息值得长期记住。

        Args:
            qa_pairs: [(question, answer), ...]
            api_key: API 密钥
            model: 模型名称
            api_provider: 提供商

        Returns:
            事实列表，失败时返回 None
        """
        try:
            # 构建 QA 文本
            qa_text = ""
            for q, a in qa_pairs:
                qa_text += f"用户问：{q[:500]}\nAI答：{a[:800]}\n\n"

            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是论文阅读助手的记忆提炼模块。从问答记录中提取值得长期记住的事实。\n"
                        "\n"
                        "可以提取两类内容：\n"
                        "1) 用户侧：用户的偏好、研究关注点、明确要求、对错误的纠正。"
                        "这类事实【只能】来自『用户问』，绝不能从『AI答』中推断——"
                        "把 AI 自己生成的内容当作用户偏好会污染用户画像。\n"
                        "2) 文档侧：论文中确定的结论、关键数值、方法与数据集名称。"
                        "这类可以来自『AI答』，但必须是论文本身的内容，"
                        "不能是 AI 的寒暄、免责声明、检索失败说明或对话过程描述。\n"
                        "文档侧的否定性结论只有在明确限定具体缺失字段时才可保存，例如层数、通道、投影或超参；"
                        "不要保存“论文未给出结构”这类无边界判断。\n"
                        "若答案同时给出架构级拓扑又说明实现细节未公开，优先保存肯定的拓扑，"
                        "缺失项必须单独限定到具体字段。\n"
                        "\n"
                        "规则：\n"
                        "- 每条事实一行，前面加 '- '，写成脱离上下文也能读懂的完整陈述\n"
                        "- 最多 5 条，只保留持久有效的信息，丢弃一次性的追问与澄清\n"
                        "- 用与用户提问相同的语言书写，数值、单位、专有名词保持原样\n"
                        "- 没有值得长期记住的内容时，只回复两个字：无\n"
                        "- 不要输出任何解释、编号或前缀\n"
                        "\n"
                        "示例一（无信息量，输出哨兵）：\n"
                        "用户问：你好\n"
                        "AI答：你好，有什么可以帮你的吗？\n"
                        "输出：无\n"
                        "\n"
                        "示例二（检索失败，不得记录）：\n"
                        "用户问：表3的F1是多少\n"
                        "AI答：抱歉，我在文档中没有找到表3的相关内容。\n"
                        "输出：无\n"
                        "\n"
                        "示例三（正常提取）：\n"
                        "用户问：我主要关心方法部分，实验可以略过。表2里他们的方法比基线高多少？\n"
                        "AI答：表2中该方法达到 82.4 Acc，最强基线为 79.1 Acc，高 3.3 个百分点。\n"
                        "输出：\n"
                        "- 用户主要关注方法部分，希望略过实验细节\n"
                        "- 表2中论文方法为 82.4 Acc，最强基线 79.1 Acc，领先 3.3 个百分点"
                    ),
                },
                {"role": "user", "content": qa_text.strip()},
            ]

            if not consume_budget(budget, "distill"):
                logger.info("[MemoryLLMBudget] 预算已尽，跳过事实提炼")
                return None
            response = call_llm_sync(
                messages,
                api_key=api_key,
                model=model,
                provider=api_provider,
                max_tokens=300,
            )
            facts = self._parse_distilled_facts(response)
            if facts is not None:
                return facts

            # 提炼失败就直接降级成截断摘要，记忆质量会明显掉一档。
            # 先带着失败原因重试一次（借鉴 paper-qa 的 prior-attempt 模式），
            # 成本只多一次调用。
            retry_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "上一次回复无法解析成事实列表。请严格按格式重新输出："
                        "每条事实一行、以 '- ' 开头；没有值得记住的内容就只回复两个字：无。"
                    ),
                }
            ]
            if not consume_budget(budget, "distill_retry"):
                return None
            response = call_llm_sync(
                retry_messages,
                api_key=api_key,
                model=model,
                provider=api_provider,
                max_tokens=300,
            )
            return self._parse_distilled_facts(response)

        except Exception as e:
            logger.warning(f"LLM 记忆提炼失败，降级为截断: {e}")
            return None

    @staticmethod
    def _parse_distilled_facts(response: str | None) -> Optional[list[str]]:
        """解析提炼结果。

        返回 None 表示"没解析出东西"（可重试）；
        返回 [] 表示模型明确说没有值得记的内容（不该重试）。
        """
        content = str(response or "").strip()
        if not content:
            return None
        if _is_no_fact_sentinel(content):
            return []
        # require_marker=True：模型返回解释性散文时要判为解析失败并触发重试，
        # 不能把那段散文当成一条"事实"存进记忆库。
        facts = [
            item for item in parse_bullet_list(content, require_marker=True)
            if len(item) > 3 and not _is_no_fact_sentinel(item)
        ]
        return facts or None

    def _extract_qa_pairs(self, chat_history: list[dict]) -> list[tuple[str, str]]:
        """从对话历史中提取 QA 对

        遍历消息列表，将相邻的 user/assistant 消息配对。

        Args:
            chat_history: 对话历史列表

        Returns:
            [(question, answer), ...] 列表
        """
        pairs = []
        i = 0
        while i < len(chat_history) - 1:
            current = chat_history[i]
            next_msg = chat_history[i + 1]

            if (
                current.get("role") == "user"
                and next_msg.get("role") == "assistant"
            ):
                question = str(current.get("content", "") or "").strip()
                answer = str(next_msg.get("content", "") or "").strip()
                if question and not is_unusable_automatic_answer(answer):
                    pairs.append((question, answer))
                i += 2  # 跳过已配对的两条消息
            else:
                i += 1

        return pairs

    def _enforce_summary_limit(self, session: dict) -> None:
        """摘要数量上限控制

        超过上限时移除最早的非重要摘要（importance < 1.0）。

        Args:
            session: 文档会话记忆字典
        """
        summaries = session.get("qa_summaries", [])
        while len(summaries) > self.max_summaries:
            # 查找最早的非重要摘要
            removed = False
            for i, s in enumerate(summaries):
                if s.get("importance", 0.5) < 1.0:
                    removed_summary = summaries.pop(i)
                    removed_id = removed_summary.get("id")
                    if removed_id:
                        try:
                            self.index.remove_entry(removed_id)
                        except Exception as e:
                            logger.debug(f"移除超限摘要索引失败 {removed_id}: {e}")
                        try:
                            self.store._append_event("delete", {"entry_id": removed_id, "scope": "qa_summary_limit"})
                        except Exception:
                            pass
                    removed = True
                    break
            if not removed:
                # 所有摘要都是重要的，无法移除，退出循环
                break
        session["qa_summaries"] = summaries

    @_serialized_memory_mutation
    def save_important_memory(
        self,
        doc_id: str,
        question: str,
        answer: str,
        source_type: str = "manual",
    ) -> MemoryEntry:
        """保存重要记忆（用户手动标记或点赞）

        同时添加到 store 和 index（支持向量检索）。

        Args:
            doc_id: 文档标识
            question: 用户问题
            answer: AI 回答
            source_type: 来源类型，"manual" 或 "liked"

        Returns:
            创建的 MemoryEntry 对象
        """
        content = f"Q: {question}\nA: {answer}"
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            source_type=source_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            doc_id=doc_id,
            importance=1.0,
            memory_tier="long_term",
            memory_kind="doc_fact" if doc_id else "profile",
            memory_scope="document" if doc_id else "profile",
            title=question[:60] or "重要记忆",
            summary=self._truncate_text(content),
            source_ref={"question": question[:200]},
            trace={"kind": "important_memory", "source": source_type},
        )

        # 保存到存储层
        self.store.add_entry(entry)
        
        # 同步写入 Markdown 源文件（每日日志）
        try:
            self.store._write_memory_markdown(entry, is_long_term=False)
        except Exception as e:
            logger.warning(f"同步写入 Markdown 失败: {e}")

        # 添加到向量索引
        try:
            self.index.add_entry(entry.id, content)
        except Exception as e:
            logger.error(f"添加重要记忆到向量索引失败: {e}")
        self._page_in_active_pool(entry)
        self._record_audit(
            entry.id,
            "add",
            new_content=entry.content,
            reason=source_type,
            actor="user",
            doc_id=doc_id,
        )

        return entry

    def update_keywords(self, query: str, *, expected_generation: int | None = None) -> bool:
        """从查询中提取关键词并更新用户画像

        提取关键词 → 更新频率统计 → 更新关注领域列表。

        Args:
            query: 用户查询文本
        """
        if not query or not query.strip():
            return False

        keywords = self.keyword_extractor.extract_keywords(query)
        if not keywords:
            return False

        with self._write_generation_lock, self._store_mutation_lock:
            expected_global_generation = self._global_generation_from_fence(expected_generation)
            if (
                expected_global_generation is not None
                and expected_global_generation != self._write_generation
            ):
                logger.info("[Memory] clear 后拒绝过期关键词写入")
                return False
            profile = self.store.load_profile()
            profile = self.keyword_extractor.update_frequency(profile, keywords)

            # 更新关注领域列表
            profile["focus_areas"] = self.keyword_extractor.get_focus_areas(
                profile, threshold=self.keyword_threshold
            )

            self.store.save_profile(profile)
            self.store.record_profile_state(profile, reason="keyword_update")
        return True

    # ==================== CRUD 操作 ====================

    def get_profile(self) -> dict:
        """获取用户画像数据"""
        profile = self.store.load_profile()
        profile_entries = sorted(
            [e for e in self.store.get_all_entries() if e.memory_scope == "profile"],
            key=self._entry_sort_key,
            reverse=True,
        )
        return {
            **profile,
            "entries": [self._serialize_entry(entry) for entry in profile_entries],
        }

    def validate_doc_id(self, doc_id: str) -> str:
        """Validate external document IDs before they reach persistent storage."""
        return self.store.validate_session_doc_id(doc_id)

    def get_session(self, doc_id: str) -> dict:
        """获取指定文档的会话记忆"""
        doc_id = self.validate_doc_id(doc_id)
        session = self.store.load_session(doc_id)
        entries = sorted(
            [e for e in self.store.get_all_entries() if e.doc_id == doc_id],
            key=self._entry_sort_key,
            reverse=True,
        )
        session["entries"] = [self._serialize_entry(entry) for entry in entries]
        return session

    def list_entries(
        self,
        *,
        doc_id: str | None = None,
        memory_kind: str | None = None,
        memory_scope: str | None = None,
        status: str | None = None,
        lifecycle: str | None = None,
        include_content: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """列出记忆条目，支持按层级/作用域/状态/生命周期筛选与分页。

        lifecycle 取值：
        - ``retrievable``：当前真正参与检索的条目
        - ``invalidated``：被裁决判定已被推翻的条目
        - ``disabled``：用户手动停用的条目
        - ``archived``：非破坏压缩后归档的原始条目
        """
        return self.list_entries_page(
            doc_id=doc_id,
            memory_kind=memory_kind,
            memory_scope=memory_scope,
            status=status,
            lifecycle=lifecycle,
            include_content=include_content,
            limit=limit,
            offset=offset,
        )["items"]

    def list_entries_page(
        self,
        *,
        doc_id: str | None = None,
        memory_kind: str | None = None,
        memory_scope: str | None = None,
        status: str | None = None,
        lifecycle: str | None = None,
        include_content: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """与 list_entries 相同的筛选，但额外返回总数便于前端分页。"""
        if doc_id is not None:
            doc_id = self.validate_doc_id(doc_id)
        entries = self.store.get_all_entries()
        filtered = []
        for entry in entries:
            if doc_id is not None and entry.doc_id != doc_id:
                continue
            if memory_kind and entry.memory_kind != memory_kind:
                continue
            if memory_scope and entry.memory_scope != memory_scope:
                continue
            if status and entry.status != status:
                continue
            if lifecycle and not self._matches_lifecycle(entry, lifecycle):
                continue
            filtered.append(entry)
        filtered.sort(key=self._entry_sort_key, reverse=True)

        total = len(filtered)
        start = max(0, int(offset or 0))
        if limit is None:
            window = filtered[start:]
        else:
            window = filtered[start : start + max(0, int(limit))]
        return {
            "items": [
                self._serialize_entry(entry, include_content=include_content)
                for entry in window
            ],
            "total": total,
            "offset": start,
            "limit": limit,
        }

    @staticmethod
    def _matches_lifecycle(entry: MemoryEntry, lifecycle: str) -> bool:
        normalized = str(lifecycle or "").strip().lower()
        if normalized == "retrievable":
            return entry.is_retrievable
        if normalized == "invalidated":
            return bool(entry.invalid_at)
        if normalized == "disabled":
            return bool(entry.disabled_at)
        if normalized == "archived":
            return entry.status == "archived_raw"
        return True

    def get_entry_trace(self, entry_id: str) -> dict[str, Any]:
        """返回指定记忆的来源链与派生关系。"""
        entry_map = self._get_entry_map()
        entry = entry_map.get(entry_id)
        if not entry:
            raise KeyError(entry_id)

        parent_entries = [
            self._serialize_entry(entry_map[parent_id])
            for parent_id in entry.derived_from
            if parent_id in entry_map
        ]
        child_entries = [
            self._serialize_entry(candidate)
            for candidate in entry_map.values()
            if entry.id in (candidate.derived_from or [])
        ]

        return {
            "entry": self._serialize_entry(entry),
            "parents": parent_entries,
            "children": sorted(child_entries, key=lambda item: (item.get("created_at", ""), item.get("id", "")), reverse=True),
            "trace": dict(entry.trace or {}),
            "source_ref": dict(entry.source_ref or {}),
        }

    def get_graph_summary(
        self,
        doc_id: str | None = None,
        parse_identity: dict | None = None,
    ) -> dict[str, Any]:
        """基于文档事实/压缩记忆生成轻量图谱摘要。"""
        if doc_id is not None:
            doc_id = self.validate_doc_id(doc_id)
        entries = [
            entry for entry in self.store.get_all_entries()
            if entry.doc_id == doc_id and entry.memory_kind in {"doc_fact", "consolidated", "graph"}
        ] if doc_id else [
            entry for entry in self.store.get_all_entries()
            if entry.memory_kind in {"doc_fact", "consolidated", "graph"}
        ]
        if doc_id and self._normalize_parse_identity(parse_identity):
            entries = [
                entry for entry in entries
                if self._entry_matches_parse_identity(
                    entry,
                    doc_id=doc_id,
                    parse_identity=parse_identity,
                )
            ]

        entries = [
            entry
            for entry in entries
            if not self._is_unscoped_architecture_absence_automatic_memory(
                entry.source_type,
                entry.content,
            )
        ]
        # 优先用缓存里的 LLM 图谱（由后台写入线程构建），拿不到就走正则降级。
        # 这个方法在检索热路径上被调用，**绝不能**在这里发起 LLM 调用。
        from services.memory_graph import build_regex_graph, facts_signature

        cached = self._graph_cache.get(doc_id or "__global__")
        if cached and cached.get("signature") == facts_signature(
            [entry.content for entry in entries]
        ):
            summary = dict(cached["summary"])
            summary["doc_id"] = doc_id
            summary["source"] = "llm"
            return summary

        summary = build_regex_graph(entries).to_summary(doc_id)
        summary["source"] = "regex"
        return summary

    def rebuild_graph(
        self,
        doc_id: str | None,
        *,
        api_key: str,
        model: str,
        api_provider: str,
        parse_identity: dict | None = None,
        force: bool = False,
        expected_generation: int | tuple[int, str, int] | None = None,
        budget: MemoryLLMBudget | None = None,
    ) -> dict[str, Any] | None:
        """用 LLM 重建文档图谱并写入缓存。

        只应从后台线程或用户显式触发的接口调用——它会发起一次 LLM 调用。
        返回 None 表示未重建（未启用/无凭证/事实太少/签名未变/增量不够/抽取失败）。
        """
        if not self._graph_llm_enabled() or not (api_key and model and api_provider):
            return None
        if not self.is_write_generation_current(expected_generation, doc_id=doc_id):
            return None

        from services.memory_graph import build_llm_graph, facts_signature

        entries = self._graph_source_entries(doc_id, parse_identity)
        facts = [entry.content for entry in entries if (entry.content or "").strip()]
        if len(facts) < 2:
            return None

        cache_key = doc_id or "__global__"
        signature = facts_signature(facts)
        cached = self._graph_cache.get(cache_key)
        if cached and not force:
            if cached.get("signature") == signature:
                return None
            # 事实增量不够就先不重建，把成本摊薄到多轮对话上
            if abs(len(facts) - cached.get("fact_count", 0)) < self._graph_rebuild_delta():
                return None

        if not consume_budget(budget, "graph"):
            logger.info("[MemoryLLMBudget] 预算已尽，跳过图谱重建")
            return None

        graph = build_llm_graph(
            facts, api_key=api_key, model=model, provider=api_provider
        )
        if graph is None:
            return None

        summary = graph.to_summary(doc_id)
        with self._write_generation_lock, self._store_mutation_lock:
            if not self.is_write_generation_current(expected_generation, doc_id=doc_id):
                logger.info("[Memory] clear 后拒绝过期图谱: doc_id=%s", doc_id)
                return None
            self._graph_cache[cache_key] = {
                "signature": signature,
                "fact_count": len(facts),
                "summary": summary,
            }
        logger.info(
            "[MemoryGraph] doc_id=%s 图谱重建完成: %d 实体 / %d 关系",
            doc_id,
            summary["node_count"],
            summary["edge_count"],
        )
        return summary

    def _graph_source_entries(
        self, doc_id: str | None, parse_identity: dict | None
    ) -> list[MemoryEntry]:
        entries = [
            entry for entry in self.store.get_all_entries()
            if entry.memory_kind in {"doc_fact", "consolidated", "graph"}
            and entry.is_retrievable
            and not self._is_unscoped_architecture_absence_automatic_memory(
                entry.source_type,
                entry.content,
            )
            and (doc_id is None or entry.doc_id == doc_id)
        ]
        if doc_id and self._normalize_parse_identity(parse_identity):
            entries = [
                entry for entry in entries
                if self._entry_matches_parse_identity(
                    entry, doc_id=doc_id, parse_identity=parse_identity
                )
            ]
        return entries

    @staticmethod
    def _graph_llm_enabled() -> bool:
        try:
            from config import settings
            return bool(settings.memory_graph_llm_enabled)
        except Exception:
            return True

    @staticmethod
    def _graph_rebuild_delta() -> int:
        try:
            from config import settings
            return max(1, int(settings.memory_graph_rebuild_delta))
        except Exception:
            return 5

    @_serialized_memory_mutation
    def add_entry(
        self, content: str, source_type: str, doc_id: str = None
    ) -> MemoryEntry:
        """添加记忆条目

        同时添加到 store 和 index，并自动打标签。

        Args:
            content: 记忆内容文本
            source_type: 来源类型
            doc_id: 关联的文档 ID（可选）

        Returns:
            创建的 MemoryEntry 对象
        """
        if doc_id is not None:
            doc_id = self.validate_doc_id(doc_id)
        importance = 1.0 if source_type in ("manual", "liked") else 0.5
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            source_type=source_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            doc_id=doc_id,
            importance=importance,
            memory_kind="profile" if doc_id is None else "episodic",
            memory_scope="profile" if doc_id is None else "document",
            title=(content.splitlines()[0][:60] if content else "记忆条目"),
            summary=self._truncate_text(content),
            trace={"kind": "manual_add"},
        )

        # 自动打标签（安全执行，失败不影响写入）
        if self.tagger:
            tags = self._safe_execute("MemoryTagger.auto_tag", self.tagger.auto_tag, content)
            if tags:
                entry.tags = tags

        # 保存到 store
        self.store.add_entry(entry)

        # 添加到向量索引
        try:
            self.index.add_entry(entry.id, content)
        except Exception as e:
            logger.error(f"添加记忆条目到向量索引失败: {e}")
        self._page_in_active_pool(entry)
        self._record_audit(
            entry.id,
            "add",
            new_content=entry.content,
            reason=source_type,
            actor="user",
            doc_id=doc_id,
        )

        return entry

    @_serialized_memory_mutation
    def delete_entry(self, entry_id: str) -> bool:
        """删除指定记忆条目

        同时从 store 和 index 中移除。

        Args:
            entry_id: 记忆条目 ID

        Returns:
            是否删除成功
        """
        existing = self._find_entry(entry_id)
        success = self.store.delete_entry(entry_id)
        if success:
            try:
                self.index.remove_entry(entry_id)
            except Exception as e:
                logger.error(f"从向量索引移除记忆条目失败: {e}")
            self._record_audit(
                entry_id,
                "delete",
                old_content=existing.content if existing else "",
                actor="user",
                doc_id=existing.doc_id if existing else None,
            )
        return success

    @_serialized_memory_mutation
    def update_entry(
        self,
        entry_id: str,
        content: str,
        *,
        actor: str = "user",
        reason: str = "",
    ) -> bool:
        """更新指定记忆条目的内容

        同时更新 store 和 index，并记一条审计。

        Args:
            entry_id: 记忆条目 ID
            content: 新的内容文本
            actor: 变更发起方，用于审计区分人工编辑与裁决改写
            reason: 变更原因，写入审计

        Returns:
            是否更新成功
        """
        existing = self._find_entry(entry_id)
        success = self.store.update_entry(entry_id, content)
        if success:
            try:
                # 先移除旧的向量，再添加新的
                self.index.remove_entry(entry_id)
                self.index.add_entry(entry_id, content)
            except Exception as e:
                logger.error(f"更新向量索引失败: {e}")
            self._record_audit(
                entry_id,
                "update",
                old_content=existing.content if existing else "",
                new_content=content,
                reason=reason,
                actor=actor,
                doc_id=existing.doc_id if existing else None,
            )
        return success

    @_serialized_memory_mutation
    def clear_document(self, doc_id: str) -> int:
        """Permanently erase every document-scoped memory artifact.

        Delayed LLM workers receive a document fence at request start. Advance
        that fence before touching storage so they cannot recreate entries,
        graph cache, vectors, audit rows or Markdown after the clear returns.
        """
        doc_id = self.validate_doc_id(doc_id)
        with self._write_generation_lock:
            self._document_write_generations[doc_id] = (
                self._document_write_generations.get(doc_id, 0) + 1
            )

        entry_ids = self.store.clear_document(doc_id)
        self._graph_cache.pop(doc_id, None)
        if self.active_pool:
            for entry_id in entry_ids:
                try:
                    self.active_pool.remove_entry(entry_id)
                except Exception as exc:
                    logger.debug("清理活跃记忆池失败 %s: %s", entry_id, exc)

        for entry_id in entry_ids:
            try:
                self.index.remove_entry(entry_id)
            except Exception as exc:
                logger.debug("清理文档向量索引失败 %s: %s", entry_id, exc)
        try:
            self.index.flush_sync(reason="document_clear")
        except Exception as exc:
            logger.warning("同步文档记忆清理索引失败: %s", exc)

        if getattr(self, "audit_log", None):
            self.audit_log.clear_document(doc_id)
        return len(entry_ids)

    @_serialized_memory_mutation
    def clear_all(self) -> None:
        """清空所有记忆数据

        同时清空 store 和 index。
        """
        with self._write_generation_lock:
            self._write_generation += 1
        with self._store_mutation_lock:
            self.store.clear_all()
            if self.active_pool:
                try:
                    self.active_pool.clear()
                except Exception as exc:
                    logger.warning("清空活跃记忆池失败: %s", exc)
        try:
            # Do not hold the service generation lock while acquiring the
            # index lock: an embedding worker checks the generation from
            # inside that index lock. The generation was already advanced
            # above, so late workers cannot publish during this gap.
            self.index.rebuild([])
            self.index.flush_sync(reason="manual")
        except Exception as e:
            logger.error(f"清空向量索引失败: {e}")
        if getattr(self, "audit_log", None):
            try:
                self.audit_log.clear()
            except Exception as e:
                logger.warning(f"清空审计日志失败: {e}")

    def rebuild_from_events(self) -> dict[str, Any]:
        """从事件日志重建 JSON 快照和向量索引。"""
        state = self.store.rebuild_snapshots_from_events()
        entries = [
            entry for entry in self.store.get_all_entries()
            if entry.status != "archived_raw"
            and entry.is_retrievable
            and not self._is_unscoped_architecture_absence_automatic_memory(
                entry.source_type,
                entry.content,
            )
        ]
        self.index.safe_reindex(entries, reason="events_restore")
        self.index.flush_sync(reason="manual")
        storage_status = self.store.get_storage_status()
        return {
            "profile_entries": len(state.get("profile", {}).get("entries", [])),
            "session_count": len(state.get("sessions", {})),
            "indexed_entries": len(entries),
            **storage_status,
        }

    @_serialized_memory_mutation
    def evaluate_and_update_importance(self) -> None:
        """自动评估并更新记忆重要性
        
        基于以下因素综合评分：
        - 命中次数：频繁使用的记忆提升重要性
        - 时间衰减：长期未使用的记忆降低重要性
        - 用户标记：手动标记的记忆保持高重要性
        
        自动升降级规则：
        - 命中次数 >= 5 且最近 7 天内使用过：提升到 0.8
        - 命中次数 >= 10 且最近 3 天内使用过：提升到 1.0
        - 超过 90 天未使用且命中次数 < 3：降低到 0.3
        - 超过 180 天未使用：降低到 0.1
        """
        try:
            from datetime import timedelta
            
            all_entries = self.store.get_all_entries()
            now = datetime.now(timezone.utc)
            updated_count = 0
            
            for entry in all_entries:
                original_importance = entry.importance
                new_importance = original_importance
                
                # 用户标记的记忆（importance >= 1.0）保持不变
                if entry.importance >= 1.0:
                    continue
                
                # 计算最后使用时间
                last_hit_time = None
                if entry.last_hit_at:
                    try:
                        last_hit_time = datetime.fromisoformat(entry.last_hit_at)
                        if last_hit_time.tzinfo is None:
                            last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        pass
                
                if not last_hit_time:
                    # 没有命中记录，使用创建时间
                    try:
                        last_hit_time = datetime.fromisoformat(entry.created_at)
                        if last_hit_time.tzinfo is None:
                            last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue
                
                days_since_use = (now - last_hit_time).total_seconds() / 86400.0
                hit_count = entry.hit_count
                
                # 自动提升规则
                if hit_count >= 10 and days_since_use <= 3:
                    new_importance = 1.0  # 最高重要性
                elif hit_count >= 5 and days_since_use <= 7:
                    new_importance = max(original_importance, 0.8)  # 提升到 0.8
                elif hit_count >= 3 and days_since_use <= 14:
                    new_importance = max(original_importance, 0.6)  # 提升到 0.6
                
                # 自动降级规则
                elif days_since_use > 180:
                    new_importance = 0.1  # 最低重要性
                elif days_since_use > 90 and hit_count < 3:
                    new_importance = min(original_importance, 0.3)  # 降低到 0.3
                elif days_since_use > 60 and hit_count < 2:
                    new_importance = min(original_importance, 0.4)  # 降低到 0.4
                
                # 如果重要性发生变化，更新条目
                if abs(new_importance - original_importance) > 0.05:  # 变化超过 5% 才更新
                    entry.importance = new_importance
                    # 更新存储
                    if entry.doc_id:
                        session = self.store.load_session(entry.doc_id)
                        # 在 important_memories 中查找并更新
                        for item in session.get("important_memories", []):
                            if item.get("id") == entry.id:
                                item["importance"] = new_importance
                                self.store.save_session(entry.doc_id, session)
                                updated_count += 1
                                break
                    else:
                        # 在 profile 中查找并更新
                        profile = self.store.load_profile()
                        for item in profile.get("entries", []):
                            if item.get("id") == entry.id:
                                item["importance"] = new_importance
                                self.store.save_profile(profile)
                                updated_count += 1
                                break
            
            if updated_count > 0:
                logger.info(f"自动评估记忆重要性: 更新了 {updated_count} 条记忆的重要性")
            
            # 评估完成后触发降级检查
            try:
                self.check_and_demote()
            except Exception as e:
                logger.debug(f"降级检查失败（不影响评估）: {e}")
        except Exception as e:
            logger.error(f"自动评估记忆重要性失败: {e}")
    
    def get_status(self) -> dict:
        """获取记忆系统状态

        Returns:
            包含 enabled、total_entries、index_size、profile_focus_areas 的字典
        """
        try:
            all_entries = self.store.get_all_entries()
            total_entries = len(all_entries)
        except Exception:
            total_entries = 0

        index_size = (
            self.index.index.ntotal
            if self.index.index is not None
            else 0
        )

        profile = self.store.load_profile()
        focus_areas = profile.get("focus_areas", [])

        return {
            "enabled": True,
            "total_entries": total_entries,
            "index_size": index_size,
            "profile_focus_areas": focus_areas,
            "llm_calls_per_turn": self._llm_calls_per_turn(),
            **self.store.get_storage_status(),
            **self.index.get_status(),
        }
