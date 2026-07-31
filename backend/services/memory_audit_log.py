"""记忆变更审计日志

与 MemoryStore 的事件日志分工不同：
- 事件日志面向**整体状态恢复**（replay 出 profile/session 快照）
- 审计日志面向**单条记忆的可解释性**：这条记忆为什么变成现在这样

每次 ADD/UPDATE/INVALIDATE/DISABLE/DELETE/ARCHIVE 记一行，保留变更前后内容。
存储用独立的 SQLite 文件，与主存储后端（JSON 或 SQLite）解耦——
即使记忆本身被删除，它的演化历史仍然查得到。

设计原则：审计写入失败**绝不能**影响记忆写入本身。所有方法都吞异常。
"""
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 审计事件类型
EVENT_ADD = "add"
EVENT_UPDATE = "update"
EVENT_INVALIDATE = "invalidate"
EVENT_REVALIDATE = "revalidate"
EVENT_DISABLE = "disable"
EVENT_ENABLE = "enable"
EVENT_DELETE = "delete"
EVENT_ARCHIVE = "archive"
EVENT_PROMOTE = "promote"

_KNOWN_EVENTS = {
    EVENT_ADD, EVENT_UPDATE, EVENT_INVALIDATE, EVENT_REVALIDATE,
    EVENT_DISABLE, EVENT_ENABLE, EVENT_DELETE, EVENT_ARCHIVE, EVENT_PROMOTE,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     TEXT NOT NULL,
    doc_id        TEXT,
    event         TEXT NOT NULL,
    old_content   TEXT,
    new_content   TEXT,
    reason        TEXT,
    actor         TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_history_memory_id
    ON memory_history (memory_id, id);
CREATE INDEX IF NOT EXISTS idx_memory_history_created_at
    ON memory_history (created_at);
"""


class MemoryAuditLog:
    """单条记忆演化历史的追加写日志。"""

    def __init__(self, data_dir: str, filename: str = "memory_history.sqlite"):
        self.db_path = os.path.join(data_dir, filename)
        self._lock = threading.Lock()
        self._available = False
        try:
            os.makedirs(data_dir, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
            self._available = True
        except Exception as exc:
            logger.warning(f"[MemoryAudit] 初始化失败，审计日志将被跳过: {exc}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def available(self) -> bool:
        return self._available

    def record(
        self,
        memory_id: str,
        event: str,
        *,
        old_content: str = "",
        new_content: str = "",
        reason: str = "",
        actor: str = "system",
        doc_id: Optional[str] = None,
    ) -> bool:
        """记一次变更。任何失败都只记日志，不向上抛。"""
        if not self._available or not memory_id:
            return False
        normalized_event = str(event or "").strip().lower()
        if normalized_event not in _KNOWN_EVENTS:
            logger.debug(f"[MemoryAudit] 未知事件类型 {event!r}，仍按原样记录")
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO memory_history "
                    "(memory_id, doc_id, event, old_content, new_content, reason, actor, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(memory_id),
                        doc_id,
                        normalized_event or str(event),
                        str(old_content or ""),
                        str(new_content or ""),
                        str(reason or ""),
                        str(actor or "system"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            return True
        except Exception as exc:
            logger.debug(f"[MemoryAudit] 写入失败 memory_id={memory_id}: {exc}")
            return False

    def record_many(self, records: list[dict[str, Any]]) -> int:
        """批量记录，返回成功条数。"""
        written = 0
        for item in records or []:
            memory_id = item.get("memory_id", "")
            event = item.get("event", "")
            if not memory_id or not event:
                continue
            if self.record(
                memory_id,
                event,
                old_content=item.get("old_content", ""),
                new_content=item.get("new_content", ""),
                reason=item.get("reason", ""),
                actor=item.get("actor", "system"),
                doc_id=item.get("doc_id"),
            ):
                written += 1
        return written

    def history(self, memory_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """按时间正序返回单条记忆的演化链。"""
        if not self._available or not memory_id:
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_history WHERE memory_id = ? "
                    "ORDER BY id ASC LIMIT ?",
                    (str(memory_id), max(1, int(limit))),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.debug(f"[MemoryAudit] 查询失败 memory_id={memory_id}: {exc}")
            return []

    def recent(
        self,
        limit: int = 50,
        doc_id: Optional[str] = None,
        event: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """按时间倒序返回最近的变更，可按文档与事件类型过滤。"""
        if not self._available:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if doc_id is not None:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if event:
            clauses.append("event = ?")
            params.append(str(event).strip().lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM memory_history {where} ORDER BY id DESC LIMIT ?",
                    params,
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.debug(f"[MemoryAudit] 查询最近变更失败: {exc}")
            return []

    def stats(self) -> dict[str, Any]:
        """审计日志概况，用于状态面板。"""
        if not self._available:
            return {"available": False, "total": 0, "by_event": {}}
        try:
            with self._connect() as conn:
                total = conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0]
                rows = conn.execute(
                    "SELECT event, COUNT(*) AS n FROM memory_history GROUP BY event"
                ).fetchall()
            return {
                "available": True,
                "total": int(total),
                "by_event": {row["event"]: int(row["n"]) for row in rows},
            }
        except Exception as exc:
            logger.debug(f"[MemoryAudit] 统计失败: {exc}")
            return {"available": False, "total": 0, "by_event": {}}

    def clear(self) -> None:
        """清空审计日志（仅在整体清空记忆时调用）。"""
        if not self._available:
            return
        try:
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM memory_history")
        except Exception as exc:
            logger.warning(f"[MemoryAudit] 清空失败: {exc}")

    def clear_document(self, doc_id: str) -> None:
        """Permanently remove one document's audit payloads on document clear."""
        if not self._available or not doc_id:
            return
        try:
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM memory_history WHERE doc_id = ?", (str(doc_id),))
        except Exception as exc:
            logger.warning(f"[MemoryAudit] 清理文档审计失败 doc_id={doc_id}: {exc}")
