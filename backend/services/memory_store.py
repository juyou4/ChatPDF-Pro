"""
记忆持久化存储层

负责记忆数据的 JSON 文件读写，提供原子化的 CRUD 操作。
存储结构：
  data/memory/
  ├── user_profile.json          # 用户画像（长期记忆）
  ├── sessions/
  │   └── {doc_id}_session.json  # 文档会话记忆
  └── memory_index/              # FAISS 向量索引（由 MemoryIndex 管理）
"""
import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from services.memory_cache import MemoryCache
from services.memory_quality import sanitize_automatic_memory_content

logger = logging.getLogger(__name__)

_SESSION_DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source_type: str = "manual"  # "auto_qa" | "manual" | "liked" | "keyword" | "llm_distilled" | "compressed"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    doc_id: Optional[str] = None
    importance: float = 0.5  # 0.0-1.0，manual/liked 默认 1.0，auto 默认 0.5
    hit_count: int = 0  # 被检索命中的次数
    last_hit_at: str = ""  # 最后一次被命中的时间
    # 新增字段：记忆层级和分类标签
    memory_tier: str = "short_term"  # "working" | "short_term" | "long_term" | "archived"
    tags: list[str] = field(default_factory=list)  # 分类标签，如 "concept" | "fact" | "preference" 等
    memory_kind: str = ""
    memory_scope: str = ""
    status: str = "active"  # "active" | "archived_raw"
    title: str = ""
    summary: str = ""
    source_ref: dict[str, Any] = field(default_factory=dict)
    derived_from: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    last_used_query: str = ""
    # 双时态：valid_at 是该事实开始成立的时间（留空表示自 created_at 起），
    # invalid_at 非空表示已被后续对话推翻/取代，不再参与检索但保留可追溯。
    valid_at: str = ""
    invalid_at: str = ""
    # 用户手动停用（负向控制）。与 invalid_at 分开，便于区分"系统判定过期"
    # 和"用户不想要"，两者都可逆。
    disabled_at: str = ""

    @property
    def is_retrievable(self) -> bool:
        """该条记忆是否应参与检索与注入。"""
        return (
            self.status == "active"
            and not self.invalid_at
            and not self.disabled_at
        )

    def __post_init__(self) -> None:
        if not self.memory_kind:
            self.memory_kind = self._infer_memory_kind(self.source_type, self.doc_id)
        if not self.memory_scope:
            self.memory_scope = self._infer_memory_scope(self.doc_id)
        if not self.summary:
            normalized = " ".join((self.content or "").split())
            self.summary = normalized[:180] + ("..." if len(normalized) > 180 else "")

    @staticmethod
    def _infer_memory_kind(source_type: str, doc_id: Optional[str]) -> str:
        if source_type == "compressed":
            return "consolidated"
        if source_type == "llm_distilled":
            return "doc_fact" if doc_id else "profile"
        if source_type in {"manual", "liked"}:
            return "doc_fact" if doc_id else "profile"
        if source_type == "keyword":
            return "profile"
        return "episodic"

    @staticmethod
    def _infer_memory_scope(doc_id: Optional[str]) -> str:
        return "document" if doc_id else "profile"

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "source_type": self.source_type,
            "created_at": self.created_at,
            "doc_id": self.doc_id,
            "importance": self.importance,
            "hit_count": self.hit_count,
            "last_hit_at": self.last_hit_at,
            "memory_tier": self.memory_tier,
            "tags": self.tags,
            "memory_kind": self.memory_kind,
            "memory_scope": self.memory_scope,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "source_ref": self.source_ref,
            "derived_from": self.derived_from,
            "trace": self.trace,
            "last_used_query": self.last_used_query,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "disabled_at": self.disabled_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """从字典反序列化，缺失的新字段使用默认值以保证向后兼容"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            source_type=data.get("source_type", "manual"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            doc_id=data.get("doc_id"),
            importance=data.get("importance", 0.5),
            hit_count=data.get("hit_count", 0),
            last_hit_at=data.get("last_hit_at", ""),
            memory_tier=data.get("memory_tier", "short_term"),
            tags=data.get("tags", []),
            memory_kind=data.get("memory_kind", cls._infer_memory_kind(data.get("source_type", "manual"), data.get("doc_id"))),
            memory_scope=data.get("memory_scope", cls._infer_memory_scope(data.get("doc_id"))),
            status=data.get("status", "active"),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            source_ref=data.get("source_ref", {}),
            derived_from=data.get("derived_from", []),
            trace=data.get("trace", {}),
            last_used_query=data.get("last_used_query", ""),
            valid_at=data.get("valid_at", ""),
            invalid_at=data.get("invalid_at", ""),
            disabled_at=data.get("disabled_at", ""),
        )


class MemoryStore:
    """记忆持久化存储"""

    def __init__(self, data_dir: str):
        """
        初始化记忆存储

        Args:
            data_dir: 记忆数据根目录，如 "data/memory/"
        """
        self.data_dir = data_dir
        self.profile_path = os.path.join(data_dir, "user_profile.json")
        self.sessions_dir = os.path.join(data_dir, "sessions")
        self.memory_dir = os.path.join(data_dir, "memory")  # Markdown 源文件目录
        self.events_dir = os.path.join(data_dir, "events")
        self.snapshots_dir = os.path.join(data_dir, "snapshots")
        self.snapshot_sessions_dir = os.path.join(self.snapshots_dir, "sessions")
        self.legacy_migration_state_path = os.path.join(self.snapshots_dir, "legacy_migration_state.json")
        # 初始化内存缓存
        self.cache = MemoryCache()
        # 确保目录结构存在
        self._ensure_dirs()
        self._migrate_legacy_json_to_events()

    def _ensure_dirs(self) -> None:
        """确保所有必需的目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)  # Markdown 源文件目录
        os.makedirs(self.events_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.snapshot_sessions_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "memory_index"), exist_ok=True)

    @staticmethod
    def validate_session_doc_id(doc_id: str) -> str:
        """Validate document IDs before using them in session file paths."""
        if not isinstance(doc_id, str) or not _SESSION_DOC_ID_PATTERN.fullmatch(doc_id):
            raise ValueError("文档 ID 只能包含字母、数字、下划线和连字符")
        return doc_id

    def _safe_session_file_path(self, directory: str, doc_id: str, suffix: str) -> str:
        safe_doc_id = self.validate_session_doc_id(doc_id)
        root = Path(directory).resolve()
        candidate = (root / f"{safe_doc_id}{suffix}").resolve()
        if candidate.parent != root:
            raise ValueError("文档 ID 生成了越界路径")
        return str(candidate)

    def _snapshot_profile_path(self) -> str:
        """获取用户画像快照文件路径。"""
        return os.path.join(self.snapshots_dir, "user_profile.snapshot.json")

    def _snapshot_session_path(self, doc_id: str) -> str:
        """获取文档会话快照文件路径。"""
        return self._safe_session_file_path(
            self.snapshot_sessions_dir,
            doc_id,
            "_session.snapshot.json",
        )

    def _has_event_records(self) -> bool:
        """是否存在可回放的事件日志。"""
        if not os.path.exists(self.events_dir):
            return False
        for filename in os.listdir(self.events_dir):
            if not filename.endswith(".jsonl"):
                continue
            filepath = os.path.join(self.events_dir, filename)
            try:
                if os.path.getsize(filepath) > 0:
                    return True
            except OSError:
                continue
        return False

    def _sync_profile_snapshot(self, profile: dict) -> None:
        """同步写入用户画像快照。"""
        self._write_json(self._snapshot_profile_path(), profile)

    def _sync_session_snapshot(self, doc_id: str, session: dict) -> None:
        """同步写入文档会话快照。"""
        self._write_json(self._snapshot_session_path(doc_id), session)

    def _remove_snapshot_session(self, doc_id: str) -> None:
        """删除指定文档的会话快照。"""
        path = self._snapshot_session_path(doc_id)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                logger.warning(f"删除 session 快照失败 {path}: {exc}")

    def _clear_session_files(self, directory: str, suffix: str) -> None:
        """批量删除指定目录下的 session 文件。"""
        if not os.path.exists(directory):
            return
        for filename in os.listdir(directory):
            if not filename.endswith(suffix):
                continue
            filepath = os.path.join(directory, filename)
            try:
                os.remove(filepath)
            except OSError as exc:
                logger.warning(f"删除 session 文件失败 {filepath}: {exc}")

    def _list_session_doc_ids(self) -> list[str]:
        """列出当前可见的文档会话 ID，优先使用快照。"""
        doc_ids: set[str] = set()

        if os.path.exists(self.snapshot_sessions_dir):
            for filename in os.listdir(self.snapshot_sessions_dir):
                if filename.endswith("_session.snapshot.json"):
                    doc_id = filename[: -len("_session.snapshot.json")]
                    try:
                        doc_ids.add(self.validate_session_doc_id(doc_id))
                    except ValueError:
                        logger.warning("忽略非法的 session 快照文件名: %s", filename)

        if not doc_ids and os.path.exists(self.sessions_dir):
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith("_session.json"):
                    doc_id = filename[: -len("_session.json")]
                    try:
                        doc_ids.add(self.validate_session_doc_id(doc_id))
                    except ValueError:
                        logger.warning("忽略非法的 session 文件名: %s", filename)

        if not doc_ids and self._has_event_records():
            state = self.replay_events()
            doc_ids.update(state.get("sessions", {}).keys())

        return sorted(doc_ids)

    def _get_event_log_stats(self) -> dict[str, Any]:
        """返回事件日志文件统计信息。"""
        if not os.path.exists(self.events_dir):
            return {"event_log_files": 0, "last_event_at": ""}

        event_files = [
            os.path.join(self.events_dir, filename)
            for filename in os.listdir(self.events_dir)
            if filename.endswith(".jsonl")
        ]
        if not event_files:
            return {"event_log_files": 0, "last_event_at": ""}

        latest_path = max(event_files, key=lambda path: os.path.getmtime(path))
        last_event_at = ""
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    last_event_at = record.get("timestamp", last_event_at)
        except OSError as exc:
            logger.warning(f"读取事件日志统计失败 {latest_path}: {exc}")

        return {
            "event_log_files": len(event_files),
            "last_event_at": last_event_at,
        }

    @staticmethod
    def _default_profile() -> dict:
        """返回默认的用户画像结构"""
        return {
            "focus_areas": [],
            "keyword_frequencies": {},
            "entries": [],
            "updated_at": "",
        }

    @staticmethod
    def _default_session(doc_id: str) -> dict:
        """返回默认的文档会话记忆结构"""
        return {
            "doc_id": doc_id,
            "qa_summaries": [],
            "important_memories": [],
            "last_accessed": "",
        }

    @staticmethod
    def _default_legacy_migration_state() -> dict[str, Any]:
        """返回旧 JSON -> 事件日志迁移状态。"""
        return {
            "profile_seeded": False,
            "session_doc_ids": [],
            "completed_at": "",
        }

    def _read_json(self, path: str) -> Optional[dict]:
        """安全读取 JSON 文件，失败时返回 None"""
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.warning(f"读取 JSON 文件失败 {path}: {e}")
        return None

    def _write_json(self, path: str, data: dict) -> None:
        """安全写入 JSON 文件，自动创建父目录"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_legacy_migration_state(self) -> dict[str, Any]:
        """加载旧 JSON 迁移状态。"""
        state = self._read_json(self.legacy_migration_state_path)
        if not isinstance(state, dict):
            return self._default_legacy_migration_state()
        return {
            "profile_seeded": bool(state.get("profile_seeded", False)),
            "session_doc_ids": list(state.get("session_doc_ids", [])),
            "completed_at": state.get("completed_at", ""),
        }

    def _save_legacy_migration_state(self, state: dict[str, Any]) -> None:
        """保存旧 JSON 迁移状态。"""
        normalized = {
            "profile_seeded": bool(state.get("profile_seeded", False)),
            "session_doc_ids": sorted(set(state.get("session_doc_ids", []))),
            "completed_at": state.get("completed_at", datetime.now(timezone.utc).isoformat()),
        }
        self._write_json(self.legacy_migration_state_path, normalized)

    def _event_log_path(self) -> str:
        """获取当天事件日志文件路径。"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.events_dir, f"{today}.jsonl")

    def _append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """追加写入记忆事件日志，作为新存储层的 append-only 基础。"""
        record = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        try:
            os.makedirs(self.events_dir, exist_ok=True)
            with open(self._event_log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"写入记忆事件日志失败 {event_type}: {e}")

    def _iter_event_records(self) -> list[dict[str, Any]]:
        """按时间顺序读取事件日志。"""
        if not os.path.exists(self.events_dir):
            return []

        records: list[dict[str, Any]] = []
        for filename in sorted(os.listdir(self.events_dir)):
            if not filename.endswith(".jsonl"):
                continue
            path = os.path.join(self.events_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            logger.warning(f"解析事件日志失败 {path}: {exc}")
                            continue
                        if isinstance(record, dict):
                            records.append(record)
            except OSError as exc:
                logger.warning(f"读取事件日志失败 {path}: {exc}")
        return records

    def _discover_legacy_seed_state(self) -> dict[str, Any]:
        """从事件日志中推断已完成的 legacy seed 迁移范围。"""
        profile_seeded = False
        session_doc_ids: set[str] = set()
        for record in self._iter_event_records():
            event_type = record.get("event_type")
            if event_type == "legacy_profile_seed":
                profile_seeded = True
            elif event_type == "legacy_session_seed" and record.get("doc_id"):
                session_doc_ids.add(record["doc_id"])
        return {
            "profile_seeded": profile_seeded,
            "session_doc_ids": sorted(session_doc_ids),
        }

    @staticmethod
    def _content_to_qa_fields(content: str, source_type: str) -> tuple[str, str]:
        if source_type == "llm_distilled":
            return content or "", ""
        if content.startswith("Q: ") and "\nA: " in content:
            question, answer = content.split("\nA: ", 1)
            return question[3:].strip(), answer.strip()
        return content or "", ""

    def _summary_to_entry(self, summary: dict[str, Any], doc_id: str) -> dict[str, Any]:
        """将 legacy session 中的 qa_summary 还原为统一 entry 结构。"""
        source_type = summary.get("source_type", "auto_qa")
        if source_type == "llm_distilled" and not summary.get("answer"):
            content = summary.get("question", "")
        else:
            content = f"Q: {summary.get('question', '')}\nA: {summary.get('answer', '')}"
        return {
            "id": summary.get("id", str(uuid.uuid4())),
            "content": content,
            "source_type": source_type,
            "created_at": summary.get("created_at", ""),
            "doc_id": doc_id,
            "importance": summary.get("importance", 0.5),
            "hit_count": summary.get("hit_count", 0),
            "last_hit_at": summary.get("last_hit_at", ""),
            "memory_tier": summary.get("memory_tier", "short_term"),
            "tags": summary.get("tags", []),
            "memory_kind": summary.get("memory_kind", ""),
            "memory_scope": summary.get("memory_scope", ""),
            "status": summary.get("status", "active"),
            "title": summary.get("title", ""),
            "summary": summary.get("summary", ""),
            "source_ref": summary.get("source_ref", {}),
            "derived_from": summary.get("derived_from", []),
            "trace": summary.get("trace", {}),
            "last_used_query": summary.get("last_used_query", ""),
        }

    def _entry_to_summary(self, entry: dict[str, Any]) -> dict[str, Any]:
        question, answer = self._content_to_qa_fields(entry.get("content", ""), entry.get("source_type", "auto_qa"))
        return {
            "id": entry.get("id", str(uuid.uuid4())),
            "question": question,
            "answer": answer,
            "source_type": entry.get("source_type", "auto_qa"),
            "created_at": entry.get("created_at", ""),
            "importance": entry.get("importance", 0.5),
            "hit_count": entry.get("hit_count", 0),
            "last_hit_at": entry.get("last_hit_at", ""),
            "memory_tier": entry.get("memory_tier", "short_term"),
            "tags": entry.get("tags", []),
            "memory_kind": entry.get("memory_kind", ""),
            "memory_scope": entry.get("memory_scope", ""),
            "status": entry.get("status", "active"),
            "title": entry.get("title", ""),
            "summary": entry.get("summary", ""),
            "source_ref": entry.get("source_ref", {}),
            "derived_from": entry.get("derived_from", []),
            "trace": entry.get("trace", {}),
            "last_used_query": entry.get("last_used_query", ""),
        }

    def _migrate_legacy_json_to_events(self) -> None:
        """将旧 JSON 快照补写为 legacy seed 事件，保证事件日志可完整回放。"""
        try:
            state = self._load_legacy_migration_state()
            discovered = self._discover_legacy_seed_state()
            profile_seeded = state.get("profile_seeded", False) or discovered.get("profile_seeded", False)
            session_doc_ids = set(state.get("session_doc_ids", [])) | set(discovered.get("session_doc_ids", []))
            changed = False

            legacy_profile = self._read_json(self.profile_path)
            if legacy_profile is not None and not profile_seeded:
                self._append_event("legacy_profile_seed", {"profile": legacy_profile})
                profile_seeded = True
                changed = True

            if os.path.exists(self.sessions_dir):
                for filename in sorted(os.listdir(self.sessions_dir)):
                    if not filename.endswith("_session.json"):
                        continue
                    doc_id = filename[: -len("_session.json")]
                    if doc_id in session_doc_ids:
                        continue
                    session = self._read_json(self._session_path(doc_id))
                    if session is None:
                        continue
                    self._append_event(
                        "legacy_session_seed",
                        {"doc_id": doc_id, "session": session},
                    )
                    session_doc_ids.add(doc_id)
                    changed = True

            if changed or set(state.get("session_doc_ids", [])) != session_doc_ids or state.get("profile_seeded", False) != profile_seeded:
                self._save_legacy_migration_state({
                    "profile_seeded": profile_seeded,
                    "session_doc_ids": sorted(session_doc_ids),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:
            logger.warning(f"旧 JSON 迁移到事件日志失败: {exc}")

    def replay_events(self) -> dict[str, Any]:
        """从 append-only 事件日志回放出 profile/session 快照。"""
        records = self._iter_event_records()
        profile = self._default_profile()
        session_meta: dict[str, dict[str, Any]] = {}
        entries: dict[str, dict[str, Any]] = {}
        placements: dict[str, dict[str, Any]] = {}
        entry_order: list[str] = []

        def ensure_session(doc_id: str) -> dict[str, Any]:
            if doc_id not in session_meta:
                session_meta[doc_id] = {
                    "doc_id": doc_id,
                    "last_accessed": "",
                }
            return session_meta[doc_id]

        def remove_entry(entry_id: str) -> None:
            entries.pop(entry_id, None)
            placements.pop(entry_id, None)
            if entry_id in entry_order:
                entry_order.remove(entry_id)

        def reset_profile_scope(seed_profile: dict[str, Any], timestamp: str) -> None:
            profile["focus_areas"] = list(seed_profile.get("focus_areas", []))
            profile["keyword_frequencies"] = dict(seed_profile.get("keyword_frequencies", {}))
            profile["updated_at"] = seed_profile.get("updated_at", timestamp)
            for entry_id in list(entry_order):
                if placements.get(entry_id, {}).get("scope") == "profile":
                    remove_entry(entry_id)
            for entry in seed_profile.get("entries", []):
                entry_id = entry.get("id")
                if not entry_id:
                    continue
                entries[entry_id] = dict(entry)
                placements[entry_id] = {"scope": "profile", "doc_id": None}
                if entry_id not in entry_order:
                    entry_order.append(entry_id)

        def reset_session_scope(doc_id: str, seed_session: dict[str, Any], timestamp: str) -> None:
            for entry_id in list(entry_order):
                if placements.get(entry_id, {}).get("doc_id") == doc_id:
                    remove_entry(entry_id)

            session_meta[doc_id] = {
                "doc_id": doc_id,
                "last_accessed": seed_session.get("last_accessed", timestamp),
            }

            for item in seed_session.get("qa_summaries", []):
                entry = self._summary_to_entry(item, doc_id)
                entry_id = entry.get("id")
                if not entry_id:
                    continue
                entries[entry_id] = entry
                placements[entry_id] = {"scope": "qa_summary", "doc_id": doc_id}
                if entry_id not in entry_order:
                    entry_order.append(entry_id)

            for item in seed_session.get("important_memories", []):
                entry = dict(item)
                entry["doc_id"] = doc_id
                entry_id = entry.get("id")
                if not entry_id:
                    continue
                entries[entry_id] = entry
                placements[entry_id] = {"scope": "important_memory", "doc_id": doc_id}
                if entry_id not in entry_order:
                    entry_order.append(entry_id)

        seed_records = [
            record for record in records
            if record.get("event_type") in {"legacy_profile_seed", "legacy_session_seed"}
        ]
        incremental_records = [
            record for record in records
            if record.get("event_type") not in {"legacy_profile_seed", "legacy_session_seed"}
        ]

        for record in [*seed_records, *incremental_records]:
            event_type = record.get("event_type")
            timestamp = record.get("timestamp", "")

            if event_type == "clear_all":
                profile = self._default_profile()
                session_meta = {}
                entries = {}
                placements = {}
                entry_order = []
                continue

            if event_type == "legacy_profile_seed":
                reset_profile_scope(dict(record.get("profile") or {}), timestamp)
                continue

            if event_type == "legacy_session_seed":
                doc_id = record.get("doc_id")
                if doc_id:
                    reset_session_scope(doc_id, dict(record.get("session") or {}), timestamp)
                continue

            if event_type == "profile_state":
                profile["focus_areas"] = record.get("focus_areas", [])
                profile["keyword_frequencies"] = record.get("keyword_frequencies", {})
                profile["updated_at"] = record.get("updated_at", timestamp)
                continue

            if event_type == "add":
                entry = dict(record.get("entry") or {})
                entry_id = entry.get("id")
                if not entry_id:
                    continue
                doc_id = entry.get("doc_id")
                scope = record.get("scope")
                if scope not in {"profile", "qa_summary", "important_memory"}:
                    scope = "profile" if doc_id is None else "important_memory"
                entries[entry_id] = entry
                placements[entry_id] = {"scope": scope, "doc_id": doc_id}
                if entry_id not in entry_order:
                    entry_order.append(entry_id)
                if doc_id:
                    ensure_session(doc_id)["last_accessed"] = timestamp
                else:
                    profile["updated_at"] = timestamp
                continue

            if event_type == "delete":
                entry_id = record.get("entry_id")
                if not entry_id:
                    continue
                remove_entry(entry_id)
                doc_id = record.get("doc_id")
                if doc_id:
                    ensure_session(doc_id)["last_accessed"] = timestamp
                else:
                    profile["updated_at"] = timestamp
                continue

            if event_type in {"update", "patch"}:
                entry_id = record.get("entry_id")
                updates = dict(record.get("updates") or {})
                if not entry_id or entry_id not in entries:
                    continue
                target = entries[entry_id]
                if "question" in updates or "answer" in updates:
                    question = updates.pop("question", "")
                    answer = updates.pop("answer", "")
                    target["content"] = f"Q: {question}\nA: {answer}".strip()
                target.update(updates)
                doc_id = placements.get(entry_id, {}).get("doc_id") or record.get("doc_id")
                if doc_id:
                    ensure_session(doc_id)["last_accessed"] = timestamp
                else:
                    profile["updated_at"] = timestamp

        profile_entries = [
            entries[entry_id]
            for entry_id in entry_order
            if placements.get(entry_id, {}).get("scope") == "profile"
        ]
        profile["entries"] = profile_entries

        sessions: dict[str, dict[str, Any]] = {}
        for doc_id, meta in session_meta.items():
            qa_summaries = [
                self._entry_to_summary(entries[entry_id])
                for entry_id in entry_order
                if placements.get(entry_id, {}).get("doc_id") == doc_id
                and placements.get(entry_id, {}).get("scope") == "qa_summary"
            ]
            important_memories = [
                entries[entry_id]
                for entry_id in entry_order
                if placements.get(entry_id, {}).get("doc_id") == doc_id
                and placements.get(entry_id, {}).get("scope") == "important_memory"
            ]
            sessions[doc_id] = {
                "doc_id": doc_id,
                "qa_summaries": qa_summaries,
                "important_memories": important_memories,
                "last_accessed": meta.get("last_accessed", ""),
            }

        return {"profile": profile, "sessions": sessions}

    def rebuild_snapshots_from_events(self, *, mirror_legacy: bool = True) -> dict[str, Any]:
        """从事件日志重建事件快照，可选镜像回旧 JSON。"""
        state = self.replay_events()
        profile = state.get("profile", self._default_profile())
        sessions = state.get("sessions", {})

        self._sync_profile_snapshot(profile)
        self._clear_session_files(self.snapshot_sessions_dir, "_session.snapshot.json")
        for doc_id, session in sessions.items():
            self._sync_session_snapshot(doc_id, session)

        if mirror_legacy:
            self._write_json(self.profile_path, profile)
            self._clear_session_files(self.sessions_dir, "_session.json")
            for doc_id, session in sessions.items():
                self._write_json(self._session_path(doc_id), session)

        self.cache.invalidate()
        return state

    def record_profile_state(self, profile: dict, reason: str = "profile_update") -> None:
        """记录用户画像状态，供事件回放恢复 focus areas/keywords。"""
        self._append_event(
            "profile_state",
            {
                "reason": reason,
                "focus_areas": list(profile.get("focus_areas", [])),
                "keyword_frequencies": dict(profile.get("keyword_frequencies", {})),
                "updated_at": profile.get("updated_at", datetime.now(timezone.utc).isoformat()),
            },
        )

    # ==================== Profile 操作 ====================

    def load_profile(self) -> dict:
        """加载用户画像，优先读取事件快照，失败时再回放事件或回退旧 JSON。"""
        snapshot = self._read_json(self._snapshot_profile_path())
        if snapshot is not None:
            return snapshot

        if self._has_event_records():
            state = self.rebuild_snapshots_from_events(mirror_legacy=False)
            return state.get("profile", self._default_profile())

        data = self._read_json(self.profile_path)
        if data is not None:
            self._sync_profile_snapshot(data)
            return data
        return self._default_profile()

    def save_profile(self, profile: dict) -> None:
        """保存用户画像"""
        self._sync_profile_snapshot(profile)
        self._write_json(self.profile_path, profile)

    # ==================== Session 操作 ====================

    def _session_path(self, doc_id: str) -> str:
        """获取文档会话记忆文件路径"""
        return self._safe_session_file_path(self.sessions_dir, doc_id, "_session.json")

    def load_session(self, doc_id: str) -> dict:
        """加载文档会话记忆，优先读取事件快照，失败时再回放事件或回退旧 JSON。"""
        snapshot = self._read_json(self._snapshot_session_path(doc_id))
        if snapshot is not None:
            return snapshot

        if self._has_event_records():
            state = self.rebuild_snapshots_from_events(mirror_legacy=False)
            session = state.get("sessions", {}).get(doc_id)
            if session is not None:
                return session

        data = self._read_json(self._session_path(doc_id))
        if data is not None:
            self._sync_session_snapshot(doc_id, data)
            return data
        return self._default_session(doc_id)

    def save_session(self, doc_id: str, session: dict) -> None:
        """保存文档会话记忆"""
        self._sync_session_snapshot(doc_id, session)
        self._write_json(self._session_path(doc_id), session)

    # ==================== 条目 CRUD ====================

    def get_all_entries(self) -> list:
        """获取所有记忆条目（从 profile + 所有 session 快照中汇总），优先使用缓存。"""
        # 先检查缓存
        cached = self.cache.get_all_entries()
        if cached is not None:
            return cached

        entries: list[MemoryEntry] = []

        # 从 profile 中收集
        profile = self.load_profile()
        for entry_data in profile.get("entries", []):
            entries.append(MemoryEntry.from_dict(entry_data))

        # 从所有 session 中收集
        for doc_id in self._list_session_doc_ids():
            data = self.load_session(doc_id)
            # 从 qa_summaries 中收集（转换为 MemoryEntry）
            for item in data.get("qa_summaries", []):
                source_type = item.get("source_type", "auto_qa")
                if source_type in {"llm_distilled", "compressed"} and not item.get("answer"):
                    content = item.get("question", "")
                else:
                    content = f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
                if source_type in {"auto_qa", "llm_distilled", "compressed"}:
                    content = sanitize_automatic_memory_content(content, source_type)
                    if not content:
                        # 保留原始快照以便审计，但不让故障响应进入检索或提示词。
                        continue
                entry = MemoryEntry(
                    id=item.get("id", str(uuid.uuid4())),
                    content=content,
                    source_type=source_type,
                    created_at=item.get("created_at", ""),
                    doc_id=data.get("doc_id"),
                    importance=item.get("importance", 0.5),
                    hit_count=item.get("hit_count", 0),
                    last_hit_at=item.get("last_hit_at", ""),
                    memory_tier=item.get("memory_tier", "short_term"),
                    tags=item.get("tags", []),
                    memory_kind=item.get("memory_kind", ""),
                    memory_scope=item.get("memory_scope", ""),
                    status=item.get("status", "active"),
                    title=item.get("title", ""),
                    summary=item.get("summary", ""),
                    source_ref=item.get("source_ref", {}),
                    derived_from=item.get("derived_from", []),
                    trace=item.get("trace", {}),
                    last_used_query=item.get("last_used_query", ""),
                    valid_at=item.get("valid_at", ""),
                    invalid_at=item.get("invalid_at", ""),
                    disabled_at=item.get("disabled_at", ""),
                )
                entries.append(entry)
            # 从 important_memories 中收集
            for item in data.get("important_memories", []):
                entry = MemoryEntry.from_dict({
                    **item,
                    "doc_id": data.get("doc_id"),
                })
                if entry.source_type in {"auto_qa", "llm_distilled", "compressed"}:
                    entry.content = sanitize_automatic_memory_content(entry.content, entry.source_type)
                    if not entry.content:
                        continue
                entries.append(entry)

        # 将结果写入缓存
        self.cache.set_all_entries(entries)
        return entries

    def add_entry(self, entry: MemoryEntry) -> None:
        """
        添加记忆条目到对应存储位置
        - 无 doc_id 的条目存入 profile
        - 有 doc_id 的条目存入对应 session 的 important_memories
        """
        if entry.doc_id is None:
            # 存入 profile
            profile = self.load_profile()
            profile["entries"].append(entry.to_dict())
            profile["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.save_profile(profile)
        else:
            # 存入 session 的 important_memories
            session = self.load_session(entry.doc_id)
            session["important_memories"].append(entry.to_dict())
            session["last_accessed"] = datetime.now(timezone.utc).isoformat()
            self.save_session(entry.doc_id, session)
        self._append_event("add", {"entry": entry.to_dict()})
        # 写入后使缓存失效
        self.cache.invalidate()

    def delete_entry(self, entry_id: str) -> bool:
        """删除指定记忆条目，返回是否成功"""
        # 先在 profile 中查找
        profile = self.load_profile()
        original_len = len(profile.get("entries", []))
        profile["entries"] = [
            e for e in profile.get("entries", []) if e.get("id") != entry_id
        ]
        if len(profile["entries"]) < original_len:
            profile["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.save_profile(profile)
            self._append_event("delete", {"entry_id": entry_id, "scope": "profile"})
            # 删除后使缓存失效
            self.cache.invalidate()
            return True

        # 在所有 session 中查找
        for doc_id in self._list_session_doc_ids():
            data = self.load_session(doc_id)

            # 在 qa_summaries 中查找
            orig_qa = len(data.get("qa_summaries", []))
            data["qa_summaries"] = [
                s for s in data.get("qa_summaries", []) if s.get("id") != entry_id
            ]
            if len(data["qa_summaries"]) < orig_qa:
                data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                self.save_session(doc_id, data)
                self._append_event("delete", {"entry_id": entry_id, "scope": "qa_summary", "doc_id": doc_id})
                # 删除后使缓存失效
                self.cache.invalidate()
                return True

            # 在 important_memories 中查找
            orig_im = len(data.get("important_memories", []))
            data["important_memories"] = [
                m for m in data.get("important_memories", []) if m.get("id") != entry_id
            ]
            if len(data["important_memories"]) < orig_im:
                data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                self.save_session(doc_id, data)
                self._append_event("delete", {"entry_id": entry_id, "scope": "important_memory", "doc_id": doc_id})
                # 删除后使缓存失效
                self.cache.invalidate()
                return True

        return False

    def batch_add_entries(self, entries: list) -> None:
        """批量写入记忆条目，按 doc_id 分组减少文件 I/O

        Args:
            entries: MemoryEntry 对象列表
        """
        if not entries:
            return

        # 按 doc_id 分组
        profile_entries = []
        session_groups: dict[str, list] = {}
        for entry in entries:
            if entry.doc_id is None:
                profile_entries.append(entry)
            else:
                if entry.doc_id not in session_groups:
                    session_groups[entry.doc_id] = []
                session_groups[entry.doc_id].append(entry)

        # 批量写入 profile
        if profile_entries:
            profile = self.load_profile()
            for entry in profile_entries:
                profile["entries"].append(entry.to_dict())
            profile["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.save_profile(profile)
            for entry in profile_entries:
                self._append_event("add", {"entry": entry.to_dict()})

        # 批量写入各 session
        for doc_id, doc_entries in session_groups.items():
            session = self.load_session(doc_id)
            for entry in doc_entries:
                session["important_memories"].append(entry.to_dict())
            session["last_accessed"] = datetime.now(timezone.utc).isoformat()
            self.save_session(doc_id, session)
            for entry in doc_entries:
                self._append_event("add", {"entry": entry.to_dict()})

        # 写入后使缓存失效
        self.cache.invalidate()

    def update_entry(self, entry_id: str, content: str) -> bool:
        """更新指定记忆条目的内容，返回是否成功"""
        # 先在 profile 中查找
        profile = self.load_profile()
        for entry in profile.get("entries", []):
            if entry.get("id") == entry_id:
                entry["content"] = content
                profile["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.save_profile(profile)
                self._append_event("update", {"entry_id": entry_id, "updates": {"content": content}})
                # 更新后使缓存失效
                self.cache.invalidate()
                return True

        # 在所有 session 中查找
        for doc_id in self._list_session_doc_ids():
            data = self.load_session(doc_id)

            # 在 qa_summaries 中查找
            for item in data.get("qa_summaries", []):
                if item.get("id") == entry_id:
                    # qa_summaries 的 content 是 question + answer 的组合
                    # 更新时直接替换整个内容
                    item["question"] = content
                    item["answer"] = ""
                    data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                    self.save_session(doc_id, data)
                    self._append_event("update", {"entry_id": entry_id, "doc_id": doc_id, "updates": {"question": content, "answer": ""}})
                    # 更新后使缓存失效
                    self.cache.invalidate()
                    return True

            # 在 important_memories 中查找
            for item in data.get("important_memories", []):
                if item.get("id") == entry_id:
                    item["content"] = content
                    data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                    self.save_session(doc_id, data)
                    self._append_event("update", {"entry_id": entry_id, "doc_id": doc_id, "updates": {"content": content}})
                    # 更新后使缓存失效
                    self.cache.invalidate()
                    return True

        return False

    def update_entry_fields(self, entry_id: str, updates: dict[str, Any]) -> bool:
        """局部更新记忆条目字段，保持旧存储结构兼容。"""
        if not updates:
            return False

        profile = self.load_profile()
        for entry in profile.get("entries", []):
            if entry.get("id") == entry_id:
                entry.update(updates)
                profile["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.save_profile(profile)
                self._append_event("patch", {"entry_id": entry_id, "scope": "profile", "updates": updates})
                self.cache.invalidate()
                return True

        for doc_id in self._list_session_doc_ids():
            data = self.load_session(doc_id)

            for item in data.get("qa_summaries", []):
                if item.get("id") == entry_id:
                    item.update(updates)
                    data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                    self.save_session(doc_id, data)
                    self._append_event("patch", {"entry_id": entry_id, "scope": "qa_summary", "doc_id": doc_id, "updates": updates})
                    self.cache.invalidate()
                    return True

            for item in data.get("important_memories", []):
                if item.get("id") == entry_id:
                    item.update(updates)
                    data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                    self.save_session(doc_id, data)
                    self._append_event("patch", {"entry_id": entry_id, "scope": "important_memory", "doc_id": doc_id, "updates": updates})
                    self.cache.invalidate()
                    return True

        return False

    # ==================== Markdown 源文件支持 ====================
    
    def _get_memory_file_path(self) -> str:
        """获取长期记忆 Markdown 文件路径"""
        return os.path.join(self.memory_dir, "MEMORY.md")
    
    def _get_daily_memory_path(self) -> str:
        """获取今日记忆日志文件路径"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.memory_dir, f"{today}.md")
    
    def _append_to_markdown(self, filepath: str, content: str) -> None:
        """追加内容到 Markdown 文件（append-only）"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(content + "\n\n")
        except Exception as e:
            logger.warning(f"写入 Markdown 文件失败 {filepath}: {e}")
    
    def _write_memory_markdown(self, entry: MemoryEntry, is_long_term: bool = False) -> None:
        """将记忆条目写入 Markdown 文件
        
        Args:
            entry: 记忆条目
            is_long_term: 是否为长期记忆（写入 MEMORY.md），否则写入每日日志
        """
        if is_long_term:
            filepath = self._get_memory_file_path()
            prefix = "##"
        else:
            filepath = self._get_daily_memory_path()
            prefix = "###"
        
        timestamp = entry.created_at or datetime.now(timezone.utc).isoformat()
        source_label = {
            "auto_qa": "自动摘要",
            "manual": "手动记忆",
            "liked": "点赞记忆",
            "keyword": "关键词",
            "llm_distilled": "LLM 提炼",
            "compressed": "压缩记忆",
        }.get(entry.source_type, entry.source_type)
        
        content = f"""{prefix} [{source_label}] {timestamp}

{entry.content}

---
"""
        self._append_to_markdown(filepath, content)
    
    def clear_all(self) -> None:
        """彻底删除所有记忆及其可回放、Markdown 和索引副本。"""
        # 重置 profile
        self.save_profile(self._default_profile())
        # 清空后使缓存失效
        self.cache.invalidate()

        # 删除所有 session 文件
        self._clear_session_files(self.sessions_dir, "_session.json")
        self._clear_session_files(self.snapshot_sessions_dir, "_session.snapshot.json")

        # 删除索引文件
        index_dir = os.path.join(self.data_dir, "memory_index")
        if os.path.exists(index_dir):
            for filename in os.listdir(index_dir):
                filepath = os.path.join(index_dir, filename)
                try:
                    os.remove(filepath)
                except OSError as e:
                    logger.warning(f"删除索引文件失败 {filepath}: {e}")
        
        # Events and daily Markdown were previously retained, so a "clear"
        # action left complete memory text on disk. These directories are
        # owned by MemoryStore; remove their contents rather than appending a
        # new event that preserves the old audit trail indefinitely.
        for directory in (self.events_dir, self.memory_dir):
            try:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
            except OSError as exc:
                logger.warning("删除记忆派生目录失败 %s: %s", directory, exc)
        try:
            if os.path.exists(self.legacy_migration_state_path):
                os.remove(self.legacy_migration_state_path)
        except OSError as exc:
            logger.warning("删除记忆迁移状态失败: %s", exc)
        self._ensure_dirs()

    def get_storage_status(self) -> dict[str, Any]:
        """返回存储层状态，用于前端展示当前快照/事件回放能力。"""
        event_stats = self._get_event_log_stats()
        profile_snapshot_exists = os.path.exists(self._snapshot_profile_path())
        session_snapshot_count = len([
            filename
            for filename in os.listdir(self.snapshot_sessions_dir)
            if filename.endswith("_session.snapshot.json")
        ]) if os.path.exists(self.snapshot_sessions_dir) else 0
        return {
            "snapshot_primary": True,
            "profile_snapshot_exists": profile_snapshot_exists,
            "session_snapshot_count": session_snapshot_count,
            **event_stats,
        }
