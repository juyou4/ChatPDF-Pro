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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source_type: str = "manual"  # "auto_qa" | "manual" | "liked" | "keyword" | "llm_distilled"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    doc_id: Optional[str] = None
    importance: float = 0.5  # 0.0-1.0，manual/liked 默认 1.0，auto 默认 0.5
    hit_count: int = 0  # 被检索命中的次数
    last_hit_at: str = ""  # 最后一次被命中的时间

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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """从字典反序列化"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            source_type=data.get("source_type", "manual"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            doc_id=data.get("doc_id"),
            importance=data.get("importance", 0.5),
            hit_count=data.get("hit_count", 0),
            last_hit_at=data.get("last_hit_at", ""),
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
        # 确保目录结构存在
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """确保所有必需的目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "memory_index"), exist_ok=True)

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

    # ==================== Profile 操作 ====================

    def load_profile(self) -> dict:
        """加载用户画像，文件不存在时返回默认结构"""
        data = self._read_json(self.profile_path)
        if data is None:
            return self._default_profile()
        return data

    def save_profile(self, profile: dict) -> None:
        """保存用户画像"""
        self._write_json(self.profile_path, profile)

    # ==================== Session 操作 ====================

    def _session_path(self, doc_id: str) -> str:
        """获取文档会话记忆文件路径"""
        return os.path.join(self.sessions_dir, f"{doc_id}_session.json")

    def load_session(self, doc_id: str) -> dict:
        """加载文档会话记忆，文件不存在时返回默认结构"""
        data = self._read_json(self._session_path(doc_id))
        if data is None:
            return self._default_session(doc_id)
        return data

    def save_session(self, doc_id: str, session: dict) -> None:
        """保存文档会话记忆"""
        self._write_json(self._session_path(doc_id), session)

    # ==================== 条目 CRUD ====================

    def get_all_entries(self) -> list:
        """获取所有记忆条目（从 profile + 所有 session 中汇总）"""
        entries: list[MemoryEntry] = []

        # 从 profile 中收集
        profile = self.load_profile()
        for entry_data in profile.get("entries", []):
            entries.append(MemoryEntry.from_dict(entry_data))

        # 从所有 session 中收集
        if os.path.exists(self.sessions_dir):
            for filename in os.listdir(self.sessions_dir):
                if not filename.endswith("_session.json"):
                    continue
                filepath = os.path.join(self.sessions_dir, filename)
                data = self._read_json(filepath)
                if data is None:
                    continue
                # 从 qa_summaries 中收集（转换为 MemoryEntry）
                for item in data.get("qa_summaries", []):
                    entry = MemoryEntry(
                        id=item.get("id", str(uuid.uuid4())),
                        content=f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}",
                        source_type=item.get("source_type", "auto_qa"),
                        created_at=item.get("created_at", ""),
                        doc_id=data.get("doc_id"),
                        importance=item.get("importance", 0.5),
                    )
                    entries.append(entry)
                # 从 important_memories 中收集
                for item in data.get("important_memories", []):
                    entries.append(MemoryEntry.from_dict({
                        **item,
                        "doc_id": data.get("doc_id"),
                    }))

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
            return True

        # 在所有 session 中查找
        if os.path.exists(self.sessions_dir):
            for filename in os.listdir(self.sessions_dir):
                if not filename.endswith("_session.json"):
                    continue
                filepath = os.path.join(self.sessions_dir, filename)
                data = self._read_json(filepath)
                if data is None:
                    continue
                doc_id = data.get("doc_id", filename.replace("_session.json", ""))

                # 在 qa_summaries 中查找
                orig_qa = len(data.get("qa_summaries", []))
                data["qa_summaries"] = [
                    s for s in data.get("qa_summaries", []) if s.get("id") != entry_id
                ]
                if len(data["qa_summaries"]) < orig_qa:
                    data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                    self.save_session(doc_id, data)
                    return True

                # 在 important_memories 中查找
                orig_im = len(data.get("important_memories", []))
                data["important_memories"] = [
                    m for m in data.get("important_memories", []) if m.get("id") != entry_id
                ]
                if len(data["important_memories"]) < orig_im:
                    data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                    self.save_session(doc_id, data)
                    return True

        return False

    def update_entry(self, entry_id: str, content: str) -> bool:
        """更新指定记忆条目的内容，返回是否成功"""
        # 先在 profile 中查找
        profile = self.load_profile()
        for entry in profile.get("entries", []):
            if entry.get("id") == entry_id:
                entry["content"] = content
                profile["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.save_profile(profile)
                return True

        # 在所有 session 中查找
        if os.path.exists(self.sessions_dir):
            for filename in os.listdir(self.sessions_dir):
                if not filename.endswith("_session.json"):
                    continue
                filepath = os.path.join(self.sessions_dir, filename)
                data = self._read_json(filepath)
                if data is None:
                    continue
                doc_id = data.get("doc_id", filename.replace("_session.json", ""))

                # 在 qa_summaries 中查找
                for item in data.get("qa_summaries", []):
                    if item.get("id") == entry_id:
                        # qa_summaries 的 content 是 question + answer 的组合
                        # 更新时直接替换整个内容
                        item["question"] = content
                        item["answer"] = ""
                        data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                        self.save_session(doc_id, data)
                        return True

                # 在 important_memories 中查找
                for item in data.get("important_memories", []):
                    if item.get("id") == entry_id:
                        item["content"] = content
                        data["last_accessed"] = datetime.now(timezone.utc).isoformat()
                        self.save_session(doc_id, data)
                        return True

        return False

    def clear_all(self) -> None:
        """清空所有记忆数据"""
        # 重置 profile
        self.save_profile(self._default_profile())

        # 删除所有 session 文件
        if os.path.exists(self.sessions_dir):
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith("_session.json"):
                    filepath = os.path.join(self.sessions_dir, filename)
                    try:
                        os.remove(filepath)
                    except OSError as e:
                        logger.warning(f"删除 session 文件失败 {filepath}: {e}")

        # 删除索引文件
        index_dir = os.path.join(self.data_dir, "memory_index")
        if os.path.exists(index_dir):
            for filename in os.listdir(index_dir):
                filepath = os.path.join(index_dir, filename)
                try:
                    os.remove(filepath)
                except OSError as e:
                    logger.warning(f"删除索引文件失败 {filepath}: {e}")
