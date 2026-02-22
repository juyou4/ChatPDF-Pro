"""
记忆 SQLite 存储层（可选增强）

使用 SQLite + FTS5 提供更好的查询性能和事务支持。
保留 JSON 作为兼容层，支持渐进式迁移。
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from services.memory_store import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)


class MemoryStoreSQLite(MemoryStore):
    """基于 SQLite 的记忆存储（继承 MemoryStore，增强查询性能）"""
    
    def __init__(self, data_dir: str, use_sqlite: bool = False):
        """
        初始化 SQLite 存储
        
        Args:
            data_dir: 记忆数据根目录
            use_sqlite: 是否启用 SQLite（默认 False，保持向后兼容）
        """
        super().__init__(data_dir)
        self.use_sqlite = use_sqlite
        self.db_path = Path(data_dir) / "memory.db"
        self._db: Optional[sqlite3.Connection] = None
        
        if use_sqlite:
            self._init_database()
    
    def _init_database(self) -> None:
        """初始化 SQLite 数据库和 FTS5 虚拟表"""
        try:
            self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            
            # 创建主表
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_type TEXT,
                    doc_id TEXT,
                    importance REAL DEFAULT 0.5,
                    created_at TEXT,
                    hit_count INTEGER DEFAULT 0,
                    last_hit_at TEXT,
                    memory_tier TEXT DEFAULT 'short_term',
                    tags TEXT DEFAULT '[]'
                )
            """)
            
            # 为旧表添加新列（如果不存在）
            self._migrate_add_columns()
            
            # 创建 FTS5 虚拟表（用于全文检索）
            try:
                self._db.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        content,
                        content='memory_entries',
                        content_rowid='rowid'
                    )
                """)
            except sqlite3.OperationalError as e:
                # FTS5 可能不可用（某些 SQLite 版本）
                logger.warning(f"FTS5 虚拟表创建失败，将使用普通全文检索: {e}")
            
            # 创建索引
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_id ON memory_entries(doc_id)
            """)
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at)
            """)
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memory_entries(importance)
            """)
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory_entries(memory_tier)
            """)
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_hit_at ON memory_entries(last_hit_at)
            """)
            
            self._db.commit()
            logger.info(f"SQLite 数据库已初始化: {self.db_path}")
        except Exception as e:
            logger.error(f"初始化 SQLite 数据库失败: {e}")
            self.use_sqlite = False
            self._db = None

    def _migrate_add_columns(self) -> None:
        """为旧表添加 memory_tier 和 tags 列（如果不存在）"""
        if not self._db:
            return
        try:
            # 检查列是否存在
            cursor = self._db.execute("PRAGMA table_info(memory_entries)")
            columns = {row[1] for row in cursor.fetchall()}
            if "memory_tier" not in columns:
                self._db.execute("ALTER TABLE memory_entries ADD COLUMN memory_tier TEXT DEFAULT 'short_term'")
                logger.info("SQLite: 已添加 memory_tier 列")
            if "tags" not in columns:
                self._db.execute("ALTER TABLE memory_entries ADD COLUMN tags TEXT DEFAULT '[]'")
                logger.info("SQLite: 已添加 tags 列")
            self._db.commit()
        except Exception as e:
            logger.warning(f"SQLite 列迁移失败: {e}")
    
    def _sync_from_json(self) -> None:
        """从 JSON 文件同步数据到 SQLite（一次性迁移）"""
        if not self.use_sqlite or not self._db:
            return
        
        try:
            # 检查是否已有数据
            count = self._db.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
            if count > 0:
                logger.debug("SQLite 数据库已有数据，跳过同步")
                return
            
            # 从 JSON 加载所有条目
            all_entries = super().get_all_entries()
            
            if not all_entries:
                return
            
            # 批量插入（包含新字段 memory_tier 和 tags）
            import json as _json
            self._db.executemany("""
                INSERT OR IGNORE INTO memory_entries 
                (id, content, source_type, doc_id, importance, created_at, hit_count, last_hit_at, memory_tier, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    e.id, e.content, e.source_type, e.doc_id,
                    e.importance, e.created_at, e.hit_count, e.last_hit_at,
                    e.memory_tier, _json.dumps(e.tags, ensure_ascii=False)
                )
                for e in all_entries
            ])
            
            # 同步 FTS5 索引
            if self._has_fts5():
                self._db.execute("""
                    INSERT INTO memory_fts(rowid, content)
                    SELECT rowid, content FROM memory_entries
                    WHERE rowid NOT IN (SELECT rowid FROM memory_fts)
                """)
            
            self._db.commit()
            logger.info(f"从 JSON 同步了 {len(all_entries)} 条记忆到 SQLite")
        except Exception as e:
            logger.error(f"从 JSON 同步到 SQLite 失败: {e}")
            self._db.rollback()

    def migrate_json_to_sqlite(self) -> int:
        """JSON → SQLite 数据迁移工具方法

        强制从 JSON 加载所有条目并写入 SQLite，即使 SQLite 已有数据。

        Returns:
            迁移的条目数量
        """
        if not self.use_sqlite or not self._db:
            logger.warning("SQLite 未启用，无法执行迁移")
            return 0
        try:
            import json as _json
            all_entries = super().get_all_entries()
            if not all_entries:
                return 0
            self._db.executemany("""
                INSERT OR REPLACE INTO memory_entries
                (id, content, source_type, doc_id, importance, created_at, hit_count, last_hit_at, memory_tier, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    e.id, e.content, e.source_type, e.doc_id,
                    e.importance, e.created_at, e.hit_count, e.last_hit_at,
                    e.memory_tier, _json.dumps(e.tags, ensure_ascii=False)
                )
                for e in all_entries
            ])
            self._db.commit()
            logger.info(f"JSON → SQLite 迁移完成: {len(all_entries)} 条记忆")
            return len(all_entries)
        except Exception as e:
            logger.error(f"JSON → SQLite 迁移失败: {e}")
            self._db.rollback()
            return 0
    
    def _has_fts5(self) -> bool:
        """检查 FTS5 是否可用"""
        if not self._db:
            return False
        try:
            self._db.execute("SELECT 1 FROM memory_fts LIMIT 1")
            return True
        except sqlite3.OperationalError:
            return False
    
    def get_all_entries(self) -> List[MemoryEntry]:
        """获取所有记忆条目（优先从 SQLite 读取，回退到 JSON）"""
        if self.use_sqlite and self._db:
            try:
                # 首次使用时同步 JSON 数据
                self._sync_from_json()
                
                rows = self._db.execute("""
                    SELECT id, content, source_type, doc_id, importance,
                           created_at, hit_count, last_hit_at, memory_tier, tags
                    FROM memory_entries
                    ORDER BY created_at DESC
                """).fetchall()
                
                import json as _json
                entries = []
                for row in rows:
                    # 解析 tags JSON 字符串
                    tags_raw = row["tags"] if "tags" in row.keys() else "[]"
                    try:
                        tags = _json.loads(tags_raw) if tags_raw else []
                    except (ValueError, TypeError):
                        tags = []
                    entries.append(MemoryEntry(
                        id=row["id"],
                        content=row["content"],
                        source_type=row["source_type"],
                        doc_id=row["doc_id"],
                        importance=row["importance"],
                        created_at=row["created_at"],
                        hit_count=row["hit_count"],
                        last_hit_at=row["last_hit_at"],
                        memory_tier=row["memory_tier"] if "memory_tier" in row.keys() else "short_term",
                        tags=tags,
                    ))
                return entries
            except Exception as e:
                logger.warning(f"从 SQLite 读取失败，回退到 JSON: {e}")
        
        # 回退到 JSON
        return super().get_all_entries()
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """添加记忆条目（同时写入 SQLite 和 JSON）"""
        # 先调用父类方法写入 JSON
        super().add_entry(entry)
        
        # 如果启用 SQLite，也写入 SQLite
        if self.use_sqlite and self._db:
            try:
                import json as _json
                self._db.execute("""
                    INSERT OR REPLACE INTO memory_entries
                    (id, content, source_type, doc_id, importance, created_at, hit_count, last_hit_at, memory_tier, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id, entry.content, entry.source_type, entry.doc_id,
                    entry.importance, entry.created_at, entry.hit_count, entry.last_hit_at,
                    entry.memory_tier, _json.dumps(entry.tags, ensure_ascii=False)
                ))
                
                # 更新 FTS5 索引
                if self._has_fts5():
                    self._db.execute("""
                        INSERT INTO memory_fts(rowid, content)
                        SELECT rowid, content FROM memory_entries WHERE id = ?
                    """, (entry.id,))
                
                self._db.commit()
            except Exception as e:
                logger.warning(f"写入 SQLite 失败，已写入 JSON: {e}")
                self._db.rollback()
    
    def delete_entry(self, entry_id: str) -> bool:
        """删除记忆条目（同时从 SQLite 和 JSON 删除）"""
        # 先调用父类方法从 JSON 删除
        success = super().delete_entry(entry_id)
        
        # 如果启用 SQLite，也从 SQLite 删除
        if success and self.use_sqlite and self._db:
            try:
                # 先删除 FTS5 索引
                if self._has_fts5():
                    rowid = self._db.execute(
                        "SELECT rowid FROM memory_entries WHERE id = ?",
                        (entry_id,)
                    ).fetchone()
                    if rowid:
                        self._db.execute(
                            "DELETE FROM memory_fts WHERE rowid = ?",
                            (rowid[0],)
                        )
                
                # 删除主表记录
                cursor = self._db.execute(
                    "DELETE FROM memory_entries WHERE id = ?",
                    (entry_id,)
                )
                self._db.commit()
                
                if cursor.rowcount > 0:
                    return True
            except Exception as e:
                logger.warning(f"从 SQLite 删除失败: {e}")
                self._db.rollback()
        
        return success
    
    def update_entry(self, entry_id: str, content: str) -> bool:
        """更新记忆条目（同时更新 SQLite 和 JSON）"""
        # 先调用父类方法更新 JSON
        success = super().update_entry(entry_id, content)
        
        # 如果启用 SQLite，也更新 SQLite
        if success and self.use_sqlite and self._db:
            try:
                self._db.execute("""
                    UPDATE memory_entries SET content = ? WHERE id = ?
                """, (content, entry_id))
                
                # 更新 FTS5 索引
                if self._has_fts5():
                    rowid = self._db.execute(
                        "SELECT rowid FROM memory_entries WHERE id = ?",
                        (entry_id,)
                    ).fetchone()
                    if rowid:
                        self._db.execute("""
                            INSERT INTO memory_fts(rowid, content)
                            VALUES (?, ?)
                            ON CONFLICT(rowid) DO UPDATE SET content = excluded.content
                        """, (rowid[0], content))
                
                self._db.commit()
            except Exception as e:
                logger.warning(f"更新 SQLite 失败: {e}")
                self._db.rollback()
        
        return success
    
    def search_fts(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """使用 FTS5 全文检索（如果可用）
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            匹配的记忆条目列表
        """
        if not self.use_sqlite or not self._db or not self._has_fts5():
            return []
        
        try:
            # FTS5 搜索
            rows = self._db.execute("""
                SELECT m.id, m.content, m.source_type, m.doc_id, m.importance,
                       m.created_at, m.hit_count, m.last_hit_at
                FROM memory_fts f
                JOIN memory_entries m ON f.rowid = m.rowid
                WHERE memory_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            
            entries = []
            for row in rows:
                entries.append(MemoryEntry(
                    id=row["id"],
                    content=row["content"],
                    source_type=row["source_type"],
                    doc_id=row["doc_id"],
                    importance=row["importance"],
                    created_at=row["created_at"],
                    hit_count=row["hit_count"],
                    last_hit_at=row["last_hit_at"],
                ))
            return entries
        except Exception as e:
            logger.warning(f"FTS5 搜索失败: {e}")
            return []
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self._db:
            try:
                self._db.close()
                self._db = None
            except Exception as e:
                logger.warning(f"关闭 SQLite 连接失败: {e}")
