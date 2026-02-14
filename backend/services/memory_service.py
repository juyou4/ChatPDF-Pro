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
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from services.keyword_extractor import KeywordExtractor
from services.memory_index import MemoryIndex
from services.memory_retriever import MemoryRetriever
from services.memory_store import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_MAX_SUMMARIES = 50  # QA 摘要数量上限
DEFAULT_KEYWORD_THRESHOLD = 3  # 关键词频率阈值
DEFAULT_RETRIEVAL_TOP_K = 3  # 记忆检索返回条数
QUESTION_MAX_LEN = 100  # 问题截取最大长度
ANSWER_MAX_LEN = 200  # 回答截取最大长度


class MemoryService:
    """记忆管理核心服务（单例）"""

    def __init__(self, data_dir: str, embedding_model_id: str = "local-minilm"):
        """
        初始化记忆管理服务

        Args:
            data_dir: 记忆数据根目录，如 "data/memory/"
            embedding_model_id: embedding 模型 ID
        """
        self.data_dir = data_dir
        self.store = MemoryStore(data_dir)
        self.index = MemoryIndex(
            os.path.join(data_dir, "memory_index"), embedding_model_id
        )
        self.retriever = MemoryRetriever(self.store, self.index)
        self.keyword_extractor = KeywordExtractor()
        self.max_summaries = DEFAULT_MAX_SUMMARIES
        self.keyword_threshold = DEFAULT_KEYWORD_THRESHOLD

        # 尝试加载已有的向量索引
        self.index.load()

    # ==================== 记忆检索 ====================

    def retrieve_memories(
        self, query: str, top_k: int = DEFAULT_RETRIEVAL_TOP_K, api_key: str = None
    ) -> str:
        """检索相关记忆并返回格式化的上下文字符串

        Args:
            query: 用户查询文本
            top_k: 返回的最大结果数
            api_key: API 密钥（远程模型需要）

        Returns:
            格式化的记忆上下文字符串，无记忆时返回空字符串
        """
        try:
            memories = self.retriever.retrieve(query, top_k=top_k, api_key=api_key)
            return self.retriever.build_memory_context(memories)
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return ""

    # ==================== 记忆写入 ====================

    def save_qa_summary(
        self, doc_id: str, chat_history: list[dict], n: int = 3
    ) -> None:
        """从对话历史中提取最后 N 轮 QA 摘要并保存

        每轮 QA 包含一问一答（user + assistant）。
        问题截取前 100 字符，回答截取前 200 字符。
        超过上限时移除最早的非重要摘要。

        Args:
            doc_id: 文档标识
            chat_history: 对话历史列表，每项包含 role 和 content
            n: 提取最后 N 轮 QA 对，默认 3
        """
        if not chat_history or not doc_id:
            return

        # 提取 QA 对：从对话历史中配对 user/assistant 消息
        qa_pairs = self._extract_qa_pairs(chat_history)
        if not qa_pairs:
            return

        # 取最后 N 轮
        recent_pairs = qa_pairs[-n:]

        # 加载当前 session
        session = self.store.load_session(doc_id)

        # 生成摘要并添加
        for question, answer in recent_pairs:
            truncated_q = question[:QUESTION_MAX_LEN]
            truncated_a = answer[:ANSWER_MAX_LEN]

            summary = {
                "id": str(uuid.uuid4()),
                "question": truncated_q,
                "answer": truncated_a,
                "source_type": "auto_qa",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "importance": 0.5,
            }
            session["qa_summaries"].append(summary)

        # 摘要数量上限控制
        self._enforce_summary_limit(session)

        # 更新最后访问时间并保存
        session["last_accessed"] = datetime.now(timezone.utc).isoformat()
        self.store.save_session(doc_id, session)

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
                question = current.get("content", "")
                answer = next_msg.get("content", "")
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
                    summaries.pop(i)
                    removed = True
                    break
            if not removed:
                # 所有摘要都是重要的，无法移除，退出循环
                break
        session["qa_summaries"] = summaries

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
            importance=1.0,  # 重要记忆默认 1.0
        )

        # 保存到 session 的 important_memories
        session = self.store.load_session(doc_id)
        session["important_memories"].append(entry.to_dict())
        session["last_accessed"] = datetime.now(timezone.utc).isoformat()
        self.store.save_session(doc_id, session)

        # 添加到向量索引
        try:
            self.index.add_entry(entry.id, content)
        except Exception as e:
            logger.error(f"添加重要记忆到向量索引失败: {e}")

        return entry

    def update_keywords(self, query: str) -> None:
        """从查询中提取关键词并更新用户画像

        提取关键词 → 更新频率统计 → 更新关注领域列表。

        Args:
            query: 用户查询文本
        """
        if not query or not query.strip():
            return

        keywords = self.keyword_extractor.extract_keywords(query)
        if not keywords:
            return

        profile = self.store.load_profile()
        profile = self.keyword_extractor.update_frequency(profile, keywords)

        # 更新关注领域列表
        profile["focus_areas"] = self.keyword_extractor.get_focus_areas(
            profile, threshold=self.keyword_threshold
        )

        self.store.save_profile(profile)

    # ==================== CRUD 操作 ====================

    def get_profile(self) -> dict:
        """获取用户画像数据"""
        return self.store.load_profile()

    def get_session(self, doc_id: str) -> dict:
        """获取指定文档的会话记忆"""
        return self.store.load_session(doc_id)

    def add_entry(
        self, content: str, source_type: str, doc_id: str = None
    ) -> MemoryEntry:
        """添加记忆条目

        同时添加到 store 和 index。

        Args:
            content: 记忆内容文本
            source_type: 来源类型
            doc_id: 关联的文档 ID（可选）

        Returns:
            创建的 MemoryEntry 对象
        """
        importance = 1.0 if source_type in ("manual", "liked") else 0.5
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            source_type=source_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            doc_id=doc_id,
            importance=importance,
        )

        # 保存到 store
        self.store.add_entry(entry)

        # 添加到向量索引
        try:
            self.index.add_entry(entry.id, content)
        except Exception as e:
            logger.error(f"添加记忆条目到向量索引失败: {e}")

        return entry

    def delete_entry(self, entry_id: str) -> bool:
        """删除指定记忆条目

        同时从 store 和 index 中移除。

        Args:
            entry_id: 记忆条目 ID

        Returns:
            是否删除成功
        """
        success = self.store.delete_entry(entry_id)
        if success:
            try:
                self.index.remove_entry(entry_id)
            except Exception as e:
                logger.error(f"从向量索引移除记忆条目失败: {e}")
        return success

    def update_entry(self, entry_id: str, content: str) -> bool:
        """更新指定记忆条目的内容

        同时更新 store 和 index。

        Args:
            entry_id: 记忆条目 ID
            content: 新的内容文本

        Returns:
            是否更新成功
        """
        success = self.store.update_entry(entry_id, content)
        if success:
            try:
                # 先移除旧的向量，再添加新的
                self.index.remove_entry(entry_id)
                self.index.add_entry(entry_id, content)
            except Exception as e:
                logger.error(f"更新向量索引失败: {e}")
        return success

    def clear_all(self) -> None:
        """清空所有记忆数据

        同时清空 store 和 index。
        """
        self.store.clear_all()
        try:
            self.index.rebuild([])
        except Exception as e:
            logger.error(f"清空向量索引失败: {e}")

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
        }
