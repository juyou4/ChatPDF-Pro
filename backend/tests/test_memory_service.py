"""测试 MemoryService 记忆管理核心服务

验证需求 2.5, 3.1, 3.2, 3.3, 3.4, 3.6：
- 手动添加记忆条目存储到 User_Profile
- 从对话历史提取最后 N 轮 QA 摘要
- 截取问题前 100 字符和回答前 200 字符
- 点击"记住这个"标记为重要记忆
- 点赞标记为重要记忆
- QA 摘要数量超过上限时移除最早的非重要摘要
"""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.memory_service import MemoryService, QUESTION_MAX_LEN, ANSWER_MAX_LEN
from services.memory_store import MemoryEntry


@pytest.fixture
def service(tmp_path):
    """创建使用临时目录的 MemoryService 实例，mock 掉向量索引"""
    data_dir = str(tmp_path / "memory")
    # mock MemoryIndex 避免依赖 embedding_service 和 FAISS
    with patch("services.memory_service.MemoryIndex") as MockIndex:
        mock_index = MagicMock()
        mock_index.index = None
        mock_index.load.return_value = False
        MockIndex.return_value = mock_index

        with patch("services.memory_service.MemoryRetriever") as MockRetriever:
            mock_retriever = MagicMock()
            MockRetriever.return_value = mock_retriever

            svc = MemoryService(data_dir)
            svc._mock_index = mock_index
            svc._mock_retriever = mock_retriever
            yield svc


# ==================== retrieve_memories 测试 ====================


class TestRetrieveMemories:
    """测试记忆检索"""

    def test_retrieve_returns_formatted_context(self, service):
        """检索应返回格式化的上下文字符串"""
        service._mock_retriever.retrieve.return_value = [
            {"text": "记忆1", "source_type": "manual"}
        ]
        service._mock_retriever.build_memory_context.return_value = (
            "用户历史记忆：\n- [manual] 记忆1"
        )
        result = service.retrieve_memories("测试查询")
        assert "记忆1" in result
        service._mock_retriever.retrieve.assert_called_once()

    def test_retrieve_handles_exception(self, service):
        """检索异常时应返回空字符串"""
        service._mock_retriever.retrieve.side_effect = Exception("检索失败")
        result = service.retrieve_memories("测试查询")
        assert result == ""


# ==================== save_qa_summary 测试 ====================


class TestSaveQaSummary:
    """测试 QA 摘要提取和保存（需求 3.1, 3.2）"""

    def test_extract_last_n_pairs(self, service):
        """应提取最后 N 轮 QA 对"""
        chat_history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
            {"role": "assistant", "content": "回答2"},
            {"role": "user", "content": "问题3"},
            {"role": "assistant", "content": "回答3"},
        ]
        service.save_qa_summary("doc-1", chat_history, n=2)
        session = service.store.load_session("doc-1")
        # 应只保存最后 2 轮
        assert len(session["qa_summaries"]) == 2
        assert session["qa_summaries"][0]["question"] == "问题2"
        assert session["qa_summaries"][1]["question"] == "问题3"

    def test_truncate_question_and_answer(self, service):
        """问题应截取前 100 字符，回答应截取前 200 字符（需求 3.2）"""
        long_question = "问" * 200  # 200 字符
        long_answer = "答" * 400  # 400 字符
        chat_history = [
            {"role": "user", "content": long_question},
            {"role": "assistant", "content": long_answer},
        ]
        service.save_qa_summary("doc-1", chat_history, n=1)
        session = service.store.load_session("doc-1")
        summary = session["qa_summaries"][0]
        assert len(summary["question"]) == QUESTION_MAX_LEN
        assert len(summary["answer"]) == ANSWER_MAX_LEN

    def test_empty_chat_history(self, service):
        """空对话历史不应保存任何摘要"""
        service.save_qa_summary("doc-1", [], n=3)
        session = service.store.load_session("doc-1")
        assert len(session["qa_summaries"]) == 0

    def test_no_doc_id(self, service):
        """空 doc_id 不应保存"""
        service.save_qa_summary("", [{"role": "user", "content": "q"}], n=1)
        # 不应抛出异常

    def test_fewer_pairs_than_n(self, service):
        """对话历史不足 N 轮时应保存所有可用的 QA 对"""
        chat_history = [
            {"role": "user", "content": "唯一的问题"},
            {"role": "assistant", "content": "唯一的回答"},
        ]
        service.save_qa_summary("doc-1", chat_history, n=5)
        session = service.store.load_session("doc-1")
        assert len(session["qa_summaries"]) == 1

    def test_non_paired_messages_skipped(self, service):
        """非配对的消息应被跳过"""
        chat_history = [
            {"role": "system", "content": "系统消息"},
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "未回答的问题"},
        ]
        service.save_qa_summary("doc-1", chat_history, n=3)
        session = service.store.load_session("doc-1")
        # 只有一个完整的 QA 对
        assert len(session["qa_summaries"]) == 1
        assert session["qa_summaries"][0]["question"] == "问题1"

    def test_summary_source_type_is_auto_qa(self, service):
        """摘要的 source_type 应为 auto_qa"""
        chat_history = [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ]
        service.save_qa_summary("doc-1", chat_history, n=1)
        session = service.store.load_session("doc-1")
        assert session["qa_summaries"][0]["source_type"] == "auto_qa"
        assert session["qa_summaries"][0]["importance"] == 0.5


# ==================== 摘要上限控制测试 ====================


class TestSummaryLimit:
    """测试摘要数量上限控制（需求 3.6）"""

    def test_enforce_limit_removes_oldest_non_important(self, service):
        """超过上限时应移除最早的非重要摘要"""
        service.max_summaries = 3

        # 先保存 3 轮摘要
        history1 = [
            {"role": "user", "content": f"问题{i}"}
            if i % 2 == 0
            else {"role": "assistant", "content": f"回答{i}"}
            for i in range(6)  # 3 轮 QA
        ]
        service.save_qa_summary("doc-1", history1, n=3)
        session = service.store.load_session("doc-1")
        assert len(session["qa_summaries"]) == 3

        # 再保存 2 轮，应触发上限控制
        history2 = [
            {"role": "user", "content": "新问题1"},
            {"role": "assistant", "content": "新回答1"},
            {"role": "user", "content": "新问题2"},
            {"role": "assistant", "content": "新回答2"},
        ]
        service.save_qa_summary("doc-1", history2, n=2)
        session = service.store.load_session("doc-1")
        # 应保持在上限内
        assert len(session["qa_summaries"]) <= 3

    def test_important_summaries_not_removed(self, service):
        """重要摘要不应被移除"""
        service.max_summaries = 2

        # 手动构建包含重要摘要的 session
        session = service.store.load_session("doc-1")
        session["qa_summaries"] = [
            {
                "id": "important-1",
                "question": "重要问题",
                "answer": "重要回答",
                "source_type": "auto_qa",
                "created_at": "2024-01-01T00:00:00Z",
                "importance": 1.0,  # 重要
            },
            {
                "id": "normal-1",
                "question": "普通问题",
                "answer": "普通回答",
                "source_type": "auto_qa",
                "created_at": "2024-01-02T00:00:00Z",
                "importance": 0.5,  # 非重要
            },
        ]
        service.store.save_session("doc-1", session)

        # 再添加一轮，触发上限
        history = [
            {"role": "user", "content": "新问题"},
            {"role": "assistant", "content": "新回答"},
        ]
        service.save_qa_summary("doc-1", history, n=1)
        session = service.store.load_session("doc-1")

        # 重要摘要应保留
        ids = [s["id"] for s in session["qa_summaries"]]
        assert "important-1" in ids
        # 非重要的应被移除
        assert "normal-1" not in ids


# ==================== save_important_memory 测试 ====================


class TestSaveImportantMemory:
    """测试保存重要记忆（需求 3.3, 3.4）"""

    def test_save_manual_memory(self, service):
        """手动标记的记忆应保存到 session"""
        entry = service.save_important_memory(
            "doc-1", "这个公式什么意思？", "这是注意力机制的公式", "manual"
        )
        assert entry.importance == 1.0
        assert entry.source_type == "manual"
        assert entry.doc_id == "doc-1"

        session = service.store.load_session("doc-1")
        assert len(session["important_memories"]) == 1
        assert "Q: 这个公式什么意思？" in session["important_memories"][0]["content"]

    def test_save_liked_memory(self, service):
        """点赞的记忆应保存到 session"""
        entry = service.save_important_memory(
            "doc-1", "总结一下", "本文主要贡献是...", "liked"
        )
        assert entry.source_type == "liked"
        assert entry.importance == 1.0

    def test_adds_to_index(self, service):
        """重要记忆应同时添加到向量索引"""
        service.save_important_memory("doc-1", "问题", "回答")
        service._mock_index.add_entry.assert_called_once()


# ==================== update_keywords 测试 ====================


class TestUpdateKeywords:
    """测试关键词更新"""

    def test_updates_profile_keywords(self, service):
        """应更新用户画像中的关键词频率"""
        service.update_keywords("机器学习和深度学习的区别")
        profile = service.store.load_profile()
        freq = profile.get("keyword_frequencies", {})
        # 应有关键词被提取和记录
        assert len(freq) > 0

    def test_empty_query_no_update(self, service):
        """空查询不应更新"""
        service.update_keywords("")
        profile = service.store.load_profile()
        assert profile["keyword_frequencies"] == {}

    def test_focus_areas_updated(self, service):
        """超过阈值的关键词应出现在关注领域"""
        service.keyword_threshold = 2
        # 多次查询相同关键词
        service.update_keywords("transformer 架构")
        service.update_keywords("transformer 模型")
        profile = service.store.load_profile()
        # transformer 出现 2 次，应在关注领域中
        assert "transformer" in profile.get("focus_areas", [])


# ==================== CRUD 操作测试 ====================


class TestCRUDOperations:
    """测试 CRUD 方法"""

    def test_get_profile(self, service):
        """应返回用户画像"""
        profile = service.get_profile()
        assert "focus_areas" in profile
        assert "keyword_frequencies" in profile

    def test_get_session(self, service):
        """应返回文档会话记忆"""
        session = service.get_session("doc-1")
        assert session["doc_id"] == "doc-1"
        assert "qa_summaries" in session

    def test_add_entry_manual(self, service):
        """手动添加的条目 importance 应为 1.0"""
        entry = service.add_entry("用户偏好中文", "manual")
        assert entry.importance == 1.0
        assert entry.source_type == "manual"
        assert entry.doc_id is None
        # 应存入 profile
        profile = service.store.load_profile()
        assert len(profile["entries"]) == 1

    def test_add_entry_auto(self, service):
        """自动添加的条目 importance 应为 0.5"""
        entry = service.add_entry("自动记忆", "auto_qa", doc_id="doc-1")
        assert entry.importance == 0.5

    def test_add_entry_adds_to_index(self, service):
        """添加条目应同时添加到向量索引"""
        service.add_entry("测试内容", "manual")
        service._mock_index.add_entry.assert_called_once()

    def test_delete_entry(self, service):
        """删除条目应同时从 store 和 index 移除"""
        entry = service.add_entry("待删除", "manual")
        result = service.delete_entry(entry.id)
        assert result is True
        service._mock_index.remove_entry.assert_called_with(entry.id)

    def test_delete_nonexistent_entry(self, service):
        """删除不存在的条目应返回 False"""
        result = service.delete_entry("nonexistent-id")
        assert result is False

    def test_update_entry(self, service):
        """更新条目应同时更新 store 和 index"""
        entry = service.add_entry("原始内容", "manual")
        result = service.update_entry(entry.id, "新内容")
        assert result is True
        # 应先移除旧向量再添加新向量
        service._mock_index.remove_entry.assert_called_with(entry.id)
        # add_entry 被调用了两次（add_entry + update_entry）
        assert service._mock_index.add_entry.call_count == 2

    def test_update_nonexistent_entry(self, service):
        """更新不存在的条目应返回 False"""
        result = service.update_entry("nonexistent-id", "新内容")
        assert result is False

    def test_clear_all(self, service):
        """清空应同时清空 store 和 index"""
        service.add_entry("条目1", "manual")
        service.add_entry("条目2", "manual")
        service.clear_all()
        profile = service.store.load_profile()
        assert profile["entries"] == []
        service._mock_index.rebuild.assert_called_with([])

    def test_get_status(self, service):
        """应返回正确的状态信息"""
        service.add_entry("条目1", "manual")
        status = service.get_status()
        assert "enabled" in status
        assert "total_entries" in status
        assert "index_size" in status
        assert "profile_focus_areas" in status
        assert status["enabled"] is True
        assert status["total_entries"] >= 1
