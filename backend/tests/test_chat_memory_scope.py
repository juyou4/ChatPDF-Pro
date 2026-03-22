"""测试聊天路由中的记忆作用域选择。"""
import os
import sys
from unittest.mock import MagicMock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.chat_routes as chat_routes


class TestChatMemoryScope:
    def test_retrieve_memory_context_filters_by_doc_when_doc_id_present(self):
        mock_service = MagicMock()
        mock_service.retrieve_memories.return_value = "用户历史记忆"
        original = chat_routes.memory_service
        chat_routes.memory_service = mock_service
        try:
            result = chat_routes._retrieve_memory_context(
                "请总结本文",
                api_key="test-key",
                doc_id="doc-1",
            )
        finally:
            chat_routes.memory_service = original

        assert result == "用户历史记忆"
        mock_service.retrieve_memories.assert_called_once_with(
            "请总结本文",
            api_key="test-key",
            doc_id="doc-1",
            filter_by_doc=True,
        )

    def test_retrieve_memory_context_keeps_global_scope_without_doc_id(self):
        mock_service = MagicMock()
        mock_service.retrieve_memories.return_value = ""
        original = chat_routes.memory_service
        chat_routes.memory_service = mock_service
        try:
            chat_routes._retrieve_memory_context("用户偏好")
        finally:
            chat_routes.memory_service = original

        mock_service.retrieve_memories.assert_called_once_with(
            "用户偏好",
            api_key=None,
            doc_id=None,
            filter_by_doc=False,
        )

    def test_retrieve_raw_memories_filters_by_doc_when_doc_id_present(self):
        mock_service = MagicMock()
        mock_service.retrieve_memories_raw.return_value = [{"content": "当前文档记忆"}]
        original = chat_routes.memory_service
        chat_routes.memory_service = mock_service
        try:
            result = chat_routes._retrieve_raw_memories(
                "请解释图2",
                api_key="test-key",
                doc_id="doc-2",
            )
        finally:
            chat_routes.memory_service = original

        assert result == [{"content": "当前文档记忆"}]
        mock_service.retrieve_memories_raw.assert_called_once_with(
            "请解释图2",
            api_key="test-key",
            doc_id="doc-2",
            filter_by_doc=True,
            chat_history=None,
        )

    def test_smart_inject_memory_returns_selected_hits(self):
        mock_service = MagicMock()
        mock_injector = MagicMock()
        mock_injector.token_budget = 800
        mock_injector.prepare_memories.return_value = [{"id": "mem-1", "summary": "当前文档记忆"}]
        mock_injector.inject.return_value = "system with memory"
        mock_service.context_injector = mock_injector
        original = chat_routes.memory_service
        chat_routes.memory_service = mock_service
        try:
            prompt, hits, meta = chat_routes._smart_inject_memory(
                "system",
                "用户历史记忆",
                [{"id": "mem-1", "summary": "当前文档记忆"}],
            )
        finally:
            chat_routes.memory_service = original

        assert prompt == "system with memory"
        assert hits == [{"id": "mem-1", "summary": "当前文档记忆"}]
        assert meta == {
            "enabled": True,
            "strategy": "context_injector",
            "retrieved_count": 1,
            "selected_count": 1,
            "truncated": False,
            "token_budget": 800,
            "selected_kinds": ["episodic"],
        }
