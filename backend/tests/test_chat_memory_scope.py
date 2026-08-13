"""测试聊天路由中的记忆作用域选择。"""
import os
import sys
from unittest.mock import MagicMock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.chat_routes as chat_routes

# 检索层现在会显式传 top_k（此前 memory_retrieval_top_k 配置项从未被读取）。
# 直接引用解析函数而不是写死数字，配置调整时断言会跟着走。
EXPECTED_TOP_K = chat_routes._resolve_memory_top_k(None)


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
            top_k=EXPECTED_TOP_K,
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
            top_k=EXPECTED_TOP_K,
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
            top_k=EXPECTED_TOP_K,
        )

    def test_smart_inject_memory_never_writes_memory_into_system_prompt(self):
        """记忆只能作为不可信证据出现，绝不能获得 system prompt 权限。

        ``_smart_inject_memory`` 是专门保留的安全 shim，它的 docstring 点名由这条
        测试守住「返回的 prompt 与传入完全相同」。任何让记忆重新流回 system prompt
        的改动都必须在这里失败——即使 injector 自己返回了一个拼好记忆的 prompt。
        """
        mock_service = MagicMock()
        mock_injector = MagicMock()
        mock_injector.token_budget = 800
        mock_injector.prepare_memories.return_value = [{"id": "mem-1", "summary": "当前文档记忆"}]
        # injector 故意返回一个已经把记忆拼进去的 prompt，shim 必须忽略它。
        mock_injector.inject.return_value = "system with memory"
        mock_service.context_injector = mock_injector
        original = chat_routes.memory_service
        chat_routes.memory_service = mock_service
        try:
            prompt, hits, _meta = chat_routes._smart_inject_memory(
                "system",
                "用户历史记忆",
                [{"id": "mem-1", "summary": "当前文档记忆"}],
            )
        finally:
            chat_routes.memory_service = original

        assert prompt == "system"
        assert "记忆" not in prompt
        assert hits == [{"id": "mem-1", "summary": "当前文档记忆"}]

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

        assert prompt == "system"
        assert hits == [{"id": "mem-1", "summary": "当前文档记忆"}]
        # 只断言这条测试关心的选择结果，不做整字典相等：meta 还携带预算与隐私模式
        # 等诊断字段，随实现演进增删，全等断言会在每次加字段时误报。
        expected = {
            "enabled": True,
            "strategy": "context_injector",
            "retrieved_count": 1,
            "selected_count": 1,
            "truncated": False,
            "token_budget": 800,
            "selected_kinds": ["episodic"],
        }
        assert {key: meta.get(key) for key in expected} == expected
