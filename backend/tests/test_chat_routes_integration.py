"""
Chat 路由集成测试

测试 chat_routes.py 中 PresetService 和引文追踪的集成：
- 生成类查询（思维导图/流程图）检测并注入对应系统提示词
- 引文指示提示词追加到系统提示词
- 响应中包含 retrieval_meta 字段（含 citations）

Requirements: 5.3, 5.4, 8.3, 9.2
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient
from fastapi import FastAPI

from routes.chat_routes import router


# ---- 测试辅助工具 ----

def _create_test_app():
    """创建测试用 FastAPI 应用"""
    app = FastAPI()
    app.include_router(router, prefix="/api")

    # 初始化文档存储和向量存储目录
    router.documents_store = {
        "test-doc-1": {
            "filename": "测试文档.pdf",
            "data": {
                "full_text": "这是一个测试文档的完整文本内容。" * 100,
                "total_pages": 10,
                "pages": [{"page": i, "text": f"第{i}页内容"} for i in range(1, 11)],
            }
        }
    }
    router.vector_store_dir = "/tmp/test_vector_store"
    return app


def _make_chat_request(question="请总结本文", doc_id="test-doc-1", **kwargs):
    """构建聊天请求体"""
    base = {
        "doc_id": doc_id,
        "question": question,
        "model": "gpt-4o-mini",
        "api_provider": "openai",
        "enable_vector_search": True,
    }
    base.update(kwargs)
    return base


def _mock_ai_response(content="这是AI的回答"):
    """构建模拟的 AI API 响应"""
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "role": "assistant",
                }
            }
        ],
        "_used_provider": "openai",
        "_used_model": "gpt-4o-mini",
        "_fallback_used": False,
    }


def _mock_vector_context_with_citations():
    """构建包含 citations 的模拟 vector_context 返回值"""
    return {
        "context": "[1]【group-0 - full | 页码: 1-3】\n内容:\n这是检索到的文本内容",
        "retrieval_meta": {
            "query_type": "specific",
            "granularities": ["full", "digest"],
            "token_used": 500,
            "citations": [
                {"ref": 1, "group_id": "group-0", "page_range": [1, 3]},
                {"ref": 2, "group_id": "group-1", "page_range": [4, 6]},
            ],
        },
    }


def _mock_vector_context_without_citations():
    """构建不包含 citations 的模拟 vector_context 返回值"""
    return {
        "context": "这是检索到的普通文本内容",
        "retrieval_meta": {
            "query_type": "overview",
            "granularities": ["summary"],
            "token_used": 200,
        },
    }


# ---- /chat 端点测试 ----

class TestChatEndpointGenerationPrompt:
    """测试 /chat 端点的生成类查询提示词注入"""

    @patch("routes.chat_routes.call_ai_api", new_callable=AsyncMock)
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_思维导图查询注入mindmap提示词(self, mock_vector_ctx, mock_ai_api):
        """当用户请求生成思维导图时，系统提示词中应包含思维导图格式要求"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_without_citations()
        mock_ai_api.return_value = _mock_ai_response("# 思维导图\n## 分支1")

        client = TestClient(app)
        response = client.post("/api/chat", json=_make_chat_request(question="生成思维导图"))

        assert response.status_code == 200

        # 验证 call_ai_api 被调用时，系统提示词中包含思维导图相关内容
        call_args = mock_ai_api.call_args
        messages = call_args[0][0]  # 第一个位置参数是 messages
        system_content = messages[0]["content"]
        assert "思维导图" in system_content or "Markdown 层级结构" in system_content

    @patch("routes.chat_routes.call_ai_api", new_callable=AsyncMock)
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_流程图查询注入mermaid提示词(self, mock_vector_ctx, mock_ai_api):
        """当用户请求生成流程图时，系统提示词中应包含 Mermaid 语法要求"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_without_citations()
        mock_ai_api.return_value = _mock_ai_response("```mermaid\ngraph TD\nA-->B\n```")

        client = TestClient(app)
        response = client.post("/api/chat", json=_make_chat_request(question="生成流程图"))

        assert response.status_code == 200

        # 验证系统提示词中包含 Mermaid 相关内容
        call_args = mock_ai_api.call_args
        messages = call_args[0][0]
        system_content = messages[0]["content"]
        assert "mermaid" in system_content.lower() or "Mermaid" in system_content

    @patch("routes.chat_routes.call_ai_api", new_callable=AsyncMock)
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_普通查询不注入生成提示词(self, mock_vector_ctx, mock_ai_api):
        """普通查询不应注入思维导图或流程图的生成提示词"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_without_citations()
        mock_ai_api.return_value = _mock_ai_response("这是普通回答")

        client = TestClient(app)
        response = client.post("/api/chat", json=_make_chat_request(question="请总结本文的主要内容"))

        assert response.status_code == 200

        # 验证系统提示词中不包含 Mermaid 或思维导图的专用提示词
        call_args = mock_ai_api.call_args
        messages = call_args[0][0]
        system_content = messages[0]["content"]
        # 不应包含 Mermaid 语法要求（"graph TD" 是 Mermaid 提示词的特征）
        assert "graph TD" not in system_content
        # 不应包含思维导图层级结构要求（"Markdown 层级结构" 是思维导图提示词的特征）
        assert "Markdown 层级结构" not in system_content


class TestChatEndpointCitationPrompt:
    """测试 /chat 端点的引文指示提示词注入"""

    @patch("routes.chat_routes.call_ai_api", new_callable=AsyncMock)
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_有citations时注入引文指示提示词(self, mock_vector_ctx, mock_ai_api):
        """当 retrieval_meta 包含 citations 时，系统提示词中应追加引文指示"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_with_citations()
        mock_ai_api.return_value = _mock_ai_response("回答内容 [1]")

        client = TestClient(app)
        response = client.post("/api/chat", json=_make_chat_request(question="这篇文章讲了什么？"))

        assert response.status_code == 200

        # 验证系统提示词中包含引文指示
        call_args = mock_ai_api.call_args
        messages = call_args[0][0]
        system_content = messages[0]["content"]
        # 应包含引用编号说明
        assert "[1]" in system_content
        assert "[2]" in system_content
        # 应包含引文使用指示
        assert "引用" in system_content or "标注" in system_content

    @patch("routes.chat_routes.call_ai_api", new_callable=AsyncMock)
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_无citations时不注入引文指示(self, mock_vector_ctx, mock_ai_api):
        """当 retrieval_meta 不包含 citations 时，不应追加引文指示提示词"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_without_citations()
        mock_ai_api.return_value = _mock_ai_response("普通回答")

        client = TestClient(app)
        response = client.post("/api/chat", json=_make_chat_request(question="请总结本文"))

        assert response.status_code == 200

        # 验证系统提示词中不包含引文指示的特征内容
        call_args = mock_ai_api.call_args
        messages = call_args[0][0]
        system_content = messages[0]["content"]
        # 不应包含"可用的引用来源"这个引文指示提示词的特征文本
        assert "可用的引用来源" not in system_content


class TestChatEndpointRetrievalMeta:
    """测试 /chat 端点响应中的 retrieval_meta 字段"""

    @patch("routes.chat_routes.call_ai_api", new_callable=AsyncMock)
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_响应包含retrieval_meta字段(self, mock_vector_ctx, mock_ai_api):
        """chat 响应中应包含 retrieval_meta 字段"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_with_citations()
        mock_ai_api.return_value = _mock_ai_response("回答")

        client = TestClient(app)
        response = client.post("/api/chat", json=_make_chat_request())

        assert response.status_code == 200
        data = response.json()
        assert "retrieval_meta" in data

    @patch("routes.chat_routes.call_ai_api", new_callable=AsyncMock)
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_retrieval_meta包含citations(self, mock_vector_ctx, mock_ai_api):
        """retrieval_meta 中应包含 citations 字段"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_with_citations()
        mock_ai_api.return_value = _mock_ai_response("回答 [1]")

        client = TestClient(app)
        response = client.post("/api/chat", json=_make_chat_request())

        assert response.status_code == 200
        data = response.json()
        citations = data["retrieval_meta"].get("citations", [])
        assert len(citations) == 2
        assert citations[0]["ref"] == 1
        assert citations[0]["group_id"] == "group-0"
        assert citations[0]["page_range"] == [1, 3]

    @patch("routes.chat_routes.call_ai_api", new_callable=AsyncMock)
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_无向量搜索时retrieval_meta为空字典(self, mock_vector_ctx, mock_ai_api):
        """禁用向量搜索时，retrieval_meta 应为空字典"""
        app = _create_test_app()
        mock_ai_api.return_value = _mock_ai_response("回答")

        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json=_make_chat_request(enable_vector_search=False),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_meta"] == {}


class TestChatEndpointCombined:
    """测试生成提示词和引文追踪的组合场景"""

    @patch("routes.chat_routes.call_ai_api", new_callable=AsyncMock)
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_思维导图查询同时有citations(self, mock_vector_ctx, mock_ai_api):
        """思维导图查询且有 citations 时，系统提示词应同时包含两种注入"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_with_citations()
        mock_ai_api.return_value = _mock_ai_response("# 思维导图")

        client = TestClient(app)
        response = client.post("/api/chat", json=_make_chat_request(question="生成思维导图"))

        assert response.status_code == 200

        call_args = mock_ai_api.call_args
        messages = call_args[0][0]
        system_content = messages[0]["content"]

        # 应同时包含思维导图提示词和引文指示
        assert "思维导图" in system_content or "Markdown 层级结构" in system_content
        assert "可用的引用来源" in system_content


# ---- /chat/stream 端点测试 ----

class TestStreamEndpointGenerationPrompt:
    """测试 /chat/stream 端点的生成类查询提示词注入"""

    @patch("routes.chat_routes.call_ai_api_stream")
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_流式端点_思维导图查询注入提示词(self, mock_vector_ctx, mock_stream):
        """流式端点中，思维导图查询也应注入对应提示词"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_without_citations()

        # 捕获传递给流式 API 的 messages
        captured_messages = []

        async def fake_stream(messages, *args, **kwargs):
            captured_messages.extend(messages)
            yield {"content": "# 思维导图", "done": False}
            yield {"content": "", "done": True}

        mock_stream.side_effect = fake_stream

        client = TestClient(app)
        response = client.post("/api/chat/stream", json=_make_chat_request(question="生成思维导图"))

        assert response.status_code == 200

        # 验证传递给流式 API 的系统提示词包含思维导图内容
        assert len(captured_messages) > 0
        system_content = captured_messages[0]["content"]
        assert "思维导图" in system_content or "Markdown 层级结构" in system_content

    @patch("routes.chat_routes.call_ai_api_stream")
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_流式端点_流程图查询注入mermaid提示词(self, mock_vector_ctx, mock_stream):
        """流式端点中，流程图查询也应注入 Mermaid 提示词"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_without_citations()

        captured_messages = []

        async def fake_stream(messages, *args, **kwargs):
            captured_messages.extend(messages)
            yield {"content": "```mermaid\ngraph TD\n```", "done": False}
            yield {"content": "", "done": True}

        mock_stream.side_effect = fake_stream

        client = TestClient(app)
        response = client.post("/api/chat/stream", json=_make_chat_request(question="生成流程图"))

        assert response.status_code == 200

        system_content = captured_messages[0]["content"]
        assert "mermaid" in system_content.lower() or "Mermaid" in system_content


class TestStreamEndpointCitationPrompt:
    """测试 /chat/stream 端点的引文指示提示词注入"""

    @patch("routes.chat_routes.call_ai_api_stream")
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_流式端点_有citations时注入引文指示(self, mock_vector_ctx, mock_stream):
        """流式端点中，有 citations 时也应注入引文指示提示词"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_with_citations()

        captured_messages = []

        async def fake_stream(messages, *args, **kwargs):
            captured_messages.extend(messages)
            yield {"content": "回答 [1]", "done": False}
            yield {"content": "", "done": True}

        mock_stream.side_effect = fake_stream

        client = TestClient(app)
        response = client.post("/api/chat/stream", json=_make_chat_request(question="这篇文章讲了什么？"))

        assert response.status_code == 200

        system_content = captured_messages[0]["content"]
        assert "[1]" in system_content
        assert "可用的引用来源" in system_content


class TestStreamEndpointRetrievalMeta:
    """测试 /chat/stream 端点响应中的 retrieval_meta"""

    @patch("routes.chat_routes.call_ai_api_stream")
    @patch("routes.chat_routes.vector_context", new_callable=AsyncMock)
    def test_流式端点_最后chunk包含retrieval_meta(self, mock_vector_ctx, mock_stream):
        """流式端点的最后一个 chunk 应包含 retrieval_meta"""
        app = _create_test_app()
        mock_vector_ctx.return_value = _mock_vector_context_with_citations()

        async def fake_stream(messages, *args, **kwargs):
            yield {"content": "回答内容", "done": False}
            yield {"content": "", "done": True}

        mock_stream.side_effect = fake_stream

        client = TestClient(app)
        response = client.post("/api/chat/stream", json=_make_chat_request())

        assert response.status_code == 200

        # 解析 SSE 事件流，找到 done=True 的 chunk
        lines = response.text.strip().split("\n")
        done_chunk = None
        for line in lines:
            if line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
                if data.get("done"):
                    done_chunk = data
                    break

        assert done_chunk is not None, "应有 done=True 的 chunk"
        assert "retrieval_meta" in done_chunk
        assert "citations" in done_chunk["retrieval_meta"]
        assert len(done_chunk["retrieval_meta"]["citations"]) == 2
