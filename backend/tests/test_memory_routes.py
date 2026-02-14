"""测试 memory_routes 记忆 API 路由

验证需求 6.1-6.8：
- GET /api/memory/profile 返回用户画像
- GET /api/memory/sessions/{doc_id} 返回文档会话记忆
- GET /api/memory/status 返回记忆系统状态
- POST /api/memory/entries 添加记忆条目
- PUT /api/memory/entries/{entry_id} 编辑记忆条目
- DELETE /api/memory/entries/{entry_id} 删除记忆条目
- DELETE /api/memory/all 清空所有记忆
- 无效 entry_id 返回 404
"""
import sys
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from routes.memory_routes import router, MemoryEntryResponse
import routes.memory_routes as memory_routes_module
from services.memory_store import MemoryEntry


@pytest.fixture
def mock_service():
    """创建 mock 的 MemoryService"""
    svc = MagicMock()
    # 默认 store 也是 mock
    svc.store = MagicMock()
    return svc


@pytest.fixture
def client(mock_service):
    """创建带有 mock service 注入的 TestClient"""
    app = FastAPI()
    app.include_router(router)
    # 注入 mock service
    original = memory_routes_module.memory_service
    memory_routes_module.memory_service = mock_service
    yield TestClient(app)
    # 恢复原始值
    memory_routes_module.memory_service = original


# ==================== GET /api/memory/profile ====================

class TestGetProfile:
    """测试获取用户画像接口"""

    def test_returns_profile(self, client, mock_service):
        """正常返回用户画像数据"""
        mock_service.get_profile.return_value = {
            "focus_areas": ["机器学习", "NLP"],
            "keyword_frequencies": {"机器学习": 5},
            "entries": [],
            "updated_at": "2024-01-15T10:00:00Z",
        }
        resp = client.get("/api/memory/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["focus_areas"] == ["机器学习", "NLP"]
        mock_service.get_profile.assert_called_once()


# ==================== GET /api/memory/sessions/{doc_id} ====================

class TestGetSession:
    """测试获取文档会话记忆接口"""

    def test_returns_session(self, client, mock_service):
        """正常返回文档会话记忆"""
        mock_service.get_session.return_value = {
            "doc_id": "doc123",
            "qa_summaries": [],
            "important_memories": [],
            "last_accessed": "2024-01-15T11:00:00Z",
        }
        resp = client.get("/api/memory/sessions/doc123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "doc123"
        mock_service.get_session.assert_called_once_with("doc123")


# ==================== GET /api/memory/status ====================

class TestGetStatus:
    """测试获取记忆系统状态接口"""

    def test_returns_status(self, client, mock_service):
        """正常返回记忆系统状态"""
        mock_service.get_status.return_value = {
            "enabled": True,
            "total_entries": 10,
            "index_size": 8,
            "profile_focus_areas": ["transformer"],
        }
        resp = client.get("/api/memory/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["total_entries"] == 10
        assert data["index_size"] == 8
        assert data["profile_focus_areas"] == ["transformer"]


# ==================== POST /api/memory/entries ====================

class TestAddEntry:
    """测试添加记忆条目接口"""

    def test_add_manual_entry(self, client, mock_service):
        """添加手动记忆条目"""
        entry = MemoryEntry(
            id="uuid-1",
            content="测试记忆内容",
            source_type="manual",
            created_at="2024-01-15T10:00:00Z",
            doc_id=None,
            importance=1.0,
        )
        mock_service.add_entry.return_value = entry

        resp = client.post("/api/memory/entries", json={
            "content": "测试记忆内容",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "uuid-1"
        assert data["content"] == "测试记忆内容"
        assert data["source_type"] == "manual"
        assert data["importance"] == 1.0
        mock_service.add_entry.assert_called_once_with(
            content="测试记忆内容",
            source_type="manual",
            doc_id=None,
        )

    def test_add_liked_entry_with_doc_id(self, client, mock_service):
        """添加点赞记忆条目，带 doc_id"""
        entry = MemoryEntry(
            id="uuid-2",
            content="点赞内容",
            source_type="liked",
            created_at="2024-01-15T10:00:00Z",
            doc_id="doc456",
            importance=1.0,
        )
        mock_service.add_entry.return_value = entry

        resp = client.post("/api/memory/entries", json={
            "content": "点赞内容",
            "source_type": "liked",
            "doc_id": "doc456",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_type"] == "liked"
        assert data["doc_id"] == "doc456"


# ==================== PUT /api/memory/entries/{entry_id} ====================

class TestUpdateEntry:
    """测试编辑记忆条目接口"""

    def test_update_existing_entry(self, client, mock_service):
        """成功更新已存在的记忆条目"""
        mock_service.update_entry.return_value = True
        updated_entry = MemoryEntry(
            id="uuid-1",
            content="更新后的内容",
            source_type="manual",
            created_at="2024-01-15T10:00:00Z",
            doc_id=None,
            importance=1.0,
        )
        mock_service.store.get_all_entries.return_value = [updated_entry]

        resp = client.put("/api/memory/entries/uuid-1", json={
            "content": "更新后的内容",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "uuid-1"
        assert data["content"] == "更新后的内容"

    def test_update_nonexistent_entry_returns_404(self, client, mock_service):
        """更新不存在的条目返回 404"""
        mock_service.update_entry.return_value = False

        entry_id = str(uuid.uuid4())
        resp = client.put(f"/api/memory/entries/{entry_id}", json={
            "content": "新内容",
        })
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]


# ==================== DELETE /api/memory/entries/{entry_id} ====================

class TestDeleteEntry:
    """测试删除记忆条目接口"""

    def test_delete_existing_entry(self, client, mock_service):
        """成功删除已存在的记忆条目"""
        mock_service.delete_entry.return_value = True

        resp = client.delete("/api/memory/entries/uuid-1")
        assert resp.status_code == 200
        assert "已删除" in resp.json()["message"]
        mock_service.delete_entry.assert_called_once_with("uuid-1")

    def test_delete_nonexistent_entry_returns_404(self, client, mock_service):
        """删除不存在的条目返回 404"""
        mock_service.delete_entry.return_value = False

        entry_id = str(uuid.uuid4())
        resp = client.delete(f"/api/memory/entries/{entry_id}")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]


# ==================== DELETE /api/memory/all ====================

class TestClearAll:
    """测试清空所有记忆接口"""

    def test_clear_all(self, client, mock_service):
        """成功清空所有记忆"""
        resp = client.delete("/api/memory/all")
        assert resp.status_code == 200
        assert "已清空" in resp.json()["message"]
        mock_service.clear_all.assert_called_once()


# ==================== 服务未初始化 ====================

class TestServiceNotInitialized:
    """测试 memory_service 未注入时的行为"""

    def test_returns_500_when_service_is_none(self):
        """memory_service 为 None 时返回 500"""
        app = FastAPI()
        app.include_router(router)
        original = memory_routes_module.memory_service
        memory_routes_module.memory_service = None
        try:
            c = TestClient(app)
            resp = c.get("/api/memory/profile")
            assert resp.status_code == 500
            assert "未初始化" in resp.json()["detail"]
        finally:
            memory_routes_module.memory_service = original
