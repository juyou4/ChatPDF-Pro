"""
预设问题 API 路由测试

测试 GET /api/presets 端点的功能。
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from routes.preset_routes import router
from services.preset_service import PRESET_QUESTIONS


@pytest.fixture
def client():
    """创建测试客户端"""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestPresetsEndpoint:
    """预设问题端点测试"""

    def test_获取预设问题列表成功(self, client):
        """GET /api/presets 应返回 200 和预设问题列表"""
        response = client.get("/api/presets")
        assert response.status_code == 200
        data = response.json()
        assert "presets" in data

    def test_返回数据与服务层一致(self, client):
        """返回的预设问题列表应与 PRESET_QUESTIONS 完全一致"""
        response = client.get("/api/presets")
        data = response.json()
        assert data["presets"] == PRESET_QUESTIONS

    def test_返回列表包含5个预设问题(self, client):
        """返回的预设问题列表应包含 5 个问题"""
        response = client.get("/api/presets")
        data = response.json()
        assert len(data["presets"]) == 5

    def test_每个预设问题包含必要字段(self, client):
        """每个预设问题应包含 id、label 和 query 字段"""
        response = client.get("/api/presets")
        data = response.json()
        for preset in data["presets"]:
            assert "id" in preset
            assert "label" in preset
            assert "query" in preset

    def test_预设问题包含总结本文(self, client):
        """预设问题列表应包含"总结本文"选项"""
        response = client.get("/api/presets")
        data = response.json()
        labels = [p["label"] for p in data["presets"]]
        assert "总结本文" in labels

    def test_预设问题包含生成思维导图(self, client):
        """预设问题列表应包含"生成思维导图"选项"""
        response = client.get("/api/presets")
        data = response.json()
        ids = [p["id"] for p in data["presets"]]
        assert "mindmap" in ids

    def test_预设问题包含生成流程图(self, client):
        """预设问题列表应包含"生成流程图"选项"""
        response = client.get("/api/presets")
        data = response.json()
        ids = [p["id"] for p in data["presets"]]
        assert "flowchart" in ids
