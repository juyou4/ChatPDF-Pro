"""
高级搜索 API 路由测试

测试 /api/search/regex 和 /api/search/boolean 端点的功能。
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from routes.search_routes import router


@pytest.fixture
def app_with_docs():
    """创建带有模拟文档存储的测试应用"""
    app = FastAPI()
    app.include_router(router)

    # 注入模拟的文档存储
    router.documents_store = {
        "test-doc-1": {
            "data": {
                "full_text": (
                    "深度学习是机器学习的一个分支。"
                    "CNN 用于图像识别，RNN 用于序列建模。"
                    "Transformer 架构在 NLP 领域取得了突破性进展。"
                    "BERT 和 GPT 是基于 Transformer 的预训练模型。"
                ),
                "pages": [],
            }
        },
        "empty-doc": {
            "data": {
                "full_text": "",
                "pages": [],
            }
        },
    }

    return app


@pytest.fixture
def client(app_with_docs):
    """创建测试客户端"""
    return TestClient(app_with_docs)


# ==================== 正则搜索端点测试 ====================


class TestRegexSearchEndpoint:
    """正则表达式搜索端点测试"""

    def test_regex_search_basic(self, client):
        """测试基本正则搜索功能"""
        response = client.post(
            "/api/search/regex",
            json={"doc_id": "test-doc-1", "pattern": "CNN"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] > 0
        # 验证匹配文本包含 CNN
        assert data["results"][0]["match_text"] == "CNN"

    def test_regex_search_with_pattern(self, client):
        """测试使用正则模式搜索"""
        response = client.post(
            "/api/search/regex",
            json={"doc_id": "test-doc-1", "pattern": r"[A-Z]{3,}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
        # 所有匹配应为 3 个以上大写字母
        for result in data["results"]:
            assert len(result["match_text"]) >= 3

    def test_regex_search_invalid_pattern(self, client):
        """测试无效正则表达式返回 400"""
        response = client.post(
            "/api/search/regex",
            json={"doc_id": "test-doc-1", "pattern": "[invalid("},
        )
        assert response.status_code == 400
        assert "正则表达式语法错误" in response.json()["detail"]

    def test_regex_search_doc_not_found(self, client):
        """测试文档不存在返回 404"""
        response = client.post(
            "/api/search/regex",
            json={"doc_id": "nonexistent", "pattern": "test"},
        )
        assert response.status_code == 404

    def test_regex_search_empty_doc(self, client):
        """测试空文档返回空结果"""
        response = client.post(
            "/api/search/regex",
            json={"doc_id": "empty-doc", "pattern": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total"] == 0

    def test_regex_search_with_limit(self, client):
        """测试 limit 参数限制结果数量"""
        response = client.post(
            "/api/search/regex",
            json={"doc_id": "test-doc-1", "pattern": r"[。]", "limit": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] <= 2

    def test_regex_search_with_context_chars(self, client):
        """测试 context_chars 参数控制上下文长度"""
        response = client.post(
            "/api/search/regex",
            json={
                "doc_id": "test-doc-1",
                "pattern": "CNN",
                "context_chars": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
        # 上下文片段应包含匹配文本
        assert "CNN" in data["results"][0]["context_snippet"]

    def test_regex_search_no_match(self, client):
        """测试无匹配结果"""
        response = client.post(
            "/api/search/regex",
            json={"doc_id": "test-doc-1", "pattern": "ZZZZNOTEXIST"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total"] == 0


# ==================== 布尔搜索端点测试 ====================


class TestBooleanSearchEndpoint:
    """布尔逻辑搜索端点测试"""

    def test_boolean_search_basic(self, client):
        """测试基本布尔搜索功能"""
        response = client.post(
            "/api/search/boolean",
            json={"doc_id": "test-doc-1", "query": "深度学习"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] > 0

    def test_boolean_search_and(self, client):
        """测试 AND 操作符"""
        response = client.post(
            "/api/search/boolean",
            json={"doc_id": "test-doc-1", "query": "CNN AND 图像"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0

    def test_boolean_search_or(self, client):
        """测试 OR 操作符"""
        response = client.post(
            "/api/search/boolean",
            json={"doc_id": "test-doc-1", "query": "CNN OR BERT"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0

    def test_boolean_search_not(self, client):
        """测试 NOT 操作符"""
        response = client.post(
            "/api/search/boolean",
            json={"doc_id": "test-doc-1", "query": "Transformer NOT 图像"},
        )
        assert response.status_code == 200
        data = response.json()
        # 结果中不应包含"图像"附近的 Transformer
        for result in data["results"]:
            assert "图像" not in result["match_text"]

    def test_boolean_search_results_sorted_by_score(self, client):
        """测试结果按分数降序排列"""
        response = client.post(
            "/api/search/boolean",
            json={"doc_id": "test-doc-1", "query": "CNN OR 深度学习"},
        )
        assert response.status_code == 200
        data = response.json()
        if len(data["results"]) > 1:
            scores = [r["score"] for r in data["results"]]
            assert scores == sorted(scores, reverse=True)

    def test_boolean_search_doc_not_found(self, client):
        """测试文档不存在返回 404"""
        response = client.post(
            "/api/search/boolean",
            json={"doc_id": "nonexistent", "query": "test"},
        )
        assert response.status_code == 404

    def test_boolean_search_empty_doc(self, client):
        """测试空文档返回空结果"""
        response = client.post(
            "/api/search/boolean",
            json={"doc_id": "empty-doc", "query": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total"] == 0

    def test_boolean_search_with_limit(self, client):
        """测试 limit 参数"""
        response = client.post(
            "/api/search/boolean",
            json={"doc_id": "test-doc-1", "query": "深度学习", "limit": 1},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] <= 1

    def test_boolean_search_no_match(self, client):
        """测试无匹配结果"""
        response = client.post(
            "/api/search/boolean",
            json={"doc_id": "test-doc-1", "query": "ZZZZNOTEXIST"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total"] == 0
