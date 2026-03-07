"""rerank API 服务回归测试"""

import os
import sys

import pytest

# 将 backend 目录加入导入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import services.rerank_api_service as rerank_api_service


class _FakeResponse:
    def raise_for_status(self):
        return None

    @staticmethod
    def json():
        return {"results": [{"index": 0, "relevance_score": 0.91}]}


def test_openai_like_rerank_requires_explicit_endpoint_for_unknown_provider(monkeypatch):
    """未知 provider 不应静默回退到 SiliconFlow 默认 endpoint。"""

    def _should_not_call(*args, **kwargs):
        raise AssertionError("httpx.post 不应被调用")

    monkeypatch.setattr(rerank_api_service.httpx, "post", _should_not_call)

    with pytest.raises(ValueError):
        rerank_api_service.openai_like_rerank(
            query="test",
            documents=["a", "b"],
            model="rerank-test",
            api_key="sk-test",
            provider="openai",
        )


def test_openai_like_rerank_accepts_explicit_endpoint_for_unknown_provider(monkeypatch):
    """未知 provider 显式提供 endpoint 时仍应正常请求。"""
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr(rerank_api_service.httpx, "post", _fake_post)

    scores = rerank_api_service.openai_like_rerank(
        query="test",
        documents=["a", "b"],
        model="rerank-test",
        api_key="sk-test",
        provider="openai",
        endpoint="https://example.com/v1/rerank",
    )

    assert captured["url"] == "https://example.com/v1/rerank"
    assert captured["payload"]["model"] == "rerank-test"
    assert scores == [(0, 0.91)]
