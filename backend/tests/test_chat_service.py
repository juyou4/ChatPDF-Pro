import types

import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import services.chat_service as cs  # noqa: E402
from utils.middleware import RetryMiddleware  # noqa: E402


def test_call_ai_api_success(monkeypatch):
    """单次调用成功"""
    class Dummy:
        async def chat(self, messages, api_key, model, timeout=None, stream=False):
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(cs.ProviderFactory, "create", lambda provider, endpoint=None: Dummy())

    resp = cs.asyncio.get_event_loop().run_until_complete(
        cs.call_ai_api([{"role": "user", "content": "hi"}], "k", "m", "openai")
    )
    assert resp.get("choices", [{}])[0].get("message", {}).get("content") == "ok"


def test_call_ai_api_retry_then_success(monkeypatch):
    """首次失败、重试成功"""
    calls = {"n": 0}

    class Dummy:
        async def chat(self, messages, api_key, model, timeout=None, stream=False):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("fail once")
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(cs.ProviderFactory, "create", lambda provider, endpoint=None: Dummy())

    resp = cs.asyncio.get_event_loop().run_until_complete(
        cs.call_ai_api(
            [{"role": "user", "content": "hi"}],
            "k",
            "m",
            "openai",
            middlewares=[RetryMiddleware(retries=1, delay=0)],
        )
    )
    assert calls["n"] == 2
    assert resp["choices"][0]["message"]["content"] == "ok"


def test_call_ai_api_retry_exhausted(monkeypatch):
    """重试耗尽返回错误"""
    class Dummy:
        async def chat(self, messages, api_key, model, timeout=None, stream=False):
            raise RuntimeError("always fail")

    monkeypatch.setattr(cs.ProviderFactory, "create", lambda provider, endpoint=None: Dummy())

    resp = cs.asyncio.get_event_loop().run_until_complete(
        cs.call_ai_api(
            [{"role": "user", "content": "hi"}],
            "k",
            "m",
            "openai",
            middlewares=[RetryMiddleware(retries=1, delay=0)],
        )
    )
    assert "error" in resp
