"""chat_routes 查询改写分流与超时回退测试"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.chat_routes as chat_routes


@pytest.mark.asyncio
async def test_overview_query_skips_llm_rewrite(monkeypatch):
    monkeypatch.setattr(chat_routes.settings, "enable_llm_query_rewrite", True)
    monkeypatch.setattr(chat_routes.settings, "query_rewrite_trigger_length", 150)

    async def _unexpected_llm_call(**kwargs):
        raise AssertionError("概览问题不应触发 LLM 查询改写")

    monkeypatch.setattr(chat_routes._query_rewriter, "rewrite_with_llm", _unexpected_llm_call)

    result = await chat_routes._maybe_rewrite_query(
        question="请总结本文的主要内容",
        chat_history=[{"role": "user", "content": "上一轮内容"}],
        selected_text=None,
        api_key="test-key",
        model="test-model",
        provider="openai",
        endpoint="https://example.com/v1/chat/completions",
    )

    assert result == "请总结本文的主要内容"


@pytest.mark.asyncio
async def test_ambiguous_query_can_still_use_llm_rewrite(monkeypatch):
    monkeypatch.setattr(chat_routes.settings, "enable_llm_query_rewrite", True)
    monkeypatch.setattr(chat_routes.settings, "query_rewrite_trigger_length", 150)

    async def _fake_llm_call(**kwargs):
        return "梯度下降方法的作用是什么"

    monkeypatch.setattr(chat_routes._query_rewriter, "rewrite_with_llm", _fake_llm_call)

    result = await chat_routes._maybe_rewrite_query(
        question="它的作用是什么",
        chat_history=[{"role": "assistant", "content": "我们在讨论梯度下降。"}],
        selected_text=None,
        api_key="test-key",
        model="test-model",
        provider="openai",
        endpoint="https://example.com/v1/chat/completions",
    )

    assert result == "梯度下降方法的作用是什么"


@pytest.mark.asyncio
async def test_query_rewrite_timeout_falls_back_to_regex(monkeypatch):
    monkeypatch.setattr(chat_routes.settings, "enable_llm_query_rewrite", True)
    monkeypatch.setattr(chat_routes.settings, "query_rewrite_trigger_length", 150)

    async def _slow_llm_call(**kwargs):
        await asyncio.sleep(5)
        return "不应返回"

    monkeypatch.setattr(chat_routes._query_rewriter, "rewrite_with_llm", _slow_llm_call)

    result = await chat_routes._maybe_rewrite_query(
        question="它的作用是什么",
        chat_history=[{"role": "assistant", "content": "我们在讨论激活函数。"}],
        selected_text=None,
        api_key="test-key",
        model="test-model",
        provider="openai",
        endpoint="https://example.com/v1/chat/completions",
    )

    assert result == "它的作用是什么"
