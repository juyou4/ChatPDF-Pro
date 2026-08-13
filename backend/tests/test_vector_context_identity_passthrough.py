"""``vector_context`` 与其兼容包装的签名契约回归。

HEAD 上 ``vector_context`` 无条件把 ``retrieval_identity`` 传给本模块的
``get_relevant_context`` 兼容包装，而包装签名缺少该参数：普通向量检索路径
每次调用都会 ``TypeError``，随后被 fail-open 吞成空上下文。这类签名漂移不会
被任何路由测试捕获（路由测试通常 mock 整个 ``vector_context``），必须在服务
层单独锁定。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import services.vector_service as vector_service  # noqa: E402


@pytest.mark.asyncio
async def test_vector_context_passes_retrieval_identity_to_core_impl(monkeypatch) -> None:
    captured: dict = {}

    def _fake_impl(doc_id, query, **kwargs):
        captured["doc_id"] = doc_id
        captured["kwargs"] = kwargs
        return "ctx", {"retrieval_mode": "test"}

    monkeypatch.setattr(vector_service, "_get_relevant_context_impl", _fake_impl)

    identity = {"parse_generation": "gen-1", "document_source_hash": "hash-1"}
    result = await vector_service.vector_context(
        doc_id="doc-identity",
        query="测试查询",
        vector_store_dir="",
        pages=[{"page": 1, "content": "正文"}],
        api_key="key",
        top_k=5,
        candidate_k=10,
        use_rerank=False,
        reranker_model=None,
        retrieval_identity=identity,
    )

    assert result["error"] is None
    assert result["context"] == "ctx"
    assert captured["doc_id"] == "doc-identity"
    assert captured["kwargs"]["retrieval_identity"] == identity


@pytest.mark.asyncio
async def test_vector_context_defaults_identity_to_none_for_legacy_callers(monkeypatch) -> None:
    captured: dict = {}

    def _fake_impl(doc_id, query, **kwargs):
        captured["kwargs"] = kwargs
        return "ctx", {}

    monkeypatch.setattr(vector_service, "_get_relevant_context_impl", _fake_impl)

    result = await vector_service.vector_context(
        doc_id="doc-legacy",
        query="q",
        vector_store_dir="",
        pages=[],
        api_key="key",
        top_k=5,
        candidate_k=10,
        use_rerank=False,
        reranker_model=None,
    )

    assert result["error"] is None
    assert captured["kwargs"]["retrieval_identity"] is None
