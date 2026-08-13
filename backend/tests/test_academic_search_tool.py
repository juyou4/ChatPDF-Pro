"""academic_search 工具的注册、限次与执行行为测试。"""

import asyncio

from services import retrieval_tools
from services.retrieval_tools import DocContext
from services.retrieval_tool_schemas import TOOL_SCHEMAS, get_tool_spec


def _make_ctx(web_enabled: bool = True) -> DocContext:
    return DocContext(
        doc_id="doc-1",
        full_text="全文",
        chunks=["全文"],
        pages=[{"page": 1, "content": "全文"}],
        web_search_executor=(lambda query: {"sources": []}) if web_enabled else None,
    )


def test_academic_search_schema_registered():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert "academic_search" in names
    spec = get_tool_spec("academic_search")
    assert spec.get("source_family") == "academic"
    assert spec.get("cost_class") == "remote_web"
    assert spec.get("planner_default") is False


def test_academic_search_follows_web_authorization():
    ctx = _make_ctx(web_enabled=False)
    assert ctx.academic_search_available() is False
    allowed, reason = ctx.claim_academic_search_slot()
    assert allowed is False
    assert reason == "academic_search_not_enabled"


def test_academic_search_slot_limit():
    ctx = _make_ctx()
    assert ctx.academic_search_available() is True
    assert ctx.claim_academic_search_slot() == (True, "")
    assert ctx.claim_academic_search_slot() == (True, "")
    allowed, reason = ctx.claim_academic_search_slot()
    assert allowed is False
    assert reason == "academic_search_limit_reached"


def test_exec_academic_search_renders_and_registers(monkeypatch):
    ctx = _make_ctx()

    async def fake_discovery(query, **kwargs):
        assert query == "test method"
        return {
            "query": query,
            "candidates": [
                {
                    "candidate_id": "external:semantic_scholar:abc",
                    "metadata": {
                        "title": "A Great Paper",
                        "authors": ["Alice", "Bob"],
                        "year": 2024,
                        "venue": "NeurIPS",
                        "doi": "10.1000/xyz",
                        "arxiv_id": "2401.00001",
                        "abstract_preview": "We propose something new.",
                        "external_url": "https://www.semanticscholar.org/paper/abc",
                        "discovery_provider": "semantic_scholar",
                    },
                }
            ],
            "providers": {"semantic_scholar": {"status": "ok", "candidate_count": 1}},
        }

    monkeypatch.setattr(retrieval_tools, "discover_subscription_papers", fake_discovery)

    result = asyncio.run(
        retrieval_tools.execute_async_tool("academic_search", {"query": "test method"}, ctx)
    )
    assert not result.get("error")
    assert result["result_count"] == 1
    rendered = result["results"][0]
    assert "A Great Paper" in rendered
    assert "学术元数据" in rendered
    assert result["web_search_sources"][0]["url"] == "https://www.semanticscholar.org/paper/abc"
    assert result["chunk_meta"][0]["source"] == "academic_search"

    registry = result["web_search_source_registry"]
    source_id = registry[0]["source_id"]
    assert source_id.startswith("academic:")
    # 注册成功后 read_web_source 才能拿到授权。
    source, _cache_key, _cached, _from_cache, reason = ctx.claim_web_source_read(source_id, 0, 6000)
    assert source is not None
    assert reason == ""


def test_exec_academic_search_failure_suggests_web_search(monkeypatch):
    ctx = _make_ctx()

    async def failing_discovery(query, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(retrieval_tools, "discover_subscription_papers", failing_discovery)
    result = asyncio.run(
        retrieval_tools.execute_async_tool("academic_search", {"query": "test"}, ctx)
    )
    assert result["error_code"] == "academic_search_failed"
    assert result["suggested_next_tool"] == "web_search"


def test_exec_academic_search_empty_candidates_suggests_web_search(monkeypatch):
    ctx = _make_ctx()

    async def empty_discovery(query, **kwargs):
        return {"query": query, "candidates": [], "providers": {}}

    monkeypatch.setattr(retrieval_tools, "discover_subscription_papers", empty_discovery)
    result = asyncio.run(
        retrieval_tools.execute_async_tool("academic_search", {"query": "obscure topic"}, ctx)
    )
    assert result["result_count"] == 0
    assert result["suggested_next_tool"] == "web_search"


def test_execute_tool_sync_rejects_academic_search():
    ctx = _make_ctx()
    result = retrieval_tools.execute_tool("academic_search", {"query": "test"}, ctx)
    assert result["error_code"] == "academic_search_requires_async_executor"
