"""联网搜索服务回退链路测试"""
import os
import sys

import pytest

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.web_search_service import SearchManager


@pytest.mark.asyncio
async def test_auto_provider_fallbacks_to_bing_when_ddg_empty(monkeypatch):
    """自动模式下 DDG 空结果时应回退到 Bing"""

    async def mock_ddg(query, max_results=5):
        return []

    async def mock_bing(query, max_results=5):
        return [{"title": "Bing 命中", "url": "https://example.com", "snippet": "ok"}]

    monkeypatch.setattr(SearchManager, "_ddg_search", staticmethod(mock_ddg))
    monkeypatch.setattr(SearchManager, "_bing_search", staticmethod(mock_bing))

    result = await SearchManager.search("OpenAI", provider="auto", max_results=3)
    assert len(result) == 1
    assert result[0]["title"] == "Bing 命中"


@pytest.mark.asyncio
async def test_key_provider_without_key_fallbacks_to_auto(monkeypatch):
    """需要 API Key 的 provider 缺 key 时应回退自动搜索"""

    async def mock_auto(query, max_results=5):
        return [{"title": "Auto 命中", "url": "https://example.com/auto", "snippet": ""}]

    monkeypatch.setattr(SearchManager, "_auto_search", staticmethod(mock_auto))

    result = await SearchManager.search(
        "OpenAI",
        provider="tavily",
        api_key=None,
        max_results=4,
    )
    assert len(result) == 1
    assert result[0]["url"] == "https://example.com/auto"


@pytest.mark.asyncio
async def test_provider_alias_bing_rss_supported(monkeypatch):
    """provider 别名 bing_rss（大小写不敏感）应正确路由到 Bing"""

    async def mock_bing(query, max_results=5):
        return [{"title": "Bing RSS", "url": "https://example.com/rss", "snippet": ""}]

    monkeypatch.setattr(SearchManager, "_bing_search", staticmethod(mock_bing))

    result = await SearchManager.search("OpenAI", provider="BING_RSS")
    assert len(result) == 1
    assert result[0]["title"] == "Bing RSS"


@pytest.mark.asyncio
async def test_ddg_failure_fallbacks_to_auto_chain(monkeypatch):
    """provider=duckduckgo 发生异常时应触发自动回退"""

    async def mock_ddg(query, max_results=5):
        raise RuntimeError("ddg down")

    async def mock_auto(query, max_results=5):
        return [{"title": "Auto Fallback", "url": "https://example.com/fallback", "snippet": ""}]

    monkeypatch.setattr(SearchManager, "_ddg_search", staticmethod(mock_ddg))
    monkeypatch.setattr(SearchManager, "_auto_search", staticmethod(mock_auto))

    result = await SearchManager.search("OpenAI", provider="duckduckgo")
    assert len(result) == 1
    assert result[0]["title"] == "Auto Fallback"
