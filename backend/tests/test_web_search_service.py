"""联网搜索服务回退链路测试"""
import os
import sys

import pytest

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.web_search_service import SearchManager

# 结果在返回前会经过相关性过滤（_postprocess_results）。这些用例考的是 provider
# 回退链路的路由，不是过滤本身，所以 mock 结果必须与查询真实相关，否则会被正当地
# 滤成空列表，让测试失去对路由的判别力。
QUERY = "OpenAI"


def _relevant(title: str, url: str) -> dict:
    return {
        "title": f"{QUERY} {title}",
        "url": url,
        "snippet": f"{QUERY} 相关的检索结果摘要，用于通过相关性过滤。",
    }


@pytest.mark.asyncio
async def test_auto_provider_fallbacks_to_bing_when_ddg_empty(monkeypatch):
    """自动模式下 DDG 空结果时应回退到 Bing"""

    async def mock_ddg(query, max_results=5):
        return []

    async def mock_bing(query, max_results=5):
        return [_relevant("Bing 命中", "https://example.com")]

    monkeypatch.setattr(SearchManager, "_ddg_search", staticmethod(mock_ddg))
    monkeypatch.setattr(SearchManager, "_bing_search", staticmethod(mock_bing))

    result = await SearchManager.search(QUERY, provider="auto", max_results=3)
    assert len(result) == 1
    assert "Bing 命中" in result[0]["title"]


@pytest.mark.asyncio
async def test_key_provider_without_key_fallbacks_to_auto(monkeypatch):
    """需要 API Key 的 provider 缺 key 时应回退自动搜索"""

    async def mock_auto(query, max_results=5):
        return [_relevant("Auto 命中", "https://example.com/auto")]

    monkeypatch.setattr(SearchManager, "_auto_search", staticmethod(mock_auto))

    result = await SearchManager.search(
        QUERY,
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
        return [_relevant("Bing RSS", "https://example.com/rss")]

    monkeypatch.setattr(SearchManager, "_bing_search", staticmethod(mock_bing))

    result = await SearchManager.search(QUERY, provider="BING_RSS")
    assert len(result) == 1
    assert "Bing RSS" in result[0]["title"]


@pytest.mark.asyncio
async def test_ddg_failure_fallbacks_to_auto_chain(monkeypatch):
    """provider=duckduckgo 发生异常时应触发自动回退"""

    async def mock_ddg(query, max_results=5):
        raise RuntimeError("ddg down")

    async def mock_auto(query, max_results=5):
        return [_relevant("Auto Fallback", "https://example.com/fallback")]

    monkeypatch.setattr(SearchManager, "_ddg_search", staticmethod(mock_ddg))
    monkeypatch.setattr(SearchManager, "_auto_search", staticmethod(mock_auto))

    result = await SearchManager.search(QUERY, provider="duckduckgo")
    assert len(result) == 1
    assert "Auto Fallback" in result[0]["title"]
