"""
多引擎联网搜索服务（工厂模式）

支持的搜索引擎：
- DuckDuckGo（默认，免费，无需 API Key）
- Tavily（AI 原生搜索，需要 API Key，免费 1000 次/月）
- Serper（Google 镜像，需要 API Key，新号 2500 次）
- Brave Search（隐私优先，独立索引，需要 API Key）
- Exa（AI 原生语义搜索，需要 API Key）
- SerpAPI（多引擎 SERP，需要 API Key）
- Google Custom Search（Google 官方 CSE，需要 API Key + CX ID）
- Firecrawl（AI 搜索 + 内容提取，需要 API Key）

所有引擎返回统一结构：[{title, url, snippet}]
搜索失败时静默降级，返回空列表，不影响正常对话。
"""

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class SearchManager:
    """多引擎搜索管理器"""

    # 需要 API Key 的引擎列表及其对应方法名
    _PROVIDERS_REQUIRING_KEY = {
        "tavily": "_tavily_search",
        "serper": "_serper_search",
        "brave": "_brave_search",
        "exa": "_exa_search",
        "serpapi": "_serpapi_search",
        "google_cse": "_google_cse_search",
        "firecrawl": "_firecrawl_search",
    }

    @staticmethod
    async def search(
        query: str,
        provider: str = "duckduckgo",
        api_key: Optional[str] = None,
        max_results: int = 5,
    ) -> list[dict]:
        """统一搜索接口，根据 provider 调度对应实现

        Args:
            query: 搜索关键词
            provider: 搜索引擎名称
            api_key: API Key（部分引擎必需，格式说明见各方法）
            max_results: 最大返回结果数

        Returns:
            统一结构列表：[{title: str, url: str, snippet: str}]
        """
        if not query or not query.strip():
            return []

        try:
            method_name = SearchManager._PROVIDERS_REQUIRING_KEY.get(provider)
            if method_name:
                if not api_key:
                    logger.warning(f"{provider} 搜索需要 API Key，回退到 DuckDuckGo")
                    return await SearchManager._ddg_search(query, max_results)
                method = getattr(SearchManager, method_name)
                return await method(query, api_key, max_results)
            else:
                return await SearchManager._ddg_search(query, max_results)
        except Exception as e:
            logger.error(f"搜索失败 (provider={provider}): {e}")
            return []

    @staticmethod
    async def _tavily_search(
        query: str, api_key: str, max_results: int = 5
    ) -> list[dict]:
        """Tavily AI 原生搜索

        API 文档: https://docs.tavily.com/docs/rest-api/api-reference
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    }
                )
            logger.info(f"Tavily 搜索完成: query='{query}', 结果数={len(results)}")
            return results

    @staticmethod
    async def _serper_search(
        query: str, api_key: str, max_results: int = 5
    ) -> list[dict]:
        """Serper.dev Google 镜像搜索

        API 文档: https://serper.dev/docs
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": max_results},
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("organic", [])[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    }
                )
            logger.info(f"Serper 搜索完成: query='{query}', 结果数={len(results)}")
            return results

    @staticmethod
    async def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
        """DuckDuckGo 免费搜索（无需 API Key）

        使用 duckduckgo_search 库，同步调用通过 asyncio.to_thread 包装。
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error(
                "duckduckgo_search 未安装，请运行: pip install duckduckgo_search"
            )
            return []

        def _sync_search():
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
                results = []
                for item in raw:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("href", ""),
                            "snippet": item.get("body", ""),
                        }
                    )
                return results

        results = await asyncio.to_thread(_sync_search)
        logger.info(f"DuckDuckGo 搜索完成: query='{query}', 结果数={len(results)}")
        return results


    @staticmethod
    async def _brave_search(
        query: str, api_key: str, max_results: int = 5
    ) -> list[dict]:
        """Brave Search 隐私优先搜索

        API 文档: https://api.search.brave.com/app/documentation/web-search
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("web", {}).get("results", [])[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("description", ""),
                    }
                )
            logger.info(f"Brave 搜索完成: query='{query}', 结果数={len(results)}")
            return results

    @staticmethod
    async def _exa_search(
        query: str, api_key: str, max_results: int = 5
    ) -> list[dict]:
        """Exa AI 原生语义搜索

        API 文档: https://docs.exa.ai/reference/search
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                json={
                    "query": query,
                    "num_results": max_results,
                    "type": "neural",
                    "use_autoprompt": True,
                    "highlights": True,
                },
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                snippet = item.get("text", "")
                if not snippet:
                    highlights = item.get("highlights", [])
                    snippet = highlights[0] if highlights else ""
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": snippet,
                    }
                )
            logger.info(f"Exa 搜索完成: query='{query}', 结果数={len(results)}")
            return results

    @staticmethod
    async def _serpapi_search(
        query: str, api_key: str, max_results: int = 5
    ) -> list[dict]:
        """SerpAPI 多引擎 SERP 搜索

        API 文档: https://serpapi.com/search-api
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://serpapi.com/search.json",
                params={
                    "q": query,
                    "num": max_results,
                    "api_key": api_key,
                    "engine": "google",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("organic_results", [])[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    }
                )
            logger.info(f"SerpAPI 搜索完成: query='{query}', 结果数={len(results)}")
            return results

    @staticmethod
    async def _google_cse_search(
        query: str, api_key: str, max_results: int = 5
    ) -> list[dict]:
        """Google Custom Search Engine

        api_key 格式: "API_KEY:CX_ID"（用冒号分隔）
        API 文档: https://developers.google.com/custom-search/v1/overview
        """
        parts = api_key.split(":", 1)
        if len(parts) != 2:
            logger.warning("Google CSE 需要 'API_KEY:CX_ID' 格式的密钥")
            return []
        goog_key, cx_id = parts
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": goog_key,
                    "cx": cx_id,
                    "q": query,
                    "num": min(max_results, 10),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("items", [])[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    }
                )
            logger.info(f"Google CSE 搜索完成: query='{query}', 结果数={len(results)}")
            return results

    @staticmethod
    async def _firecrawl_search(
        query: str, api_key: str, max_results: int = 5
    ) -> list[dict]:
        """Firecrawl AI 搜索 + 内容提取

        API 文档: https://docs.firecrawl.dev/features/search
        """
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.firecrawl.dev/v1/search",
                json={
                    "query": query,
                    "limit": max_results,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("data", [])[:max_results]:
                snippet = item.get("description", "") or item.get("markdown", "")[:200]
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": snippet,
                    }
                )
            logger.info(f"Firecrawl 搜索完成: query='{query}', 结果数={len(results)}")
            return results


def format_search_results(results: list[dict]) -> str:
    """将搜索结果格式化为编号列表，用于注入 system prompt

    Args:
        results: SearchManager.search() 返回的结果列表

    Returns:
        格式化的字符串，如：
        [1] 标题 - URL
        摘要内容...

        [2] ...
    """
    if not results:
        return ""

    parts = []
    for i, item in enumerate(results, 1):
        title = item.get("title", "未知标题")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        entry = f"[{i}] {title}"
        if url:
            entry += f" - {url}"
        if snippet:
            entry += f"\n{snippet}"
        parts.append(entry)

    return "\n\n".join(parts)
