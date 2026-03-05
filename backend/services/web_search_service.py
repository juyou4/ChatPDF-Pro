"""
多引擎联网搜索服务（工厂模式）

支持的搜索引擎：
- Auto（默认，自动回退链路：Bing RSS -> DuckDuckGo）
- DuckDuckGo（免费，无需 API Key）
- Bing RSS（免费，无需 API Key）
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
import xml.etree.ElementTree as ET

import httpx

logger = logging.getLogger(__name__)


class SearchManager:
    """多引擎搜索管理器"""

    _logged_ddgs_missing = False

    # 无需 API Key 的引擎
    _PROVIDERS_NO_KEY = {
        "auto": "_auto_search",
        "duckduckgo": "_ddg_search",
        "bing": "_bing_search",
        "bing_rss": "_bing_search",
    }

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

        provider = (provider or "auto").strip().lower()
        max_results = max(1, int(max_results))

        try:
            key_method_name = SearchManager._PROVIDERS_REQUIRING_KEY.get(provider)
            if key_method_name:
                if not api_key:
                    logger.warning(f"{provider} 搜索需要 API Key，已回退到自动搜索")
                    return await SearchManager._auto_search(query, max_results)
                method = getattr(SearchManager, key_method_name)
                return await method(query, api_key, max_results)

            no_key_method_name = SearchManager._PROVIDERS_NO_KEY.get(provider)
            if no_key_method_name:
                method = getattr(SearchManager, no_key_method_name)
                results = await method(query, max_results=max_results)
                if provider == "duckduckgo" and not results:
                    logger.warning("DuckDuckGo 返回空结果，已回退到 Bing RSS")
                    return await SearchManager._bing_search(query, max_results=max_results)
                return results

            logger.warning(f"未知搜索 provider='{provider}'，已回退到自动搜索")
            return await SearchManager._auto_search(query, max_results)
        except Exception as e:
            logger.error(f"搜索失败 (provider={provider}): {e}")
            if provider != "auto":
                try:
                    return await SearchManager._auto_search(query, max_results)
                except Exception as fallback_error:
                    logger.error(f"自动回退搜索失败: {fallback_error}")
            return []

    @staticmethod
    async def _auto_search(query: str, max_results: int = 5) -> list[dict]:
        """自动搜索：先 Bing RSS，再回退到 DuckDuckGo"""
        providers = (
            ("bing", SearchManager._bing_search),
            ("duckduckgo", SearchManager._ddg_search),
        )
        for name, method in providers:
            try:
                results = await method(query, max_results=max_results)
            except Exception as e:
                logger.warning(f"自动搜索 {name} 失败: {e}")
                continue
            if results:
                logger.info(f"自动搜索命中 provider={name}, 结果数={len(results)}")
                return results
            logger.info(f"自动搜索 provider={name} 返回空结果，尝试下一个引擎")
        logger.warning("自动搜索未返回结果")
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

        使用 ddgs 包（duckduckgo_search 已弃用）。
        同步调用通过 asyncio.to_thread 包装。
        """
        try:
            from ddgs import DDGS
        except ImportError:
            if not SearchManager._logged_ddgs_missing:
                logger.warning("未安装 ddgs，DuckDuckGo 搜索不可用（请运行: pip install ddgs）")
                SearchManager._logged_ddgs_missing = True
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
    async def _bing_search(query: str, max_results: int = 5) -> list[dict]:
        """Bing RSS 免费搜索（无需 API Key）"""
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://www.bing.com/search",
                params={
                    "q": query,
                    "format": "rss",
                    "count": min(max_results, 50),
                },
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
                },
            )
            resp.raise_for_status()

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as e:
            logger.warning(f"Bing RSS 解析失败: {e}")
            return []

        results = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            url = (item.findtext("link") or "").strip()
            snippet = (item.findtext("description") or "").strip()
            if not title or not url:
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )
            if len(results) >= max_results:
                break

        logger.info(f"Bing RSS 搜索完成: query='{query}', 结果数={len(results)}")
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
