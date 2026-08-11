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
import hashlib
import logging
import math
import re
from typing import Optional
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


def _query_log_fingerprint(query: str) -> tuple[int, str]:
    """Return diagnostic metadata without retaining a user's search terms."""
    normalized = str(query or "")
    return len(normalized), hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _log_search_complete(provider: str, query: str, result_count: int) -> None:
    query_length, query_hash = _query_log_fingerprint(query)
    logger.info(
        "%s 搜索完成: query_chars=%s query_hash=%s 结果数=%s",
        provider,
        query_length,
        query_hash,
        result_count,
    )


def _safe_search_error(exc: BaseException) -> str:
    """Avoid serializing request URLs, query strings, or API keys into logs."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    suffix = f" status={status_code}" if isinstance(status_code, int) else ""
    return f"{type(exc).__name__}{suffix}"


class SearchManager:
    """多引擎搜索管理器"""

    _logged_ddgs_missing = False
    _TECH_QUERY_HINTS = (
        "论文", "方法", "模型", "算法", "攻击", "防御", "鲁棒", "公式", "实验", "asr",
        "adversarial", "model", "method", "algorithm", "benchmark", "loss", "dataset",
        "github", "repository", "repo", "source code", "代码仓库",
    )
    _NOISY_DOMAIN_HINTS = (
        "instagram.com", "ameblo.jp", "weibo.com", "x.com", "twitter.com", "facebook.com",
        "hanyuguoxue.com", "zdic.net", "dict.youdao.com", "wenku.baidu.com", "zhidao.baidu.com",
    )
    _MIN_RELEVANCE_SCORE = 0.08
    _TITLE_ANCHOR_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "is", "of", "on", "or", "the", "to", "via", "with",
    }
    _REPOSITORY_PRIORITY_DOMAINS = {
        "github.com", "gitlab.com", "gitee.com", "codeberg.org",
    }

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
        blacklist: Optional[list[str]] = None,
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
            # Auto 模式不能只看上游是否返回了原始结果：某些网络环境会
            # 把 Bing RSS 重定向到门户首页，原始列表非空但相关性全为零。
            # 只有经过黑名单、相关性和仓库官方来源排序后仍有可用结果，
            # 才算该引擎命中，否则继续尝试下一个引擎。
            if provider == "auto":
                return await SearchManager._auto_search_relevant(
                    query,
                    max_results=max_results,
                    blacklist=blacklist,
                )

            key_method_name = SearchManager._PROVIDERS_REQUIRING_KEY.get(provider)
            if key_method_name:
                if not api_key:
                    logger.warning(f"{provider} 搜索需要 API Key，已回退到自动搜索")
                    raw_results = await SearchManager._auto_search(query, max_results)
                    return SearchManager._postprocess_results(query, raw_results, max_results, blacklist)
                method = getattr(SearchManager, key_method_name)
                raw_results = await method(query, api_key, max_results)
                return SearchManager._postprocess_results(query, raw_results, max_results, blacklist)

            no_key_method_name = SearchManager._PROVIDERS_NO_KEY.get(provider)
            if no_key_method_name:
                method = getattr(SearchManager, no_key_method_name)
                results = await method(query, max_results=max_results)
                if provider == "duckduckgo" and not results:
                    logger.warning("DuckDuckGo 返回空结果，已回退到 Bing RSS")
                    results = await SearchManager._bing_search(query, max_results=max_results)
                return SearchManager._postprocess_results(query, results, max_results, blacklist)

            logger.warning(f"未知搜索 provider='{provider}'，已回退到自动搜索")
            raw_results = await SearchManager._auto_search(query, max_results)
            return SearchManager._postprocess_results(query, raw_results, max_results, blacklist)
        except Exception as e:
            logger.error("搜索失败 (provider=%s): %s", provider, _safe_search_error(e))
            if provider != "auto":
                try:
                    raw_results = await SearchManager._auto_search(query, max_results)
                    return SearchManager._postprocess_results(query, raw_results, max_results, blacklist)
                except Exception as fallback_error:
                    logger.error(
                        "自动回退搜索失败: %s",
                        _safe_search_error(fallback_error),
                    )
            return []

    @staticmethod
    async def _auto_search(query: str, max_results: int = 5) -> list[dict]:
        """自动搜索：当前引擎必须产出相关结果，否则继续回退。"""
        providers = []
        if (
            SearchManager._quoted_query_anchors(query)
            and re.search(r"\b(?:venue|publication|journal|conference|doi|authors?)\b", query, re.IGNORECASE)
        ):
            providers.append(("academic_metadata", SearchManager._academic_metadata_search))
        providers.extend((
            ("bing", SearchManager._bing_search),
            ("duckduckgo", SearchManager._ddg_search),
        ))
        for name, method in providers:
            try:
                results = await method(query, max_results=max_results)
            except Exception as e:
                logger.warning("自动搜索 %s 失败: %s", name, _safe_search_error(e))
                continue
            if not results:
                logger.info(f"自动搜索 provider={name} 返回空结果，尝试下一个引擎")
                continue
            relevant_results = SearchManager._rerank_and_filter_results(
                query,
                SearchManager._deduplicate_results(results),
            )
            if relevant_results:
                logger.info(
                    "自动搜索命中 provider=%s, 原始结果数=%s, 相关结果数=%s",
                    name,
                    len(results),
                    len(relevant_results),
                )
                return relevant_results[:max_results]
            logger.info(
                "自动搜索 provider=%s 仅返回离题结果，尝试下一个引擎",
                name,
            )
        logger.warning("自动搜索未返回结果")
        return []

    @staticmethod
    async def _academic_metadata_search(query: str, max_results: int = 5) -> list[dict]:
        """Resolve a quoted paper title through keyless academic metadata providers."""
        anchors = SearchManager._quoted_query_anchors(query)
        if not anchors:
            return []
        title = anchors[0]
        try:
            openalex_sources = await SearchManager._openalex_metadata_sources(
                title,
                anchors,
                max_results=max_results,
            )
            if openalex_sources:
                return openalex_sources
        except Exception as exc:
            logger.info("OpenAlex 论文元数据搜索失败: %s", type(exc).__name__)

        try:
            from services.paper_metadata_hydration_service import hydrate_paper_metadata

            hydration = await hydrate_paper_metadata(
                {"title": title},
                timeout_seconds=6.0,
                required_fields=("title", "venue"),
                enabled_provider_names=("crossref",),
            )
        except Exception as exc:
            logger.info("Crossref 论文元数据搜索失败: %s", type(exc).__name__)
            return []

        metadata = hydration.get("metadata") if isinstance(hydration, dict) else {}
        if not isinstance(metadata, dict):
            return []
        resolved_title = str(metadata.get("title") or "").strip()
        if not resolved_title:
            return []
        venue = str(metadata.get("venue") or "").strip()
        year = str(metadata.get("year") or "").strip()
        doi = str(metadata.get("doi") or "").strip()
        url = str(metadata.get("url") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"
        provenance = hydration.get("field_provenance") if isinstance(hydration, dict) else {}
        venue_provider = str((provenance or {}).get("venue") or "academic metadata").strip()
        detail_parts = [f"Metadata provider: {venue_provider}."]
        if venue:
            detail_parts.append(f"Venue: {venue}.")
        if year:
            detail_parts.append(f"Publication year: {year}.")
        if doi:
            detail_parts.append(f"DOI: {doi}.")
        result = {
            "title": f"{resolved_title}{f' — {venue}' if venue else ''}",
            "url": url,
            "snippet": " ".join(detail_parts),
        }
        if not SearchManager._result_matches_quoted_anchor(result, anchors):
            return []
        return [result][:max_results]

    @staticmethod
    async def _openalex_metadata_sources(
        title: str,
        anchors: list[str],
        *,
        max_results: int,
    ) -> list[dict]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(6.0),
            follow_redirects=True,
            trust_env=False,
            headers={"User-Agent": "ChatPDF/3 academic-web-search"},
        ) as client:
            response = await client.get(
                "https://api.openalex.org/works",
                params={"search": title, "per-page": min(8, max(3, max_results))},
            )
            response.raise_for_status()
            payload = response.json()
        return SearchManager._openalex_payload_to_sources(
            payload,
            anchors,
            max_results=max_results,
        )

    @staticmethod
    def _openalex_payload_to_sources(
        payload: dict,
        anchors: list[str],
        *,
        max_results: int,
    ) -> list[dict]:
        records = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            return []
        sources: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            resolved_title = str(record.get("title") or "").strip()
            location = record.get("primary_location") if isinstance(record.get("primary_location"), dict) else {}
            venue_record = location.get("source") if isinstance(location.get("source"), dict) else {}
            venue = str(venue_record.get("display_name") or "").strip()
            year = str(record.get("publication_year") or "").strip()
            doi = str(record.get("doi") or "").strip()
            url = str(location.get("landing_page_url") or doi or record.get("id") or "").strip()
            work_type = str(record.get("type") or "").strip()
            if not resolved_title or not url:
                continue
            detail_parts = ["Metadata provider: OpenAlex."]
            if venue:
                detail_parts.append(f"Venue: {venue}.")
            if work_type:
                detail_parts.append(f"Work type: {work_type}.")
            if year:
                detail_parts.append(f"Publication year: {year}.")
            if doi:
                detail_parts.append(f"DOI: {doi.removeprefix('https://doi.org/')}.")
            source = {
                "title": f"{resolved_title}{f' - {venue}' if venue else ''}{f' ({year})' if year else ''}",
                "url": url,
                "snippet": " ".join(detail_parts),
            }
            if not SearchManager._result_matches_quoted_anchor(source, anchors):
                continue
            key = (url.casefold(), venue.casefold(), year)
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
            if len(sources) >= max_results:
                break
        return sources

    @staticmethod
    async def _auto_search_relevant(
        query: str,
        max_results: int = 5,
        blacklist: Optional[list[str]] = None,
    ) -> list[dict]:
        """自动搜索并以处理后的相关结果决定是否切换引擎。"""
        providers = []
        if (
            SearchManager._quoted_query_anchors(query)
            and re.search(
                r"\b(?:venue|publication|journal|conference|doi|authors?)\b",
                query,
                re.IGNORECASE,
            )
        ):
            providers.append(("academic_metadata", SearchManager._academic_metadata_search))
        providers.extend((
            ("bing", SearchManager._bing_search),
            ("duckduckgo", SearchManager._ddg_search),
        ))
        for name, method in providers:
            try:
                raw_results = await method(query, max_results=max_results)
            except Exception as exc:
                logger.warning("自动搜索 %s 失败: %s", name, _safe_search_error(exc))
                continue
            processed = SearchManager._postprocess_results(
                query,
                raw_results,
                max_results,
                blacklist,
            )
            if processed:
                logger.info(
                    "自动搜索命中 provider=%s, 相关结果数=%s",
                    name,
                    len(processed),
                )
                return processed
            if raw_results:
                logger.info(
                    "自动搜索 provider=%s 返回结果但相关性不足，尝试下一个引擎",
                    name,
                )
            else:
                logger.info("自动搜索 provider=%s 返回空结果，尝试下一个引擎", name)
        logger.warning("自动搜索未返回可用相关结果")
        return []

    @staticmethod
    def _postprocess_results(
        query: str,
        results: list[dict],
        max_results: int,
        blacklist: Optional[list[str]] = None,
    ) -> list[dict]:
        """对原始搜索结果做去重 + 黑名单过滤 + 相关性重排过滤。"""
        deduped = SearchManager._deduplicate_results(results)
        if blacklist:
            deduped = SearchManager._filter_by_blacklist(deduped, blacklist)
        ranked = SearchManager._rerank_and_filter_results(query, deduped)
        return ranked[:max_results]

    @staticmethod
    def _filter_by_blacklist(results: list[dict], blacklist: list[str]) -> list[dict]:
        """过滤黑名单域名，支持精确域名和子域名（例如 example.com 同时屏蔽 sub.example.com）"""
        if not blacklist:
            return results
        filtered = []
        for r in results:
            domain = SearchManager._domain(r.get("url", ""))
            if not any(domain == b or domain.endswith("." + b) for b in blacklist):
                filtered.append(r)
        removed = len(results) - len(filtered)
        if removed:
            logger.debug(f"黑名单过滤：移除 {removed} 条结果")
        return filtered

    @staticmethod
    def _deduplicate_results(results: list[dict]) -> list[dict]:
        if not results:
            return []
        deduped = []
        seen = set()
        for item in results:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            if not title and not url:
                continue
            key = (title.lower(), url.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"title": title, "url": url, "snippet": snippet})
        return deduped

    @staticmethod
    def _tokenize_for_relevance(text: str) -> set[str]:
        if not text:
            return set()
        lowered = text.lower()
        tokens = set(re.findall(r"[a-z0-9]{2,}", lowered))
        for seq in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
            # 保留整词，并补充 2-gram 以提升中文召回
            tokens.add(seq)
            for i in range(len(seq) - 1):
                tokens.add(seq[i:i + 2])
        return tokens

    @staticmethod
    def _has_kana(text: str) -> bool:
        return bool(re.search(r"[\u3040-\u30ff]", text or ""))

    @staticmethod
    def _has_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    @staticmethod
    def _domain(url: str) -> str:
        try:
            return (urlparse(url).netloc or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _is_technical_query(query: str) -> bool:
        q = (query or "").lower()
        return any(h in q for h in SearchManager._TECH_QUERY_HINTS)

    @staticmethod
    def _is_repository_query(query: str) -> bool:
        q = (query or "").lower()
        return any(
            hint in q
            for hint in (
                "github", "gitlab", "gitee", "repository", "repo", "source code",
                "代码仓库", "源码", "仓库", "代码地址",
            )
        )

    @staticmethod
    def _score_result(query: str, item: dict) -> float:
        q_tokens = SearchManager._tokenize_for_relevance(query)
        if not q_tokens:
            return 0.0

        title = item.get("title", "") or ""
        snippet = item.get("snippet", "") or ""
        url = item.get("url", "") or ""
        title_tokens = SearchManager._tokenize_for_relevance(title)
        body_tokens = SearchManager._tokenize_for_relevance(f"{title} {snippet} {url}")

        overlap_title = len(q_tokens & title_tokens) / max(1, len(q_tokens))
        overlap_body = len(q_tokens & body_tokens) / max(1, len(q_tokens))
        score = overlap_title * 0.65 + overlap_body * 0.35

        q_lower = (query or "").lower()
        corpus_lower = f"{title} {snippet}".lower()
        if q_lower and q_lower in corpus_lower:
            score += 0.2

        # 中文查询遇到大量假名结果时降权（典型日文娱乐站点误命中）
        if SearchManager._has_cjk(query) and SearchManager._has_kana(f"{title} {snippet}"):
            score *= 0.45

        # 技术问题下，对社交/娱乐噪音域名降权（精确后缀匹配避免误伤）
        domain = SearchManager._domain(url)
        if SearchManager._is_technical_query(query) and any(
            domain == h or domain.endswith("." + h) for h in SearchManager._NOISY_DOMAIN_HINTS
        ):
            score *= 0.5

        if SearchManager._is_repository_query(query):
            if domain in SearchManager._REPOSITORY_PRIORITY_DOMAINS or any(
                domain.endswith("." + h) for h in SearchManager._REPOSITORY_PRIORITY_DOMAINS
            ):
                score += 0.4
            elif domain.endswith(".github.io"):
                score += 0.25
            elif any(
                domain == h or domain.endswith("." + h)
                for h in SearchManager._NOISY_DOMAIN_HINTS
            ):
                score *= 0.08

        return score

    @staticmethod
    def _quoted_query_anchors(query: str) -> list[str]:
        return [
            anchor.strip()
            for anchor in re.findall(r'"([^"\r\n]{6,})"', str(query or ""))
            if anchor.strip()
        ]

    @staticmethod
    def _compact_anchor_text(text: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(text or "").lower())

    @staticmethod
    def _result_matches_quoted_anchor(item: dict, anchors: list[str]) -> bool:
        if not anchors:
            return True
        title = str(item.get("title") or "")
        snippet = str(item.get("snippet") or "")
        corpus_tokens = SearchManager._tokenize_for_relevance(f"{title} {snippet}")
        title_compact = SearchManager._compact_anchor_text(title)
        corpus_compact = SearchManager._compact_anchor_text(f"{title} {snippet}")

        for anchor in anchors:
            anchor_compact = SearchManager._compact_anchor_text(anchor)
            if (
                anchor_compact
                and corpus_compact
                and (
                    anchor_compact in corpus_compact
                    or title_compact in anchor_compact and len(title_compact) >= 24
                    or (
                        title_compact.startswith(anchor_compact[:24])
                        and len(title_compact) >= 24
                    )
                )
            ):
                return True

            anchor_tokens = {
                token
                for token in SearchManager._tokenize_for_relevance(anchor)
                if token not in SearchManager._TITLE_ANCHOR_STOPWORDS
            }
            if not anchor_tokens:
                continue
            overlap_count = len(anchor_tokens & corpus_tokens)
            # Quoted titles are identity anchors, not broad topical queries.
            # Requiring high coverage prevents a different paper in the same
            # topic (for example another diffusion detector) from being cited.
            required = len(anchor_tokens) if len(anchor_tokens) <= 3 else math.ceil(len(anchor_tokens) * 0.75)
            if overlap_count >= required:
                return True
        return False

    @staticmethod
    def _rerank_and_filter_results(query: str, results: list[dict]) -> list[dict]:
        if not results:
            return []

        quoted_anchors = SearchManager._quoted_query_anchors(query)
        if quoted_anchors:
            results = [
                item
                for item in results
                if SearchManager._result_matches_quoted_anchor(item, quoted_anchors)
            ]
            if not results:
                logger.debug("相关性重排：没有结果通过精确查询锚点，返回空")
                return []

        scored = []
        for item in results:
            score = SearchManager._score_result(query, item)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)

        max_score = scored[0][0] if scored else 0.0
        if max_score < SearchManager._MIN_RELEVANCE_SCORE:
            logger.debug(f"相关性重排：所有结果得分过低(max={max_score:.3f})，返回空")
            return []

        # A repository request must not degrade to a random dictionary/news
        # page merely because it shares one generic Chinese bigram.
        if SearchManager._is_repository_query(query) and max_score < 0.12:
            logger.debug("相关性重排：仓库查询没有达到最低相关性阈值(max=%.3f)", max_score)
            return []

        # 自适应阈值：保留与最佳结果接近的项，过滤明显离题项
        threshold = max(SearchManager._MIN_RELEVANCE_SCORE, max_score * 0.35)
        filtered = [item for score, item in scored if score >= threshold]
        logger.debug(
            f"相关性重排：共 {len(scored)} 条 → 过滤后 {len(filtered)} 条 "
            f"(max_score={max_score:.3f}, threshold={threshold:.3f})"
        )

        return filtered

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
            _log_search_complete("Tavily", query, len(results))
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
            _log_search_complete("Serper", query, len(results))
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
        _log_search_complete("DuckDuckGo", query, len(results))
        return results

    @staticmethod
    async def _bing_search(query: str, max_results: int = 5) -> list[dict]:
        """Bing RSS 免费搜索（无需 API Key）"""
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", query or ""))
        market = "zh-CN" if has_cjk else "en-US"
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://www.bing.com/search",
                params={
                    "q": query,
                    "format": "rss",
                    "count": min(max_results, 50),
                    "mkt": market,
                },
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
                },
            )
            resp.raise_for_status()

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as e:
            logger.warning("Bing RSS 解析失败: %s", _safe_search_error(e))
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

        _log_search_complete("Bing RSS", query, len(results))
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
            _log_search_complete("Brave", query, len(results))
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
                    "numResults": max_results,
                    "type": "neural",
                    "highlights": {"numSentences": 3, "highlightsPerUrl": 1},
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
                highlights = item.get("highlights", [])
                snippet = highlights[0] if highlights else item.get("text", "")
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": snippet,
                    }
                )
            _log_search_complete("Exa", query, len(results))
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
            _log_search_complete("SerpAPI", query, len(results))
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
            _log_search_complete("Google CSE", query, len(results))
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
            _log_search_complete("Firecrawl", query, len(results))
            return results


def format_search_results(results: list[dict]) -> str:
    """将搜索结果格式化为独立的网页引用编号，用于注入证据消息。

    Args:
        results: SearchManager.search() 返回的结果列表

    Returns:
        格式化的字符串，如：
        [W1] 标题 - URL
        摘要内容...

        [W2] ...
    """
    if not results:
        return ""

    _SNIPPET_MAX_LEN = 2400

    parts = []
    for i, item in enumerate(results, 1):
        title = item.get("title", "未知标题")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        if len(snippet) > _SNIPPET_MAX_LEN:
            snippet = snippet[:_SNIPPET_MAX_LEN] + "…"
        evidence_type = str(item.get("evidence_type") or "search_snippet")
        evidence_label = {
            "academic_metadata": "学术元数据",
            "provider_content": "搜索服务正文",
            "webpage_excerpt": "网页正文摘录",
        }.get(evidence_type, "搜索摘要")
        entry = f"[W{i}] {title}"
        if url:
            entry += f" - {url}"
        if snippet:
            entry += f"\n证据类型：{evidence_label}\n{snippet}"
        parts.append(entry)

    return "\n\n".join(parts)
