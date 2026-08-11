"""渐进式外部论文元数据补全。

本模块只补充来源身份、开放获取状态和撤稿信号。所有 provider 失败均
fail-open；返回的期刊、被引量和撤稿信号不参与文档事实真伪判断。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

import httpx

from services.paper_metadata_provider_registry import (
    ProviderRegistry,
    ProviderSpec,
    build_provider_cache_identity,
    default_provider_names,
    fields_satisfied,
    missing_fields,
)

HYDRATION_VERSION = "paper-metadata-hydration-v1"
_S2_FIELDS = "title,authors,year,venue,externalIds,url,publicationTypes,citationCount,openAccessPdf"
_DEFAULT_REQUIRED_FIELDS = ("title", "authors", "year")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _year_from_parts(value: Any) -> int | None:
    try:
        return int(value[0][0])
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _crossref_record(payload: dict[str, Any], *, query_title: str = "") -> dict[str, Any]:
    message = payload.get("message") if isinstance(payload, dict) else None
    if isinstance(message, dict) and isinstance(message.get("items"), list):
        candidates = [item for item in message["items"] if isinstance(item, dict)]
        if not candidates:
            return {}
        query_norm = re.sub(r"\W+", "", query_title.casefold())
        message = max(
            candidates[:3],
            key=lambda item: int(
                re.sub(r"\W+", "", _text((item.get("title") or [""])[0]).casefold())
                == query_norm
            ),
        )
    if not isinstance(message, dict):
        return {}
    authors = []
    for author in message.get("author") or []:
        if not isinstance(author, dict):
            continue
        name = _text(" ".join(filter(None, (author.get("given"), author.get("family")))), 120)
        if name:
            authors.append(name)
    relation = message.get("relation") if isinstance(message.get("relation"), dict) else {}
    updates = message.get("update-to") if isinstance(message.get("update-to"), list) else []
    return {
        "title": _text((message.get("title") or [""])[0], 300),
        "authors": authors[:30],
        "year": _year_from_parts((message.get("published-print") or message.get("published-online") or {}).get("date-parts")),
        "venue": _text((message.get("container-title") or [""])[0], 200),
        "doi": _text(message.get("DOI"), 300).lower(),
        "url": _text(message.get("URL"), 1200),
        "citation_count": int(message.get("is-referenced-by-count") or 0),
        "relation": relation,
        "updates": updates,
    }


def _s2_record(payload: dict[str, Any], *, query_title: str = "") -> dict[str, Any]:
    if isinstance(payload.get("data"), list):
        candidates = [item for item in payload["data"] if isinstance(item, dict)]
        if not candidates:
            return {}
        query_norm = re.sub(r"\W+", "", query_title.casefold())
        payload = max(
            candidates[:3],
            key=lambda item: int(re.sub(r"\W+", "", _text(item.get("title")).casefold()) == query_norm),
        )
    if not isinstance(payload, dict):
        return {}
    external_ids = payload.get("externalIds") if isinstance(payload.get("externalIds"), dict) else {}
    authors = [
        _text(item.get("name"), 120)
        for item in (payload.get("authors") or [])
        if isinstance(item, dict) and _text(item.get("name"), 120)
    ]
    oa_pdf = payload.get("openAccessPdf") if isinstance(payload.get("openAccessPdf"), dict) else {}
    return {
        "title": _text(payload.get("title"), 300),
        "authors": authors[:30],
        "year": int(payload.get("year")) if str(payload.get("year") or "").isdigit() else None,
        "venue": _text(payload.get("venue"), 200),
        "doi": _text(external_ids.get("DOI"), 300).lower(),
        "arxiv_id": _text(external_ids.get("ArXiv"), 120),
        "url": _text(payload.get("url"), 1200),
        "citation_count": int(payload.get("citationCount") or 0),
        "publication_types": [
            _text(value, 80) for value in (payload.get("publicationTypes") or []) if _text(value, 80)
        ],
        "open_access_url": _text(oa_pdf.get("url"), 1200),
    }


def _unpaywall_record(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best_oa_location") if isinstance(payload.get("best_oa_location"), dict) else {}
    return {
        "is_open_access": bool(payload.get("is_oa")),
        "oa_status": _text(payload.get("oa_status"), 80),
        "open_access_url": _text(best.get("url_for_pdf") or best.get("url"), 1200),
        "license": _text(best.get("license"), 120),
    }


def _openalex_record(payload: dict[str, Any], *, query_title: str = "") -> dict[str, Any]:
    """Normalize the small subset of OpenAlex fields used by ChatPDF."""
    if isinstance(payload.get("results"), list):
        candidates = [item for item in payload["results"] if isinstance(item, dict)]
        if not candidates:
            return {}
        query_norm = re.sub(r"\W+", "", query_title.casefold())
        payload = max(
            candidates[:3],
            key=lambda item: int(re.sub(r"\W+", "", _text(item.get("title")).casefold()) == query_norm),
        )
    if not isinstance(payload, dict):
        return {}
    ids = payload.get("ids") if isinstance(payload.get("ids"), dict) else {}
    authorships = payload.get("authorships") if isinstance(payload.get("authorships"), list) else []
    authors = []
    for authorship in authorships:
        author = authorship.get("author") if isinstance(authorship, dict) else {}
        name = _text(author.get("display_name"), 120) if isinstance(author, dict) else ""
        if name:
            authors.append(name)
    primary_location = payload.get("primary_location") if isinstance(payload.get("primary_location"), dict) else {}
    source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
    oa = payload.get("best_oa_location") if isinstance(payload.get("best_oa_location"), dict) else {}
    doi = _text(ids.get("doi") or payload.get("doi"), 300).lower()
    if doi.startswith("https://doi.org/"):
        doi = doi.rsplit("/", 1)[-1]
    return {
        "title": _text(payload.get("title"), 300),
        "authors": authors[:30],
        "year": int(payload.get("publication_year")) if str(payload.get("publication_year") or "").isdigit() else None,
        "venue": _text(source.get("display_name"), 200),
        "doi": doi,
        "url": _text(payload.get("id"), 1200),
        "citation_count": int(payload.get("cited_by_count") or 0),
        "open_access_url": _text(oa.get("pdf_url") or oa.get("landing_page_url"), 1200),
    }


def _arxiv_record(payload: Any, *, query_title: str = "") -> dict[str, Any]:
    """Normalize Atom XML returned by export.arxiv.org."""
    if not isinstance(payload, str) or not payload.strip():
        return {}
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        return {}
    namespace = "{http://www.w3.org/2005/Atom}"
    entries = [item for item in root.findall(f"{namespace}entry")]
    if not entries:
        return {}
    query_norm = re.sub(r"\W+", "", query_title.casefold())

    def entry_title(entry) -> str:
        title = entry.findtext(f"{namespace}title")
        return _text(title, 300)

    entry = max(
        entries[:3],
        key=lambda item: int(re.sub(r"\W+", "", entry_title(item).casefold()) == query_norm),
    )
    title = entry_title(entry)
    authors = []
    for author in entry.findall(f"{namespace}author"):
        name = _text(author.findtext(f"{namespace}name"), 120)
        if name:
            authors.append(name)
    published = _text(entry.findtext(f"{namespace}published"), 40)
    year_match = re.match(r"(\d{4})", published)
    arxiv_url = _text(entry.findtext(f"{namespace}id"), 1200)
    arxiv_id = arxiv_url.rsplit("/", 1)[-1] if arxiv_url else ""
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    summary = _text(entry.findtext(f"{namespace}summary"), 1200)
    return {
        "title": title,
        "authors": authors[:30],
        "year": int(year_match.group(1)) if year_match else None,
        "arxiv_id": _text(arxiv_id, 120),
        "url": arxiv_url,
        "abstract_preview": summary,
    }


def _openreview_record(payload: dict[str, Any], *, query_title: str = "") -> dict[str, Any]:
    """Normalize the public OpenReview notes response when available."""
    notes = payload.get("notes") if isinstance(payload, dict) else None
    candidates = [item for item in notes if isinstance(item, dict)] if isinstance(notes, list) else []
    if candidates:
        query_norm = re.sub(r"\W+", "", query_title.casefold())
        payload = max(
            candidates[:3],
            key=lambda item: int(
                re.sub(r"\W+", "", _text((item.get("content") or {}).get("title")).casefold()) == query_norm
            ),
        )
    if not isinstance(payload, dict):
        return {}
    content = payload.get("content") if isinstance(payload.get("content"), dict) else payload
    title = _text(content.get("title"), 300)
    authors_raw = content.get("authors") or content.get("authorids") or []
    authors = [_text(value, 120) for value in authors_raw if _text(value, 120)] if isinstance(authors_raw, list) else []
    timestamp = payload.get("cdate") or payload.get("odate")
    try:
        year = datetime.fromtimestamp(float(timestamp) / 1000, tz=timezone.utc).year if timestamp else None
    except (TypeError, ValueError, OverflowError):
        year = None
    note_id = _text(payload.get("id") or payload.get("forum"), 200)
    return {
        "title": title,
        "authors": authors[:30],
        "year": year,
        "venue": _text(payload.get("invitation"), 200),
        "url": f"https://openreview.net/forum?id={quote(note_id, safe='')}" if note_id else "",
        "abstract_preview": _text(content.get("abstract"), 1200),
    }


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


async def _get_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    return str(response.text or "")


def _build_default_provider_registry() -> ProviderRegistry:
    """构造内置 provider；函数闭包只接收短元数据上下文。"""

    async def fetch_crossref(client, context):
        doi = str(context.get("doi") or "").strip()
        title = str(context.get("title") or "").strip()
        if doi:
            return await _get_json(client, f"https://api.crossref.org/works/{quote(doi, safe='')}")
        if title:
            return await _get_json(
                client,
                "https://api.crossref.org/works",
                params={"query.bibliographic": title, "rows": 3},
            )
        return {}

    async def fetch_semantic_scholar(client, context):
        doi = str(context.get("doi") or "").strip()
        arxiv_id = str(context.get("arxiv_id") or "").strip()
        title = str(context.get("title") or "").strip()
        s2_id = f"DOI:{doi}" if doi else (f"ARXIV:{arxiv_id}" if arxiv_id else "")
        semantic_key = str(context.get("semantic_scholar_api_key") or "").strip()
        headers = {"x-api-key": semantic_key} if semantic_key else None
        if s2_id:
            return await _get_json(
                client,
                f"https://api.semanticscholar.org/graph/v1/paper/{quote(s2_id, safe=':')}",
                params={"fields": _S2_FIELDS},
                headers=headers,
            )
        if title:
            return await _get_json(
                client,
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": title, "limit": 3, "fields": _S2_FIELDS},
                headers=headers,
            )
        return {}

    async def fetch_unpaywall(client, context):
        doi = str(context.get("resolved_doi") or context.get("doi") or "").strip()
        email = str(context.get("unpaywall_email") or "").strip()
        if not doi or not email:
            return {}
        return await _get_json(
            client,
            f"https://api.unpaywall.org/v2/{quote(doi, safe='')}",
            params={"email": email},
        )

    async def fetch_openalex(client, context):
        doi = str(context.get("doi") or "").strip()
        title = str(context.get("title") or "").strip()
        if doi:
            return await _get_json(client, f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}")
        if title:
            return await _get_json(client, "https://api.openalex.org/works", params={"search": title, "per-page": 3})
        return {}

    async def fetch_arxiv(client, context):
        arxiv_id = str(context.get("arxiv_id") or "").strip()
        title = str(context.get("title") or "").strip()
        params = {"id_list": arxiv_id, "max_results": 3} if arxiv_id else {
            "search_query": f'all:"{title}"',
            "max_results": 3,
        }
        if not arxiv_id and not title:
            return ""
        return await _get_text(client, "https://export.arxiv.org/api/query", params=params)

    async def fetch_openreview(client, context):
        title = str(context.get("title") or "").strip()
        if not title:
            return {}
        return await _get_json(
            client,
            "https://api2.openreview.net/notes",
            params={"content.title": title, "limit": 3},
        )

    return ProviderRegistry([
        ProviderSpec(
            name="crossref",
            capabilities=("identity", "citation_count", "retraction_signal"),
            enabled_by_default=True,
            priority=20,
            supported_fields=("title", "authors", "year", "venue", "doi", "url", "citation_count"),
            fetch=fetch_crossref,
            normalize=lambda payload, context: _crossref_record(payload, query_title=str(context.get("title") or "")),
            provenance="crossref",
            diagnostics={"privacy": "no_api_key", "fact_role": "source_hint_only"},
        ),
        ProviderSpec(
            name="semantic_scholar",
            capabilities=("identity", "citation_count", "open_access_hint"),
            enabled_by_default=True,
            priority=20,
            supported_fields=("title", "authors", "year", "venue", "doi", "arxiv_id", "url", "citation_count"),
            fetch=fetch_semantic_scholar,
            normalize=lambda payload, context: _s2_record(payload, query_title=str(context.get("title") or "")),
            provenance="semantic_scholar",
            diagnostics={"optional_key": True, "fact_role": "source_hint_only"},
        ),
        ProviderSpec(
            name="unpaywall",
            capabilities=("open_access_hint",),
            requires_key=True,
            enabled_by_default=False,
            priority=40,
            supported_fields=("is_open_access", "oa_status", "open_access_url", "license"),
            fetch=fetch_unpaywall,
            normalize=lambda payload, context: _unpaywall_record(payload),
            provenance="unpaywall",
            diagnostics={"credential": "email", "fact_role": "source_hint_only"},
        ),
        ProviderSpec(
            name="openalex",
            capabilities=("identity", "citation_count", "open_access_hint"),
            enabled_by_default=False,
            priority=60,
            supported_fields=("title", "authors", "year", "venue", "doi", "url", "citation_count", "open_access_url"),
            fetch=fetch_openalex,
            normalize=lambda payload, context: _openalex_record(payload, query_title=str(context.get("title") or "")),
            provenance="openalex",
            diagnostics={"optional": True, "fact_role": "source_hint_only"},
        ),
        ProviderSpec(
            name="arxiv",
            capabilities=("identity", "abstract_hint"),
            enabled_by_default=False,
            priority=70,
            supported_fields=("title", "authors", "year", "arxiv_id", "url", "abstract_preview"),
            fetch=fetch_arxiv,
            normalize=lambda payload, context: _arxiv_record(payload, query_title=str(context.get("title") or "")),
            provenance="arxiv",
            diagnostics={"optional": True, "fact_role": "source_hint_only"},
        ),
        ProviderSpec(
            name="openreview",
            capabilities=("identity", "abstract_hint"),
            enabled_by_default=False,
            priority=80,
            supported_fields=("title", "authors", "year", "venue", "url", "abstract_preview"),
            fetch=fetch_openreview,
            normalize=lambda payload, context: _openreview_record(payload, query_title=str(context.get("title") or "")),
            provenance="openreview",
            diagnostics={"optional": True, "fact_role": "source_hint_only"},
        ),
    ])


def hydration_cache_identity(
    *,
    parse_generation: str,
    document_source_hash: str,
    unpaywall_email: str = "",
    semantic_scholar_api_key: str = "",
    enable_openalex: bool = False,
    enable_arxiv: bool = False,
    enable_openreview: bool = False,
    required_fields: tuple[str, ...] = _DEFAULT_REQUIRED_FIELDS,
) -> dict[str, Any]:
    provider_names = default_provider_names(
        unpaywall_email=unpaywall_email,
        semantic_scholar_api_key=semantic_scholar_api_key,
        enable_openalex=enable_openalex,
        enable_arxiv=enable_arxiv,
        enable_openreview=enable_openreview,
    )
    return build_provider_cache_identity(
        parse_generation=parse_generation,
        source_hash=document_source_hash,
        provider_names=provider_names,
        credentials={
            "semantic_scholar": bool(str(semantic_scholar_api_key or "").strip()),
            "unpaywall": bool(str(unpaywall_email or "").strip()),
        },
        required_fields=required_fields,
        hydration_version=HYDRATION_VERSION,
    )


def _retraction_signal(provider_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence: list[dict[str, str]] = []
    crossref = provider_records.get("crossref") or {}
    relation = crossref.get("relation") if isinstance(crossref.get("relation"), dict) else {}
    for relation_type, values in relation.items():
        if "retract" in str(relation_type).lower():
            evidence.append({"provider": "crossref", "signal": str(relation_type)[:120]})
        for value in values if isinstance(values, list) else []:
            if "retract" in str(value).lower():
                evidence.append({"provider": "crossref", "signal": str(value)[:120]})
    for update in crossref.get("updates") or []:
        if isinstance(update, dict) and "retract" in str(update.get("type") or "").lower():
            evidence.append({"provider": "crossref", "signal": _text(update.get("type"), 120)})
    s2 = provider_records.get("semantic_scholar") or {}
    if any("retract" in value.lower() for value in (s2.get("publication_types") or [])):
        evidence.append({"provider": "semantic_scholar", "signal": "publication_type:retraction"})
    if _text(s2.get("title")).lower().startswith("retracted"):
        evidence.append({"provider": "semantic_scholar", "signal": "title_prefix:retracted"})
    return {
        "status": "retracted" if evidence else "no_signal",
        "evidence": evidence[:8],
        "checked_providers": sorted(provider_records),
        "notice": "仅表示外部来源信号；无信号不等于确认未撤稿。",
    }


def _merge_metadata(
    local: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    provider_order: list[str] | tuple[str, ...] = (),
) -> tuple[dict, dict]:
    merged = dict(local or {})
    provenance: dict[str, str] = {
        field: "local" for field, value in merged.items() if value not in (None, "", [], {})
    }
    order = list(provider_order) or ["crossref", "semantic_scholar", "unpaywall", "openalex", "arxiv", "openreview"]
    for provider in order:
        record = records.get(provider) or {}
        for field in (
            "title", "authors", "year", "venue", "doi", "arxiv_id", "url",
            "citation_count", "abstract_preview", "is_open_access", "oa_status", "open_access_url", "license",
        ):
            value = record.get(field)
            if value in (None, "", [], {}):
                continue
            if merged.get(field) in (None, "", [], {}):
                merged[field] = value
                provenance[field] = provider
    return merged, provenance


async def hydrate_paper_metadata(
    local_metadata: dict[str, Any],
    *,
    parse_generation: str = "",
    document_source_hash: str = "",
    unpaywall_email: str = "",
    semantic_scholar_api_key: str = "",
    timeout_seconds: float = 8.0,
    client: httpx.AsyncClient | None = None,
    enable_openalex: bool = False,
    enable_arxiv: bool = False,
    enable_openreview: bool = False,
    required_fields: tuple[str, ...] = _DEFAULT_REQUIRED_FIELDS,
    enabled_provider_names: tuple[str, ...] | None = None,
    registry: ProviderRegistry | None = None,
) -> dict[str, Any]:
    """按 provider 分层补全元数据，并在字段满足时跳过低优先级来源。

    ``enabled_provider_names`` 和 ``registry`` 主要用于测试/高级部署；普通路由
    只使用配置开关。所有失败均转换为 provider diagnostics，不阻塞本地结果。
    """
    local = dict(local_metadata or {})
    doi = _text(local.get("doi"), 300).lower()
    arxiv_id = _text(local.get("arxiv_id"), 120)
    title = _text(local.get("title"), 300)
    diagnostics: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    provider_order: list[str] = []
    active_registry = registry or _build_default_provider_registry()
    requested_names = tuple(enabled_provider_names or default_provider_names(
        unpaywall_email=unpaywall_email,
        semantic_scholar_api_key=semantic_scholar_api_key,
        enable_openalex=enable_openalex,
        enable_arxiv=enable_arxiv,
        enable_openreview=enable_openreview,
    ))
    credential_presence = {
        "semantic_scholar": bool(str(semantic_scholar_api_key or "").strip()),
        "unpaywall": bool(str(unpaywall_email or "").strip()),
    }
    cache_identity = build_provider_cache_identity(
        parse_generation=parse_generation,
        source_hash=document_source_hash,
        provider_names=requested_names,
        credentials=credential_presence,
        required_fields=required_fields,
        hydration_version=HYDRATION_VERSION,
    )
    owns_client = client is None
    active_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(max(1.0, float(timeout_seconds or 8.0))),
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": "ChatPDF/2 metadata-hydration"},
    )

    context: dict[str, Any] = {
        "doi": doi,
        "arxiv_id": arxiv_id,
        "title": title,
        "unpaywall_email": str(unpaywall_email or "").strip(),
        "semantic_scholar_api_key": str(semantic_scholar_api_key or "").strip(),
    }

    async def run_provider(spec: ProviderSpec) -> None:
        name = spec.name
        if spec.fetch is None or spec.normalize is None:
            diagnostics[name] = {
                "status": "misconfigured",
                "provider": spec.public_contract(),
            }
            return
        try:
            timeout = max(0.001, min(float(timeout_seconds or 8.0), float(spec.timeout)))
            raw = await asyncio.wait_for(spec.fetch(active_client, context), timeout=timeout)
            record = spec.normalize(raw, context)
            records[name] = record if isinstance(record, dict) else {}
            diagnostics[name] = {
                "status": "ok" if records[name] else "empty",
                "field_count": len([value for value in records[name].values() if value not in (None, "", [], {})]),
                "provider": spec.public_contract(),
            }
        except asyncio.TimeoutError:
            diagnostics[name] = {"status": "timeout", "provider": spec.public_contract()}
        except httpx.HTTPStatusError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            diagnostics[name] = {
                "status": "rate_limited" if status_code == 429 else "failed",
                "error": f"http_{status_code}" if status_code else "http_error",
                "provider": spec.public_contract(),
            }
        except Exception as exc:
            diagnostics[name] = {
                "status": "failed",
                "error": type(exc).__name__,
                "provider": spec.public_contract(),
            }

    async def run_specs(specs: list[ProviderSpec]) -> None:
        if specs:
            await asyncio.gather(*(run_provider(spec) for spec in specs))

    def mark_skipped(name: str, reason: str, *, missing: list[str] | None = None) -> None:
        spec = active_registry.get(name)
        if not spec:
            return
        diagnostics[name] = {
            "status": "skipped",
            "reason": reason,
            "missing_fields": list(missing or []),
            "provider": spec.public_contract(),
        }

    try:
        specs = active_registry.enabled_specs(requested_names, credentials=credential_presence)
        primary = [
            spec
            for spec in specs
            if spec.name in {"crossref", "semantic_scholar"} and (doi or arxiv_id or title)
        ]
        await run_specs(primary)
        resolved_doi = doi or _text((records.get("crossref") or {}).get("doi"), 300)
        context["resolved_doi"] = resolved_doi
        unpaywall = [spec for spec in specs if spec.name == "unpaywall"]
        if resolved_doi and str(unpaywall_email or "").strip():
            await run_specs(unpaywall)
        elif "unpaywall" in requested_names:
            mark_skipped("unpaywall", "missing_doi_or_email")

        provider_order = [
            spec.name
            for spec in sorted(specs, key=lambda item: (int(item.priority), item.name))
            if spec.name in records
        ]
        merged, _ = _merge_metadata(local, records, provider_order=provider_order)
        optional = [
            spec
            for spec in specs
            if spec.name in {"openalex", "arxiv", "openreview"}
        ]
        optional_groups: dict[int, list[ProviderSpec]] = {}
        for spec in optional:
            optional_groups.setdefault(int(spec.priority), []).append(spec)
        for priority in sorted(optional_groups):
            group = optional_groups[priority]
            if fields_satisfied(merged, required_fields):
                for spec in optional:
                    if spec.name not in diagnostics:
                        mark_skipped(spec.name, "required_fields_satisfied")
                break
            await run_specs(group)
            provider_order = [
                spec.name
                for spec in sorted(specs, key=lambda item: (int(item.priority), item.name))
                if spec.name in records
            ]
            merged, _ = _merge_metadata(local, records, provider_order=provider_order)
    finally:
        if owns_client:
            await active_client.aclose()

    provider_order = [
        spec.name
        for spec in sorted(active_registry.enabled_specs(requested_names, credentials=credential_presence), key=lambda item: (int(item.priority), item.name))
        if spec.name in records
    ]
    merged, provenance = _merge_metadata(local, records, provider_order=provider_order)
    successful = [name for name, detail in diagnostics.items() if detail.get("status") == "ok"]
    return {
        "version": HYDRATION_VERSION,
        "provider_registry_version": cache_identity.get("registry_version"),
        "cache_identity": cache_identity,
        "provider_names": list(requested_names),
        "status": "completed" if successful else "unavailable",
        "hydrated_at": _now(),
        "parse_generation": str(parse_generation or ""),
        "document_source_hash": str(document_source_hash or ""),
        "metadata": merged,
        "field_provenance": provenance,
        "providers": diagnostics,
        "required_fields": list(required_fields),
        "missing_fields": missing_fields(merged, required_fields),
        "retraction": _retraction_signal(records),
        "notice": "外部元数据仅用于来源提示，不用于判断文档陈述真假。",
    }
