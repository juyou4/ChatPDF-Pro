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

import httpx

HYDRATION_VERSION = "paper-metadata-hydration-v1"
_S2_FIELDS = "title,authors,year,venue,externalIds,url,publicationTypes,citationCount,openAccessPdf"


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


def _merge_metadata(local: dict[str, Any], records: dict[str, dict[str, Any]]) -> tuple[dict, dict]:
    merged = dict(local or {})
    provenance: dict[str, str] = {
        field: "local" for field, value in merged.items() if value not in (None, "", [], {})
    }
    for provider in ("crossref", "semantic_scholar", "unpaywall"):
        record = records.get(provider) or {}
        for field in (
            "title", "authors", "year", "venue", "doi", "arxiv_id", "url",
            "citation_count", "is_open_access", "oa_status", "open_access_url", "license",
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
) -> dict[str, Any]:
    """Hydrate in parallel, preserve local fields, and return provider diagnostics."""
    local = dict(local_metadata or {})
    doi = _text(local.get("doi"), 300).lower()
    arxiv_id = _text(local.get("arxiv_id"), 120)
    title = _text(local.get("title"), 300)
    diagnostics: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    owns_client = client is None
    active_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(max(1.0, float(timeout_seconds or 8.0))),
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": "ChatPDF/2 metadata-hydration"},
    )

    async def run_provider(name: str, call) -> None:
        try:
            record = await call()
            records[name] = record if isinstance(record, dict) else {}
            diagnostics[name] = {
                "status": "ok" if records[name] else "empty",
                "field_count": len([value for value in records[name].values() if value not in (None, "", [], {})]),
            }
        except Exception as exc:
            diagnostics[name] = {"status": "failed", "error": type(exc).__name__}

    try:
        initial = []
        if doi:
            initial.append(("crossref", lambda: _get_json(active_client, f"https://api.crossref.org/works/{quote(doi, safe='')}")))
        elif title:
            initial.append(("crossref", lambda: _get_json(
                active_client,
                "https://api.crossref.org/works",
                params={"query.bibliographic": title, "rows": 3},
            )))
        s2_id = f"DOI:{doi}" if doi else (f"ARXIV:{arxiv_id}" if arxiv_id else "")
        s2_headers = {"x-api-key": semantic_scholar_api_key} if semantic_scholar_api_key else None
        if s2_id:
            initial.append(("semantic_scholar", lambda: _get_json(
                active_client,
                f"https://api.semanticscholar.org/graph/v1/paper/{quote(s2_id, safe=':')}",
                params={"fields": _S2_FIELDS},
                headers=s2_headers,
            )))
        elif title:
            initial.append(("semantic_scholar", lambda: _get_json(
                active_client,
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": title, "limit": 3, "fields": _S2_FIELDS},
                headers=s2_headers,
            )))
        await asyncio.gather(*(run_provider(name, call) for name, call in initial))
        if "crossref" in records:
            records["crossref"] = _crossref_record(records["crossref"], query_title=title)
        if "semantic_scholar" in records:
            records["semantic_scholar"] = _s2_record(records["semantic_scholar"], query_title=title)

        resolved_doi = doi or _text((records.get("crossref") or {}).get("doi"), 300)
        if resolved_doi and unpaywall_email:
            await run_provider("unpaywall", lambda: _get_json(
                active_client,
                f"https://api.unpaywall.org/v2/{quote(resolved_doi, safe='')}",
                params={"email": unpaywall_email},
            ))
            if "unpaywall" in records:
                records["unpaywall"] = _unpaywall_record(records["unpaywall"])
        else:
            diagnostics["unpaywall"] = {
                "status": "skipped",
                "reason": "missing_doi_or_email",
            }
    finally:
        if owns_client:
            await active_client.aclose()

    merged, provenance = _merge_metadata(local, records)
    successful = [name for name, detail in diagnostics.items() if detail.get("status") == "ok"]
    return {
        "version": HYDRATION_VERSION,
        "status": "completed" if successful else "unavailable",
        "hydrated_at": _now(),
        "parse_generation": str(parse_generation or ""),
        "document_source_hash": str(document_source_hash or ""),
        "metadata": merged,
        "field_provenance": provenance,
        "providers": diagnostics,
        "retraction": _retraction_signal(records),
        "notice": "外部元数据仅用于来源提示，不用于判断文档陈述真假。",
    }
