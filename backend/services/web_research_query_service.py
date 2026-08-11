"""Construct safe, document-grounded queries for external web research.

The document is an untrusted input.  This module deliberately extracts only
small, public-looking identifiers from search results; it never forwards a
document paragraph, filename, URL query string, or document instruction to a
search provider.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import unquote, urlsplit


_MAX_EVIDENCE_CHARS = 24_000
_MAX_ANCHORS = 8
_MAX_ANCHOR_TOKENS = 5
_MAX_QUERY_LENGTH = 320

_URL_RE = re.compile(
    r"https?://[^\s<>\"'`()\[\]{}]+",
    re.IGNORECASE,
)
_BARE_DOMAIN_RE = re.compile(
    r"(?<![@A-Za-z0-9_-])((?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,})(/[A-Za-z0-9._~%/-]{1,180})?",
)
_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9][A-Za-z0-9_.-]{0,38})/([A-Za-z0-9][A-Za-z0-9_.-]{0,99})(?:[/?#\s]|$)",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\barXiv\s*:\s*(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)
_LABELED_VALUE_RE = re.compile(
    r"(?:project\s*(?:page|website)|repository|repo|github|code(?:base)?|source\s*code|"
    r"项目主页|项目网站|代码(?:仓库|地址)?|源码|仓库|实现)\s*[:：]\s*([^\n\r]{1,180})",
    re.IGNORECASE,
)
_REPOSITORY_HINT_RE = re.compile(
    r"(?:仓库|代码|源码|github|gitlab|repository|repo|source\s*code|implementation)",
    re.IGNORECASE,
)
_PROJECT_HINT_RE = re.compile(
    r"(?:项目主页|项目网站|官网|project\s*(?:page|website)|official\s*(?:site|page))",
    re.IGNORECASE,
)
_DATASET_HINT_RE = re.compile(r"(?:数据集|dataset|下载包|download)", re.IGNORECASE)
_PAPER_HINT_RE = re.compile(r"(?:论文|paper|doi|arxiv)", re.IGNORECASE)
_FILLER_RE = re.compile(
    r"(?:请|帮我|帮忙|麻烦|能否|可以|有没有|有的话|如果有|找一下|查一下|看一下|"
    r"请问|是否|给出|提供|吗|么|呢|please|could you|can you|if any|help me|find|look up)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,63}")
_CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,20}")
_BLOCKED_HOST_SUFFIXES = (
    ".local",
    ".localhost",
    ".internal",
    ".intranet",
    ".lan",
)
_BLOCKED_HOSTS = {
    "example.com",
    "example.org",
    "example.net",
    "attacker.example",
    "test",
}
_INJECTION_HINTS = {
    "ignore",
    "previous",
    "instruction",
    "instructions",
    "system",
    "assistant",
    "developer",
    "send",
    "upload",
    "secret",
    "token",
    "password",
    "pdf",
    "execute",
    "tool",
}
_GENERIC_PATH_NAMES = {
    "index",
    "home",
    "page",
    "project",
    "projects",
    "paper",
    "papers",
    "html",
    "docs",
    "doc",
}
_QUERY_STOPWORDS = {
    "repository",
    "repo",
    "github",
    "official",
    "project",
    "page",
    "website",
    "source",
    "code",
    "the",
    "this",
    "that",
}


def _compact(value: object, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _is_public_host(host: str) -> bool:
    normalized = str(host or "").strip().lower().rstrip(".")
    if (
        not normalized
        or normalized == "localhost"
        or normalized in _BLOCKED_HOSTS
        or normalized.endswith(_BLOCKED_HOST_SUFFIXES)
    ):
        return False
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return bool("." in normalized and " " not in normalized and len(normalized) <= 253)
    return not (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved)


def _safe_url(value: object) -> dict | None:
    raw = str(value or "").strip().rstrip(".,;:)]}")
    if not raw or len(raw) > 1200:
        return None
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or parsed.username or parsed.password:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if not _is_public_host(host):
        return None
    path = unquote(parsed.path or "").strip("/")
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) > 5 or any(len(segment) > 80 for segment in segments):
        return None
    if any(not re.fullmatch(r"[A-Za-z0-9._~%+-]+", segment) for segment in segments):
        return None
    # Query strings and fragments commonly contain tokens or credentials.  They
    # are intentionally dropped before an anchor is stored or rendered.
    return {"host": host, "path": "/".join(segments[:5])}


def _tokens(*values: object) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _compact(value, 180)
        for token in [*_TOKEN_RE.findall(text), *_CJK_TOKEN_RE.findall(text)]:
            normalized = token.strip("._-")
            if len(normalized) < 2 or normalized.lower() in _QUERY_STOPWORDS:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(normalized)
            if len(output) >= _MAX_ANCHOR_TOKENS:
                return output
    return output


def _anchor(kind: str, *, host: str = "", path: str = "", tokens: list[str] | None = None, value: str = "") -> dict:
    clean_tokens = _tokens(*(tokens or []))
    identity = "/".join(part for part in (host, path) if part)
    return {
        "kind": kind,
        "host": host,
        "path": path,
        "value": _compact(value or identity, 180),
        "tokens": clean_tokens,
    }


def _append_anchor(output: list[dict], seen: set[str], item: dict | None) -> None:
    if not isinstance(item, dict):
        return
    tokens = item.get("tokens") or []
    if not tokens:
        return
    key = "|".join(
        [str(item.get("kind") or ""), str(item.get("host") or ""), str(item.get("path") or ""), *tokens]
    ).casefold()
    if key in seen:
        return
    seen.add(key)
    output.append(item)


def _anchor_from_url(url: str, *, labeled: bool = False) -> dict | None:
    parsed = _safe_url(url)
    if not parsed:
        return None
    host = parsed["host"]
    path = parsed["path"]
    github_match = re.search(r"^github\.com/([^/]+)/([^/]+)", f"{host}/{path}", re.IGNORECASE)
    if github_match:
        owner, repo = github_match.groups()
        return _anchor("github", host="github.com", path=f"{owner}/{repo}", tokens=[owner, repo], value=f"{owner}/{repo}")
    segments = [segment for segment in path.split("/") if segment]
    if host.endswith(".github.io"):
        owner = host.split(".", 1)[0]
        useful = [segment for segment in segments if segment.casefold() not in _GENERIC_PATH_NAMES]
        return _anchor("github_pages", host=host, path="/".join(useful[:2]), tokens=[owner, *useful[:2]], value=f"{owner} {' '.join(useful[:2])}")
    useful = [segment for segment in segments if segment.casefold() not in _GENERIC_PATH_NAMES]
    if labeled or useful:
        return _anchor("public_page", host=host, path="/".join(useful[:3]), tokens=[host, *useful[:3]], value=f"{host} {' '.join(useful[:3])}")
    return _anchor("public_domain", host=host, tokens=[host], value=host)


def _labeled_identifier(value: str) -> list[str]:
    candidate = _tokens(value)
    if not candidate or len(candidate) > 3:
        return []
    if any(token.casefold() in _INJECTION_HINTS for token in candidate):
        return []
    return candidate


def _iter_evidence_texts(evidence: object):
    if isinstance(evidence, dict):
        for key in ("results", "candidate_meta", "chunk_meta", "detail"):
            value = evidence.get(key)
            if isinstance(value, list):
                yield from _iter_evidence_texts(value)
        for key in ("text", "chunk", "source_text", "display_text", "title", "summary"):
            if evidence.get(key):
                yield _compact(evidence.get(key), 2400)
        return
    if isinstance(evidence, (list, tuple, set)):
        used = 0
        for item in evidence:
            for text in _iter_evidence_texts(item):
                if used >= _MAX_EVIDENCE_CHARS:
                    return
                remaining = _MAX_EVIDENCE_CHARS - used
                clipped = text[:remaining]
                used += len(clipped)
                if clipped:
                    yield clipped
        return
    if evidence:
        yield _compact(evidence, 2400)


def extract_safe_web_anchors(evidence: object) -> list[dict]:
    """Extract public identifiers from already retrieved document evidence."""
    output: list[dict] = []
    seen: set[str] = set()
    texts = list(_iter_evidence_texts(evidence))
    bounded = "\n".join(texts)[:_MAX_EVIDENCE_CHARS]

    # Explicit GitHub links are strongest because they already identify a repo.
    for match in _GITHUB_RE.finditer(bounded):
        owner, repo = match.groups()
        _append_anchor(
            output,
            seen,
            _anchor("github", host="github.com", path=f"{owner}/{repo}", tokens=[owner, repo], value=f"{owner}/{repo}"),
        )
        if len(output) >= _MAX_ANCHORS:
            return output

    labeled_spans: list[tuple[int, int]] = []
    for match in _LABELED_VALUE_RE.finditer(bounded):
        value = match.group(1)
        labeled_spans.append((match.start(1), match.end(1)))
        url = _URL_RE.search(value)
        if url:
            _append_anchor(output, seen, _anchor_from_url(url.group(0), labeled=True))
            continue
        bare_url = _BARE_DOMAIN_RE.search(value)
        if bare_url and (bare_url.group(1).lower().endswith(".github.io") or bare_url.group(1).lower() == "github.com"):
            _append_anchor(
                output,
                seen,
                _anchor_from_url(
                    f"https://{bare_url.group(1)}{bare_url.group(2) or ''}",
                    labeled=True,
                ),
            )
            continue
        # A label may contain a project name without a URL.  Keep only compact
        # identifier-like tokens; prose and prompt-injection text are ignored.
        candidate = _labeled_identifier(value)
        if candidate:
            _append_anchor(output, seen, _anchor("labeled_identifier", tokens=candidate, value=" ".join(candidate)))

    for match in _URL_RE.finditer(bounded):
        _append_anchor(output, seen, _anchor_from_url(match.group(0)))
        if len(output) >= _MAX_ANCHORS:
            return output

    # Bare project domains are common in papers (for example, a GitHub Pages
    # link printed without the scheme).  Only use them when they are not part of
    # a sensitive query string and retain host/path, never surrounding prose.
    for match in _BARE_DOMAIN_RE.finditer(bounded):
        if any(start <= match.start() < end for start, end in labeled_spans):
            # Labeled values were already tokenized above, but a bare URL may
            # still be worth preserving as a structured public page anchor.
            pass
        parsed = _safe_url(f"https://{match.group(1)}{match.group(2) or ''}")
        if parsed:
            _append_anchor(output, seen, _anchor_from_url(f"https://{match.group(1)}{match.group(2) or ''}"))
        if len(output) >= _MAX_ANCHORS:
            return output

    for match in _DOI_RE.finditer(bounded):
        _append_anchor(output, seen, _anchor("doi", tokens=[match.group(0)], value=match.group(0)))
    for match in _ARXIV_RE.finditer(bounded):
        _append_anchor(output, seen, _anchor("arxiv", tokens=[match.group(1)], value=f"arXiv:{match.group(1)}"))
    return output[:_MAX_ANCHORS]


def _target_type(query: str) -> str:
    text = str(query or "")
    if _REPOSITORY_HINT_RE.search(text):
        return "repository"
    if _PROJECT_HINT_RE.search(text):
        return "project"
    if _DATASET_HINT_RE.search(text):
        return "dataset"
    if _PAPER_HINT_RE.search(text):
        return "paper"
    return "general"


def _safe_intent_text(query: str) -> str:
    text = _compact(query, 180)
    text = _FILLER_RE.sub(" ", text)
    text = re.sub(r"[\x00-\x1f<>\[\]{}|;`$]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_web_research_query(
    user_query: str,
    *,
    planner_query: str = "",
    document_evidence: object = None,
    anchors: list[dict] | None = None,
) -> dict:
    """Return a bounded outbound query and diagnostics.

    ``planner_query`` is used only to infer the target type.  It is never
    copied verbatim into the outbound query because it may contain untrusted
    text produced after reading a PDF.
    """
    base = _safe_intent_text(user_query)
    target = _target_type(f"{user_query} {planner_query}")
    if target == "repository":
        base = re.sub(r"[有吗呢么，,?？。.!！]+", " ", base)
        base = re.sub(r"\s+", " ", base).strip()
    safe_anchors = list(anchors or []) or extract_safe_web_anchors(document_evidence)
    tokens: list[str] = []
    for item in safe_anchors:
        if not isinstance(item, dict):
            continue
        tokens.extend(item.get("tokens") or [])
    tokens = _tokens(*tokens)

    if target == "repository" and tokens:
        query = "site:github.com " + " ".join(tokens[:4]) + " repository"
    elif target == "repository":
        query = f"{base} GitHub repository".strip()
    elif target == "project" and tokens:
        query = " ".join(tokens[:5]) + " official project page"
    elif target == "dataset" and tokens:
        query = " ".join(tokens[:5]) + " official dataset"
    elif target == "paper" and tokens:
        query = " ".join(tokens[:5]) + " paper"
    elif tokens and not base:
        query = " ".join(tokens[:5])
    elif tokens and _target_type(user_query) in {"repository", "project", "dataset", "paper"}:
        query = f"{base} {' '.join(tokens[:4])}".strip()
    else:
        query = base or _safe_intent_text(planner_query)

    query = re.sub(r"\s+", " ", query).strip()[:_MAX_QUERY_LENGTH]
    return {
        "query": query,
        "target": target,
        "anchors": safe_anchors[:_MAX_ANCHORS],
        "anchor_count": len(safe_anchors),
        "used_document_anchors": bool(safe_anchors),
    }
