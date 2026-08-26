"""受控外部研究适配器。

第一阶段只读取已经由 ChatPDF 搜索结果登记的公开网页。Agent-Reach 的
安装器、Skill、MCP 和全局 CLI 不进入本项目；这里保留一个窄的、可替换的
适配器边界，后续 GitHub/YouTube 适配器可以复用同一份安全与证据协议。
"""

from __future__ import annotations

import hashlib
import base64
import ipaddress
import json
import logging
import os
import re
import socket
import xml.etree.ElementTree as ET
from html import unescape
from typing import Any, Protocol
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

import httpx

logger = logging.getLogger(__name__)

_READER_HOST = "r.jina.ai"
_DEFAULT_TIMEOUT_S = 15.0
_DEFAULT_MAX_BYTES = 1_500_000
_DEFAULT_MAX_CHARS = 12_000
_MAX_URL_LENGTH = 4096
_MAX_REDIRECTS = 0
_MAX_ADAPTER_RESPONSE_BYTES = 4_000_000
_GITHUB_HOSTS = {"github.com", "www.github.com", "raw.githubusercontent.com"}
_GITHUB_API_HOST = "api.github.com"
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
_YOUTUBE_TRANSCRIPT_HOST = "video.google.com"
_YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
_GITHUB_PATH_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_GITHUB_ISSUE_NUMBER_RE = re.compile(r"^[0-9]{1,12}$")
_BLOCKED_HOST_SUFFIXES = (
    ".local",
    ".internal",
    ".lan",
    ".localhost",
)
_BLOCKED_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.google.com",
    "169.254.169.254",
    "100.100.100.200",
}

# 后续适配器（GitHub 公共页面、YouTube 字幕等）只需实现同一窄接口；
# 它们不会获得任意 URL、Cookie 或写操作权限。
EXTERNAL_RESEARCH_ADAPTERS = (
    "jina_reader",
    "github_public",
    "youtube_transcript",
    "browser_render",
)


class ExternalResearchAdapter(Protocol):
    name: str

    async def read(
        self,
        url: str,
        *,
        max_chars: int = _DEFAULT_MAX_CHARS,
        start_char: int = 0,
    ) -> dict:
        """读取一个已通过来源授权的公开资源并返回统一结果。"""


class ExternalResearchError(ValueError):
    """可安全展示给调用方的外部研究错误。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code or "external_research_error")
        self.message = str(message or "外部来源读取失败")


def _is_blocked_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _resolve_public_host(hostname: str) -> None:
    host = str(hostname or "").strip().rstrip(".").casefold()
    if not host:
        raise ExternalResearchError("invalid_url", "网页地址缺少主机名")
    if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise ExternalResearchError("private_host_blocked", "不允许读取本地或内网地址")
    if _is_blocked_ip(host):
        raise ExternalResearchError("private_ip_blocked", "不允许读取本地或保留 IP 地址")
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror) as exc:
        raise ExternalResearchError("dns_resolution_failed", "网页地址无法解析") from exc
    if not addresses:
        raise ExternalResearchError("dns_resolution_failed", "网页地址无法解析")
    for item in addresses:
        sockaddr = item[4] if len(item) > 4 else ()
        address = str(sockaddr[0] if sockaddr else "").strip()
        if address and _is_blocked_ip(address):
            raise ExternalResearchError("private_ip_blocked", "网页地址解析到不允许的 IP")


def validate_public_url(url: str) -> str:
    """校验并规范一个公开 HTTP(S) URL。

    解析时不接受用户信息、控制字符、反斜杠或本地网段；DNS 结果也会
    逐个检查，避免常见的 SSRF 绕过。重定向不会由适配器自动跟随。
    """

    value = str(url or "").strip()
    if not value or len(value) > _MAX_URL_LENGTH:
        raise ExternalResearchError("invalid_url", "网页地址无效")
    if any(ord(char) < 0x20 for char in value) or "\\" in value or any(char.isspace() for char in value):
        raise ExternalResearchError("invalid_url", "网页地址包含不允许的字符")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ExternalResearchError("invalid_url", "网页地址解析失败") from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        raise ExternalResearchError("invalid_url", "只允许读取 HTTP 或 HTTPS 网页")
    if parsed.username or parsed.password:
        raise ExternalResearchError("invalid_url", "网页地址不允许包含用户名或密码")
    try:
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError as exc:
        raise ExternalResearchError("invalid_url", "网页端口无效") from exc
    if port is not None and not (1 <= int(port) <= 65535):
        raise ExternalResearchError("invalid_url", "网页端口无效")
    _resolve_public_host(hostname)
    return parsed.geturl()


def external_adapter_for_url(url: str) -> str:
    """Return the narrow adapter selected for one already-authorized URL.

    This is intentionally a pure host dispatch. Authorization still happens in
    ``DocContext.register_web_sources``/``claim_web_source_read`` and every
    adapter validates its own fixed endpoint before making a request.
    """

    try:
        host = (urlsplit(str(url or "").strip()).hostname or "").casefold().rstrip(".")
    except ValueError:
        host = ""
    if host in _GITHUB_HOSTS:
        return "github_public"
    if host in _YOUTUBE_HOSTS:
        return "youtube_transcript"
    return "jina_reader"


def _bounded_window(text: str, *, start_char: int, max_chars: int) -> tuple[str, int, bool]:
    normalized = str(text or "").replace("\x00", "").strip()
    try:
        start = max(0, min(1_000_000, int(start_char or 0)))
    except (TypeError, ValueError):
        start = 0
    try:
        limit = max(256, min(_DEFAULT_MAX_CHARS, int(max_chars or _DEFAULT_MAX_CHARS)))
    except (TypeError, ValueError):
        limit = _DEFAULT_MAX_CHARS
    window = normalized[start:start + limit].rstrip()
    return window, start, start + len(window) < len(normalized)


def _text_result(
    *,
    url: str,
    adapter: str,
    text: str,
    max_chars: int,
    start_char: int,
    status: str = "completed",
    error_code: str = "",
    error: str = "",
    content_kind: str = "web_page",
    metadata: dict[str, Any] | None = None,
) -> dict:
    window, start, truncated = _bounded_window(text, start_char=start_char, max_chars=max_chars)
    effective_status = status if status != "completed" or window else "empty"
    return {
        "url": str(url or "")[:_MAX_URL_LENGTH],
        "backend": adapter,
        "adapter": adapter,
        "status": effective_status,
        "error_code": error_code if effective_status != "completed" else "",
        "error": error if effective_status != "completed" else "",
        "text": window,
        "content_hash": hashlib.sha256(window.encode("utf-8", errors="ignore")).hexdigest() if window else "",
        "char_count": len(window),
        "truncated": truncated,
        "content_start": start,
        "max_chars": max_chars,
        "content_kind": content_kind,
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def _failure_result(
    *,
    url: str,
    adapter: str,
    code: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> dict:
    return {
        "url": str(url or "")[:_MAX_URL_LENGTH],
        "backend": adapter,
        "adapter": adapter,
        "status": "failed",
        "error_code": code,
        "error": message,
        "text": "",
        "content_hash": "",
        "char_count": 0,
        "truncated": False,
        "content_start": 0,
        "content_kind": "",
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


async def _fetch_public_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_bytes: int = _MAX_ADAPTER_RESPONSE_BYTES,
) -> dict:
    """Fetch a fixed public endpoint without following redirects."""

    try:
        safe_url = validate_public_url(url)
    except ExternalResearchError as exc:
        return {
            "ok": False,
            "url": str(url or "")[:_MAX_URL_LENGTH],
            "error_code": exc.code,
            "error": exc.message,
            "status_code": 0,
            "body": b"",
        }
    timeout_value = max(2.0, min(60.0, float(timeout_s or _DEFAULT_TIMEOUT_S)))
    byte_limit = max(16_384, min(10_000_000, int(max_bytes or _MAX_ADAPTER_RESPONSE_BYTES)))
    chunks: list[bytes] = []
    total_bytes = 0
    try:
        timeout = httpx.Timeout(timeout_value, connect=min(8.0, timeout_value))
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            headers={
                "Accept": "*/*",
                "User-Agent": "ChatPDF-ExternalResearch/1.0",
                **(headers or {}),
            },
        ) as client:
            async with client.stream("GET", safe_url) as response:
                if 300 <= response.status_code < 400:
                    return {
                        "ok": False,
                        "url": safe_url,
                        "error_code": "redirect_not_followed",
                        "error": "外部网页发生重定向，已停止读取",
                        "status_code": response.status_code,
                        "body": b"",
                    }
                if response.status_code >= 400:
                    return {
                        "ok": False,
                        "url": safe_url,
                        "error_code": "http_status",
                        "error": f"外部网页读取失败（HTTP {response.status_code}）",
                        "status_code": response.status_code,
                        "body": b"",
                    }
                declared = response.headers.get("content-length")
                try:
                    if declared and int(declared) > byte_limit:
                        return {
                            "ok": False,
                            "url": safe_url,
                            "error_code": "response_too_large",
                            "error": "外部网页响应超过大小限制",
                            "status_code": response.status_code,
                            "body": b"",
                        }
                except (TypeError, ValueError):
                    pass
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    remaining = byte_limit - total_bytes
                    if remaining <= 0:
                        break
                    piece = bytes(chunk[:remaining])
                    chunks.append(piece)
                    total_bytes += len(piece)
                    if len(piece) < len(chunk) or total_bytes >= byte_limit:
                        break
    except (httpx.TimeoutException, TimeoutError):
        return {
            "ok": False,
            "url": str(url or "")[:_MAX_URL_LENGTH],
            "error_code": "timeout",
            "error": "外部网页读取超时",
            "status_code": 0,
            "body": b"",
        }
    except (httpx.HTTPError, OSError) as exc:
        logger.info("external adapter request failed: %s", type(exc).__name__)
        return {
            "ok": False,
            "url": str(url or "")[:_MAX_URL_LENGTH],
            "error_code": "network_error",
            "error": "外部网页暂时无法读取",
            "status_code": 0,
            "body": b"",
        }
    return {
        "ok": True,
        "url": safe_url,
        "error_code": "",
        "error": "",
        "status_code": 200,
        "body": b"".join(chunks),
    }


def _reader_url(source_url: str) -> str:
    # Jina Reader 将原始公开 URL 放在路径中，返回 markdown/纯文本。
    # source_url 已先通过 validate_public_url；不允许 Planner 改写该地址。
    return f"https://{_READER_HOST}/{source_url}"


def _parse_github_target(url: str) -> dict[str, str] | None:
    try:
        parsed = urlsplit(validate_public_url(url))
    except ExternalResearchError:
        return None
    host = (parsed.hostname or "").casefold().rstrip(".")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if host == "raw.githubusercontent.com":
        if len(parts) < 4 or not all(_GITHUB_PATH_PART_RE.match(part) for part in parts[:2]):
            return None
        return {
            "kind": "raw",
            "owner": parts[0],
            "repo": parts[1].removesuffix(".git"),
        }
    if host not in {"github.com", "www.github.com"} or len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if not _GITHUB_PATH_PART_RE.match(owner) or not _GITHUB_PATH_PART_RE.match(repo):
        return None
    if len(parts) == 2:
        return {"kind": "repo", "owner": owner, "repo": repo}
    resource, rest = parts[2], parts[3:]
    if resource in {"issues", "pull", "pulls"} and rest and _GITHUB_ISSUE_NUMBER_RE.match(rest[0]):
        return {
            "kind": "pull" if resource in {"pull", "pulls"} else "issue",
            "owner": owner,
            "repo": repo,
            "number": rest[0],
        }
    if resource == "blob" and len(rest) >= 2:
        return {
            "kind": "blob",
            "owner": owner,
            "repo": repo,
            "ref": rest[0],
            "path": "/".join(rest[1:]),
        }
    if resource == "tree" and rest:
        return {
            "kind": "tree",
            "owner": owner,
            "repo": repo,
            "ref": rest[0],
            "path": "/".join(rest[1:]),
        }
    return {"kind": "repo", "owner": owner, "repo": repo}


def _github_api_url(path: str, query: dict[str, str] | None = None) -> str:
    query_string = urlencode(query or {})
    return f"https://{_GITHUB_API_HOST}{path}" + (f"?{query_string}" if query_string else "")


def _decode_response_text(body: bytes) -> str:
    return body.decode("utf-8", errors="replace").replace("\x00", "").strip()


async def read_github_public_source(
    url: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_chars: int = _DEFAULT_MAX_CHARS,
    start_char: int = 0,
) -> dict:
    """Read public GitHub repository, README, file, Issue or PR content.

    The planner still supplies only a source id. This adapter receives the
    already-authorized URL and uses anonymous GitHub APIs, never a token or
    repository write endpoint.
    """

    target = _parse_github_target(url)
    if not target:
        return _failure_result(
            url=url,
            adapter="github_public",
            code="unsupported_github_url",
            message="GitHub 来源地址格式不受支持",
        )
    owner = target["owner"]
    repo = target["repo"]
    kind = target["kind"]
    metadata: dict[str, Any] = {
        "owner": owner,
        "repo": repo,
        "resource": kind,
    }
    api_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if kind == "raw":
        response = await _fetch_public_bytes(
            url,
            headers={"Accept": "text/plain, text/markdown;q=0.9"},
            timeout_s=timeout_s,
            max_bytes=_MAX_ADAPTER_RESPONSE_BYTES,
        )
        if not response.get("ok"):
            return _failure_result(
                url=url,
                adapter="github_public",
                code=str(response.get("error_code") or "github_read_failed"),
                message=str(response.get("error") or "GitHub 文件读取失败"),
                metadata=metadata,
            )
        text = _decode_response_text(response.get("body") or b"")
        metadata["content_source"] = "raw.githubusercontent.com"
        return _text_result(
            url=url,
            adapter="github_public",
            text=text,
            max_chars=max_chars,
            start_char=start_char,
            content_kind="github_file",
            metadata=metadata,
        )

    if kind in {"issue", "pull"}:
        endpoint = (
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/"
            f"{'pulls' if kind == 'pull' else 'issues'}/{target['number']}"
        )
    elif kind == "blob":
        endpoint = (
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/contents/"
            f"{quote(target['path'], safe='/')}"
        )
    elif kind == "tree":
        endpoint = (
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/contents/"
            f"{quote(target.get('path') or '', safe='/')}"
        )
    else:
        endpoint = f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/readme"

    query: dict[str, str] = {}
    if target.get("ref"):
        query["ref"] = target["ref"]
    response = await _fetch_public_bytes(
        _github_api_url(endpoint, query),
        headers=api_headers,
        timeout_s=timeout_s,
        max_bytes=_MAX_ADAPTER_RESPONSE_BYTES,
    )
    if not response.get("ok"):
        # Repository pages commonly have no README. Return a small metadata
        # record rather than hiding the fact that GitHub was reachable.
        if kind == "repo" and response.get("error_code") == "http_status":
            metadata_only = f"GitHub 公共仓库: {owner}/{repo}"
            metadata["content_source"] = "github_api"
            metadata["warning"] = "README 不可用"
            return _text_result(
                url=url,
                adapter="github_public",
                text=metadata_only,
                max_chars=max_chars,
                start_char=start_char,
                content_kind="github_metadata",
                metadata=metadata,
            )
        return _failure_result(
            url=url,
            adapter="github_public",
            code=str(response.get("error_code") or "github_read_failed"),
            message=str(response.get("error") or "GitHub 内容读取失败"),
            metadata=metadata,
        )

    body_text = _decode_response_text(response.get("body") or b"")
    try:
        payload = json.loads(body_text)
    except (TypeError, ValueError):
        payload = None

    if kind == "blob":
        if not isinstance(payload, dict):
            return _failure_result(
                url=url,
                adapter="github_public",
                code="invalid_github_response",
                message="GitHub 文件响应格式无效",
                metadata=metadata,
            )
        encoded = str(payload.get("content") or "").replace("\n", "")
        try:
            text = base64.b64decode(encoded, validate=False).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            text = str(payload.get("content") or "")
        metadata.update({
            "path": target.get("path", ""),
            "ref": target.get("ref", ""),
            "content_source": "github_api",
        })
        return _text_result(
            url=url,
            adapter="github_public",
            text=text,
            max_chars=max_chars,
            start_char=start_char,
            content_kind="github_file",
            metadata=metadata,
        )

    if kind == "tree":
        if not isinstance(payload, list):
            return _failure_result(
                url=url,
                adapter="github_public",
                code="invalid_github_response",
                message="GitHub 目录响应格式无效",
                metadata=metadata,
            )
        rows = [
            f"{item.get('type', 'unknown')}: {item.get('path', '')}"
            for item in payload
            if isinstance(item, dict) and item.get("path")
        ]
        metadata.update({"ref": target.get("ref", ""), "content_source": "github_api"})
        return _text_result(
            url=url,
            adapter="github_public",
            text="\n".join(rows),
            max_chars=max_chars,
            start_char=start_char,
            content_kind="github_tree",
            metadata=metadata,
        )

    if not isinstance(payload, dict):
        return _failure_result(
            url=url,
            adapter="github_public",
            code="invalid_github_response",
            message="GitHub API 响应格式无效",
            metadata=metadata,
        )

    title = str(payload.get("title") or payload.get("name") or f"{owner}/{repo}").strip()
    body = str(payload.get("body") or payload.get("description") or "").strip()
    if kind == "repo":
        encoded_readme = str(payload.get("content") or "").replace("\n", "")
        if encoded_readme:
            try:
                body = base64.b64decode(encoded_readme, validate=False).decode("utf-8", errors="replace").strip()
            except (ValueError, TypeError):
                body = str(payload.get("content") or "").strip()
        text = f"仓库: {owner}/{repo}\n标题: {title}\n\n{body}".strip()
        content_kind = "github_readme"
    else:
        labels = ", ".join(
            str(label.get("name") or "").strip()
            for label in (payload.get("labels") or [])
            if isinstance(label, dict) and label.get("name")
        )
        state = str(payload.get("state") or "").strip()
        author = str((payload.get("user") or {}).get("login") or "").strip()
        text = "\n".join(
            part
            for part in (
                f"{'Pull Request' if kind == 'pull' else 'Issue'}: {title}",
                f"状态: {state}" if state else "",
                f"作者: {author}" if author else "",
                f"标签: {labels}" if labels else "",
                "",
                body,
            )
            if part
        ).strip()
        content_kind = "github_pull" if kind == "pull" else "github_issue"
    metadata.update({
        "title": title,
        "content_source": "github_api",
        "number": target.get("number", ""),
    })
    return _text_result(
        url=url,
        adapter="github_public",
        text=text,
        max_chars=max_chars,
        start_char=start_char,
        content_kind=content_kind,
        metadata=metadata,
    )


def _github_repo_parts(owner: str, repo: str) -> tuple[str, str] | None:
    normalized_owner = str(owner or "").strip()
    normalized_repo = str(repo or "").strip().removesuffix(".git")
    if not _GITHUB_PATH_PART_RE.match(normalized_owner):
        return None
    if not _GITHUB_PATH_PART_RE.match(normalized_repo):
        return None
    return normalized_owner, normalized_repo


def _github_tree_failure(code: str, message: str, **extra: Any) -> dict:
    return {
        "status": "failed",
        "error_code": code,
        "error": message,
        "entries": [],
        "entry_count": 0,
        "truncated": False,
        "ref": "",
        **extra,
    }


async def read_github_default_branch(
    owner: str,
    repo: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict:
    """Resolve the default branch of a public repository over the anonymous API."""

    parts = _github_repo_parts(owner, repo)
    if parts is None:
        return {"status": "failed", "error_code": "unsupported_github_repo", "error": "GitHub 仓库标识无效", "branch": ""}
    safe_owner, safe_repo = parts
    response = await _fetch_public_bytes(
        _github_api_url(f"/repos/{quote(safe_owner, safe='')}/{quote(safe_repo, safe='')}"),
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout_s=timeout_s,
        max_bytes=_MAX_ADAPTER_RESPONSE_BYTES,
    )
    if not response.get("ok"):
        return {
            "status": "failed",
            "error_code": str(response.get("error_code") or "github_read_failed"),
            "error": str(response.get("error") or "GitHub 仓库元数据读取失败"),
            "branch": "",
        }
    try:
        payload = json.loads(_decode_response_text(response.get("body") or b""))
    except (TypeError, ValueError):
        payload = None
    branch = str((payload or {}).get("default_branch") or "").strip() if isinstance(payload, dict) else ""
    if not branch:
        return {
            "status": "failed",
            "error_code": "invalid_github_response",
            "error": "GitHub 仓库元数据缺少默认分支",
            "branch": "",
        }
    return {"status": "completed", "error_code": "", "error": "", "branch": branch}


async def read_github_repo_tree(
    owner: str,
    repo: str,
    *,
    ref: str = "",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_entries: int = 4000,
) -> dict:
    """Read one public repository's recursive git tree.

    This is the directory-listing counterpart of ``read_github_public_source``:
    read-only, anonymous, no token, no clone, and no write endpoint. The single
    ``/contents`` listing that adapter performs only covers one directory level,
    which is not enough to locate a training script or loss definition.
    """

    parts = _github_repo_parts(owner, repo)
    if parts is None:
        return _github_tree_failure("unsupported_github_repo", "GitHub 仓库标识无效", owner=str(owner or ""), repo=str(repo or ""))
    safe_owner, safe_repo = parts
    resolved_ref = str(ref or "").strip()
    if resolved_ref and not re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", resolved_ref):
        return _github_tree_failure("unsupported_github_ref", "GitHub 分支或提交标识无效", owner=safe_owner, repo=safe_repo)
    if not resolved_ref:
        branch_result = await read_github_default_branch(safe_owner, safe_repo, timeout_s=timeout_s)
        if branch_result.get("status") != "completed":
            return _github_tree_failure(
                str(branch_result.get("error_code") or "github_read_failed"),
                str(branch_result.get("error") or "GitHub 默认分支解析失败"),
                owner=safe_owner,
                repo=safe_repo,
            )
        resolved_ref = str(branch_result.get("branch") or "")

    response = await _fetch_public_bytes(
        _github_api_url(
            f"/repos/{quote(safe_owner, safe='')}/{quote(safe_repo, safe='')}/"
            f"git/trees/{quote(resolved_ref, safe='/')}",
            {"recursive": "1"},
        ),
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout_s=timeout_s,
        max_bytes=_MAX_ADAPTER_RESPONSE_BYTES,
    )
    if not response.get("ok"):
        return _github_tree_failure(
            str(response.get("error_code") or "github_read_failed"),
            str(response.get("error") or "GitHub 目录树读取失败"),
            owner=safe_owner,
            repo=safe_repo,
            ref=resolved_ref,
        )
    try:
        payload = json.loads(_decode_response_text(response.get("body") or b""))
    except (TypeError, ValueError):
        payload = None
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), list):
        return _github_tree_failure(
            "invalid_github_response",
            "GitHub 目录树响应格式无效",
            owner=safe_owner,
            repo=safe_repo,
            ref=resolved_ref,
        )
    limit = max(1, min(20_000, int(max_entries or 4000)))
    entries: list[dict[str, Any]] = []
    for item in payload["tree"]:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        entries.append({
            "path": path[:400],
            "type": str(item.get("type") or "blob").strip() or "blob",
            "size": max(0, size),
        })
        if len(entries) >= limit:
            break
    return {
        "status": "completed",
        "error_code": "",
        "error": "",
        "owner": safe_owner,
        "repo": safe_repo,
        "ref": resolved_ref,
        "entries": entries,
        "entry_count": len(entries),
        "truncated": bool(payload.get("truncated")) or len(entries) >= limit,
    }


def _parse_youtube_video_id(url: str) -> tuple[str, str] | None:
    try:
        safe_url = validate_public_url(url)
        parsed = urlsplit(safe_url)
    except ExternalResearchError:
        return None
    host = (parsed.hostname or "").casefold().rstrip(".")
    if host not in _YOUTUBE_HOSTS:
        return None
    video_id = ""
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/", 1)[0]
    else:
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        path_parts = [part for part in parsed.path.split("/") if part]
        if query_id:
            video_id = query_id
        elif len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            video_id = path_parts[1]
    if not _YOUTUBE_VIDEO_ID_RE.match(video_id):
        return None
    return video_id, safe_url


def _youtube_timestamp(seconds: str) -> str:
    try:
        total = max(0, int(float(seconds or 0)))
    except (TypeError, ValueError):
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}" if hours else f"{minutes:02d}:{seconds_value:02d}"


def _parse_youtube_tracks(body: str) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(body)
    except (ET.ParseError, TypeError, ValueError):
        return []
    tracks: list[dict[str, str]] = []
    for item in root.findall(".//track"):
        if not isinstance(item, ET.Element):
            continue
        lang = str(item.attrib.get("lang_code") or item.attrib.get("lang") or "").strip()
        if lang:
            tracks.append({
                "lang": lang,
                "name": str(item.attrib.get("name") or "").strip(),
                "kind": str(item.attrib.get("kind") or "").strip(),
            })
    return tracks


def _choose_youtube_track(tracks: list[dict[str, str]]) -> dict[str, str] | None:
    if not tracks:
        return None
    preferred = ("zh-Hans", "zh-CN", "zh", "en", "en-US", "en-GB")
    for wanted in preferred:
        for track in tracks:
            if track.get("lang", "").casefold() == wanted.casefold():
                return track
    return tracks[0]


def _parse_youtube_transcript(body: str) -> str:
    try:
        root = ET.fromstring(body)
    except (ET.ParseError, TypeError, ValueError):
        return ""
    lines: list[str] = []
    for item in root.findall(".//text"):
        if not isinstance(item, ET.Element):
            continue
        value = " ".join(unescape("".join(item.itertext())).split())
        if not value:
            continue
        timestamp = _youtube_timestamp(item.attrib.get("start", "0"))
        lines.append(f"[{timestamp}] {value}")
    return "\n".join(lines)


async def read_youtube_public_source(
    url: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_chars: int = _DEFAULT_MAX_CHARS,
    start_char: int = 0,
) -> dict:
    """Read public YouTube metadata and the first available transcript track."""

    target = _parse_youtube_video_id(url)
    if not target:
        return _failure_result(
            url=url,
            adapter="youtube_transcript",
            code="unsupported_youtube_url",
            message="YouTube 来源地址格式不受支持",
        )
    video_id, safe_url = target
    metadata: dict[str, Any] = {"video_id": video_id}
    oembed_url = "https://www.youtube.com/oembed?" + urlencode({"url": safe_url, "format": "json"})
    oembed = await _fetch_public_bytes(
        oembed_url,
        headers={"Accept": "application/json"},
        timeout_s=timeout_s,
        max_bytes=256_000,
    )
    oembed_payload: dict[str, Any] = {}
    if oembed.get("ok"):
        try:
            parsed_oembed = json.loads(_decode_response_text(oembed.get("body") or b""))
            if isinstance(parsed_oembed, dict):
                oembed_payload = parsed_oembed
        except (TypeError, ValueError):
            pass
    title = str(oembed_payload.get("title") or "").strip()
    author = str(oembed_payload.get("author_name") or "").strip()
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author

    tracks_response = await _fetch_public_bytes(
        "https://" + _YOUTUBE_TRANSCRIPT_HOST + "/timedtext?" + urlencode({"type": "list", "v": video_id}),
        headers={"Accept": "application/xml, text/xml;q=0.9"},
        timeout_s=timeout_s,
        max_bytes=512_000,
    )
    tracks = _parse_youtube_tracks(_decode_response_text(tracks_response.get("body") or b"")) if tracks_response.get("ok") else []
    selected = _choose_youtube_track(tracks)
    transcript = ""
    if selected:
        metadata["language"] = selected.get("lang", "")
        transcript_url = "https://" + _YOUTUBE_TRANSCRIPT_HOST + "/timedtext?" + urlencode({
            "lang": selected.get("lang", ""),
            "v": video_id,
            "fmt": "srv3",
        })
        transcript_response = await _fetch_public_bytes(
            transcript_url,
            headers={"Accept": "application/xml, text/xml;q=0.9"},
            timeout_s=timeout_s,
            max_bytes=_MAX_ADAPTER_RESPONSE_BYTES,
        )
        if transcript_response.get("ok"):
            transcript = _parse_youtube_transcript(_decode_response_text(transcript_response.get("body") or b""))

    header = "\n".join(
        part
        for part in (
            f"视频标题: {title}" if title else "",
            f"频道: {author}" if author else "",
            f"视频 ID: {video_id}",
            f"字幕语言: {metadata.get('language')}" if metadata.get("language") else "",
        )
        if part
    )
    if transcript:
        text = f"{header}\n\n[字幕]\n{transcript}" if header else transcript
        metadata["content_source"] = "youtube_timedtext"
        return _text_result(
            url=safe_url,
            adapter="youtube_transcript",
            text=text,
            max_chars=max_chars,
            start_char=start_char,
            content_kind="youtube_transcript",
            metadata=metadata,
        )
    if header:
        metadata["content_source"] = "youtube_oembed"
        metadata["warning"] = "未找到公开字幕，仅返回视频元数据"
        return _text_result(
            url=safe_url,
            adapter="youtube_transcript",
            text=header,
            max_chars=max_chars,
            start_char=start_char,
            content_kind="youtube_metadata",
            metadata=metadata,
        )
    return _failure_result(
        url=safe_url,
        adapter="youtube_transcript",
        code="transcript_unavailable",
        message="YouTube 没有可读取的公开字幕或元数据",
        metadata=metadata,
    )


def _browser_render_enabled() -> bool:
    value = str(os.getenv("CHATPDF_EXTERNAL_BROWSER_ENABLED", "")).strip().casefold()
    return value in {"1", "true", "yes", "on"}


async def render_public_web_source(
    url: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_chars: int = _DEFAULT_MAX_CHARS,
    start_char: int = 0,
) -> dict:
    """Optionally render a public page with Playwright.

    Playwright is deliberately an optional runtime dependency. The packaged
    desktop build excludes it by default; enabling this path is an explicit
    server-side deployment choice via ``CHATPDF_EXTERNAL_BROWSER_ENABLED``.
    All HTTP requests made by the page are filtered through the same public URL
    policy, and images/media/fonts are blocked to keep the fallback text-only.
    """

    try:
        safe_url = validate_public_url(url)
    except ExternalResearchError as exc:
        return _failure_result(
            url=url,
            adapter="browser_render",
            code=exc.code,
            message=exc.message,
        )
    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except ImportError:
        return _failure_result(
            url=safe_url,
            adapter="browser_render",
            code="browser_renderer_unavailable",
            message="浏览器渲染后端未安装",
        )

    timeout_value = max(2.0, min(60.0, float(timeout_s or _DEFAULT_TIMEOUT_S)))
    timeout_ms = int(timeout_value * 1000)
    checked_hosts: set[str] = set()
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent="ChatPDF-ExternalResearch/1.0",
                    java_script_enabled=True,
                )

                async def filter_request(route):
                    request_url = str(route.request.url or "")
                    scheme = (urlsplit(request_url).scheme or "").casefold()
                    if scheme in {"data", "about", "blob"}:
                        await route.continue_()
                        return
                    if scheme not in {"http", "https"}:
                        await route.abort()
                        return
                    try:
                        host = (urlsplit(request_url).hostname or "").casefold().rstrip(".")
                        if host not in checked_hosts:
                            validate_public_url(request_url)
                            checked_hosts.add(host)
                    except ExternalResearchError:
                        await route.abort()
                        return
                    if route.request.resource_type in {"image", "media", "font", "stylesheet"}:
                        await route.abort()
                        return
                    await route.continue_()

                await context.route("**/*", filter_request)
                page = await context.new_page()
                response = await page.goto(safe_url, wait_until="domcontentloaded", timeout=timeout_ms)
                final_url = str(page.url or safe_url)
                validate_public_url(final_url)
                if response is not None and int(response.status) >= 400:
                    return _failure_result(
                        url=safe_url,
                        adapter="browser_render",
                        code="http_status",
                        message=f"外部网页读取失败（HTTP {response.status}）",
                    )
                try:
                    await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5000))
                except PlaywrightTimeoutError:
                    # 依赖长轮询的页面不应阻塞全文读取；DOM 已经可用即可继续。
                    pass
                body = await page.locator("body").inner_text(timeout=timeout_ms)
                return _text_result(
                    url=safe_url,
                    adapter="browser_render",
                    text=body,
                    max_chars=max_chars,
                    start_char=start_char,
                    content_kind="rendered_web_page",
                    metadata={"final_url": final_url, "rendered": True},
                )
            finally:
                await browser.close()
    except Exception as exc:
        logger.info("external browser render failed: %s", type(exc).__name__)
        return _failure_result(
            url=safe_url,
            adapter="browser_render",
            code="browser_render_failed",
            message="外部网页浏览器渲染失败",
        )


async def read_external_research_source(
    url: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_chars: int = _DEFAULT_MAX_CHARS,
    start_char: int = 0,
) -> dict:
    """Dispatch one authorized public URL to its source-specific adapter."""

    adapter = external_adapter_for_url(url)
    if adapter == "github_public":
        return await read_github_public_source(
            url,
            timeout_s=timeout_s,
            max_chars=max_chars,
            start_char=start_char,
        )
    if adapter == "youtube_transcript":
        return await read_youtube_public_source(
            url,
            timeout_s=timeout_s,
            max_chars=max_chars,
            start_char=start_char,
        )
    reader_result = await read_public_web_source(
        url,
        timeout_s=timeout_s,
        max_chars=max_chars,
        start_char=start_char,
    )
    if not _browser_render_enabled():
        return reader_result
    # Only retry recoverable reader failures. Invalid/private URLs must remain
    # rejected by the first policy gate and should never reach a browser.
    if str(reader_result.get("status") or "").strip().lower() == "completed":
        return reader_result
    if str(reader_result.get("error_code") or "").strip() in {
        "invalid_url",
        "private_host_blocked",
        "private_ip_blocked",
        "dns_resolution_failed",
    }:
        return reader_result
    rendered = await render_public_web_source(
        url,
        timeout_s=timeout_s,
        max_chars=max_chars,
        start_char=start_char,
    )
    if str(rendered.get("status") or "").strip().lower() == "completed":
        metadata = rendered.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["fallback_from"] = "jina_reader"
        return rendered
    return reader_result


async def read_public_web_source(
    url: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_chars: int = _DEFAULT_MAX_CHARS,
    start_char: int = 0,
) -> dict:
    """通过 Jina Reader 读取一个已授权的公开网页。

    适配器不跟随重定向，并以字节数和字符数双重限制响应。所有失败均以
    结构化结果返回，调用方可以只降级当前外部来源而不影响文档回答。
    """

    try:
        safe_url = validate_public_url(url)
    except ExternalResearchError as exc:
        return {
            "url": str(url or "")[:_MAX_URL_LENGTH],
            "backend": "jina_reader",
            "status": "failed",
            "error_code": exc.code,
            "error": exc.message,
            "text": "",
            "content_hash": "",
            "char_count": 0,
            "truncated": False,
        }

    timeout_value = max(2.0, min(60.0, float(timeout_s or _DEFAULT_TIMEOUT_S)))
    byte_limit = max(16_384, min(10_000_000, int(max_bytes or _DEFAULT_MAX_BYTES)))
    char_limit = max(256, min(_DEFAULT_MAX_CHARS, int(max_chars or _DEFAULT_MAX_CHARS)))
    try:
        start_offset = max(0, min(1_000_000, int(start_char or 0)))
    except (TypeError, ValueError):
        start_offset = 0
    reader_url = _reader_url(safe_url)
    chunks: list[bytes] = []
    total_bytes = 0
    try:
        timeout = httpx.Timeout(timeout_value, connect=min(8.0, timeout_value))
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            headers={
                "Accept": "text/plain, text/markdown;q=0.9",
                "User-Agent": "ChatPDF-ExternalResearch/1.0",
            },
        ) as client:
            async with client.stream("GET", reader_url) as response:
                if 300 <= response.status_code < 400:
                    return {
                        "url": safe_url,
                        "backend": "jina_reader",
                        "status": "failed",
                        "error_code": "redirect_not_followed",
                        "error": "外部网页发生重定向，已停止读取",
                        "text": "",
                        "content_hash": "",
                        "char_count": 0,
                        "truncated": False,
                    }
                if response.status_code >= 400:
                    return {
                        "url": safe_url,
                        "backend": "jina_reader",
                        "status": "failed",
                        "error_code": "http_status",
                        "error": f"外部网页读取失败（HTTP {response.status_code}）",
                        "text": "",
                        "content_hash": "",
                        "char_count": 0,
                        "truncated": False,
                    }
                declared = response.headers.get("content-length")
                try:
                    if declared and int(declared) > byte_limit:
                        return {
                            "url": safe_url,
                            "backend": "jina_reader",
                            "status": "failed",
                            "error_code": "response_too_large",
                            "error": "外部网页响应超过大小限制",
                            "text": "",
                            "content_hash": "",
                            "char_count": 0,
                            "truncated": False,
                        }
                except (TypeError, ValueError):
                    pass
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    remaining = byte_limit - total_bytes
                    if remaining <= 0:
                        break
                    piece = bytes(chunk[:remaining])
                    chunks.append(piece)
                    total_bytes += len(piece)
                    if len(piece) < len(chunk) or total_bytes >= byte_limit:
                        break
    except (httpx.TimeoutException, TimeoutError):
        return {
            "url": safe_url,
            "backend": "jina_reader",
            "status": "failed",
            "error_code": "timeout",
            "error": "外部网页读取超时",
            "text": "",
            "content_hash": "",
            "char_count": 0,
            "truncated": False,
        }
    except (httpx.HTTPError, OSError) as exc:
        logger.info("external web source read failed: %s", type(exc).__name__)
        return {
            "url": safe_url,
            "backend": "jina_reader",
            "status": "failed",
            "error_code": "network_error",
            "error": "外部网页暂时无法读取",
            "text": "",
            "content_hash": "",
            "char_count": 0,
            "truncated": False,
        }

    raw = b"".join(chunks)
    decoded_text = raw.decode("utf-8", errors="replace").replace("\x00", "").strip()
    byte_truncated = total_bytes >= byte_limit
    text = decoded_text[start_offset:start_offset + char_limit]
    truncated = byte_truncated or start_offset + len(text) < len(decoded_text)
    text = text.rstrip()
    content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return {
        "url": safe_url,
        "backend": "jina_reader",
        "status": "completed" if text else "empty",
        "error_code": "" if text else "empty_content",
        "error": "" if text else "外部网页没有可读取的正文",
        "text": text,
        "content_hash": content_hash,
        "char_count": len(text),
        "truncated": truncated,
        "content_start": start_offset,
        "max_bytes": byte_limit,
        "max_chars": char_limit,
        "redirects_followed": _MAX_REDIRECTS,
    }


__all__ = [
    "EXTERNAL_RESEARCH_ADAPTERS",
    "ExternalResearchAdapter",
    "ExternalResearchError",
    "external_adapter_for_url",
    "validate_public_url",
    "read_public_web_source",
    "read_github_default_branch",
    "read_github_public_source",
    "read_github_repo_tree",
    "read_youtube_public_source",
    "render_public_web_source",
    "read_external_research_source",
]
