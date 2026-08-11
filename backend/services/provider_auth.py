"""Provider API key authentication helpers.

Dynamic providers only persist the non-sensitive authentication shape. The
actual key is supplied by the desktop client for each request and is selected
from a comma-separated pool at the last possible moment.
"""

from __future__ import annotations

import re
from typing import Mapping

from models.api_key_selector import select_api_key


_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")
_FORBIDDEN_HEADERS = {
    "accept",
    "connection",
    "content-length",
    "content-type",
    "cookie",
    "host",
    "origin",
    "proxy-authorization",
    "te",
    "transfer-encoding",
    "upgrade",
}


def normalize_api_key_header(value: str | None, *, default: str = "Authorization") -> str:
    """Normalize and validate a header name before it reaches httpx."""

    normalized = str(value or "").strip() or default
    if not _HEADER_NAME_RE.fullmatch(normalized) or normalized.lower() in _FORBIDDEN_HEADERS:
        raise ValueError("API Key 请求头名称不合法")
    return normalized


def normalize_api_key_prefix(value: str | None, *, default: str = "Bearer ") -> str:
    """Keep a small, header-safe authentication prefix."""

    if value is None:
        return default
    normalized = str(value)
    if len(normalized) > 64 or "\r" in normalized or "\n" in normalized:
        raise ValueError("API Key 请求头前缀不合法")
    return normalized


def resolve_api_key_auth(
    *,
    provider_type: str | None = None,
    api_key_header: str | None = None,
    api_key_prefix: str | None = None,
) -> tuple[str, str]:
    """Return ``(header, prefix)`` for a provider.

    Anthropic's native protocol uses ``x-api-key`` without a prefix. All
    OpenAI-compatible and custom providers retain the conventional Bearer
    header unless explicitly configured otherwise.
    """

    protocol = str(provider_type or "").strip().lower()
    default_header = "x-api-key" if protocol == "anthropic" else "Authorization"
    default_prefix = "" if protocol == "anthropic" else "Bearer "
    return (
        normalize_api_key_header(api_key_header, default=default_header),
        normalize_api_key_prefix(api_key_prefix, default=default_prefix),
    )


def build_api_key_headers(
    api_key: str | None,
    *,
    provider_type: str | None = None,
    api_key_header: str | None = None,
    api_key_prefix: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build request headers without persisting or logging the secret."""

    header, prefix = resolve_api_key_auth(
        provider_type=provider_type,
        api_key_header=api_key_header,
        api_key_prefix=api_key_prefix,
    )
    headers = {"Content-Type": "application/json"}
    actual_key = select_api_key(api_key) if api_key else ""
    if actual_key:
        value = str(actual_key).strip()
        prefix_without_space = prefix.strip()
        if prefix_without_space and value.lower().startswith(prefix_without_space.lower() + " "):
            value = value[len(prefix_without_space):].lstrip()
        headers[header] = f"{prefix}{value}"
    if extra_headers:
        for key, value in extra_headers.items():
            normalized_key = normalize_api_key_header(key)
            if normalized_key.lower() in {"authorization", "x-api-key"}:
                # Explicit auth headers must be represented by the API key
                # fields above, otherwise a custom header could silently
                # override the credential selected from the key pool.
                continue
            headers[normalized_key] = str(value)
    return headers
