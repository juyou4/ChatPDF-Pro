"""Shared configuration and invocation helpers for visual-language work.

This module deliberately has no parser-route logic.  A visual model is an
optional, scoped enhancement; callers remain responsible for deciding whether
their source content is risky enough to warrant an image request.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


_KEYLESS_LOCAL_PROVIDERS = {"local", "ollama"}
_VISUAL_STRATEGIES = {"privacy", "balanced", "quality"}


def _normalized(value: str | None) -> str:
    return str(value or "").strip()


def _is_loopback_endpoint(endpoint: str) -> bool:
    value = _normalized(endpoint)
    if not value or value.startswith("/"):
        return True
    try:
        parsed = urlparse(value if "://" in value else f"//{value}")
        host = str(parsed.hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(address.is_loopback or address.is_unspecified)


def model_config_identity(
    provider: str,
    model: str,
    endpoint: str = "",
    *,
    enabled: bool | None = None,
    available: bool | None = None,
) -> str:
    """Return a stable, non-secret identity for a model configuration."""
    payload = {
        "provider": _normalized(provider).lower(),
        "model": _normalized(model),
        "endpoint": _normalized(endpoint).rstrip("/"),
        "enabled": None if enabled is None else bool(enabled),
        "available": None if available is None else bool(available),
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def visual_model_identity(
    provider: str,
    model: str,
    endpoint: str = "",
    enabled: bool | None = None,
    available: bool | None = None,
) -> str:
    """Return a stable, non-secret identity for a visual model configuration."""
    return model_config_identity(
        provider,
        model,
        endpoint,
        enabled=enabled,
        available=available,
    )


@dataclass(frozen=True)
class VisualModelConfig:
    """Resolved VLM credentials for one bounded visual enhancement request."""

    provider: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)
    endpoint: str = ""
    source: str = "follow_chat"
    enabled: bool = True

    @property
    def identity(self) -> str:
        return visual_model_identity(
            self.provider,
            self.model,
            self.endpoint,
            self.enabled,
            self.can_call,
        )

    @property
    def can_call(self) -> bool:
        if not self.enabled or not self.provider or not self.model:
            return False
        return bool(self.api_key) or self.provider.strip().lower() in _KEYLESS_LOCAL_PROVIDERS

    @property
    def is_local(self) -> bool:
        return (
            self.provider.strip().lower() in _KEYLESS_LOCAL_PROVIDERS
            and _is_loopback_endpoint(self.endpoint)
        )

    def public_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "endpoint_identity": self.identity,
            "source": self.source,
            "enabled": self.enabled,
            "available": self.can_call,
            "local_execution": self.is_local,
        }


@dataclass(frozen=True)
class VisualEnrichmentPolicy:
    """Select a local or strong VLM without changing the parser route."""

    strategy: str
    strong_model: VisualModelConfig
    local_model: VisualModelConfig

    @property
    def normalized_strategy(self) -> str:
        value = _normalized(self.strategy).lower()
        return value if value in _VISUAL_STRATEGIES else "balanced"

    @property
    def risk_threshold(self) -> float:
        return {"privacy": 0.70, "balanced": 0.45, "quality": 0.25}[self.normalized_strategy]

    @property
    def document_budget(self) -> int:
        return {"privacy": 8, "balanced": 16, "quality": 32}[self.normalized_strategy]

    @property
    def identity(self) -> str:
        return model_config_identity(
            f"visual-policy:{self.normalized_strategy}:{self.strong_model.identity}",
            self.local_model.identity,
            "",
            enabled=True,
            available=bool(self.strong_model.can_call or self.local_model.can_call),
        )

    def select(self, *, risk_level: str = "medium", purpose: str = "") -> VisualModelConfig:
        strategy = self.normalized_strategy
        strong = self.strong_model
        local = self.local_model
        if strategy == "privacy":
            if local.can_call and local.is_local:
                return local
            if strong.can_call and strong.is_local:
                return strong
            return VisualModelConfig(source="privacy_blocked", enabled=False)

        precise_numeric = str(purpose or "").strip().lower() in {
            "numeric_table_verification",
            "numeric_table_visual_verification",
            "table_cell_verification",
        }
        if strategy == "quality" or precise_numeric or str(risk_level or "").lower() == "high":
            return strong if strong.can_call else local
        return local if local.can_call else strong

    def public_metadata(self) -> dict[str, Any]:
        return {
            "strategy": self.normalized_strategy,
            "identity": self.identity,
            "risk_threshold": self.risk_threshold,
            "document_budget": self.document_budget,
            "strong_model": self.strong_model.public_metadata(),
            "local_model": self.local_model.public_metadata(),
        }


def resolve_visual_model_config(
    *,
    primary_provider: str = "",
    primary_model: str = "",
    primary_api_key: str = "",
    primary_endpoint: str = "",
    visual_provider: str = "",
    visual_model: str = "",
    visual_api_key: str = "",
    visual_endpoint: str = "",
    visual_enabled: bool | None = None,
) -> VisualModelConfig:
    """Resolve explicit visual settings, otherwise intentionally follow chat.

    ``visual_enabled`` is a capability hint supplied by the desktop client.  A
    missing hint preserves API compatibility for non-UI callers and lets the
    provider decide at request time.
    """
    explicit_provider = _normalized(visual_provider)
    explicit_model = _normalized(visual_model)
    explicit = bool(explicit_provider or explicit_model)
    provider = explicit_provider or _normalized(primary_provider)
    model = explicit_model or _normalized(primary_model)
    endpoint = _normalized(visual_endpoint) if explicit else _normalized(primary_endpoint)
    api_key = _normalized(visual_api_key) if explicit else _normalized(primary_api_key)
    return VisualModelConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        source="dedicated" if explicit else "follow_chat",
        enabled=True if visual_enabled is None else bool(visual_enabled),
    )


def resolve_visual_enrichment_policy(
    *,
    strategy: str = "balanced",
    primary_provider: str = "",
    primary_model: str = "",
    primary_api_key: str = "",
    primary_endpoint: str = "",
    visual_provider: str = "",
    visual_model: str = "",
    visual_api_key: str = "",
    visual_endpoint: str = "",
    visual_enabled: bool | None = None,
    local_visual_provider: str = "",
    local_visual_model: str = "",
    local_visual_api_key: str = "",
    local_visual_endpoint: str = "",
) -> VisualEnrichmentPolicy:
    strong = resolve_visual_model_config(
        primary_provider=primary_provider,
        primary_model=primary_model,
        primary_api_key=primary_api_key,
        primary_endpoint=primary_endpoint,
        visual_provider=visual_provider,
        visual_model=visual_model,
        visual_api_key=visual_api_key,
        visual_endpoint=visual_endpoint,
        visual_enabled=visual_enabled,
    )
    local_provider = _normalized(local_visual_provider).lower()
    local_enabled = bool(local_provider in _KEYLESS_LOCAL_PROVIDERS and _normalized(local_visual_model))
    local = VisualModelConfig(
        provider=local_provider if local_enabled else "",
        model=_normalized(local_visual_model) if local_enabled else "",
        api_key=_normalized(local_visual_api_key) if local_enabled else "",
        endpoint=_normalized(local_visual_endpoint) if local_enabled else "",
        source="local_tier" if local_enabled else "local_tier_unavailable",
        enabled=local_enabled,
    )
    return VisualEnrichmentPolicy(
        strategy=strategy,
        strong_model=strong,
        local_model=local,
    )


class VisualModelUnavailable(RuntimeError):
    """Raised before a visual request when the selected model cannot be used."""


async def call_visual_model(
    *,
    messages: list[dict[str, Any]],
    config: VisualModelConfig,
    purpose: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    middlewares: list[Any] | None = None,
) -> Any:
    """Call the common AI client with a resolved VLM configuration."""
    if not config.can_call:
        raise VisualModelUnavailable("No enabled visual-language model is available")

    # Import lazily: chat_service has broad provider imports and this helper is
    # also used by lightweight cache/identity code during test collection.
    from services.chat_service import call_ai_api

    kwargs: dict[str, Any] = {
        "messages": messages,
        "api_key": config.api_key,
        "model": config.model,
        "provider": config.provider,
        "endpoint": config.endpoint,
        "purpose": purpose,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if middlewares is not None:
        kwargs["middlewares"] = middlewares
    return await call_ai_api(**kwargs)
