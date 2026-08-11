"""论文元数据 provider 合同、注册表和缓存身份工具。

Provider 只负责外部来源的身份/来源提示。该模块不参与文档内容检索、
回答事实判断或 claim verifier。网络调用由 hydration service 注入，便于
测试时使用 fake client，也避免 provider 自己保存密钥或原文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Awaitable, Callable, Iterable, Mapping


REGISTRY_VERSION = "paper-metadata-provider-registry-v1"
EMPTY_VALUES = (None, "", [], {})

ProviderFetch = Callable[[Any, Mapping[str, Any]], Awaitable[Any]]
ProviderNormalize = Callable[[Any, Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ProviderSpec:
    """一个可诊断、可测试的元数据来源定义。"""

    name: str
    capabilities: tuple[str, ...] = ()
    requires_key: bool = False
    enabled_by_default: bool = True
    timeout: float = 8.0
    rate_limit: str = "unspecified"
    priority: int = 100
    supported_fields: tuple[str, ...] = ()
    fetch: ProviderFetch | None = None
    normalize: ProviderNormalize | None = None
    provenance: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("provider name cannot be empty")
        if self.fetch is not None and not callable(self.fetch):
            raise TypeError("provider fetch must be callable")
        if self.normalize is not None and not callable(self.normalize):
            raise TypeError("provider normalize must be callable")
        if float(self.timeout) <= 0:
            raise ValueError("provider timeout must be positive")

    def public_contract(self) -> dict[str, Any]:
        """返回不含调用函数和密钥的稳定诊断描述。"""
        return {
            "name": self.name,
            "capabilities": list(self.capabilities),
            "requires_key": bool(self.requires_key),
            "enabled_by_default": bool(self.enabled_by_default),
            "timeout": float(self.timeout),
            "rate_limit": self.rate_limit,
            "priority": int(self.priority),
            "supported_fields": list(self.supported_fields),
            "provenance": self.provenance or self.name,
            "diagnostics": dict(self.diagnostics or {}),
        }


class ProviderRegistry:
    """按优先级管理 provider，并提供字段满足和缓存身份辅助。"""

    def __init__(self, specs: Iterable[ProviderSpec] = ()) -> None:
        self._specs: dict[str, ProviderSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ProviderSpec) -> ProviderSpec:
        if spec.name in self._specs:
            raise ValueError(f"duplicate metadata provider: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ProviderSpec | None:
        return self._specs.get(str(name or "").strip())

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def enabled_specs(
        self,
        names: Iterable[str] | None = None,
        *,
        credentials: Mapping[str, bool] | None = None,
    ) -> list[ProviderSpec]:
        requested = None if names is None else {str(name).strip() for name in names if str(name).strip()}
        credentials = credentials or {}
        selected: list[ProviderSpec] = []
        for spec in self._specs.values():
            if requested is None:
                wanted = spec.enabled_by_default
            else:
                wanted = spec.name in requested
            if not wanted:
                continue
            if spec.requires_key and not bool(credentials.get(spec.name)):
                continue
            selected.append(spec)
        return sorted(selected, key=lambda item: (int(item.priority), item.name))

    def contracts(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        specs = self.enabled_specs(names)
        return [spec.public_contract() for spec in specs]


def non_empty_fields(metadata: Mapping[str, Any] | None) -> set[str]:
    """返回 metadata 中已有的非空字段名，不读取正文或嵌套原文。"""
    return {
        str(field)
        for field, value in (metadata or {}).items()
        if value not in EMPTY_VALUES
    }


def fields_satisfied(
    metadata: Mapping[str, Any] | None,
    required_fields: Iterable[str] | None,
) -> bool:
    required = {str(field).strip() for field in (required_fields or ()) if str(field).strip()}
    if not required:
        return True
    return required.issubset(non_empty_fields(metadata))


def missing_fields(
    metadata: Mapping[str, Any] | None,
    required_fields: Iterable[str] | None,
) -> list[str]:
    required = {str(field).strip() for field in (required_fields or ()) if str(field).strip()}
    return sorted(required - non_empty_fields(metadata))


def build_provider_cache_identity(
    *,
    parse_generation: str,
    source_hash: str,
    provider_names: Iterable[str],
    credentials: Mapping[str, bool] | None = None,
    required_fields: Iterable[str] | None = None,
    hydration_version: str = "",
) -> dict[str, Any]:
    """构建可比较的缓存身份；只记录凭据是否存在，不记录密钥本身。"""
    names = sorted({str(name).strip() for name in provider_names if str(name).strip()})
    credential_presence = {
        str(name): bool(value)
        for name, value in sorted((credentials or {}).items())
        if str(name).strip()
    }
    payload = {
        "registry_version": REGISTRY_VERSION,
        "hydration_version": str(hydration_version or ""),
        "parse_generation": str(parse_generation or ""),
        "source_hash": str(source_hash or ""),
        "providers": names,
        "credential_presence": credential_presence,
        "required_fields": sorted({str(field).strip() for field in (required_fields or ()) if str(field).strip()}),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "key": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def default_provider_names(
    *,
    unpaywall_email: str = "",
    semantic_scholar_api_key: str = "",
    enable_openalex: bool = False,
    enable_arxiv: bool = False,
    enable_openreview: bool = False,
) -> tuple[str, ...]:
    """返回配置期望的 provider 集合，供 hydration 和缓存失效共用。"""
    names = ["crossref", "semantic_scholar"]
    if str(unpaywall_email or "").strip():
        names.append("unpaywall")
    if enable_openalex:
        names.append("openalex")
    if enable_arxiv:
        names.append("arxiv")
    if enable_openreview:
        names.append("openreview")
    return tuple(names)


__all__ = [
    "ProviderRegistry",
    "ProviderSpec",
    "REGISTRY_VERSION",
    "build_provider_cache_identity",
    "default_provider_names",
    "fields_satisfied",
    "missing_fields",
    "non_empty_fields",
]
