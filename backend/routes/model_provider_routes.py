from datetime import datetime
import os
import re
import threading
from typing import Any, List, Optional
from urllib.parse import unquote, urlparse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from models.provider_registry import PROVIDER_CONFIG
from models.rerank_registry import RERANK_PROVIDERS
from models.dynamic_store import (
    load_dynamic_providers,
    save_dynamic_providers,
    load_dynamic_models,
    save_dynamic_models,
)
from models.model_detector import infer_model_tags, is_embedding_model, is_rerank_model, NOT_SUPPORTED_REGEX
from models.api_key_selector import select_api_key
from runtime_mode import runtime
from services.ocr_service import validate_external_ocr_service_url
from services.reasoning_effort_service import reasoning_options_for_frontend
from services.provider_auth import (
    build_api_key_headers,
    normalize_api_key_header,
    normalize_api_key_prefix,
)


router = APIRouter()

# ``load -> mutate -> save`` must be one critical section. Atomic files protect
# readers from partial JSON, while this lock prevents two requests in this
# process from silently dropping each other's settings change.
_DYNAMIC_CONFIG_LOCK = threading.RLock()
# Provider ID 需要保持稳定、适合作为配置键；模型 ID 允许供应商常见的
# ``org/model``、``provider:model`` 和版本后缀，但仍拒绝路径穿越字符。
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@-]{0,255}$")
_ALLOWED_CUSTOM_PROVIDER_TYPES = {
    "openai", "anthropic", "gemini", "ollama", "cohere", "jina", "custom",
}
_ALLOWED_CUSTOM_MODEL_TYPES = {"chat", "embedding", "rerank", "image", "vision"}
_MODEL_METADATA_FIELDS = {
    "base_url",
    "model_name",
    "embedding_endpoint",
    "rerank_endpoint",
    "max_tokens",
    "context_window",
    "dimension",
    "description",
    "price",
    "reasoning_mode",
    "reasoning_options",
    "reasoning_default",
    "reasoning_always_enabled",
    "reasoning_off_control",
    "reasoning_on_control",
}
_REASONING_MODES = {
    "openai_effort", "anthropic_adaptive", "anthropic_budget", "gemini_level",
    "gemini_budget", "qwen_budget", "thinking_toggle", "ollama_think", "fixed",
}
_REASONING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
_REASONING_OFF_CONTROLS = {
    "reasoning_effort_none", "thinking_disabled", "enable_thinking_false",
    "gemini_budget_zero", "ollama_think_false",
}
_REASONING_ON_CONTROLS = {
    "thinking_enabled", "thinking_adaptive", "enable_thinking_true",
    "reasoning_split_true", "provider_default",
}


_LATEST_MODEL_IDS = {
    "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.7-flash",
    "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp",
    "kimi-k3", "kimi-k2.7-code", "kimi-k2.6",
    "glm-5.2", "MiniMax-M3",
    "mimo-v2.5-pro", "mimo-v2.5",
    "claude-opus-5", "claude-fable-5", "claude-sonnet-5",
    "gemini-3.6-flash", "gemini-3.1-pro-preview",
    "grok-4.5",
    "doubao-seed-evolving",
    "gemini-embedding-2-preview",
}


def _pick_first(*values):
    """返回首个非空值。"""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        return value
    return None


def _allow_private_provider_urls() -> bool:
    """Local desktop providers are a user-selected capability, not server SSRF.

    A remote/server deployment rejects private targets unless its operator has
    made the exceptional opt-in explicit. Desktop mode keeps Ollama/LM Studio
    working without weakening a network-exposed backend.
    """
    value = os.environ.get("CHATPDF_ALLOW_PRIVATE_PROVIDER_URLS", "")
    return runtime.is_desktop or value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_identifier(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} 只能包含字母、数字、点、下划线、连字符和冒号，且长度不超过 128")
    return normalized


def _validate_model_id(value: str, *, field_name: str = "model_id") -> str:
    normalized = str(value or "").strip()
    if not _MODEL_ID_RE.fullmatch(normalized) or ".." in normalized:
        raise ValueError(
            f"{field_name} 只能包含字母、数字、点、斜杠、冒号、下划线、连字符和 @，且长度不超过 256"
        )
    return normalized


def _validate_provider_url(value: str, *, field_name: str) -> str:
    """Validate a provider origin before it can receive an API credential."""
    try:
        safe_url = validate_external_ocr_service_url(
            str(value or "").strip(),
            service_name=field_name,
            allow_private=_allow_private_provider_urls(),
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    parsed = urlparse(safe_url)
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} 不允许包含查询参数或片段")
    return safe_url.rstrip("/")


def _validate_relative_endpoint(value: str | None, *, field_name: str) -> str:
    """Allow API paths but never let a path switch the configured origin."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) > 512:
        raise ValueError(f"{field_name} 过长")
    parsed = urlparse(raw)
    decoded_path = unquote(parsed.path or "")
    if (
        parsed.scheme
        or parsed.netloc
        or raw.startswith("//")
        or "\\" in decoded_path
        or any(part in {"", ".", ".."} for part in decoded_path.split("/") if part in {".", ".."})
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{field_name} 必须是站内相对路径")
    return "/" + decoded_path.lstrip("/")


def _validate_optional_model_url(value: Any, *, field_name: str, allow_relative: bool = False) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if allow_relative and not parsed.scheme and not parsed.netloc:
        return _validate_relative_endpoint(raw, field_name=field_name)
    return _validate_provider_url(raw, field_name=field_name)


def _normalize_model_metadata(metadata: dict | None) -> dict:
    """将前端 metadata 归一化为后端调用链使用的字段格式。"""
    raw = dict(metadata or {})
    # Dynamic metadata becomes executable endpoint configuration later in the
    # embedding/rerank path. Persist only fields the backend actually consumes
    # instead of keeping arbitrary opaque settings around indefinitely.
    normalized = {}
    field_aliases = {
        "base_url": ["base_url", "baseUrl", "api_host", "apiHost"],
        "model_name": ["model_name", "modelName"],
        "embedding_endpoint": ["embedding_endpoint", "embeddingEndpoint"],
        "rerank_endpoint": ["rerank_endpoint", "rerankEndpoint"],
        "max_tokens": ["max_tokens", "maxTokens"],
        "context_window": ["context_window", "contextWindow"],
        "dimension": ["dimension"],
        "description": ["description"],
        "price": ["price"],
        "reasoning_mode": ["reasoning_mode", "reasoningMode", "thinking_mode", "thinkingMode"],
        "reasoning_options": ["reasoning_options", "reasoningOptions", "thinking_levels", "thinkingLevels"],
        "reasoning_default": ["reasoning_default", "reasoningDefault", "thinking_default", "thinkingDefault"],
        "reasoning_always_enabled": ["reasoning_always_enabled", "reasoningAlwaysEnabled", "always_enabled", "alwaysEnabled"],
        "reasoning_off_control": ["reasoning_off_control", "reasoningOffControl", "thinking_off_control", "thinkingOffControl"],
        "reasoning_on_control": ["reasoning_on_control", "reasoningOnControl", "thinking_on_control", "thinkingOnControl"],
    }
    for target_field, aliases in field_aliases.items():
        picked = _pick_first(*(raw.get(alias) for alias in aliases))
        if picked is not None:
            normalized[target_field] = picked

    mode = str(normalized.get("reasoning_mode") or "").strip().lower()
    if mode:
        if mode not in _REASONING_MODES:
            raise ValueError("reasoning_mode 不受支持")
        normalized["reasoning_mode"] = mode

    raw_options = normalized.get("reasoning_options")
    if raw_options not in (None, ""):
        if isinstance(raw_options, str):
            raw_options = re.split(r"[,|/\s]+", raw_options)
        if not isinstance(raw_options, (list, tuple, set)):
            raise ValueError("reasoning_options 必须是档位数组")
        options = []
        for option in raw_options:
            value = str(option or "").strip().lower()
            if value not in _REASONING_LEVELS:
                raise ValueError(f"不支持的思考档位: {value}")
            if value not in options:
                options.append(value)
        normalized["reasoning_options"] = options

    default = str(normalized.get("reasoning_default") or "").strip().lower()
    if default:
        if default not in _REASONING_LEVELS:
            raise ValueError("reasoning_default 不受支持")
        normalized["reasoning_default"] = default

    if "reasoning_always_enabled" in normalized:
        normalized["reasoning_always_enabled"] = bool(normalized["reasoning_always_enabled"])

    off_control = str(normalized.get("reasoning_off_control") or "").strip().lower()
    if off_control:
        if off_control not in _REASONING_OFF_CONTROLS:
            raise ValueError("reasoning_off_control 不受支持")
        normalized["reasoning_off_control"] = off_control
    on_control = str(normalized.get("reasoning_on_control") or "").strip().lower()
    if on_control:
        if on_control not in _REASONING_ON_CONTROLS:
            raise ValueError("reasoning_on_control 不受支持")
        normalized["reasoning_on_control"] = on_control
    return {key: value for key, value in normalized.items() if key in _MODEL_METADATA_FIELDS}


def _get_provider_type(provider_id: str) -> str:
    """获取 provider 的后端类型（如 silicon -> openai）。"""
    merged_providers = {**PROVIDER_CONFIG, **load_dynamic_providers()}
    return merged_providers.get(provider_id, {}).get("type", provider_id)


def _build_public_model_details(model_id: str, name: str, model_config: dict | None = None) -> dict:
    """Keep the legacy ``models`` name map stable while exposing typed tags."""
    config = model_config if isinstance(model_config, dict) else {}
    model_type = str(config.get("type") or "").strip().lower()
    if not model_type:
        model_type = "rerank" if is_rerank_model(model_id) else "embedding" if is_embedding_model(model_id) else "chat"

    configured_tags = config.get("tags") if isinstance(config.get("tags"), list) else []
    tags = [str(tag).strip() for tag in configured_tags if str(tag).strip()]
    tags.extend(infer_model_tags(model_id, model_type))
    if model_id in _LATEST_MODEL_IDS:
        tags.append("latest")
    return {
        "id": model_id,
        "name": name,
        "type": model_type,
        "tags": list(dict.fromkeys(tags)),
    }


@router.get("/models")
async def get_models():
    """获取可用模型/Provider列表（含静态+动态），按 provider 分组

    返回结构：
    {
        "provider_id": {
            "name": "Provider名称",
            "endpoint": "...",
            "type": "openai",
            "models": {
                "model_id": "模型显示名称",
                ...
            }
        },
        ...
    }

    前端通过 availableModels[apiProvider]?.models 访问。
    """
    from models.model_registry import EMBEDDING_MODELS
    from urllib.parse import urlparse

    merged_providers = {**PROVIDER_CONFIG, **load_dynamic_providers()}
    merged_models = {**EMBEDDING_MODELS, **load_dynamic_models()}

    # 与前端 systemModels.ts 对齐的推荐 chat 模型目录。
    # 历史模型只保留在 DefaultsContext 的迁移映射中，不应再作为新配置默认展示。
    CHAT_MODELS = {
        "openai": {
            "gpt-5.6": "GPT-5.6",
            "gpt-5.6-sol": "GPT-5.6 Sol",
            "gpt-5.6-terra": "GPT-5.6 Terra",
            "gpt-5.6-luna": "GPT-5.6 Luna",
            "gpt-5.5": "GPT-5.5",
            "gpt-5.5-pro": "GPT-5.5 Pro",
            "gpt-5.4-mini": "GPT-5.4 mini",
            "gpt-5.4-nano": "GPT-5.4 nano",
            "gpt-4.1": "GPT-4.1",
            "gpt-4.1-mini": "GPT-4.1 mini",
            "gpt-4.1-nano": "GPT-4.1 nano",
            "o3": "OpenAI o3",
            "o4-mini": "OpenAI o4-mini",
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o mini",
        },
        "aliyun": {
            "qwen3.7-max": "Qwen3.7-Max",
            "qwen3.7-plus": "Qwen3.7-Plus",
            "qwen3.7-flash": "Qwen3.7-Flash",
            "qwen3.6-flash": "Qwen3.6-Flash",
        },
        "deepseek": {
            "deepseek-v4-flash": "DeepSeek V4 Flash",
            "deepseek-v4-pro": "DeepSeek V4 Pro",
            "deepseek-v4-flash-vision-exp": "DeepSeek V4 Flash Vision Exp",
        },
        "moonshot": {
            "kimi-k3": "Kimi K3",
            "kimi-k2.7-code": "Kimi K2.7 Code",
            "kimi-k2.6": "Kimi K2.6",
        },
        "zhipu": {
            "glm-5.2": "GLM-5.2",
            "glm-5.1": "GLM-5.1",
        },
        "minimax": {
            "MiniMax-M3": "MiniMax M3",
        },
        "xiaomi": {
            "mimo-v2.5-pro": "MiMo V2.5 Pro",
            "mimo-v2.5": "MiMo V2.5",
        },
        "silicon": {
            "deepseek-ai/DeepSeek-V4-Flash": "DeepSeek V4 Flash (SiliconFlow)",
            "Pro/zai-org/GLM-4.7": "GLM-4.7 Pro (SiliconFlow)",
            "deepseek-ai/DeepSeek-R1": "DeepSeek R1 (SiliconFlow)",
            "Qwen/Qwen3-32B": "Qwen3-32B (SiliconFlow)",
        },
        "anthropic": {
            "claude-opus-5": "Claude Opus 5",
            "claude-fable-5": "Claude Fable 5",
            "claude-opus-4-8": "Claude Opus 4.8",
            "claude-sonnet-5": "Claude Sonnet 5",
            "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
        },
        "gemini": {
            "gemini-3.6-flash": "Gemini 3.6 Flash",
            "gemini-3.5-flash": "Gemini 3.5 Flash",
            "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
            "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
            "gemini-2.5-pro": "Gemini 2.5 Pro",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
        },
        "grok": {
            "grok-4.5": "Grok 4.5",
            "grok-4.3": "Grok 4.3",
            "grok-build-0.1": "Grok Build 0.1",
        },
        "doubao": {
            "doubao-seed-evolving": "Doubao Seed Evolving",
            "doubao-seed-2-1-pro-260628": "Doubao Seed 2.1 Pro",
            "doubao-seed-2-1-turbo-260628": "Doubao Seed 2.1 Turbo",
        },
    }

    def _extract_domain(url: str) -> str:
        """从 URL 中提取域名，用于匹配 provider"""
        if not url:
            return ""
        parsed = urlparse(url)
        return parsed.netloc or ""

    # 预计算每个 provider 的域名，用于通过 base_url 区分同 type 的不同服务商
    provider_domains = {}
    for pid, pconfig in merged_providers.items():
        endpoint = pconfig.get("endpoint", "")
        provider_domains[pid] = _extract_domain(endpoint)

    result = {}
    for provider_id, provider_config in merged_providers.items():
        # 收集该 provider 下的 embedding/rerank 模型
        provider_models = {}
        provider_domain = provider_domains.get(provider_id, "")
        provider_type = provider_config.get("type", provider_id)

        for model_id, model_config in merged_models.items():
            model_provider_type = model_config.get("provider", "")
            model_provider_id = model_config.get("provider_id") or model_config.get("providerId")

            # 本地模型只归属于 local provider
            if model_provider_type == "local":
                if provider_id == "local":
                    provider_models[model_id] = model_config.get("name", model_id)
                continue

            # 新格式动态模型：显式保存 provider_id，优先按 provider_id 归属
            if model_provider_id:
                if model_provider_id == provider_id:
                    provider_models[model_id] = model_config.get("name", model_id)
                continue

            # 非本地模型：先匹配 provider type，再通过 base_url 域名区分
            if model_provider_type not in {provider_type, provider_id}:
                continue

            model_base_url = model_config.get("base_url", "")
            model_domain = _extract_domain(model_base_url)

            # 通过域名匹配区分同 type 的不同服务商
            if provider_domain and model_domain and provider_domain == model_domain:
                provider_models[model_id] = model_config.get("name", model_id)
            elif not model_domain and model_provider_type == provider_id:
                # 兼容旧动态数据：未配置 base_url 时按 providerId 直接归属
                provider_models[model_id] = model_config.get("name", model_id)

        # 合并 chat 模型
        chat_models = CHAT_MODELS.get(provider_id, {})
        provider_models.update(chat_models)
        model_details = {
            model_id: _build_public_model_details(
                model_id,
                model_name,
                merged_models.get(model_id),
            )
            for model_id, model_name in provider_models.items()
        }

        result[provider_id] = {
            **provider_config,
            "models": provider_models,
            # ``models`` 仍保持旧版 id -> name 映射，避免 ChatPDF 现有选择器
            # 破坏性变更；模型服务可使用 modelDetails 读取 tags/type。
            "modelDetails": model_details,
        }

    return result


@router.get("/rerank/providers")
async def get_rerank_providers():
    """获取支持的重排提供商及默认模型"""
    return RERANK_PROVIDERS


class ProviderTestRequest(BaseModel):
    providerId: str = Field(min_length=1, max_length=128)
    apiKey: str = Field(max_length=32_768)
    apiHost: str = Field(max_length=2_048)
    fetchModelsEndpoint: str | None = Field(default=None, max_length=512)
    modelId: str | None = Field(default=None, max_length=256)
    modelType: str = Field(default="chat", max_length=32)
    chatEndpoint: str | None = Field(default=None, max_length=512)
    providerType: str | None = Field(default=None, max_length=64)
    apiKeyHeader: str | None = Field(default=None, max_length=128)
    apiKeyPrefix: str | None = Field(default=None, max_length=64)

    @field_validator("providerId")
    @classmethod
    def _validate_provider_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="providerId")

    @field_validator("fetchModelsEndpoint")
    @classmethod
    def _validate_models_endpoint(cls, value: str | None) -> str | None:
        return _validate_relative_endpoint(value, field_name="fetchModelsEndpoint") or None

    @field_validator("modelId")
    @classmethod
    def _validate_model_id_field(cls, value: str | None) -> str | None:
        return _validate_model_id(value, field_name="modelId") if value else None

    @field_validator("chatEndpoint")
    @classmethod
    def _validate_chat_endpoint(cls, value: str | None) -> str | None:
        return _validate_relative_endpoint(value, field_name="chatEndpoint") or None

    @field_validator("modelType")
    @classmethod
    def _validate_chat_model_type(cls, value: str) -> str:
        normalized = str(value or "chat").strip().lower()
        if normalized != "chat":
            raise ValueError("Provider 连接测试目前只支持 chat 模型")
        return normalized

    @field_validator("apiKeyHeader")
    @classmethod
    def _validate_api_key_header(cls, value: str | None) -> str | None:
        return normalize_api_key_header(value) if value is not None and str(value).strip() else None

    @field_validator("apiKeyPrefix")
    @classmethod
    def _validate_api_key_prefix(cls, value: str | None) -> str | None:
        return normalize_api_key_prefix(value) if value is not None else None


def _build_endpoint(base: str, path: str | None) -> str:
    """Join a validated relative endpoint to a provider base URL.

    Users commonly enter either ``https://host/v1`` with
    ``/chat/completions`` or ``https://host`` with
    ``/v1/chat/completions``. A plain string join turns the former into
    ``/v1/v1/...`` and duplicates a full endpoint pasted into API Host.
    """
    if not base:
        return path or ""
    base_clean = str(base).rstrip('/')
    if not path:
        return base_clean

    path_clean = "/" + str(path).lstrip('/')
    try:
        parsed = urlparse(base_clean)
    except ValueError:
        parsed = None

    if parsed and parsed.scheme and parsed.netloc:
        base_path = (parsed.path or "").rstrip('/')
        base_parts = [part for part in base_path.split('/') if part]
        path_parts = [part for part in path_clean.split('/') if part]
        # API Host may already be the exact endpoint (or a versioned prefix
        # ending in it). Do not append the configured path a second time.
        if base_path and (base_path == path_clean or base_path.endswith(path_clean)):
            return base_clean
        # The endpoint may carry the same version prefix as the host. Replace
        # the path instead of appending it to avoid /v1/v1/... URLs.
        if base_path and (path_clean == base_path or path_clean.startswith(base_path + '/')):
            return parsed._replace(path=path_clean).geturl().rstrip('/')
        # 网关可能使用 ``/api/v1`` 这样的前缀，而配置的 endpoint 写成
        # ``/v1/models``。保留网关前缀，只去掉重复的版本段。
        version_segments = {"v1", "v1beta", "v2", "beta", "openai", "compat", "compatible"}
        if (
            base_parts
            and path_parts
            and base_parts[-1].lower() in version_segments
            and path_parts[0].lower() == base_parts[-1].lower()
        ):
            suffix = "/" + "/".join(path_parts[1:])
            return parsed._replace(path=f"{base_path}{suffix}").geturl().rstrip('/')

    return f"{base_clean}{path_clean}"


def _normalize_api_host(provider_id: str, api_host: str | None) -> str:
    """
    把传入的 api_host 还原成可拼接 /models 的 base url。
    只去掉已知的 chat/embedding 尾部路径（如 /chat/completions），保留有意义的路径前缀（如 /api/v3）。
    """
    host = (api_host or "").strip()
    if not host:
        host = PROVIDER_CONFIG.get(provider_id, {}).get("endpoint", "")

    if not host:
        return host

    # 去掉已知的 API 尾部路径，保留 base path
    known_suffixes = [
        "/chat/completions",
        "/completions",
        "/embeddings",
        "/v1/chat/completions",
    ]
    for suffix in known_suffixes:
        if host.endswith(suffix):
            host = host[: -len(suffix)]
            break

    return host.rstrip("/")


def _resolve_safe_provider_host(provider_id: str, api_host: str | None, *, field_name: str) -> str:
    host = _normalize_api_host(provider_id, api_host)
    return _validate_provider_url(host, field_name=field_name)


def _provider_auth_values(
    provider_id: str,
    *,
    provider_type: str | None = None,
    api_key_header: str | None = None,
    api_key_prefix: str | None = None,
) -> tuple[str | None, str | None, str]:
    """Resolve persisted non-secret auth metadata with request overrides."""

    dynamic = load_dynamic_providers().get(provider_id) or {}
    configured_type = str(provider_type or dynamic.get("type") or _provider_protocol(provider_id)).strip().lower()
    header = api_key_header if api_key_header is not None else dynamic.get("api_key_header")
    prefix = api_key_prefix if api_key_prefix is not None else dynamic.get("api_key_prefix")
    # Validate here so malformed persisted configuration fails as a normal
    # provider error instead of being handed to the HTTP client.
    if header is not None:
        header = normalize_api_key_header(header)
    if prefix is not None:
        prefix = normalize_api_key_prefix(prefix)
    return header, prefix, configured_type


def _model_list_endpoint_candidates(api_host: str, endpoints: List[str]) -> list[str]:
    """Build stable model-list candidates without duplicating version paths."""

    base = str(api_host or "").rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/messages", "/embeddings", "/rerank"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    tail = base.rsplit("/", 1)[-1].lower()
    version_segments = {"v1", "v1beta", "v2", "beta", "openai", "compat", "compatible"}
    requested = [str(item or "").strip() for item in endpoints if str(item or "").strip()]
    if not requested:
        requested = ["/models"]
    candidates: list[str] = []
    for raw_path in requested:
        path = "/" + raw_path.lstrip("/")
        path_parts = [part for part in path.split("/") if part]
        path_has_version = bool(path_parts and path_parts[0].lower() in version_segments)
        # 已经带 /v1、/v1beta 等版本段时直接使用；只有裸 /models
        # 才补一个版本候选，避免产生 /v1/v1/models 的无效请求。
        if path_has_version or tail in version_segments:
            candidates.append(_build_endpoint(base, path))
        else:
            candidates.append(_build_endpoint(base, f"/v1{path}"))
            candidates.append(_build_endpoint(base, path))
    # A user may pass the exact model list URL. Keep it first and de-duplicate.
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _extract_model_items(payload: Any) -> list[dict[str, Any]]:
    """Normalize common OpenAI-compatible model-list response shapes."""

    if isinstance(payload, str):
        value = payload.strip()
        return [{"id": value}] if value else []
    if isinstance(payload, list):
        items: list[dict[str, Any]] = []
        for item in payload:
            items.extend(_extract_model_items(item))
        return items
    if not isinstance(payload, dict):
        return []
    for key in ("data", "models", "items", "results"):
        if key in payload:
            nested = _extract_model_items(payload[key])
            if nested:
                return nested
    model_id = payload.get("id") or payload.get("name")
    if isinstance(model_id, str) and model_id.strip():
        item = {"id": model_id.strip()}
        if payload.get("owned_by") is not None:
            item["owned_by"] = payload.get("owned_by")
        return [item]
    return []


def _next_model_list_cursor(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    has_more = payload.get("has_more")
    if has_more is False:
        return ""
    for key in ("last_id", "next_cursor", "next_page_token", "cursor"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "" if has_more is not True else "__missing_cursor__"


def _dedupe_model_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并分页结果时按模型 ID 去重，保留首次出现的完整元数据。"""

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or item.get("name") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        normalized = dict(item)
        normalized["id"] = model_id
        result.append(normalized)
    return result


async def _fetch_models_with_fallback(
    api_host: str,
    api_key: str,
    endpoints: List[str],
    *,
    provider_id: str = "",
    provider_type: str | None = None,
    api_key_header: str | None = None,
    api_key_prefix: str | None = None,
):
    api_host = _validate_provider_url(api_host, field_name="模型服务 API 地址")
    # 从 API Key 池中随机选择一个有效 Key（支持逗号分隔的多 Key 轮换）
    actual_key = select_api_key(api_key) if api_key else None
    if not actual_key:
        return None, "API Key 池为空，无法发送请求"
    resolved_header, resolved_prefix, protocol = _provider_auth_values(
        provider_id,
        provider_type=provider_type,
        api_key_header=api_key_header,
        api_key_prefix=api_key_prefix,
    )
    headers = build_api_key_headers(
        actual_key,
        provider_type=protocol,
        api_key_header=resolved_header,
        api_key_prefix=resolved_prefix,
        extra_headers={"anthropic-version": "2023-06-01"} if protocol == "anthropic" else None,
    )
    last_error = None

    for url in _model_list_endpoint_candidates(api_host, endpoints):
        last_error = None
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                cursor = ""
                all_items: list[dict[str, Any]] = []
                first_payload: Any = None
                for page in range(50):
                    request_url = url
                    if cursor:
                        from urllib.parse import urlencode
                        separator = "&" if "?" in request_url else "?"
                        request_url = f"{request_url}{separator}{urlencode({'limit': 1000, 'after_id': cursor})}"
                    response = await client.get(request_url, headers=headers)
                    body = response.text[:8192]
                    if response.status_code < 200 or response.status_code >= 300:
                        # 401/403 are credential failures; do not hide them
                        # behind a later endpoint that will produce a 404.
                        if response.status_code in (401, 403):
                            return None, f"API Key 无效或格式错误（HTTP {response.status_code}）：{body[:500]}"
                        last_error = f"HTTP {response.status_code}：{body[:300]}"
                        break
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        last_error = f"模型列表响应不是合法 JSON：{exc}"
                        break
                    if first_payload is None:
                        first_payload = payload
                    all_items.extend(_extract_model_items(payload))
                    next_cursor = _next_model_list_cursor(payload)
                    if next_cursor in ("", "__missing_cursor__"):
                        if next_cursor == "__missing_cursor__":
                            last_error = "模型列表分页缺少下一页游标"
                        break
                    if page >= 49:
                        last_error = "模型列表分页超过 50 页，结果可能不完整"
                        break
                    cursor = next_cursor
                else:
                    continue
                if first_payload is not None and last_error is None:
                    # 与 cursor-byok 的模型列表合同保持一致：HTTP 2xx 但没有
                    # 可用模型并不算“连接成功”。继续尝试下一个候选地址，
                    # 避免把空响应误报成可用 Provider。
                    all_items = _dedupe_model_items(all_items)
                    if not all_items:
                        last_error = "模型列表响应中没有可用模型"
                    else:
                        normalized = dict(first_payload) if isinstance(first_payload, dict) else {}
                        normalized["data"] = all_items
                        normalized["models"] = all_items
                        return normalized, url
        except Exception as e:
            last_error = str(e)
            continue
    return None, last_error


def _provider_protocol(provider_id: str, explicit_type: str | None = None) -> str:
    merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
    configured = str((merged.get(provider_id) or {}).get("type") or "").strip().lower()
    explicit = str(explicit_type or "").strip().lower()
    # 静态 Provider 的 id（silicon、deepseek 等）不是协议名，优先采用
    # registry 中声明的 openai/anthropic/gemini；动态 Provider 才用显式协议。
    return configured or explicit or str(provider_id).strip().lower()


def _resolve_chat_test_endpoint(request: ProviderTestRequest, api_host: str) -> str:
    protocol = _provider_protocol(request.providerId, request.providerType)
    explicit = request.chatEndpoint
    if explicit:
        return _build_endpoint(api_host, explicit)
    dynamic = load_dynamic_providers().get(request.providerId) or {}
    configured = str(dynamic.get("chat_endpoint") or "").strip()
    if configured:
        return _build_endpoint(api_host, _validate_relative_endpoint(configured, field_name="chatEndpoint"))
    static_endpoint = str((PROVIDER_CONFIG.get(request.providerId) or {}).get("endpoint") or "").strip()
    if static_endpoint.endswith("/chat/completions"):
        return static_endpoint
    return _build_endpoint(api_host, "/messages" if protocol == "anthropic" else "/chat/completions")


async def _probe_openai_compatible_chat(
    *,
    endpoint: str,
    api_key: str,
    model_id: str,
    provider_type: str | None = None,
    api_key_header: str | None = None,
    api_key_prefix: str | None = None,
) -> None:
    actual_key = select_api_key(api_key) if api_key else ""
    if not actual_key:
        raise ValueError("API Key 池为空，无法执行 Chat 模型测试")
    headers = build_api_key_headers(
        actual_key,
        provider_type=provider_type,
        api_key_header=api_key_header,
        api_key_prefix=api_key_prefix,
    )
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with OK only."}],
        "stream": False,
        "max_tokens": 8,
    }
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
    if response.status_code < 200 or response.status_code >= 300:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Chat 模型调用失败（HTTP {response.status_code}）：{response.text[:500]}",
        )
    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    content = ((choices or [{}])[0].get("message") or {}).get("content") if choices else None
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Chat 接口返回成功，但响应中没有 choices.message.content")


async def _probe_anthropic_chat(
    *,
    endpoint: str,
    api_key: str,
    model_id: str,
    api_key_header: str | None = None,
    api_key_prefix: str | None = None,
) -> None:
    """使用 Anthropic Messages 协议验证真实 Chat 调用。"""

    actual_key = select_api_key(api_key) if api_key else ""
    if not actual_key:
        raise ValueError("API Key 池为空，无法执行 Chat 模型测试")
    headers = build_api_key_headers(
        actual_key,
        provider_type="anthropic",
        api_key_header=api_key_header,
        api_key_prefix=api_key_prefix,
        extra_headers={"anthropic-version": "2023-06-01"},
    )
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with OK only."}],
        "max_tokens": 8,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
    if response.status_code < 200 or response.status_code >= 300:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Anthropic Messages 调用失败（HTTP {response.status_code}）：{response.text[:500]}",
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError(f"Anthropic Messages 返回了无效 JSON：{exc}") from exc
    blocks = data.get("content") if isinstance(data, dict) else None
    text = "".join(
        str(block.get("text") or "")
        for block in (blocks or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if not text.strip():
        raise ValueError("Anthropic Messages 返回成功，但响应中没有文本 content")


@router.post("/api/providers/test")
async def test_provider_connection(request: ProviderTestRequest):
    """测试 Provider 的真实模型能力，而不是只判断 /models 是否可达。"""
    from time import time
    start_time = time()
    try:
        if request.providerId == 'local':
            # 本地模型无需网络请求，但仍记录延迟
            latency = int((time() - start_time) * 1000)
            return {
                "success": True,
                "message": "本地模型无需连接测试",
                "availableModels": 2,
                "latency": latency
            }

        api_host = _resolve_safe_provider_host(
            request.providerId,
            request.apiHost,
            field_name="模型服务 API 地址",
        )
        protocol = _provider_protocol(request.providerId, request.providerType)
        endpoints = [request.fetchModelsEndpoint or "/models", "/v1/models", "/models"]
        data, last_error = await _fetch_models_with_fallback(
            api_host,
            request.apiKey,
            endpoints,
            provider_id=request.providerId,
            provider_type=protocol,
            api_key_header=request.apiKeyHeader,
            api_key_prefix=request.apiKeyPrefix,
        )

        model_count = len(data.get('data', [])) if isinstance(data, dict) and isinstance(data.get('data'), list) else 0
        listed_model_ids = [
            str(item.get("id") or "").strip()
            for item in (data.get("data") if isinstance(data, dict) else [])
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        ]

        # /models 不是所有兼容服务都提供。只要调用方明确给了模型 ID，
        # 仍继续验证真实 Chat endpoint，避免把“没有模型列表”误判为不可用。
        model_id = request.modelId or (listed_model_ids[0] if listed_model_ids else "")
        if not model_id:
            if last_error and ("401" in last_error or "403" in last_error or "API Key 无效" in last_error):
                return {"success": False, "message": last_error}
            return {
                "success": False,
                "message": f"无法验证 Chat 模型：请先填写模型 ID（模型列表请求：{last_error or '无响应'}）",
                "availableModels": model_count,
            }

        if protocol not in {"openai", "custom", "anthropic"}:
            return {
                "success": False,
                "message": f"已连接到 Provider，但暂不支持自动验证 {protocol} 协议的 Chat 模型",
                "availableModels": model_count,
                "verifiedModel": model_id,
            }

        chat_endpoint = _resolve_chat_test_endpoint(request, api_host)
        if protocol == "anthropic":
            await _probe_anthropic_chat(
                endpoint=chat_endpoint,
                api_key=request.apiKey,
                model_id=model_id,
                api_key_header=request.apiKeyHeader,
                api_key_prefix=request.apiKeyPrefix,
            )
        else:
            await _probe_openai_compatible_chat(
                endpoint=chat_endpoint,
                api_key=request.apiKey,
                model_id=model_id,
                provider_type=protocol,
                api_key_header=request.apiKeyHeader,
                api_key_prefix=request.apiKeyPrefix,
            )
        latency = int((time() - start_time) * 1000)
        return {
            "success": True,
            "message": "Chat 模型调用成功",
            "availableModels": model_count,
            "verifiedModel": model_id,
            "chatEndpoint": chat_endpoint,
            "latency": latency,
        }

    except httpx.ConnectError:
        # 失败时不返回 latency
        return {"success": False, "message": "无法连接到API服务器，请检查网络或API地址"}
    except httpx.TimeoutException:
        # 失败时不返回 latency
        return {"success": False, "message": "连接超时，请稍后重试"}
    except Exception as e:
        # 失败时不返回 latency
        return {"success": False, "message": f"测试失败：{str(e)}"}


class ModelFetchRequest(BaseModel):
    providerId: str = Field(min_length=1, max_length=128)
    apiKey: str = Field(max_length=32_768)
    apiHost: str = Field(max_length=2_048)
    fetchModelsEndpoint: str | None = Field(default=None, max_length=512)
    providerType: str | None = Field(default=None, max_length=64)  # "openai" | "anthropic" | "gemini" | ...
    apiKeyHeader: str | None = Field(default=None, max_length=128)
    apiKeyPrefix: str | None = Field(default=None, max_length=64)

    @field_validator("providerId")
    @classmethod
    def _validate_provider_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="providerId")

    @field_validator("fetchModelsEndpoint")
    @classmethod
    def _validate_models_endpoint(cls, value: str | None) -> str | None:
        return _validate_relative_endpoint(value, field_name="fetchModelsEndpoint") or None

    @field_validator("apiKeyHeader")
    @classmethod
    def _validate_api_key_header(cls, value: str | None) -> str | None:
        return normalize_api_key_header(value) if value is not None and str(value).strip() else None

    @field_validator("apiKeyPrefix")
    @classmethod
    def _validate_api_key_prefix(cls, value: str | None) -> str | None:
        return normalize_api_key_prefix(value) if value is not None else None


@router.post("/api/models/fetch")
async def fetch_provider_models(request: ModelFetchRequest):
    """从Provider API获取模型列表（支持动态/静态）"""
    try:
        # Anthropic Claude：使用非 OpenAI 格式 API，返回预设模型列表
        if request.providerId == 'anthropic':
            ANTHROPIC_PRESET_MODELS = [
                {"id": "claude-opus-5", "name": "Claude Opus 5", "type": "chat",
                 "metadata": {"description": "Anthropic 面向复杂 Agent、编码与企业任务的最新旗舰模型"}},
                {"id": "claude-fable-5", "name": "Claude Fable 5", "type": "chat",
                 "metadata": {"description": "Anthropic 最新 Claude 5 系列模型，400K 上下文，适合通用与创作任务"}},
                {"id": "claude-opus-4-8", "name": "Claude Opus 4.8", "type": "chat",
                 "metadata": {"description": "Claude Opus 最新旗舰，复杂推理与编码能力强"}},
                {"id": "claude-sonnet-5", "name": "Claude Sonnet 5", "type": "chat",
                 "metadata": {"description": "Claude Sonnet 5 均衡旗舰，适合编码、Agent 和通用推理"}},
                {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "type": "chat",
                 "metadata": {"description": "Claude 高速轻量模型，适合低延迟和成本敏感场景"}},
            ]
            return {
                "models": [
                    {
                        "id": m["id"],
                        "name": m["name"],
                        "providerId": "anthropic",
                        "type": m["type"],
                        "capabilities": [{"type": m["type"], "isUserSelected": False}],
                        "tags": infer_model_tags(m["id"], m["type"]),
                        "metadata": m["metadata"],
                        "isSystem": True,
                        "isUserAdded": False
                    }
                    for m in ANTHROPIC_PRESET_MODELS
                ],
                "providerId": "anthropic",
                "timestamp": int(datetime.now().timestamp()),
                "message": "已返回 Claude 预设模型列表（Anthropic 使用自定义 API 格式，如有新模型请手动添加）"
            }

        # Google Gemini：返回预设模型列表
        if request.providerId == 'gemini':
            GEMINI_PRESET_MODELS = [
                {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash", "type": "chat",
                 "metadata": {"description": "Google 当前最新稳定模型，兼顾速度、智能水平与 Agent/多模态任务"}},
                {"id": "gemini-3.5-flash", "name": "Gemini 3.5 Flash", "type": "chat",
                 "metadata": {"description": "Google 最新稳定 Gemini 3.5 Flash，强调速度、多模态与思考能力"}},
                {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro Preview", "type": "chat",
                 "metadata": {"description": "Gemini 3.1 Pro 预览版，适合复杂多模态和推理任务"}},
                {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash-Lite", "type": "chat",
                 "metadata": {"description": "Gemini 3.1 轻量高速版本，适合大规模低成本任务"}},
                {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "type": "chat",
                 "metadata": {"description": "Gemini 2.5 旗舰稳定版，1M 上下文，自适应思考"}},
                {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "type": "chat",
                 "metadata": {"description": "Gemini 2.5 快速均衡版，可控推理预算"}},
                {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash-Lite", "type": "chat",
                 "metadata": {"description": "Gemini 2.5 超轻量版，大规模低成本场景"}},
            ]
            return {
                "models": [
                    {
                        "id": m["id"],
                        "name": m["name"],
                        "providerId": "gemini",
                        "type": m["type"],
                        "capabilities": [{"type": m["type"], "isUserSelected": False}],
                        "tags": infer_model_tags(m["id"], m["type"]),
                        "metadata": m["metadata"],
                        "isSystem": True,
                        "isUserAdded": False
                    }
                    for m in GEMINI_PRESET_MODELS
                ],
                "providerId": "gemini",
                "timestamp": int(datetime.now().timestamp()),
                "message": "已返回 Gemini 预设模型列表（如有新模型请手动添加）"
            }

        # 其他不支持自动拉取的自定义 provider
        unsupported_providers: set = set()
        if request.providerId in unsupported_providers or (
            request.providerType and request.providerType.lower() in unsupported_providers
        ):
            return {
                "models": [],
                "providerId": request.providerId,
                "providerType": request.providerType,
                "timestamp": int(datetime.now().timestamp()),
                "message": "该提供商不支持自动拉取模型列表，请在前端手动输入模型 ID"
            }

        # 字节跳动豆包：火山引擎 Ark API 不提供 GET /models 端点，返回预设模型列表。
        # Seed-Evolving 是官方推荐的滚动别名，旧版本号仅用于兼容已有配置。
        if request.providerId == 'doubao':
            DOUBAO_PRESET_MODELS = [
                {"id": "doubao-seed-evolving", "name": "Doubao Seed Evolving", "type": "chat",
                 "metadata": {"description": "豆包当前滚动更新的 Agent/Coding 模型，一个 ID 自动获得最新版本能力"}},
                {"id": "doubao-seed-2-1-pro-260628", "name": "Doubao Seed 2.1 Pro", "type": "chat",
                 "metadata": {"description": "豆包 Seed 2.1 旗舰模型，适合复杂推理、文档和多模态 Agent 场景"}},
                {"id": "doubao-seed-2-1-turbo-260628", "name": "Doubao Seed 2.1 Turbo", "type": "chat",
                 "metadata": {"description": "豆包 Seed 2.1 高速版，兼顾推理质量与低延迟"}},
                {"id": "doubao-embedding-large-250104", "name": "Doubao Embedding Large", "type": "embedding",
                 "metadata": {"dimension": 4096, "maxTokens": 32768, "description": "豆包大尺寸嵌入模型"}},
                {"id": "doubao-embedding-250104", "name": "Doubao Embedding", "type": "embedding",
                 "metadata": {"dimension": 2048, "maxTokens": 32768, "description": "豆包标准嵌入模型"}},
            ]
            return {
                "models": [
                    {
                        "id": m["id"],
                        "name": m["name"],
                        "providerId": "doubao",
                        "type": m["type"],
                        "capabilities": [{"type": m["type"], "isUserSelected": False}],
                        "tags": infer_model_tags(m["id"], m["type"]),
                        "metadata": m["metadata"],
                        "isSystem": True,
                        "isUserAdded": False
                    }
                    for m in DOUBAO_PRESET_MODELS
                ],
                "providerId": "doubao",
                "timestamp": int(datetime.now().timestamp()),
                "message": "已返回豆包预设模型列表（火山引擎不支持动态拉取，如有新模型请手动添加）"
            }

        if request.providerId == 'local':
            return {
                "models": [
                    {
                        "id": "all-MiniLM-L6-v2",
                        "name": "MiniLM-L6-v2",
                        "providerId": "local",
                        "type": "embedding",
                        "tags": ["embedding", "free"],
                        "metadata": {"dimension": 384, "maxTokens": 256, "description": "快速通用模型"},
                        "isSystem": True,
                        "isUserAdded": False
                    },
                    {
                        "id": "paraphrase-multilingual-MiniLM-L12-v2",
                        "name": "Multilingual MiniLM-L12-v2",
                        "providerId": "local",
                        "type": "embedding",
                        "tags": ["embedding", "free"],
                        "metadata": {"dimension": 384, "maxTokens": 128, "description": "多语言支持"},
                        "isSystem": True,
                        "isUserAdded": False
                    }
                ],
                "providerId": "local",
                "timestamp": int(datetime.now().timestamp())
            }

        api_host = _resolve_safe_provider_host(
            request.providerId,
            request.apiHost,
            field_name="模型服务 API 地址",
        )
        endpoints = [request.fetchModelsEndpoint or "/models", "/v1/models", "/models"]
        protocol = _provider_protocol(request.providerId, request.providerType)
        data, last_error = await _fetch_models_with_fallback(
            api_host,
            request.apiKey,
            endpoints,
            provider_id=request.providerId,
            provider_type=protocol,
            api_key_header=request.apiKeyHeader,
            api_key_prefix=request.apiKeyPrefix,
        )

        if data is None:
            return {
                "models": [],
                "providerId": request.providerId,
                "timestamp": int(datetime.now().timestamp()),
                "message": f"获取模型失败: {last_error or '无响应'}，可在前端手动添加模型 ID"
            }

        models = []
        if 'data' in data and isinstance(data['data'], list):
            for item in data['data']:
                model_id = item.get('id', '')
                # 过滤不支持的模型（TTS、语音、审核等）
                if NOT_SUPPORTED_REGEX.search(model_id):
                    continue
                model_type = _detect_model_type(model_id)
                # 推断模型标签（如 free、vision、reasoning 等）
                tags = infer_model_tags(model_id, model_type)
                model = {
                    "id": model_id,
                    "name": model_id,
                    "providerId": request.providerId,
                    "type": model_type,
                    # 模型能力声明，默认由正则检测推断，isUserSelected=False 表示非用户手动指定
                    "capabilities": [{"type": model_type, "isUserSelected": False}],
                    "tags": tags,
                    "metadata": _infer_model_metadata(model_id, model_type),
                    "isSystem": False,
                    "isUserAdded": False
                }
                if 'owned_by' in item:
                    model["metadata"]["description"] = f"Owned by: {item['owned_by']}"
                models.append(model)

        return {
            "models": models,
            "providerId": request.providerId,
            "timestamp": int(datetime.now().timestamp()),
            "message": None
        }

    except Exception as e:
        # 统一用 200 返回空列表和错误消息，避免前端 500
        return {
            "models": [],
            "providerId": request.providerId,
            "timestamp": int(datetime.now().timestamp()),
            "message": f"获取模型失败: {str(e)}，可在前端手动添加模型 ID"
        }


class ModelTestRequest(BaseModel):
    providerId: str = Field(min_length=1, max_length=128)
    modelId: str = Field(min_length=1, max_length=256)
    apiKey: str = Field(max_length=32_768)
    apiHost: str = Field(max_length=2_048)
    modelType: str = Field(min_length=1, max_length=32)  # 'embedding' or 'rerank'
    providerType: str | None = Field(default=None, max_length=64)
    embeddingEndpoint: str | None = Field(default=None, max_length=512)
    rerankEndpoint: str | None = Field(default=None, max_length=2_048)
    chatEndpoint: str | None = Field(default=None, max_length=512)
    apiKeyHeader: str | None = Field(default=None, max_length=128)
    apiKeyPrefix: str | None = Field(default=None, max_length=64)

    @field_validator("providerId")
    @classmethod
    def _validate_provider_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="providerId")

    @field_validator("modelType")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"chat", "embedding", "rerank"}:
            raise ValueError("modelType 仅支持 chat、embedding 或 rerank")
        return normalized

    @field_validator("embeddingEndpoint")
    @classmethod
    def _validate_embedding_endpoint(cls, value: str | None) -> str | None:
        return _validate_relative_endpoint(value, field_name="embeddingEndpoint") or None

    @field_validator("rerankEndpoint")
    @classmethod
    def _validate_rerank_endpoint(cls, value: str | None) -> str | None:
        return _validate_optional_model_url(value, field_name="rerankEndpoint", allow_relative=True) or None

    @field_validator("chatEndpoint")
    @classmethod
    def _validate_chat_endpoint(cls, value: str | None) -> str | None:
        return _validate_relative_endpoint(value, field_name="chatEndpoint") or None

    @field_validator("apiKeyHeader")
    @classmethod
    def _validate_api_key_header(cls, value: str | None) -> str | None:
        return normalize_api_key_header(value) if value is not None and str(value).strip() else None

    @field_validator("apiKeyPrefix")
    @classmethod
    def _validate_api_key_prefix(cls, value: str | None) -> str | None:
        return normalize_api_key_prefix(value) if value is not None else None


@router.post("/api/models/test")
async def test_model(request: ModelTestRequest):
    """测试具体模型的功能"""
    from time import time

    start_time = time()

    try:
        if request.providerId == 'local':
            try:
                from sentence_transformers import SentenceTransformer
            except (ImportError, OSError):
                raise HTTPException(
                    status_code=400,
                    detail="本地模型不可用（sentence-transformers 未安装）。"
                           "请使用远程模型，或安装完整依赖: pip install -r requirements.txt"
                )
            model = SentenceTransformer(request.modelId)
            test_text = "这是一个测试句子用于验证模型功能"
            embedding = model.encode([test_text])
            response_time = int((time() - start_time) * 1000)
            return {
                "success": True,
                "modelId": request.modelId,
                "providerId": "local",
                "dimension": int(embedding.shape[1]) if hasattr(embedding, "shape") else None,
                "responseTime": response_time,
            }

        if request.modelType == "chat":
            protocol = _provider_protocol(request.providerId, request.providerType)
            if protocol not in {"openai", "custom", "anthropic"}:
                raise HTTPException(status_code=400, detail=f"暂不支持自动测试 {protocol} 协议的 Chat 模型")
            api_host = _resolve_safe_provider_host(
                request.providerId,
                request.apiHost,
                field_name="Chat API 地址",
            )
            test_request = ProviderTestRequest(
                providerId=request.providerId,
                apiKey=request.apiKey,
                apiHost=api_host,
                modelId=request.modelId,
                chatEndpoint=request.chatEndpoint,
                providerType=request.providerType,
                apiKeyHeader=request.apiKeyHeader,
                apiKeyPrefix=request.apiKeyPrefix,
            )
            endpoint = _resolve_chat_test_endpoint(test_request, api_host)
            if protocol == "anthropic":
                await _probe_anthropic_chat(
                    endpoint=endpoint,
                    api_key=request.apiKey,
                    model_id=request.modelId,
                    api_key_header=request.apiKeyHeader,
                    api_key_prefix=request.apiKeyPrefix,
                )
            else:
                await _probe_openai_compatible_chat(
                    endpoint=endpoint,
                    api_key=request.apiKey,
                    model_id=request.modelId,
                    provider_type=protocol,
                    api_key_header=request.apiKeyHeader,
                    api_key_prefix=request.apiKeyPrefix,
                )
            return {
                "success": True,
                "modelId": request.modelId,
                "providerId": request.providerId,
                "message": "Chat 模型调用成功",
                "responseTime": int((time() - start_time) * 1000),
                "chatEndpoint": endpoint,
            }

        if request.modelType == 'embedding':
            protocol = _provider_protocol(request.providerId, request.providerType)
            dynamic = load_dynamic_providers().get(request.providerId) or {}
            headers = build_api_key_headers(
                request.apiKey,
                provider_type=protocol,
                api_key_header=request.apiKeyHeader if request.apiKeyHeader is not None else dynamic.get("api_key_header"),
                api_key_prefix=request.apiKeyPrefix if request.apiKeyPrefix is not None else dynamic.get("api_key_prefix"),
            )
            payload = {
                "input": ["Hello world"],
                "model": request.modelId
            }
            api_host = _resolve_safe_provider_host(
                request.providerId,
                request.apiHost,
                field_name="Embedding API 地址",
            )
            dynamic_config = (load_dynamic_providers().get(request.providerId) or {})
            embedding_path = request.embeddingEndpoint or dynamic_config.get("embedding_endpoint") or "/embeddings"
            url = _build_endpoint(api_host, embedding_path)
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code < 200 or resp.status_code >= 300:
                raise HTTPException(status_code=resp.status_code, detail=f"Embedding接口返回错误: {resp.text[:500]}")
            try:
                data = resp.json()
            except ValueError as exc:
                raise HTTPException(status_code=502, detail=f"Embedding接口返回了无效 JSON：{exc}") from exc
            rows = data.get("data") if isinstance(data, dict) else None
            embedding = rows[0].get("embedding") if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None
            if not isinstance(embedding, list) or not embedding:
                raise HTTPException(status_code=502, detail="Embedding接口返回成功，但没有有效向量")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in embedding):
                raise HTTPException(status_code=502, detail="Embedding接口返回的向量格式无效")
            dim = len(embedding)
            return {
                "success": True,
                "modelId": request.modelId,
                "providerId": request.providerId,
                "dimension": dim,
                "responseTime": int((time() - start_time) * 1000),
            }

        if request.modelType == 'rerank':
            protocol = _provider_protocol(request.providerId, request.providerType)
            dynamic = load_dynamic_providers().get(request.providerId) or {}
            headers = build_api_key_headers(
                request.apiKey,
                provider_type=protocol,
                api_key_header=request.apiKeyHeader if request.apiKeyHeader is not None else dynamic.get("api_key_header"),
                api_key_prefix=request.apiKeyPrefix if request.apiKeyPrefix is not None else dynamic.get("api_key_prefix"),
            )
            payload = {
                "model": request.modelId,
                "query": "test",
                "documents": ["a", "b"]
            }
            rerank_value = request.rerankEndpoint
            if rerank_value and not urlparse(rerank_value).scheme:
                api_host = _resolve_safe_provider_host(
                    request.providerId,
                    request.apiHost,
                    field_name="Rerank API 地址",
                )
                url = _build_endpoint(api_host, rerank_value)
            else:
                url = _validate_provider_url(
                    rerank_value or "https://api.cohere.com/v1/rerank",
                    field_name="Rerank API 地址",
                )
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code < 200 or resp.status_code >= 300:
                raise HTTPException(status_code=resp.status_code, detail=f"Rerank接口返回错误: {resp.text[:500]}")
            try:
                data = resp.json()
            except ValueError as exc:
                raise HTTPException(status_code=502, detail=f"Rerank接口返回了无效 JSON：{exc}") from exc
            results = data.get("results") if isinstance(data, dict) else None
            if not isinstance(results, list) or not results:
                raise HTTPException(status_code=502, detail="Rerank接口返回成功，但没有有效排序结果")
            valid_results = 0
            for item in results:
                if not isinstance(item, dict):
                    continue
                score = item.get("relevance_score", item.get("score"))
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    valid_results += 1
            if valid_results == 0:
                raise HTTPException(status_code=502, detail="Rerank接口返回的排序结果格式无效")
            return {
                "success": True,
                "modelId": request.modelId,
                "providerId": request.providerId,
                "resultCount": valid_results,
                "responseTime": int((time() - start_time) * 1000),
            }

        raise HTTPException(status_code=400, detail="不支持的模型类型")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"测试失败：{str(e)}")


class ProviderUpsertRequest(BaseModel):
    providerId: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=160)
    endpoint: str = Field(min_length=1, max_length=2_048)
    type: str = Field(default="openai", max_length=64)  # openai | anthropic | gemini | ollama
    fetchModelsEndpoint: str | None = Field(default=None, max_length=512)
    chatEndpoint: str | None = Field(default="/chat/completions", max_length=512)
    embeddingEndpoint: str | None = Field(default="/embeddings", max_length=512)
    rerankEndpoint: str | None = Field(default=None, max_length=512)
    supportsStreaming: bool = True
    supportsReasoning: bool = False
    reasoningMode: str | None = None
    reasoningOptions: list[str] | None = Field(default=None, max_length=8)
    reasoningDefault: str | None = None
    reasoningAlwaysEnabled: bool | None = None
    reasoningOffControl: str | None = None
    reasoningOnControl: str | None = None
    apiKeyHeader: str | None = Field(default=None, max_length=128)
    apiKeyPrefix: str | None = Field(default=None, max_length=64)
    capabilities: dict[str, bool] | None = None

    @field_validator("providerId")
    @classmethod
    def _validate_provider_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="providerId")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("name 不能为空")
        return normalized

    @field_validator("type")
    @classmethod
    def _validate_provider_type(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in _ALLOWED_CUSTOM_PROVIDER_TYPES:
            raise ValueError("不支持的自定义 provider 类型")
        return normalized

    @field_validator("fetchModelsEndpoint", "chatEndpoint", "embeddingEndpoint", "rerankEndpoint")
    @classmethod
    def _validate_endpoint_path(cls, value: str | None, info) -> str | None:
        return _validate_relative_endpoint(value, field_name=info.field_name) or None

    @field_validator("apiKeyHeader")
    @classmethod
    def _validate_api_key_header(cls, value: str | None) -> str | None:
        return normalize_api_key_header(value) if value is not None and str(value).strip() else None

    @field_validator("apiKeyPrefix")
    @classmethod
    def _validate_api_key_prefix(cls, value: str | None) -> str | None:
        return normalize_api_key_prefix(value) if value is not None else None

    @field_validator("reasoningMode")
    @classmethod
    def _validate_reasoning_mode(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return None
        if normalized not in _REASONING_MODES:
            raise ValueError("reasoningMode 不受支持")
        return normalized

    @field_validator("reasoningOptions")
    @classmethod
    def _validate_reasoning_options(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = list(dict.fromkeys(str(item or "").strip().lower() for item in value))
        if any(item not in _REASONING_LEVELS for item in normalized):
            raise ValueError("reasoningOptions 包含不支持的档位")
        return normalized

    @field_validator("reasoningDefault")
    @classmethod
    def _validate_reasoning_default(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return None
        if normalized not in _REASONING_LEVELS:
            raise ValueError("reasoningDefault 不受支持")
        return normalized

    @field_validator("reasoningOffControl")
    @classmethod
    def _validate_reasoning_off_control(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return None
        if normalized not in _REASONING_OFF_CONTROLS:
            raise ValueError("reasoningOffControl 不受支持")
        return normalized

    @field_validator("reasoningOnControl")
    @classmethod
    def _validate_reasoning_on_control(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return None
        if normalized not in _REASONING_ON_CONTROLS:
            raise ValueError("reasoningOnControl 不受支持")
        return normalized


@router.get("/api/providers/custom")
async def list_custom_providers():
    """列出动态配置的 provider"""
    return load_dynamic_providers()


@router.post("/api/providers/custom")
async def upsert_custom_provider(req: ProviderUpsertRequest):
    if req.providerId in PROVIDER_CONFIG:
        raise HTTPException(status_code=409, detail="内置 provider 不允许由自定义配置覆盖")
    try:
        endpoint = _validate_provider_url(req.endpoint, field_name="自定义 Provider API 地址")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _DYNAMIC_CONFIG_LOCK:
        providers = load_dynamic_providers()
        providers[req.providerId] = {
            "name": req.name,
            "endpoint": endpoint,
            "type": req.type,
            "fetch_models_endpoint": req.fetchModelsEndpoint or "/models",
            "chat_endpoint": req.chatEndpoint or "/chat/completions",
            "embedding_endpoint": req.embeddingEndpoint or "/embeddings",
            "rerank_endpoint": req.rerankEndpoint or "",
            "supports_streaming": bool(req.supportsStreaming),
            "supports_reasoning": bool(req.supportsReasoning),
            "reasoning_mode": req.reasoningMode,
            "reasoning_options": req.reasoningOptions or [],
            "reasoning_default": req.reasoningDefault,
            "reasoning_always_enabled": req.reasoningAlwaysEnabled,
            "reasoning_off_control": req.reasoningOffControl,
            "reasoning_on_control": req.reasoningOnControl,
            "api_key_header": req.apiKeyHeader,
            "api_key_prefix": req.apiKeyPrefix,
            "capabilities": {
                key: bool(value)
                for key, value in (req.capabilities or {}).items()
                if key in {"chat", "embedding", "rerank", "imageGeneration"}
            },
        }
        save_dynamic_providers(providers)
    return {"success": True, "providers": providers}


@router.delete("/api/providers/custom/{provider_id}")
async def delete_custom_provider(provider_id: str):
    try:
        provider_id = _validate_identifier(provider_id, field_name="provider_id")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if provider_id in PROVIDER_CONFIG:
        raise HTTPException(status_code=409, detail="内置 provider 不允许删除")
    with _DYNAMIC_CONFIG_LOCK:
        providers = load_dynamic_providers()
        removed = providers.pop(provider_id, None)
        removed_model_ids = []
        if removed is not None:
            save_dynamic_providers(providers)
        # 自定义模型记录以 provider_id 绑定 Provider。只清理明确绑定到
        # 当前 Provider 的记录，避免误删旧数据中仅有 provider/type 的模型。
        models = load_dynamic_models()
        remaining_models = {}
        for model_id, model in models.items():
            bound_provider_id = None
            if isinstance(model, dict):
                bound_provider_id = model.get("provider_id") or model.get("providerId")
            if bound_provider_id == provider_id:
                removed_model_ids.append(model_id)
            else:
                remaining_models[model_id] = model
        if removed_model_ids:
            save_dynamic_models(remaining_models)
    return {
        "success": True,
        "providers": providers,
        "removedModelIds": removed_model_ids,
    }


@router.get("/api/models/reasoning-capabilities")
async def get_reasoning_capabilities(
    provider: str = "",
    model: str = "",
    provider_type: str | None = None,
    supports_reasoning: bool | None = None,
    reasoning_mode: str | None = None,
    reasoning_options: str | None = None,
    reasoning_default: str | None = None,
    reasoning_always_enabled: bool | None = None,
    reasoning_off_control: str | None = None,
    reasoning_on_control: str | None = None,
):
    """返回当前 provider/model 真正可用的思考档位。

    前端不再凭 provider 名称硬编码菜单。能力接口只读取非敏感的
    provider/model 元数据，不会触碰 API Key，也不会向上游发探测请求。
    """
    try:
        provider_id = _validate_identifier(provider, field_name="provider")
        model_id = _validate_model_id(model, field_name="model")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    options = reasoning_options_for_frontend(
        provider_id,
        model_id,
        provider_type=provider_type,
        supports_reasoning=supports_reasoning,
        declared_mode=reasoning_mode,
        declared_options=reasoning_options,
        declared_default=reasoning_default,
        declared_always_enabled=reasoning_always_enabled,
        declared_off_control=reasoning_off_control,
        declared_on_control=reasoning_on_control,
    )
    return {
        "provider": provider_id,
        "model": model_id,
        "provider_type": provider_type or _get_provider_type(provider_id),
        **options,
    }


# ===== 动态模型管理 =====

class ModelUpsertRequest(BaseModel):
    modelId: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    providerId: str = Field(min_length=1, max_length=128)
    type: str = Field(default="embedding", max_length=32)  # embedding | rerank | chat
    metadata: dict | None = None
    capabilities: list[dict] | None = Field(default=None, max_length=16)  # 模型能力声明列表，每个元素包含 type 和 isUserSelected 字段
    tags: list[str] | None = Field(default=None, max_length=24)  # 模型标签列表（如 free、vision、reasoning 等）

    @field_validator("modelId")
    @classmethod
    def _validate_model_id_field(cls, value: str) -> str:
        return _validate_model_id(value, field_name="modelId")

    @field_validator("providerId")
    @classmethod
    def _validate_provider_id_field(cls, value: str) -> str:
        return _validate_identifier(value, field_name="providerId")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("name 不能为空")
        return normalized

    @field_validator("type")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in _ALLOWED_CUSTOM_MODEL_TYPES:
            raise ValueError("不支持的自定义模型类型")
        return normalized

    @field_validator("metadata")
    @classmethod
    def _validate_metadata_shape(cls, value: dict | None) -> dict | None:
        if value is None:
            return None
        if len(value) > 24:
            raise ValueError("metadata 字段过多")
        for key, item in value.items():
            if len(str(key)) > 64 or len(str(item)) > 2_048:
                raise ValueError("metadata 中存在过长字段")
        return value

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        result = []
        for tag in value:
            normalized = str(tag or "").strip()
            if not normalized or len(normalized) > 48:
                raise ValueError("tags 包含无效值")
            result.append(normalized)
        return list(dict.fromkeys(result))


def _validate_model_metadata_endpoints(metadata: dict) -> dict:
    normalized = dict(metadata or {})
    for field_name in ("base_url", "embedding_endpoint", "rerank_endpoint"):
        value = normalized.get(field_name)
        if value in (None, ""):
            continue
        try:
            normalized[field_name] = _validate_optional_model_url(
                value,
                field_name=field_name,
                allow_relative=field_name in {"embedding_endpoint", "rerank_endpoint"},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return normalized


@router.get("/api/models/custom")
async def list_custom_models():
    return load_dynamic_models()


@router.post("/api/models/custom")
async def upsert_custom_model(req: ModelUpsertRequest):
    from models.model_registry import EMBEDDING_MODELS

    if req.modelId in EMBEDDING_MODELS:
        raise HTTPException(status_code=409, detail="内置模型不允许由自定义配置覆盖")
    merged_providers = {**PROVIDER_CONFIG, **load_dynamic_providers()}
    if req.providerId not in merged_providers:
        raise HTTPException(status_code=400, detail="providerId 不存在，请先添加自定义 Provider")

    try:
        normalized_metadata = _validate_model_metadata_endpoints(
            _normalize_model_metadata(req.metadata)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    provider_type = _get_provider_type(req.providerId)
    provider_config = ({**PROVIDER_CONFIG, **load_dynamic_providers()}).get(req.providerId) or {}
    if req.type == "embedding":
        normalized_metadata.setdefault("base_url", provider_config.get("endpoint", ""))
        normalized_metadata.setdefault(
            "embedding_endpoint",
            provider_config.get("embedding_endpoint", "/embeddings"),
        )
    if req.type == "rerank":
        normalized_metadata.setdefault(
            "base_url",
            provider_config.get("endpoint", ""),
        )
        normalized_metadata.setdefault(
            "rerank_endpoint",
            provider_config.get("rerank_endpoint", ""),
        )
    model_data = {
        "name": req.name,
        "provider": provider_type,
        "provider_id": req.providerId,
        "provider_type": provider_type,
        "type": req.type,
        **normalized_metadata,
    }
    # 持久化 capabilities 和 tags 字段到动态存储
    if req.capabilities is not None:
        model_data["capabilities"] = req.capabilities
    if req.tags is not None:
        model_data["tags"] = req.tags
    with _DYNAMIC_CONFIG_LOCK:
        models = load_dynamic_models()
        models[req.modelId] = model_data
        save_dynamic_models(models)
    return {"success": True, "models": models}


@router.delete("/api/models/custom/{model_id:path}")
async def delete_custom_model(model_id: str):
    try:
        model_id = _validate_model_id(model_id, field_name="model_id")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _DYNAMIC_CONFIG_LOCK:
        models = load_dynamic_models()
        if model_id in models:
            models.pop(model_id)
            save_dynamic_models(models)
    return {"success": True, "models": models}


# Helpers reused from app.py for model inference
import re


def _detect_model_type(model_id: str) -> str:
    """统一使用 model_detector.py 的正则检测模型类型

    优先级：rerank > embedding > image > chat（默认）
    """
    if is_rerank_model(model_id):
        return 'rerank'
    if is_embedding_model(model_id):
        return 'embedding'
    lower_id = model_id.lower()
    if re.search(r'image|img|diffusion|sd|dall-e|dalle', lower_id):
        return 'image'
    return 'chat'


def _infer_model_metadata(model_id: str, model_type: str) -> dict:
    metadata = {}
    lower_id = model_id.lower()
    if model_type == 'embedding':
        if 'text-embedding-3-large' in model_id:
            metadata['dimension'] = 3072
            metadata['maxTokens'] = 8191
        elif 'text-embedding-3-small' in model_id:
            metadata['dimension'] = 1536
            metadata['maxTokens'] = 8191
        elif 'text-embedding-ada-002' in model_id:
            metadata['dimension'] = 1536
            metadata['maxTokens'] = 8191
        elif 'bge-m3' in model_id:
            metadata['dimension'] = 1024
            metadata['maxTokens'] = 8192
        else:
            metadata['dimension'] = 1024
            metadata['maxTokens'] = 512
    elif model_type == 'chat':
        if 'gpt-4' in model_id:
            metadata['contextWindow'] = 32768 if '32k' in model_id else 8192
        elif 'gpt-3.5' in model_id:
            metadata['contextWindow'] = 16384 if '16k' in model_id else 4096
        elif 'claude-3' in model_id:
            metadata['contextWindow'] = 200000
        elif 'gemini-1.5' in lower_id:
            metadata['contextWindow'] = 1000000
        else:
            metadata['contextWindow'] = 4096
    return metadata
