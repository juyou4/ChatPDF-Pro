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


router = APIRouter()

# ``load -> mutate -> save`` must be one critical section. Atomic files protect
# readers from partial JSON, while this lock prevents two requests in this
# process from silently dropping each other's settings change.
_DYNAMIC_CONFIG_LOCK = threading.RLock()
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
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
}


_LATEST_MODEL_IDS = {
    "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "qwen3.7-max", "qwen3.7-plus",
    "deepseek-v4-flash", "deepseek-v4-pro", "kimi-k2.6",
    "glm-5.2", "MiniMax-M3",
    "mimo-v2.5-pro", "mimo-v2.5",
    "claude-fable-5", "claude-opus-4-8", "claude-sonnet-5",
    "gemini-3.5-flash", "gemini-3.1-pro-preview",
    "grok-4.3", "grok-build-0.1",
    "doubao-seed-2-1-pro-260628", "doubao-seed-2-1-turbo-260628",
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
    }
    for target_field, aliases in field_aliases.items():
        picked = _pick_first(*(raw.get(alias) for alias in aliases))
        if picked is not None:
            normalized[target_field] = picked
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
            "qwen3.6-flash": "Qwen3.6-Flash",
        },
        "deepseek": {
            "deepseek-v4-flash": "DeepSeek V4 Flash",
            "deepseek-v4-pro": "DeepSeek V4 Pro",
        },
        "moonshot": {
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
            "claude-fable-5": "Claude Fable 5",
            "claude-opus-4-8": "Claude Opus 4.8",
            "claude-sonnet-5": "Claude Sonnet 5",
            "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
        },
        "gemini": {
            "gemini-3.5-flash": "Gemini 3.5 Flash",
            "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
            "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
            "gemini-2.5-pro": "Gemini 2.5 Pro",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
        },
        "grok": {
            "grok-4.3": "Grok 4.3",
            "grok-build-0.1": "Grok Build 0.1",
        },
        "doubao": {
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

    @field_validator("providerId")
    @classmethod
    def _validate_provider_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="providerId")

    @field_validator("fetchModelsEndpoint")
    @classmethod
    def _validate_models_endpoint(cls, value: str | None) -> str | None:
        return _validate_relative_endpoint(value, field_name="fetchModelsEndpoint") or None


def _build_endpoint(base: str, path: str | None) -> str:
    if not base:
        return path or ""
    base_clean = base.rstrip('/')
    if not path:
        return base_clean
    path_clean = path.lstrip('/')
    return f"{base_clean}/{path_clean}"


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



async def _fetch_models_with_fallback(api_host: str, api_key: str, endpoints: List[str]):
    api_host = _validate_provider_url(api_host, field_name="模型服务 API 地址")
    # 从 API Key 池中随机选择一个有效 Key（支持逗号分隔的多 Key 轮换）
    actual_key = select_api_key(api_key) if api_key else None
    if not actual_key:
        return None, "API Key 池为空，无法发送请求"
    headers = {
        "Authorization": f"Bearer {actual_key}",
        "Content-Type": "application/json"
    }
    last_error = None
    auth_error = None  # 记录 401/403 认证失败，用于区分「key错误」和「端点不存在」

    for ep in endpoints:
        if not ep:
            continue
        path = _validate_relative_endpoint(ep, field_name="模型列表路径")
        url = _build_endpoint(api_host, path)
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return response.json(), url
            # 401/403 表示 API Key 无效，立即返回认证失败（不继续尝试其他 endpoint）
            if response.status_code in (401, 403):
                try:
                    err_body = response.json()
                    err_msg = (
                        err_body.get("error", {}).get("message")
                        or err_body.get("message")
                        or str(err_body)
                    )
                except Exception:
                    err_msg = response.text[:200]
                auth_error = f"API Key 无效或格式错误（HTTP {response.status_code}）: {err_msg}"
                return None, auth_error
            last_error = f"HTTP {response.status_code}"
        except Exception as e:
            last_error = str(e)
            continue
    return None, last_error


@router.post("/api/providers/test")
async def test_provider_connection(request: ProviderTestRequest):
    """测试Provider连接，成功时返回延迟毫秒数"""
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
        endpoints = [request.fetchModelsEndpoint or "/models", "/v1/models", "/models"]
        data, last_error = await _fetch_models_with_fallback(api_host, request.apiKey, endpoints)

        if data is not None:
            model_count = len(data.get('data', [])) if isinstance(data.get('data'), list) else 0
            latency = int((time() - start_time) * 1000)
            return {
                "success": True,
                "message": "连接成功",
                "availableModels": model_count,
                "latency": latency
            }

        # 401/403 认证失败：API Key 无效或格式错误
        if last_error and ("401" in last_error or "403" in last_error or "API Key 无效" in last_error):
            return {"success": False, "message": last_error}

        # 虽然未获取到模型列表，但连接本身成功（服务器可达且未返回 401/403）
        latency = int((time() - start_time) * 1000)
        return {
            "success": True,
            "message": f"连接成功（无法获取模型列表: {last_error or '无响应'})",
            "availableModels": 0,
            "latency": latency
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

    @field_validator("providerId")
    @classmethod
    def _validate_provider_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="providerId")

    @field_validator("fetchModelsEndpoint")
    @classmethod
    def _validate_models_endpoint(cls, value: str | None) -> str | None:
        return _validate_relative_endpoint(value, field_name="fetchModelsEndpoint") or None


@router.post("/api/models/fetch")
async def fetch_provider_models(request: ModelFetchRequest):
    """从Provider API获取模型列表（支持动态/静态）"""
    try:
        # Anthropic Claude：使用非 OpenAI 格式 API，返回预设模型列表
        if request.providerId == 'anthropic':
            ANTHROPIC_PRESET_MODELS = [
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

        # 字节跳动豆包：火山引擎 Ark API 不提供 GET /models 端点，返回预设模型列表
        # 注意：2.0 系列的官方 model id 带版本号，旧短 id 会在调用时 404
        if request.providerId == 'doubao':
            DOUBAO_PRESET_MODELS = [
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
        data, last_error = await _fetch_models_with_fallback(api_host, request.apiKey, endpoints)

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
    embeddingEndpoint: str | None = Field(default=None, max_length=512)
    rerankEndpoint: str | None = Field(default=None, max_length=2_048)

    @field_validator("providerId")
    @classmethod
    def _validate_provider_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="providerId")

    @field_validator("modelType")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"embedding", "rerank"}:
            raise ValueError("modelType 仅支持 embedding 或 rerank")
        return normalized

    @field_validator("embeddingEndpoint")
    @classmethod
    def _validate_embedding_endpoint(cls, value: str | None) -> str | None:
        return _validate_relative_endpoint(value, field_name="embeddingEndpoint") or None

    @field_validator("rerankEndpoint")
    @classmethod
    def _validate_rerank_endpoint(cls, value: str | None) -> str | None:
        return _validate_optional_model_url(value, field_name="rerankEndpoint") or None


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

        if request.modelType == 'embedding':
            headers = {
                "Authorization": f"Bearer {request.apiKey}",
                "Content-Type": "application/json"
            }
            payload = {
                "input": ["Hello world"],
                "model": request.modelId
            }
            api_host = _resolve_safe_provider_host(
                request.providerId,
                request.apiHost,
                field_name="Embedding API 地址",
            )
            url = _build_endpoint(api_host, request.embeddingEndpoint or "/v1/embeddings")
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Embedding接口返回错误: {resp.text}")
            data = resp.json()
            dim = len(data.get("data", [{}])[0].get("embedding", [])) if data.get("data") else None
            return {
                "success": True,
                "modelId": request.modelId,
                "providerId": request.providerId,
                "dimension": dim,
                "responseTime": int((time() - start_time) * 1000),
            }

        if request.modelType == 'rerank':
            headers = {
                "Authorization": f"Bearer {request.apiKey}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": request.modelId,
                "query": "test",
                "documents": ["a", "b"]
            }
            url = _validate_provider_url(
                request.rerankEndpoint or "https://api.cohere.com/v1/rerank",
                field_name="Rerank API 地址",
            )
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Rerank接口返回错误: {resp.text}")
            return {
                "success": True,
                "modelId": request.modelId,
                "providerId": request.providerId,
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
        }
        save_dynamic_providers(providers)
    return {"success": True, "providers": providers}


@router.delete("/api/providers/custom/{provider_id}")
async def delete_custom_provider(provider_id: str):
    try:
        provider_id = _validate_identifier(provider_id, field_name="provider_id")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _DYNAMIC_CONFIG_LOCK:
        providers = load_dynamic_providers()
        if provider_id in providers:
            providers.pop(provider_id)
            save_dynamic_providers(providers)
    return {"success": True, "providers": providers}


# ===== 动态模型管理 =====

class ModelUpsertRequest(BaseModel):
    modelId: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    providerId: str = Field(min_length=1, max_length=128)
    type: str = Field(default="embedding", max_length=32)  # embedding | rerank | chat
    metadata: dict | None = None
    capabilities: list[dict] | None = Field(default=None, max_length=16)  # 模型能力声明列表，每个元素包含 type 和 isUserSelected 字段
    tags: list[str] | None = Field(default=None, max_length=24)  # 模型标签列表（如 free、vision、reasoning 等）

    @field_validator("modelId", "providerId")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _validate_identifier(value, field_name=info.field_name)

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

    normalized_metadata = _validate_model_metadata_endpoints(
        _normalize_model_metadata(req.metadata)
    )
    provider_type = _get_provider_type(req.providerId)
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


@router.delete("/api/models/custom/{model_id}")
async def delete_custom_model(model_id: str):
    try:
        model_id = _validate_identifier(model_id, field_name="model_id")
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
