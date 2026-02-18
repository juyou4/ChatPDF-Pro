from datetime import datetime
from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.provider_registry import PROVIDER_CONFIG
from models.rerank_registry import RERANK_PROVIDERS
from models.dynamic_store import (
    load_dynamic_providers,
    save_dynamic_providers,
    load_dynamic_models,
    save_dynamic_models,
)
from models.model_detector import infer_model_tags
from models.api_key_selector import select_api_key


router = APIRouter()


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

    # 从前端 systemModels.ts 同步的 chat 模型列表
    # 这些模型不在 EMBEDDING_MODELS 中，但前端需要通过 /models API 获取
    CHAT_MODELS = {
        "openai": {
            "gpt-4.1": "GPT-4.1",
            "gpt-4.1-mini": "GPT-4.1 mini",
            "o3": "OpenAI o3",
            "o4-mini": "OpenAI o4-mini",
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o mini",
        },
        "aliyun": {
            "qwen3-max": "Qwen3-Max",
            "qwen3.5-plus": "Qwen3.5-Plus",
            "qwen-plus": "Qwen-Plus",
            "qwen-turbo": "Qwen-Turbo",
        },
        "deepseek": {
            "deepseek-chat": "DeepSeek V3",
            "deepseek-reasoner": "DeepSeek R1",
        },
        "moonshot": {
            "kimi-k2.5": "Kimi K2.5",
            "kimi-k2": "Kimi K2",
            "moonshot-v1-128k": "Moonshot v1 128K",
            "moonshot-v1-32k": "Moonshot v1 32K",
            "moonshot-v1-8k": "Moonshot v1 8K",
        },
        "zhipu": {
            "glm-5": "GLM-5",
            "glm-4.7": "GLM-4.7",
            "glm-4.5": "GLM-4.5",
            "glm-4.5-air": "GLM-4.5-Air",
            "glm-4-air": "GLM-4-Air",
        },
        "minimax": {
            "MiniMax-Text-01": "MiniMax Text-01",
            "abab6.5s-chat": "abab6.5s-chat",
        },
        "silicon": {
            "deepseek-ai/DeepSeek-R1": "DeepSeek R1 (SiliconFlow)",
            "deepseek-ai/DeepSeek-V3": "DeepSeek V3 (SiliconFlow)",
            "Qwen/Qwen3-235B-A22B": "Qwen3-235B (SiliconFlow)",
            "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B (SiliconFlow)",
        },
        "anthropic": {
            "claude-opus-4-6": "Claude Opus 4.6",
            "claude-sonnet-4-6": "Claude Sonnet 4.6",
            "claude-opus-4-5": "Claude Opus 4.5",
            "claude-sonnet-4-5": "Claude Sonnet 4.5",
            "claude-haiku-3-5": "Claude Haiku 3.5",
        },
        "gemini": {
            "gemini-3-pro": "Gemini 3 Pro",
            "gemini-3-flash": "Gemini 3 Flash",
            "gemini-2.5-pro": "Gemini 2.5 Pro",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
        },
        "grok": {
            "grok-4": "Grok 4",
            "grok-4-1-fast": "Grok 4.1 Fast",
            "grok-3": "Grok 3",
            "grok-3-mini": "Grok 3 Mini",
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

            # 本地模型只归属于 local provider
            if model_provider_type == "local":
                if provider_id == "local":
                    provider_models[model_id] = model_config.get("name", model_id)
                continue

            # 非本地模型：先匹配 provider type，再通过 base_url 域名区分
            if model_provider_type != provider_type:
                continue

            model_base_url = model_config.get("base_url", "")
            model_domain = _extract_domain(model_base_url)

            # 通过域名匹配区分同 type 的不同服务商
            if provider_domain and model_domain and provider_domain == model_domain:
                provider_models[model_id] = model_config.get("name", model_id)

        # 合并 chat 模型
        chat_models = CHAT_MODELS.get(provider_id, {})
        provider_models.update(chat_models)

        result[provider_id] = {
            **provider_config,
            "models": provider_models,
        }

    return result


@router.get("/rerank/providers")
async def get_rerank_providers():
    """获取支持的重排提供商及默认模型"""
    return RERANK_PROVIDERS


class ProviderTestRequest(BaseModel):
    providerId: str
    apiKey: str
    apiHost: str
    fetchModelsEndpoint: str | None = None


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



async def _fetch_models_with_fallback(api_host: str, api_key: str, endpoints: List[str]):
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
        url = _build_endpoint(api_host, ep)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
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

        endpoints = [request.fetchModelsEndpoint or "/models", "/v1/models", "/models"]
        data, last_error = await _fetch_models_with_fallback(request.apiHost, request.apiKey, endpoints)

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
    providerId: str
    apiKey: str
    apiHost: str
    fetchModelsEndpoint: str | None = None
    providerType: str | None = None  # "openai" | "anthropic" | "gemini" | ...


@router.post("/api/models/fetch")
async def fetch_provider_models(request: ModelFetchRequest):
    """从Provider API获取模型列表（支持动态/静态）"""
    try:
        # Anthropic Claude：使用非 OpenAI 格式 API，返回预设模型列表
        if request.providerId == 'anthropic':
            ANTHROPIC_PRESET_MODELS = [
                {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "type": "chat",
                 "metadata": {"description": "Anthropic 旗舰模型，200K 上下文，最强编程与推理，支持 1M 上下文 (beta)"}},
                {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "type": "chat",
                 "metadata": {"description": "Anthropic 最新均衡模型，Opus 级别推理能力，200K 上下文，同等价格"}},
                {"id": "claude-opus-4-5", "name": "Claude Opus 4.5", "type": "chat",
                 "metadata": {"description": "Claude Opus 系列前代，超强编程、Agent 工作流"}},
                {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "type": "chat",
                 "metadata": {"description": "Claude 均衡前代版本，高性价比"}},
                {"id": "claude-haiku-3-5", "name": "Claude Haiku 3.5", "type": "chat",
                 "metadata": {"description": "最快速轻量 Claude 模型，低成本高并发"}},
            ]
            return {
                "models": [
                    {
                        "id": m["id"],
                        "name": m["name"],
                        "providerId": "anthropic",
                        "type": m["type"],
                        "capabilities": [{"type": m["type"], "isUserSelected": False}],
                        "tags": infer_model_tags(m["id"]),
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
                {"id": "gemini-3-pro", "name": "Gemini 3 Pro", "type": "chat",
                 "metadata": {"description": "Google 最新旗舰推理模型，1M 上下文，自适应思考，强多模态 (preview)"}},
                {"id": "gemini-3-flash", "name": "Gemini 3 Flash", "type": "chat",
                 "metadata": {"description": "Google 最新多模态理解模型，强编程与推理 (preview)"}},
                {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "type": "chat",
                 "metadata": {"description": "Gemini 旗舰稳定版，1M 上下文，自适应思考"}},
                {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "type": "chat",
                 "metadata": {"description": "Gemini 快速均衡版，可控推理预算"}},
                {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash-Lite", "type": "chat",
                 "metadata": {"description": "Gemini 超轻量版，大规模低成本场景"}},
            ]
            return {
                "models": [
                    {
                        "id": m["id"],
                        "name": m["name"],
                        "providerId": "gemini",
                        "type": m["type"],
                        "capabilities": [{"type": m["type"], "isUserSelected": False}],
                        "tags": infer_model_tags(m["id"]),
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
        if request.providerId == 'doubao':
            DOUBAO_PRESET_MODELS = [
                {"id": "doubao-seed-2-0-pro", "name": "Doubao Seed 2.0 Pro", "type": "chat",
                 "metadata": {"description": "豆包 2.0 旗舰模型，对标 GPT-5.2 / Gemini 3 Pro，支持长链路推理与多模态"}},
                {"id": "doubao-seed-2-0-lite-260215", "name": "Doubao Seed 2.0 Lite", "type": "chat",
                 "metadata": {"description": "豆包 2.0 Lite，均衡性能与成本，能力超越上一代豆包 1.8"}},
                {"id": "doubao-seed-2-0-mini-260215", "name": "Doubao Seed 2.0 Mini", "type": "chat",
                 "metadata": {"description": "豆包 2.0 Mini，低延迟高并发，适合成本敏感场景"}},
                {"id": "doubao-seed-2-0-code-preview-260215", "name": "Doubao Seed 2.0 Code", "type": "chat",
                 "metadata": {"description": "豆包 2.0 编程专项模型，深度优化 Agentic Coding 场景"}},
                {"id": "doubao-seed-1-8", "name": "Doubao Seed 1.8", "type": "chat",
                 "metadata": {"description": "豆包 1.8，上一代主力模型，多模态 Agent 场景优化"}},
                {"id": "doubao-1-5-pro-32k-250115", "name": "Doubao 1.5 Pro 32K", "type": "chat",
                 "metadata": {"description": "豆包 1.5 Pro，32K 上下文"}},
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
                        "tags": infer_model_tags(m["id"]),
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
                        "metadata": {"dimension": 384, "maxTokens": 256, "description": "快速通用模型"},
                        "isSystem": True,
                        "isUserAdded": False
                    },
                    {
                        "id": "paraphrase-multilingual-MiniLM-L12-v2",
                        "name": "Multilingual MiniLM-L12-v2",
                        "providerId": "local",
                        "type": "embedding",
                        "metadata": {"dimension": 384, "maxTokens": 128, "description": "多语言支持"},
                        "isSystem": True,
                        "isUserAdded": False
                    }
                ],
                "providerId": "local",
                "timestamp": int(datetime.now().timestamp())
            }

        api_host = _normalize_api_host(request.providerId, request.apiHost)
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
                model_type = _detect_model_type(model_id)
                # 推断模型标签（如 free、vision、reasoning 等）
                tags = infer_model_tags(model_id)
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
    providerId: str
    modelId: str
    apiKey: str
    apiHost: str
    modelType: str  # 'embedding' or 'rerank'
    embeddingEndpoint: str | None = None
    rerankEndpoint: str | None = None


@router.post("/api/models/test")
async def test_model(request: ModelTestRequest):
    """测试具体模型的功能"""
    from time import time

    start_time = time()

    try:
        if request.providerId == 'local':
            from sentence_transformers import SentenceTransformer
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
            url = _build_endpoint(request.apiHost, request.embeddingEndpoint or "/v1/embeddings")
            async with httpx.AsyncClient(timeout=15.0) as client:
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
            url = request.rerankEndpoint or "https://api.cohere.com/v1/rerank"
            async with httpx.AsyncClient(timeout=15.0) as client:
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
    providerId: str
    name: str
    endpoint: str
    type: str = "openai"  # openai | anthropic | gemini | ollama


@router.get("/api/providers/custom")
async def list_custom_providers():
    """列出动态配置的 provider"""
    return load_dynamic_providers()


@router.post("/api/providers/custom")
async def upsert_custom_provider(req: ProviderUpsertRequest):
    providers = load_dynamic_providers()
    providers[req.providerId] = {
        "name": req.name,
        "endpoint": req.endpoint,
        "type": req.type
    }
    save_dynamic_providers(providers)
    return {"success": True, "providers": providers}


@router.delete("/api/providers/custom/{provider_id}")
async def delete_custom_provider(provider_id: str):
    providers = load_dynamic_providers()
    if provider_id in providers:
        providers.pop(provider_id)
        save_dynamic_providers(providers)
    return {"success": True, "providers": providers}


# ===== 动态模型管理 =====

class ModelUpsertRequest(BaseModel):
    modelId: str
    name: str
    providerId: str
    type: str = "embedding"  # embedding | rerank | chat
    metadata: dict | None = None
    capabilities: list[dict] | None = None  # 模型能力声明列表，每个元素包含 type 和 isUserSelected 字段
    tags: list[str] | None = None  # 模型标签列表（如 free、vision、reasoning 等）


@router.get("/api/models/custom")
async def list_custom_models():
    return load_dynamic_models()


@router.post("/api/models/custom")
async def upsert_custom_model(req: ModelUpsertRequest):
    models = load_dynamic_models()
    model_data = {
        "name": req.name,
        "provider": req.providerId,
        "type": req.type,
        **(req.metadata or {})
    }
    # 持久化 capabilities 和 tags 字段到动态存储
    if req.capabilities is not None:
        model_data["capabilities"] = req.capabilities
    if req.tags is not None:
        model_data["tags"] = req.tags
    models[req.modelId] = model_data
    save_dynamic_models(models)
    return {"success": True, "models": models}


@router.delete("/api/models/custom/{model_id}")
async def delete_custom_model(model_id: str):
    models = load_dynamic_models()
    if model_id in models:
        models.pop(model_id)
        save_dynamic_models(models)
    return {"success": True, "models": models}


# Helpers reused from app.py for model inference
import re


def _detect_model_type(model_id: str) -> str:
    lower_id = model_id.lower()
    if any(k in lower_id for k in ['embedding', 'embed', 'vector']):
        return 'embedding'
    if any(k in lower_id for k in ['rerank', 're-rank']):
        return 'rerank'
    if any(k in lower_id for k in ['vision', 'gpt-4', 'claude', 'gemini', 'gpt']):
        return 'chat'
    if re.search(r'image|img|diffusion|sd', lower_id):
        return 'image'
    else:
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
