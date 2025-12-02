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


router = APIRouter()


@router.get("/models")
async def get_models():
    """获取可用模型/Provider列表（含静态+动态）"""
    merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
    return merged


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


async def _fetch_models_with_fallback(api_host: str, api_key: str, endpoints: List[str]):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    last_error = None

    for ep in endpoints:
        if not ep:
            continue
        url = _build_endpoint(api_host, ep)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return response.json(), url
            last_error = f"HTTP {response.status_code}"
        except Exception as e:
            last_error = str(e)
            continue
    return None, last_error


@router.post("/api/providers/test")
async def test_provider_connection(request: ProviderTestRequest):
    """测试Provider连接"""
    try:
        if request.providerId == 'local':
            return {
                "success": True,
                "message": "本地模型无需连接测试",
                "availableModels": 2
            }

        endpoints = [request.fetchModelsEndpoint or "/models", "/v1/models", "/models"]
        data, last_error = await _fetch_models_with_fallback(request.apiHost, request.apiKey, endpoints)

        if data is not None:
            model_count = len(data.get('data', [])) if isinstance(data.get('data'), list) else 0
            return {
                "success": True,
                "message": "连接成功",
                "availableModels": model_count
            }

        return {
            "success": True,
            "message": f"连接成功（无法获取模型列表: {last_error or '无响应'})",
            "availableModels": 0
        }

    except httpx.ConnectError:
        return {"success": False, "message": "无法连接到API服务器，请检查网络或API地址"}
    except httpx.TimeoutException:
        return {"success": False, "message": "连接超时，请稍后重试"}
    except Exception as e:
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
        if request.providerType and request.providerType.lower() in {"anthropic", "gemini"}:
            return {
                "models": [],
                "providerId": request.providerId,
                "providerType": request.providerType,
                "timestamp": int(datetime.now().timestamp()),
                "message": "该提供商不支持自动拉取模型列表，请在前端手动选择/输入模型 ID"
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

        endpoints = [request.fetchModelsEndpoint or "/models", "/v1/models", "/models"]
        data, last_error = await _fetch_models_with_fallback(request.apiHost, request.apiKey, endpoints)

        if data is None:
            raise HTTPException(status_code=502, detail=f"获取模型失败: {last_error or '无响应'}")

        models = []
        if 'data' in data and isinstance(data['data'], list):
            for item in data['data']:
                model_id = item.get('id', '')
                model_type = _detect_model_type(model_id)
                model = {
                    "id": model_id,
                    "name": model_id,
                    "providerId": request.providerId,
                    "type": model_type,
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
            "timestamp": int(datetime.now().timestamp())
        }

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接到Provider API")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Provider API响应超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型失败：{str(e)}")


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


@router.get("/api/models/custom")
async def list_custom_models():
    return load_dynamic_models()


@router.post("/api/models/custom")
async def upsert_custom_model(req: ModelUpsertRequest):
    models = load_dynamic_models()
    models[req.modelId] = {
        "name": req.name,
        "provider": req.providerId,
        "type": req.type,
        **(req.metadata or {})
    }
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
