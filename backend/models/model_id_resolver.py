"""模型 ID 统一解析器

负责将前端传入的模型 ID（composite key 或 plain key）解析为
后端 Model_Registry 中的键名和完整配置。

支持格式：
- composite key: "provider:modelId"（如 "silicon:BAAI/bge-m3"）
- plain key: 纯 "modelId"（如 "text-embedding-3-large"、"local-minilm"）
"""

from typing import Optional, Tuple

from models.dynamic_store import load_dynamic_models
from models.model_registry import EMBEDDING_MODELS

# 前端 providerId → 后端 provider 字段的映射
# 解决 aliyun vs openai 等语义差异：
# 许多国内服务商在后端使用 OpenAI 兼容接口，provider 字段统一标记为 "openai"
PROVIDER_ALIAS_MAP = {
    "aliyun": ["openai"],      # 阿里云在后端标记为 openai（OpenAI 兼容）
    "silicon": ["openai"],     # 硅基流动同理
    "moonshot": ["openai"],
    "deepseek": ["openai"],
    "zhipu": ["openai"],
    "minimax": ["openai"],
    "local": ["local"],
    "openai": ["openai"],
}

# 前端 providerId → 后端 base_url 中的域名关键字映射
# 用于在 provider 字段相同（都是 "openai"）时，通过 base_url 区分不同服务商
PROVIDER_BASE_URL_HINTS = {
    "aliyun": "dashscope.aliyuncs.com",
    "silicon": "siliconflow.cn",
    "moonshot": "moonshot.cn",
    "deepseek": "deepseek.com",
    "zhipu": "bigmodel.cn",
    "minimax": "minimax",
    "openai": "openai.com",
}

# 废弃模型 ID 映射：用于兼容旧安装包/旧配置中的历史模型键名
DEPRECATED_MODEL_ID_ALIASES = {
    # SiliconFlow 旧 ID（已逐步下线） -> 新 ID
    "Qwen/Qwen-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    # OpenAI 经典 embedding 旧 ID（兼容迁移到 3-small）
    "text-embedding-ada-002": "text-embedding-3-small",
    # MiniMax 旧 embedding ID
    "embo-01": "minimax-embedding-v2",
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


def _normalize_dynamic_embedding_config(model_id: str, config: dict) -> Optional[dict]:
    """将动态模型记录归一化为 embedding 调用链可直接消费的格式。"""
    if not isinstance(config, dict):
        return None

    model_type = str(config.get("type") or "").lower()
    capabilities = config.get("capabilities") or []
    has_embedding_capability = any(
        isinstance(cap, dict)
        and cap.get("type") == "embedding"
        and cap.get("isUserSelected") is not False
        for cap in capabilities
    )
    if model_type and model_type != "embedding" and not has_embedding_capability:
        return None

    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    provider_id = _pick_first(
        config.get("provider_id"),
        config.get("providerId"),
        metadata.get("provider_id"),
        metadata.get("providerId"),
        config.get("provider"),
    )
    provider_type = _pick_first(
        config.get("provider_type"),
        config.get("providerType"),
        metadata.get("provider_type"),
        metadata.get("providerType"),
    )
    if not provider_type and provider_id:
        provider_type = PROVIDER_ALIAS_MAP.get(provider_id, [provider_id])[0]

    normalized = dict(config)
    normalized["provider_id"] = provider_id or normalized.get("provider_id")
    normalized["provider"] = provider_type or normalized.get("provider") or "openai"

    field_aliases = {
        "base_url": ["base_url", "baseUrl", "api_host", "apiHost"],
        "model_name": ["model_name", "modelName"],
        "embedding_endpoint": ["embedding_endpoint", "embeddingEndpoint"],
        "max_tokens": ["max_tokens", "maxTokens"],
        "dimension": ["dimension"],
        "description": ["description"],
        "price": ["price"],
    }
    for target_field, aliases in field_aliases.items():
        values = [normalized.get(alias) for alias in aliases]
        values.extend(metadata.get(alias) for alias in aliases)
        picked = _pick_first(*values)
        if picked is not None:
            normalized[target_field] = picked

    normalized.setdefault("model_name", model_id)
    return normalized


def _load_embedding_models() -> dict:
    """加载静态 + 动态 embedding 模型并做统一归一化。"""
    merged = dict(EMBEDDING_MODELS)
    for model_id, config in load_dynamic_models().items():
        normalized = _normalize_dynamic_embedding_config(model_id, config)
        if normalized is not None:
            merged[model_id] = normalized
    return merged


def normalize_deprecated_model_id(raw_model_id: str) -> str:
    """将历史模型 ID 归一到当前可用模型 ID。

    支持 plain key 和 composite key（provider:modelId）两种输入。
    """
    if not raw_model_id:
        return raw_model_id

    if ":" in raw_model_id:
        provider_part, model_part = raw_model_id.split(":", 1)
        mapped_model_part = DEPRECATED_MODEL_ID_ALIASES.get(model_part, model_part)
        return f"{provider_part}:{mapped_model_part}"

    return DEPRECATED_MODEL_ID_ALIASES.get(raw_model_id, raw_model_id)


def resolve_model_id(
    raw_model_id: str,
) -> Tuple[Optional[str], Optional[dict]]:
    """解析模型 ID，返回 (registry_key, config) 或 (None, None)

    解析策略（按优先级）：
    1. 直接匹配：raw_model_id 作为键在 EMBEDDING_MODELS 中查找
    2. composite key 解析：拆分 "provider:modelId"
       2a. model_part 直接匹配 EMBEDDING_MODELS 键
       2b. 通过 provider 的 base_url hint 辅助定位

    Args:
        raw_model_id: 前端传入的模型 ID，支持 "provider:modelId" 或纯 "modelId" 格式

    Returns:
        (registry_key, config) 元组，解析失败时返回 (None, None)
    """
    if not raw_model_id:
        return None, None

    raw_model_id = normalize_deprecated_model_id(raw_model_id)
    embedding_models = _load_embedding_models()

    # 1. 直接匹配：纯 modelId 或旧格式（如 "local-minilm"）
    if raw_model_id in embedding_models:
        return raw_model_id, embedding_models[raw_model_id]

    # 2. composite key 格式：provider:modelId
    if ":" in raw_model_id:
        provider_part, model_part = raw_model_id.split(":", 1)

        # 2a. model_part 直接匹配 EMBEDDING_MODELS 键
        if model_part in embedding_models:
            config = embedding_models[model_part]
            # 验证 provider 兼容性（宽松模式：即使不匹配也返回结果）
            backend_provider = config.get("provider", "")
            expected_providers = PROVIDER_ALIAS_MAP.get(provider_part, [provider_part])
            if backend_provider in expected_providers:
                return model_part, config
            # provider 不匹配但 model_part 唯一存在，仍然返回（宽松模式）
            return model_part, config

        # 2b. 通过 model_name + provider 兼容性匹配
        # 适用于 model_part 不是注册表键名，但可能是 model_name 的情况
        # 例如本地模型：键名为 "local-minilm"，但 model_name 为 "all-MiniLM-L6-v2"
        expected_providers = PROVIDER_ALIAS_MAP.get(provider_part, [provider_part])
        base_url_hint = PROVIDER_BASE_URL_HINTS.get(provider_part, "")
        for key, config in embedding_models.items():
            model_name = config.get("model_name", key)
            backend_provider = config.get("provider", "")
            base_url = config.get("base_url", "")
            # model_name 或 key 匹配 model_part
            if model_name == model_part or key == model_part:
                # 验证 provider 兼容性
                if backend_provider in expected_providers:
                    # 如果有 base_url_hint，进一步验证 base_url 匹配
                    if base_url_hint:
                        if base_url_hint in base_url:
                            return key, config
                    else:
                        # 无 base_url_hint（如本地模型），仅通过 provider 匹配
                        return key, config

    return None, None


def get_available_model_ids() -> list:
    """返回所有可用的模型 ID 列表，用于错误提示

    Returns:
        EMBEDDING_MODELS 中所有注册的模型键名列表
    """
    return list(_load_embedding_models().keys())
