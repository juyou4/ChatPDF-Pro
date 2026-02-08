"""模型 ID 统一解析器

负责将前端传入的模型 ID（composite key 或 plain key）解析为
后端 Model_Registry 中的键名和完整配置。

支持格式：
- composite key: "provider:modelId"（如 "silicon:BAAI/bge-m3"）
- plain key: 纯 "modelId"（如 "text-embedding-3-large"、"local-minilm"）
"""

from typing import Optional, Tuple

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

    # 1. 直接匹配：纯 modelId 或旧格式（如 "local-minilm"）
    if raw_model_id in EMBEDDING_MODELS:
        return raw_model_id, EMBEDDING_MODELS[raw_model_id]

    # 2. composite key 格式：provider:modelId
    if ":" in raw_model_id:
        provider_part, model_part = raw_model_id.split(":", 1)

        # 2a. model_part 直接匹配 EMBEDDING_MODELS 键
        if model_part in EMBEDDING_MODELS:
            config = EMBEDDING_MODELS[model_part]
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
        for key, config in EMBEDDING_MODELS.items():
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
    return list(EMBEDDING_MODELS.keys())
