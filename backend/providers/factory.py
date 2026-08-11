from typing import Optional

from .base import BaseProvider
from .openai_provider import OpenAICompatibleProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .grok_provider import GrokProvider
from .ollama_provider import OllamaProvider
from .provider_ids import OPENAI_LIKE, ANTHROPIC, GEMINI, OLLAMA
from models.provider_registry import PROVIDER_CONFIG


class ProviderFactory:
    """简单的 Provider 工厂"""

    @staticmethod
    def create(
        provider_id: str,
        endpoint: Optional[str] = None,
        *,
        provider_type: Optional[str] = None,
        api_key_header: Optional[str] = None,
        api_key_prefix: Optional[str] = None,
    ) -> BaseProvider:
        pid = (provider_id or "").lower()
        default_endpoint = PROVIDER_CONFIG.get(pid, {}).get("endpoint", "")
        endpoint = endpoint or default_endpoint
        protocol = (provider_type or PROVIDER_CONFIG.get(pid, {}).get("type") or pid).strip().lower()
        # 动态 Provider 的 ID 不在固定白名单中，协议类型必须优先于 ID
        # 判断，否则自定义 Anthropic 网关会误走 OpenAI 兼容请求体。
        if protocol == "anthropic":
            return AnthropicProvider(endpoint, api_key_header, api_key_prefix)
        if pid in OPENAI_LIKE:
            return OpenAICompatibleProvider(endpoint, api_key_header, api_key_prefix)
        if pid in ANTHROPIC:
            return AnthropicProvider(endpoint, api_key_header, api_key_prefix)
        if pid in GEMINI:
            return GeminiProvider()
        if pid == "grok":
            return GrokProvider()
        if pid in OLLAMA:
            return OllamaProvider()
        # 其他视为 OpenAI 兼容
        return OpenAICompatibleProvider(endpoint, api_key_header, api_key_prefix)
