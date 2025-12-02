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
    def create(provider_id: str, endpoint: Optional[str] = None) -> BaseProvider:
        pid = (provider_id or "").lower()
        default_endpoint = PROVIDER_CONFIG.get(pid, {}).get("endpoint", "")
        endpoint = endpoint or default_endpoint
        if pid in OPENAI_LIKE:
            return OpenAICompatibleProvider(endpoint)
        if pid in ANTHROPIC:
            return AnthropicProvider()
        if pid in GEMINI:
            return GeminiProvider()
        if pid == "grok":
            return GrokProvider()
        if pid in OLLAMA:
            return OllamaProvider()
        # 其他视为 OpenAI 兼容
        return OpenAICompatibleProvider(endpoint)
