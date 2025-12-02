import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from providers.factory import ProviderFactory
from providers.openai_provider import OpenAICompatibleProvider
from providers.ollama_provider import OllamaProvider


def test_factory_openai():
    client = ProviderFactory.create("openai")
    assert isinstance(client, OpenAICompatibleProvider)


def test_factory_ollama():
    client = ProviderFactory.create("ollama")
    assert isinstance(client, OllamaProvider)


def test_factory_fallback():
    client = ProviderFactory.create("unknown")
    assert isinstance(client, OpenAICompatibleProvider)
