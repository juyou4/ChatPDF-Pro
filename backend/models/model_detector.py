import re
from typing import Optional

# Embedding and rerank model detection regex (inspired by cherry-studio)
EMBEDDING_REGEX = re.compile(r'(?:^text-|embed|bge-|e5-|LLM2Vec|retrieval|uae-|gte-|jina-clip|jina-embeddings|voyage-)', re.I)
RERANKING_REGEX = re.compile(r'(?:rerank|re-rank|re-ranker|re-ranking|retrieval|retriever)', re.I)


def is_embedding_model(model_id: str) -> bool:
    """Check if a model ID is an embedding model using regex"""
    if not model_id:
        return False
    if is_rerank_model(model_id):
        return False
    return bool(EMBEDDING_REGEX.search(model_id.lower()))


def is_rerank_model(model_id: str) -> bool:
    """Check if a model ID is a rerank model using regex"""
    if not model_id:
        return False
    return bool(RERANKING_REGEX.search(model_id.lower()))


def get_model_provider(model_id: str) -> str:
    """Infer provider from model ID"""
    if not model_id:
        return "openai"

    model_id_lower = model_id.lower()

    if "doubao" in model_id_lower:
        return "doubao"
    if "moonshot" in model_id_lower or "kimi" in model_id_lower:
        return "moonshot"
    if "zhipu" in model_id_lower or "glm" in model_id_lower:
        return "zhipu"
    if "minimax" in model_id_lower:
        return "minimax"
    if "qwen" in model_id_lower or "alibaba" in model_id_lower:
        return "openai"
    if model_id_lower.startswith("gpt") or model_id_lower.startswith("text-embedding"):
        return "openai"
    if model_id_lower.startswith("claude"):
        return "anthropic"
    if model_id_lower.startswith("gemini"):
        return "gemini"
    return "openai"


def normalize_embedding_model_id(embedding_model_id: Optional[str]) -> Optional[str]:
    """Normalize embedding model id to a configured key (supports provider:model or plain id)"""
    if not embedding_model_id:
        return None
    return embedding_model_id
