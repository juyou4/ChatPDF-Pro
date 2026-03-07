"""Rerank provider registry."""

RERANK_PROVIDERS = {
    "local": {
        "name": "Local CrossEncoder",
        "requires_api_key": False,
        "default_model": "BAAI/bge-reranker-base",
        "endpoint": None
    },
    "cohere": {
        "name": "Cohere Rerank",
        "requires_api_key": True,
        "default_model": "rerank-multilingual-v3.0",
        "endpoint": "https://api.cohere.com/v1/rerank"
    },
    "jina": {
        "name": "Jina Rerank",
        "requires_api_key": True,
        "default_model": "jina-reranker-v2-base-multilingual",
        "endpoint": "https://api.jina.ai/v1/rerank"
    },
    "silicon": {
        "name": "SiliconFlow Rerank",
        "requires_api_key": True,
        "default_model": "BAAI/bge-reranker-v2-m3",
        "endpoint": "https://api.siliconflow.cn/v1/rerank"
    },
    "aliyun": {
        "name": "Aliyun Rerank",
        "requires_api_key": True,
        "default_model": "gte-rerank-v2",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/rerank"
    }
}
