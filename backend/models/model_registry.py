"""集中管理模型配置"""

EMBEDDING_MODELS = {
    "local-minilm": {
        "name": "Local: MiniLM-L6 (Default)",
        "provider": "local",
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "max_tokens": 256,
        "price": "Free (Local)",
        "description": "Fast, general purpose",
        "embedding_endpoint": None
    },
    "local-multilingual": {
        "name": "Local: Multilingual",
        "provider": "local",
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "max_tokens": 128,
        "price": "Free (Local)",
        "description": "Better for Chinese/multilingual"
    },
    # OpenAI
    "text-embedding-3-large": {
        "name": "OpenAI: text-embedding-3-large",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "embedding_endpoint": "/embeddings",
        "dimension": 3072,
        "max_tokens": 8191,
        "price": "$0.13/M tokens",
        "description": "Best overall quality"
    },
    "text-embedding-3-small": {
        "name": "OpenAI: text-embedding-3-small",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "dimension": 1536,
        "max_tokens": 8191,
        "price": "$0.02/M tokens",
        "description": "Best value"
    },
    # Alibaba
    "text-embedding-v4": {
        "name": "Alibaba: text-embedding-v4",
        "provider": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding_endpoint": "/embeddings",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "CNY 0.000514/1K tokens",
        "description": "Qwen3-Embedding based multilingual embedding model"
    },
    "qwen3.7-text-embedding": {
        "name": "Alibaba: qwen3.7-text-embedding",
        "provider": "openai",
        "provider_id": "aliyun",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding_endpoint": "/embeddings",
        "dimension": 1024,
        "max_tokens": 131072,
        "price": "USD 0.07/M tokens",
        "description": "Qwen3.7 多语言文本向量模型，支持 256-2560 维自定义输出"
    },
    "qwen3-rerank": {
        "name": "Alibaba: qwen3-rerank",
        "provider": "openai",
        "type": "rerank",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "rerank_endpoint": "/rerank",
        "max_tokens": 4000,
        "price": "CNY 0.0005/1K tokens",
        "description": "Qwen3 multilingual rerank model"
    },
    # SiliconFlow
    "BAAI/bge-m3": {
        "name": "SiliconFlow: BGE-M3",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "Free (Limited)",
        "description": "State-of-the-art multilingual"
    },
    "BAAI/bge-large-zh-v1.5": {
        "name": "SiliconFlow: BGE-Large-ZH",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "dimension": 1024,
        "max_tokens": 512,
        "price": "Free (Limited)",
        "description": "Best for Chinese"
    },
    "Pro/BAAI/bge-m3": {
        "name": "SiliconFlow Pro: BGE-M3",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "Paid",
        "description": "High performance BGE-M3"
    },
    # Zhipu
    "embedding-3": {
        "name": "Zhipu: embedding-3",
        "provider": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "dimension": 2048,
        "max_tokens": 8192,
        "price": "Paid",
        "description": "ChatGLM embedding"
    },
    # SiliconFlow - BGE-Large-EN
    "BAAI/bge-large-en-v1.5": {
        "name": "SiliconFlow: BGE-Large-EN",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "dimension": 1024,
        "max_tokens": 512,
        "price": "Free (Limited)",
        "description": "English embedding model"
    },
    # SiliconFlow - BCE Embedding
    "netease-youdao/bce-embedding-base_v1": {
        "name": "SiliconFlow: BCE Embedding",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "dimension": 768,
        "max_tokens": 512,
        "price": "Free (Limited)",
        "description": "NetEase Youdao cross-lingual embedding"
    },
    # SiliconFlow - Qwen embeddings (OpenAI兼容)
    "Qwen/Qwen3-Embedding-8B": {
        "name": "SiliconFlow: Qwen3-Embedding-8B",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "Free/Limited",
        "description": "Hosted Qwen3 embedding (SiliconFlow)"
    },
    "Qwen/Qwen3-Embedding-4B": {
        "name": "SiliconFlow: Qwen3-Embedding-4B",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "Paid",
        "description": "Qwen3 embedding 4B, balanced performance"
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "name": "SiliconFlow: Qwen3-Embedding-0.6B",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "Paid",
        "description": "Qwen3 embedding 0.6B, lightweight"
    },
    # MiniMax - 官方 Embeddings API
    "embo-01": {
        "name": "MiniMax: embo-01",
        "provider": "openai",
        "provider_id": "minimax",
        "type": "embedding",
        "base_url": "https://api.minimax.chat/v1",
        "embedding_endpoint": "/embeddings",
        "max_tokens": 4096,
        "price": "Paid",
        "description": "MiniMax official text embedding model"
    },
    # Gemini OpenAI compatibility embeddings
    "gemini-embedding-2-preview": {
        "name": "Gemini: gemini-embedding-2-preview",
        "provider": "openai",
        "provider_id": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "embedding_endpoint": "/embeddings",
        "max_tokens": 8192,
        "price": "Preview",
        "description": "Google multimodal embedding preview model"
    },
    "gemini-embedding-001": {
        "name": "Gemini: gemini-embedding-001",
        "provider": "openai",
        "provider_id": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "embedding_endpoint": "/embeddings",
        "max_tokens": 2048,
        "price": "Paid",
        "description": "Google stable text embedding model"
    }
}
