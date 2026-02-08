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
    "text-embedding-v3": {
        "name": "Alibaba: text-embedding-v3",
        "provider": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding_endpoint": "/embeddings",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "$0.007/M tokens",
        "description": "Chinese optimized, cheapest"
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
    # Moonshot（moonshot-v1-8k 是 chat 模型，已移除；替换为正确的 embedding 模型）
    "moonshot-embedding-v1": {
        "name": "Moonshot: moonshot-embedding-v1",
        "provider": "openai",
        "base_url": "https://api.moonshot.cn/v1",
        "embedding_endpoint": "/embeddings",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "Paid",
        "description": "Moonshot AI 嵌入模型"
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
    # SiliconFlow - Qwen Embedding 8B（前端 systemModels.ts 中定义的模型）
    "Qwen/Qwen-Embedding-8B": {
        "name": "SiliconFlow: Qwen-Embedding-8B",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "embedding_endpoint": "/embeddings",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "Paid",
        "description": "阿里通义千问嵌入模型（硅基流动托管）"
    },
    # MiniMax（保留 embo-01 以兼容旧数据）
    "embo-01": {
        "name": "MiniMax: embo-01",
        "provider": "openai",
        "base_url": "https://api.minimax.chat/v1",
        "dimension": 1536,
        "max_tokens": 4096,
        "price": "Paid",
        "description": "MiniMax embedding (旧版模型 ID，保留兼容)"
    },
    # MiniMax - 与前端 systemModels.ts 同步的模型 ID
    "minimax-embedding-v2": {
        "name": "MiniMax: minimax-embedding-v2",
        "provider": "openai",
        "base_url": "https://api.minimax.chat/v1",
        "embedding_endpoint": "/embeddings",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "Paid",
        "description": "MiniMax 嵌入模型"
    },
    # DeepSeek - 与前端 systemModels.ts 同步
    "deepseek-embedding-v1": {
        "name": "DeepSeek: deepseek-embedding-v1",
        "provider": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "embedding_endpoint": "/embeddings",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "Paid",
        "description": "DeepSeek 嵌入模型"
    }
}

