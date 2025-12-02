"""Provider registry with endpoints for single and integrated providers."""

PROVIDER_CONFIG = {
    "openai": {
        "name": "OpenAI",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "type": "openai"
    },
    "anthropic": {
        "name": "Anthropic",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "type": "anthropic"
    },
    "gemini": {
        "name": "Google Gemini",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "type": "gemini"
    },
    "grok": {
        "name": "xAI Grok",
        "endpoint": "https://api.x.ai/v1/chat/completions",
        "type": "openai"
    },
    "ollama": {
        "name": "Ollama (local)",
        "endpoint": "http://localhost:11434/api/chat",
        "type": "ollama"
    },
    # Integrated providers (OpenAI-compatible)
    "silicon": {
        "name": "SiliconFlow",
        "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
        "type": "openai"
    },
    "aliyun": {
        "name": "Aliyun DashScope",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "type": "openai"
    },
    "moonshot": {
        "name": "Moonshot",
        "endpoint": "https://api.moonshot.cn/v1/chat/completions",
        "type": "openai"
    },
    "deepseek": {
        "name": "DeepSeek",
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "type": "openai"
    },
    "zhipu": {
        "name": "Zhipu",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "type": "openai"
    },
    "minimax": {
        "name": "MiniMax",
        "endpoint": "https://api.minimaxi.com/v1/chat/completions",
        "type": "openai"
    },
    "qwen": {
        "name": "Qwen",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "type": "openai"
    },
    "doubao": {
        "name": "Doubao",
        "endpoint": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "type": "openai"
    }
}

