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


# 预定义标签集合
PREDEFINED_TAGS = {
    "vision", "embedding", "rerank", "free", "reasoning",
    "function_calling", "web_search", "chinese_optimized"
}


def get_model_type_with_capabilities(
    model_id: str,
    capabilities: list[dict] | None = None
) -> str:
    """带 capabilities 优先级的模型类型检测

    优先级：
    1. capabilities 中 isUserSelected=true 的条目（用户手动指定）
    2. capabilities 中 isUserSelected=false 排除该类型（用户明确禁用）
    3. 正则检测（rerank 优先于 embedding）
    4. 默认返回 "chat"
    """
    # 检查用户覆盖：isUserSelected=true 的条目优先级最高
    if capabilities:
        for cap in capabilities:
            if cap.get("isUserSelected") is True:
                return cap["type"]

    # 收集用户禁用的类型：isUserSelected=false 排除该类型
    disabled_types = set()
    if capabilities:
        for cap in capabilities:
            if cap.get("isUserSelected") is False:
                disabled_types.add(cap["type"])

    # 正则检测（rerank 优先于 embedding，避免 "retrieval" 关键字误分类）
    if "rerank" not in disabled_types and is_rerank_model(model_id):
        return "rerank"
    if "embedding" not in disabled_types and is_embedding_model(model_id):
        return "embedding"

    return "chat"


def infer_model_tags(model_id: str) -> list[str]:
    """根据模型 ID 推断标签

    基于模型 ID 中的关键字匹配，自动推断模型能力标签。
    输出始终为 PREDEFINED_TAGS 的子集。
    """
    tags = []
    lower_id = model_id.lower()

    # 免费模型标签
    if "free" in lower_id:
        tags.append("free")

    # 视觉能力标签 — 增强版
    # 优先级：关键字匹配 → 系列匹配
    if "vision" in lower_id or "vl" in lower_id:
        tags.append("vision")
    elif re.search(r'^gpt-(4o|4-turbo|4\.1|5)', lower_id):
        # OpenAI GPT-4o、GPT-4 Turbo、GPT-4.1、GPT-5 系列均支持视觉
        tags.append("vision")
    elif re.search(r'^(o3|o4)(-|$)', lower_id):
        # OpenAI o3/o4 推理模型同样支持视觉输入
        tags.append("vision")
    elif re.search(r'^claude-(3|sonnet-4|opus-4|haiku-3|haiku-4)', lower_id):
        # Anthropic Claude 3+ 全系（含 claude-haiku-3.x）均支持视觉
        tags.append("vision")
    elif re.search(r'^gemini-(2|[3-9])', lower_id):
        # Google Gemini 2+ 系列均支持视觉
        tags.append("vision")
    elif re.search(r'^(qwen-vl|qwen-max)', lower_id):
        # 阿里云 Qwen-VL 和 Qwen-Max 系列
        tags.append("vision")
    elif re.search(r'^(grok-vision|grok-4)', lower_id):
        # xAI Grok 视觉模型和 Grok-4 系列
        tags.append("vision")
    elif re.search(r'^abab6\.5', lower_id):
        # MiniMax abab6.5 系列支持视觉
        tags.append("vision")
    elif re.search(r'^(doubao-1\.5-pro|doubao-seed)', lower_id):
        # 字节跳动豆包：1.5-Pro 及全部 Seed 系列（Seed 均为多模态）
        tags.append("vision")
    elif re.search(r'^moonshot-v1', lower_id):
        # Moonshot moonshot-v1 系列支持图片输入
        tags.append("vision")

    # 中文优化标签
    if any(k in lower_id for k in ["chinese", "zh", "multilingual"]):
        tags.append("chinese_optimized")

    # 推理能力标签 — 覆盖各家思考模型
    # DeepSeek: deepseek-reasoner, *-thinking
    # 智谱: glm-4.5/4.6/4.7 系列（均支持 thinking 参数）
    # Moonshot/Kimi: kimi-k2-thinking 等 thinking 变体
    # MiniMax: minimax-m2, m2.1（原生支持 reasoning_split）
    # OpenAI: o1, o3, o4, gpt-4.5, gpt-5 系列（reasoning_effort）
    # Anthropic: claude-*-thinking
    if any(k in lower_id for k in ["reasoning", "think", "reasoner"]):
        tags.append("reasoning")
    elif re.search(r'\b(o1|o3|o4)\b', lower_id):
        # OpenAI o 系列推理模型
        tags.append("reasoning")
    elif re.search(r'gpt-4\.5', lower_id):
        # OpenAI GPT-4.5 系列支持 reasoning_effort
        tags.append("reasoning")
    elif re.search(r'gpt-5', lower_id):
        # OpenAI GPT-5 系列（gpt-5, gpt-5-mini, gpt-5-nano）
        tags.append("reasoning")
    elif re.search(r'glm-4\.[5-9]', lower_id):
        # 智谱 GLM-4.5+ 系列支持思考模式
        tags.append("reasoning")
    elif re.search(r'minimax-m2', lower_id):
        # MiniMax M2 系列原生支持思考
        tags.append("reasoning")
    elif "deepseek-r" in lower_id:
        # DeepSeek-R1 等推理模型
        tags.append("reasoning")

    return tags
