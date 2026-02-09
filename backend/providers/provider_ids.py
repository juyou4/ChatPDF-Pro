OPENAI_LIKE = {"openai", "aliyun", "silicon", "moonshot", "deepseek", "zhipu", "minimax", "grok", "qwen", "doubao"}
ANTHROPIC = {"anthropic"}
GEMINI = {"gemini"}
OLLAMA = {"ollama"}

# 需要特殊思考参数的 provider 子集
# OpenAI 原生 API 使用 reasoning_effort 参数控制思考深度
OPENAI_NATIVE = {"openai"}
# MiniMax 使用 reasoning_split 参数开启思考分离
MINIMAX = {"minimax"}
# Moonshot/Kimi 的思考模型自动输出 reasoning_content，无需额外参数
MOONSHOT = {"moonshot"}
