/**
 * 系统预设模型配置
 * 三层架构的中层：模型定义（与Provider分离）
 */

import type { Model } from '../types/model'

/**
 * 本地embedding模型
 */
export const LOCAL_MODELS: Model[] = [
    {
        id: 'all-MiniLM-L6-v2',
        name: 'MiniLM-L6-v2',
        providerId: 'local',
        type: 'embedding',
        metadata: {
            dimension: 384,
            maxTokens: 256,
            description: '快速通用模型，适合大多数场景'
        },
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'paraphrase-multilingual-MiniLM-L12-v2',
        name: 'Multilingual MiniLM-L12-v2',
        providerId: 'local',
        type: 'embedding',
        metadata: {
            dimension: 384,
            maxTokens: 128,
            description: '多语言支持，特别是中文效果更好'
        },
        isSystem: true,
        isUserAdded: false
    }
]

/**
 * OpenAI模型
 */
export const OPENAI_MODELS: Model[] = [
    {
        id: 'text-embedding-3-large',
        name: 'text-embedding-3-large',
        providerId: 'openai',
        type: 'embedding',
        metadata: {
            dimension: 3072,
            maxTokens: 8191,
            description: '最高质量'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.13,
            currency: 'USD'
        }
    },
    {
        id: 'text-embedding-3-small',
        name: 'text-embedding-3-small',
        providerId: 'openai',
        type: 'embedding',
        metadata: {
            dimension: 1536,
            maxTokens: 8191,
            description: '性价比最高'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.02,
            currency: 'USD'
        }
    }
]

/**
 * 阿里云模型
 */
export const ALIYUN_MODELS: Model[] = [
    {
        id: 'text-embedding-v3',
        name: 'text-embedding-v3',
        providerId: 'aliyun',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: '中文优化，价格最便宜'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.007,
            currency: 'USD'
        }
    }
]

/**
 * 硅基流动模型
 */
export const SILICON_MODELS: Model[] = [
    {
        id: 'BAAI/bge-m3',
        name: 'BAAI/bge-m3',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: '开源，托管在硅基流动'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.02,
            currency: 'USD'
        }
    },
    {
        id: 'Qwen/Qwen-Embedding-8B',
        name: 'Qwen Embedding 8B',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: '阿里通义千问嵌入模型'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.28,
            currency: 'CNY'
        }
    },
    {
        id: 'BAAI/bge-reranker-v2-m3',
        name: 'BGE Reranker v2-M3',
        providerId: 'silicon',
        type: 'rerank',
        metadata: {
            dimension: 0,
            maxTokens: 8192,
            description: '重排模型，用于结果重新排序'
        },
        isSystem: true,
        isUserAdded: false
    }
]

/**
 * Moonshot模型
 */
export const MOONSHOT_MODELS: Model[] = [
    {
        id: 'moonshot-embedding-v1',
        name: 'moonshot-embedding-v1',
        providerId: 'moonshot',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: 'Moonshot AI 嵌入模型'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.011,
            currency: 'USD'
        }
    }
]

/**
 * DeepSeek模型
 */
export const DEEPSEEK_MODELS: Model[] = [
    {
        id: 'deepseek-embedding-v1',
        name: 'deepseek-embedding-v1',
        providerId: 'deepseek',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: 'DeepSeek 嵌入模型'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.01,
            currency: 'USD'
        }
    }
]

/**
 * Chat 模型（对话）
 */
export const CHAT_MODELS: Model[] = [
    // ── OpenAI ──
    {
        id: 'gpt-4.1',
        name: 'GPT-4.1',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 最强非推理模型，1M Token 上下文，指令遵循能力大幅提升' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-4.1-mini',
        name: 'GPT-4.1 mini',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'GPT-4.1 轻量版，速度快成本低，1M Token 上下文' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'o3',
        name: 'OpenAI o3',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 强推理模型，适合数学、科学、代码复杂任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'o4-mini',
        name: 'OpenAI o4-mini',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 快速推理模型，高吞吐量高性价比' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-4o',
        name: 'GPT-4o',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 多模态旗舰（前代）' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-4o-mini',
        name: 'GPT-4o mini',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'GPT-4o 轻量版，高性价比' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    // ── 阿里云 通义千问 ──
    {
        id: 'qwen3-max',
        name: 'Qwen3-Max',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '通义千问旗舰模型，1T+ 参数，全球排名第三' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen3.5-plus',
        name: 'Qwen3.5-Plus',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '最新 Qwen3.5 均衡版，性能优于 Qwen3-Max，成本更低' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen-plus',
        name: 'Qwen-Plus',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '千问均衡版，性能与成本兼顾' },
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen-turbo',
        name: 'Qwen-Turbo',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '千问快速版，低延迟低成本' },
        isSystem: true, isUserAdded: false
    },
    // ── DeepSeek ──
    {
        id: 'deepseek-chat',
        name: 'DeepSeek V3',
        providerId: 'deepseek',
        type: 'chat',
        metadata: { description: 'DeepSeek 旗舰对话模型 (V3)' },
        isSystem: true, isUserAdded: false
    },
    {
        id: 'deepseek-reasoner',
        name: 'DeepSeek R1',
        providerId: 'deepseek',
        type: 'chat',
        metadata: { description: 'DeepSeek 推理增强模型 (R1)' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    // ── Moonshot (Kimi) ──
    {
        id: 'kimi-k2.5',
        name: 'Kimi K2.5',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi 最新旗舰，1T 参数 MoE，支持 Agent Swarm 并行协作' },
        isSystem: true, isUserAdded: false
    },
    {
        id: 'kimi-k2',
        name: 'Kimi K2',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi K2 开源 MoE 模型，256K 上下文，强 Agent 能力' },
        isSystem: true, isUserAdded: false
    },
    {
        id: 'moonshot-v1-128k',
        name: 'Moonshot v1 128K',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi 128K 超长上下文' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'moonshot-v1-32k',
        name: 'Moonshot v1 32K',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi 32K 长上下文' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'moonshot-v1-8k',
        name: 'Moonshot v1 8K',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi 通用 8K' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    // ── 智谱 GLM ──
    {
        id: 'glm-5',
        name: 'GLM-5',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱最新旗舰，744B MoE，全球开源第一，2026-02 发布' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-4.7',
        name: 'GLM-4.7',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱编程增强模型，Agentic Coding 能力优化' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-4.5',
        name: 'GLM-4.5',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱 Agent 基座模型，推理+代码+Agentic 原生融合' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-4.5-air',
        name: 'GLM-4.5-Air',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: 'GLM-4.5 轻量版，高性价比' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-4-air',
        name: 'GLM-4-Air',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: 'GLM-4 轻量版（前代）' },
        isSystem: true, isUserAdded: false
    },
    // ── MiniMax ──
    {
        id: 'MiniMax-Text-01',
        name: 'MiniMax Text-01',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax 旗舰对话模型，百万长上下文' },
        isSystem: true, isUserAdded: false
    },
    {
        id: 'abab6.5s-chat',
        name: 'abab6.5s-chat',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax 轻量聊天（前代）' },
        isSystem: true, isUserAdded: false
    },
    // ── Anthropic (Claude) ──
    {
        id: 'claude-opus-4-6',
        name: 'Claude Opus 4.6',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Anthropic 旗舰模型，200K 上下文，最强编程与推理，支持 1M 上下文 (beta)' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-sonnet-4-6',
        name: 'Claude Sonnet 4.6',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Anthropic 最新均衡模型，Opus 级别推理能力，200K 上下文，同等价格' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-opus-4-5',
        name: 'Claude Opus 4.5',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Claude Opus 系列前代，超强编程、Agent 工作流' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-sonnet-4-5',
        name: 'Claude Sonnet 4.5',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Claude 均衡前代版本，高性价比' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-haiku-3-5',
        name: 'Claude Haiku 3.5',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: '最快速轻量 Claude 模型，低成本高并发' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    // ── Google (Gemini) ──
    {
        id: 'gemini-3-pro',
        name: 'Gemini 3 Pro',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Google 最新旗舰推理模型，1M 上下文，自适应思考，强多模态 (preview)' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-3-flash',
        name: 'Gemini 3 Flash',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Google 最新多模态理解模型，强编程与推理 (preview)' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-2.5-pro',
        name: 'Gemini 2.5 Pro',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Gemini 旗舰稳定版，1M 上下文，自适应思考' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-2.5-flash',
        name: 'Gemini 2.5 Flash',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Gemini 快速均衡版，可控推理预算' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-2.5-flash-lite',
        name: 'Gemini 2.5 Flash-Lite',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Gemini 超轻量版，大规模低成本场景' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    // ── xAI (Grok) ──
    {
        id: 'grok-4',
        name: 'Grok 4',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'xAI 最新旗舰，256K 上下文，原生工具调用与实时搜索' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'grok-4-1-fast',
        name: 'Grok 4.1 Fast',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'Grok 4.1 快速版，2M 上下文，强 Agent 工具调用' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'grok-3',
        name: 'Grok 3',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'Grok 3 旗舰，强推理能力' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'grok-3-mini',
        name: 'Grok 3 Mini',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'Grok 3 轻量版，低成本快速响应' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    // ── 硬基流动 (SiliconFlow) ──
    {
        id: 'deepseek-ai/DeepSeek-R1',
        name: 'DeepSeek R1 (SiliconFlow)',
        providerId: 'silicon',
        type: 'chat',
        metadata: { description: '托管于硅基流动的 DeepSeek R1 推理模型' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'deepseek-ai/DeepSeek-V3',
        name: 'DeepSeek V3 (SiliconFlow)',
        providerId: 'silicon',
        type: 'chat',
        metadata: { description: '托管于硅基流动的 DeepSeek V3' },
        isSystem: true, isUserAdded: false
    },
    {
        id: 'Qwen/Qwen3-235B-A22B',
        name: 'Qwen3-235B (SiliconFlow)',
        providerId: 'silicon',
        type: 'chat',
        metadata: { description: '托管于硅基流动的通义千问3 旗舰开源版' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'Qwen/Qwen2.5-7B-Instruct',
        name: 'Qwen2.5 7B (SiliconFlow)',
        providerId: 'silicon',
        type: 'chat',
        metadata: { description: '托管于硅基流动的通义 2.5 7B 指令模型' },
        isSystem: true, isUserAdded: false
    }
]

/**
 * 智谱模型
 */
export const ZHIPU_MODELS: Model[] = [
    {
        id: 'embedding-3',
        name: 'Embedding-3',
        providerId: 'zhipu',
        type: 'embedding',
        metadata: {
            dimension: 2048,
            maxTokens: 8192,
            description: '智谱 GLM 嵌入模型'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.014,
            currency: 'USD'
        }
    }
]

/**
 * 字节跳动 豆包模型
 */
export const DOUBAO_MODELS: Model[] = [
    {
        id: 'doubao-seed-2-0-pro',
        name: 'Doubao Seed 2.0 Pro',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 2.0 旗舰模型，对标 GPT-5.2 / Gemini 3 Pro，支持长链路推理与多模态'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-seed-2-0-lite-260215',
        name: 'Doubao Seed 2.0 Lite',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 2.0 Lite，均衡性能与成本，能力超越上一代豆包 1.8'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-seed-2-0-mini-260215',
        name: 'Doubao Seed 2.0 Mini',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 2.0 Mini，低延迟高并发，适合成本敏感场景'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-seed-2-0-code-preview-260215',
        name: 'Doubao Seed 2.0 Code',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 2.0 编程专项模型，深度优化 Agentic Coding 场景'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-seed-1-8',
        name: 'Doubao Seed 1.8',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 1.8，上一代主力模型，多模态 Agent 场景优化'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-1-5-pro-32k-250115',
        name: 'Doubao 1.5 Pro 32K',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 1.5 Pro，32K 上下文'
        },
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-embedding-large-250104',
        name: 'Doubao Embedding Large',
        providerId: 'doubao',
        type: 'embedding',
        metadata: {
            dimension: 4096,
            maxTokens: 32768,
            description: '豆包大尺寸嵌入模型'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.0005,
            currency: 'CNY'
        }
    },
    {
        id: 'doubao-embedding-250104',
        name: 'Doubao Embedding',
        providerId: 'doubao',
        type: 'embedding',
        metadata: {
            dimension: 2048,
            maxTokens: 32768,
            description: '豆包标准嵌入模型'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.0005,
            currency: 'CNY'
        }
    }
]

/**
 * MiniMax模型
 */
export const MINIMAX_MODELS: Model[] = [
    {
        id: 'minimax-embedding-v2',
        name: 'minimax-embedding-v2',
        providerId: 'minimax',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: 'MiniMax 嵌入模型'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.014,
            currency: 'USD'
        }
    }
]

/**
 * 所有系统预设模型
 */
export const SYSTEM_MODELS: Model[] = [
    ...LOCAL_MODELS,
    ...OPENAI_MODELS,
    ...ALIYUN_MODELS,
    ...SILICON_MODELS,
    ...MOONSHOT_MODELS,
    ...DEEPSEEK_MODELS,
    ...ZHIPU_MODELS,
    ...MINIMAX_MODELS,
    ...DOUBAO_MODELS,
    ...CHAT_MODELS
]

/**
 * 根据provider ID获取系统模型
 */
export function getSystemModelsByProvider(providerId: string): Model[] {
    return SYSTEM_MODELS.filter(m => m.providerId === providerId)
}

/**
 * 根据模型ID获取系统模型
 */
export function getSystemModelById(modelId: string): Model | undefined {
    return SYSTEM_MODELS.find(m => m.id === modelId)
}

/**
 * 获取所有embedding类型的系统模型
 */
export function getSystemEmbeddingModels(): Model[] {
    return SYSTEM_MODELS.filter(m => m.type === 'embedding')
}

/**
 * 获取所有rerank类型的系统模型
 */
export function getSystemRerankModels(): Model[] {
    return SYSTEM_MODELS.filter(m => m.type === 'rerank')
}
