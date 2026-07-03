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
        id: 'text-embedding-v4',
        name: 'text-embedding-v4',
        providerId: 'aliyun',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: 'Qwen3-Embedding 系列，默认 1024 维，支持 100+ 语言'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.000514,
            currency: 'CNY'
        }
    },
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
    },
    {
        id: 'qwen3-rerank',
        name: 'qwen3-rerank',
        providerId: 'aliyun',
        type: 'rerank',
        metadata: {
            maxTokens: 4000,
            description: '通义千问第三代文本重排模型，支持 100+ 语言，最多 500 个文档'
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
 * 硅基流动模型
 */
export const SILICON_MODELS: Model[] = [
    //  Embedding 模型 
    {
        id: 'BAAI/bge-m3',
        name: 'BAAI/bge-m3',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: '多语言多功能嵌入模型，支持稠密/稀疏/多向量检索'
        },
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'Pro/BAAI/bge-m3',
        name: 'Pro/BAAI/bge-m3',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: 'bge-m3 加速版，更快推理速度'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.07,
            currency: 'CNY'
        }
    },
    {
        id: 'BAAI/bge-large-zh-v1.5',
        name: 'BAAI/bge-large-zh-v1.5',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 512,
            description: '中文大型嵌入模型，中文场景效果最佳'
        },
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'BAAI/bge-large-en-v1.5',
        name: 'BAAI/bge-large-en-v1.5',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 512,
            description: '英文大型嵌入模型'
        },
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'Qwen/Qwen3-Embedding-8B',
        name: 'Qwen3 Embedding 8B',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: '通义千问第三代嵌入模型，8B 参数，效果最强'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.28,
            currency: 'CNY'
        }
    },
    {
        id: 'Qwen/Qwen3-Embedding-4B',
        name: 'Qwen3 Embedding 4B',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: '通义千问第三代嵌入模型，4B 参数，均衡性能与成本'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.14,
            currency: 'CNY'
        }
    },
    {
        id: 'Qwen/Qwen3-Embedding-0.6B',
        name: 'Qwen3 Embedding 0.6B',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 8192,
            description: '通义千问第三代嵌入模型，0.6B 参数，轻量低成本'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.07,
            currency: 'CNY'
        }
    },
    {
        id: 'netease-youdao/bce-embedding-base_v1',
        name: 'BCE Embedding Base v1',
        providerId: 'silicon',
        type: 'embedding',
        metadata: {
            dimension: 768,
            maxTokens: 512,
            description: '网易有道跨语言嵌入模型，中英文效果好'
        },
        isSystem: true,
        isUserAdded: false
    },
    //  Rerank 模型 
    {
        id: 'BAAI/bge-reranker-v2-m3',
        name: 'BGE Reranker v2-M3',
        providerId: 'silicon',
        type: 'rerank',
        metadata: {
            maxTokens: 8192,
            description: 'BGE 多语言重排模型，用于结果重新排序'
        },
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'Qwen/Qwen3-Reranker-8B',
        name: 'Qwen3 Reranker 8B',
        providerId: 'silicon',
        type: 'rerank',
        metadata: {
            maxTokens: 8192,
            description: '通义千问第三代重排模型，8B 参数'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.28,
            currency: 'CNY'
        }
    },
    {
        id: 'Qwen/Qwen3-Reranker-4B',
        name: 'Qwen3 Reranker 4B',
        providerId: 'silicon',
        type: 'rerank',
        metadata: {
            maxTokens: 8192,
            description: '通义千问第三代重排模型，4B 参数'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.14,
            currency: 'CNY'
        }
    },
    {
        id: 'Qwen/Qwen3-Reranker-0.6B',
        name: 'Qwen3 Reranker 0.6B',
        providerId: 'silicon',
        type: 'rerank',
        metadata: {
            maxTokens: 8192,
            description: '通义千问第三代重排模型，0.6B 参数，轻量低成本'
        },
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.07,
            currency: 'CNY'
        }
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
    // ── OpenAI GPT-5 系列 ──
    {
        id: 'gpt-5.5',
        name: 'GPT-5.5',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 最新默认文本与推理模型，适合编码、工具调用和复杂任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.5-pro',
        name: 'GPT-5.5 Pro',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 最高质量推理模型，适合成本和延迟不敏感的复杂场景' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.4',
        name: 'GPT-5.4',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 上一代默认文本与推理模型，适合已有 GPT-5.4 集成' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.4-pro',
        name: 'GPT-5.4 Pro',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'GPT-5.4 高质量版本，保留兼容已有 Pro 配置' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.4-mini',
        name: 'GPT-5.4 mini',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'GPT-5.4 低成本轻量版，适合测试和轻量生产任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.4-nano',
        name: 'GPT-5.4 nano',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'GPT-5.4 超轻量版，适合高吞吐分类和简单任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.3-codex',
        name: 'GPT-5.3 Codex',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 面向代码编辑、Agent 和工具密集流程优化的 Codex 模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.1-codex-mini',
        name: 'GPT-5.1 Codex mini',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 轻量 Codex 模型，适合低成本代码工作流' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.2',
        name: 'GPT-5.2',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 专业工作与长时 Agent 优化版本，保留兼容' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.1',
        name: 'GPT-5.1',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'GPT-5.1 兼容模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5',
        name: 'GPT-5',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'GPT-5 经典旗舰模型，强推理与编程能力' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    // ── OpenAI GPT-4 / o 系列 ──
    {
        id: 'gpt-4.1',
        name: 'GPT-4.1',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 非推理长上下文模型，指令遵循能力强' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-4.1-mini',
        name: 'GPT-4.1 mini',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'GPT-4.1 轻量版，速度快成本低' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-4.1-nano',
        name: 'GPT-4.1 nano',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'GPT-4.1 最轻量文本模型，适合低延迟简单任务' },
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
        metadata: { description: 'OpenAI 前代多模态旗舰' },
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
        id: 'qwen3.7-max',
        name: 'Qwen3.7-Max',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '通义千问最新旗舰 Max 系列，适合复杂推理与多模态任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen3.7-plus',
        name: 'Qwen3.7-Plus',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '通义千问 3.7 均衡版，性能与成本兼顾' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen3.6-flash',
        name: 'Qwen3.6-Flash',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '通义千问 3.6 快速版，低延迟高性价比' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen3.5-flash',
        name: 'Qwen3.5-Flash',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: 'Qwen3.5 快速多模态版，保留兼容' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen3-max',
        name: 'Qwen3-Max',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '通义千问 3 Max，上一代旗舰兼容项' },
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
        id: 'deepseek-v4-pro',
        name: 'DeepSeek V4 Pro',
        providerId: 'deepseek',
        type: 'chat',
        metadata: { description: 'DeepSeek 最新高质量 V4-Pro 模型，适合深度推理和复杂问答' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'deepseek-v4-flash',
        name: 'DeepSeek V4 Flash',
        providerId: 'deepseek',
        type: 'chat',
        metadata: { description: 'DeepSeek 最新 V4-Flash 模型，兼容旧 deepseek-chat / deepseek-reasoner 模式' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'deepseek-chat',
        name: 'DeepSeek Chat (legacy)',
        providerId: 'deepseek',
        type: 'chat',
        metadata: { description: '旧版兼容 ID，将迁移到 DeepSeek V4 Flash' },
        isSystem: true, isUserAdded: false
    },
    {
        id: 'deepseek-reasoner',
        name: 'DeepSeek Reasoner (legacy)',
        providerId: 'deepseek',
        type: 'chat',
        metadata: { description: '旧版推理兼容 ID，将迁移到 DeepSeek V4 Pro' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    // ── Moonshot (Kimi) ──
    {
        id: 'kimi-k2.7-code',
        name: 'Kimi K2.7 Code',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi K2.7 代码与 Agent 模型，适合编程和工具调用' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'kimi-k2.7-code-highspeed',
        name: 'Kimi K2.7 Code Highspeed',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi K2.7 代码高速版，适合低延迟开发助手场景' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'kimi-k2.6',
        name: 'Kimi K2.6',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi 最新通用多模态模型，适合问答、文档和视觉任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'kimi-k2.5',
        name: 'Kimi K2.5',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi K2.5 兼容模型，适合通用多模态任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'moonshot-v1-vision-preview',
        name: 'Moonshot v1 Vision Preview',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Moonshot 视觉预览模型，兼容图片输入' },
        tags: ['vision'],
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
        id: 'glm-5.2',
        name: 'GLM-5.2',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱最新旗舰 GLM-5.2，支持深度推理与 Agent 场景' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-5.1',
        name: 'GLM-5.1',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱 GLM-5.1，复杂推理与代码能力增强' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-5',
        name: 'GLM-5',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱 GLM-5 系列基础旗舰，保留兼容' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-4.7',
        name: 'GLM-4.7',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱编程增强模型，Agentic Coding 能力优化' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-4.6',
        name: 'GLM-4.6',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱 GLM-4.6，增强推理与代码能力' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-4.5',
        name: 'GLM-4.5',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱 Agent 基座模型，推理+代码+Agentic 原生融合' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-4.5-air',
        name: 'GLM-4.5-Air',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: 'GLM-4.5 轻量版，高性价比' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-4-air-250414',
        name: 'GLM-4-Air 250414',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: 'GLM-4-Air 250414，前代轻量版' },
        isSystem: true, isUserAdded: false
    },
    // ── MiniMax ──
    {
        id: 'MiniMax-M3',
        name: 'MiniMax M3',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax 最新旗舰，原生 Agent 与长上下文推理模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'MiniMax-M2.7',
        name: 'MiniMax M2.7',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax M2.7，通用推理与长上下文增强版' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'MiniMax-M2.7-highspeed',
        name: 'MiniMax M2.7 Highspeed',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax M2.7 高速版，适合低延迟场景' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'MiniMax-M2.5',
        name: 'MiniMax M2.5',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax M2.5 兼容模型，百万长上下文' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'MiniMax-M2.5-highspeed',
        name: 'MiniMax M2.5 Highspeed',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax M2.5 高速兼容模型' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'MiniMax-M2.1',
        name: 'MiniMax M2.1',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax 均衡版，兼顾效果与成本' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'MiniMax-M2.1-highspeed',
        name: 'MiniMax M2.1 Highspeed',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax M2.1 高速兼容模型' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'MiniMax-M2',
        name: 'MiniMax M2',
        providerId: 'minimax',
        type: 'chat',
        metadata: { description: 'MiniMax 前代基座模型' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    // ── Anthropic (Claude) ──
    {
        id: 'claude-fable-5',
        name: 'Claude Fable 5',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Anthropic 最新 Claude 5 系列模型，400K 上下文，适合通用与创作任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-opus-4-8',
        name: 'Claude Opus 4.8',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Claude Opus 最新旗舰，复杂推理与编码能力强' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-sonnet-5',
        name: 'Claude Sonnet 5',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Claude Sonnet 5 均衡旗舰，适合编码、Agent 和通用推理' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-haiku-4-5-20251001',
        name: 'Claude Haiku 4.5',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Claude 高速轻量模型，适合低延迟和成本敏感场景' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-opus-4-1-20250805',
        name: 'Claude Opus 4.1',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Claude Opus 4.1 兼容模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-opus-4-20250514',
        name: 'Claude Opus 4',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Claude Opus 4 兼容模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'claude-sonnet-4-20250514',
        name: 'Claude Sonnet 4',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Claude Sonnet 4 兼容模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    // ── Google (Gemini) ──
    {
        id: 'gemini-3.5-flash',
        name: 'Gemini 3.5 Flash',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Google 最新稳定 Gemini 3.5 Flash，强调速度、多模态与思考能力' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-3.1-pro-preview',
        name: 'Gemini 3.1 Pro Preview',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Gemini 3.1 Pro 预览版，适合复杂多模态和推理任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-3-flash-preview',
        name: 'Gemini 3 Flash Preview',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Gemini 3 Flash 预览版，保留兼容' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-3.1-flash-lite',
        name: 'Gemini 3.1 Flash-Lite',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Gemini 3.1 轻量高速版本，适合大规模低成本任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-2.5-pro',
        name: 'Gemini 2.5 Pro',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Gemini 2.5 旗舰稳定版，1M 上下文，自适应思考' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-2.5-flash',
        name: 'Gemini 2.5 Flash',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Gemini 2.5 快速均衡版，可控推理预算' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-2.5-flash-lite',
        name: 'Gemini 2.5 Flash-Lite',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Gemini 2.5 超轻量版，大规模低成本场景' },
        tags: ['vision'],
        isSystem: true, isUserAdded: false
    },
    // ── xAI (Grok) ──
    {
        id: 'grok-4.3',
        name: 'Grok 4.3',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'xAI 最新旗舰模型，适合通用推理、工具调用和多模态任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'grok-build-0.1',
        name: 'Grok Build 0.1',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'xAI 面向应用构建与工具工作流的模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'grok-4',
        name: 'Grok 4',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'xAI Grok 4 兼容模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'grok-4-1-fast-reasoning',
        name: 'Grok 4.1 Fast',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'Grok 4.1 快速推理版，保留兼容' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'grok-3',
        name: 'Grok 3',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'Grok 3 兼容模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    // ── 硅基流动 (SiliconFlow) ──
    {
        id: 'deepseek-ai/DeepSeek-V4-Flash',
        name: 'DeepSeek V4 Flash (SiliconFlow)',
        providerId: 'silicon',
        type: 'chat',
        metadata: { description: '托管于硅基流动的 DeepSeek V4 Flash，支持推理参数' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'Pro/zai-org/GLM-4.7',
        name: 'GLM-4.7 Pro (SiliconFlow)',
        providerId: 'silicon',
        type: 'chat',
        metadata: { description: '托管于硅基流动的 GLM-4.7 Pro，支持思考参数' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
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
        id: 'deepseek-ai/DeepSeek-V3.2',
        name: 'DeepSeek V3.2 (SiliconFlow)',
        providerId: 'silicon',
        type: 'chat',
        metadata: { description: '托管于硅基流动的 DeepSeek V3.2 兼容模型' },
        isSystem: true, isUserAdded: false
    },
    {
        id: 'Qwen/Qwen3-32B',
        name: 'Qwen3-32B (SiliconFlow)',
        providerId: 'silicon',
        type: 'chat',
        metadata: { description: '托管于硅基流动的通义千问3 32B 开源版' },
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
        id: 'doubao-seed-2-1-pro-260628',
        name: 'Doubao Seed 2.1 Pro',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 Seed 2.1 旗舰模型，适合复杂推理、文档和多模态 Agent 场景'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-seed-2-1-turbo-260628',
        name: 'Doubao Seed 2.1 Turbo',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 Seed 2.1 高速版，兼顾推理质量与低延迟'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-seed-2-0-pro-260215',
        name: 'Doubao Seed 2.0 Pro',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 2.0 Pro 兼容模型，支持长链路推理与多模态'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-seed-2-0-lite-260428',
        name: 'Doubao Seed 2.0 Lite',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 2.0 Lite 最新版本号，均衡性能与成本'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'doubao-seed-2-0-mini-260428',
        name: 'Doubao Seed 2.0 Mini',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包 2.0 Mini 最新版本号，低延迟高并发，适合成本敏感场景'
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
