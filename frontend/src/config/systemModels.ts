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
 * Google Gemini embedding 模型
 *
 * Gemini OpenAI 兼容层支持 /embeddings；v2 为预览版多模态向量模型，
 * 当前 ChatPDF 仅将其用于文本块索引。
 */
export const GEMINI_MODELS: Model[] = [
    {
        id: 'gemini-embedding-2-preview',
        name: 'Gemini Embedding 2 Preview',
        providerId: 'gemini',
        type: 'embedding',
        metadata: {
            maxTokens: 8192,
            description: 'Google 最新预览多模态向量模型；可用于文本、图片、视频、音频和 PDF 的统一检索空间'
        },
        tags: ['latest'],
        isSystem: true,
        isUserAdded: false
    },
    {
        id: 'gemini-embedding-001',
        name: 'Gemini Embedding 001',
        providerId: 'gemini',
        type: 'embedding',
        metadata: {
            maxTokens: 2048,
            description: 'Google 稳定版文本向量模型，适合语义搜索、文档检索与推荐'
        },
        isSystem: true,
        isUserAdded: false
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
        id: 'qwen3.7-text-embedding',
        name: 'Qwen3.7 Text Embedding',
        providerId: 'aliyun',
        type: 'embedding',
        metadata: {
            dimension: 1024,
            maxTokens: 131072,
            description: '通义千问最新多语言文本向量模型，默认 1024 维，支持 256-2560 维自定义输出'
        },
        tags: ['latest'],
        isSystem: true,
        isUserAdded: false,
        pricing: {
            input: 0.07,
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
 * Chat 模型（对话）
 */
export const CHAT_MODELS: Model[] = [
    // ── OpenAI GPT-5 系列 ──
    {
        id: 'gpt-5.6',
        name: 'GPT-5.6',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 最新旗舰别名，当前指向 GPT-5.6 Sol，适合复杂推理、编码和工具工作流' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.6-sol',
        name: 'GPT-5.6 Sol',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI GPT-5.6 旗舰型号，适合复杂专业任务、深度推理和高质量编码' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.6-terra',
        name: 'GPT-5.6 Terra',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI GPT-5.6 均衡型号，兼顾智能水平、延迟和成本' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.6-luna',
        name: 'GPT-5.6 Luna',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI GPT-5.6 高吞吐型号，适合成本敏感的批量和轻量任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gpt-5.5',
        name: 'GPT-5.5',
        providerId: 'openai',
        type: 'chat',
        metadata: { description: 'OpenAI 上一代通用文本与推理模型，保留用于兼容已有配置' },
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
        id: 'qwen3.8-max',
        name: 'Qwen3.8-Max',
        providerId: 'aliyun',
        type: 'chat',
        metadata: {
            description: '通义千问最新旗舰模型，原生多模态与 1M 上下文，适合复杂 Agent 和长文档任务',
            reasoningMode: 'openai_effort',
            reasoningOptions: ['off', 'low', 'medium', 'xhigh'],
            reasoningDefault: 'xhigh',
            reasoningOffControl: 'enable_thinking_false',
            reasoningOnControl: 'enable_thinking_true'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen3.8-flash',
        name: 'Qwen3.8-Flash',
        providerId: 'aliyun',
        type: 'chat',
        metadata: {
            description: '通义千问最新高速多模态模型，原生 1M 上下文，适合低延迟 Agent 和高吞吐任务',
            reasoningMode: 'openai_effort',
            reasoningOptions: ['off', 'low', 'medium', 'xhigh'],
            reasoningDefault: 'xhigh',
            reasoningOffControl: 'enable_thinking_false',
            reasoningOnControl: 'enable_thinking_true'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen3.7-max',
        name: 'Qwen3.7-Max',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '通义千问上一代 Max 模型，兼容已有配置并支持多模态任务' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'qwen3.7-flash',
        name: 'Qwen3.7-Flash',
        providerId: 'aliyun',
        type: 'chat',
        metadata: { description: '通义千问上一代高速多模态模型，适合低延迟 Agent 与高吞吐任务' },
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
    // ── DeepSeek ──
    {
        id: 'deepseek-v4-flash',
        name: 'DeepSeek V4 Flash',
        providerId: 'deepseek',
        type: 'chat',
        metadata: { description: 'DeepSeek 当前官方模型；兼容旧 deepseek-chat / deepseek-reasoner，并支持思考与非思考模式' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'deepseek-v4-pro',
        name: 'DeepSeek V4 Pro',
        providerId: 'deepseek',
        type: 'chat',
        metadata: { description: 'DeepSeek V4 高性能版，支持思考与非思考模式，适合复杂推理和长文档任务' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'deepseek-v4-flash-vision-exp',
        name: 'DeepSeek V4 Flash Vision Exp',
        providerId: 'deepseek',
        type: 'chat',
        metadata: { description: 'DeepSeek 实验性多模态视觉模型；纯文本能力与 V4 Flash 持平，支持图片描述、截图识字和图表分析' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    // ── Moonshot (Kimi) ──
    {
        id: 'kimi-k3',
        name: 'Kimi K3',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi 当前旗舰模型，1M 上下文，支持视觉理解、工具调用与 low/high/max 推理强度' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'kimi-k2.7-code',
        name: 'Kimi K2.7 Code',
        providerId: 'moonshot',
        type: 'chat',
        metadata: { description: 'Kimi 编程 Agent 模型，支持文本/图片/视频输入与思考模式' },
        tags: ['vision', 'reasoning'],
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
    // ── 智谱 GLM ──
    {
        id: 'glm-5.3',
        name: 'GLM-5.3',
        providerId: 'zhipu',
        type: 'chat',
        metadata: {
            description: '智谱最新旗舰模型，文本 Agent 与复杂软件工程能力增强；始终启用思考，支持 1M 上下文',
            reasoningMode: 'openai_effort',
            reasoningOptions: ['low', 'high', 'max'],
            reasoningDefault: 'max',
            reasoningAlwaysEnabled: true,
            reasoningOnControl: 'thinking_enabled'
        },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-5.3-flash',
        name: 'GLM-5.3-Flash',
        providerId: 'zhipu',
        type: 'chat',
        metadata: {
            description: '智谱最新原生多模态模型，支持图像、视频和文件理解；始终启用思考，适合视觉 Agent',
            reasoningMode: 'openai_effort',
            reasoningOptions: ['low', 'high', 'max'],
            reasoningDefault: 'max',
            reasoningAlwaysEnabled: true,
            reasoningOnControl: 'thinking_enabled'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-5.2',
        name: 'GLM-5.2',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱上一代旗舰，保留用于兼容已有配置与迁移' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'glm-5.1',
        name: 'GLM-5.1',
        providerId: 'zhipu',
        type: 'chat',
        metadata: { description: '智谱上一代推理模型，保留用于兼容已有配置' },
        tags: ['reasoning'],
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
    // ── 小米 MiMo ──
    {
        id: 'mimo-v2.5-pro',
        name: 'MiMo V2.5 Pro',
        providerId: 'xiaomi',
        type: 'chat',
        metadata: { description: '小米 MiMo 旗舰文本模型，支持深度思考、函数调用、结构化输出和联网搜索' },
        tags: ['reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'mimo-v2.5',
        name: 'MiMo V2.5',
        providerId: 'xiaomi',
        type: 'chat',
        metadata: { description: '小米 MiMo 全模态模型，支持图像、音频、视频理解，以及深度思考和工具调用' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    // ── Anthropic (Claude) ──
    {
        id: 'claude-opus-5',
        name: 'Claude Opus 5',
        providerId: 'anthropic',
        type: 'chat',
        metadata: { description: 'Anthropic 面向复杂 Agent、编码与企业任务的最新旗舰模型' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
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
        metadata: { description: 'Claude Opus 上一代版本，保留用于兼容已有配置' },
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
    // ── Google (Gemini) ──
    {
        id: 'gemini-3.7-flash',
        name: 'Gemini 3.7 Flash',
        providerId: 'gemini',
        type: 'chat',
        metadata: {
            description: 'Google 最新稳定 Flash 模型，原生多模态、1M 上下文，支持 Search grounding 与工具调用',
            reasoningMode: 'gemini_level',
            reasoningOptions: ['low', 'medium', 'high'],
            reasoningDefault: 'high',
            reasoningAlwaysEnabled: true
        },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-3.6-flash',
        name: 'Gemini 3.6 Flash',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Google 上一代稳定 Flash 模型，保留用于兼容已有配置' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'gemini-3.5-flash',
        name: 'Gemini 3.5 Flash',
        providerId: 'gemini',
        type: 'chat',
        metadata: { description: 'Google 旧版 Flash 模型，建议迁移到 Gemini 3.7 Flash' },
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
        id: 'grok-4.6',
        name: 'Grok 4.6',
        providerId: 'grok',
        type: 'chat',
        metadata: {
            description: 'xAI 最新旗舰模型，支持文本与图像输入、工具调用和可配置深度推理',
            reasoningMode: 'openai_effort',
            reasoningOptions: ['low', 'medium', 'high', 'xhigh'],
            reasoningDefault: 'high',
            reasoningAlwaysEnabled: true
        },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
    {
        id: 'grok-4.5',
        name: 'Grok 4.5',
        providerId: 'grok',
        type: 'chat',
        metadata: { description: 'xAI 上一代旗舰模型，保留用于兼容已有配置' },
        tags: ['vision', 'reasoning'],
        isSystem: true, isUserAdded: false
    },
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
        id: 'Qwen/Qwen3-32B',
        name: 'Qwen3-32B (SiliconFlow)',
        providerId: 'silicon',
        type: 'chat',
        metadata: { description: '托管于硅基流动的通义千问3 32B 开源版' },
        tags: ['reasoning'],
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
        id: 'doubao-seed-evolving',
        name: 'Doubao Seed Evolving',
        providerId: 'doubao',
        type: 'chat',
        metadata: {
            description: '豆包当前滚动更新的 Agent/Coding 模型，一个 ID 自动获得最新版本能力'
        },
        tags: ['vision', 'reasoning'],
        isSystem: true,
        isUserAdded: false
    },
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
        id: 'embo-01',
        name: 'Embo-01',
        providerId: 'minimax',
        type: 'embedding',
        metadata: {
            maxTokens: 4096,
            description: 'MiniMax 官方文本向量模型，适合知识库与长记忆检索'
        },
        isSystem: true,
        isUserAdded: false
    }
]

const LATEST_SYSTEM_MODEL_IDS = new Set([
    'qwen3.7-text-embedding',
    'gpt-5.6',
    'gpt-5.6-sol',
    'gpt-5.6-terra',
    'gpt-5.6-luna',
    'qwen3.8-max',
    'qwen3.8-flash',
    'qwen3.7-plus',
    'deepseek-v4-flash',
    'deepseek-v4-pro',
    'deepseek-v4-flash-vision-exp',
    'kimi-k3',
    'kimi-k2.7-code',
    'kimi-k2.6',
    'glm-5.3',
    'glm-5.3-flash',
    'MiniMax-M3',
    'mimo-v2.5-pro',
    'mimo-v2.5',
    'claude-opus-5',
    'claude-fable-5',
    'claude-sonnet-5',
    'claude-haiku-4-5-20251001',
    'gemini-3.7-flash',
    'grok-4.6',
    'doubao-seed-evolving',
    'doubao-seed-2-1-pro-260628',
    'doubao-seed-2-1-turbo-260628',
])

/**
 * 各官方 Provider 当前推荐的默认聊天模型。
 * 这是推荐目录元数据，不会覆盖用户已经保存的模型选择。
 */
export const LATEST_PROVIDER_DEFAULTS: Record<string, string> = {
    openai: 'gpt-5.6',
    aliyun: 'qwen3.8-max',
    deepseek: 'deepseek-v4-pro',
    moonshot: 'kimi-k3',
    zhipu: 'glm-5.3',
    minimax: 'MiniMax-M3',
    xiaomi: 'mimo-v2.5-pro',
    anthropic: 'claude-opus-5',
    gemini: 'gemini-3.7-flash',
    grok: 'grok-4.6',
    doubao: 'doubao-seed-evolving',
}

function withSystemModelTags(model: Model): Model {
    const tags = new Set(model.tags || [])
    if (model.type === 'embedding') tags.add('embedding')
    if (model.type === 'rerank') tags.add('rerank')
    if (model.providerId === 'local') tags.add('free')
    if (LATEST_SYSTEM_MODEL_IDS.has(model.id)) tags.add('latest')
    return { ...model, tags: Array.from(tags) }
}

/**
 * 所有系统预设模型。每项都带有稳定 tags，供模型服务、快捷切换和视觉模型
 * 选择共用；历史模型不再作为默认推荐暴露。
 */
export const SYSTEM_MODELS: Model[] = [
    ...LOCAL_MODELS,
    ...OPENAI_MODELS,
    ...GEMINI_MODELS,
    ...ALIYUN_MODELS,
    ...SILICON_MODELS,
    ...ZHIPU_MODELS,
    ...MINIMAX_MODELS,
    ...DOUBAO_MODELS,
    ...CHAT_MODELS
]
    .map(withSystemModelTags)

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
