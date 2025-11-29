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
        isUserAdded: false
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
    ...MINIMAX_MODELS
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
