import type { EmbeddingProvider, EmbeddingModel, EmbeddingModelType } from '../types/embedding'

/**
 * 本地嵌入模型配置
 */
export const LOCAL_EMBEDDING_MODELS: EmbeddingModel[] = [
    {
        id: 'all-MiniLM-L6-v2',
        name: 'MiniLM-L6-v2',
        provider: 'local',
        type: 'embedding',
        dimension: 384,
        maxTokens: 256,
        description: '快速通用模型，适合大多数场景'
    },
    {
        id: 'paraphrase-multilingual-MiniLM-L12-v2',
        name: 'Multilingual MiniLM-L12-v2',
        provider: 'local',
        type: 'embedding',
        dimension: 384,
        maxTokens: 128,
        description: '多语言支持，特别是中文效果更好'
    }
]

/**
 * OpenAI 嵌入模型配置
 */
export const OPENAI_EMBEDDING_MODELS: EmbeddingModel[] = [
    {
        id: 'text-embedding-3-large',
        name: 'text-embedding-3-large',
        provider: 'openai',
        type: 'embedding',
        dimension: 3072,
        maxTokens: 8191,
        description: '最高质量',
        pricing: { perMillionTokens: 0.13, currency: 'USD' }
    },
    {
        id: 'text-embedding-3-small',
        name: 'text-embedding-3-small',
        provider: 'openai',
        type: 'embedding',
        dimension: 1536,
        maxTokens: 8191,
        description: '性价比最高',
        pricing: { perMillionTokens: 0.02, currency: 'USD' }
    }
]

/**
 * 阿里云嵌入模型配置
 */
export const ALIYUN_EMBEDDING_MODELS: EmbeddingModel[] = [
    {
        id: 'text-embedding-v3',
        name: 'text-embedding-v3',
        provider: 'aliyun',
        type: 'embedding',
        dimension: 1024,
        maxTokens: 8192,
        description: '中文优化，价格最便宜',
        pricing: { perMillionTokens: 0.007, currency: 'USD' }
    }
]

/**
 * 硅基流动嵌入模型配置
 */
export const SILICON_EMBEDDING_MODELS: EmbeddingModel[] = [
    {
        id: 'BAAI/bge-m3',
        name: 'BAAI/bge-m3',
        provider: 'silicon',
        type: 'embedding',
        dimension: 1024,
        maxTokens: 8192,
        description: '开源，托管在硅基流动',
        pricing: { perMillionTokens: 0.02, currency: 'USD' }
    },
    {
        id: 'Qwen/Qwen-Embedding-8B',
        name: 'Qwen Embedding 8B',
        provider: 'silicon',
        type: 'embedding',
        dimension: 1024,
        maxTokens: 8192,
        description: '阿里通义千问嵌入模型'
    },
    {
        id: 'BAAI/bge-reranker-v2-m3',
        name: 'BGE Reranker v2-M3',
        provider: 'silicon',
        type: 'rerank',
        dimension: 0,
        maxTokens: 8192,
        description: '重排模型，用于结果重新排序'
    }
]

/**
 * Moonshot 嵌入模型配置
 */
export const MOONSHOT_EMBEDDING_MODELS: EmbeddingModel[] = [
    {
        id: 'moonshot-embedding-v1',
        name: 'moonshot-embedding-v1',
        provider: 'moonshot',
        type: 'embedding',
        dimension: 1024,
        maxTokens: 8192,
        description: 'Moonshot AI 嵌入模型',
        pricing: { perMillionTokens: 0.011, currency: 'USD' }
    }
]

/**
 * DeepSeek 嵌入模型配置
 */
export const DEEPSEEK_EMBEDDING_MODELS: EmbeddingModel[] = [
    {
        id: 'deepseek-embedding-v1',
        name: 'deepseek-embedding-v1',
        provider: 'deepseek',
        type: 'embedding',
        dimension: 1024,
        maxTokens: 8192,
        description: 'DeepSeek 嵌入模型',
        pricing: { perMillionTokens: 0.01, currency: 'USD' }
    }
]

/**
 * 智谱 嵌入模型配置
 */
export const ZHIPU_EMBEDDING_MODELS: EmbeddingModel[] = [
    {
        id: 'embedding-3',
        name: 'Embedding-3',
        provider: 'zhipu',
        type: 'embedding',
        dimension: 2048,
        maxTokens: 8192,
        description: '智谱 GLM 嵌入模型',
        pricing: { perMillionTokens: 0.014, currency: 'USD' }
    }
]

/**
 * MiniMax 嵌入模型配置
 */
export const MINIMAX_EMBEDDING_MODELS: EmbeddingModel[] = [
    {
        id: 'minimax-embedding-v2',
        name: 'minimax-embedding-v2',
        provider: 'minimax',
        type: 'embedding',
        dimension: 1024,
        maxTokens: 8192,
        description: 'MiniMax 嵌入模型',
        pricing: { perMillionTokens: 0.014, currency: 'USD' }
    }
]

/**
 * 系统内置服务商配置
 */
export const SYSTEM_EMBEDDING_PROVIDERS: EmbeddingProvider[] = [
    {
        id: 'local',
        name: '本地模型 (免费)',
        type: 'local',
        apiKey: '',
        apiHost: '',
        models: LOCAL_EMBEDDING_MODELS,
        enabled: true,
        isSystem: true,
        logo: '💻'
    },
    {
        id: 'openai',
        name: 'OpenAI',
        type: 'openai',
        apiKey: '',
        apiHost: 'https://api.openai.com/v1',
        models: OPENAI_EMBEDDING_MODELS,
        enabled: false,
        isSystem: true,
        logo: '🤖'
    },
    {
        id: 'aliyun',
        name: '阿里云 (通义千问)',
        type: 'openai',
        apiKey: '',
        apiHost: 'https://dashscope.aliyuncs.com/api/v1',
        models: ALIYUN_EMBEDDING_MODELS,
        enabled: false,
        isSystem: true,
        logo: '☁️'
    },
    {
        id: 'silicon',
        name: '硅基流动 (SiliconFlow)',
        type: 'openai',
        apiKey: '',
        apiHost: 'https://api.siliconflow.cn',
        models: SILICON_EMBEDDING_MODELS,
        enabled: false,
        isSystem: true,
        logo: '🔷'
    },
    {
        id: 'moonshot',
        name: 'Moonshot (Kimi)',
        type: 'openai',
        apiKey: '',
        apiHost: 'https://api.moonshot.cn/v1',
        models: MOONSHOT_EMBEDDING_MODELS,
        enabled: false,
        isSystem: true,
        logo: '🌙'
    },
    {
        id: 'deepseek',
        name: 'DeepSeek',
        type: 'openai',
        apiKey: '',
        apiHost: 'https://api.deepseek.com/v1',
        models: DEEPSEEK_EMBEDDING_MODELS,
        enabled: false,
        isSystem: true,
        logo: '🧠'
    },
    {
        id: 'zhipu',
        name: '智谱 (GLM)',
        type: 'openai',
        apiKey: '',
        apiHost: 'https://open.bigmodel.cn/api/paas/v4',
        models: ZHIPU_EMBEDDING_MODELS,
        enabled: false,
        isSystem: true,
        logo: '✨'
    },
    {
        id: 'minimax',
        name: 'MiniMax',
        type: 'openai',
        apiKey: '',
        apiHost: 'https://api.minimaxi.com/v1',
        models: MINIMAX_EMBEDDING_MODELS,
        enabled: false,
        isSystem: true,
        logo: '🎯'
    }
]

/**
 * 根据名称自动识别模型类型
 */
export function detectModelType(modelId: string): EmbeddingModelType {
    const lowerModelId = modelId.toLowerCase()

    // 重排模型识别
    if (/rerank|re-rank|ranker|ranking/i.test(lowerModelId)) {
        return 'rerank'
    }

    // 默认为嵌入模型
    return 'embedding'
}

/**
 * 根据 Provider 获取所有嵌入模型
 */
export function getEmbeddingModelsByProvider(
    provider: EmbeddingProvider
): EmbeddingModel[] {
    return provider.models.filter(m => m.type === 'embedding')
}

/**
 * 根据 Provider 获取所有重排模型
 */
export function getRerankModelsByProvider(
    provider: EmbeddingProvider
): EmbeddingModel[] {
    return provider.models.filter(m => m.type === 'rerank')
}
