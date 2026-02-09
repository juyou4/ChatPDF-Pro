/**
 * Provider配置文件
 * 三层架构的底层：服务商配置（不包含模型列表）
 */

import type { Provider } from '../types/provider'
import { getProviderLogo } from './providerLogos'

/**
 * 系统内置Provider配置
 * 注意：在新架构中，Provider不包含models，models由ModelContext单独管理
 */
export const SYSTEM_PROVIDERS: Provider[] = [
    {
        id: 'local',
        name: '本地模型 (免费)',
        logo: getProviderLogo('local'),
        apiKey: '',
        apiHost: '',
        enabled: true,
        isSystem: true,
        capabilities: {
            embedding: true,
            rerank: false
        }
    },
    {
        id: 'openai',
        name: 'OpenAI',
        logo: getProviderLogo('openai'),
        apiKey: '',
        apiHost: 'https://api.openai.com/v1',
        enabled: false,
        isSystem: true,
        capabilities: {
            chat: true,
            embedding: true,
            rerank: false,
            imageGeneration: true
        },
        apiConfig: {
            fetchModelsEndpoint: '/models',
            chatEndpoint: '/chat/completions',
            embeddingEndpoint: '/embeddings'
        }
    },
    {
        id: 'aliyun',
        name: '阿里云 (通义千问)',
        logo: getProviderLogo('aliyun'),
        apiKey: '',
        apiHost: 'https://dashscope.aliyuncs.com/api/v1',
        enabled: false,
        isSystem: true,
        capabilities: {
            chat: true,
            embedding: true,
            rerank: false
        },
        apiConfig: {
            fetchModelsEndpoint: '/models',
            embeddingEndpoint: '/embeddings'
        }
    },
    {
        id: 'silicon',
        name: '硅基流动 (SiliconFlow)',
        logo: getProviderLogo('silicon'),
        apiKey: '',
        apiHost: 'https://api.siliconflow.cn',
        enabled: false,
        isSystem: true,
        capabilities: {
            chat: true,
            embedding: true,
            rerank: true,
            imageGeneration: true
        },
        apiConfig: {
            fetchModelsEndpoint: '/v1/models',
            chatEndpoint: '/v1/chat/completions',
            embeddingEndpoint: '/v1/embeddings',
            rerankEndpoint: '/v1/rerank'
        }
    },
    {
        id: 'moonshot',
        name: 'Moonshot (Kimi)',
        logo: getProviderLogo('moonshot'),
        apiKey: '',
        apiHost: 'https://api.moonshot.cn/v1',
        enabled: false,
        isSystem: true,
        capabilities: {
            chat: true,
            embedding: true,
            rerank: false
        },
        apiConfig: {
            fetchModelsEndpoint: '/models',
            chatEndpoint: '/chat/completions',
            embeddingEndpoint: '/embeddings'
        }
    },
    {
        id: 'deepseek',
        name: 'DeepSeek',
        logo: getProviderLogo('deepseek'),
        apiKey: '',
        apiHost: 'https://api.deepseek.com/v1',
        enabled: false,
        isSystem: true,
        capabilities: {
            chat: true,
            embedding: true,
            rerank: false
        },
        apiConfig: {
            fetchModelsEndpoint: '/models',
            chatEndpoint: '/chat/completions',
            embeddingEndpoint: '/embeddings'
        }
    },
    {
        id: 'zhipu',
        name: '智谱 (GLM)',
        logo: getProviderLogo('zhipu'),
        apiKey: '',
        apiHost: 'https://open.bigmodel.cn/api/paas/v4',
        enabled: false,
        isSystem: true,
        capabilities: {
            chat: true,
            embedding: true,
            rerank: false
        },
        apiConfig: {
            fetchModelsEndpoint: '/models',
            embeddingEndpoint: '/embeddings'
        }
    },
    {
        id: 'minimax',
        name: 'MiniMax',
        logo: getProviderLogo('minimax'),
        apiKey: '',
        apiHost: 'https://api.minimaxi.com/v1',
        enabled: false,
        isSystem: true,
        capabilities: {
            chat: true,
            embedding: true,
            rerank: false
        },
        apiConfig: {
            fetchModelsEndpoint: '/models',
            chatEndpoint: '/chat/completions',
            embeddingEndpoint: '/embeddings'
        }
    },
    {
        id: 'doubao',
        name: '字节跳动 (豆包)',
        logo: getProviderLogo('doubao'),
        apiKey: '',
        apiHost: 'https://ark.cn-beijing.volces.com/api/v3',
        enabled: false,
        isSystem: true,
        capabilities: {
            chat: true,
            embedding: true,
            rerank: false
        },
        apiConfig: {
            fetchModelsEndpoint: '/models',
            chatEndpoint: '/chat/completions',
            embeddingEndpoint: '/embeddings'
        }
    }
]

/**
 * 根据ID获取Provider
 */
export function getProviderById(id: string): Provider | undefined {
    return SYSTEM_PROVIDERS.find(p => p.id === id)
}

/**
 * 获取所有启用的Providers
 */
export function getEnabledProviders(providers: Provider[]): Provider[] {
    return providers.filter(p => p.enabled)
}

/**
 * 获取支持特定能力的Providers
 */
export function getProvidersByCapability(
    providers: Provider[],
    capability: keyof Provider['capabilities']
): Provider[] {
    return providers.filter(p => p.capabilities[capability])
}
