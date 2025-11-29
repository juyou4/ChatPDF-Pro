/**
 * 嵌入模型服务商类型
 */
export type EmbeddingProviderType = 'local' | 'openai' | 'custom'

/**
 * 嵌入模型类型
 */
export type EmbeddingModelType = 'embedding' | 'rerank'

/**
 * 嵌入模型接口
 */
export interface EmbeddingModel {
    id: string                      // 模型ID
    name: string                    // 显示名称
    provider: string                // 所属服务商ID
    type: EmbeddingModelType        // 模型类型
    dimension: number               // 向量维度
    maxTokens: number               // 最大token数
    description?: string            // 描述
    pricing?: {                     // 价格信息
        perMillionTokens: number
        currency: string
    }
}

/**
 * 嵌入服务商接口
 */
export interface EmbeddingProvider {
    id: string                      // 服务商ID
    name: string                    // 显示名称
    type: EmbeddingProviderType     // 服务商类型
    apiKey: string                  // API密钥
    apiHost: string                 // API地址
    models: EmbeddingModel[]        // 模型列表
    enabled: boolean                // 是否启用
    isSystem: boolean               // 是否系统内置
    logo?: string                   // Logo图片
}

/**
 * 嵌入服务配置
 */
export interface EmbeddingConfig {
    providers: EmbeddingProvider[]  // 服务商列表
    defaultProviderId: string       // 默认服务商ID
    defaultEmbeddingModelId: string // 默认嵌入模型ID
    defaultRerankModelId?: string   // 默认重排模型ID
}

/**
 * 类型适配器：新类型系统兼容层
 * 用于在新旧类型系统之间转换
 */

import type { Provider, ProviderCapabilities } from './provider'
import type { Model, ModelType } from './model'
import type { DefaultModels } from './defaults'

/**
 * 将旧的EmbeddingProvider转换为新的Provider格式
 */
export function embeddingProviderToProvider(embProvider: EmbeddingProvider): Provider {
    const capabilities: ProviderCapabilities = {
        embedding: true,
        rerank: embProvider.models.some(m => m.type === 'rerank')
    }

    return {
        id: embProvider.id,
        name: embProvider.name,
        logo: embProvider.logo,
        apiKey: embProvider.apiKey,
        apiHost: embProvider.apiHost,
        enabled: embProvider.enabled,
        isSystem: embProvider.isSystem,
        capabilities
    }
}

/**
 * 将新的Provider转换为旧的EmbeddingProvider格式
 * 注意：需要传入models，因为Provider本身不包含models
 */
export function providerToEmbeddingProvider(
    provider: Provider,
    models: EmbeddingModel[]
): EmbeddingProvider {
    return {
        id: provider.id,
        name: provider.name,
        type: provider.isSystem ? 'custom' : 'custom', // 简化映射
        apiKey: provider.apiKey,
        apiHost: provider.apiHost,
        models,
        enabled: provider.enabled,
        isSystem: provider.isSystem,
        logo: provider.logo
    }
}

/**
 * 将旧的EmbeddingModel转换为新的Model格式
 */
export function embeddingModelToModel(embModel: EmbeddingModel): Model {
    const modelType: ModelType = embModel.type === 'rerank' ? 'rerank' : 'embedding'

    return {
        id: embModel.id,
        name: embModel.name,
        providerId: embModel.provider,
        type: modelType,
        metadata: {
            dimension: embModel.dimension,
            maxTokens: embModel.maxTokens,
            description: embModel.description
        },
        isSystem: true, // EmbeddingModel都是系统预设
        isUserAdded: false,
        pricing: embModel.pricing ? {
            input: embModel.pricing.perMillionTokens,
            currency: embModel.pricing.currency
        } : undefined
    }
}

/**
 * 将新的Model转换为旧的EmbeddingModel格式
 */
export function modelToEmbeddingModel(model: Model): EmbeddingModel {
    const embeddingType: EmbeddingModelType = model.type === 'rerank' ? 'rerank' : 'embedding'

    return {
        id: model.id,
        name: model.name,
        provider: model.providerId,
        type: embeddingType,
        dimension: model.metadata.dimension || 0,
        maxTokens: model.metadata.maxTokens || 0,
        description: model.metadata.description,
        pricing: model.pricing ? {
            perMillionTokens: model.pricing.input || 0,
            currency: model.pricing.currency
        } : undefined
    }
}

/**
 * 将新的DefaultModels转换为旧的EmbeddingConfig部分
 */
export function defaultModelsToEmbeddingConfig(
    defaults: DefaultModels,
    providers: EmbeddingProvider[]
): EmbeddingConfig {
    return {
        providers,
        defaultProviderId: 'local', // 需要从embeddingModel推断
        defaultEmbeddingModelId: defaults.embeddingModel || '',
        defaultRerankModelId: defaults.rerankModel
    }
}
