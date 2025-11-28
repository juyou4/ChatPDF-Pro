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
