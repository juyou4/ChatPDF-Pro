/**
 * Provider类型定义
 * 三层架构的底层：服务商配置
 */

/**
 * Provider能力标识
 */
export interface ProviderCapabilities {
  chat?: boolean              // 支持对话模型
  embedding?: boolean         // 支持embedding模型
  rerank?: boolean            // 支持rerank模型
  imageGeneration?: boolean   // 支持图像生成
}

/**
 * Provider API配置
 */
export interface ProviderApiConfig {
  fetchModelsEndpoint?: string    // 获取模型列表的endpoint
  chatEndpoint?: string           // 对话endpoint
  embeddingEndpoint?: string      // embedding endpoint
  rerankEndpoint?: string         // rerank endpoint
}

/**
 * Provider（服务商）接口
 * 代表一个AI服务提供商（如OpenAI、硅基流动等）
 */
export interface Provider {
  id: string                      // 服务商ID（openai, silicon等）
  name: string                    // 显示名称
  logo?: string                   // 图标路径
  apiKey: string                  // API密钥
  apiHost: string                 // API地址
  enabled: boolean                // 是否启用
  isSystem: boolean               // 是否系统内置

  capabilities: ProviderCapabilities  // Provider能力标识
  apiConfig?: ProviderApiConfig       // API配置
}

/**
 * Provider配置更新接口
 */
export interface ProviderUpdate {
  apiKey?: string
  apiHost?: string
  enabled?: boolean
}

/**
 * Provider测试连接结果
 */
export interface ProviderTestResult {
  success: boolean
  message?: string
  availableModels?: number
  error?: string
  latency?: number  // 延迟毫秒数，连接测试成功时返回
}
