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
  protocol?: 'openai' | 'anthropic' | 'gemini' | 'ollama' | 'custom'
  fetchModelsEndpoint?: string    // 获取模型列表的endpoint
  chatEndpoint?: string           // 对话endpoint
  embeddingEndpoint?: string      // embedding endpoint
  rerankEndpoint?: string         // rerank endpoint
  supportsStreaming?: boolean     // Provider 是否支持 SSE 流式输出
  supportsReasoning?: boolean     // 是否允许自动注入 thinking 参数
  reasoningMode?: 'openai_effort' | 'anthropic_adaptive' | 'anthropic_budget' | 'gemini_level' | 'gemini_budget' | 'qwen_budget' | 'thinking_toggle' | 'ollama_think' | 'fixed'
  reasoningOptions?: string[]
  reasoningDefault?: string
  reasoningAlwaysEnabled?: boolean
  reasoningOffControl?: string
  reasoningOnControl?: 'thinking_enabled' | 'thinking_adaptive' | 'enable_thinking_true' | 'reasoning_split_true' | 'provider_default'
  apiKeyHeader?: string           // API Key 请求头名称，默认 Authorization
  apiKeyPrefix?: string           // API Key 请求头前缀，默认 Bearer 空格
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
  apiConfig?: ProviderApiConfig
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
  verifiedModel?: string
  chatEndpoint?: string
}
