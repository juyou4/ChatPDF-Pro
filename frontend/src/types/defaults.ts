/**
 * DefaultModels类型定义
 * 三层架构的顶层：默认模型选择
 */

/**
 * 默认模型配置
 * 存储用户选择的各种场景下的默认模型
 */
export interface DefaultModels {
  // LLM默认模型（未来扩展）
  assistantModel?: string       // 默认助手模型ID
  quickModel?: string           // 快速模型ID
  translateModel?: string       // 翻译模型ID

  // Embedding默认模型（当前实现）
  embeddingModel?: string       // 默认embedding模型ID
  rerankModel?: string          // 默认rerank模型ID（可选）
}

/**
 * 默认模型设置类型
 */
export type DefaultModelType = keyof DefaultModels

/**
 * 默认模型更新接口
 */
export interface DefaultModelUpdate {
  type: DefaultModelType
  modelId: string | null
}
