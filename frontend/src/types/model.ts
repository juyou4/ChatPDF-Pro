/**
 * Model类型定义
 * 三层架构的中层：模型管理
 */

/**
 * 模型类型
 */
export type ModelType = 'chat' | 'embedding' | 'rerank' | 'image'

/**
 * 模型元数据
 */
export interface ModelMetadata {
  dimension?: number          // embedding维度
  maxTokens?: number          // 最大token数
  contextWindow?: number      // 上下文窗口
  description?: string        // 描述
  languages?: string[]        // 支持的语言
}

/**
 * 模型价格信息
 */
export interface ModelPricing {
  input?: number              // 输入价格（per million tokens）
  output?: number             // 输出价格（per million tokens）
  currency: string            // 货币单位
}

/**
 * Model（模型）接口
 * 代表一个具体的AI模型
 */
export interface Model {
  id: string                  // 完整模型ID（如 gpt-4, text-embedding-3-large）
  name: string                // 显示名称
  providerId: string          // 所属provider ID
  type: ModelType             // 模型类型

  metadata: ModelMetadata     // 模型元数据

  // 状态标识
  isSystem: boolean           // 是否系统预设
  isUserAdded: boolean        // 用户是否添加到collection
  isFavorite?: boolean        // 是否收藏

  pricing?: ModelPricing      // 价格信息（可选）
}

/**
 * 用户模型集合
 * 存储用户添加到collection中的模型
 */
export interface UserModelCollection {
  models: Model[]                         // 用户启用的模型列表
  lastSync: Record<string, number>        // 每个provider的最后同步时间（Unix timestamp）
}

/**
 * 从Provider API获取的模型响应
 */
export interface FetchModelsResponse {
  models: Model[]
  providerId: string
  timestamp: number
}

/**
 * 模型过滤选项
 */
export interface ModelFilterOptions {
  type?: ModelType
  providerId?: string
  isUserAdded?: boolean
  searchQuery?: string
}
