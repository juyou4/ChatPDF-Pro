/**
 * 模型快速切换器辅助函数
 *
 * 提供纯函数用于模型过滤、分组和格式化，
 * 便于单独测试和复用。
 */

/**
 * 过滤出已启用 Provider 下的 chat 类型模型
 *
 * 仅保留 type === 'chat' 且其 providerId 对应的 Provider 已启用的模型。
 *
 * @param {Array<{id: string, name: string, type: string, providerId: string}>} models - 所有模型列表
 * @param {Array<{id: string, name: string, enabled: boolean}>} enabledProviders - 已启用的 Provider 列表
 * @returns {Array} 过滤后的 chat 类型模型列表
 */
export function filterChatModels(models, enabledProviders) {
  // 构建已启用 Provider 的 id 集合，提高查找效率
  const enabledProviderIds = new Set(enabledProviders.map(p => p.id))

  return models.filter(
    model => model.type === 'chat' && enabledProviderIds.has(model.providerId)
  )
}

/**
 * 将模型按 Provider 分组
 *
 * 按已启用的 Provider 对模型进行分组，仅返回包含模型的分组。
 * 每个分组包含 provider 对象和该 Provider 下的模型数组。
 *
 * @param {Array<{id: string, name: string, type: string, providerId: string}>} models - 已过滤的模型列表（通常为 chat 类型）
 * @param {Array<{id: string, name: string, enabled: boolean}>} enabledProviders - 已启用的 Provider 列表
 * @returns {Array<{provider: object, models: Array}>} 按 Provider 分组的模型列表，仅包含有模型的分组
 */
export function groupModelsByProvider(models, enabledProviders) {
  return enabledProviders
    .map(provider => ({
      provider,
      models: models.filter(m => m.providerId === provider.id)
    }))
    .filter(group => group.models.length > 0)
}

/**
 * 格式化模型键值
 *
 * 将 providerId 和 modelId 拼接为标准格式 `providerId:modelId`，
 * 用于 setDefaultModel 的值。
 *
 * @param {string} providerId - Provider ID
 * @param {string} modelId - 模型 ID
 * @returns {string} 格式化后的模型键值，格式为 `providerId:modelId`
 */
export function formatModelKey(providerId, modelId) {
  return `${providerId}:${modelId}`
}

/**
 * 按关键词过滤模型列表
 *
 * 对模型名称进行不区分大小写的子串匹配。
 * 关键词 trim() 后为空时返回完整列表。
 *
 * @param {Array<{id: string, name: string, type: string, providerId: string}>} models - 模型列表
 * @param {string} keyword - 搜索关键词
 * @returns {Array} 过滤后的模型列表
 */
export function filterModelsByKeyword(models, keyword) {
  const trimmed = keyword.trim()
  if (trimmed === '') {
    return models
  }
  const lowerKeyword = trimmed.toLowerCase()
  return models.filter(model => model.name.toLowerCase().includes(lowerKeyword))
}
