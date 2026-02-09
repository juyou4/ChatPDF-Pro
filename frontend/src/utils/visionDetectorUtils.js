/**
 * 视觉模型检测工具函数
 *
 * 提供纯函数用于判断模型是否支持视觉（多模态）能力，
 * 采用 tags 优先 → provider+modelId 正则 → 关键字兜底的三级判定逻辑。
 * 结构与 ModelQuickSwitch 中 THINKING_CAPABLE_PROVIDERS 模式一致。
 */

/**
 * 视觉能力 Provider 白名单及模型 ID 匹配规则
 *
 * 键为 providerId，值为匹配该 Provider 下支持视觉能力的模型 ID 的正则表达式。
 * 覆盖 OpenAI、Anthropic、Google、阿里云、xAI、MiniMax、字节跳动等主流多模态模型系列。
 */
export const VISION_CAPABLE_RULES = {
  openai: /^(gpt-4o|gpt-4-turbo|gpt-4\.1|gpt-5)/i,
  anthropic: /^(claude-3|claude-sonnet-4|claude-opus-4|claude-haiku-4)/i,
  gemini: /^gemini-(2|[3-9])/i,
  qwen: /^(qwen-vl|qwen-max)/i,
  grok: /^(grok-vision|grok-4)/i,
  minimax: /^abab6\.5/i,
  doubao: /^doubao-1\.5-pro/i,
}

/**
 * 判断模型是否支持视觉能力
 *
 * 三级判定优先级：
 * 1. tags 明确包含 "vision" → 直接判定支持
 * 2. providerId 在白名单中且 modelId 匹配对应正则 → 判定支持
 * 3. modelId 包含 "vision" 或 "-vl" 关键字 → 判定支持（兜底逻辑，保留现有行为）
 *
 * @param {object} model - 模型对象 { id, name, tags, providerId }
 * @returns {boolean} 是否支持视觉能力
 */
export function supportsVision(model) {
  if (!model) return false

  // 1. tags 明确包含 vision
  if (model.tags?.includes('vision')) return true

  // 2. provider 白名单 + modelId 正则匹配
  const rule = VISION_CAPABLE_RULES[model.providerId]
  if (rule && rule.test(model.id)) return true

  // 3. modelId 包含 vision/vl 关键字（保留现有逻辑）
  const lowerId = model.id.toLowerCase()
  if (lowerId.includes('vision') || lowerId.includes('-vl')) return true

  return false
}
