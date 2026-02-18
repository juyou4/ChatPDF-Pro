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
  // OpenAI: GPT-4o/4-Turbo/4.1/5 系列 + o3/o4 推理模型均支持视觉
  openai: /^(gpt-4o|gpt-4-turbo|gpt-4\.1|gpt-5|o3|o4)/i,
  // Anthropic: Claude 3 系列全系 + Claude 4 Sonnet/Opus/Haiku（含 haiku-3.x）
  anthropic: /^(claude-3|claude-sonnet-4|claude-opus-4|claude-haiku-3|claude-haiku-4)/i,
  // Google Gemini 2+ 全系均支持视觉
  gemini: /^gemini-(2|[3-9])/i,
  // 通义千问 VL 和 Max 系列
  qwen: /^(qwen-vl|qwen-max)/i,
  // xAI Grok 4 系列及专用视觉模型
  grok: /^(grok-vision|grok-4)/i,
  // MiniMax abab6.5 系列
  minimax: /^abab6\.5/i,
  // 豆包：1.5-Pro 系列及全部 Seed 系列（Seed 1.x / 2.0 Pro / Lite / Mini 均为多模态通用模型）
  doubao: /^(doubao-1\.5-pro|doubao-seed)/i,
  // Moonshot：moonshot-v1 系列支持图片输入
  moonshot: /^moonshot-v1/i,
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
