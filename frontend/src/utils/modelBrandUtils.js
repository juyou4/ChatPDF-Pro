const MODEL_BRAND_RULES = [
  ['deepseek', /deepseek/i],
  ['anthropic', /(?:^|[/_.:-])claude(?:$|[/_.:-])/i],
  ['gemini', /(?:^|[/_.:-])gemini(?:$|[/_.:-])/i],
  ['openai', /(?:^|[/_.:-])(?:gpt|o1|o3|o4|codex)(?:$|[/_.:-])/i],
  ['qwen', /(?:^|[/_.:-])(?:qwen|qwq)(?:$|[/_.:-])/i],
  ['moonshot', /(?:^|[/_.:-])(?:kimi|moonshot)(?:$|[/_.:-])/i],
  ['zhipu', /(?:^|[/_.:-])(?:glm|chatglm)(?:$|[/_.:-])/i],
  ['minimax', /minimax/i],
  ['grok', /(?:^|[/_.:-])grok(?:$|[/_.:-])/i],
  ['mistral', /(?:^|[/_.:-])(?:mistral|mixtral|codestral)(?:$|[/_.:-])/i],
  ['doubao', /(?:^|[/_.:-])(?:doubao|seed)(?:$|[/_.:-])/i],
  ['hunyuan', /hunyuan/i],
  ['xiaomi', /(?:^|[/_.:-])mimo(?:$|[/_.:-])/i],
  ['baichuan', /baichuan/i],
  ['step', /(?:^|[/_.:-])step(?:$|[/_.:-])/i],
  ['cohere', /(?:^|[/_.:-])command-r(?:$|[/_.:-])/i],
  ['spark', /(?:^|[/_.:-])(?:spark|xinghuo)(?:$|[/_.:-])/i],
  ['baidu', /(?:^|[/_.:-])ernie(?:$|[/_.:-])/i],
  ['yi', /(?:^|[/_.:-])yi(?:$|[/_.:-])/i],
]

/**
 * 解析消息应展示的模型品牌。模型家族优先于 API 网关，例如通过
 * SiliconFlow 调用 deepseek-ai/DeepSeek-R1 时仍展示 DeepSeek 标识。
 */
export function resolveModelBrandProviderId(modelId, apiProviderId = '') {
  const normalizedModel = String(modelId || '').trim()
  const matchedRule = MODEL_BRAND_RULES.find(([, pattern]) => pattern.test(normalizedModel))
  if (matchedRule) return matchedRule[0]
  return String(apiProviderId || '').trim().toLowerCase() || 'openai'
}
