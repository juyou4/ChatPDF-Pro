// 思考深度的产品级枚举。真正可用的选项由后端能力接口裁剪。
export const REASONING_EFFORT_OPTIONS = [
  { value: 'off', label: 'off', description: '使用模型原生关闭方式' },
  { value: 'minimal', label: 'minimal', description: '最低原生推理强度' },
  { value: 'low', label: 'low', description: '更快，适合简单问题' },
  { value: 'medium', label: 'medium', description: '速度与质量平衡' },
  { value: 'high', label: 'high', description: '适合复杂分析' },
  { value: 'xhigh', label: 'xhigh', description: '更充分的推理预算' },
  { value: 'max', label: 'max', description: '质量优先，消耗更多额度' },
  { value: 'ultra', label: 'ultra', description: '仅在模型明确支持时可用，消耗最多额度' },
]

const EFFORT_RANK = Object.fromEntries(REASONING_EFFORT_OPTIONS.map((item, index) => [item.value, index]))
const REASONING_PREFERENCES_KEY = 'reasoningEffortByModel:v1'
const REASONING_PREFERENCES_KEY_V2 = 'reasoningEffortByModel:v2'

export const getReasoningEffortOption = (value) => (
  REASONING_EFFORT_OPTIONS.find((item) => item.value === value)
  || REASONING_EFFORT_OPTIONS[0]
)

export const normalizeReasoningProfile = (payload, fallback = null) => {
  const source = payload && typeof payload === 'object' ? payload : fallback
  const mode = String(source?.mode || 'unsupported')
  const rawOptions = Array.isArray(source?.options) ? source.options : ['off']
  let options = [...new Set(rawOptions.map((value) => String(value || '').trim().toLowerCase()))]
    .filter((value) => EFFORT_RANK[value] !== undefined)
    .sort((a, b) => EFFORT_RANK[a] - EFFORT_RANK[b])
  if (mode !== 'unsupported' && options.includes('off') && !source?.off_control) {
    options = options.filter((value) => value !== 'off')
  }
  const safeOptions = options.length > 0 ? options : mode === 'unsupported' ? ['off'] : ['medium']
  const requestedDefault = String(source?.default || '').trim().toLowerCase()
  const defaultValue = safeOptions.includes(requestedDefault)
    ? requestedDefault
    : safeOptions.find((value) => value !== 'off') || 'off'
  const offControl = safeOptions.includes('off') ? source?.off_control || null : null
  // 档位是跨 Provider 的协议值，不允许旧版本后端或缓存把它重新本地化。
  // 中文说明仍由菜单的 description/note 提供；这里始终输出 low/high/max
  // 等 canonical 名称，避免前端在服务重启前继续显示旧的“轻量/最大”标签。
  return {
    mode,
    options: safeOptions,
    default: defaultValue,
    always_enabled: source?.always_enabled === true,
    note: String(source?.note || '').trim(),
    source: String(source?.source || 'fallback'),
    cost_warning_from: source?.cost_warning_from || null,
    off_control: offControl,
    on_control: source?.on_control || null,
    split_reasoning_output: source?.split_reasoning_output === true,
    can_disable: safeOptions.includes('off') && source?.can_disable === true,
    off_is_guaranteed: safeOptions.includes('off') && source?.off_is_guaranteed === true,
    labels: Object.fromEntries(safeOptions.map((value) => [value, value])),
  }
}

export const inferReasoningProfile = ({ providerId, modelId, model, provider }) => {
  const pid = String(providerId || '').toLowerCase()
  const mid = String(modelId || '').toLowerCase()
  const tags = Array.isArray(model?.tags) ? model.tags : []
  const modelConfig = model?.metadata || {}
  const providerConfig = provider?.apiConfig || {}
  const declaredMode = modelConfig.reasoningMode || providerConfig.reasoningMode
  const declaredOptions = modelConfig.reasoningOptions || providerConfig.reasoningOptions
  if (declaredMode || (Array.isArray(declaredOptions) && declaredOptions.length > 0)) {
    const mode = declaredMode || 'openai_effort'
    const options = Array.isArray(declaredOptions) && declaredOptions.length > 0
      ? declaredOptions
      : mode === 'fixed'
        ? ['medium']
        : mode === 'thinking_toggle' || mode === 'ollama_think'
          ? ['off', 'medium']
          : ['off', 'low', 'medium', 'high']
    return normalizeReasoningProfile({
      mode,
      options,
      default: modelConfig.reasoningDefault || providerConfig.reasoningDefault,
      always_enabled: typeof modelConfig.reasoningAlwaysEnabled === 'boolean'
        ? modelConfig.reasoningAlwaysEnabled
        : providerConfig.reasoningAlwaysEnabled,
      off_control: modelConfig.reasoningOffControl || providerConfig.reasoningOffControl,
      on_control: modelConfig.reasoningOnControl || providerConfig.reasoningOnControl,
      source: modelConfig.reasoningMode || modelConfig.reasoningOptions ? 'explicit_model' : 'explicit_provider',
    })
  }
  const isReasoning = tags.includes('reasoning') || /reason|reasoner|thinking|think|qwen3|qwq|gpt-5|^o[134]|deepseek-v4|grok-[34]|grok-build|seed|(?:^|\/)minimax-m[23](?:[.\-:]|$)|(?:^|\/)mimo-v2\.5(?:-pro)?(?:[.\-:]|$)/.test(mid)
  if (pid === 'local' || pid === 'ollama') {
    if (mid.includes('gpt-oss')) {
      return normalizeReasoningProfile({
        mode: 'ollama_think',
        options: ['low', 'medium', 'high'],
        default: 'medium',
        always_enabled: true,
        note: 'Ollama GPT-OSS 仅接受 low / medium / high，不能关闭思考',
      })
    }
    return normalizeReasoningProfile(isReasoning
      ? { mode: 'ollama_think', options: ['off', 'medium'], note: 'Ollama think 开关', off_control: 'ollama_think_false', off_is_guaranteed: true }
      : { mode: 'unsupported', options: ['off'], note: '当前本地模型未声明思考能力' })
  }
  if (pid === 'moonshot') {
    if (mid.includes('kimi-k3')) {
      return normalizeReasoningProfile({
        mode: 'openai_effort',
        options: ['low', 'high', 'max'],
        default: 'max',
        always_enabled: true,
        note: 'Kimi K3 始终思考，支持 low / high / max',
        on_control: 'provider_default',
      })
    }
    if (/kimi-k2[.-]?(?:5|6)/.test(mid)) {
      return normalizeReasoningProfile({
        mode: 'thinking_toggle',
        options: ['off', 'medium'],
        default: 'medium',
        note: 'Kimi K2 思考开关',
        off_control: 'thinking_disabled',
        on_control: 'thinking_enabled',
        can_disable: true,
      })
    }
    if (/kimi-k2[.-]?7-code/.test(mid)) {
      return normalizeReasoningProfile({ mode: 'fixed', options: ['medium'], default: 'medium', always_enabled: true, note: 'Kimi Code 始终思考' })
    }
    return normalizeReasoningProfile({ mode: 'unsupported', options: ['off'], note: '当前模型未声明思考能力' })
  }
  if (pid === 'doubao') {
    return normalizeReasoningProfile(isReasoning
      ? { mode: 'fixed', options: ['medium'], default: 'medium', always_enabled: true, note: '该模型由厂商自动决定思考深度' }
      : { mode: 'unsupported', options: ['off'], note: '当前模型未声明思考能力' })
  }
  if (pid === 'gemini' && /gemini-3/.test(mid)) {
    let options = ['low', 'medium', 'high']
    let defaultValue = 'high'
    let canDisable = false
    if (/gemini-3\.7-flash/.test(mid)) {
      options = ['off', 'minimal', 'low', 'medium', 'high']
      defaultValue = 'high'
      canDisable = true
    } else if (/gemini-3\.6-flash/.test(mid)) {
      options = ['minimal', 'low', 'medium', 'high']
      defaultValue = 'medium'
    } else if (/gemini-3\.5-flash-lite/.test(mid)) {
      options = ['minimal', 'low', 'medium', 'high']
      defaultValue = 'minimal'
    } else if (/gemini-3\.5-flash/.test(mid)) {
      options = ['minimal', 'low', 'medium', 'high']
      defaultValue = 'medium'
    } else if (/gemini-3\.1-pro-preview/.test(mid)) {
      options = ['low', 'high']
      defaultValue = 'high'
    } else if (/gemini-3\.1-flash-lite-image/.test(mid)) {
      options = ['minimal', 'high']
      defaultValue = 'minimal'
    } else if (/gemini-3\.1-flash-lite/.test(mid)) {
      options = ['minimal', 'low', 'medium', 'high']
      defaultValue = 'minimal'
    } else if (/gemini-3-flash-preview/.test(mid)) {
      options = ['minimal', 'low', 'medium', 'high']
      defaultValue = 'high'
    } else if (/gemini-3-pro-preview/.test(mid)) {
      options = ['low', 'high']
      defaultValue = 'high'
    }
    return normalizeReasoningProfile({
      mode: 'gemini_level',
      options,
      default: defaultValue,
      always_enabled: !canDisable,
      note: canDisable ? 'Gemini thinkingLevel；off 使用兼容的 thinkingBudget=0' : 'Gemini thinkingLevel 按具体型号裁剪',
      off_control: canDisable ? 'gemini_budget_zero' : null,
      can_disable: canDisable,
    })
  }
  if (pid === 'gemini' && /gemini-2\.5-pro/.test(mid)) {
    return normalizeReasoningProfile({ mode: 'gemini_budget', options: ['low', 'medium', 'high', 'max'], default: 'medium', always_enabled: true, note: 'Gemini Pro 保留最小思考预算' })
  }
  if (pid === 'gemini' && /gemini-2\.5/.test(mid)) {
    return normalizeReasoningProfile({ mode: 'gemini_budget', options: ['off', 'low', 'medium', 'high', 'max'], note: 'Gemini thinking budget', off_control: 'gemini_budget_zero', off_is_guaranteed: true })
  }
  if (pid === 'grok') {
    if (mid.includes('grok-4.6')) {
      return normalizeReasoningProfile({ mode: 'openai_effort', options: ['off', 'low', 'medium', 'high', 'xhigh'], default: 'high', note: 'Grok 4.6 支持 none / low / medium / high / xhigh', off_control: 'reasoning_effort_none', can_disable: true })
    }
    if (mid.includes('grok-4.5')) {
      return normalizeReasoningProfile({ mode: 'openai_effort', options: ['off', 'low', 'medium', 'high'], default: 'high', note: 'Grok 4.5 支持 none / low / medium / high', off_control: 'reasoning_effort_none', can_disable: true })
    }
    if (mid.includes('grok-3-mini')) {
      return normalizeReasoningProfile({ mode: 'openai_effort', options: ['low', 'medium', 'high'], default: 'medium', always_enabled: true, note: 'Grok 3 Mini reasoning effort' })
    }
    return normalizeReasoningProfile(isReasoning
      ? { mode: 'fixed', options: ['medium'], default: 'medium', always_enabled: true, note: '该 xAI 模型使用固定推理模式，不接受 reasoning_effort' }
      : { mode: 'unsupported', options: ['off'], note: '当前 Grok 模型未声明思考能力' })
  }
  if (pid === 'anthropic') {
    const adaptive = /claude-(?:fable|mythos|opus|sonnet)-5(?:[.-]|$)|claude-(?:opus|sonnet)-4[-.]?6(?:[.-]|$)|claude-opus-4[-.]?[78](?:[.-]|$)/.test(mid)
      || mid.includes('claude-mythos-preview')
    const extended = /claude-(?:opus|sonnet|haiku)-4[-.]?[0-5](?:[.-]|$)|claude-sonnet-3[-.]?7(?:[.-]|$)/.test(mid)
    const supportsXhigh = /claude-(?:(?:fable|mythos|opus|sonnet)-)?5(?:[.-]|$)|claude-opus-?4[.-]?[78](?:[.-]|$)/.test(mid)
    const mandatory = /claude-(?:fable|mythos)-5(?:[.-]|$)/.test(mid) || mid.includes('claude-mythos-preview')
    const adaptiveOptions = supportsXhigh
      ? ['off', 'low', 'medium', 'high', 'xhigh', 'max']
      : ['off', 'low', 'medium', 'high', 'max']
    if (!adaptive && !extended) {
      return normalizeReasoningProfile({ mode: 'unsupported', options: ['off'], note: '该 Claude 型号未声明 adaptive/extended thinking 能力' })
    }
    return normalizeReasoningProfile({
      mode: adaptive ? 'anthropic_adaptive' : 'anthropic_budget',
      options: mandatory ? adaptiveOptions.filter((item) => item !== 'off') : adaptiveOptions,
      default: adaptive ? 'high' : 'medium',
      always_enabled: mandatory,
      note: mandatory
        ? '该 Claude 型号强制 adaptive thinking；关闭参数会返回 400'
        : adaptive ? 'Claude effort 与 adaptive thinking 独立控制' : 'Anthropic extended thinking',
      off_control: mandatory ? null : 'thinking_disabled',
      on_control: mandatory ? 'provider_default' : null,
      can_disable: !mandatory,
    })
  }
  if (pid === 'aliyun' || pid === 'qwen' || /qwen3|qwq/.test(mid)) {
    if (/qwen3\.8-(?:max|flash)/.test(mid)) {
      return normalizeReasoningProfile({
        mode: 'qwen_budget',
        options: ['off', 'low', 'medium', 'high', 'max'],
        default: 'high',
        note: 'Qwen3.8 使用 enable_thinking + thinking_budget（应用档位映射）',
        off_control: 'enable_thinking_false',
        on_control: 'enable_thinking_true',
        can_disable: true,
      })
    }
    return normalizeReasoningProfile(isReasoning || provider?.apiConfig?.supportsReasoning === true
      ? { mode: 'qwen_budget', options: ['off', 'low', 'medium', 'high', 'max'], default: 'medium', note: 'Qwen thinking budget', off_control: 'enable_thinking_false', can_disable: true }
      : { mode: 'unsupported', options: ['off'], note: '当前模型未声明思考能力' })
  }
  if (pid === 'silicon') {
    if (/deepseek-v4|glm-5\.2/.test(mid)) {
      return normalizeReasoningProfile({
        mode: 'openai_effort',
        options: ['off', 'high', 'max'],
        default: 'high',
        note: 'SiliconFlow V4/GLM-5.2 使用 enable_thinking + reasoning_effort',
        off_control: 'enable_thinking_false',
        on_control: 'enable_thinking_true',
        can_disable: true,
      })
    }
    return normalizeReasoningProfile(isReasoning
      ? { mode: 'qwen_budget', options: ['off', 'low', 'medium', 'high', 'max'], default: 'medium', note: 'SiliconFlow 使用 enable_thinking + thinking_budget', off_control: 'enable_thinking_false', on_control: 'enable_thinking_true', can_disable: true }
      : { mode: 'unsupported', options: ['off'], note: '当前模型未声明思考能力' })
  }
  if (pid === 'deepseek') {
    if (mid.includes('deepseek-v4')) {
      return normalizeReasoningProfile({
        mode: 'openai_effort',
        options: ['off', 'low', 'high', 'max'],
        default: 'high',
        note: 'DeepSeek V4 原生 reasoning_effort：low / high / max',
        off_control: 'thinking_disabled',
        on_control: 'thinking_enabled',
        can_disable: true,
      })
    }
    return normalizeReasoningProfile(isReasoning
      ? { mode: 'thinking_toggle', options: ['off', 'medium'], default: 'medium', note: '当前接口只提供思考开关', off_control: 'thinking_disabled', on_control: 'thinking_enabled', can_disable: true }
      : { mode: 'unsupported', options: ['off'], note: '当前模型未声明思考能力' })
  }
  if (pid === 'zhipu') {
    if (mid.includes('glm-5.3')) {
      return normalizeReasoningProfile({
        mode: 'openai_effort',
        options: ['low', 'high', 'max'],
        default: 'max',
        always_enabled: true,
        note: 'GLM-5.3 始终思考，仅支持 low / high / max',
        on_control: 'thinking_enabled',
      })
    }
    if (mid.includes('glm-5.2')) {
      return normalizeReasoningProfile({
        mode: 'openai_effort',
        options: ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'],
        default: 'max',
        note: 'GLM-5.2 支持完整 reasoning_effort 档位',
        off_control: 'thinking_disabled',
        on_control: 'thinking_enabled',
        can_disable: true,
      })
    }
    return normalizeReasoningProfile(isReasoning
      ? { mode: 'thinking_toggle', options: ['off', 'medium'], default: 'medium', note: '当前接口只提供思考开关', off_control: 'thinking_disabled', on_control: 'thinking_enabled', can_disable: true }
      : { mode: 'unsupported', options: ['off'], note: '当前模型未声明思考能力' })
  }
  if (pid === 'xiaomi') {
    return normalizeReasoningProfile(isReasoning
      ? { mode: 'thinking_toggle', options: ['off', 'medium'], default: 'medium', note: 'MiMo 使用 thinking.type 开关', off_control: 'thinking_disabled', on_control: 'thinking_enabled', can_disable: true }
      : { mode: 'unsupported', options: ['off'], note: '当前模型未声明思考能力' })
  }
  if (pid === 'minimax') {
    if (/(?:^|\/)minimax-m3(?:[.\-:]|$)/.test(mid)) {
      return normalizeReasoningProfile({ mode: 'thinking_toggle', options: ['off', 'medium'], default: 'medium', note: 'MiniMax M3 使用 adaptive / disabled；reasoning_split 只分离输出', off_control: 'thinking_disabled', on_control: 'thinking_adaptive', split_reasoning_output: true, can_disable: true })
    }
    if (/(?:^|\/)minimax-m2(?:[.\-:]|$)/.test(mid)) {
      return normalizeReasoningProfile({ mode: 'fixed', options: ['medium'], default: 'medium', always_enabled: true, note: 'MiniMax M2.x 始终思考；thinking.type=disabled 不会关闭思考', on_control: 'provider_default', split_reasoning_output: true })
    }
    return normalizeReasoningProfile({ mode: 'unsupported', options: ['off'], note: '当前模型未声明思考能力' })
  }
  if (isReasoning || provider?.apiConfig?.supportsReasoning === true) {
    if (/gpt-5(?:\.\d+)?-pro(?:[-.]|$)/.test(mid)) {
      return normalizeReasoningProfile({ mode: 'openai_effort', options: ['high'], default: 'high', always_enabled: true, note: 'OpenAI Pro 使用固定 high 档位' })
    }
    if (/gpt-5\.(?:2|4|5|6)(?:[-.]|$)/.test(mid)) {
      return normalizeReasoningProfile({ mode: 'openai_effort', options: ['off', 'low', 'medium', 'high', 'xhigh'], default: 'medium', note: 'OpenAI reasoning_effort（保守兼容集）', off_control: 'reasoning_effort_none', can_disable: true })
    }
    if (/gpt-5\.(?:1|3)(?:[-.]|$)/.test(mid)) {
      return normalizeReasoningProfile({ mode: 'openai_effort', options: ['off', 'low', 'medium', 'high'], default: 'medium', note: 'OpenAI reasoning_effort（保守兼容集）', off_control: 'reasoning_effort_none', can_disable: true })
    }
    if (/gpt-5(?:[-.]|$)/.test(mid)) {
      return normalizeReasoningProfile({ mode: 'openai_effort', options: ['minimal', 'low', 'medium', 'high'], default: 'medium', always_enabled: true, note: 'OpenAI reasoning_effort（保守兼容集）' })
    }
    if (/^o[134](?:[-.]|$)/.test(mid)) {
      return normalizeReasoningProfile({ mode: 'openai_effort', options: ['low', 'medium', 'high'], default: 'medium', always_enabled: true, note: 'OpenAI reasoning model' })
    }
    return normalizeReasoningProfile({ mode: 'openai_effort', options: ['off', 'low', 'medium', 'high'], note: 'OpenAI-compatible reasoning_effort' })
  }
  return normalizeReasoningProfile({ mode: 'unsupported', options: ['off'], note: '当前模型未声明思考能力' })
}

// 这些模型要求多轮或工具调用时原样回传上一轮 reasoning_content。
// 只按明确的模型族启用，避免普通 OpenAI 兼容接口拒绝未知消息字段。
export const requiresPreservedReasoning = ({ modelId } = {}) => {
  const mid = String(modelId || '').trim().toLowerCase()
  return /deepseek-v4|glm-5\.[23]|kimi-k3|kimi-k2[.-]?7-code|qwen3\.8-(?:max|flash)|mimo-v2\.5|(?:^|\/)minimax-m[23](?:[.\-:]|$)/.test(mid)
}

export const getStoredReasoningEffort = (modelKey) => {
  if (!modelKey || typeof localStorage === 'undefined') return null
  try {
    const v2 = JSON.parse(localStorage.getItem(REASONING_PREFERENCES_KEY_V2) || '{}') || {}
    const v2Value = String(v2?.[modelKey] || '').trim().toLowerCase()
    if (EFFORT_RANK[v2Value] !== undefined) return v2Value
    const v1 = JSON.parse(localStorage.getItem(REASONING_PREFERENCES_KEY) || '{}') || {}
    const v1Value = String(v1?.[modelKey] || '').trim().toLowerCase()
    if (EFFORT_RANK[v1Value] === undefined) return null
    // v1 曾在档案加载前把全局 off 写进 DeepSeek V4，那不是用户选择。
    if (v1Value === 'off' && String(modelKey).toLowerCase().includes('deepseek-v4')) return null
    return v1Value
  } catch {
    return null
  }
}

export const resolveModelReasoningEffort = ({
  stored,
  profile,
  current = 'off',
} = {}) => {
  const options = Array.isArray(profile?.options) ? profile.options : []
  const defaultValue = String(profile?.default || '').trim().toLowerCase()
  if (stored) return stored
  if (defaultValue && (options.length === 0 || options.includes(defaultValue))) return defaultValue
  return current || 'off'
}

export const setStoredReasoningEffort = (modelKey, effort) => {
  if (!modelKey || EFFORT_RANK[effort] === undefined || typeof localStorage === 'undefined') return
  let values = {}
  let valuesV2 = {}
  try {
    values = JSON.parse(localStorage.getItem(REASONING_PREFERENCES_KEY) || '{}') || {}
  } catch {
    values = {}
  }
  try {
    valuesV2 = JSON.parse(localStorage.getItem(REASONING_PREFERENCES_KEY_V2) || '{}') || {}
  } catch {
    valuesV2 = {}
  }
  values[modelKey] = effort
  valuesV2[modelKey] = effort
  localStorage.setItem(REASONING_PREFERENCES_KEY, JSON.stringify(values))
  localStorage.setItem(REASONING_PREFERENCES_KEY_V2, JSON.stringify(valuesV2))
}

export async function fetchReasoningCapabilities({ providerId, modelId, model, provider, signal }) {
  if (!providerId || !modelId) {
    return inferReasoningProfile({ providerId, modelId, model, provider })
  }
  const params = new URLSearchParams({
    provider: providerId,
    model: modelId,
  })
  const providerType = provider?.apiConfig?.protocol
  if (providerType) params.set('provider_type', providerType)
  if (typeof provider?.apiConfig?.supportsReasoning === 'boolean') {
    params.set('supports_reasoning', String(provider.apiConfig.supportsReasoning))
  }
  // Model declarations are more specific than provider defaults and are sent
  // with the same read-only capability request. This removes the race between
  // adding a custom model in the UI and its asynchronous models.json write.
  const modelConfig = model?.metadata || {}
  const providerConfig = provider?.apiConfig || {}
  const reasoningConfig = {
    reasoningMode: modelConfig.reasoningMode || providerConfig.reasoningMode,
    reasoningOptions: modelConfig.reasoningOptions || providerConfig.reasoningOptions,
    reasoningDefault: modelConfig.reasoningDefault || providerConfig.reasoningDefault,
    reasoningAlwaysEnabled: typeof modelConfig.reasoningAlwaysEnabled === 'boolean'
      ? modelConfig.reasoningAlwaysEnabled
      : providerConfig.reasoningAlwaysEnabled,
    reasoningOffControl: modelConfig.reasoningOffControl || providerConfig.reasoningOffControl,
    reasoningOnControl: modelConfig.reasoningOnControl || providerConfig.reasoningOnControl,
  }
  if (reasoningConfig.reasoningMode) params.set('reasoning_mode', reasoningConfig.reasoningMode)
  if (Array.isArray(reasoningConfig.reasoningOptions) && reasoningConfig.reasoningOptions.length > 0) {
    params.set('reasoning_options', reasoningConfig.reasoningOptions.join(','))
  }
  if (reasoningConfig.reasoningDefault) params.set('reasoning_default', reasoningConfig.reasoningDefault)
  if (typeof reasoningConfig.reasoningAlwaysEnabled === 'boolean') {
    params.set('reasoning_always_enabled', String(reasoningConfig.reasoningAlwaysEnabled))
  }
  if (reasoningConfig.reasoningOffControl) params.set('reasoning_off_control', reasoningConfig.reasoningOffControl)
  if (reasoningConfig.reasoningOnControl) params.set('reasoning_on_control', reasoningConfig.reasoningOnControl)
  const response = await fetch(`/api/models/reasoning-capabilities?${params.toString()}`, { signal })
  if (!response.ok) throw new Error(`reasoning_capabilities_${response.status}`)
  return normalizeReasoningProfile(await response.json(), inferReasoningProfile({ providerId, modelId, model, provider }))
}

export const getReasoningFallbackText = (resolution) => {
  if (resolution?.enabled && resolution?.output_observed === false) {
    return '当前接口本次没有返回可展示的思考文本'
  }
  if (!resolution?.fallback || !resolution?.effective) return ''
  const requestedOption = getReasoningEffortOption(resolution.requested)
  const effectiveOption = getReasoningEffortOption(resolution.effective)
  // 即使响应来自尚未重启的旧后端，也按 canonical value 展示，避免把
  // 历史中文 labels 泄漏到当前 UI。
  const requestedLabel = requestedOption.label
  const effectiveLabel = effectiveOption.label
  return `当前模型不支持“${requestedLabel}”，已按“${effectiveLabel}”执行`
}
