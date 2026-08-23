import React, { useState, useRef, useEffect, useMemo } from 'react'
import { ChevronUp, Check, Search, ChevronRight, Brain, LockKeyhole } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { useDefaults } from '../contexts/DefaultsContext'
import { useModel } from '../contexts/ModelContext'
import { useProvider } from '../contexts/ProviderContext'
import { useChatParams } from '../contexts/ChatParamsContext'
import ProviderAvatar from './ProviderAvatar'
import { filterChatModels, groupModelsByProvider, formatModelKey, filterModelsByKeyword } from '../utils/modelQuickSwitchUtils'
import {
  REASONING_EFFORT_OPTIONS,
  fetchReasoningCapabilities,
  getReasoningEffortOption,
  getStoredReasoningEffort,
  inferReasoningProfile,
  normalizeReasoningProfile,
  resolveModelReasoningEffort,
  setStoredReasoningEffort,
} from '../services/reasoningEffortService'

// 能力查询是本地元数据请求，不包含凭据；按 provider:model 缓存可避免切换菜单时重复读取。
const reasoningProfileCache = new Map()
const EFFORT_RANK = Object.fromEntries(
  REASONING_EFFORT_OPTIONS.map((option, index) => [option.value, index])
)

const nearestSupportedEffort = (requested, profile) => {
  const options = Array.isArray(profile?.options) && profile.options.length > 0
    ? profile.options
    : ['off']
  if (profile?.always_enabled) {
    if (requested === 'off') {
      return profile.default && options.includes(profile.default)
        ? profile.default
        : options.find((value) => value !== 'off') || 'off'
    }
    if (options.includes(requested)) return requested
  }
  if (options.includes(requested)) return requested
  if (requested === 'off' && options.includes('off')) return 'off'
  const activeOptions = options.filter((value) => value !== 'off')
  if (activeOptions.length === 0) return 'off'
  const requestedRank = EFFORT_RANK[requested] ?? EFFORT_RANK.medium
  return activeOptions.reduce((best, value) => {
    const distance = Math.abs((EFFORT_RANK[value] ?? 0) - requestedRank)
    const bestDistance = Math.abs((EFFORT_RANK[best] ?? 0) - requestedRank)
    return distance < bestDistance || (distance === bestDistance && (EFFORT_RANK[value] ?? 0) < (EFFORT_RANK[best] ?? 0))
      ? value
      : best
  }, activeOptions[0])
}

const syncReasoningSliderVisual = (
  slider,
  progressElement,
  position,
  options,
  { snap = false, syncValue = false } = {}
) => {
  const sliderOptions = Array.isArray(options) ? options : []
  const max = Math.max(0, sliderOptions.length - 1)
  const numericPosition = Number(position)
  const clampedPosition = Math.min(
    max,
    Math.max(0, Number.isFinite(numericPosition) ? numericPosition : 0)
  )
  const index = Math.round(clampedPosition)
  const visualPosition = snap ? index : clampedPosition
  const progress = max > 0 ? visualPosition / max : 0
  const option = sliderOptions[index]

  if (slider) {
    if (syncValue) slider.value = String(visualPosition)
    slider.setAttribute('aria-valuenow', String(index))
    if (option) {
      slider.setAttribute('aria-valuetext', `${option.label}，${option.description}`)
    }
  }
  if (progressElement) {
    progressElement.style.transform = `scaleX(${progress})`
  }

  return { index, option, visualPosition }
}

const applyReasoningSliderDetent = (position, max) => {
  const numericPosition = Number(position)
  const clampedPosition = Math.min(
    max,
    Math.max(0, Number.isFinite(numericPosition) ? numericPosition : 0)
  )
  const nearestIndex = Math.round(clampedPosition)
  const distance = clampedPosition - nearestIndex
  const absoluteDistance = Math.abs(distance)
  const holdRadius = 0.045
  const releaseRadius = 0.18

  if (absoluteDistance <= holdRadius) return nearestIndex
  if (absoluteDistance >= releaseRadius) return clampedPosition

  // 档位中心保留一个很窄的驻留区，越过后用 Hermite 曲线连续恢复到 1:1 跟手。
  const releaseSpan = releaseRadius - holdRadius
  const progress = (absoluteDistance - holdRadius) / releaseSpan
  const progressSquared = progress * progress
  const progressCubed = progressSquared * progress
  const endPositionWeight = (-2 * progressCubed) + (3 * progressSquared)
  const endSlopeWeight = progressCubed - progressSquared
  const releasedDistance = (endPositionWeight * releaseRadius) + (endSlopeWeight * releaseSpan)
  return nearestIndex + (Math.sign(distance) * releasedDistance)
}

/**
 * 模型快速切换器组件
 *
 * 在聊天输入框工具栏中显示当前模型，并提供向上弹出的下拉菜单
 * 允许用户快速切换 chat 类型的 AI 模型，无需进入设置页面。
 *
 * 数据流：
 * - 通过 useDefaults 读取/写入默认助手模型（assistantModel）
 * - 通过 useModel 获取模型列表和模型详情
 * - 通过 useProvider 获取已启用的 Provider 列表和详情
 *
 * 无 Props，所有数据通过 Context Hooks 获取。
 */
export default function ModelQuickSwitch({ onThinkingChange }) {
  // ========== Context Hooks ==========
  const { getDefaultModel, setDefaultModel } = useDefaults()
  const { getModelsByType, getModelById } = useModel()
  const { getEnabledProviders, getProviderById } = useProvider()

  // ========== ChatParamsContext — 深度思考力度 ==========
  const { reasoningEffort, setReasoningEffort } = useChatParams()

  // ========== 内部状态 ==========
  // 控制模型下拉菜单的显示/隐藏
  const [isOpen, setIsOpen] = useState(false)
  // 控制深度思考力度菜单的显示/隐藏
  const [isEffortMenuOpen, setIsEffortMenuOpen] = useState(false)
  // 搜索关键词状态
  const [searchQuery, setSearchQuery] = useState('')
  // 搜索输入框 ref，用于自动聚焦
  const searchInputRef = useRef(null)

  // 用于点击外部关闭的 ref
  const dropdownRef = useRef(null)
  // 深度思考力度菜单 ref
  const effortMenuRef = useRef(null)
  const effortSliderRef = useRef(null)
  const effortProgressRef = useRef(null)
  const effortSliderPointerActiveRef = useRef(false)

  // 注意：expandedProviders 状态在 currentProviderId 解析之后声明（见下方）

  // ========== 当前选中模型解析 ==========
  // 从 DefaultsContext 获取当前助手模型 key（格式：providerId:modelId）
  const assistantModelKey = getDefaultModel('assistantModel')

  // 解析 providerId 和 modelId
  const [currentProviderId, currentModelId] = useMemo(() => {
    if (!assistantModelKey || !assistantModelKey.includes(':')) {
      return [null, null]
    }
    const parts = assistantModelKey.split(':')
    return [parts[0], parts.slice(1).join(':')]
  }, [assistantModelKey])

  // 获取当前 Provider 和 Model 的详细信息
  const currentProvider = currentProviderId ? getProviderById(currentProviderId) : null
  const currentModel = (currentModelId && currentProviderId)
    ? getModelById(currentModelId, currentProviderId)
    : null

  // ========== 深度思考能力 ==========
  // 后端能力接口是唯一事实来源；本地推断只在接口不可用时使用。
  const reasoningProviderFingerprint = JSON.stringify({
    protocol: currentProvider?.apiConfig?.protocol || '',
    supportsReasoning: typeof currentProvider?.apiConfig?.supportsReasoning === 'boolean'
      ? currentProvider.apiConfig.supportsReasoning
      : 'auto',
    mode: currentProvider?.apiConfig?.reasoningMode || '',
    options: currentProvider?.apiConfig?.reasoningOptions || [],
    default: currentProvider?.apiConfig?.reasoningDefault || '',
    alwaysEnabled: currentProvider?.apiConfig?.reasoningAlwaysEnabled ?? null,
    offControl: currentProvider?.apiConfig?.reasoningOffControl || '',
    onControl: currentProvider?.apiConfig?.reasoningOnControl || '',
    modelMode: currentModel?.metadata?.reasoningMode || '',
    modelOptions: currentModel?.metadata?.reasoningOptions || [],
    modelDefault: currentModel?.metadata?.reasoningDefault || '',
    modelAlwaysEnabled: currentModel?.metadata?.reasoningAlwaysEnabled ?? null,
    modelOffControl: currentModel?.metadata?.reasoningOffControl || '',
    modelOnControl: currentModel?.metadata?.reasoningOnControl || '',
  })
  const reasoningPreferenceKey = currentProviderId && currentModelId
    ? `${currentProviderId}:${currentModelId}`
    : ''
  const reasoningModelKey = currentProviderId && currentModelId
    ? `${currentProviderId}:${currentModelId}:${reasoningProviderFingerprint}`
    : ''
  const inferredReasoningProfile = useMemo(() => (
    normalizeReasoningProfile(inferReasoningProfile({
      providerId: currentProviderId,
      modelId: currentModelId,
      model: currentModel,
      provider: currentProvider,
    }))
  ), [currentModel, currentModelId, currentProvider, currentProviderId])
  const [reasoningProfileState, setReasoningProfileState] = useState(null)
  const [reasoningProfileLoading, setReasoningProfileLoading] = useState(false)
  const [reasoningNotice, setReasoningNotice] = useState('')

  useEffect(() => {
    if (!reasoningModelKey) {
      setReasoningProfileState(null)
      setReasoningProfileLoading(false)
      return undefined
    }

    const cached = reasoningProfileCache.get(reasoningModelKey)
    if (cached) {
      setReasoningProfileState({ key: reasoningModelKey, profile: cached })
      setReasoningProfileLoading(false)
      return undefined
    }

    const controller = new AbortController()
    let active = true
    setReasoningProfileLoading(true)
    setReasoningProfileState(null)
    fetchReasoningCapabilities({
      providerId: currentProviderId,
      modelId: currentModelId,
      model: currentModel,
      provider: currentProvider,
      signal: controller.signal,
    }).then((profile) => {
      if (!active) return
      const normalized = normalizeReasoningProfile(profile, inferredReasoningProfile)
      reasoningProfileCache.set(reasoningModelKey, normalized)
      setReasoningProfileState({ key: reasoningModelKey, profile: normalized })
    }).catch((error) => {
      if (!active || error?.name === 'AbortError') return
      // 保守兜底：接口失败时仍允许已知模型使用本地推断，但不会凭空增加 Ultra。
      const fallback = normalizeReasoningProfile(inferredReasoningProfile)
      reasoningProfileCache.set(reasoningModelKey, fallback)
      setReasoningProfileState({ key: reasoningModelKey, profile: fallback })
      setReasoningNotice('模型能力读取失败，已使用本地兼容设置')
    }).finally(() => {
      if (active) setReasoningProfileLoading(false)
    })

    return () => {
      active = false
      controller.abort()
    }
  }, [currentModel, currentModelId, currentProvider, currentProviderId, inferredReasoningProfile, reasoningModelKey])

  const reasoningProfile = reasoningProfileState?.key === reasoningModelKey
    ? reasoningProfileState.profile
    : null
  const isAlwaysThinking = Boolean(reasoningProfile?.always_enabled)
  const reasoningOptions = useMemo(() => (
    (reasoningProfile?.options || []).map((value) => ({
      ...getReasoningEffortOption(value),
      // 档位名称始终使用协议原文；profile.labels 只兼容旧接口，不能覆盖
      // low/high/max 等 canonical value。
      label: getReasoningEffortOption(value).label,
    }))
  ), [reasoningProfile])
  const supportsThinking = reasoningOptions.some((option) => option.value !== 'off')
  const effectiveReasoningEffort = reasoningProfile
    ? nearestSupportedEffort(reasoningEffort, reasoningProfile)
    : reasoningEffort
  const isThinkingActive = isAlwaysThinking || effectiveReasoningEffort !== 'off'
  const isEffortLocked = isAlwaysThinking && reasoningOptions.length <= 1

  // 每个模型保存自己的请求档位。必须等能力档案就绪再写入，否则会把
  // 全局 off 抢先存成 V4 的偏好，思考被 thinking.disabled 关掉。
  useEffect(() => {
    if (!reasoningPreferenceKey || !reasoningProfile || reasoningProfileLoading) return
    const stored = getStoredReasoningEffort(reasoningPreferenceKey)
    const nextEffort = resolveModelReasoningEffort({
      stored,
      profile: reasoningProfile,
      current: reasoningEffort,
    })
    if (nextEffort !== reasoningEffort) setReasoningEffort(nextEffort)
    if (stored !== nextEffort) setStoredReasoningEffort(reasoningPreferenceKey, nextEffort)
  // 只在模型身份 / 档案变化时恢复；主动选择由 handleSelectEffort 持久化。
  }, [reasoningPreferenceKey, reasoningProfile, reasoningProfileLoading])

  // 模型切换或旧配置升级后只提示实际降级，不改写用户请求值。
  useEffect(() => {
    if (!reasoningProfile || reasoningProfileLoading) return
    const nextEffort = nearestSupportedEffort(reasoningEffort, reasoningProfile)
    onThinkingChange?.(nextEffort !== 'off' || reasoningProfile.always_enabled)
    if (nextEffort === reasoningEffort) return
    const requestedLabel = getReasoningEffortOption(reasoningEffort).label
    const effectiveLabel = getReasoningEffortOption(nextEffort).label
    setReasoningNotice(`当前模型不支持“${requestedLabel}”，本轮按“${effectiveLabel}”执行`)
  }, [onThinkingChange, reasoningEffort, reasoningProfile, reasoningProfileLoading])

  useEffect(() => {
    if (!reasoningNotice) return undefined
    const timer = setTimeout(() => setReasoningNotice(''), 4800)
    return () => clearTimeout(timer)
  }, [reasoningNotice])

  // 选择力度级别
  const handleSelectEffort = (effortValue, { closeMenu = true } = {}) => {
    if (!reasoningProfile || !reasoningProfile.options.includes(effortValue)) return
    setReasoningEffort(effortValue)
    setStoredReasoningEffort(reasoningPreferenceKey, effortValue)
    onThinkingChange?.(effortValue !== 'off' || reasoningProfile.always_enabled)
    setReasoningNotice('')
    if (closeMenu) setIsEffortMenuOpen(false)
  }

  // 获取当前力度的显示配置
  const currentEffort = reasoningOptions.find((option) => option.value === effectiveReasoningEffort)
    || reasoningOptions.find((option) => option.value !== 'off')
    || getReasoningEffortOption('off')
  const currentEffortIndex = Math.max(
    0,
    reasoningOptions.findIndex((option) => option.value === currentEffort.value)
  )
  const effortSliderMax = Math.max(0, reasoningOptions.length - 1)
  const effortProgress = effortSliderMax > 0
    ? currentEffortIndex / effortSliderMax
    : 0

  useEffect(() => {
    if (!isEffortMenuOpen || effortSliderPointerActiveRef.current) return
    syncReasoningSliderVisual(
      effortSliderRef.current,
      effortProgressRef.current,
      currentEffortIndex,
      reasoningOptions,
      { snap: true, syncValue: true }
    )
  }, [currentEffortIndex, isEffortMenuOpen, reasoningModelKey, reasoningOptions])

  const commitEffortSliderPosition = (slider, position) => {
    const { option } = syncReasoningSliderVisual(
      slider,
      effortProgressRef.current,
      position,
      reasoningOptions,
      { snap: true, syncValue: true }
    )
    if (option && option.value !== effectiveReasoningEffort) {
      handleSelectEffort(option.value, { closeMenu: false })
    }
  }

  const finishEffortSliderDrag = (slider) => {
    if (!effortSliderPointerActiveRef.current) return
    effortSliderPointerActiveRef.current = false
    commitEffortSliderPosition(slider, slider?.value)
  }

  const handleEffortSliderKeyDown = (event) => {
    const keyOffset = {
      ArrowLeft: -1,
      ArrowDown: -1,
      PageDown: -1,
      ArrowRight: 1,
      ArrowUp: 1,
      PageUp: 1,
    }[event.key]
    let nextIndex = null
    if (typeof keyOffset === 'number') {
      nextIndex = Math.min(effortSliderMax, Math.max(0, currentEffortIndex + keyOffset))
    } else if (event.key === 'Home') {
      nextIndex = 0
    } else if (event.key === 'End') {
      nextIndex = effortSliderMax
    }
    if (nextIndex === null) return
    event.preventDefault()
    commitEffortSliderPosition(event.currentTarget, nextIndex)
  }

  // ========== 折叠分组状态 ==========
  // 展开的 Provider 集合，初始化时仅展开当前选中模型所在的 Provider
  const [expandedProviders, setExpandedProviders] = useState(() => {
    return currentProviderId ? new Set([currentProviderId]) : new Set()
  })

  // ========== Chat 模型过滤和按 Provider 分组 ==========
  const groupedModels = useMemo(() => {
    // 获取所有 chat 类型模型
    const chatModels = getModelsByType('chat')
    // 获取所有已启用的 Provider
    const enabledProviders = getEnabledProviders()

    // 使用辅助函数过滤并分组
    const filteredModels = filterChatModels(chatModels, enabledProviders)
    // 新增：按搜索关键词过滤
    const searchFiltered = filterModelsByKeyword(filteredModels, searchQuery)
    return groupModelsByProvider(searchFiltered, enabledProviders)
  }, [getModelsByType, getEnabledProviders, searchQuery])

  // ========== 模型切换处理 ==========
  /**
   * 选择模型：更新默认助手模型并关闭下拉菜单
   * @param {string} providerId - Provider ID
   * @param {string} modelId - 模型 ID
   */
  const handleSelectModel = (providerId, modelId) => {
    setDefaultModel('assistantModel', formatModelKey(providerId, modelId))
    setIsOpen(false)
  }

  // ========== Provider 折叠/展开切换 ==========
  // 切换 Provider 折叠/展开
  const toggleProvider = (providerId) => {
    setExpandedProviders(prev => {
      const next = new Set(prev)
      if (next.has(providerId)) {
        next.delete(providerId)
      } else {
        next.add(providerId)
      }
      return next
    })
  }

  // 是否正在搜索
  const isSearching = searchQuery.trim().length > 0

  // ========== 下拉菜单打开时自动聚焦搜索框，关闭时清空搜索 ==========
  useEffect(() => {
    if (isOpen) {
      // 延迟聚焦，等待动画完成
      setTimeout(() => searchInputRef.current?.focus(), 100)
    } else {
      setSearchQuery('')
    }
  }, [isOpen])

  // ========== 点击外部关闭（模型下拉菜单） ==========
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  // ========== 点击外部关闭（力度选择菜单） ==========
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (effortMenuRef.current && !effortMenuRef.current.contains(event.target)) {
        setIsEffortMenuOpen(false)
      }
    }

    if (isEffortMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isEffortMenuOpen])

  // ========== 渲染 ==========
  return (
    <div className="relative flex shrink-0 items-center gap-1.5 pr-1">
    <div ref={dropdownRef} className="relative shrink-0">
      {/* 触发按钮 - 显示当前模型信息和展开指示器 */}
      <button
        type="button"
        onClick={() => setIsOpen(prev => !prev)}
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        aria-label={`切换模型，当前 ${currentModel?.name || '未选择'}`}
        title={currentModel?.name || '选择模型'}
        className="group flex h-7 w-[176px] items-center gap-1.5 overflow-hidden whitespace-nowrap rounded-full border border-black/[0.07] bg-white px-2 text-[11px] text-gray-700 shadow-[0_3px_10px_-6px_rgba(0,0,0,0.28)] transition-[background-color,border-color,box-shadow,transform] hover:border-black/[0.11] hover:bg-[#FCFCFB] hover:shadow-[0_5px_13px_-7px_rgba(0,0,0,0.32)] active:scale-[0.99] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black/10 dark:border-white/[0.09] dark:bg-white/[0.055] dark:text-gray-200 dark:shadow-none dark:hover:border-white/[0.15] dark:hover:bg-white/[0.09]"
      >
        {/* 当前模型的 Provider 图标 */}
        {currentProvider && (
          <ProviderAvatar provider={currentProvider} size={13} className="flex-shrink-0" />
        )}
        {/* 模型名称或占位文本 */}
        <span className="min-w-0 flex-1 truncate text-left font-medium">
          {currentModel?.name || '选择模型'}
        </span>
        {/* 向上箭头图标，指示可展开 */}
        <ChevronUp className={`h-3 w-3 shrink-0 text-gray-400 transition-transform duration-200 group-hover:text-gray-600 dark:group-hover:text-gray-300 ${isOpen ? '' : 'rotate-180'}`} />
      </button>

      {/* 向上弹出的下拉菜单 - 使用 framer-motion 动画 */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            role="dialog"
            aria-label="选择模型"
            initial={{ opacity: 0, y: 8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.95 }}
            transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            style={{ transformOrigin: 'bottom' }}
            className="absolute bottom-full left-0 z-50 mb-2 w-[296px] overflow-hidden rounded-[16px] border border-[#E9E4DE] bg-white p-1.5 text-xs shadow-[0_18px_42px_-20px_rgba(48,42,36,0.34),0_4px_12px_-8px_rgba(48,42,36,0.12)] dark:border-white/[0.10] dark:bg-[#282b31] dark:shadow-[0_18px_42px_-18px_rgba(0,0,0,0.78)]"
          >
            {/* 搜索输入框 */}
            <div className="flex items-center gap-2 rounded-[10px] bg-[#F7F5F2] px-2.5 py-2 dark:bg-white/[0.055]">
              <Search className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="搜索模型..."
                className="flex-1 bg-transparent text-xs text-gray-700 placeholder-gray-400 outline-none dark:text-gray-200 dark:placeholder-gray-500"
              />
            </div>
            {/* 可滚动的内容区域 */}
            <div className="mt-1 max-h-[300px] overflow-y-auto overscroll-contain">
              {groupedModels.length === 0 ? (
                /* 空状态提示：根据是否有搜索词显示不同文案 */
                <div className="py-6 text-center text-gray-400 dark:text-gray-500">
                  {searchQuery.trim() ? '无匹配模型' : '没有可用的模型'}
                </div>
              ) : (
                groupedModels.map(({ provider, models }, groupIndex) => (
                  <div key={provider.id} className={groupIndex > 0 ? 'mt-1' : ''}>
                    {/* Provider 分组标题 - 可点击折叠/展开 */}
                    <button
                      onClick={() => toggleProvider(provider.id)}
                      className="flex w-full select-none items-center gap-1.5 rounded-[9px] px-2 py-1.5 font-medium text-gray-500 transition-colors hover:bg-[#F7F5F2] dark:text-gray-400 dark:hover:bg-white/[0.055] dark:hover:text-gray-200"
                    >
                      <ChevronRight className={`w-3 h-3 transition-transform ${
                        (isSearching || expandedProviders.has(provider.id)) ? 'rotate-90' : ''
                      }`} />
                      <ProviderAvatar provider={provider} size={16} className="flex-shrink-0" />
                      <span className="truncate flex-1 text-left">{provider.name}</span>
                      <span className="text-gray-400 text-[10px]">{models.length}</span>
                    </button>
                    {/* 该 Provider 下的模型列表 — 搜索时强制展开，否则根据折叠状态 */}
                    <AnimatePresence initial={false}>
                    {(isSearching || expandedProviders.has(provider.id)) && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ type: 'spring', damping: 20, stiffness: 300 }}
                        style={{ overflow: 'hidden' }}
                      >
                      {models.map(model => {
                        // 判断是否为当前选中的模型
                        const isSelected = currentProviderId === provider.id && currentModelId === model.id
                        return (
                          <button
                            key={`${provider.id}:${model.id}`}
                            onClick={() => handleSelectModel(provider.id, model.id)}
                            title={model.name}
                            className={`flex w-full items-center justify-between gap-2 rounded-[9px] px-2.5 py-1.5 transition-colors ${
                              isSelected
                                ? 'bg-[#FDF0EA] font-medium text-[#B85F47] dark:bg-[#FFA07A]/12 dark:text-[#FFD1C1]'
                                : 'text-gray-700 hover:bg-[#F7F5F2] dark:text-gray-300 dark:hover:bg-white/[0.055]'
                            }`}
                          >
                            {/* 模型名称 */}
                            <span className="truncate">{model.name}</span>
                            {/* 选中状态的勾选图标 */}
                            {isSelected && (
                              <Check className="h-3.5 w-3.5 flex-shrink-0 text-[#B85F47] dark:text-[#FFA07A]" />
                            )}
                          </button>
                        )
                      })}
                      </motion.div>
                    )}
                    </AnimatePresence>
                  </div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>

    {/* 深度思考力度按钮 — 菜单由当前模型能力动态裁剪 */}
    {(supportsThinking || reasoningProfileLoading) && (
      <div ref={effortMenuRef} className="relative w-[88px] shrink-0">
        <button
          type="button"
          disabled={reasoningProfileLoading || isEffortLocked}
          onClick={() => !reasoningProfileLoading && !isEffortLocked && setIsEffortMenuOpen(prev => !prev)}
          aria-label={`思考档位 ${reasoningProfileLoading ? '读取中' : currentEffort.label}`}
          aria-haspopup="dialog"
          aria-expanded={isEffortMenuOpen}
          title={isEffortLocked
            ? `${currentModel?.name || '当前模型'} 始终启用固定思考档位`
            : isAlwaysThinking
              ? `思考始终启用，可选择深度：${currentEffort.label}`
            : reasoningProfileLoading
              ? '正在读取当前模型的思考能力'
              : `${reasoningProfile?.note || '深度思考'}：${currentEffort.label}`
          }
          className={`group flex h-7 w-full items-center justify-center gap-1 whitespace-nowrap rounded-full border px-2 text-[11px] outline-none transition-[background-color,border-color,color,box-shadow,transform] active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-black/10 ${
            isEffortLocked
              ? 'cursor-default border-transparent bg-[#F0F0EF] text-gray-600 shadow-[0_3px_10px_-6px_rgba(0,0,0,0.24)] dark:border-white/[0.08] dark:bg-white/[0.08] dark:text-gray-200 dark:shadow-none'
              : reasoningProfileLoading
                ? 'border-transparent bg-[#F2F2F1] text-gray-400 shadow-[0_3px_10px_-6px_rgba(0,0,0,0.24)] dark:border-white/[0.09] dark:bg-white/[0.055] dark:text-gray-500 dark:shadow-none'
              : isThinkingActive
                ? 'border-transparent bg-[#F0F0EF] text-gray-700 shadow-[0_3px_10px_-6px_rgba(0,0,0,0.26)] hover:bg-[#EAEAE8] hover:shadow-[0_5px_13px_-7px_rgba(0,0,0,0.30)] dark:border-white/[0.08] dark:bg-white/[0.08] dark:text-gray-200 dark:hover:bg-white/[0.12]'
                : 'border-black/[0.06] bg-white text-gray-500 shadow-[0_3px_10px_-6px_rgba(0,0,0,0.24)] hover:border-black/[0.10] hover:bg-[#FCFCFB] dark:border-white/[0.09] dark:bg-white/[0.055] dark:text-gray-300 dark:shadow-none dark:hover:border-white/[0.15] dark:hover:bg-white/[0.09]'
          }`}
        >
          <span className="relative flex h-3.5 w-3.5 shrink-0 items-center justify-center">
            <Brain size={13} strokeWidth={1.9} className={isThinkingActive ? 'text-gray-600 dark:text-gray-300' : 'text-gray-400 dark:text-gray-500'} />
            {isAlwaysThinking && (
              <LockKeyhole size={7} className="absolute -right-1 -top-1 opacity-60" aria-hidden="true" />
            )}
          </span>
          <span className="min-w-0 flex-1 text-center font-semibold tabular-nums">
            {reasoningProfileLoading ? '...' : currentEffort.label}
          </span>
          {!reasoningProfileLoading && !isEffortLocked && (
            <ChevronUp size={11} className={`shrink-0 opacity-55 transition-transform duration-200 ${isEffortMenuOpen ? 'rotate-180' : ''}`} aria-hidden="true" />
          )}
          {(reasoningProfileLoading || isEffortLocked) && <span className="h-3 w-3 shrink-0" aria-hidden="true" />}
        </button>

        {/* 力度选择弹出菜单 — 向上弹出（原生推理模型不弹出） */}
        <AnimatePresence>
          {isEffortMenuOpen && !isEffortLocked && reasoningProfile && (
            <motion.div
              role="dialog"
              aria-label="选择思考档位"
              initial={{ opacity: 0, y: 6, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 6, scale: 0.95 }}
              transition={{ type: 'spring', damping: 26, stiffness: 380, mass: 0.72 }}
              style={{ transformOrigin: 'bottom' }}
              className="absolute bottom-full right-0 z-50 mb-2 w-[244px] overflow-hidden rounded-[18px] border border-black/[0.06] bg-white px-4 py-3 text-xs shadow-[0_20px_46px_-18px_rgba(0,0,0,0.34),0_5px_16px_-9px_rgba(0,0,0,0.18)] dark:border-white/[0.10] dark:bg-[#282b31] dark:shadow-[0_20px_46px_-16px_rgba(0,0,0,0.82)]"
            >
              <div className="relative h-8" title={reasoningProfile.note || '思考深度'}>
                <div className="pointer-events-none absolute left-[13px] right-[13px] top-1/2 h-2 -translate-y-1/2 rounded-full bg-[#E7E7E5] dark:bg-white/[0.10]" />
                <div className="pointer-events-none absolute left-[13px] right-[13px] top-1/2 h-2 -translate-y-1/2 overflow-hidden rounded-full">
                  <div
                    ref={effortProgressRef}
                    className="h-full w-full origin-left rounded-full bg-[#92918D] will-change-transform dark:bg-gray-400"
                    style={{ transform: `scaleX(${effortProgress})` }}
                  />
                </div>
                <div className="pointer-events-none absolute left-[15px] right-[15px] top-1/2 flex -translate-y-1/2 items-center justify-between">
                  {reasoningOptions.map((option, index) => (
                    <span
                      key={option.value}
                      className={`h-1 w-1 rounded-full ${index <= currentEffortIndex ? 'bg-white/85' : 'bg-[#BABAB7] dark:bg-gray-500'}`}
                    />
                  ))}
                </div>
                <input
                  ref={effortSliderRef}
                  type="range"
                  min="0"
                  max={effortSliderMax}
                  step="0.01"
                  defaultValue={currentEffortIndex}
                  onPointerDown={(event) => {
                    effortSliderPointerActiveRef.current = true
                    event.currentTarget.setPointerCapture?.(event.pointerId)
                  }}
                  onInput={(event) => {
                    const detentedPosition = applyReasoningSliderDetent(
                      event.currentTarget.value,
                      effortSliderMax
                    )
                    syncReasoningSliderVisual(
                      event.currentTarget,
                      effortProgressRef.current,
                      detentedPosition,
                      reasoningOptions,
                      { syncValue: true }
                    )
                  }}
                  onPointerUp={(event) => finishEffortSliderDrag(event.currentTarget)}
                  onPointerCancel={(event) => finishEffortSliderDrag(event.currentTarget)}
                  onBlur={(event) => finishEffortSliderDrag(event.currentTarget)}
                  onKeyDown={handleEffortSliderKeyDown}
                  aria-label="思考档位"
                  aria-valuenow={currentEffortIndex}
                  aria-valuetext={`${currentEffort.label}，${currentEffort.description}`}
                  className="reasoning-effort-range absolute inset-0 z-10 h-8 w-full"
                />
              </div>
              <div className="mt-1 flex items-center justify-between px-0.5 text-[10px] font-medium text-gray-400 dark:text-gray-500">
                <span>Faster</span>
                <span>Smarter</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    )}
    <AnimatePresence initial={false}>
      {reasoningNotice && (
        <motion.span
          initial={{ opacity: 0, y: 3 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 3 }}
          role="status"
          className="pointer-events-none absolute bottom-full left-0 z-[60] mb-2 max-w-[260px] truncate rounded-lg bg-[#fff8f3] px-2.5 py-1.5 text-[10px] text-[#9a5b45] shadow-md ring-1 ring-[#e7cfc3]/70 dark:bg-[#3b2d29] dark:text-[#ffc5ae] dark:ring-[#806052]/70"
        >
          {reasoningNotice}
        </motion.span>
      )}
    </AnimatePresence>
    </div>
  )
}
