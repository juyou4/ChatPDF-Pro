import React, { useState, useRef, useEffect, useMemo } from 'react'
import { ChevronUp, Check, Search, ChevronRight } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { useDefaults } from '../contexts/DefaultsContext'
import { useModel } from '../contexts/ModelContext'
import { useProvider } from '../contexts/ProviderContext'
import { useGlobalSettings } from '../contexts/GlobalSettingsContext'
import ProviderAvatar from './ProviderAvatar'
import { filterChatModels, groupModelsByProvider, formatModelKey, filterModelsByKeyword } from '../utils/modelQuickSwitchUtils'

// 深度思考力度选项配置
const EFFORT_OPTIONS = [
  { value: 'off', label: '关闭', color: 'gray' },
  { value: 'low', label: '低', color: 'green' },
  { value: 'medium', label: '中', color: 'yellow' },
  { value: 'high', label: '高', color: 'blue' },
]

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

  // ========== GlobalSettingsContext — 深度思考力度 ==========
  const { reasoningEffort, setReasoningEffort } = useGlobalSettings()

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

  // ========== 深度思考相关 ==========
  // 已知支持通过参数开启思考模式的 provider（非推理模型也能开启）
  const THINKING_CAPABLE_PROVIDERS = new Set([
    'deepseek',  // deepseek-chat 支持 thinking 参数
    'zhipu',     // GLM-4.5+ 支持 thinking 参数
    'openai',    // GPT-5/o3/o4 支持 reasoning_effort
    'minimax',   // M2 系列支持 reasoning_split
    'moonshot',  // kimi-k2-thinking 等自动输出
  ])

  // 原生推理模型自动思考的 provider（这些 provider 的推理模型无需额外参数，始终输出思考内容）
  const ALWAYS_THINKING_PROVIDERS = new Set([
    'moonshot',  // Kimi 思考模型自动输出 reasoning_content
    'doubao',    // 豆包 Seed 系列自动思考
  ])

  // 当前模型是否为原生推理模型（始终思考，不可关闭）
  const isAlwaysThinking = useMemo(() => {
    if (!currentModel?.tags?.includes('reasoning')) return false
    if (!currentProviderId) return false
    // 这些 provider 的推理模型始终自动思考
    if (ALWAYS_THINKING_PROVIDERS.has(currentProviderId)) return true
    // OpenAI o 系列、Gemini 推理系列等也是原生推理模型
    const nativeReasoningIds = new Set([
      'o3', 'o4-mini', 'o3-mini', 'o1', 'o1-mini',  // OpenAI 推理系列
      'deepseek-reasoner',                             // DeepSeek R1
      'grok-4', 'grok-4-1-fast', 'grok-3', 'grok-3-mini',  // Grok 推理系列
    ])
    if (nativeReasoningIds.has(currentModel.id)) return true
    // 硅基流动托管的推理模型
    if (currentModel.id.includes('DeepSeek-R1') || currentModel.id.includes('Qwen3-235B')) return true
    return false
  }, [currentModel, currentProviderId])

  // 当前模型是否支持深度思考（显示按钮的条件）
  const supportsThinking = useMemo(() => {
    if (!currentProviderId) return false
    // 原生推理模型 — 始终显示按钮（锁定状态）
    if (isAlwaysThinking) return true
    // 当前模型带 reasoning 标签但不是"始终思考"类型 — 可选开关
    if (currentModel?.tags?.includes('reasoning')) return true
    // provider 已知支持思考模式
    if (THINKING_CAPABLE_PROVIDERS.has(currentProviderId)) return true
    // 同 provider 下有 reasoning 模型
    const chatModels = getModelsByType('chat')
    return chatModels.some(
      m => m.providerId === currentProviderId && m.tags?.includes('reasoning')
    )
  }, [currentProviderId, currentModel, isAlwaysThinking, getModelsByType])

  // 当前力度是否已开启（非 off）
  const isThinkingActive = isAlwaysThinking || reasoningEffort !== 'off'

  // 切换模型时的自动调整
  useEffect(() => {
    if (isAlwaysThinking) {
      // 原生推理模型：强制设为 high（实际不影响后端行为，仅 UI 状态一致）
      if (reasoningEffort === 'off') {
        setReasoningEffort('high')
        onThinkingChange?.(true)
      }
    } else if (!supportsThinking && reasoningEffort !== 'off') {
      // 不支持思考的模型：自动关闭
      setReasoningEffort('off')
      onThinkingChange?.(false)
    }
  }, [supportsThinking, isAlwaysThinking, reasoningEffort, onThinkingChange, setReasoningEffort])

  // 选择力度级别
  const handleSelectEffort = (effortValue) => {
    setReasoningEffort(effortValue)
    onThinkingChange?.(effortValue !== 'off')
    setIsEffortMenuOpen(false)
  }

  // 获取当前力度的显示配置
  const currentEffort = EFFORT_OPTIONS.find(o => o.value === reasoningEffort) || EFFORT_OPTIONS[0]

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
    <div className="flex items-center gap-1">
    <div ref={dropdownRef} className="relative">
      {/* 触发按钮 - 显示当前模型信息和展开指示器 */}
      <button
        onClick={() => setIsOpen(prev => !prev)}
        className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-600 transition-colors p-1 rounded-md hover:bg-gray-50"
      >
        {/* 当前模型的 Provider 图标 */}
        {currentProvider && (
          <ProviderAvatar provider={currentProvider} size={18} className="flex-shrink-0" />
        )}
        {/* 模型名称或占位文本 */}
        <span className="truncate max-w-[150px]">
          {currentModel?.name || '选择模型'}
        </span>
        {/* 向上箭头图标，指示可展开 */}
        <ChevronUp className={`w-3.5 h-3.5 flex-shrink-0 transition-transform ${isOpen ? '' : 'rotate-180'}`} />
      </button>

      {/* 向上弹出的下拉菜单 - 使用 framer-motion 动画 */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.95 }}
            transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            style={{ transformOrigin: 'bottom' }}
            className="absolute bottom-full mb-2 left-0 min-w-[240px] rounded-xl shadow-lg border border-gray-100 bg-white p-1.5 text-xs z-50"
          >
            {/* 搜索输入框 */}
            <div className="flex items-center gap-1.5 px-2 py-1.5 border-b border-gray-100">
              <Search className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="搜索模型..."
                className="flex-1 bg-transparent text-xs text-gray-700 placeholder-gray-400 outline-none"
              />
            </div>
            {/* 可滚动的内容区域 */}
            <div className="max-h-[300px] overflow-y-auto overscroll-contain">
              {groupedModels.length === 0 ? (
                /* 空状态提示：根据是否有搜索词显示不同文案 */
                <div className="text-gray-400 text-center py-6">
                  {searchQuery.trim() ? '无匹配模型' : '没有可用的模型'}
                </div>
              ) : (
                groupedModels.map(({ provider, models }, groupIndex) => (
                  <div key={provider.id} className={groupIndex > 0 ? 'mt-1' : ''}>
                    {/* Provider 分组标题 - 可点击折叠/展开 */}
                    <button
                      onClick={() => toggleProvider(provider.id)}
                      className="w-full flex items-center gap-1.5 px-2 py-1.5 text-gray-500 font-medium select-none hover:bg-gray-50/60 rounded-lg transition-colors"
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
                            className={`w-full flex items-center justify-between gap-2 px-2 py-1.5 rounded-lg transition-colors ${
                              isSelected
                                ? 'bg-blue-50/80 text-blue-600 font-medium'
                                : 'text-gray-700 hover:bg-gray-100/60'
                            }`}
                          >
                            {/* 模型名称 */}
                            <span className="truncate">{model.name}</span>
                            {/* 选中状态的勾选图标 */}
                            {isSelected && (
                              <Check className="w-3.5 h-3.5 flex-shrink-0 text-blue-600" />
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

    {/* 深度思考力度按钮 — 点击弹出力度选择菜单，仅当 provider 支持思考模式时显示 */}
    {supportsThinking && (
      <div ref={effortMenuRef} className="relative">
        <button
          onClick={() => !isAlwaysThinking && setIsEffortMenuOpen(prev => !prev)}
          title={isAlwaysThinking
            ? `${currentModel?.name || '当前模型'} 为原生推理模型，始终启用深度思考`
            : `深度思考：${currentEffort.label}`
          }
          className={`flex items-center gap-1.5 text-xs px-3 py-1 rounded-full border transition-all ${
            isAlwaysThinking
              ? 'border-purple-300 bg-purple-50 text-purple-700 cursor-default'
              : isThinkingActive
                ? 'border-blue-300 bg-blue-50 text-blue-700 shadow-sm'
                : 'border-gray-200 bg-white text-gray-500 hover:border-gray-300 hover:text-gray-700'
          }`}
        >
          {/* 原子图标 */}
          <svg className="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <ellipse cx="12" cy="12" rx="10" ry="4" />
            <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)" />
            <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)" />
            <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
          </svg>
          {/* 显示当前力度级别标识 */}
          <span>
            {isAlwaysThinking
              ? '推理模型'
              : isThinkingActive ? `思考·${currentEffort.label}` : 'DeepThink'
            }
          </span>
        </button>

        {/* 力度选择弹出菜单 — 向上弹出（原生推理模型不弹出） */}
        <AnimatePresence>
          {isEffortMenuOpen && !isAlwaysThinking && (
            <motion.div
              initial={{ opacity: 0, y: 6, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 6, scale: 0.95 }}
              transition={{ type: 'spring', damping: 22, stiffness: 350 }}
              style={{ transformOrigin: 'bottom' }}
              className="absolute bottom-full mb-2 left-0 min-w-[120px] rounded-xl shadow-lg border border-gray-100 bg-white p-1 text-xs z-50"
            >
              {EFFORT_OPTIONS.map(option => {
                const isSelected = reasoningEffort === option.value
                return (
                  <button
                    key={option.value}
                    onClick={() => handleSelectEffort(option.value)}
                    className={`w-full flex items-center justify-between gap-2 px-3 py-1.5 rounded-lg transition-colors ${
                      isSelected
                        ? 'bg-blue-50/80 text-blue-600 font-medium'
                        : 'text-gray-700 hover:bg-gray-100/60'
                    }`}
                  >
                    <span>{option.label}</span>
                    {isSelected && (
                      <Check className="w-3.5 h-3.5 flex-shrink-0 text-blue-600" />
                    )}
                  </button>
                )
              })}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    )}
    </div>
  )
}
