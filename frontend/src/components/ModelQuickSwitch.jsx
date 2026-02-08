import React, { useState, useRef, useEffect, useMemo } from 'react'
import { ChevronUp, Check, Search, ChevronRight } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { useDefaults } from '../contexts/DefaultsContext'
import { useModel } from '../contexts/ModelContext'
import { useProvider } from '../contexts/ProviderContext'
import ProviderAvatar from './ProviderAvatar'
import { filterChatModels, groupModelsByProvider, formatModelKey, filterModelsByKeyword } from '../utils/modelQuickSwitchUtils'

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
export default function ModelQuickSwitch() {
  // ========== Context Hooks ==========
  const { getDefaultModel, setDefaultModel } = useDefaults()
  const { getModelsByType, getModelById } = useModel()
  const { getEnabledProviders, getProviderById } = useProvider()

  // ========== 内部状态 ==========
  // 控制下拉菜单的显示/隐藏
  const [isOpen, setIsOpen] = useState(false)
  // 搜索关键词状态
  const [searchQuery, setSearchQuery] = useState('')
  // 搜索输入框 ref，用于自动聚焦
  const searchInputRef = useRef(null)

  // 用于点击外部关闭的 ref
  const dropdownRef = useRef(null)

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

  // ========== 点击外部关闭 ==========
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

  // ========== 渲染 ==========
  return (
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
            initial={{ opacity: 0, y: 8, scaleY: 0.95 }}
            animate={{ opacity: 1, y: 0, scaleY: 1 }}
            exit={{ opacity: 0, y: 8, scaleY: 0.95 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            style={{ transformOrigin: 'bottom' }}
            className="absolute bottom-full mb-2 left-0 min-w-[240px] rounded-xl shadow-xl border border-white/20 backdrop-blur-md bg-white/80 p-1.5 text-xs z-50"
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
                    {(isSearching || expandedProviders.has(provider.id)) && (
                      models.map(model => {
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
                      })
                    )}
                  </div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
