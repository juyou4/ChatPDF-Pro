import React, { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Check,
  CheckCircle2,
  ChevronDown,
  Key,
  Plug,
  Plus,
  RefreshCw,
  Search,
  Server,
  Settings,
  Shield,
  Trash2,
  ChevronLeft,
  Box, Edit3, Link2, Play, ChevronRight, CheckSquare, Sparkles, Cloud, Moon, Fish, Cpu, MessageSquare, Zap, Settings2
} from 'lucide-react'

const GlassInput = ({ icon: Icon, placeholder, value, onChange, type = "text", disabled = false, ...props }) => (
  <div className="relative group w-full">
    {Icon && (
      <div className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400 group-focus-within:text-[#B85F47] transition-colors">
        <Icon size={18} />
      </div>
    )}
    <input 
      type={type} 
      placeholder={placeholder}
      value={value}
      onChange={onChange}
      disabled={disabled}
      className={`w-full bg-white border border-gray-200 rounded-[14px] text-[14px] text-gray-800 font-medium focus:outline-none focus:ring-2 focus:ring-[#FFA07A]/25 focus:border-[#FFA07A]/50 transition-colors ${Icon ? 'pl-10 pr-4 py-3' : 'px-4 py-3'} ${disabled ? 'opacity-60 cursor-not-allowed bg-gray-100' : ''} ${props.className || ''}`}
      {...props}
    />
  </div>
);

const Tag = ({ text, active, onClick }) => (
  <span 
    onClick={onClick}
    className={`${active ? 'bg-[#FFF4EF] text-[#B85F47] border-[#FFDCCF]' : 'bg-white text-gray-600 border-gray-200 hover:border-gray-300 hover:bg-gray-50'} border text-[12px] font-semibold px-3.5 py-1.5 rounded-[10px] cursor-pointer transition-colors`}
  >
    {text}
  </span>
);

/* 扁平列表行：选中态 = 白色悬浮卡（中性投影），切换时卡片在行间滑动跟随 */
const ProviderItem = ({ provider, isActive, onClick }) => (
  <div
    onClick={onClick}
    className={`relative flex items-center px-2.5 py-2 rounded-[14px] cursor-pointer transition-colors ${
      isActive ? '' : 'hover:bg-gray-100/70'
    }`}>
    {isActive && (
      <motion.span
        layoutId="provider-active-card"
        transition={{ type: 'spring', stiffness: 420, damping: 34, mass: 0.8 }}
        aria-hidden="true"
        className="absolute inset-0 rounded-[14px] bg-white ring-1 ring-gray-200/80 shadow-[0_10px_24px_-8px_rgba(30,30,35,0.2),0_3px_8px_-2px_rgba(30,30,35,0.08)]"
      >
        <span className="absolute left-0 top-1/2 h-6 w-[3px] -translate-y-1/2 rounded-full bg-[#F0653A]" />
      </motion.span>
    )}
    <div className={`relative z-10 w-9 h-9 rounded-full flex items-center justify-center shrink-0 overflow-hidden bg-white ${isActive ? 'shadow-[0_2px_6px_rgba(30,30,35,0.12)]' : 'shadow-[0_1px_3px_rgba(30,30,35,0.08)]'}`}>
      <ProviderAvatar providerId={provider.id} size={22} />
    </div>
    <div className="relative z-10 ml-2.5 flex-1 min-w-0">
      <h3 className={`text-[13px] truncate ${isActive ? 'font-extrabold text-gray-900' : 'font-bold text-gray-800'}`}>{provider.name}</h3>
      <p className={`text-[11px] font-medium truncate ${isActive ? 'text-[#B85F47]' : 'text-gray-400'}`}>{provider.id}</p>
    </div>
    <div className={`relative z-10 w-2 h-2 rounded-full shrink-0 ml-2 ${provider.enabled ? 'bg-green-400' : 'bg-gray-300'}`}></div>
  </div>
);
import { useProvider } from '../contexts/ProviderContext'
import { useModel } from '../contexts/ModelContext'
import { useDefaults } from '../contexts/DefaultsContext'
import ProviderAvatar from './ProviderAvatar'

/**
 * 模型标签选项列表
 * 用于新增模型表单中的多选标签组件
 */
const TAG_OPTIONS = [
  { value: 'free', label: '免费' },
  { value: 'vision', label: '视觉' },
  { value: 'chinese_optimized', label: '中文优化' },
  { value: 'reasoning', label: '推理' },
  { value: 'function_calling', label: '函数调用' },
  { value: 'web_search', label: '网络搜索' },
]

/**
 * 标签值到中文显示名称的映射
 * 用于模型列表中标签徽章的渲染
 */
const TAG_LABELS = {
  free: '免费',
  vision: '视觉',
  chinese_optimized: '中文优化',
  reasoning: '推理',
  function_calling: '函数调用',
  web_search: '网络搜索',
  embedding: 'Embedding',
  rerank: 'Rerank',
}

/**
 * “模型服务管理”面板
 * 对齐 cherry-studio 的三栏结构：左侧 Provider 列表，中间连接配置，右侧模型清单。
 */
export default function EmbeddingSettings({ isOpen, onClose, onExitComplete }) {
  const {
    providers,
    addProvider,
    updateProvider,
    testConnection,
    getProviderById
  } = useProvider()

  const {
    userCollection,
    systemModels,
    addModelToCollection,
    removeModelFromCollection,
    fetchAndAddModels,
    isFetching,
    fetchError
  } = useModel()

  const { getDefaultModel, setDefaultModel } = useDefaults()

  const [activeProviderId, setActiveProviderId] = useState(
    providers[0]?.id || null
  )
  const [providerSearch, setProviderSearch] = useState('')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [collapsedTypes, setCollapsedTypes] = useState({ chat: true, embedding: true, rerank: true, image: true })
  const [addModelForm, setAddModelForm] = useState({
    id: '',
    name: '',
    type: 'chat'
  })
  // 新增模型表单的标签选择状态
  const [newModelTags, setNewModelTags] = useState([])
  const [addSuccess, setAddSuccess] = useState(null)
  const [lastAddedModelKey, setLastAddedModelKey] = useState(null)
  // Optimistic update：保存进局部 state，不依赖 context 传播时序
  const [locallyAddedModels, setLocallyAddedModels] = useState([])
  const [customProviderFormOpen, setCustomProviderFormOpen] = useState(false)
  const [customProviderForm, setCustomProviderForm] = useState({
    id: '',
    name: '',
    apiHost: '',
    chat: true,
    embedding: true,
    rerank: false
  })

  // 默认模型键名映射
  const DEFAULT_TYPE_MAP = {
    embedding: 'embeddingModel',
    rerank: 'rerankModel',
    chat: 'assistantModel'
  }
  const TYPE_META = {
    chat: { label: 'Chat 对话' },
    embedding: { label: 'Embedding 向量' },
    rerank: { label: 'Rerank 重排' },
    image: { label: 'Image 图像' }
  }
  const GLASS_CARD_CLASS = 'soft-card'
  const RADIUS_CLASS = 'rounded-[18px]'

  const activeProvider = useMemo(
    () => providers.find(p => p.id === activeProviderId) || providers[0] || null,
    [providers, activeProviderId]
  )

  // 当 providers 变化时，保持选中第一项
  useEffect(() => {
    if (!activeProvider && providers.length > 0) {
      setActiveProviderId(providers[0].id)
    }
  }, [providers, activeProvider])

  const filteredProviders = useMemo(() => {
    const normalizedSearch = providerSearch.trim().toLowerCase()
    if (!normalizedSearch) return providers
    return providers.filter(p =>
      `${p.name} ${p.id}`.toLowerCase().includes(normalizedSearch)
    )
  }, [providerSearch, providers])

  // 直接订阅 userCollection + systemModels 基础状态（对齐 Cherry Studio 单一数据源模式）
  // 避免通过 getModelsByProvider 函数派生，确保状态更新时立即反映在 UI 上
  const modelsByType = useMemo(() => {
    if (!activeProvider) return {}
    const sysForProvider = systemModels.filter(m => m.providerId === activeProvider.id)
    const userForProvider = userCollection.filter(m => m.providerId === activeProvider.id)
    const localForProvider = locallyAddedModels.filter(m => m.providerId === activeProvider.id)
    // 优先级：local > user > system
    const map = new Map()
    sysForProvider.forEach(m => map.set(m.id, m))
    userForProvider.forEach(m => map.set(m.id, m))
    localForProvider.forEach(m => map.set(m.id, m))
    const list = Array.from(map.values())

    const grouped = list.reduce((acc, model) => {
      acc[model.type] = acc[model.type] || []
      acc[model.type].push(model)
      return acc
    }, {})

    Object.values(grouped).forEach(group => {
      group.sort((a, b) => {
        const aKey = `${a.providerId}:${a.id}`
        const bKey = `${b.providerId}:${b.id}`
        if (aKey === lastAddedModelKey) return -1
        if (bKey === lastAddedModelKey) return 1
        if (!!a.isUserAdded !== !!b.isUserAdded) return a.isUserAdded ? -1 : 1
        return (a.name || a.id).localeCompare(b.name || b.id, 'zh-Hans-CN', { sensitivity: 'base' })
      })
    })

    return grouped
  }, [activeProvider, systemModels, userCollection, locallyAddedModels, lastAddedModelKey])

  const handleProviderUpdate = (field, value) => {
    if (!activeProvider) return
    updateProvider(activeProvider.id, { [field]: value })
  }

  const handleTest = async () => {
    if (!activeProvider) {
      setTesting(false)
      return
    }
    setTesting(true)
    setTestResult(null)
    const result = await testConnection(activeProvider.id)
    setTestResult(result)
    setTesting(false)
  }

  const handleSyncModels = async () => {
    if (!activeProvider) return
    await fetchAndAddModels(activeProvider)
  }

  const handleAddModel = () => {
    if (!activeProvider || !addModelForm.id.trim()) return
    const modelType = addModelForm.type
    const newId = addModelForm.id.trim()
    const newModel = {
      id: newId,
      name: addModelForm.name.trim() || newId,
      providerId: activeProvider.id,
      type: modelType,
      capabilities: [{ type: modelType, isUserSelected: true }],
      tags: newModelTags,
      metadata: {},
      isSystem: false,
      isUserAdded: true
    }
    // 将模型写入局部 state（立即显示）和 context（持久化）
    setLocallyAddedModels(prev => {
      const exists = prev.some(m => m.id === newId && m.providerId === activeProvider.id)
      return exists ? prev : [...prev, newModel]
    })
    addModelToCollection(newModel)
    setLastAddedModelKey(`${activeProvider.id}:${newId}`)
    // 自动展开对应类型的分组，让用户看到新增的模型
    setCollapsedTypes(prev => ({ ...prev, [modelType]: false }))
    setAddSuccess(`已添加: ${newId} (${modelType})`)
    setTimeout(() => setAddSuccess(null), 3000)
    setAddModelForm({ id: '', name: '', type: 'chat' })
    // 重置标签选择
    setNewModelTags([])
  }

  const buildDefaultKey = (type, modelId) => `${activeProvider?.id || ''}:${modelId}`
  const isDefaultModel = (type, modelId) => {
    const key = DEFAULT_TYPE_MAP[type]
    if (!key) return false
    return getDefaultModel(key) === buildDefaultKey(type, modelId)
  }

  const handleSetDefault = (type, modelId) => {
    const key = DEFAULT_TYPE_MAP[type]
    if (!key) return
    setDefaultModel(key, buildDefaultKey(type, modelId))
  }

  const toggleCollapse = (type) => {
    setCollapsedTypes(prev => ({
      ...prev,
      [type]: !prev?.[type]
    }))
  }

  const handleAddCustomProvider = () => {
    if (!customProviderForm.id.trim() || !customProviderForm.name.trim() || !customProviderForm.apiHost.trim()) {
      alert('请填写 Provider ID、名称与 API 地址')
      return
    }

    addProvider({
      id: customProviderForm.id.trim(),
      name: customProviderForm.name.trim(),
      apiKey: '',
      apiHost: customProviderForm.apiHost.trim(),
      enabled: true,
      isSystem: false,
      capabilities: {
        chat: customProviderForm.chat,
        embedding: customProviderForm.embedding,
        rerank: customProviderForm.rerank
      },
      apiConfig: {
        chatEndpoint: '/chat/completions',
        embeddingEndpoint: '/embeddings',
        rerankEndpoint: '/rerank'
      }
    })

    setActiveProviderId(customProviderForm.id.trim())
    setCustomProviderForm({
      id: '',
      name: '',
      apiHost: '',
      chat: true,
      embedding: true,
      rerank: false
    })
  }

  /**
   * 根据模型ID获取对应的图标Provider ID
   * 用于在聚合厂商（如SiliconFlow）中显示具体的模型厂商图标
   */
  const getIconProviderId = (model) => {
    // 如果不是聚合厂商，直接返回原providerId
    // 目前主要针对 silicon (SiliconFlow) 做特殊处理，也可以扩展到其他聚合厂商
    if (model.providerId !== 'silicon' && model.providerId !== 'openrouter') {
      return model.providerId
    }

    const modelId = model.id.toLowerCase()

    // 映射规则
    if (modelId.includes('qwen')) return 'qwen'
    if (modelId.includes('deepseek')) return 'deepseek'
    if (modelId.includes('thudm') || modelId.includes('glm')) return 'zhipu'
    if (modelId.includes('01-ai') || modelId.includes('yi-')) return 'yi'
    if (modelId.includes('mistral')) return 'mistral'
    if (modelId.includes('google') || modelId.includes('gemma')) return 'google'
    if (modelId.includes('meta') || modelId.includes('llama')) return 'meta' // We don't have meta icon yet, might fallback to local or silicon
    if (modelId.includes('nvidia')) return 'nvidia'
    if (modelId.includes('baichuan')) return 'baichuan'
    if (modelId.includes('internlm')) return 'internlm' // No icon yet
    if (modelId.includes('hunyuan')) return 'hunyuan'
    if (modelId.includes('step')) return 'step'
    if (modelId.includes('cohere')) return 'cohere'

    return model.providerId
  }

  const renderModelRow = (model) => {
    const isRecentlyAdded = `${model.providerId}:${model.id}` === lastAddedModelKey
    return (
    <div
      key={`${model.providerId}-${model.id}`}
      className={`group flex items-center justify-between px-4 py-3 rounded-xl border transition-all ${isRecentlyAdded ? 'bg-green-50 border-green-200 shadow-sm' : 'hover:bg-[var(--color-bg-subtle)] border-transparent hover:border-purple-100'}`}
    >
      <div className="flex items-center gap-4 overflow-hidden">
        <ProviderAvatar providerId={getIconProviderId(model)} size={36} className="flex-shrink-0 shadow-sm" />
        <div className="min-w-0 flex flex-col gap-0.5">
          <div className="flex items-center gap-2 flex-wrap">
            <div className="text-sm font-bold text-gray-900 truncate" title={model.name || model.id}>
              {model.name || model.id}
            </div>
            {isRecentlyAdded && (
              <span className="flex-shrink-0 text-[10px] font-medium text-green-700 bg-green-100 px-1.5 py-0.5 rounded border border-green-200">
                刚添加
              </span>
            )}
            {model.metadata?.dimension && (
              <span className="flex-shrink-0 text-[10px] font-medium text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-100">
                {model.metadata.dimension}维
              </span>
            )}
            {/* 模型标签徽章 */}
            {model.tags?.map(tag => (
              <span key={tag} className="flex-shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-gray-50 text-gray-600 border border-gray-100">
                {TAG_LABELS[tag] || tag}
              </span>
            ))}
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="truncate max-w-[200px]" title={model.id}>{model.id}</span>
            {model.type === 'chat' && (
              <span className="flex-shrink-0 px-1.5 py-0.5 rounded bg-purple-50 text-purple-600 text-[10px] font-medium border border-purple-100">
                Chat
              </span>
            )}
            {model.type === 'embedding' && (
              <span className="flex-shrink-0 px-1.5 py-0.5 rounded bg-purple-50 text-purple-600 text-[10px] font-medium border border-purple-100">
                Embedding
              </span>
            )}
            {model.type === 'rerank' && (
              <span className="flex-shrink-0 px-1.5 py-0.5 rounded bg-green-50 text-green-600 text-[10px] font-medium border border-green-100">
                Rerank
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 pl-4 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => handleSetDefault(model.type, model.id)}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${isDefaultModel(model.type, model.id)
            ? 'bg-purple-600 text-white shadow-sm hover:bg-purple-700'
            : 'bg-white border border-gray-200 text-gray-600 hover:border-purple-300 hover:text-purple-600'
            }`}
        >
          {isDefaultModel(model.type, model.id) ? '默认' : '设为默认'}
        </button>
        {model.isUserAdded && (
          <button
            onClick={() => removeModelFromCollection(model.id, model.providerId)}
            className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors"
            title="删除模型"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
      {/* Always show default badge if it is default, even when not hovering */}
      {isDefaultModel(model.type, model.id) && (
        <div className="group-hover:hidden px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-50 text-purple-700 border border-purple-100">
          默认
        </div>
      )}
    </div>
  )
  }

  try {
    return (
      <AnimatePresence initial={false} onExitComplete={onExitComplete}>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.1 }}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/25 p-3 sm:p-6 font-sans overflow-hidden"
          >
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0, transition: { duration: 0.16, ease: [0.22, 1, 0.36, 1] } }}
              exit={{ opacity: 0, y: 6, transition: { duration: 0.09, ease: [0.4, 0, 1, 1] } }}
              className="settings-modal-surface settings-solid settings-shell w-full max-w-[1150px] h-[92vh] min-h-0 bg-[#f6f7f9] p-4 sm:p-5 border border-white/80 relative z-10 flex flex-col"
            >
              {/* Header */}
              <div className="flex items-center mb-4 px-2">
                <div className="flex items-center gap-2">
                  <button onClick={onClose} className="w-9 h-9 flex items-center justify-center bg-white hover:bg-gray-100 rounded-[12px] border border-gray-200 transition-colors text-gray-600" title="返回设置中心" aria-label="返回设置中心">
                    <ChevronLeft size={18} />
                  </button>
                  <div className="p-2 rounded-[12px] bg-[#FFF4EF] text-[#B85F47]">
                    <Settings2 className="w-5 h-5" />
                  </div>
                  <div>
                    <h2 className="text-[18px] font-bold text-gray-900">模型服务管理</h2>
                  </div>
                </div>
              </div>

              {/* Main Content Area */}
              <div className="flex flex-col lg:flex-row gap-4 flex-1 min-h-0 overflow-y-auto lg:overflow-hidden pb-2">
                {/* Left Column: Sidebar */}
                <div className="w-full lg:w-[220px] flex flex-col gap-4 shrink-0 min-h-[220px] lg:min-h-0">
                  {/* Search Bar */}
                  <div className="relative">
                    <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
                    <input 
                      type="text" 
                      value={providerSearch}
                      onChange={(e) => setProviderSearch(e.target.value)}
                      placeholder="搜索模型平台..." 
                      className="w-full bg-white border border-gray-200 rounded-[14px] pl-10 pr-4 py-3 text-[13px] text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#FFA07A]/25 focus:border-[#FFA07A]/50"
                    />
                  </div>

                  {/* Provider List */}
                  <div className="flex-1 overflow-y-auto pr-1 space-y-2 custom-scrollbar pb-2">
                    {filteredProviders.map(p => (
                      <ProviderItem 
                        key={p.id}
                        provider={p}
                        isActive={p.id === activeProvider?.id}
                        onClick={() => setActiveProviderId(p.id)}
                      />
                    ))}

                    <div className="mt-4 pt-3 border-t border-gray-200">
                      <div 
                        onClick={() => setCustomProviderFormOpen(!customProviderFormOpen)}
                        className="settings-card settings-card-interactive flex items-center p-3 rounded-[16px] cursor-pointer bg-white"
                      >
                        <div className="accent-control w-10 h-10 rounded-[14px] flex items-center justify-center shrink-0 shadow-inner">
                          <Settings2 size={20} />
                        </div>
                        <div className="ml-3 flex-1">
                          <h3 className="text-[14px] font-bold text-gray-800">自定义 Provider</h3>
                          <p className="text-[12px] text-gray-500 font-medium">(OpenAI 兼容)</p>
                        </div>
                        <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${customProviderFormOpen ? 'rotate-180' : ''}`} />
                      </div>

                      <AnimatePresence>
                        {customProviderFormOpen && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden mt-2"
                          >
                            <div className="settings-inset p-4 rounded-[16px] space-y-3">
                              <GlassInput 
                                placeholder="Provider ID (如 my-openai)" 
                                value={customProviderForm.id}
                                onChange={e => setCustomProviderForm({ ...customProviderForm, id: e.target.value })}
                              />
                              <GlassInput 
                                placeholder="显示名称" 
                                value={customProviderForm.name}
                                onChange={e => setCustomProviderForm({ ...customProviderForm, name: e.target.value })}
                              />
                              <GlassInput 
                                placeholder="API Host" 
                                value={customProviderForm.apiHost}
                                onChange={e => setCustomProviderForm({ ...customProviderForm, apiHost: e.target.value })}
                              />
                              <div className="flex items-center gap-3 text-[12px] text-gray-600 px-1">
                                <label className="flex items-center gap-1 cursor-pointer">
                                  <input type="checkbox" className="accent-[#FFA07A]" checked={customProviderForm.chat} onChange={e => setCustomProviderForm({ ...customProviderForm, chat: e.target.checked })} />
                                  Chat
                                </label>
                                <label className="flex items-center gap-1 cursor-pointer">
                                  <input type="checkbox" className="accent-[#FFA07A]" checked={customProviderForm.embedding} onChange={e => setCustomProviderForm({ ...customProviderForm, embedding: e.target.checked })} />
                                  Embedding
                                </label>
                                <label className="flex items-center gap-1 cursor-pointer">
                                  <input type="checkbox" className="accent-[#FFA07A]" checked={customProviderForm.rerank} onChange={e => setCustomProviderForm({ ...customProviderForm, rerank: e.target.checked })} />
                                  Rerank
                                </label>
                              </div>
                              <button
                                onClick={handleAddCustomProvider}
                                className="accent-surface w-full text-[13px] font-bold py-2.5 rounded-[14px] transition-all flex items-center justify-center gap-2"
                              >
                                <Plus size={16} /> 添加
                              </button>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </div>
                </div>

                {/* Middle Column: Configuration */}
                <div className="flex flex-col gap-4 min-w-0 w-full flex-1 overflow-visible lg:overflow-y-auto custom-scrollbar lg:pr-2 pb-2">
                  {/* Header Card */}
                  <div className="settings-card bg-white p-5 border border-gray-200/90 flex justify-between items-center shrink-0">
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-white rounded-[16px] flex items-center justify-center shadow-sm border border-gray-100 overflow-hidden">
                        <ProviderAvatar providerId={activeProvider?.id} size={32} />
                      </div>
                      <div>
                        <h2 className="text-[18px] font-bold text-gray-900">{activeProvider?.name || '未选择'}</h2>
                        <p className="text-[14px] text-gray-500 font-medium">{activeProvider?.id}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <label className="flex items-center gap-2.5 cursor-pointer select-none">
                        <span className={`text-[13px] font-bold ${activeProvider?.enabled ? 'text-gray-800' : 'text-gray-400'}`}>
                          {activeProvider?.enabled ? '已启用' : '已停用'}
                        </span>
                        <input
                          type="checkbox"
                          checked={!!activeProvider?.enabled}
                          onChange={e => handleProviderUpdate('enabled', e.target.checked)}
                          className="sr-only"
                        />
                        <span
                          aria-hidden="true"
                          className={`relative h-[22px] w-[38px] shrink-0 rounded-full transition-colors ${activeProvider?.enabled ? 'bg-[#F0653A]' : 'bg-gray-300'}`}
                        >
                          <span className={`absolute top-[3px] h-4 w-4 rounded-full bg-white shadow-sm transition-[left] duration-200 ${activeProvider?.enabled ? 'left-[18px]' : 'left-[3px]'}`} />
                        </span>
                      </label>
                    </div>
                  </div>

                  {/* API Settings Card */}
                  <div className="settings-card bg-white p-6 border border-gray-200/90 space-y-4 shrink-0">
                    <div className="space-y-1.5">
                      <label className="text-[13px] font-bold text-gray-700 ml-1">API Key</label>
                      <GlassInput 
                        icon={Key} 
                        type="password"
                        placeholder="sk-... (多个 Key 用逗号分隔)" 
                        value={activeProvider?.apiKey || ''}
                        onChange={e => handleProviderUpdate('apiKey', e.target.value)}
                      />
                    </div>
                    <div className="space-y-1.5 mt-2">
                      <label className="text-[13px] font-bold text-gray-700 ml-1">API 地址</label>
                      <GlassInput 
                        icon={Link2} 
                        placeholder="https://api.openai.com/v1" 
                        value={activeProvider?.apiHost || ''}
                        onChange={e => handleProviderUpdate('apiHost', e.target.value)}
                      />
                    </div>
                    <div className="flex gap-4 pt-3">
                      <button
                        onClick={handleTest}
                        disabled={!activeProvider || testing}
                        className="accent-cta flex-1 disabled:opacity-60 disabled:cursor-not-allowed text-[14px] font-bold py-3 rounded-full flex items-center justify-center gap-2"
                      >
                        {testing ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} className="fill-current" />}
                        <span>测试连接</span>
                      </button>
                      <button 
                        onClick={handleSyncModels}
                        disabled={!activeProvider || isFetching}
                        className="flex-1 bg-white hover:bg-gray-50 disabled:opacity-70 transition-colors text-gray-700 text-[14px] font-bold py-3 rounded-[14px] border border-gray-200 flex items-center justify-center gap-2"
                      >
                        <RefreshCw size={16} className={isFetching ? "animate-spin" : ""} />
                        {isFetching ? '同步中...' : '同步模型'}
                      </button>
                    </div>

                    {testResult && (
                      <div className={`mt-2 rounded-[16px] p-3 text-[13px] border font-medium ${testResult.success ? 'border-green-200 bg-green-50/80 text-green-700' : 'border-red-200 bg-red-50/80 text-red-700'}`}>
                        {testResult.success
                          ? `连接成功${testResult.latency ? ` (${testResult.latency}ms)` : ''}`
                          : '连接失败'
                        } {testResult.message || testResult.error || ''}
                      </div>
                    )}
                    {fetchError && (
                      <div className="mt-2 rounded-[16px] p-3 text-[13px] border border-amber-200 bg-amber-50/80 text-amber-700 font-medium">
                        {fetchError}
                      </div>
                    )}
                  </div>

                  {/* Add Model Card */}
                  <div className="settings-card bg-white p-6 border border-gray-200/90 flex flex-col flex-1 shrink-0">
                    <h3 className="text-[15px] font-bold text-gray-900 mb-4 shrink-0">手动新增模型</h3>
                    <div className="space-y-4 pr-2">
                      <div className="flex gap-4 shrink-0">
                        <div className="flex-[2]">
                          <GlassInput 
                            placeholder="模型 ID (如 gpt-4)" 
                            value={addModelForm.id}
                            onChange={e => setAddModelForm({ ...addModelForm, id: e.target.value })}
                          />
                        </div>
                        <div className="flex-1 relative">
                          <select 
                            className="w-full bg-white border border-gray-200 rounded-[14px] px-4 py-3 text-[14px] text-gray-800 font-medium focus:outline-none focus:ring-2 focus:ring-[#FFA07A]/25 appearance-none cursor-pointer"
                            value={addModelForm.type}
                            onChange={e => setAddModelForm({ ...addModelForm, type: e.target.value })}
                          >
                            <option value="chat">Chat</option>
                            <option value="embedding">Embedding</option>
                            <option value="rerank">Rerank</option>
                            <option value="image">Image</option>
                          </select>
                          <ChevronRight size={16} className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none rotate-90" />
                        </div>
                      </div>
                      <div className="shrink-0">
                        <GlassInput 
                          placeholder="显示名称 (可选)" 
                          value={addModelForm.name}
                          onChange={e => setAddModelForm({ ...addModelForm, name: e.target.value })}
                        />
                      </div>
                      
                      <div className="space-y-2 pt-1 shrink-0 pb-2">
                        <label className="text-[13px] font-bold text-gray-700 ml-1">标签 (可选)</label>
                        <div className="flex flex-wrap gap-2.5">
                          {TAG_OPTIONS.map(tag => (
                            <Tag 
                              key={tag.value} 
                              text={tag.label} 
                              active={newModelTags.includes(tag.value)}
                              onClick={() => {
                                if (newModelTags.includes(tag.value)) {
                                  setNewModelTags(newModelTags.filter(t => t !== tag.value))
                                } else {
                                  setNewModelTags([...newModelTags, tag.value])
                                }
                              }}
                            />
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="pt-4 shrink-0">
                      <button
                        onClick={handleAddModel}
                        className="accent-cta w-full text-[14px] font-bold py-3 rounded-full flex items-center justify-center gap-2"
                      >
                        <Plus size={18} />
                        <span>保存模型</span>
                      </button>
                      {addSuccess && (
                        <div className="mt-2 text-center text-[12px] text-green-600 font-medium animate-pulse">
                          {addSuccess}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right Column: Model List */}
                <div className="settings-card w-full lg:w-[340px] min-h-[360px] bg-white p-5 border border-gray-200/90 flex flex-col shrink-0">
                  <h3 className="text-[16px] font-bold text-gray-900 mt-2">模型列表</h3>
                  <p className="text-[12px] text-gray-500 font-medium mt-1 mb-4">按类型分组: 对话 / 嵌入 / 重排</p>
                  
                  <div className="flex-1 overflow-y-auto custom-scrollbar pr-1 space-y-4">
                    {['chat', 'embedding', 'rerank', 'image'].every(t => !(modelsByType[t] || []).length) && (
                      <div className="flex flex-col items-center justify-center py-12 text-center">
                        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gray-100/80 text-gray-400">
                          <Cpu size={20} />
                        </div>
                        <p className="mt-3 text-[13px] font-semibold text-gray-500">这个平台还没有模型</p>
                        <p className="mt-1 text-[11px] leading-relaxed text-gray-400">
                          填好 API Key 后点「同步模型」自动拉取，
                          <br />
                          或在「手动新增模型」中添加
                        </p>
                      </div>
                    )}
                    {['chat', 'embedding', 'rerank', 'image'].map(type => {
                      const list = modelsByType[type] || [];
                      if (list.length === 0) return null;
                      const meta = TYPE_META[type] || { label: type };
                      const isCollapsed = !!collapsedTypes[type];
                      const defaultLabel = (() => {
                        const key = DEFAULT_TYPE_MAP[type];
                        if (!key) return '—';
                        const current = getDefaultModel(key);
                        return current || '未选择';
                      })();
                      
                      return (
                        <div key={type} className="settings-inset rounded-[16px] overflow-hidden">
                          <button
                            type="button"
                            aria-expanded={!isCollapsed}
                            onClick={() => toggleCollapse(type)}
                            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors focus:outline-none"
                          >
                            <div className="flex items-center gap-2 shrink-0">
                              {type === 'chat' && <MessageSquare size={16} className="text-[#B85F47]" />}
                              {type === 'embedding' && <Cpu size={16} className="text-[#B85F47]" />}
                              {type === 'rerank' && <Zap size={16} className="text-[#B85F47]" />}
                              {type === 'image' && <Sparkles size={16} className="text-[#B85F47]" />}
                              <span className="text-[14px] font-bold text-gray-800 whitespace-nowrap">{meta.label}</span>
                              <span className="text-[11px] font-medium text-[#B85F47] bg-[#FFA07A]/10 px-1.5 py-0.5 rounded-full">{list.length}</span>
                            </div>
                            <div className="flex items-center gap-2 min-w-0 ml-2">
                              <CheckCircle2 size={12} className="text-gray-400 shrink-0" />
                              <span className="text-[11px] text-gray-500 font-medium truncate max-w-[160px]" title={defaultLabel}>默认: {defaultLabel}</span>
                              <ChevronDown size={14} className={`text-gray-400 shrink-0 transition-transform ${isCollapsed ? '-rotate-90' : ''}`} />
                            </div>
                          </button>
                          
                          <AnimatePresence initial={false}>
                            {!isCollapsed && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ type: 'spring', stiffness: 320, damping: 24 }}
                                className="overflow-hidden"
                              >
                                <div className="p-2 pt-0 space-y-1">
                                  {list.map(model => (
                                      <div key={model.id} className={`bg-gray-50 hover:bg-gray-100 rounded-[14px] p-2.5 flex items-center justify-between border ${lastAddedModelKey === `${model.providerId}:${model.id}` ? 'border-green-300' : 'border-gray-100'} cursor-pointer transition-colors group`}>
                                        <div className="flex items-center space-x-2.5 overflow-hidden min-w-0 flex-1">
                                          <div className="w-8 h-8 rounded-[10px] bg-white flex items-center justify-center shrink-0 border border-gray-100 shadow-sm">
                                            <ProviderAvatar providerId={getIconProviderId(model)} size={20} />
                                          </div>
                                          <div className="flex flex-col flex-1 min-w-0 py-0.5">
                                            <div className="flex flex-wrap items-center gap-1.5 mb-1">
                                              <h4 className="text-[13px] font-bold text-gray-800 break-words line-clamp-2 leading-snug" title={model.name || model.id}>{model.name || model.id}</h4>
                                              {model.tags?.map(tag => (
                                                <span key={tag} className="shrink-0 text-[9px] px-1 py-0.5 rounded bg-blue-50 text-blue-600 border border-blue-100">
                                                  {TAG_LABELS[tag] || tag}
                                                </span>
                                              ))}
                                              {model.metadata?.dimension && (
                                                <span className="shrink-0 text-[9px] px-1 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-100">
                                                  {model.metadata.dimension}维
                                                </span>
                                              )}
                                            </div>
                                            <div className="flex items-start gap-1.5 mt-0.5">
                                               {isDefaultModel(model.type, model.id) && <span className="text-[10px] bg-[#FFA07A]/10 text-[#B85F47] px-1.5 py-0.5 rounded-sm font-bold shrink-0 mt-px">默认</span>}
                                               <p className="text-[11px] text-gray-400 font-medium break-all line-clamp-2 leading-snug" title={model.id}>{model.id}</p>
                                            </div>
                                          </div>
                                        </div>
                                        <div className="flex gap-1 shrink-0 ml-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                            {!isDefaultModel(model.type, model.id) && (
                                              <button onClick={(e) => { e.stopPropagation(); handleSetDefault(model.type, model.id); }} className="p-1.5 text-gray-400 hover:text-[#B85F47] rounded-md hover:bg-[#FFA07A]/10" title="设为默认">
                                                <CheckCircle2 size={16} />
                                              </button>
                                            )}
                                            {model.isUserAdded && (
                                              <button onClick={(e) => { e.stopPropagation(); removeModelFromCollection(model.id, model.providerId); }} className="p-1.5 text-gray-400 hover:text-red-500 rounded-md hover:bg-red-50" title="删除">
                                                <Trash2 size={16} />
                                              </button>
                                            )}
                                        </div>
                                      </div>
                                  ))}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      );
                    })}
                    
                    {(!modelsByType || Object.keys(modelsByType).length === 0) && (
                      <div className="text-center text-sm text-gray-500 py-8 bg-gray-50 rounded-[16px] border border-dashed border-gray-300">
                        暂无模型
                      </div>
                    )}
                  </div>
                </div>

              </div>
              <style dangerouslySetInnerHTML={{__html: `
                @keyframes shimmer {
                  100% { left: 125%; }
                }
                .animate-shimmer {
                  animation: shimmer 1.5s infinite;
                }
                /* Hide scrollbar for clean UI but allow scrolling */
                .custom-scrollbar::-webkit-scrollbar {
                  width: 4px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                  background: transparent;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                  background: rgba(184, 95, 71, 0.12);
                  border-radius: 10px;
                }
                .custom-scrollbar:hover::-webkit-scrollbar-thumb {
                  background: rgba(184, 95, 71, 0.24);
                }
                .animate-blob {
                  animation: blob 7s infinite;
                }
                .animation-delay-2000 {
                  animation-delay: 2s;
                }
                @keyframes blob {
                  0% { transform: translate(0px, 0px) scale(1); }
                  33% { transform: translate(30px, -50px) scale(1.1); }
                  66% { transform: translate(-20px, 20px) scale(0.9); }
                  100% { transform: translate(0px, 0px) scale(1); }
                }
              `}} />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    )
  } catch (err) {
    console.error('EmbeddingSettings render error', err)
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/25 p-4 font-sans">
        <div className="settings-solid settings-shell bg-white border border-white/80 p-6 max-w-lg w-full">
          <div className="text-lg font-bold text-gray-900 mb-2">模型服务管理加载失败</div>
          <div className="text-sm text-gray-600 mb-4 bg-red-50 p-3 rounded-xl border border-red-100">{err?.message || '未知错误'}</div>
          <button
            onClick={onClose}
            className="accent-surface w-full text-[14px] font-bold py-3 rounded-[16px] transition-colors"
          >
            关闭
          </button>
        </div>
      </div>
    )
  }
}
