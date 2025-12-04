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
  X
} from 'lucide-react'
import { useProvider } from '../contexts/ProviderContext'
import { useModel } from '../contexts/ModelContext'
import { useDefaults } from '../contexts/DefaultsContext'
import ProviderAvatar from './ProviderAvatar'

/**
 * “模型服务管理”面板
 * 对齐 cherry-studio 的三栏结构：左侧 Provider 列表，中间连接配置，右侧模型清单。
 */
export default function EmbeddingSettings({ isOpen, onClose }) {
  const {
    providers,
    addProvider,
    updateProvider,
    testConnection,
    getProviderById
  } = useProvider()

  const {
    getModelsByProvider,
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
  const [collapsedTypes, setCollapsedTypes] = useState({})
  const [addModelForm, setAddModelForm] = useState({
    id: '',
    name: '',
    type: 'chat'
  })
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
  const GLASS_CARD_CLASS = 'bg-white/60 backdrop-blur-md border border-white/45 shadow-[0_14px_40px_-20px_rgba(15,23,42,0.35)]'
  const RADIUS_CLASS = 'rounded-[32px]'

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

  const filteredProviders = providers.filter(p =>
    `${p.name} ${p.id}`.toLowerCase().includes(providerSearch.toLowerCase())
  )

  const modelsByType = useMemo(() => {
    if (!activeProvider) return {}
    const list = getModelsByProvider(activeProvider.id)
    return list.reduce((acc, model) => {
      acc[model.type] = acc[model.type] || []
      acc[model.type].push(model)
      return acc
    }, {})
  }, [activeProvider, getModelsByProvider])

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
    addModelToCollection({
      id: addModelForm.id.trim(),
      name: addModelForm.name.trim() || addModelForm.id.trim(),
      providerId: activeProvider.id,
      type: addModelForm.type,
      metadata: {},
      isSystem: false,
      isUserAdded: true
    })
    setAddModelForm({ id: '', name: '', type: 'chat' })
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

  const renderModelRow = (model) => (
    <div
      key={`${model.providerId}-${model.id}`}
      className="group flex items-center justify-between px-4 py-3 rounded-xl hover:bg-gray-50 border border-transparent hover:border-blue-100 transition-all"
    >
      <div className="flex items-center gap-4 overflow-hidden">
        <ProviderAvatar providerId={getIconProviderId(model)} size={36} className="flex-shrink-0 shadow-sm" />
        <div className="min-w-0 flex flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <div className="text-sm font-bold text-gray-900 truncate" title={model.name || model.id}>
              {model.name || model.id}
            </div>
            {model.metadata?.dimension && (
              <span className="flex-shrink-0 text-[10px] font-medium text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-100">
                {model.metadata.dimension}维
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="truncate max-w-[200px]" title={model.id}>{model.id}</span>
            {model.type === 'chat' && (
              <span className="flex-shrink-0 px-1.5 py-0.5 rounded bg-blue-50 text-blue-600 text-[10px] font-medium border border-blue-100">
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
            ? 'bg-blue-600 text-white shadow-sm hover:bg-blue-700'
            : 'bg-white border border-gray-200 text-gray-600 hover:border-blue-300 hover:text-blue-600'
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
        <div className="group-hover:hidden px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-50 text-blue-700 border border-blue-100">
          默认
        </div>
      )}
    </div>
  )

  try {
    return (
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/25 backdrop-blur-sm p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 10 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30, mass: 0.8 }}
              className={`w-full max-w-6xl max-h-[92vh] bg-white/80 backdrop-blur-2xl border border-white/50 shadow-[0_30px_80px_-35px_rgba(15,23,42,0.6)] ${RADIUS_CLASS} overflow-hidden flex flex-col`}
            >
              {/* Header */}
              <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-xl bg-blue-50 text-blue-700">
                    <Server className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-lg font-bold text-gray-900">模型服务管理</div>
                    <div className="text-xs text-gray-500">集中配置所有厂商与模型（对话 / 嵌入 / 重排）</div>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 rounded-full hover:bg-gray-100 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex flex-1 min-h-0">
                {/* Left: provider list */}
                <div className="w-64 border-r border-gray-100 p-4 flex flex-col">
                  <div className="relative mb-3">
                    <Search className="w-4 h-4 text-gray-400 absolute left-3 top-3" />
                    <input
                      value={providerSearch}
                      onChange={(e) => setProviderSearch(e.target.value)}
                      placeholder="搜索模型平台..."
                      className="w-full pl-9 pr-3 py-2 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                    />
                  </div>
                  <div className="space-y-2 overflow-y-auto pr-1">
                    {filteredProviders.length === 0 && (
                      <div className="text-xs text-gray-500 px-3 py-2">
                        暂无服务商，请先添加或检查配置。
                      </div>
                    )}
                    {filteredProviders.map(p => (
                      <button
                        key={p.id}
                        onClick={() => setActiveProviderId(p.id)}
                        className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl border text-left transition-colors ${p.id === activeProvider?.id
                          ? 'border-blue-200 bg-blue-50/60 text-blue-700'
                          : 'border-gray-200 hover:border-blue-200 hover:bg-blue-50/30 text-gray-700'
                          }`}
                      >
                        <ProviderAvatar providerId={p.id} className="w-8 h-8" />
                        <div className="flex-1">
                          <div className="text-sm font-semibold">{p.name}</div>
                          <div className="text-[11px] text-gray-500">{p.id}</div>
                        </div>
                        <div
                          className={`w-2.5 h-2.5 rounded-full ${p.enabled ? 'bg-green-500' : 'bg-gray-300'}`}
                          title={p.enabled ? '已启用' : '未启用'}
                        />
                      </button>
                    ))}
                  </div>

                  <div className="mt-4 space-y-2">
                    <button
                      onClick={() => setCustomProviderFormOpen(v => !v)}
                      className="w-full flex items-center justify-between text-xs font-semibold text-gray-700 px-2 py-2 rounded-lg border border-gray-200 hover:border-blue-200 hover:bg-blue-50/40 transition"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-blue-400" />
                        自定义 Provider（OpenAI 兼容）
                      </div>
                      <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${customProviderFormOpen ? '' : '-rotate-90'}`} />
                    </button>
                    <AnimatePresence initial={false}>
                      {customProviderFormOpen && (
                        <motion.div
                          key="custom-provider-form"
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ type: 'spring', stiffness: 260, damping: 28 }}
                          className="overflow-hidden"
                        >
                          <div className="p-3 rounded-xl border border-gray-100 bg-white/70 space-y-2 shadow-sm">
                            <input
                              className="soft-input px-3 py-2 rounded-lg border border-gray-200 w-full"
                              placeholder="providerId"
                              value={customProviderForm.id}
                              onChange={e => setCustomProviderForm({ ...customProviderForm, id: e.target.value })}
                            />
                            <input
                              className="soft-input px-3 py-2 rounded-lg border border-gray-200 w-full"
                              placeholder="名称"
                              value={customProviderForm.name}
                              onChange={e => setCustomProviderForm({ ...customProviderForm, name: e.target.value })}
                            />
                            <input
                              className="soft-input px-3 py-2 rounded-lg border border-gray-200 w-full"
                              placeholder="API Host（OpenAI 兼容地址，如 https://api.your.com/v1）"
                              value={customProviderForm.apiHost}
                              onChange={e => setCustomProviderForm({ ...customProviderForm, apiHost: e.target.value })}
                            />
                            <div className="flex items-center gap-3 text-xs text-gray-600">
                              <label className="flex items-center gap-1">
                                <input type="checkbox" checked={customProviderForm.chat} onChange={e => setCustomProviderForm({ ...customProviderForm, chat: e.target.checked })} />
                                Chat
                              </label>
                              <label className="flex items-center gap-1">
                                <input type="checkbox" checked={customProviderForm.embedding} onChange={e => setCustomProviderForm({ ...customProviderForm, embedding: e.target.checked })} />
                                Embedding
                              </label>
                              <label className="flex items-center gap-1">
                                <input type="checkbox" checked={customProviderForm.rerank} onChange={e => setCustomProviderForm({ ...customProviderForm, rerank: e.target.checked })} />
                                Rerank
                              </label>
                            </div>
                            <button
                              onClick={handleAddCustomProvider}
                              className="w-full soft-button soft-button-primary rounded-lg py-2 text-sm flex items-center justify-center gap-2"
                            >
                              <Plus className="w-4 h-4" />
                              添加并启用
                            </button>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>

                {/* Middle + Right */}
                <div className="flex-1 grid grid-cols-2 gap-0 min-w-0">
                  {/* Provider detail */}
                  <div className="border-r border-gray-100 p-4 flex flex-col gap-4 min-h-0">
                    <div className="flex items-center gap-3">
                      <ProviderAvatar providerId={activeProvider?.id} className="w-10 h-10" />
                      <div>
                        <div className="text-base font-semibold text-gray-900">{activeProvider?.name || '未选择'}</div>
                        <div className="text-xs text-gray-500">{activeProvider?.id}</div>
                      </div>
                      <label className="ml-auto inline-flex items-center gap-2 text-sm text-gray-700">
                        <input
                          type="checkbox"
                          checked={!!activeProvider?.enabled}
                          onChange={e => handleProviderUpdate('enabled', e.target.checked)}
                          className="accent-blue-600 w-4 h-4"
                        />
                        启用
                      </label>
                    </div>

                    <div className="space-y-3">
                      <div>
                        <label className="text-xs text-gray-600">API Key</label>
                        <div className="relative mt-1">
                          <Key className="w-4 h-4 text-gray-400 absolute left-3 top-3" />
                          <input
                            value={activeProvider?.apiKey || ''}
                            onChange={e => handleProviderUpdate('apiKey', e.target.value)}
                            placeholder="sk-..."
                            type="password"
                            className="w-full pl-10 pr-3 py-2 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                          />
                        </div>
                      </div>
                      <div>
                        <label className="text-xs text-gray-600">API 地址</label>
                        <div className="relative mt-1">
                          <Plug className="w-4 h-4 text-gray-400 absolute left-3 top-3" />
                          <input
                            value={activeProvider?.apiHost || ''}
                            onChange={e => handleProviderUpdate('apiHost', e.target.value)}
                            placeholder="https://api.openai.com/v1"
                            className="w-full pl-10 pr-3 py-2 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleTest}
                        disabled={!activeProvider || testing}
                        className="soft-button soft-button-primary px-4 py-2 rounded-xl text-sm flex items-center gap-2 disabled:opacity-60"
                      >
                        {testing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Shield className="w-4 h-4" />}
                        测试连接
                      </button>
                      <button
                        onClick={handleSyncModels}
                        disabled={!activeProvider || isFetching}
                        className="px-4 py-2 rounded-xl border border-gray-200 text-sm text-gray-700 hover:border-blue-200 hover:text-blue-700 flex items-center gap-2 disabled:opacity-60"
                      >
                        <RefreshCw className="w-4 h-4" />
                        同步模型
                      </button>
                    </div>

                    {testResult && (
                      <div className={`rounded-xl p-3 text-sm border ${testResult.success ? 'border-green-200 bg-green-50 text-green-700' : 'border-red-200 bg-red-50 text-red-700'}`}>
                        {testResult.success ? '连接成功' : '连接失败'} {testResult.message || testResult.error || ''}
                      </div>
                    )}
                    {fetchError && (
                      <div className="rounded-xl p-3 text-sm border border-amber-200 bg-amber-50 text-amber-700">
                        {fetchError}
                      </div>
                    )}

                    {/* Add model form */}
                    <div className="mt-auto border-t border-gray-100 pt-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-sm font-semibold text-gray-800">新增模型</div>
                        <Settings className="w-4 h-4 text-gray-400" />
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <input
                          className="soft-input px-3 py-2 rounded-lg border border-gray-200"
                          placeholder="modelId"
                          value={addModelForm.id}
                          onChange={e => setAddModelForm({ ...addModelForm, id: e.target.value })}
                        />
                        <select
                          className="soft-input px-3 py-2 rounded-lg border border-gray-200"
                          value={addModelForm.type}
                          onChange={e => setAddModelForm({ ...addModelForm, type: e.target.value })}
                        >
                          <option value="chat">Chat</option>
                          <option value="embedding">Embedding</option>
                          <option value="rerank">Rerank</option>
                          <option value="image">Image</option>
                        </select>
                        <input
                          className="soft-input px-3 py-2 rounded-lg border border-gray-200 col-span-2"
                          placeholder="显示名称（可选）"
                          value={addModelForm.name}
                          onChange={e => setAddModelForm({ ...addModelForm, name: e.target.value })}
                        />
                        <button
                          onClick={handleAddModel}
                          className="col-span-2 soft-button soft-button-primary rounded-lg py-2 text-sm flex items-center justify-center gap-2"
                        >
                          <Plus className="w-4 h-4" />
                          保存模型
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Models list */}
                  <div className="p-4 flex flex-col gap-3 min-h-0">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-base font-semibold text-gray-900">模型列表</div>
                        <div className="text-xs text-gray-500">按类型分组：对话 / 嵌入 / 重排</div>
                      </div>
                      <div className="text-xs text-gray-400 flex items-center gap-1">
                        <ChevronDown className="w-4 h-4" />
                        {getModelsByProvider(activeProvider?.id || '').length || 0} 个
                      </div>
                    </div>

                    <div className="flex-1 overflow-y-auto space-y-4 pr-1">
                      {['chat', 'embedding', 'rerank', 'image'].map(type => {
                        const list = modelsByType[type] || []
                        if (list.length === 0) return null
                        const meta = TYPE_META[type] || { label: type }
                        const isCollapsed = !!collapsedTypes[type]
                        const defaultLabel = (() => {
                          const key = DEFAULT_TYPE_MAP[type]
                          if (!key) return '—'
                          const current = getDefaultModel(key)
                          return current || '未选择'
                        })()

                        return (
                          <div key={type} className={`${GLASS_CARD_CLASS} ${RADIUS_CLASS}`}>
                            <button
                              type="button"
                              aria-expanded={!isCollapsed}
                              onClick={() => toggleCollapse(type)}
                              className={`w-full px-3 py-2 flex items-center justify-between bg-white/35 border-b border-white/40 ${isCollapsed ? RADIUS_CLASS : `${RADIUS_CLASS} rounded-b-none`} cursor-pointer hover:bg-white/55 focus:outline-none focus:ring-2 focus:ring-blue-200/60`}
                            >
                              <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                                {meta.label}
                                <span className="text-[11px] text-gray-400">({list.length})</span>
                              </div>
                              <div className="text-[11px] text-gray-500 flex items-center gap-2 min-w-0">
                                <CheckCircle2 className="w-3 h-3 shrink-0" />
                                <span className="truncate max-w-[180px]">默认：{defaultLabel}</span>
                                <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${isCollapsed ? '-rotate-90' : ''}`} />
                              </div>
                            </button>
                            <AnimatePresence initial={false}>
                              {!isCollapsed && (
                                <motion.div
                                  key={`${type}-list`}
                                  initial={{ height: 0, opacity: 0 }}
                                  animate={{ height: 'auto', opacity: 1 }}
                                  exit={{ height: 0, opacity: 0 }}
                                  transition={{ type: 'spring', stiffness: 320, damping: 24, mass: 0.9 }}
                                  className="overflow-hidden"
                                >
                                  <div className="p-3 space-y-2">
                                    {list.map(renderModelRow)}
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        )
                      })}

                      {(!modelsByType || Object.keys(modelsByType).length === 0) && (
                        <div className={`${RADIUS_CLASS} border border-dashed border-gray-200 p-6 text-center text-sm text-gray-500`}>
                          暂无模型，请点击“同步模型”或手动新增。
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    )
  } catch (err) {
    console.error('EmbeddingSettings render error', err)
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm p-4">
        <div className={`bg-white ${RADIUS_CLASS} shadow-2xl p-6 max-w-lg w-full`}>
          <div className="text-lg font-semibold text-gray-900 mb-2">模型服务管理加载失败</div>
          <div className="text-sm text-gray-600 mb-4">{err?.message || '未知错误'}</div>
          <button
            onClick={onClose}
            className="soft-button soft-button-primary px-4 py-2 rounded-lg"
          >
            关闭
          </button>
        </div>
      </div>
    )
  }
}
