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
  const [addModelForm, setAddModelForm] = useState({
    id: '',
    name: '',
    type: 'chat'
  })

  // 默认模型键名映射
  const DEFAULT_TYPE_MAP = {
    embedding: 'embeddingModel',
    rerank: 'rerankModel',
    chat: 'assistantModel'
  }

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

  if (!isOpen) return null

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

  const renderModelRow = (model) => (
    <div
      key={`${model.providerId}-${model.id}`}
      className="flex items-center justify-between px-3 py-2 rounded-lg hover:bg-gray-50 border border-gray-100"
    >
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center text-sm text-blue-600 font-semibold">
          {model.name?.[0]?.toUpperCase() || 'M'}
        </div>
        <div>
          <div className="text-sm font-semibold text-gray-800">{model.name || model.id}</div>
          <div className="text-xs text-gray-500">{model.id}</div>
        </div>
        {model.metadata?.dimension && (
          <span className="text-[11px] text-amber-700 bg-amber-50 px-2 py-0.5 rounded-full border border-amber-100">
            {model.metadata.dimension} 维
          </span>
        )}
        {model.type === 'chat' && (
          <span className="text-[11px] text-blue-700 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-100">
            Chat
          </span>
        )}
        {model.type === 'embedding' && (
          <span className="text-[11px] text-purple-700 bg-purple-50 px-2 py-0.5 rounded-full border border-purple-100">
            Embedding
          </span>
        )}
        {model.type === 'rerank' && (
          <span className="text-[11px] text-green-700 bg-green-50 px-2 py-0.5 rounded-full border border-green-100">
            Rerank
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={() => handleSetDefault(model.type, model.id)}
          className={`px-2 py-1 rounded-lg text-xs border ${isDefaultModel(model.type, model.id)
            ? 'border-blue-200 text-blue-700 bg-blue-50'
            : 'border-gray-200 text-gray-600 hover:border-blue-200 hover:text-blue-700'
            }`}
        >
          {isDefaultModel(model.type, model.id) ? '当前默认' : '设为默认'}
        </button>
        {model.isUserAdded && (
          <button
            onClick={() => removeModelFromCollection(model.id, model.providerId)}
            className="p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50"
            title="删除模型"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  )

  try {
    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/25 backdrop-blur-sm p-4"
        >
          <motion.div
            initial={{ scale: 0.96, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            className="w-full max-w-6xl max-h-[92vh] bg-white rounded-2xl shadow-2xl overflow-hidden flex flex-col"
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
                    return (
                      <div key={type} className="border border-gray-100 rounded-xl">
                        <div className="px-3 py-2 flex items-center justify-between bg-gray-50 border-b border-gray-100 rounded-t-xl">
                          <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                            <span className="w-2.5 h-2.5 rounded-full bg-blue-400" />
                            {type === 'chat' ? 'Chat 对话' : type === 'embedding' ? 'Embedding 向量' : type === 'rerank' ? 'Rerank 重排' : 'Image 图像'}
                          </div>
                          <div className="text-[11px] text-gray-500 flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" />
                            默认：{(() => {
                              const key = DEFAULT_TYPE_MAP[type]
                              if (!key) return '—'
                              const current = getDefaultModel(key)
                              return current || '未选择'
                            })()}
                          </div>
                        </div>
                        <div className="p-3 space-y-2">
                          {list.map(renderModelRow)}
                        </div>
                      </div>
                    )
                  })}

                  {(!modelsByType || Object.keys(modelsByType).length === 0) && (
                    <div className="rounded-xl border border-dashed border-gray-200 p-6 text-center text-sm text-gray-500">
                      暂无模型，请点击“同步模型”或手动新增。
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
    )
  } catch (err) {
    console.error('EmbeddingSettings render error', err)
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm p-4">
        <div className="bg-white rounded-2xl shadow-2xl p-6 max-w-lg w-full">
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
