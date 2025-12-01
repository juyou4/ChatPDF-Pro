import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Search, Download, Plus, Trash2, Zap, DollarSign, Check } from 'lucide-react'
import { useProvider } from '../contexts/ProviderContext'
import { useModel } from '../contexts/ModelContext'
import ProviderAvatar from './ProviderAvatar'

// 价格映射（前端兜底，防止接口未返回定价）
const PRICING_OVERRIDES = [
    { match: /qwen3-embedding-8b/i, price: 0.28, currency: 'CNY' },
    { match: /qwen3-embedding-4b/i, price: 0.14, currency: 'CNY' },
    { match: /qwen3-embedding-0\.6b/i, price: 0.07, currency: 'CNY' },
    { match: /qwen-embedding-8b/i, price: 0.28, currency: 'CNY' },
    { match: /qwen-embedding-4b/i, price: 0.14, currency: 'CNY' },
    { match: /qwen-embedding-0\.6b/i, price: 0.07, currency: 'CNY' },
    { match: /pro\/baai\/bge-m3/i, price: 0.07, currency: 'CNY' }
]

// 简单格式化价格显示
const getPricingLabel = (model) => {
    const pricing = model.pricing || model.metadata?.pricing
    let priceInfo = pricing

    if (!priceInfo) {
        const nameDesc = `${model.name || ''} ${model.metadata?.description || ''}`.toLowerCase()
        const looksFree = nameDesc.includes('免费') || nameDesc.includes('free')

        // 仅对硅基流动的常见付费模型做兜底，且不覆盖“免费”模型
        if (model.providerId === 'silicon' && !looksFree) {
            const override = PRICING_OVERRIDES.find(item => item.match.test(model.id))
            if (override) {
                priceInfo = {
                    perMillionTokens: override.price,
                    currency: override.currency
                }
            }
        }
    }

    if (!priceInfo) return null

    // 兼容旧字段 perMillionTokens
    const perMillion = priceInfo.perMillionTokens ?? priceInfo.input ?? priceInfo.output
    const currency = priceInfo.currency || 'USD'
    if (perMillion !== undefined && perMillion !== 0) {
        return `${currency} ${perMillion}/M`
    }

    // 显式 0 也视为免费
    if (perMillion === 0) {
        return null
    }

    return null
}

export default function ManageModelsPopup({ isOpen, onClose, providerId }) {
    const { getProviderById } = useProvider()
    const {
        addModelToCollection,
        removeModelFromCollection,
        fetchAndAddModels,
        isFetching,
        systemModels,
        userCollection,
        isModelInCollection
    } = useModel()

    const [activeTab, setActiveTab] = useState('available')
    const [selectedModel, setSelectedModel] = useState(null)
    const [searchQuery, setSearchQuery] = useState('')
    const [filterType, setFilterType] = useState('all') // 'all' | 'embedding' | 'rerank'
    const [fetchedModels, setFetchedModels] = useState([])
    const [showAddModal, setShowAddModal] = useState(false)
    const [newModelForm, setNewModelForm] = useState({
        id: '',
        name: '',
        type: 'embedding',
        dimension: ''
    })

    const provider = getProviderById(providerId)
    const providerSystemModels = systemModels.filter(m => m.providerId === providerId)
    const providerUserModels = userCollection.filter(m => m.providerId === providerId)

    // 合并系统模型、已获取但未添加的模型，以及用户已添加的模型，用于“可用模型”列表
    const availableModels = React.useMemo(() => {
        const mergeMap = new Map()
        const pushModels = (models) => {
            models.forEach(m => {
                mergeMap.set(m.id, m)
            })
        }
        pushModels(providerSystemModels)
        pushModels(fetchedModels)
        pushModels(providerUserModels)
        return Array.from(mergeMap.values())
    }, [providerSystemModels, fetchedModels, providerUserModels])

    // “我的模型”仅展示用户手动添加的模型，避免系统预设模型误导删除操作
    const myModels = providerUserModels

    // Get available models for this provider
    useEffect(() => {
        if (provider) {
            setSelectedModel(null)
        }
    }, [provider, providerId])

    // Filter models based on search and type
    const filterModels = (models) => {
        return models.filter(model => {
            const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                model.id.toLowerCase().includes(searchQuery.toLowerCase())
            const matchesType = filterType === 'all' || model.type === filterType
            return matchesSearch && matchesType
        })
    }

    const filteredAvailableModels = filterModels(availableModels)
    const filteredMyModels = filterModels(myModels)

    // Check if a model is already added
    const isModelAdded = (modelId) => {
        return isModelInCollection(modelId, providerId)
    }

    // Handle fetch models from API
    const handleFetchModels = async () => {
        if (!provider) return
        const models = await fetchAndAddModels(provider, { autoAdd: false })
        setFetchedModels(models || [])
    }

    // Statistics
    const embeddingCount = myModels.filter(m => m.type === 'embedding').length
    const rerankCount = myModels.filter(m => m.type === 'rerank').length

    const handleSubmitCustomModel = () => {
        if (!newModelForm.id.trim()) return
        const model = {
            id: newModelForm.id.trim(),
            name: newModelForm.name.trim() || newModelForm.id.trim(),
            providerId,
            type: newModelForm.type,
            metadata: {
                dimension: newModelForm.dimension ? Number(newModelForm.dimension) : undefined
            },
            isSystem: false,
            isUserAdded: true
        }
        addModelToCollection(model)
        setShowAddModal(false)
        setNewModelForm({
            id: '',
            name: '',
            type: 'embedding',
            dimension: ''
        })
    }

    if (!provider) return null

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
                    onClick={onClose}
                >
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        transition={{ duration: 0.2 }}
                        className="soft-panel rounded-3xl shadow-2xl w-full max-w-5xl max-h-[85vh] flex flex-col m-4"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between p-6 border-b border-white/20">
                            <div className="flex items-center gap-3">
                                <ProviderAvatar provider={provider} size={32} />
                                <div>
                                    <h2 className="text-xl font-bold text-gray-900">
                                        管理模型
                                    </h2>
                                    <p className="text-sm text-gray-600 mt-0.5">
                                        {provider.name}
                                    </p>
                                </div>
                            </div>
                            <motion.button
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.9 }}
                                onClick={onClose}
                                className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                            >
                                <X className="w-5 h-5 text-gray-600" />
                            </motion.button>
                        </div>

                        {/* Tabs */}
                        <div className="flex gap-1 px-6 pt-4">
                            <button
                                onClick={() => setActiveTab('available')}
                                className={`
                                    px-4 py-2 rounded-xl font-semibold text-sm transition-all
                                    ${activeTab === 'available'
                                        ? 'bg-blue-500 text-white shadow-md'
                                        : 'text-gray-600 hover:bg-gray-100'
                                    }
                                `}
                            >
                                可用模型
                            </button>
                            <button
                                onClick={() => setActiveTab('my-models')}
                                className={`
                                    px-4 py-2 rounded-xl font-semibold text-sm transition-all
                                    ${activeTab === 'my-models'
                                        ? 'bg-blue-500 text-white shadow-md'
                                        : 'text-gray-600 hover:bg-gray-100'
                                    }
                                `}
                            >
                                我的模型
                                {myModels.length > 0 && (
                                    <span className="ml-2 px-2 py-0.5 bg-white/20 rounded-full text-xs">
                                        {myModels.length}
                                    </span>
                                )}
                            </button>
                        </div>

                        {/* Search and Filter Bar */}
                        <div className="px-6 pt-4 pb-2">
                            <div className="flex gap-3">
                                {/* Search */}
                                <div className="flex-1 relative">
                                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                    <input
                                        type="text"
                                        placeholder="搜索模型名称..."
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        className="w-full pl-10 pr-4 py-2 soft-input rounded-xl text-sm"
                                    />
                                </div>

                                {/* Type Filter */}
                                <select
                                    value={filterType}
                                    onChange={(e) => setFilterType(e.target.value)}
                                    className="soft-input rounded-xl px-4 py-2 text-sm font-semibold text-gray-700"
                                >
                                    <option value="all">全部类型</option>
                                    <option value="embedding">Embedding</option>
                                    <option value="rerank">Rerank</option>
                                </select>

                                {/* Fetch Models Button */}
                                {activeTab === 'available' && (
                                    <div className="flex items-center gap-2">
                                        <motion.button
                                            whileHover={{ scale: 1.02 }}
                                            whileTap={{ scale: 0.98 }}
                                            onClick={handleFetchModels}
                                            disabled={isFetching}
                                            className="px-4 py-2 bg-blue-500 text-white rounded-xl font-semibold text-sm shadow-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                                        >
                                            <Download className="w-4 h-4" />
                                            {isFetching ? '获取中...' : '获取模型'}
                                        </motion.button>
                                        <motion.button
                                            whileHover={{ scale: 1.02 }}
                                            whileTap={{ scale: 0.98 }}
                                            onClick={() => setShowAddModal(true)}
                                            className="px-4 py-2 bg-green-500 text-white rounded-xl font-semibold text-sm shadow-md hover:bg-green-600 flex items-center gap-2"
                                        >
                                            <Plus className="w-4 h-4" />
                                            手动添加
                                        </motion.button>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Main Content - Two Panel Layout */}
                        <div className="flex-1 overflow-hidden px-6 pb-6 pt-2">
                            <div className="flex gap-4 h-full">
                                {/* Left Panel - Model List (60%) */}
                                <div className="flex-[3] flex flex-col">
                                    <div className="soft-card rounded-2xl overflow-hidden flex flex-col h-full">
                                        <div className="overflow-y-auto flex-1">
                                            {activeTab === 'available' ? (
                                                <ModelList
                                                    models={filteredAvailableModels}
                                                    selectedModel={selectedModel}
                                                    onSelectModel={setSelectedModel}
                                                    isAdded={isModelAdded}
                                                    onAddModel={addModelToCollection}
                                                    providerId={providerId}
                                                    emptyMessage="暂无可用模型"
                                                    emptyHint='点击"获取模型"从API获取模型列表'
                                                />
                                            ) : (
                                                <ModelList
                                                    models={filteredMyModels}
                                                    selectedModel={selectedModel}
                                                    onSelectModel={setSelectedModel}
                                                    isAdded={() => true}
                                                    onRemoveModel={removeModelFromCollection}
                                                    providerId={providerId}
                                                    emptyMessage="暂无已添加的模型"
                                                    emptyHint='前往"可用模型"标签添加模型'
                                                />
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Right Panel - Model Details (40%) */}
                                <div className="flex-[2] flex flex-col">
                                    <div className="soft-card rounded-2xl p-6 h-full overflow-y-auto">
                                        {selectedModel ? (
                                            <ModelDetails
                                                model={selectedModel}
                                                provider={provider}
                                                isAdded={isModelAdded(selectedModel.id)}
                                                onAddModel={() => addModelToCollection(selectedModel, providerId)}
                                                onRemoveModel={() => removeModelFromCollection(selectedModel.id, providerId)}
                                            />
                                        ) : (
                                            <div className="flex items-center justify-center h-full text-center">
                                                <div>
                                                    <div className="text-6xl mb-4">👈</div>
                                                    <p className="text-gray-500 font-semibold">选择一个模型</p>
                                                    <p className="text-sm text-gray-400 mt-1">查看详细信息</p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Footer */}
                        <div className="px-6 py-4 border-t border-white/20">
                            <div className="flex items-center justify-between text-sm">
                                <div className="text-gray-600">
                                    <span className="font-semibold text-gray-900">{embeddingCount}</span> 个Embedding模型 ·
                                    <span className="font-semibold text-gray-900 ml-1">{rerankCount}</span> 个Rerank模型
                                </div>
                                <div className="text-gray-500">
                                    共 <span className="font-semibold text-gray-900">{myModels.length}</span> 个模型
                                </div>
                            </div>
                        </div>

                        {/* Add custom model modal */}
                        <AnimatePresence>
                            {showAddModal && (
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="absolute inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50"
                                    onClick={() => setShowAddModal(false)}
                                >
                                    <motion.div
                                        initial={{ scale: 0.95, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        exit={{ scale: 0.95, opacity: 0 }}
                                        onClick={(e) => e.stopPropagation()}
                                        className="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-md space-y-4"
                                    >
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <h3 className="text-lg font-bold text-gray-900">添加自定义模型</h3>
                                                <p className="text-sm text-gray-500 mt-1">手动录入模型ID和类型</p>
                                            </div>
                                            <button onClick={() => setShowAddModal(false)} className="p-2 hover:bg-gray-100 rounded-lg">
                                                <X className="w-4 h-4 text-gray-600" />
                                            </button>
                                        </div>
                                        <div className="space-y-3">
                                            <div>
                                                <label className="text-xs text-gray-600 font-semibold">模型ID *</label>
                                                <input
                                                    className="soft-input w-full px-3 py-2 rounded-lg mt-1"
                                                    placeholder="例如 text-embedding-3-small"
                                                    value={newModelForm.id}
                                                    onChange={(e) => setNewModelForm({ ...newModelForm, id: e.target.value })}
                                                />
                                            </div>
                                            <div>
                                                <label className="text-xs text-gray-600 font-semibold">模型名称</label>
                                                <input
                                                    className="soft-input w-full px-3 py-2 rounded-lg mt-1"
                                                    placeholder="显示名称（可选）"
                                                    value={newModelForm.name}
                                                    onChange={(e) => setNewModelForm({ ...newModelForm, name: e.target.value })}
                                                />
                                            </div>
                                            <div className="grid grid-cols-2 gap-3">
                                                <div>
                                                    <label className="text-xs text-gray-600 font-semibold">类型</label>
                                                    <select
                                                        className="soft-input w-full px-3 py-2 rounded-lg mt-1"
                                                        value={newModelForm.type}
                                                        onChange={(e) => setNewModelForm({ ...newModelForm, type: e.target.value })}
                                                    >
                                                        <option value="embedding">Embedding</option>
                                                        <option value="rerank">Rerank</option>
                                                    </select>
                                                </div>
                                                <div>
                                                    <label className="text-xs text-gray-600 font-semibold">维度（可选）</label>
                                                    <input
                                                        className="soft-input w-full px-3 py-2 rounded-lg mt-1"
                                                        placeholder="如 1024"
                                                        value={newModelForm.dimension}
                                                        onChange={(e) => setNewModelForm({ ...newModelForm, dimension: e.target.value })}
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex justify-end gap-2 pt-2">
                                            <button
                                                onClick={() => setShowAddModal(false)}
                                                className="px-4 py-2 rounded-lg text-sm font-semibold text-gray-600 hover:bg-gray-100"
                                            >
                                                取消
                                            </button>
                                            <button
                                                onClick={handleSubmitCustomModel}
                                                className="px-4 py-2 rounded-lg text-sm font-semibold text-white bg-green-500 hover:bg-green-600 disabled:opacity-50"
                                                disabled={!newModelForm.id.trim()}
                                            >
                                                添加模型
                                            </button>
                                        </div>
                                    </motion.div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    )
}

// Model List Component
function ModelList({ models, selectedModel, onSelectModel, isAdded, onAddModel, onRemoveModel, providerId, emptyMessage, emptyHint }) {
    if (models.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-center p-8">
                <div>
                    <div className="text-6xl mb-4">📦</div>
                    <p className="text-gray-500 font-semibold">{emptyMessage}</p>
                    <p className="text-sm text-gray-400 mt-1">{emptyHint}</p>
                </div>
            </div>
        )
    }

    // Group models by type
    const embeddingModels = models.filter(m => m.type === 'embedding')
    const rerankModels = models.filter(m => m.type === 'rerank')

    return (
        <div className="divide-y divide-white/30">
            {embeddingModels.length > 0 && (
                <div className="p-4">
                    <div className="text-xs font-bold text-gray-500 mb-3 px-2">
                        EMBEDDING 模型 ({embeddingModels.length})
                    </div>
                    <div className="space-y-2">
                        {embeddingModels.map(model => (
                            <ModelCard
                                key={model.id}
                                model={model}
                                isSelected={selectedModel?.id === model.id}
                                isAdded={isAdded(model.id)}
                                onSelect={onSelectModel}
                                onAdd={onAddModel ? () => onAddModel(model, providerId) : null}
                                onRemove={onRemoveModel ? () => onRemoveModel(model.id, providerId) : null}
                            />
                        ))}
                    </div>
                </div>
            )}

            {rerankModels.length > 0 && (
                <div className="p-4">
                    <div className="text-xs font-bold text-gray-500 mb-3 px-2">
                        RERANK 模型 ({rerankModels.length})
                    </div>
                    <div className="space-y-2">
                        {rerankModels.map(model => (
                            <ModelCard
                                key={model.id}
                                model={model}
                                isSelected={selectedModel?.id === model.id}
                                isAdded={isAdded(model.id)}
                                onSelect={onSelectModel}
                                onAdd={onAddModel ? () => onAddModel(model, providerId) : null}
                                onRemove={onRemoveModel ? () => onRemoveModel(model.id, providerId) : null}
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

// Model Card Component
function ModelCard({ model, isSelected, isAdded, onSelect, onAdd, onRemove }) {
    return (
        <motion.div
            whileHover={{ scale: 1.01 }}
            className={`
                p-3 rounded-xl cursor-pointer transition-all
                ${isSelected ? 'bg-blue-50/80 ring-2 ring-blue-500/50' : 'bg-white/40 hover:bg-white/60'}
            `}
            onClick={() => onSelect(model)}
        >
            <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                    <div className="font-semibold text-sm text-gray-900 truncate">
                        {model.name}
                    </div>
                    <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                        <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-semibold">
                            {model.type === 'embedding' ? 'Embedding' : 'Rerank'}
                        </span>
                        {model.isSystem && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs font-semibold">
                                内置
                            </span>
                        )}
                        {model.isUserAdded && !model.isSystem && (
                            <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded text-xs font-semibold">
                                自定义
                            </span>
                        )}
                        {model.metadata?.dimension && (
                            <span className="px-2 py-0.5 soft-panel rounded text-xs text-gray-600">
                                📐 {model.metadata.dimension}维
                            </span>
                        )}
                        {(() => {
                            const priceText = getPricingLabel(model)
                            if (priceText) {
                                return (
                                    <span className="px-2 py-0.5 bg-amber-500/20 text-amber-700 rounded text-xs font-semibold">
                                        {priceText}
                                    </span>
                                )
                            }
                            return (
                                <span className="px-2 py-0.5 bg-green-500/20 text-green-700 rounded text-xs font-semibold">
                                    免费
                                </span>
                            )
                        })()}
                    </div>
                </div>

                {/* Action Button */}
                <div onClick={(e) => e.stopPropagation()}>
                    {isAdded ? (
                        onRemove && (
                            <motion.button
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.9 }}
                                onClick={onRemove}
                                className="p-1.5 hover:bg-red-50 rounded-lg transition-colors"
                                title="移除模型"
                            >
                                <Trash2 className="w-4 h-4 text-red-500" />
                            </motion.button>
                        )
                    ) : (
                        onAdd && (
                            <motion.button
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.9 }}
                                onClick={onAdd}
                                className="p-1.5 hover:bg-green-50 rounded-lg transition-colors"
                                title="添加模型"
                            >
                                <Plus className="w-4 h-4 text-green-500" />
                            </motion.button>
                        )
                    )}
                </div>
            </div>
        </motion.div>
    )
}

// Model Details Component
function ModelDetails({ model, provider, isAdded, onAddModel, onRemoveModel }) {
    const [testingModel, setTestingModel] = useState(false)
    const [modelTestResult, setModelTestResult] = useState(null)

    const handleTestModel = async () => {
        setTestingModel(true)
        setModelTestResult(null)

        try {
            const response = await fetch('/api/models/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    providerId: provider.id,
                    modelId: model.id,
                    apiKey: provider.apiKey || '',
                    apiHost: provider.apiHost || '',
                    modelType: model.type,
                    embeddingEndpoint: provider.apiConfig?.embeddingEndpoint,
                    rerankEndpoint: provider.apiConfig?.rerankEndpoint
                })
            })

            const result = await response.json()
            setModelTestResult(result)
        } catch (error) {
            setModelTestResult({
                success: false,
                message: error.message || '网络错误，请检查后端服务是否运行'
            })
        } finally {
            setTestingModel(false)
        }
    }

    return (
        <div className="space-y-6">
            {/* Model Name */}
            <div>
                <h3 className="text-lg font-bold text-gray-900 mb-1">
                    {model.name}
                </h3>
                <p className="text-sm text-gray-500">{model.id}</p>
            </div>

            {/* Type Badge */}
            <div>
                <div className="inline-block px-3 py-1 bg-blue-100 text-blue-700 rounded-lg text-sm font-semibold">
                    {model.type === 'embedding' ? 'Embedding 模型' : 'Rerank 模型'}
                </div>
            </div>

            {/* Metadata */}
            {model.metadata && (
                <div className="space-y-3">
                    {model.metadata.dimension && (
                        <div className="soft-panel rounded-xl p-3">
                            <div className="text-xs text-gray-500 mb-1">向量维度</div>
                            <div className="text-lg font-bold text-gray-900">
                                {model.metadata.dimension} 维
                            </div>
                        </div>
                    )}

                    {(() => {
                        const priceText = getPricingLabel(model)
                        if (priceText) {
                            return (
                                <div className="soft-panel rounded-xl p-3">
                                    <div className="text-xs text-gray-500 mb-1">价格</div>
                                    <div className="text-lg font-bold text-amber-700">
                                        {priceText}
                                    </div>
                                </div>
                            )
                        }
                        return (
                            <div className="soft-panel rounded-xl p-3">
                                <div className="flex items-center gap-2">
                                    <Zap className="w-5 h-5 text-green-600" />
                                    <div className="text-lg font-bold text-green-700">免费模型</div>
                                </div>
                            </div>
                        )
                    })()}

                    {model.metadata.description && (
                        <div className="soft-panel rounded-xl p-3">
                            <div className="text-xs text-gray-500 mb-1">描述</div>
                            <div className="text-sm text-gray-700">
                                {model.metadata.description}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Test Result */}
            {modelTestResult && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`
                        soft-panel rounded-xl p-4 border-2
                        ${modelTestResult.success
                            ? 'border-green-500/50 bg-green-50/50'
                            : 'border-red-500/50 bg-red-50/50'}
                    `}
                >
                    <div className="flex items-start gap-3">
                        <div className={`
                            w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0
                            ${modelTestResult.success ? 'bg-green-500' : 'bg-red-500'}
                        `}>
                            {modelTestResult.success ? (
                                <Check className="w-5 h-5 text-white" />
                            ) : (
                                <X className="w-5 h-5 text-white" />
                            )}
                        </div>
                        <div className="flex-1">
                            <div className={`font-semibold mb-1 ${
                                modelTestResult.success ? 'text-green-900' : 'text-red-900'
                            }`}>
                                {modelTestResult.message}
                            </div>
                            {modelTestResult.success && (
                                <div className="space-y-1 text-sm">
                                    {modelTestResult.embeddingDimension && (
                                        <div className="text-green-700">
                                            📐 维度: <span className="font-bold">{modelTestResult.embeddingDimension}</span>
                                        </div>
                                    )}
                                    {modelTestResult.responseTime && (
                                        <div className="text-green-700">
                                            ⚡ 响应时间: <span className="font-bold">{modelTestResult.responseTime}ms</span>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </motion.div>
            )}

            {/* Actions */}
            <div className="pt-4 space-y-2">
                {isAdded ? (
                    <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={onRemoveModel}
                        className="w-full px-4 py-3 bg-red-500 text-white rounded-xl font-semibold shadow-md hover:bg-red-600 flex items-center justify-center gap-2"
                    >
                        <Trash2 className="w-4 h-4" />
                        从集合中移除
                    </motion.button>
                ) : (
                    <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={onAddModel}
                        className="w-full px-4 py-3 bg-green-500 text-white rounded-xl font-semibold shadow-md hover:bg-green-600 flex items-center justify-center gap-2"
                    >
                        <Plus className="w-4 h-4" />
                        添加到集合
                    </motion.button>
                )}

                {/* Test Model Button */}
                <motion.button
                    whileHover={{ scale: testingModel ? 1 : 1.02 }}
                    whileTap={{ scale: testingModel ? 1 : 0.98 }}
                    onClick={handleTestModel}
                    disabled={testingModel}
                    className={`
                        w-full px-4 py-3 rounded-xl font-semibold shadow-md flex items-center justify-center gap-2
                        ${testingModel
                            ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
                            : 'bg-blue-500 text-white hover:bg-blue-600'}
                    `}
                >
                    {testingModel ? (
                        <>
                            <div className="w-4 h-4 border-2 border-gray-600 border-t-transparent rounded-full animate-spin" />
                            测试中...
                        </>
                    ) : (
                        <>
                            🧪 测试此模型
                        </>
                    )}
                </motion.button>
            </div>
        </div>
    )
}
