import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Settings, X, ChevronRight, Check, Key, Server,
    Zap, DollarSign, Search, AlertCircle, CheckCircle,
    Database, TrendingUp, Layers
} from 'lucide-react'
import { useProvider } from '../contexts/ProviderContext'
import { useModel } from '../contexts/ModelContext'
import { useDefaults } from '../contexts/DefaultsContext'
import ProviderAvatar from './ProviderAvatar'
import ManageModelsPopup from './ManageModelsPopup'

export default function EmbeddingSettings({ isOpen, onClose }) {
    const {
        providers,
        updateProvider,
        getProviderById
    } = useProvider()

    const {
        getModelsByProvider,
        removeModelFromCollection,
        fetchAndAddModels,
        isFetching,
        fetchError
    } = useModel()

    const {
        defaults,
        setDefaultModel,
        getDefaultModel
    } = useDefaults()

    const [activeTab, setActiveTab] = useState('providers') // 'providers' | 'defaults'
    const [activeProviderId, setActiveProviderId] = useState(
        providers.length > 0 ? providers[0].id : null
    )
    const [searchQuery, setSearchQuery] = useState('')
    const [testingProvider, setTestingProvider] = useState(null)
    const [testResult, setTestResult] = useState(null)
    const [manageModelsOpen, setManageModelsOpen] = useState(false)

    const activeProvider = providers.find(p => p.id === activeProviderId)

    // Tab配置
    const tabs = [
        { id: 'providers', label: '服务商配置', icon: Server },
        { id: 'defaults', label: '默认模型', icon: CheckCircle }
    ]

    // 切换Provider时清除测试结果
    React.useEffect(() => {
        setTestResult(null)
    }, [activeProviderId])

    // 确保在providers加载后设置activeProviderId
    React.useEffect(() => {
        if (!activeProviderId && providers.length > 0) {
            setActiveProviderId(providers[0].id)
        }
    }, [providers, activeProviderId])

    // 测试Provider连接
    const handleTestConnection = async (provider) => {
        setTestingProvider(provider.id)
        setTestResult(null)

        try {
            const response = await fetch('/api/providers/test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    providerId: provider.id,
                    apiKey: provider.apiKey,
                    apiHost: provider.apiHost
                })
            })

            // 添加：检查HTTP状态
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`)
            }

            const result = await response.json()

            // 修改：添加 providerId 字段
            setTestResult({
                ...result,
                providerId: provider.id
            })
        } catch (error) {
            setTestResult({
                success: false,
                message: error.message || '网络错误，请检查后端服务是否运行',
                providerId: provider.id
            })
        } finally {
            setTestingProvider(null)
        }
    }


    // 过滤服务商
    const filteredProviders = providers.filter(p =>
        p.name.toLowerCase().includes(searchQuery.toLowerCase())
    )

    if (!isOpen) return null

    return (
        <>
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/20 backdrop-blur-sm"
                onClick={onClose}
            >
                <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.95, opacity: 0 }}
                    transition={{ type: "spring", duration: 0.3 }}
                    onClick={(e) => e.stopPropagation()}
                    className="soft-panel rounded-[32px] w-full max-w-5xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden"
                >
                    {/* Header */}
                    <div className="flex items-center justify-between px-8 py-6 border-b border-white/20">
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                                <Settings className="w-6 h-6 text-white" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-gray-900">嵌入模型管理</h2>
                                <p className="text-sm text-gray-500 mt-0.5">管理向量检索和重排模型服务</p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-black/5 rounded-xl transition-colors"
                        >
                            <X className="w-6 h-6 text-gray-600" />
                        </button>
                    </div>

                    {/* Tab Navigation */}
                    <div className="border-b border-white/20 bg-white/30">
                        <div className="flex px-8 gap-2">
                            {tabs.map(tab => {
                                const Icon = tab.icon
                                const isActive = activeTab === tab.id

                                return (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`
                                            relative px-6 py-4 flex items-center gap-2 font-semibold transition-all
                                            ${isActive
                                                ? 'text-blue-600'
                                                : 'text-gray-600 hover:text-gray-900'
                                            }
                                        `}
                                    >
                                        <Icon className="w-5 h-5" />
                                        {tab.label}

                                        {isActive && (
                                            <motion.div
                                                layoutId="activeTab"
                                                className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-blue-600 rounded-t-full"
                                                transition={{ type: "spring", duration: 0.5 }}
                                            />
                                        )}
                                    </button>
                                )
                            })}
                        </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1 flex overflow-hidden">
                        {/* Tab 1: 服务商配置 */}
                        {activeTab === 'providers' && (
                            <>
                                {/* Left Sidebar - Provider List */}
                                <div className="w-80 border-r border-white/20 flex flex-col">
                                    {/* Search Box */}
                                    <div className="p-4 border-b border-white/20">
                                        <div className="relative">
                                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                            <input
                                                type="text"
                                                value={searchQuery}
                                                onChange={(e) => setSearchQuery(e.target.value)}
                                                placeholder="搜索服务商..."
                                                className="soft-input w-full pl-10 pr-4 py-2.5 rounded-xl text-sm"
                                            />
                                        </div>
                                    </div>

                                    {/* Provider List */}
                                    <div className="flex-1 overflow-y-auto p-3 space-y-2">
                                        {filteredProviders.map(provider => (
                                            <motion.div
                                                key={provider.id}
                                                whileHover={{ scale: 1.02 }}
                                                whileTap={{ scale: 0.98 }}
                                                onClick={() => setActiveProviderId(provider.id)}
                                                className={`
                      relative p-4 rounded-2xl cursor-pointer transition-all duration-200
                      ${activeProviderId === provider.id
                                                        ? 'soft-card shadow-lg'
                                                        : 'hover:bg-white/30 bg-white/10'}
                    `}
                                            >
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        <ProviderAvatar provider={provider} size={32} />
                                                        <div>
                                                            <div className="font-semibold text-gray-900 text-sm">
                                                                {provider.name}
                                                            </div>
                                                            <div className="text-xs text-gray-500 mt-0.5">
                                                                {getModelsByProvider(provider.id).length} 个模型
                                                            </div>
                                                        </div>
                                                    </div>
                                                    {provider.enabled && (
                                                        <div className="px-2.5 py-1 bg-green-500/20 text-green-700 text-xs font-bold rounded-lg backdrop-blur-sm">
                                                            ON
                                                        </div>
                                                    )}
                                                </div>
                                                {activeProviderId === provider.id && (
                                                    <motion.div
                                                        layoutId="activeIndicator"
                                                        className="absolute inset-0 border-2 border-blue-500 rounded-2xl pointer-events-none"
                                                        transition={{ type: "spring", duration: 0.5 }}
                                                    />
                                                )}
                                            </motion.div>
                                        ))}
                                    </div>
                                </div>

                                {/* Right Content - Provider Config */}
                                {activeProvider ? (
                                    <div className="flex-1 overflow-y-auto p-6">
                                        <motion.div
                                            key={activeProvider.id}
                                            initial={{ opacity: 0, x: 20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ duration: 0.3 }}
                                        >
                                            {/* API Configuration */}
                                            <div className="soft-card rounded-2xl p-6 mb-6">
                                                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                                                    <Server className="w-5 h-5 text-blue-600" />
                                                    API 配置
                                                </h3>

                                                {activeProvider.type === 'local' ? (
                                                    <div className="soft-card bg-gradient-to-br from-green-50/50 to-emerald-50/50 rounded-xl p-5 border border-green-200/50">
                                                        <div className="flex items-start gap-3">
                                                            <div className="w-10 h-10 bg-green-500 rounded-xl flex items-center justify-center flex-shrink-0">
                                                                <Zap className="w-5 h-5 text-white" />
                                                            </div>
                                                            <div>
                                                                <div className="font-semibold text-green-900">本地模型 (免费)</div>
                                                                <div className="text-sm text-green-700 mt-1">
                                                                    ✨ 无需 API Key，首次使用会自动下载模型文件
                                                                </div>
                                                                <div className="text-xs text-green-600/80 mt-2">
                                                                    模型会缓存在本地，后续使用无需联网
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <div className="space-y-4">
                                                        <div>
                                                            <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                                                                <Key className="w-4 h-4 text-blue-600" />
                                                                API Key
                                                            </label>
                                                            <input
                                                                type="password"
                                                                value={activeProvider.apiKey || ''}
                                                                onChange={(e) => updateProvider(activeProvider.id, { apiKey: e.target.value })}
                                                                placeholder="sk-..."
                                                                className="soft-input w-full px-4 py-3 rounded-xl text-sm"
                                                            />
                                                        </div>

                                                        <div>
                                                            <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                                                                <Server className="w-4 h-4 text-blue-600" />
                                                                API 地址
                                                            </label>
                                                            <input
                                                                type="text"
                                                                value={activeProvider.apiHost || ''}
                                                                onChange={(e) => updateProvider(activeProvider.id, { apiHost: e.target.value })}
                                                                placeholder="https://..."
                                                                className="soft-input w-full px-4 py-3 rounded-xl text-sm font-mono"
                                                            />
                                                        </div>

                                                        <div className="flex items-center gap-3 pt-2">
                                                            <button
                                                                onClick={() => updateProvider(activeProvider.id, { enabled: !activeProvider.enabled })}
                                                                className={`
                              px-6 py-3 rounded-xl font-semibold transition-all duration-200
                              ${activeProvider.enabled
                                                                        ? 'bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg shadow-green-500/30 hover:shadow-green-500/50 hover:scale-105'
                                                                        : 'soft-card text-gray-700 hover:scale-105'}
                            `}
                                                            >
                                                                {activeProvider.enabled ? (
                                                                    <span className="flex items-center gap-2">
                                                                        <CheckCircle className="w-4 h-4" />
                                                                        已启用
                                                                    </span>
                                                                ) : (
                                                                    '点击启用'
                                                                )}
                                                            </button>

                                                            <button
                                                                onClick={() => handleTestConnection(activeProvider)}
                                                                disabled={testingProvider === activeProvider.id}
                                                                className={`
                                                            soft-card px-6 py-3 rounded-xl font-semibold transition-all duration-200
                                                            ${testingProvider === activeProvider.id
                                                                        ? 'opacity-50 cursor-not-allowed'
                                                                        : 'hover:scale-105'}
                                                        `}
                                                            >
                                                                {testingProvider === activeProvider.id ? (
                                                                    <span className="flex items-center gap-2">
                                                                        <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                                                                        测试中...
                                                                    </span>
                                                                ) : (
                                                                    '测试连接'
                                                                )}
                                                            </button>
                                                        </div>

                                                        {/* Test Result Display */}
                                                        {testResult && testResult.providerId === activeProvider.id && (
                                                            <motion.div
                                                                initial={{ opacity: 0, y: -10 }}
                                                                animate={{ opacity: 1, y: 0 }}
                                                                transition={{ duration: 0.3 }}
                                                                className={`
                                                            mt-4 p-4 rounded-xl border-2 flex items-start gap-3
                                                            ${testResult.success
                                                                        ? 'bg-green-50 border-green-300 text-green-800'
                                                                        : 'bg-red-50 border-red-300 text-red-800'}
                                                        `}
                                                            >
                                                                {testResult.success ? (
                                                                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                                                                ) : (
                                                                    <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                                                                )}
                                                                <div className="flex-1">
                                                                    <p className="font-semibold mb-1">
                                                                        {testResult.success ? '✅ 连接成功' : '❌ 连接失败'}
                                                                    </p>
                                                                    <p className="text-sm opacity-90">
                                                                        {testResult.message}
                                                                    </p>
                                                                    {testResult.success && testResult.availableModels !== undefined && (
                                                                        <p className="text-sm mt-2 opacity-80">
                                                                            可用模型数: <span className="font-semibold">{testResult.availableModels}</span>
                                                                        </p>
                                                                    )}
                                                                </div>
                                                            </motion.div>
                                                        )}
                                                    </div>
                                                )}
                                            </div>

                                            {/* Provider Models Preview */}
                                            <div className="soft-card rounded-2xl p-6 mt-6">
                                                <div className="flex items-center justify-between mb-4">
                                                    <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                                                        <Zap className="w-5 h-5 text-green-600" />
                                                        模型预览
                                                    </h3>
                                                    <button
                                                        onClick={() => setManageModelsOpen(true)}
                                                        className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:scale-105 transition-all flex items-center gap-2 shadow-lg"
                                                    >
                                                        <Zap className="w-4 h-4" />
                                                        管理模型
                                                    </button>
                                                </div>

                                                {/* Display first 5 models for this provider */}
                                                {(() => {
                                                    const providerModels = getModelsByProvider(activeProvider.id)
                                                    const previewModels = providerModels.slice(0, 5)

                                                    if (providerModels.length === 0) {
                                                        return (
                                                            <div className="text-center py-12">
                                                                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                                                    <Zap className="w-8 h-8 text-gray-400" />
                                                                </div>
                                                                <p className="text-gray-600 font-semibold mb-2">暂无模型</p>
                                                                <p className="text-sm text-gray-500">点击「管理模型」添加模型到此服务商</p>
                                                            </div>
                                                        )
                                                    }

                                                    return (
                                                        <div className="space-y-2">
                                                            {previewModels.map(model => (
                                                                <div
                                                                    key={`${model.providerId}-${model.id}`}
                                                                    className="soft-card rounded-xl p-3 flex items-center gap-3"
                                                                >
                                                                    <div className="flex-1">
                                                                        <div className="font-semibold text-gray-900 text-sm">{model.name}</div>
                                                                        <div className="text-xs text-gray-500 mt-0.5">
                                                                            {model.type === 'embedding' ? '📊 Embedding' : '🔄 Rerank'}
                                                                            {model.metadata?.dimension && ` · 维度: ${model.metadata.dimension}`}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            ))}
                                                            {providerModels.length > 5 && (
                                                                <div className="text-center pt-2 text-sm text-gray-500">
                                                                    还有 {providerModels.length - 5} 个模型，点击「管理模型」查看全部
                                                                </div>
                                                            )}
                                                        </div>
                                                    )
                                                })()}
                                            </div>
                                        </motion.div>
                                    </div>
                                ) : (
                                    <div className="flex-1 overflow-y-auto p-6 flex items-center justify-center">
                                        <div className="text-center max-w-md">
                                            <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                                                <Server className="w-10 h-10 text-gray-400" />
                                            </div>
                                            <h3 className="text-xl font-bold text-gray-900 mb-3">暂无服务商</h3>
                                            <p className="text-gray-600 text-sm leading-relaxed">
                                                服务商配置正在加载中...
                                                <br />
                                                如果此问题持续存在，请尝试刷新页面
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}

                        {/* Tab 2: 默认模型 */}
                        {activeTab === 'defaults' && (
                            <div className="flex-1 overflow-y-auto p-6">
                                {(() => {
                                    const { getModelsByType } = useModel()
                                    const embeddingModels = getModelsByType('embedding')
                                    const rerankModels = getModelsByType('rerank')

                                    // 获取当前默认模型的完整键（providerId:modelId）
                                    const currentEmbeddingKey = getDefaultModel('embeddingModel')
                                    const currentRerankKey = getDefaultModel('rerankModel')

                                    // 从复合键解析出当前选中的模型
                                    const parseModelKey = (key) => {
                                        if (!key) return null
                                        const [providerId, modelId] = key.split(':')
                                        return embeddingModels.concat(rerankModels).find(
                                            m => m.providerId === providerId && m.id === modelId
                                        )
                                    }

                                    const currentEmbeddingModel = parseModelKey(currentEmbeddingKey)
                                    const currentRerankModel = parseModelKey(currentRerankKey)

                                    return (
                                        <div className="max-w-3xl mx-auto space-y-6">
                                            {/* Header */}
                                            <div className="text-center mb-8">
                                                <CheckCircle className="w-12 h-12 mx-auto text-green-600 mb-3" />
                                                <h3 className="text-2xl font-bold text-gray-900 mb-2">默认模型配置</h3>
                                                <p className="text-gray-600 text-sm">
                                                    选择您偏好的默认模型，用于文档处理和问答
                                                </p>
                                            </div>

                                            {/* Embedding 默认模型 */}
                                            <div className="soft-card rounded-2xl p-6 space-y-4">
                                                <div className="flex items-center gap-3 mb-4">
                                                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
                                                        <Zap className="w-5 h-5 text-white" />
                                                    </div>
                                                    <h4 className="text-lg font-bold text-gray-900">Embedding 模型</h4>
                                                    <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs font-semibold rounded-lg">
                                                        必选
                                                    </span>
                                                </div>

                                                {embeddingModels.length > 0 ? (
                                                    <>
                                                        <div>
                                                            <label className="text-sm font-semibold text-gray-700 mb-2 block">
                                                                选择默认 Embedding 模型
                                                            </label>
                                                            <select
                                                                value={currentEmbeddingKey || ''}
                                                                onChange={(e) => {
                                                                    const value = e.target.value
                                                                    setDefaultModel('embeddingModel', value || null)
                                                                }}
                                                                className="soft-input w-full px-4 py-3 rounded-xl text-sm font-medium"
                                                            >
                                                                <option value="">请选择...</option>
                                                                {embeddingModels.map(model => {
                                                                    const provider = providers.find(p => p.id === model.providerId)
                                                                    const key = `${model.providerId}:${model.id}`
                                                                    return (
                                                                        <option key={key} value={key}>
                                                                            {provider?.name || model.providerId} - {model.name}
                                                                        </option>
                                                                    )
                                                                })}
                                                            </select>
                                                        </div>

                                                        {/* Current Selection Info */}
                                                        {currentEmbeddingModel && (
                                                            <motion.div
                                                                initial={{ opacity: 0, y: -10 }}
                                                                animate={{ opacity: 1, y: 0 }}
                                                                className="mt-4 p-4 bg-blue-50 border-2 border-blue-200 rounded-xl"
                                                            >
                                                                <div className="flex items-start gap-4">
                                                                    {(() => {
                                                                        const provider = providers.find(p => p.id === currentEmbeddingModel.providerId)
                                                                        return provider && <ProviderAvatar provider={provider} size={48} />
                                                                    })()}
                                                                    <div className="flex-1">
                                                                        <div className="flex items-center gap-2 mb-2">
                                                                            <CheckCircle className="w-4 h-4 text-green-600" />
                                                                            <span className="font-semibold text-gray-900">
                                                                                当前默认模型
                                                                            </span>
                                                                        </div>
                                                                        <div className="space-y-1 text-sm">
                                                                            <div>
                                                                                <span className="text-gray-600">服务商:</span>{' '}
                                                                                <span className="font-semibold">
                                                                                    {providers.find(p => p.id === currentEmbeddingModel.providerId)?.name || currentEmbeddingModel.providerId}
                                                                                </span>
                                                                            </div>
                                                                            <div>
                                                                                <span className="text-gray-600">模型:</span>{' '}
                                                                                <span className="font-semibold">{currentEmbeddingModel.name}</span>
                                                                            </div>
                                                                            {currentEmbeddingModel.metadata?.dimension && (
                                                                                <div>
                                                                                    <span className="text-gray-600">向量维度:</span>{' '}
                                                                                    <span className="font-semibold">{currentEmbeddingModel.metadata.dimension}</span>
                                                                                </div>
                                                                            )}
                                                                            {currentEmbeddingModel.metadata?.description && (
                                                                                <div>
                                                                                    <span className="text-gray-600">说明:</span>{' '}
                                                                                    <span className="text-gray-800">{currentEmbeddingModel.metadata.description}</span>
                                                                                </div>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            </motion.div>
                                                        )}
                                                    </>
                                                ) : (
                                                    <div className="p-4 bg-gray-50 border-2 border-gray-200 rounded-xl text-center text-sm text-gray-600">
                                                        暂无可用的 Embedding 模型
                                                        <br />
                                                        <span className="text-xs text-gray-500 mt-1 block">
                                                            前往「服务商配置」添加 Embedding 模型
                                                        </span>
                                                    </div>
                                                )}
                                            </div>

                                            {/* Rerank 默认模型（可选） */}
                                            <div className="soft-card rounded-2xl p-6 space-y-4">
                                                <div className="flex items-center gap-3 mb-4">
                                                    <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center">
                                                        <Zap className="w-5 h-5 text-white" />
                                                    </div>
                                                    <h4 className="text-lg font-bold text-gray-900">Rerank 模型</h4>
                                                    <span className="px-2 py-1 bg-purple-100 text-purple-700 text-xs font-semibold rounded-lg">
                                                        可选
                                                    </span>
                                                </div>

                                                {rerankModels.length > 0 ? (
                                                    <>
                                                        <div>
                                                            <label className="text-sm font-semibold text-gray-700 mb-2 block">
                                                                选择重排模型（可提高搜索准确性）
                                                            </label>
                                                            <select
                                                                value={currentRerankKey || ''}
                                                                onChange={(e) => {
                                                                    const value = e.target.value
                                                                    setDefaultModel('rerankModel', value || null)
                                                                }}
                                                                className="soft-input w-full px-4 py-3 rounded-xl text-sm font-medium"
                                                            >
                                                                <option value="">不使用重排模型</option>
                                                                {rerankModels.map(model => {
                                                                    const provider = providers.find(p => p.id === model.providerId)
                                                                    const key = `${model.providerId}:${model.id}`
                                                                    return (
                                                                        <option key={key} value={key}>
                                                                            {provider?.name || model.providerId} - {model.name}
                                                                        </option>
                                                                    )
                                                                })}
                                                            </select>
                                                        </div>

                                                        {/* Current Rerank Selection */}
                                                        {currentRerankModel && (
                                                            <motion.div
                                                                initial={{ opacity: 0, y: -10 }}
                                                                animate={{ opacity: 1, y: 0 }}
                                                                className="mt-4 p-4 bg-purple-50 border-2 border-purple-200 rounded-xl"
                                                            >
                                                                <div className="flex items-start gap-4">
                                                                    {(() => {
                                                                        const provider = providers.find(p => p.id === currentRerankModel.providerId)
                                                                        return provider && <ProviderAvatar provider={provider} size={48} />
                                                                    })()}
                                                                    <div className="flex-1">
                                                                        <div className="flex items-center gap-2 mb-2">
                                                                            <CheckCircle className="w-4 h-4 text-green-600" />
                                                                            <span className="font-semibold text-gray-900">
                                                                                当前 Rerank 模型
                                                                            </span>
                                                                        </div>
                                                                        <div className="space-y-1 text-sm">
                                                                            <div>
                                                                                <span className="text-gray-600">服务商:</span>{' '}
                                                                                <span className="font-semibold">
                                                                                    {providers.find(p => p.id === currentRerankModel.providerId)?.name || currentRerankModel.providerId}
                                                                                </span>
                                                                            </div>
                                                                            <div>
                                                                                <span className="text-gray-600">模型:</span>{' '}
                                                                                <span className="font-semibold">{currentRerankModel.name}</span>
                                                                            </div>
                                                                            {currentRerankModel.metadata?.description && (
                                                                                <div>
                                                                                    <span className="text-gray-600">说明:</span>{' '}
                                                                                    <span className="text-gray-800">{currentRerankModel.metadata.description}</span>
                                                                                </div>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            </motion.div>
                                                        )}
                                                    </>
                                                ) : (
                                                    <div className="p-4 bg-gray-50 border-2 border-gray-200 rounded-xl text-center text-sm text-gray-600">
                                                        暂无可用的 Rerank 模型
                                                        <br />
                                                        <span className="text-xs text-gray-500 mt-1 block">
                                                            前往「服务商配置」添加 Rerank 模型
                                                        </span>
                                                    </div>
                                                )}
                                            </div>

                                            {/* Configuration Preview */}
                                            <div className="soft-card rounded-2xl p-6 bg-gradient-to-br from-green-50 to-blue-50">
                                                <div className="flex items-center gap-3 mb-4">
                                                    <Settings className="w-6 h-6 text-green-600" />
                                                    <h4 className="text-lg font-bold text-gray-900">配置预览</h4>
                                                </div>
                                                <div className="grid grid-cols-2 gap-4">
                                                    <div className="bg-white rounded-xl p-4 border border-gray-200">
                                                        <div className="text-xs text-gray-600 mb-1">Embedding 模型</div>
                                                        <div className="font-semibold text-gray-900">
                                                            {currentEmbeddingModel?.name || '未设置'}
                                                        </div>
                                                        {currentEmbeddingModel && (
                                                            <div className="text-xs text-gray-500 mt-1">
                                                                {providers.find(p => p.id === currentEmbeddingModel.providerId)?.name || currentEmbeddingModel.providerId}
                                                            </div>
                                                        )}
                                                    </div>
                                                    <div className="bg-white rounded-xl p-4 border border-gray-200">
                                                        <div className="text-xs text-gray-600 mb-1">Rerank 模型</div>
                                                        <div className="font-semibold text-gray-900">
                                                            {currentRerankModel?.name || '未设置'}
                                                        </div>
                                                        {currentRerankModel && (
                                                            <div className="text-xs text-gray-500 mt-1">
                                                                {providers.find(p => p.id === currentRerankModel.providerId)?.name || currentRerankModel.providerId}
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                                <div className="mt-4 text-xs text-gray-600 flex items-center gap-2">
                                                    <AlertCircle className="w-3 h-3" />
                                                    配置更改会立即保存到本地存储
                                                </div>
                                            </div>
                                        </div>
                                    )
                                })()}
                            </div>
                        )}
                    </div>
                </motion.div>
            </motion.div>
        </AnimatePresence>

        {/* Manage Models Popup */}
        {activeProvider && (
            <ManageModelsPopup
                isOpen={manageModelsOpen}
                onClose={() => setManageModelsOpen(false)}
                providerId={activeProvider.id}
            />
        )}
    </>
    )
}
