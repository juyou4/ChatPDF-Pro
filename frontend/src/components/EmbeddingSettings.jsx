import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Settings, X, ChevronRight, Check, Key, Server,
    Zap, DollarSign, Search, AlertCircle, CheckCircle
} from 'lucide-react'
import { useEmbedding } from '../contexts/EmbeddingContext'

export default function EmbeddingSettings({ isOpen, onClose }) {
    const {
        providers,
        selectedProviderId,
        selectProvider,
        setProviderApiKey,
        setProviderApiHost,
        toggleProviderEnabled
    } = useEmbedding()

    const [activeProviderId, setActiveProviderId] = useState(selectedProviderId)
    const [searchQuery, setSearchQuery] = useState('')

    const activeProvider = providers.find(p => p.id === activeProviderId)

    // 过滤服务商
    const filteredProviders = providers.filter(p =>
        p.name.toLowerCase().includes(searchQuery.toLowerCase())
    )

    if (!isOpen) return null

    return (
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
                    className="glass-panel rounded-[32px] w-full max-w-5xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden"
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

                    {/* Content */}
                    <div className="flex-1 flex overflow-hidden">
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
                                        className="glass-input w-full pl-10 pr-4 py-2.5 rounded-xl text-sm"
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
                                                ? 'glass-3d shadow-lg'
                                                : 'hover:bg-white/30 bg-white/10'}
                    `}
                                    >
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                <span className="text-3xl">{provider.logo}</span>
                                                <div>
                                                    <div className="font-semibold text-gray-900 text-sm">
                                                        {provider.name}
                                                    </div>
                                                    <div className="text-xs text-gray-500 mt-0.5">
                                                        {provider.models.length} 个模型
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
                        {activeProvider && (
                            <div className="flex-1 overflow-y-auto p-6">
                                <motion.div
                                    key={activeProvider.id}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    {/* API Configuration */}
                                    <div className="glass-3d rounded-2xl p-6 mb-6">
                                        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                                            <Server className="w-5 h-5 text-blue-600" />
                                            API 配置
                                        </h3>

                                        {activeProvider.type === 'local' ? (
                                            <div className="glass-3d bg-gradient-to-br from-green-50/50 to-emerald-50/50 rounded-xl p-5 border border-green-200/50">
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
                                                        value={activeProvider.apiKey}
                                                        onChange={(e) => setProviderApiKey(activeProvider.id, e.target.value)}
                                                        placeholder="sk-..."
                                                        className="glass-input w-full px-4 py-3 rounded-xl text-sm"
                                                    />
                                                </div>

                                                <div>
                                                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                                                        <Server className="w-4 h-4 text-blue-600" />
                                                        API 地址
                                                    </label>
                                                    <input
                                                        type="text"
                                                        value={activeProvider.apiHost}
                                                        onChange={(e) => setProviderApiHost(activeProvider.id, e.target.value)}
                                                        placeholder="https://..."
                                                        className="glass-input w-full px-4 py-3 rounded-xl text-sm font-mono"
                                                    />
                                                </div>

                                                <div className="flex items-center gap-3 pt-2">
                                                    <button
                                                        onClick={() => toggleProviderEnabled(activeProvider.id)}
                                                        className={`
                              px-6 py-3 rounded-xl font-semibold transition-all duration-200
                              ${activeProvider.enabled
                                                                ? 'bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg shadow-green-500/30 hover:shadow-green-500/50 hover:scale-105'
                                                                : 'glass-3d text-gray-700 hover:scale-105'}
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

                                                    <button className="glass-3d px-6 py-3 rounded-xl font-semibold hover:scale-105 transition-transform">
                                                        测试连接
                                                    </button>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Models List */}
                                    <div className="space-y-4">
                                        <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                                            <Zap className="w-5 h-5 text-blue-600" />
                                            可用模型
                                        </h3>

                                        {/* Embedding Models */}
                                        {activeProvider.models.filter(m => m.type === 'embedding').length > 0 && (
                                            <div>
                                                <div className="text-sm font-semibold text-gray-500 mb-3 px-2">
                                                    📊 嵌入模型
                                                </div>
                                                <div className="space-y-3">
                                                    {activeProvider.models
                                                        .filter(m => m.type === 'embedding')
                                                        .map(model => (
                                                            <ModelCard key={model.id} model={model} />
                                                        ))}
                                                </div>
                                            </div>
                                        )}

                                        {/* Rerank Models */}
                                        {activeProvider.models.filter(m => m.type === 'rerank').length > 0 && (
                                            <div className="mt-6">
                                                <div className="text-sm font-semibold text-gray-500 mb-3 px-2">
                                                    🔄 重排模型
                                                </div>
                                                <div className="space-y-3">
                                                    {activeProvider.models
                                                        .filter(m => m.type === 'rerank')
                                                        .map(model => (
                                                            <ModelCard key={model.id} model={model} isRerank />
                                                        ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </motion.div>
                            </div>
                        )}
                    </div>
                </motion.div>
            </motion.div>
        </AnimatePresence>
    )
}

function ModelCard({ model, isRerank = false }) {
    return (
        <motion.div
            whileHover={{ scale: 1.02, y: -2 }}
            className="glass-3d rounded-2xl p-5 hover:shadow-xl transition-all duration-200"
        >
            <div className="flex items-start justify-between">
                <div className="flex-1">
                    <div className="flex items-center gap-2">
                        <div className="font-semibold text-gray-900">{model.name}</div>
                        {isRerank && (
                            <span className="px-2 py-0.5 bg-purple-500/20 text-purple-700 text-xs font-bold rounded-lg">
                                重排
                            </span>
                        )}
                    </div>

                    {model.description && (
                        <div className="text-sm text-gray-600 mt-1.5 leading-relaxed">
                            {model.description}
                        </div>
                    )}

                    <div className="flex items-center gap-4 mt-3 flex-wrap">
                        <span className="px-3 py-1 glass-panel rounded-lg text-xs font-medium text-gray-700">
                            📐 维度: {model.dimension}
                        </span>
                        <span className="px-3 py-1 glass-panel rounded-lg text-xs font-medium text-gray-700">
                            📝 最大: {model.maxTokens} tokens
                        </span>
                        {model.pricing ? (
                            <span className="px-3 py-1 bg-gradient-to-r from-amber-500/20 to-orange-500/20 rounded-lg text-xs font-bold text-amber-700 flex items-center gap-1">
                                <DollarSign className="w-3 h-3" />
                                ${model.pricing.perMillionTokens}/M
                            </span>
                        ) : (
                            <span className="px-3 py-1 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-lg text-xs font-bold text-green-700 flex items-center gap-1">
                                <Zap className="w-3 h-3" />
                                免费
                            </span>
                        )}
                    </div>
                </div>

                <button className="ml-4 px-5 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white text-sm font-semibold rounded-xl shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:scale-105 transition-all">
                    选择
                </button>
            </div>
        </motion.div>
    )
}
