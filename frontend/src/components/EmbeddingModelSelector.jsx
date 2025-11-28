import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Check, Zap, DollarSign } from 'lucide-react'
import { useEmbedding } from '../contexts/EmbeddingContext'

export default function EmbeddingModelSelector() {
    const {
        providers,
        selectedProviderId,
        selectedEmbeddingModelId,
        selectProvider,
        selectEmbeddingModel,
        getCurrentProvider,
        getCurrentEmbeddingModel
    } = useEmbedding()

    const [isOpen, setIsOpen] = useState(false)
    const dropdownRef = useRef(null)

    const currentProvider = getCurrentProvider()
    const currentModel = getCurrentEmbeddingModel()
    const enabledProviders = providers.filter(p => p.enabled)

    // 点击外部关闭
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

    return (
        <div className="relative" ref={dropdownRef}>
            {/* Trigger Button */}
            <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setIsOpen(!isOpen)}
                className={`
          w-full glass-3d rounded-2xl p-4 transition-all duration-200
          ${isOpen ? 'shadow-xl ring-2 ring-blue-500/50' : 'shadow-lg hover:shadow-xl'}
        `}
            >
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        {currentProvider && (
                            <span className="text-2xl">{currentProvider.logo}</span>
                        )}
                        <div className="text-left">
                            <div className="text-sm font-semibold text-gray-900">
                                {currentModel?.name || '选择嵌入模型'}
                            </div>
                            {currentModel && (
                                <div className="flex items-center gap-3 text-xs text-gray-600 mt-1">
                                    <span className="flex items-center gap-1">
                                        📐 {currentModel.dimension}维
                                    </span>
                                    {currentModel.pricing ? (
                                        <span className="flex items-center gap-1 px-2 py-0.5 bg-amber-500/20 text-amber-700 rounded font-semibold">
                                            <DollarSign className="w-3 h-3" />
                                            ${currentModel.pricing.perMillionTokens}/M
                                        </span>
                                    ) : (
                                        <span className="flex items-center gap-1 px-2 py-0.5 bg-green-500/20 text-green-700 rounded font-semibold">
                                            <Zap className="w-3 h-3" />
                                            免费
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                    <motion.div
                        animate={{ rotate: isOpen ? 180 : 0 }}
                        transition={{ duration: 0.2 }}
                    >
                        <ChevronDown className="w-5 h-5 text-gray-600" />
                    </motion.div>
                </div>
            </motion.button>

            {/* Dropdown */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className="absolute z-50 mt-2 w-full glass-panel rounded-2xl shadow-2xl overflow-hidden max-h-96"
                    >
                        <div className="overflow-y-auto max-h-96">
                            {enabledProviders.length > 0 ? (
                                enabledProviders.map((provider, providerIndex) => (
                                    <div key={provider.id}>
                                        {/* Provider Header */}
                                        <div className="sticky top-0 px-4 py-2 bg-white/60 backdrop-blur-md border-b border-white/40 z-10">
                                            <div className="flex items-center gap-2 text-xs font-bold text-gray-600">
                                                <span className="text-lg">{provider.logo}</span>
                                                <span>{provider.name}</span>
                                            </div>
                                        </div>

                                        {/* Embedding Models */}
                                        {provider.models.filter(m => m.type === 'embedding').map((model, modelIndex) => (
                                            <motion.button
                                                key={model.id}
                                                whileHover={{ backgroundColor: 'rgba(59, 130, 246, 0.05)' }}
                                                onClick={() => {
                                                    selectProvider(provider.id)
                                                    selectEmbeddingModel(model.id)
                                                    setIsOpen(false)
                                                }}
                                                className={`
                          w-full px-4 py-3 text-left transition-colors
                          ${selectedEmbeddingModelId === model.id ? 'bg-blue-50/50' : ''}
                          ${modelIndex === 0 ? '' : 'border-t border-white/30'}
                        `}
                                            >
                                                <div className="flex items-center justify-between">
                                                    <div className="flex-1 pr-3">
                                                        <div className="text-sm font-semibold text-gray-900">
                                                            {model.name}
                                                        </div>
                                                        {model.description && (
                                                            <div className="text-xs text-gray-600 mt-0.5 line-clamp-1">
                                                                {model.description}
                                                            </div>
                                                        )}
                                                        <div className="flex items-center gap-2 mt-1.5">
                                                            <span className="px-2 py-0.5 glass-panel rounded text-xs text-gray-600">
                                                                📐 {model.dimension}维
                                                            </span>
                                                            {model.pricing ? (
                                                                <span className="px-2 py-0.5 bg-amber-500/20 text-amber-700 rounded text-xs font-semibold">
                                                                    ${model.pricing.perMillionTokens}/M
                                                                </span>
                                                            ) : (
                                                                <span className="px-2 py-0.5 bg-green-500/20 text-green-700 rounded text-xs font-semibold">
                                                                    免费
                                                                </span>
                                                            )}
                                                        </div>
                                                    </div>
                                                    {selectedEmbeddingModelId === model.id && (
                                                        <motion.div
                                                            initial={{ scale: 0 }}
                                                            animate={{ scale: 1 }}
                                                            className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0"
                                                        >
                                                            <Check className="w-4 h-4 text-white" />
                                                        </motion.div>
                                                    )}
                                                </div>
                                            </motion.button>
                                        ))}
                                    </div>
                                ))
                            ) : (
                                <div className="px-4 py-8 text-center">
                                    <div className="text-gray-500 mb-2">😔 没有启用的服务商</div>
                                    <p className="text-sm text-gray-400">请在设置中启用至少一个服务商</p>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
