import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Check, Zap, DollarSign } from 'lucide-react'
import { useProvider } from '../contexts/ProviderContext'
import { useModel } from '../contexts/ModelContext'
import { useDefaults } from '../contexts/DefaultsContext'
import ProviderAvatar from './ProviderAvatar'

export default function EmbeddingModelSelector() {
    const { providers, getProviderById } = useProvider()
    const { getModelById, getModelsByType } = useModel()
    const { getDefaultModel, setDefaultModel } = useDefaults()

    const [isOpen, setIsOpen] = useState(false)
    const dropdownRef = useRef(null)

    // Get current embedding model
    const embeddingModelKey = getDefaultModel('embeddingModel')
    const [currentProviderId, currentModelId] = embeddingModelKey?.split(':') || [null, null]
    const currentProvider = currentProviderId ? getProviderById(currentProviderId) : null
    const currentModel = currentModelId ? getModelById(currentModelId, currentProviderId) : null

    // Get all embedding models grouped by provider
    const embeddingModels = getModelsByType('embedding')
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
          w-full soft-card rounded-2xl p-4 transition-all duration-200
          ${isOpen ? 'shadow-xl ring-2 ring-primary-500/50' : 'shadow-lg hover:shadow-xl'}
        `}
            >
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        {currentProvider && (
                            <ProviderAvatar provider={currentProvider} size={24} />
                        )}
                        <div className="text-left">
                            <div className="text-sm font-semibold text-gray-900">
                                {currentModel?.name || '选择嵌入模型'}
                            </div>
                            {currentModel && (
                                <div className="flex items-center gap-3 text-xs text-gray-600 mt-1">
                                    <span className="flex items-center gap-1">
                                        📐 {currentModel.metadata?.dimension || '?'}维
                                    </span>
                                    {currentModel.metadata?.pricing ? (
                                        <span className="flex items-center gap-1 px-2 py-0.5 bg-amber-500/20 text-amber-700 rounded font-semibold">
                                            <DollarSign className="w-3 h-3" />
                                            ${currentModel.metadata.pricing.perMillionTokens}/M
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
                        className="absolute z-50 mt-2 w-full soft-panel rounded-2xl shadow-2xl overflow-hidden max-h-96"
                    >
                        <div className="overflow-y-auto max-h-96">
                            {enabledProviders.length > 0 ? (
                                enabledProviders.map((provider, providerIndex) => {
                                    // Filter embedding models for this provider
                                    const providerEmbeddingModels = embeddingModels.filter(m => m.providerId === provider.id)

                                    if (providerEmbeddingModels.length === 0) return null

                                    return (
                                        <div key={provider.id}>
                                            {/* Provider Header */}
                                            <div className="sticky top-0 px-4 py-2 bg-white/60 backdrop-blur-md border-b border-white/40 z-10">
                                                <div className="flex items-center gap-2 text-xs font-bold text-gray-600">
                                                    <ProviderAvatar provider={provider} size={20} />
                                                    <span>{provider.name}</span>
                                                </div>
                                            </div>

                                            {/* Embedding Models */}
                                            {providerEmbeddingModels.map((model, modelIndex) => {
                                                const isSelected = embeddingModelKey === `${provider.id}:${model.id}`

                                                return (
                                                    <motion.button
                                                        key={model.id}
                                                        whileHover={{ backgroundColor: 'rgba(59, 130, 246, 0.05)' }}
                                                        onClick={() => {
                                                            setDefaultModel('embeddingModel', `${provider.id}:${model.id}`)
                                                            setIsOpen(false)
                                                        }}
                                                        className={`
                                  w-full px-4 py-3 text-left transition-colors
                                  ${isSelected ? 'bg-primary-50/50' : ''}
                                  ${modelIndex === 0 ? '' : 'border-t border-white/30'}
                                `}
                                                    >
                                                        <div className="flex items-center justify-between">
                                                            <div className="flex-1 pr-3">
                                                                <div className="text-sm font-semibold text-gray-900">
                                                                    {model.name}
                                                                </div>
                                                                {model.metadata?.description && (
                                                                    <div className="text-xs text-gray-600 mt-0.5 line-clamp-1">
                                                                        {model.metadata.description}
                                                                    </div>
                                                                )}
                                                                <div className="flex items-center gap-2 mt-1.5">
                                                                    <span className="px-2 py-0.5 soft-panel rounded text-xs text-gray-600">
                                                                        📐 {model.metadata?.dimension || '?'}维
                                                                    </span>
                                                                    {model.metadata?.pricing ? (
                                                                        <span className="px-2 py-0.5 bg-amber-500/20 text-amber-700 rounded text-xs font-semibold">
                                                                            ${model.metadata.pricing.perMillionTokens}/M
                                                                        </span>
                                                                    ) : (
                                                                        <span className="px-2 py-0.5 bg-green-500/20 text-green-700 rounded text-xs font-semibold">
                                                                            免费
                                                                        </span>
                                                                    )}
                                                                </div>
                                                            </div>
                                                            {isSelected && (
                                                                <motion.div
                                                                    initial={{ scale: 0 }}
                                                                    animate={{ scale: 1 }}
                                                                    className="w-6 h-6 bg-primary-500 rounded-full flex items-center justify-center flex-shrink-0"
                                                                >
                                                                    <Check className="w-4 h-4 text-white" />
                                                                </motion.div>
                                                            )}
                                                        </div>
                                                    </motion.button>
                                                )
                                            })}
                                        </div>
                                    )
                                })
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
