import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { SYSTEM_EMBEDDING_PROVIDERS } from '../config/embeddingProviders'
import type { EmbeddingProvider, EmbeddingModel } from '../types/embedding'

interface EmbeddingContextType {
    providers: EmbeddingProvider[]
    selectedProviderId: string
    selectedEmbeddingModelId: string
    selectedRerankModelId: string | null

    // Provider 操作
    setProviderApiKey: (providerId: string, apiKey: string) => void
    setProviderApiHost: (providerId: string, apiHost: string) => void
    toggleProviderEnabled: (providerId: string) => void

    // 模型选择
    selectProvider: (providerId: string) => void
    selectEmbeddingModel: (modelId: string) => void
    selectRerankModel: (modelId: string | null) => void

    // 获取当前选择
    getCurrentProvider: () => EmbeddingProvider | null
    getCurrentEmbeddingModel: () => EmbeddingModel | null
    getCurrentRerankModel: () => EmbeddingModel | null
}

const EmbeddingContext = createContext<EmbeddingContextType | undefined>(undefined)

export function EmbeddingProvider({ children }: { children: ReactNode }) {
    const [providers, setProviders] = useState<EmbeddingProvider[]>(() => {
        // 从 localStorage 加载保存的配置
        const saved = localStorage.getItem('embeddingProviders')
        if (saved) {
            try {
                return JSON.parse(saved)
            } catch {
                console.warn('Failed to parse saved embedding providers, using defaults')
            }
        }
        return SYSTEM_EMBEDDING_PROVIDERS
    })

    const [selectedProviderId, setSelectedProviderId] = useState(() => {
        return localStorage.getItem('selectedProviderId') || 'local'
    })

    const [selectedEmbeddingModelId, setSelectedEmbeddingModelId] = useState(() => {
        return localStorage.getItem('selectedEmbeddingModelId') || 'all-MiniLM-L6-v2'
    })

    const [selectedRerankModelId, setSelectedRerankModelId] = useState<string | null>(() => {
        return localStorage.getItem('selectedRerankModelId') || null
    })

    // 保存到 localStorage
    useEffect(() => {
        localStorage.setItem('embeddingProviders', JSON.stringify(providers))
    }, [providers])

    useEffect(() => {
        localStorage.setItem('selectedProviderId', selectedProviderId)
    }, [selectedProviderId])

    useEffect(() => {
        localStorage.setItem('selectedEmbeddingModelId', selectedEmbeddingModelId)
    }, [selectedEmbeddingModelId])

    useEffect(() => {
        if (selectedRerankModelId) {
            localStorage.setItem('selectedRerankModelId', selectedRerankModelId)
        } else {
            localStorage.removeItem('selectedRerankModelId')
        }
    }, [selectedRerankModelId])

    // Provider 操作
    const setProviderApiKey = (providerId: string, apiKey: string) => {
        setProviders(prev =>
            prev.map(p => p.id === providerId ? { ...p, apiKey } : p)
        )
    }

    const setProviderApiHost = (providerId: string, apiHost: string) => {
        setProviders(prev =>
            prev.map(p => p.id === providerId ? { ...p, apiHost } : p)
        )
    }

    const toggleProviderEnabled = (providerId: string) => {
        setProviders(prev =>
            prev.map(p => p.id === providerId ? { ...p, enabled: !p.enabled } : p)
        )
    }

    // 获取当前选择
    const getCurrentProvider = () => {
        return providers.find(p => p.id === selectedProviderId) || null
    }

    const getCurrentEmbeddingModel = () => {
        const provider = getCurrentProvider()
        return provider?.models.find(m => m.id === selectedEmbeddingModelId) || null
    }

    const getCurrentRerankModel = () => {
        if (!selectedRerankModelId) return null
        const provider = getCurrentProvider()
        return provider?.models.find(m => m.id === selectedRerankModelId) || null
    }

    return (
        <EmbeddingContext.Provider
            value={{
                providers,
                selectedProviderId,
                selectedEmbeddingModelId,
                selectedRerankModelId,
                setProviderApiKey,
                setProviderApiHost,
                toggleProviderEnabled,
                selectProvider: setSelectedProviderId,
                selectEmbeddingModel: setSelectedEmbeddingModelId,
                selectRerankModel: setSelectedRerankModelId,
                getCurrentProvider,
                getCurrentEmbeddingModel,
                getCurrentRerankModel
            }}
        >
            {children}
        </EmbeddingContext.Provider>
    )
}

export function useEmbedding() {
    const context = useContext(EmbeddingContext)
    if (!context) {
        throw new Error('useEmbedding must be used within EmbeddingProvider')
    }
    return context
}
