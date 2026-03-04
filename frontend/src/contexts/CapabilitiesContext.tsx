/**
 * 后端能力上下文
 *
 * 启动时从 /capabilities 获取后端运行模式和可用功能，
 * 供 ProviderContext、EmbeddingSettings 等组件根据能力动态调整 UI。
 *
 * 能力字段：
 * - mode: "desktop" | "server"
 * - has_local_embedding: 是否支持本地 embedding 模型
 * - has_local_rerank: 是否支持本地 rerank 模型
 * - needs_api_key: 是否必须配置 API Key 才能使用
 * - embedding_providers: 可用 embedding provider 列表
 * - rerank_providers: 可用 rerank provider 列表
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

export interface Capabilities {
    mode: 'desktop' | 'server'
    version: string
    has_local_embedding: boolean
    has_local_rerank: boolean
    embedding_providers: string[]
    rerank_providers: string[]
    needs_api_key: boolean
    data_dir: string
    uptime: number
}

const DEFAULT_CAPABILITIES: Capabilities = {
    mode: 'server',
    version: '',
    has_local_embedding: true,
    has_local_rerank: true,
    embedding_providers: ['local', 'openai', 'silicon', 'aliyun'],
    rerank_providers: ['local', 'cohere', 'jina', 'silicon'],
    needs_api_key: false,
    data_dir: '',
    uptime: 0,
}

interface CapabilitiesContextType {
    capabilities: Capabilities
    isDesktopMode: boolean
    hasLocalEmbedding: boolean
    hasLocalRerank: boolean
    isLoading: boolean
}

const CapabilitiesContext = createContext<CapabilitiesContextType | undefined>(undefined)

export function CapabilitiesProvider({ children }: { children: ReactNode }) {
    const [capabilities, setCapabilities] = useState<Capabilities>(DEFAULT_CAPABILITIES)
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        let cancelled = false

        async function fetchCapabilities() {
            try {
                const res = await fetch('/capabilities')
                if (res.ok && !cancelled) {
                    const data = await res.json()
                    setCapabilities(data)
                }
            } catch (err) {
                // 网络错误时使用默认值（假设 server 模式，全功能）
                console.warn('[Capabilities] Failed to fetch, using defaults:', err)
            } finally {
                if (!cancelled) setIsLoading(false)
            }
        }

        fetchCapabilities()
        return () => { cancelled = true }
    }, [])

    const value: CapabilitiesContextType = {
        capabilities,
        isDesktopMode: capabilities.mode === 'desktop',
        hasLocalEmbedding: capabilities.has_local_embedding,
        hasLocalRerank: capabilities.has_local_rerank,
        isLoading,
    }

    return (
        <CapabilitiesContext.Provider value={value}>
            {children}
        </CapabilitiesContext.Provider>
    )
}

export function useCapabilities() {
    const context = useContext(CapabilitiesContext)
    if (!context) {
        throw new Error('useCapabilities must be used within CapabilitiesProvider')
    }
    return context
}
