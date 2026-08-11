import React, { createContext, useContext, useState, useEffect, useMemo, ReactNode } from 'react'
import { SYSTEM_PROVIDERS } from '../config/providers'
import type { Provider, ProviderUpdate, ProviderTestResult } from '../types/provider'
import { useCapabilities } from './CapabilitiesContext'

/**
 * ProviderContext接口定义
 * 三层架构的底层：Provider配置管理
 */
interface ProviderContextType {
    providers: Provider[]
    addProvider: (provider: Provider) => Promise<void>
    deleteProvider: (id: string) => Promise<void>
    updateProvider: (id: string, updates: ProviderUpdate) => void
    testConnection: (id: string, options?: { modelId?: string }) => Promise<ProviderTestResult>
    getProviderById: (id: string) => Provider | null
    getEnabledProviders: () => Provider[]
}

const ProviderContext = createContext<ProviderContextType | undefined>(undefined)

const CONFIG_VERSION = '4.3'
const STORAGE_KEY = 'providers'
const VERSION_KEY = 'providersVersion'
const CUSTOM_PROVIDER_SAVE_TIMEOUT_MS = 10_000

// 旧架构的localStorage键名（需要清理）
const OLD_KEYS = [
    'embeddingProviders',
    'embeddingProvidersVersion',
    'selectedProviderId',
    'selectedEmbeddingModelId',
    'selectedRerankModelId'
]

/**
 * 版本迁移：从旧版本数据中提取用户配置的 apiKey 和 apiHost，
 * 合并到新版本的系统 Provider 中，而非直接清除所有数据。
 *
 * @param oldData - localStorage 中的旧版本 JSON 字符串
 * @returns 迁移后的 Provider 数组，解析失败时返回 null（回退到默认配置）
 */
export function migrateProviders(oldData: string): Provider[] | null {
    try {
        const parsed = JSON.parse(oldData)
        if (!Array.isArray(parsed)) return null

        // 从旧数据中提取用户配置的 apiKey 和 apiHost
        const userConfigs = new Map<string, { apiKey: string; apiHost: string }>()
        for (const p of parsed) {
            if (p && typeof p === 'object' && p.id && p.apiKey) {
                userConfigs.set(p.id, {
                    apiKey: p.apiKey,
                    apiHost: p.apiHost || '',
                })
            }
        }
        const systemIds = new Set(SYSTEM_PROVIDERS.map(provider => provider.id))
        const customProviders = parsed
            .filter(p => p && typeof p === 'object' && p.id && !systemIds.has(p.id))
            .map(p => ({ ...p, isSystem: false }))

        // 只有系统 Provider 且没有任何用户配置时才回退默认值。自定义
        // Provider 可能在填写 API Key 之前就已经保存，不能因 Key 为空而丢失。
        if (userConfigs.size === 0 && customProviders.length === 0) return null

        // 将用户配置合并到新版本的系统 Provider 中
        const newProviders = SYSTEM_PROVIDERS.map(sp => {
            const userConfig = userConfigs.get(sp.id)
            if (userConfig) {
                return {
                    ...sp,
                    apiKey: userConfig.apiKey,
                    apiHost: userConfig.apiHost || sp.apiHost,
                    enabled: true,
                }
            }
            return sp
        })

        // 旧版本已经允许用户添加自定义 Provider，迁移时不能把它们
        // 静默丢掉；后续启动会再尝试同步到后端动态存储。
        return [...newProviders, ...customProviders]
    } catch {
        return null
    }
}

export function ProviderProvider({ children }: { children: ReactNode }) {
    const { hasLocalEmbedding, hasLocalRerank } = useCapabilities()

    const [providers, setProviders] = useState<Provider[]>(() => {
        const savedVersion = localStorage.getItem(VERSION_KEY)
        const saved = localStorage.getItem(STORAGE_KEY)

        // 版本不匹配时尝试迁移旧数据
        if (savedVersion !== CONFIG_VERSION) {
            // 尝试从旧数据迁移用户配置
            if (saved) {
                const migrated = migrateProviders(saved)
                if (migrated) {
                    localStorage.setItem(VERSION_KEY, CONFIG_VERSION)
                    // 清除旧架构的键名
                    OLD_KEYS.forEach(key => localStorage.removeItem(key))
                    return migrated
                }
                console.warn('⚠️ 旧版本数据迁移失败，使用默认配置')
            }

            // 清除旧版本数据
            localStorage.removeItem(STORAGE_KEY)

            // 清除旧架构的键名
            OLD_KEYS.forEach(key => localStorage.removeItem(key))
        }

        // 版本匹配时使用保存的配置，并补全缺失的系统 Provider
        if (saved && savedVersion === CONFIG_VERSION) {
            try {
                const parsed = JSON.parse(saved) as Provider[]
                // 检查是否有新增的系统 Provider 不在缓存中，补全到列表末尾
                const cachedIds = new Set(parsed.map(p => p.id))
                const missing = SYSTEM_PROVIDERS.filter(sp => !cachedIds.has(sp.id))
                if (missing.length > 0) {
                    const reconciled = [...parsed, ...missing]
                    localStorage.setItem(STORAGE_KEY, JSON.stringify(reconciled))
                    return reconciled
                }
                return parsed
            } catch (error) {
                console.warn('Failed to parse saved providers, using defaults')
            }
        }

        // 保存新版本号
        localStorage.setItem(VERSION_KEY, CONFIG_VERSION)

        // 返回默认配置
        return [...SYSTEM_PROVIDERS]
    })

    // 保存到localStorage
    useEffect(() => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(providers))
    }, [providers])

    const persistCustomProvider = async (provider: Provider): Promise<void> => {
        const controller = new AbortController()
        const timeoutId = window.setTimeout(() => controller.abort(), CUSTOM_PROVIDER_SAVE_TIMEOUT_MS)
        try {
            const response = await fetch('/api/providers/custom', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: controller.signal,
                body: JSON.stringify({
                    providerId: provider.id,
                    name: provider.name,
                    endpoint: provider.apiHost,
                    type: provider.apiConfig?.protocol || 'custom',
                    fetchModelsEndpoint: provider.apiConfig?.fetchModelsEndpoint || '/models',
                    chatEndpoint: provider.apiConfig?.chatEndpoint || '/chat/completions',
                    embeddingEndpoint: provider.apiConfig?.embeddingEndpoint || '/embeddings',
                    rerankEndpoint: provider.apiConfig?.rerankEndpoint || null,
                    supportsStreaming: provider.apiConfig?.supportsStreaming !== false,
                    supportsReasoning: provider.apiConfig?.supportsReasoning === true,
                    reasoningMode: provider.apiConfig?.reasoningMode || null,
                    reasoningOptions: provider.apiConfig?.reasoningOptions || null,
                    reasoningDefault: provider.apiConfig?.reasoningDefault || null,
                    reasoningAlwaysEnabled: provider.apiConfig?.reasoningAlwaysEnabled ?? null,
                    reasoningOffControl: provider.apiConfig?.reasoningOffControl || null,
                    reasoningOnControl: provider.apiConfig?.reasoningOnControl || null,
                    apiKeyHeader: provider.apiConfig?.apiKeyHeader || null,
                    apiKeyPrefix: provider.apiConfig?.apiKeyPrefix ?? null,
                    capabilities: provider.capabilities || {},
                })
            })
            if (!response.ok) {
                let message = '保存自定义 Provider 失败'
                try {
                    const payload = await response.json()
                    message = payload.detail || payload.message || message
                } catch {
                    // 保留默认错误
                }
                throw new Error(message)
            }
        } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
                throw new Error('保存超时，请确认后端服务可用后重试')
            }
            throw error
        } finally {
            window.clearTimeout(timeoutId)
        }
    }

    // 动态 Provider 的权威配置保存在后端（不包含 API Key）。启动时回灌
    // 非敏感 endpoint/protocol，API Key 仍只从当前桌面的 localStorage 读取。
    useEffect(() => {
        let cancelled = false
        fetch('/api/providers/custom')
            .then(async response => {
                if (!response.ok) return null
                return response.json()
            })
            .then(async (dynamic: Record<string, any> | null) => {
                if (cancelled || !dynamic || typeof dynamic !== 'object') return
                setProviders(prev => {
                    const byId = new Map(prev.map(provider => [provider.id, provider]))
                    Object.entries(dynamic).forEach(([id, config]) => {
                        if (!config || typeof config !== 'object') return
                        const previous = byId.get(id)
                        const apiConfig = {
                            ...(previous?.apiConfig || {}),
                            protocol: config.type || previous?.apiConfig?.protocol || 'custom',
                            fetchModelsEndpoint: config.fetch_models_endpoint || previous?.apiConfig?.fetchModelsEndpoint || '/models',
                            chatEndpoint: config.chat_endpoint || previous?.apiConfig?.chatEndpoint || '/chat/completions',
                            embeddingEndpoint: config.embedding_endpoint || previous?.apiConfig?.embeddingEndpoint || '/embeddings',
                            rerankEndpoint: config.rerank_endpoint || previous?.apiConfig?.rerankEndpoint || '',
                            supportsStreaming: config.supports_streaming !== false,
                            supportsReasoning: config.supports_reasoning === true,
                            reasoningMode: config.reasoning_mode || undefined,
                            reasoningOptions: Array.isArray(config.reasoning_options) ? config.reasoning_options : undefined,
                            reasoningDefault: config.reasoning_default || undefined,
                            reasoningAlwaysEnabled: typeof config.reasoning_always_enabled === 'boolean'
                                ? config.reasoning_always_enabled
                                : undefined,
                            reasoningOffControl: config.reasoning_off_control || undefined,
                            reasoningOnControl: config.reasoning_on_control || undefined,
                            // 后端保存的是动态 Provider 的权威非敏感配置。
                            // 使用 nullish 判断而不是 ``||``，这样用户清空认证头后
                            // 刷新不会又被旧的 localStorage 值覆盖。
                            apiKeyHeader: config.api_key_header ?? '',
                            apiKeyPrefix: config.api_key_prefix ?? '',
                        }
                        byId.set(id, {
                            ...(previous || {
                                id,
                                name: config.name || id,
                                apiKey: '',
                                enabled: true,
                                isSystem: false,
                                capabilities: { chat: true },
                            }),
                            name: config.name || previous?.name || id,
                            apiHost: config.endpoint || previous?.apiHost || '',
                            enabled: previous?.enabled ?? true,
                            isSystem: false,
                            capabilities: config.capabilities || previous?.capabilities || { chat: true },
                            apiConfig,
                        })
                    })
                    return Array.from(byId.values())
                })

                // 旧版本可能只把自定义 Provider 留在 localStorage。补写
                // 非敏感配置后，后端的 endpoint 元数据与前端状态保持一致；
                // API Key 仍不会离开当前桌面。
                const dynamicIds = new Set(Object.keys(dynamic))
                const missing = providers.filter(provider =>
                    !provider.isSystem &&
                    !dynamicIds.has(provider.id) &&
                    Boolean(provider.apiHost?.trim())
                )
                await Promise.all(
                    missing.map(provider =>
                        persistCustomProvider(provider).catch(error => {
                            console.warn('[Provider] 补写历史自定义 Provider 失败:', error)
                        })
                    )
                )
            })
            .catch(() => {
                // 后端不可达时继续使用本地缓存，桌面应用稍后可再次保存。
            })
        return () => { cancelled = true }
    }, [])

    /**
     * 更新Provider配置
     */
    const updateProvider = (id: string, updates: ProviderUpdate) => {
        const current = providers.find(provider => provider.id === id)
        const next = current ? { ...current, ...updates } : null
        setProviders(prev => prev.map(p => p.id === id ? { ...p, ...updates } : p))
        if (next && !next.isSystem) {
            persistCustomProvider(next).catch(error => {
                console.warn('[Provider] 保存自定义 Provider 失败:', error)
            })
        }
    }

    /**
     * 测试Provider连接
     * 调用后端API验证provider配置是否正确
     */
    const testConnection = async (id: string, options?: { modelId?: string }): Promise<ProviderTestResult> => {
        const provider = providers.find(p => p.id === id)

        if (!provider) {
            return {
                success: false,
                error: 'Provider not found'
            }
        }

        // 本地provider不需要测试
        if (provider.id === 'local') {
            return {
                success: true,
                message: '本地模型无需连接测试'
            }
        }

        // 检查API key是否配置
        if (!provider.apiKey) {
            return {
                success: false,
                error: '请先配置API Key'
            }
        }

        try {
            const response = await fetch('/api/providers/test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    providerId: provider.id,
                    apiKey: provider.apiKey,
                    apiHost: provider.apiHost,
                    fetchModelsEndpoint: provider.apiConfig?.fetchModelsEndpoint,
                    modelId: options?.modelId || undefined,
                    modelType: 'chat',
                    chatEndpoint: provider.apiConfig?.chatEndpoint,
                    providerType: provider.apiConfig?.protocol,
                    apiKeyHeader: provider.apiConfig?.apiKeyHeader,
                    apiKeyPrefix: provider.apiConfig?.apiKeyPrefix
                })
            })

            if (!response.ok) {
                const error = await response.json()
                return {
                    success: false,
                    error: error.message || '连接失败'
                }
            }

            const result = await response.json()
            return {
                success: result.success !== false,
                message: result.message || '连接成功',
                error: result.success === false ? (result.message || '连接失败') : undefined,
                availableModels: result.availableModels,
                latency: result.latency,
                verifiedModel: result.verifiedModel,
                chatEndpoint: result.chatEndpoint
            }
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : '网络错误'
            }
        }
    }

    // 根据后端能力过滤 provider 列表（桌面模式下隐藏不可用的本地模型）
    const filteredProviders = useMemo(() => {
        if (hasLocalEmbedding) return providers
        return providers.filter(p => p.id !== 'local')
    }, [providers, hasLocalEmbedding])

    /**
     * 根据ID获取Provider
     */
    const getProviderById = (id: string): Provider | null => {
        return filteredProviders.find(p => p.id === id) || null
    }

    /**
     * 获取所有启用的Providers
     */
    const getEnabledProviders = (): Provider[] => {
        return filteredProviders.filter(p => p.enabled)
    }

    /**
     * 新增自定义 Provider（OpenAI 兼容或自建网关）
     */
    const addProvider = async (provider: Provider): Promise<void> => {
        const normalized = { ...provider, isSystem: false }
        await persistCustomProvider(normalized)
        setProviders(prev => {
            const exists = prev.some(p => p.id === provider.id)
            if (exists) return prev.map(p => p.id === provider.id ? { ...p, ...normalized } : p)
            return [...prev, normalized]
        })
    }

    /**
     * 删除用户新增的 Provider。
     * 内置 Provider 不能通过设置界面删除，后端也会再次校验这一约束。
     */
    const deleteProvider = async (id: string): Promise<void> => {
        const provider = providers.find(item => item.id === id)
        if (!provider) return
        if (provider.isSystem) {
            throw new Error('内置模型服务不能删除')
        }

        const response = await fetch(`/api/providers/custom/${encodeURIComponent(id)}`, {
            method: 'DELETE',
        })
        if (!response.ok) {
            let message = '删除自定义模型服务失败'
            try {
                const payload = await response.json()
                message = payload.detail || payload.message || message
            } catch {
                // 保留默认错误
            }
            throw new Error(message)
        }

        setProviders(prev => prev.filter(item => item.id !== id))
    }

    return (
        <ProviderContext.Provider
            value={{
                providers: filteredProviders,
                addProvider,
                deleteProvider,
                updateProvider,
                testConnection,
                getProviderById,
                getEnabledProviders
            }}
        >
            {children}
        </ProviderContext.Provider>
    )
}

/**
 * useProvider Hook
 */
export function useProvider() {
    const context = useContext(ProviderContext)
    if (!context) {
        throw new Error('useProvider must be used within ProviderProvider')
    }
    return context
}
