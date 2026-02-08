import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { SYSTEM_PROVIDERS } from '../config/providers'
import type { Provider, ProviderUpdate, ProviderTestResult } from '../types/provider'

/**
 * ProviderContext接口定义
 * 三层架构的底层：Provider配置管理
 */
interface ProviderContextType {
    providers: Provider[]
    addProvider: (provider: Provider) => void
    updateProvider: (id: string, updates: ProviderUpdate) => void
    testConnection: (id: string) => Promise<ProviderTestResult>
    getProviderById: (id: string) => Provider | null
    getEnabledProviders: () => Provider[]
}

const ProviderContext = createContext<ProviderContextType | undefined>(undefined)

const CONFIG_VERSION = '4.0'
const STORAGE_KEY = 'providers'
const VERSION_KEY = 'providersVersion'

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

        if (userConfigs.size === 0) return null

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

        return newProviders
    } catch {
        return null
    }
}

export function ProviderProvider({ children }: { children: ReactNode }) {
    const [providers, setProviders] = useState<Provider[]>(() => {
        const savedVersion = localStorage.getItem(VERSION_KEY)
        const saved = localStorage.getItem(STORAGE_KEY)

        // 版本不匹配时尝试迁移旧数据
        if (savedVersion !== CONFIG_VERSION) {
            console.log('🔄 Upgrading to version', CONFIG_VERSION)

            // 尝试从旧数据迁移用户配置
            if (saved) {
                const migrated = migrateProviders(saved)
                if (migrated) {
                    console.log('✅ 成功从旧版本迁移 Provider 配置')
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

        // 版本匹配时使用保存的配置
        if (saved && savedVersion === CONFIG_VERSION) {
            try {
                const parsed = JSON.parse(saved) as Provider[]
                console.log('✅ Loaded providers from cache (v' + CONFIG_VERSION + ')')
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

    /**
     * 更新Provider配置
     */
    const updateProvider = (id: string, updates: ProviderUpdate) => {
        setProviders(prev =>
            prev.map(p => {
                if (p.id === id) {
                    return { ...p, ...updates }
                }
                return p
            })
        )
    }

    /**
     * 测试Provider连接
     * 调用后端API验证provider配置是否正确
     */
    const testConnection = async (id: string): Promise<ProviderTestResult> => {
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
                    fetchModelsEndpoint: provider.apiConfig?.fetchModelsEndpoint
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
                success: true,
                message: '连接成功',
                availableModels: result.availableModels,
                latency: result.latency  // 传递后端返回的延迟毫秒数（可选字段）
            }
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : '网络错误'
            }
        }
    }

    /**
     * 根据ID获取Provider
     */
    const getProviderById = (id: string): Provider | null => {
        return providers.find(p => p.id === id) || null
    }

    /**
     * 获取所有启用的Providers
     */
    const getEnabledProviders = (): Provider[] => {
        return providers.filter(p => p.enabled)
    }

    /**
     * 新增自定义 Provider（OpenAI 兼容或自建网关）
     */
    const addProvider = (provider: Provider) => {
        setProviders(prev => {
            const exists = prev.some(p => p.id === provider.id)
            if (exists) return prev
            return [...prev, { ...provider, isSystem: false }]
        })
    }

    return (
        <ProviderContext.Provider
            value={{
                providers,
                addProvider,
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
