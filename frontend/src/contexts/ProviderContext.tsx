import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { SYSTEM_PROVIDERS } from '../config/providers'
import type { Provider, ProviderUpdate, ProviderTestResult } from '../types/provider'

/**
 * ProviderContext接口定义
 * 三层架构的底层：Provider配置管理
 */
interface ProviderContextType {
    providers: Provider[]
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

export function ProviderProvider({ children }: { children: ReactNode }) {
    const [providers, setProviders] = useState<Provider[]>(() => {
        const savedVersion = localStorage.getItem(VERSION_KEY)
        const saved = localStorage.getItem(STORAGE_KEY)

        // 版本不匹配时清理旧数据并升级
        if (savedVersion !== CONFIG_VERSION) {
            console.log('🔄 Upgrading to version', CONFIG_VERSION)

            // 清除旧版本数据
            localStorage.removeItem(STORAGE_KEY)

            // 清除旧架构的键名
            OLD_KEYS.forEach(key => localStorage.removeItem(key))
            console.log('🧹 Cleaned up old configuration keys')
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
                    apiHost: provider.apiHost
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
                availableModels: result.availableModels
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

    return (
        <ProviderContext.Provider
            value={{
                providers,
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
