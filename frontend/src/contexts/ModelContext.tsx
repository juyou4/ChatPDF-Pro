import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { SYSTEM_MODELS } from '../config/systemModels'
import type { Model, ModelType, UserModelCollection } from '../types/model'
import {
    fetchModelsFromProvider,
    filterModels,
    mergeModels,
    groupModelsByType
} from '../services/modelService'
import type { Provider } from '../types/provider'

/**
 * ModelContext接口定义
 * 三层架构的中层：模型管理
 */
interface ModelContextType {
    allModels: Model[]                                        // 所有可用模型（系统+用户）
    userCollection: Model[]                                   // 用户collection
    systemModels: Model[]                                     // 系统预设模型

    // 模型操作
    addModelToCollection: (model: Model) => void
    removeModelFromCollection: (modelId: string, providerId: string) => void
    isModelInCollection: (modelId: string, providerId: string) => boolean

    // 模型获取
    getModelsByType: (type: ModelType) => Model[]
    getModelsByProvider: (providerId: string) => Model[]
    getModelById: (modelId: string, providerId: string) => Model | null

    // 从provider API获取模型
    fetchAndAddModels: (provider: Provider) => Promise<void>
    isFetching: boolean
    fetchError: string | null
}

const ModelContext = createContext<ModelContextType | undefined>(undefined)

const CONFIG_VERSION = '4.0'
const STORAGE_KEY = 'userModels'
const VERSION_KEY = 'userModelsVersion'
const LAST_SYNC_KEY = 'modelsLastSync'

export function ModelProvider({ children }: { children: ReactNode }) {
    const [systemModels] = useState<Model[]>(SYSTEM_MODELS)

    const [userCollection, setUserCollection] = useState<Model[]>(() => {
        const savedVersion = localStorage.getItem(VERSION_KEY)
        const saved = localStorage.getItem(STORAGE_KEY)

        // 版本不匹配时清除旧数据
        if (saved && savedVersion !== CONFIG_VERSION) {
            console.log('🔄 Upgrading user model collection to version', CONFIG_VERSION)
            localStorage.removeItem(STORAGE_KEY)
            localStorage.removeItem(LAST_SYNC_KEY)
        }

        // 版本匹配时加载
        if (saved && savedVersion === CONFIG_VERSION) {
            try {
                const parsed = JSON.parse(saved) as Model[]
                console.log('✅ Loaded user model collection (v' + CONFIG_VERSION + ')')
                return parsed
            } catch (error) {
                console.warn('Failed to parse saved user models')
            }
        }

        // 保存新版本号
        localStorage.setItem(VERSION_KEY, CONFIG_VERSION)

        // 默认将所有系统模型添加到collection
        // 不标记为用户添加，这样删除后不会重新出现
        return systemModels.map(m => ({ ...m, isUserAdded: false }))
    })

    const [isFetching, setIsFetching] = useState(false)
    const [fetchError, setFetchError] = useState<string | null>(null)

    // 合并系统模型和用户模型
    const allModels = mergeModels(systemModels, userCollection)

    // 保存用户collection到localStorage
    useEffect(() => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(userCollection))
    }, [userCollection])

    /**
     * 添加模型到用户collection
     */
    const addModelToCollection = (model: Model) => {
        setUserCollection(prev => {
            // 避免重复添加
            const exists = prev.some(
                m => m.id === model.id && m.providerId === model.providerId
            )

            if (exists) {
                return prev
            }

            return [...prev, { ...model, isUserAdded: true }]
        })
    }

    /**
     * 从用户collection移除模型
     */
    const removeModelFromCollection = (modelId: string, providerId: string) => {
        setUserCollection(prev =>
            prev.filter(m => !(m.id === modelId && m.providerId === providerId))
        )
    }

    /**
     * 检查模型是否在collection中
     */
    const isModelInCollection = (modelId: string, providerId: string): boolean => {
        return userCollection.some(
            m => m.id === modelId && m.providerId === providerId
        )
    }

    /**
     * 按类型获取模型
     */
    const getModelsByType = (type: ModelType): Model[] => {
        return filterModels(allModels, { type })
    }

    /**
     * 按provider获取模型
     */
    const getModelsByProvider = (providerId: string): Model[] => {
        return filterModels(allModels, { providerId })
    }

    /**
     * 根据ID获取模型
     */
    const getModelById = (modelId: string, providerId: string): Model | null => {
        return allModels.find(
            m => m.id === modelId && m.providerId === providerId
        ) || null
    }

    /**
     * 从provider API获取模型并添加到collection
     */
    const fetchAndAddModels = async (provider: Provider) => {
        setIsFetching(true)
        setFetchError(null)

        try {
            const models = await fetchModelsFromProvider(provider)

            // 将获取到的模型添加到collection
            models.forEach(model => {
                addModelToCollection(model)
            })

            // 更新最后同步时间
            const lastSync = JSON.parse(localStorage.getItem(LAST_SYNC_KEY) || '{}')
            lastSync[provider.id] = Date.now()
            localStorage.setItem(LAST_SYNC_KEY, JSON.stringify(lastSync))

            console.log(`✅ Fetched ${models.length} models from ${provider.name}`)
        } catch (error) {
            const message = error instanceof Error ? error.message : '获取模型失败'
            setFetchError(message)
            console.error('Error fetching models:', error)
        } finally {
            setIsFetching(false)
        }
    }

    return (
        <ModelContext.Provider
            value={{
                allModels,
                userCollection,
                systemModels,
                addModelToCollection,
                removeModelFromCollection,
                isModelInCollection,
                getModelsByType,
                getModelsByProvider,
                getModelById,
                fetchAndAddModels,
                isFetching,
                fetchError
            }}
        >
            {children}
        </ModelContext.Provider>
    )
}

/**
 * useModel Hook
 */
export function useModel() {
    const context = useContext(ModelContext)
    if (!context) {
        throw new Error('useModel must be used within ModelProvider')
    }
    return context
}
