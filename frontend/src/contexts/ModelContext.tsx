import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { SYSTEM_MODELS } from '../config/systemModels'
import type { Model, ModelType, ModelCapability, UserModelCollection } from '../types/model'
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
    updateModelInCollection: (modelId: string, providerId: string, updates: Partial<Model>) => void
    isModelInCollection: (modelId: string, providerId: string) => boolean

    // 模型获取
    getModelsByType: (type: ModelType) => Model[]
    getModelsByProvider: (providerId: string) => Model[]
    getModelById: (modelId: string, providerId: string) => Model | null

    // 从provider API获取模型
    fetchAndAddModels: (provider: Provider, options?: { autoAdd?: boolean }) => Promise<Model[]>
    isFetching: boolean
    fetchError: string | null
}

const ModelContext = createContext<ModelContextType | undefined>(undefined)

const CONFIG_VERSION = '4.1'
const STORAGE_KEY = 'userModels'
const VERSION_KEY = 'userModelsVersion'
const LAST_SYNC_KEY = 'modelsLastSync'

/**
 * 版本迁移：从旧版本数据中保留用户手动添加的模型（isUserAdded 为 true），
 * 仅清除系统预设模型的缓存。
 *
 * @param oldData - localStorage 中的旧版本 JSON 字符串
 * @returns 迁移后的用户模型数组，解析失败时返回 null（回退到空数组）
 */
export function migrateUserModels(oldData: string): Model[] | null {
    try {
        const parsed = JSON.parse(oldData)
        if (!Array.isArray(parsed)) return null

        // 仅保留用户手动添加的模型
        const userModels = parsed.filter(
            (m: any) => m && typeof m === 'object' && m.isUserAdded === true
        ) as Model[]

        if (userModels.length === 0) return null

        return userModels
    } catch {
        return null
    }
}

export function ModelProvider({ children }: { children: ReactNode }) {
    const [systemModels] = useState<Model[]>(SYSTEM_MODELS)

    const [userCollection, setUserCollection] = useState<Model[]>(() => {
        const savedVersion = localStorage.getItem(VERSION_KEY)
        const saved = localStorage.getItem(STORAGE_KEY)

        // 版本不匹配时尝试迁移用户模型
        if (saved && savedVersion !== CONFIG_VERSION) {
            console.log('🔄 Upgrading user model collection to version', CONFIG_VERSION)

            // 尝试从旧数据迁移用户手动添加的模型
            const migrated = migrateUserModels(saved)
            if (migrated) {
                console.log('✅ 成功迁移用户模型，保留', migrated.length, '个用户添加的模型')
                localStorage.setItem(VERSION_KEY, CONFIG_VERSION)
                localStorage.removeItem(LAST_SYNC_KEY)
                return migrated
            }
            console.warn('⚠️ 用户模型迁移失败或无用户模型，使用空集合')

            localStorage.removeItem(STORAGE_KEY)
            localStorage.removeItem(LAST_SYNC_KEY)
        }

        // 版本匹配时加载
        if (saved && savedVersion === CONFIG_VERSION) {
            try {
                const parsed = JSON.parse(saved) as Model[]
                // 仅保留用户真正添加的模型，过滤掉之前缓存的系统模型
                const filtered = parsed.filter(m => m.isUserAdded || !m.isSystem)
                console.log('✅ Loaded user model collection (v' + CONFIG_VERSION + ')')
                return filtered
            } catch (error) {
                console.warn('Failed to parse saved user models')
            }
        }

        // 保存新版本号
        localStorage.setItem(VERSION_KEY, CONFIG_VERSION)

        // 默认不注入系统模型，用户模型集合仅存储用户新增的模型
        return []
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
     * 确保 capabilities 和 tags 字段被正确持久化：
     * - 如果模型已包含 capabilities，直接保留
     * - 如果模型未包含 capabilities，根据 type 自动生成默认的 capability（isUserSelected=true 表示用户手动添加）
     * - tags 字段直接透传保留
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

            // 确保 capabilities 字段存在：若缺失则根据 type 自动生成
            const capabilities: ModelCapability[] = model.capabilities && model.capabilities.length > 0
                ? model.capabilities
                : [{ type: model.type, isUserSelected: true }]

            // 确保 tags 字段存在：若缺失则默认为空数组
            const tags: string[] = model.tags || []

            return [...prev, {
                ...model,
                capabilities,
                tags,
                isUserAdded: true
            }]
        })
    }

    /**
     * 更新用户collection中已有模型的信息
     * 支持更新 capabilities、tags 等字段并持久化到 localStorage
     * 满足需求 2.4：用户编辑已有模型的类型时，更新 capabilities 数组中对应条目的 isUserSelected 标志
     */
    const updateModelInCollection = (modelId: string, providerId: string, updates: Partial<Model>) => {
        setUserCollection(prev =>
            prev.map(m => {
                if (m.id === modelId && m.providerId === providerId) {
                    return { ...m, ...updates }
                }
                return m
            })
        )
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
    const fetchAndAddModels = async (provider: Provider, options?: { autoAdd?: boolean }) => {
        setIsFetching(true)
        setFetchError(null)

        try {
            const models = await fetchModelsFromProvider(provider)

            // 可选：将获取到的模型添加到collection
            if (options?.autoAdd !== false) {
                models.forEach(model => {
                    addModelToCollection(model)
                })

                // 更新最后同步时间
                const lastSync = JSON.parse(localStorage.getItem(LAST_SYNC_KEY) || '{}')
                lastSync[provider.id] = Date.now()
                localStorage.setItem(LAST_SYNC_KEY, JSON.stringify(lastSync))
            }

            console.log(`✅ Fetched ${models.length} models from ${provider.name}`)
            return models
        } catch (error) {
            const message = error instanceof Error ? error.message : '获取模型失败'
            setFetchError(message)
            console.error('Error fetching models:', error)
            return []
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
                updateModelInCollection,
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
