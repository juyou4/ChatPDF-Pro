import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import type { DefaultModels, DefaultModelType } from '../types/defaults'

/**
 * DefaultsContext接口定义
 * 三层架构的顶层：默认模型选择
 */
interface DefaultsContextType {
    defaults: DefaultModels
    setDefaultModel: (type: DefaultModelType, modelId: string | null) => void
    getDefaultModel: (type: DefaultModelType) => string | undefined
    clearDefaultModel: (type: DefaultModelType) => void
    resetToDefaults: () => void
}

const DefaultsContext = createContext<DefaultsContextType | undefined>(undefined)

const CONFIG_VERSION = '4.0'
const STORAGE_KEY = 'defaultModels'
const VERSION_KEY = 'defaultModelsVersion'

/**
 * 初始默认配置
 * 使用系统推荐的模型作为默认值
 */
const normalizeEmbeddingKey = (value?: string | null) => {
    if (!value) return undefined
    // 如果已经包含 provider 前缀则直接返回
    if (value.includes(':')) return value
    // 旧格式只存模型ID时，默认加上 local 前缀
    return `local:${value}`
}

const INITIAL_DEFAULTS: DefaultModels = {
    embeddingModel: 'local:all-MiniLM-L6-v2',  // 本地模型作为默认（带前缀）
    rerankModel: undefined,                    // rerank为可选
    assistantModel: 'deepseek:deepseek-chat',  // 默认 Chat 模型
}

/**
 * 版本迁移：从旧版本数据中迁移用户选择的默认模型配置。
 * 保留用户已选择的非空字段值。
 *
 * @param oldData - localStorage 中的旧版本 JSON 字符串
 * @returns 迁移后的默认模型配置，解析失败时返回 null（回退到默认配置）
 */
export function migrateDefaults(oldData: string): DefaultModels | null {
    try {
        const parsed = JSON.parse(oldData)
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null

        // 从旧数据中提取用户选择的非空字段
        const migrated: DefaultModels = { ...INITIAL_DEFAULTS }
        let hasUserData = false

        if (parsed.embeddingModel) {
            migrated.embeddingModel = normalizeEmbeddingKey(parsed.embeddingModel) || INITIAL_DEFAULTS.embeddingModel
            hasUserData = true
        }
        if (parsed.assistantModel) {
            migrated.assistantModel = parsed.assistantModel
            hasUserData = true
        }
        if (parsed.rerankModel) {
            migrated.rerankModel = parsed.rerankModel
            hasUserData = true
        }

        return hasUserData ? migrated : null
    } catch {
        return null
    }
}

export function DefaultsProvider({ children }: { children: ReactNode }) {
    const [defaults, setDefaults] = useState<DefaultModels>(() => {
        const savedVersion = localStorage.getItem(VERSION_KEY)
        const saved = localStorage.getItem(STORAGE_KEY)

        // 版本不匹配时尝试迁移旧数据
        if (saved && savedVersion !== CONFIG_VERSION) {
            console.log('🔄 Upgrading default models to version', CONFIG_VERSION)

            // 尝试从旧数据迁移用户选择的默认模型
            const migrated = migrateDefaults(saved)
            if (migrated) {
                console.log('✅ 成功从旧版本迁移默认模型配置')
                localStorage.setItem(VERSION_KEY, CONFIG_VERSION)
                return migrated
            }
            console.warn('⚠️ 默认模型迁移失败，使用默认配置')

            localStorage.removeItem(STORAGE_KEY)
        }

        // 版本匹配时加载
        if (saved && savedVersion === CONFIG_VERSION) {
            try {
                const parsed = JSON.parse(saved) as DefaultModels
                console.log('✅ Loaded default models (v' + CONFIG_VERSION + ')')
                return {
                    ...parsed,
                    embeddingModel: normalizeEmbeddingKey(parsed.embeddingModel) || INITIAL_DEFAULTS.embeddingModel
                }
            } catch (error) {
                console.warn('Failed to parse saved default models')
            }
        }

        // 保存新版本号
        localStorage.setItem(VERSION_KEY, CONFIG_VERSION)

        // 尝试从旧的localStorage迁移
        const oldEmbeddingModel = localStorage.getItem('selectedEmbeddingModelId')
        const oldRerankModel = localStorage.getItem('selectedRerankModelId')

        if (oldEmbeddingModel || oldRerankModel) {
            console.log('📦 Migrating old default models configuration')
            return {
                embeddingModel: normalizeEmbeddingKey(oldEmbeddingModel) || INITIAL_DEFAULTS.embeddingModel,
                rerankModel: oldRerankModel || undefined
            }
        }

        // 返回初始默认配置
        return INITIAL_DEFAULTS
    })

    // 保存到localStorage
    useEffect(() => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(defaults))
    }, [defaults])

    /**
     * 设置默认模型
     */
    const setDefaultModel = (type: DefaultModelType, modelId: string | null) => {
        setDefaults(prev => ({
            ...prev,
            [type]: type === 'embeddingModel'
                ? normalizeEmbeddingKey(modelId) || INITIAL_DEFAULTS.embeddingModel
                : modelId || undefined
        }))
    }

    /**
     * 获取默认模型
     */
    const getDefaultModel = (type: DefaultModelType): string | undefined => {
        return defaults[type]
    }

    /**
     * 清除默认模型
     */
    const clearDefaultModel = (type: DefaultModelType) => {
        setDefaults(prev => ({
            ...prev,
            [type]: undefined
        }))
    }

    /**
     * 重置为初始默认配置
     */
    const resetToDefaults = () => {
        setDefaults(INITIAL_DEFAULTS)
        console.log('🔄 Reset to initial default models')
    }

    return (
        <DefaultsContext.Provider
            value={{
                defaults,
                setDefaultModel,
                getDefaultModel,
                clearDefaultModel,
                resetToDefaults
            }}
        >
            {children}
        </DefaultsContext.Provider>
    )
}

/**
 * useDefaults Hook
 */
export function useDefaults() {
    const context = useContext(DefaultsContext)
    if (!context) {
        throw new Error('useDefaults must be used within DefaultsProvider')
    }
    return context
}
