import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import type { DefaultModels, DefaultModelType } from '../types/defaults'
import { useCapabilities } from './CapabilitiesContext'

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

const DEPRECATED_EMBEDDING_MODEL_ALIASES: Record<string, string> = {
    'Qwen/Qwen-Embedding-8B': 'Qwen/Qwen3-Embedding-8B',
    'text-embedding-ada-002': 'text-embedding-3-small',
    'embo-01': 'minimax-embedding-v2',
}

const DEPRECATED_ASSISTANT_MODEL_ALIASES: Record<string, string> = {
    'gpt-5.4': 'gpt-5.2',
    'gpt-5.4-pro': 'gpt-5.2',
    'gpt-5.4-mini': 'gpt-5-mini',
    'gpt-5.4-nano': 'gpt-5-nano',
    'Doubao-Seed-1.6-lite': 'doubao-seed-2-0-lite-260215',
    'doubao-seed-1.6-lite': 'doubao-seed-2-0-lite-260215',
    'doubao-seed-2-0-pro': 'doubao-seed-2-0-pro-260215',
    'doubao-seed-2.0-pro': 'doubao-seed-2-0-pro-260215',
    'doubao-seed-2-0-lite': 'doubao-seed-2-0-lite-260215',
    'doubao-seed-2.0-lite': 'doubao-seed-2-0-lite-260215',
    'doubao-seed-2-0-mini': 'doubao-seed-2-0-mini-260215',
    'doubao-seed-2.0-mini': 'doubao-seed-2-0-mini-260215',
    'doubao-seed-2-0-code-preview-260215': 'doubao-seed-2-0-pro-260215',
    'doubao-seed-2.0-code-preview': 'doubao-seed-2-0-pro-260215',
    'doubao-seed-code-preview': 'doubao-seed-2-0-pro-260215',
    'doubao-seed-code-preview-latest': 'doubao-seed-2-0-pro-260215',
    'claude-haiku-4-5': 'claude-3-5-haiku-20241022',
    'claude-opus-4-6': 'claude-opus-4-1-20250805',
    'claude-sonnet-4-6': 'claude-sonnet-4-20250514',
    'claude-opus-4-5': 'claude-opus-4-20250514',
    'claude-sonnet-4-5': 'claude-sonnet-4-20250514',
    'claude-haiku-3-5': 'claude-3-5-haiku-20241022',
    'gemini-3.1-pro': 'gemini-3-pro-preview',
    'gemini-3.1-pro-preview': 'gemini-3-pro-preview',
    'gemini-3-flash': 'gemini-3-flash-preview',
    'kimi-k2.5': 'kimi-thinking-preview',
    'kimi-k2': 'kimi-k2-0905-preview',
    'grok-4.20': 'grok-4.20-beta-latest-non-reasoning',
    'grok-4-1-fast': 'grok-4-1-fast-reasoning',
    'MiniMax-Text-01': 'MiniMax-M2.5',
    'abab6.5s-chat': 'MiniMax-M2.1',
    'glm-4-air': 'glm-4-air-250414',
    'deepseek-ai/DeepSeek-V3': 'deepseek-ai/DeepSeek-V3.2',
    'Qwen/Qwen3-235B-A22B': 'Qwen/Qwen3-32B',
}

/**
 * 初始默认配置
 * 使用系统推荐的模型作为默认值
 */
const normalizeEmbeddingKey = (value?: string | null) => {
    if (!value) return undefined

    const mapModelId = (modelId: string) =>
        DEPRECATED_EMBEDDING_MODEL_ALIASES[modelId] || modelId

    // provider:modelId 格式
    if (value.includes(':')) {
        const [providerId, ...rest] = value.split(':')
        const modelId = rest.join(':')
        if (!modelId) return value
        return `${providerId}:${mapModelId(modelId)}`
    }

    // 旧格式只存模型ID时，默认加上 local 前缀
    return `local:${mapModelId(value)}`
}

export const normalizeAssistantKey = (value?: string | null) => {
    if (!value) return undefined

    const mapModelId = (modelId: string) =>
        DEPRECATED_ASSISTANT_MODEL_ALIASES[modelId] ||
        DEPRECATED_ASSISTANT_MODEL_ALIASES[modelId.toLowerCase()] ||
        modelId

    if (value.includes(':')) {
        const [providerId, ...rest] = value.split(':')
        const modelId = rest.join(':')
        if (!modelId) return value
        return `${providerId}:${mapModelId(modelId)}`
    }

    return mapModelId(value)
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
            migrated.assistantModel = normalizeAssistantKey(parsed.assistantModel) || INITIAL_DEFAULTS.assistantModel
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
    const { hasLocalEmbedding } = useCapabilities()

    const [defaults, setDefaults] = useState<DefaultModels>(() => {
        const savedVersion = localStorage.getItem(VERSION_KEY)
        const saved = localStorage.getItem(STORAGE_KEY)

        // 版本不匹配时尝试迁移旧数据
        if (saved && savedVersion !== CONFIG_VERSION) {
            // 尝试从旧数据迁移用户选择的默认模型
            const migrated = migrateDefaults(saved)
            if (migrated) {
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
                return {
                    ...parsed,
                    embeddingModel: normalizeEmbeddingKey(parsed.embeddingModel) || INITIAL_DEFAULTS.embeddingModel,
                    assistantModel: normalizeAssistantKey(parsed.assistantModel) || INITIAL_DEFAULTS.assistantModel,
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
            return {
                embeddingModel: normalizeEmbeddingKey(oldEmbeddingModel) || INITIAL_DEFAULTS.embeddingModel,
                assistantModel: INITIAL_DEFAULTS.assistantModel,
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
                : type === 'assistantModel'
                    ? normalizeAssistantKey(modelId) || INITIAL_DEFAULTS.assistantModel
                    : modelId || undefined
        }))
    }

    /**
     * 获取默认模型
     * 当本地 embedding 不可用时，如果默认值指向 local provider，返回 undefined
     */
    const getDefaultModel = (type: DefaultModelType): string | undefined => {
        const value = defaults[type]
        if (!hasLocalEmbedding && value?.startsWith('local:')) {
            return undefined
        }
        return value
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
