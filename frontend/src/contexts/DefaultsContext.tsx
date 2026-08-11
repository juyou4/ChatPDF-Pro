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

const CONFIG_VERSION = '4.4'
const STORAGE_KEY = 'defaultModels'
const VERSION_KEY = 'defaultModelsVersion'

const DEPRECATED_EMBEDDING_MODEL_ALIASES: Record<string, string> = {
    'Qwen/Qwen-Embedding-8B': 'Qwen/Qwen3-Embedding-8B',
    'text-embedding-ada-002': 'text-embedding-3-small',
    'text-embedding-v3': 'text-embedding-v4',
    'minimax-embedding-v2': 'embo-01',
}

const RETIRED_EMBEDDING_KEYS = new Set([
    'moonshot:moonshot-embedding-v1',
    'deepseek:deepseek-embedding-v1',
    'moonshot-embedding-v1',
    'deepseek-embedding-v1',
])

const DEPRECATED_ASSISTANT_MODEL_ALIASES: Record<string, string> = {
    'gpt-5.5': 'gpt-5.6-terra',
    'gpt-5.5-pro': 'gpt-5.6-sol',
    'gpt-5.4': 'gpt-5.6-terra',
    'gpt-5.4-pro': 'gpt-5.6-sol',
    'gpt-5.3-codex': 'gpt-5.6-terra',
    'gpt-5.1-codex-mini': 'gpt-5.6-luna',
    'gpt-5.2': 'gpt-5.6-terra',
    'gpt-5.1': 'gpt-5.6-terra',
    'gpt-5-mini': 'gpt-5.6-luna',
    'gpt-5-nano': 'gpt-5.6-luna',
    'Doubao-Seed-1.6-lite': 'doubao-seed-evolving',
    'doubao-seed-1.6-lite': 'doubao-seed-evolving',
    'doubao-seed-2-0-pro': 'doubao-seed-evolving',
    'doubao-seed-2.0-pro': 'doubao-seed-evolving',
    'doubao-seed-2-0-pro-260215': 'doubao-seed-evolving',
    'doubao-seed-2-0-lite': 'doubao-seed-evolving',
    'doubao-seed-2.0-lite': 'doubao-seed-evolving',
    'doubao-seed-2-0-lite-260215': 'doubao-seed-evolving',
    'doubao-seed-2-0-mini': 'doubao-seed-evolving',
    'doubao-seed-2.0-mini': 'doubao-seed-evolving',
    'doubao-seed-2-0-mini-260215': 'doubao-seed-evolving',
    'doubao-seed-2-0-code-preview-260215': 'doubao-seed-evolving',
    'doubao-seed-2.0-code-preview': 'doubao-seed-evolving',
    'doubao-seed-code-preview': 'doubao-seed-evolving',
    'doubao-seed-code-preview-latest': 'doubao-seed-evolving',
    'doubao-seed-2-1-pro': 'doubao-seed-evolving',
    'doubao-seed-2.1-pro': 'doubao-seed-evolving',
    'doubao-seed-2-1-turbo': 'doubao-seed-evolving',
    'doubao-seed-2.1-turbo': 'doubao-seed-evolving',
    'claude-fable-5-20260701': 'claude-fable-5',
    'claude-opus-4-8': 'claude-opus-5',
    'claude-opus-4-8-20260219': 'claude-opus-5',
    'claude-sonnet-5-20260614': 'claude-sonnet-5',
    'claude-haiku-4-5': 'claude-haiku-4-5-20251001',
    'claude-opus-4-6': 'claude-opus-5',
    'claude-sonnet-4-6': 'claude-sonnet-5',
    'claude-opus-4-5': 'claude-opus-5',
    'claude-sonnet-4-5': 'claude-sonnet-5',
    'claude-opus-4-1-20250805': 'claude-opus-5',
    'claude-opus-4-20250514': 'claude-opus-5',
    'claude-sonnet-4-20250514': 'claude-sonnet-5',
    'claude-3-7-sonnet-20250219': 'claude-sonnet-5',
    'claude-haiku-3-5': 'claude-haiku-4-5-20251001',
    'claude-3-5-haiku-20241022': 'claude-haiku-4-5-20251001',
    'gemini-3-pro-preview': 'gemini-3.1-pro-preview',
    'gemini-3.1-pro': 'gemini-3.1-pro-preview',
    'gemini-3-flash': 'gemini-3.6-flash',
    'gemini-3-flash-preview': 'gemini-3.6-flash',
    'kimi-latest': 'kimi-k3',
    'kimi-thinking-preview': 'kimi-k3',
    'kimi-k2-0905-preview': 'kimi-k2.6',
    'kimi-k2-turbo-preview': 'kimi-k2.6',
    'kimi-k2': 'kimi-k3',
    'grok-4.20': 'grok-4.5',
    'grok-4.20-beta-latest-non-reasoning': 'grok-4.5',
    'grok-4-1-fast': 'grok-4.5',
    'grok-4-1-fast-reasoning': 'grok-4.5',
    'MiniMax-Text-01': 'MiniMax-M3',
    'minimax-text-01': 'MiniMax-M3',
    'MiniMax-M2.5': 'MiniMax-M3',
    'minimax-m2.5': 'MiniMax-M3',
    'abab6.5s-chat': 'MiniMax-M3',
    'glm-5': 'glm-5.2',
    'glm-4.7': 'glm-5.2',
    'glm-4.6': 'glm-5.2',
    'glm-4.5': 'glm-5.2',
    'glm-4-air': 'glm-5.2',
    'qwen3-max': 'qwen3.7-max',
    'qwen3.5-plus': 'qwen3.7-plus',
    'qwen3.5-flash': 'qwen3.6-flash',
    'qwen-max': 'qwen3.7-max',
    'qwen-plus': 'qwen3.7-plus',
    'deepseek-chat': 'deepseek-v4-flash',
    'deepseek-reasoner': 'deepseek-v4-flash',
    'deepseek-v3.2': 'deepseek-v4-flash',
    'deepseek-ai/DeepSeek-V3': 'deepseek-ai/DeepSeek-V4-Flash',
    'deepseek-ai/DeepSeek-V3.2': 'deepseek-ai/DeepSeek-V4-Flash',
    'deepseek-ai/deepseek-v3': 'deepseek-ai/DeepSeek-V4-Flash',
    'deepseek-ai/deepseek-v3.2': 'deepseek-ai/DeepSeek-V4-Flash',
    'Qwen/Qwen3-235B-A22B': 'Qwen/Qwen3-32B',
    'qwen/qwen3-235b-a22b': 'Qwen/Qwen3-32B',
    'kimi-k2.5': 'kimi-k2.6',
    'kimi-k2.7-code': 'kimi-k2.7-code',
    'kimi-k2.7-code-highspeed': 'kimi-k2.7-code',
    'moonshot-v1-vision-preview': 'kimi-k2.6',
    'moonshot-v1-128k': 'kimi-k2.6',
    'moonshot-v1-32k': 'kimi-k2.6',
    'moonshot-v1-8k': 'kimi-k2.6',
    'glm-4.5-air': 'glm-5.2',
    'glm-4-air-250414': 'glm-5.2',
    'MiniMax-M2.7': 'MiniMax-M3',
    'MiniMax-M2.7-highspeed': 'MiniMax-M3',
    'MiniMax-M2.1': 'MiniMax-M3',
    'MiniMax-M2.1-highspeed': 'MiniMax-M3',
    'MiniMax-M2': 'MiniMax-M3',
    'grok-4': 'grok-4.5',
    'grok-3': 'grok-4.5',
    'doubao-seed-2-1-pro-260628': 'doubao-seed-evolving',
    'doubao-seed-2-1-turbo-260628': 'doubao-seed-evolving',
    'doubao-seed-2-0-lite-260428': 'doubao-seed-evolving',
    'doubao-seed-2-0-mini-260428': 'doubao-seed-evolving',
    'doubao-seed-1-8': 'doubao-seed-evolving',
    'doubao-1-5-pro-32k-250115': 'doubao-seed-evolving',
}

/**
 * 初始默认配置
 * 使用系统推荐的模型作为默认值
 */
export const normalizeEmbeddingKey = (value?: string | null) => {
    if (!value) return undefined

    const normalizedValue = value.trim()
    if (RETIRED_EMBEDDING_KEYS.has(normalizedValue.toLowerCase())) {
        return 'local:all-MiniLM-L6-v2'
    }

    const mapModelId = (modelId: string) =>
        DEPRECATED_EMBEDDING_MODEL_ALIASES[modelId] ||
        DEPRECATED_EMBEDDING_MODEL_ALIASES[modelId.toLowerCase()] ||
        modelId

    // provider:modelId 格式
    if (normalizedValue.includes(':')) {
        const [providerId, ...rest] = normalizedValue.split(':')
        const modelId = rest.join(':')
        if (!modelId) return normalizedValue
        if (RETIRED_EMBEDDING_KEYS.has(`${providerId}:${modelId}`.toLowerCase())) {
            return 'local:all-MiniLM-L6-v2'
        }
        return `${providerId}:${mapModelId(modelId)}`
    }

    // 旧格式只存模型ID时，默认加上 local 前缀
    return `local:${mapModelId(normalizedValue)}`
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
    assistantModel: 'deepseek:deepseek-v4-pro',  // DeepSeek 官方快速开始推荐的默认 Chat 模型
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
