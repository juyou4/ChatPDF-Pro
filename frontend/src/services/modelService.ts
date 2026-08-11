/**
 * Model服务层
 * 负责从provider API获取模型列表，并解析为统一的Model格式
 */

import type { Model, ModelType, ModelCapability, FetchModelsResponse } from '../types/model'
import type { Provider } from '../types/provider'

/**
 * 从Provider API获取模型列表
 * 通过后端代理调用，避免CORS问题
 */
export async function fetchModelsFromProvider(
    provider: Provider
): Promise<Model[]> {
    try {
        // 调用后端API，由后端代理请求provider的模型列表
        const response = await fetch('/api/models/fetch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                providerId: provider.id,
                apiKey: provider.apiKey,
                apiHost: provider.apiHost,
                fetchModelsEndpoint: provider.apiConfig?.fetchModelsEndpoint,
                providerType: provider.apiConfig?.protocol || provider.id,
                apiKeyHeader: provider.apiConfig?.apiKeyHeader,
                apiKeyPrefix: provider.apiConfig?.apiKeyPrefix,
            })
        })

        if (!response.ok) {
            throw new Error(`Failed to fetch models: ${response.statusText}`)
        }

        const data: FetchModelsResponse = await response.json()
        return data.models
    } catch (error) {
        console.error(`Error fetching models from ${provider.name}:`, error)
        throw error
    }
}

/**
 * 解析OpenAI格式的模型响应
 * OpenAI API返回格式：{ data: [{ id, object, created, owned_by }] }
 */
export function parseOpenAIModels(
    data: any,
    providerId: string
): Model[] {
    if (!data.data || !Array.isArray(data.data)) {
        return []
    }

    return data.data.map((item: any) => {
        const modelId = item.id
        const modelType = detectModelType(modelId)
        const modelTags = detectModelTags(modelId, modelType)

        return {
            id: modelId,
            name: modelId,
            providerId,
            type: modelType,
            tags: modelTags.length > 0 ? modelTags : undefined,
            metadata: {
                description: item.owned_by ? `Owned by: ${item.owned_by}` : undefined
            },
            isSystem: false,
            isUserAdded: false
        }
    })
}

/**
 * 根据模型 ID 与类型推断标签（embedding / rerank / vision / reasoning）
 * 用于API动态获取的模型自动分类，覆盖主流商业模型命名规律
 */
export function detectModelTags(modelId: string, modelType?: ModelType): string[] {
    const tags: string[] = []
    const id = modelId.toLowerCase()

    if (modelType === 'embedding') tags.push('embedding')
    if (modelType === 'rerank') tags.push('rerank')

    // 向量/重排模型的名称可能包含 Gemini、Qwen、GPT 等聊天系列前缀，
    // 不能因此被标为视觉或推理模型。
    if (modelType === 'embedding' || modelType === 'rerank') return tags

    // 视觉能力检测
    const hasVision =
        /vision|visual|\bvl\b|vl-|vlm|multimodal/.test(id) ||
        /gpt-4o|gpt-4\.1|gpt-5/.test(id) ||
        /\bo[34]\b/.test(id) ||
        /claude-(3|opus|sonnet|haiku|fable|mythos)/.test(id) ||
        /gemini/.test(id) ||
        /grok-[34]/.test(id) ||
        /grok-build/.test(id) ||
        /kimi-(latest|k2|k3)/.test(id) ||
        /doubao-seed/.test(id) ||
        /moonshot-v1/.test(id) ||
        /qwen3\.[567]/.test(id) ||
        /minimax-m3/.test(id) ||
        /^mimo-v2\.5$/.test(id)
    if (hasVision) tags.push('vision')

    // 推理能力检测（含思维链/深度推理）
    const hasReasoning =
        /reasoning|reasoner/.test(id) ||
        /\bthink(ing)?\b/.test(id) ||
        /\bqwq\b|\bqvq\b/.test(id) ||
        /deepseek-r\d|deepseek-reasoner|deepseek-v4/.test(id) ||
        /\bo[1-9](-mini|-preview|-pro)?\b/.test(id) ||
        /gemini-2\.5|gemini-3/.test(id) ||
        /grok-[34]/.test(id) ||
        /grok-build/.test(id) ||
        /claude-(fable|mythos|opus|sonnet|haiku)-(4|5)/.test(id) ||
        /glm-[45]\.\d|glm-5/.test(id) ||
        /doubao-seed-2/.test(id) ||
        /kimi-thinking-preview/.test(id) ||
        /kimi-k(?:2|3)(\.\d+)?|k2\.\d+/.test(id) ||
        /qwen3(\.\d+)?/.test(id) ||
        /minimax-m[23]/.test(id) ||
        /^mimo-v2\.5(?:-pro)?$/.test(id)
    if (hasReasoning) tags.push('reasoning')

    return tags
}

/**
 * 根据模型ID推断模型类型
 * 正则与后端 model_detector.py 保持一致
 */
export function detectModelType(modelId: string): ModelType {
    const lowerModelId = modelId.toLowerCase()

    // Rerank模型识别（优先级最高，避免 retrieval 关键字误分类）
    if (/rerank|re-rank|re-ranker|re-ranking|retrieval|retriever/i.test(lowerModelId)) {
        return 'rerank'
    }

    // Embedding模型识别（与后端 EMBEDDING_REGEX 对齐）
    if (/(?:^text-|embed|embo-|bge-|e5-|LLM2Vec|retrieval|uae-|gte-|jina-clip|jina-embeddings|voyage-|minilm|qwen.*embedding)/i.test(lowerModelId)) {
        return 'embedding'
    }

    // 图像生成模型识别
    if (/dall-e|dalle|stable-diffusion|midjourney|imagen|diffusion|sd/i.test(lowerModelId)) {
        return 'image'
    }

    // 默认为chat模型
    return 'chat'
}

/**
 * 带 capabilities 优先级的模型类型检测
 *
 * 优先级逻辑：
 * 1. 优先检查 capabilities 中 isUserSelected=true 的条目（用户覆盖）
 * 2. 若无用户覆盖，回退到基于正则的 detectModelType 检测
 *
 * 与后端 get_model_type_with_capabilities 保持一致的优先级策略
 */
export function detectModelTypeWithCapabilities(
    modelId: string,
    capabilities?: ModelCapability[]
): ModelType {
    // 优先检查用户覆盖（isUserSelected=true 的条目）
    if (capabilities) {
        const userSelected = capabilities.find(c => c.isUserSelected === true)
        if (userSelected) return userSelected.type
    }
    // 回退到正则检测
    return detectModelType(modelId)
}

/**
 * 根据模型ID推断元数据（维度、最大token等）
 * 基于常见模型的已知信息
 */
export function inferModelMetadata(modelId: string, type: ModelType) {
    const metadata: any = {}

    if (type === 'embedding') {
        // OpenAI
        if (modelId.includes('text-embedding-3-large')) {
            metadata.dimension = 3072
            metadata.maxTokens = 8191
        } else if (modelId.includes('text-embedding-3-small')) {
            metadata.dimension = 1536
            metadata.maxTokens = 8191
        } else if (modelId.includes('text-embedding-ada-002')) {
            metadata.dimension = 1536
            metadata.maxTokens = 8191
        }
        // BAAI
        else if (modelId.includes('bge-large')) {
            metadata.dimension = 1024
            metadata.maxTokens = 512
        } else if (modelId.includes('bge-m3')) {
            metadata.dimension = 1024
            metadata.maxTokens = 8192
        }
        // 默认值
        else {
            metadata.dimension = 1024
            metadata.maxTokens = 512
        }
    }

    if (type === 'chat') {
        // GPT-4
        if (modelId.includes('gpt-4')) {
            metadata.contextWindow = modelId.includes('32k') ? 32768 : 8192
        }
        // GPT-3.5
        else if (modelId.includes('gpt-3.5')) {
            metadata.contextWindow = modelId.includes('16k') ? 16384 : 4096
        }
        // 默认值
        else {
            metadata.contextWindow = 4096
        }
    }

    return metadata
}

/**
 * 过滤模型列表
 * 根据类型、provider等条件筛选
 */
export function filterModels(
    models: Model[],
    options: {
        type?: ModelType
        providerId?: string
        isUserAdded?: boolean
        searchQuery?: string
    }
): Model[] {
    let filtered = models

    if (options.type) {
        filtered = filtered.filter(m => m.type === options.type)
    }

    if (options.providerId) {
        filtered = filtered.filter(m => m.providerId === options.providerId)
    }

    if (options.isUserAdded !== undefined) {
        filtered = filtered.filter(m => m.isUserAdded === options.isUserAdded)
    }

    if (options.searchQuery) {
        const query = options.searchQuery.toLowerCase()
        filtered = filtered.filter(m =>
            m.id.toLowerCase().includes(query) ||
            m.name.toLowerCase().includes(query) ||
            m.metadata.description?.toLowerCase().includes(query)
        )
    }

    return filtered
}

/**
 * 合并系统模型和用户模型
 * 用户添加的模型优先级更高（可能有自定义配置）
 */
export function mergeModels(
    systemModels: Model[],
    userModels: Model[]
): Model[] {
    const modelMap = new Map<string, Model>()

    // 先添加系统模型
    systemModels.forEach(model => {
        const key = `${model.providerId}:${model.id}`
        modelMap.set(key, model)
    })

    // 用户模型覆盖同名的系统模型，但保留系统模型的 type（防止误操作改变模型类型）
    userModels.forEach(model => {
        const key = `${model.providerId}:${model.id}`
        const existing = modelMap.get(key)
        if (existing && existing.isSystem) {
            modelMap.set(key, { ...existing, ...model, type: existing.type })
        } else {
            modelMap.set(key, model)
        }
    })

    return Array.from(modelMap.values())
}

/**
 * 按类型分组模型
 */
export function groupModelsByType(models: Model[]): Record<ModelType, Model[]> {
    const grouped: Record<string, Model[]> = {
        chat: [],
        embedding: [],
        rerank: [],
        image: []
    }

    models.forEach(model => {
        if (grouped[model.type]) {
            grouped[model.type].push(model)
        }
    })

    return grouped as Record<ModelType, Model[]>
}

/**
 * 按provider分组模型
 */
export function groupModelsByProvider(models: Model[]): Record<string, Model[]> {
    const grouped: Record<string, Model[]> = {}

    models.forEach(model => {
        if (!grouped[model.providerId]) {
            grouped[model.providerId] = []
        }
        grouped[model.providerId].push(model)
    })

    return grouped
}
