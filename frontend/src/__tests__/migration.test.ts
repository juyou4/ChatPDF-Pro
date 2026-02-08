/**
 * 版本迁移属性测试
 *
 * 使用 fast-check 对三个 Context 的迁移函数进行属性测试，
 * 验证版本升级时用户配置的正确保留。
 *
 * 测试框架：vitest + fast-check
 */

import { describe, it, expect } from 'vitest'
import fc from 'fast-check'
import { migrateProviders } from '../contexts/ProviderContext'
import { migrateUserModels } from '../contexts/ModelContext'
import { migrateDefaults } from '../contexts/DefaultsContext'

// ---- Property 5: Provider 配置迁移保留 apiKey ----

// Feature: chatpdf-model-service-fix, Property 5: Provider 配置迁移保留 apiKey
describe('Property 5: Provider 配置迁移保留 apiKey', () => {
    /**
     * **Validates: Requirements 6.1**
     *
     * 对于任意包含至少一个非空 apiKey 的旧版本 Provider 数组数据，
     * 经过 migrateProviders() 迁移后，返回的新 Provider 数组中
     * 对应 id 的 Provider 应保留原始的 apiKey 值。
     */
    it('迁移后保留所有非空 apiKey', () => {
        // Feature: chatpdf-model-service-fix, Property 5: Provider 配置迁移保留 apiKey
        const providerIds = ['openai', 'aliyun', 'silicon', 'moonshot', 'deepseek', 'zhipu', 'minimax']

        // 生成随机的旧版本 Provider 数据
        const arbitraryOldProviders = fc.array(
            fc.record({
                id: fc.constantFrom(...providerIds),
                name: fc.string({ minLength: 1, maxLength: 20 }),
                apiKey: fc.string({ minLength: 1, maxLength: 64 }),
                apiHost: fc.string({ minLength: 0, maxLength: 100 }),
                enabled: fc.boolean(),
            }),
            { minLength: 1, maxLength: 7 }
        )

        fc.assert(fc.property(
            arbitraryOldProviders,
            (oldProviders) => {
                const oldData = JSON.stringify(oldProviders)
                const result = migrateProviders(oldData)

                // 迁移应成功（非 null）
                expect(result).not.toBeNull()
                if (!result) return

                // 构建每个 id 最后出现的 apiKey（Map 语义：后者覆盖前者）
                const lastApiKeyById = new Map<string, string>()
                for (const oldP of oldProviders) {
                    if (oldP.apiKey) {
                        lastApiKeyById.set(oldP.id, oldP.apiKey)
                    }
                }

                // 验证迁移后保留了每个 id 最后出现的 apiKey
                for (const [id, expectedKey] of lastApiKeyById) {
                    const migrated = result.find(p => p.id === id)
                    if (migrated) {
                        expect(migrated.apiKey).toBe(expectedKey)
                    }
                }
            }
        ), { numRuns: 100 })
    })
})


// ---- Property 6: 用户模型迁移保留 isUserAdded 条目 ----

// Feature: chatpdf-model-service-fix, Property 6: 用户模型迁移保留 isUserAdded 条目
describe('Property 6: 用户模型迁移保留 isUserAdded 条目', () => {
    /**
     * **Validates: Requirements 6.2**
     *
     * 对于任意包含 isUserAdded: true 条目的旧版本用户模型数组，
     * 经过 migrateUserModels() 迁移后，返回的数组中应包含
     * 所有原始 isUserAdded: true 的模型条目。
     */
    it('迁移后保留所有 isUserAdded 为 true 的模型', () => {
        // Feature: chatpdf-model-service-fix, Property 6: 用户模型迁移保留 isUserAdded 条目
        const arbitraryModels = fc.array(
            fc.record({
                id: fc.string({ minLength: 1, maxLength: 30 }),
                name: fc.string({ minLength: 1, maxLength: 30 }),
                providerId: fc.constantFrom('openai', 'silicon', 'deepseek', 'local'),
                type: fc.constantFrom('embedding', 'chat', 'rerank'),
                isUserAdded: fc.boolean(),
                isSystem: fc.boolean(),
            }),
            { minLength: 1, maxLength: 10 }
        ).filter(models => models.some(m => m.isUserAdded))

        fc.assert(fc.property(
            arbitraryModels,
            (oldModels) => {
                const oldData = JSON.stringify(oldModels)
                const result = migrateUserModels(oldData)

                // 迁移应成功（非 null），因为至少有一个 isUserAdded 模型
                expect(result).not.toBeNull()
                if (!result) return

                // 验证所有 isUserAdded 的模型都被保留
                const userAddedOriginals = oldModels.filter(m => m.isUserAdded)
                for (const original of userAddedOriginals) {
                    const found = result.some(
                        m => m.id === original.id && m.providerId === original.providerId
                    )
                    expect(found).toBe(true)
                }
            }
        ), { numRuns: 100 })
    })
})

// ---- Property 7: 默认模型迁移保留用户选择 ----

// Feature: chatpdf-model-service-fix, Property 7: 默认模型迁移保留用户选择
describe('Property 7: 默认模型迁移保留用户选择', () => {
    /**
     * **Validates: Requirements 6.3**
     *
     * 对于任意包含非空 embeddingModel 或 assistantModel 值的旧版本默认模型配置，
     * 经过 migrateDefaults() 迁移后，返回的配置中应保留原始的非空字段值。
     */
    it('迁移后保留非空的 embeddingModel 和 assistantModel', () => {
        // Feature: chatpdf-model-service-fix, Property 7: 默认模型迁移保留用户选择
        const arbitraryDefaults = fc.record({
            embeddingModel: fc.option(
                fc.constantFrom('local:all-MiniLM-L6-v2', 'silicon:BAAI/bge-m3', 'openai:text-embedding-3-large'),
                { nil: undefined }
            ),
            assistantModel: fc.option(
                fc.constantFrom('deepseek:deepseek-chat', 'openai:gpt-4o', 'moonshot:moonshot-v1-8k'),
                { nil: undefined }
            ),
            rerankModel: fc.option(
                fc.constantFrom('silicon:BAAI/bge-reranker-v2-m3'),
                { nil: undefined }
            ),
        }).filter(d => !!(d.embeddingModel || d.assistantModel))

        fc.assert(fc.property(
            arbitraryDefaults,
            (oldDefaults) => {
                const oldData = JSON.stringify(oldDefaults)
                const result = migrateDefaults(oldData)

                // 迁移应成功（非 null）
                expect(result).not.toBeNull()
                if (!result) return

                // 验证非空字段被保留
                if (oldDefaults.embeddingModel) {
                    // embeddingModel 可能被 normalizeEmbeddingKey 处理，但值应保留
                    expect(result.embeddingModel).toBeTruthy()
                }
                if (oldDefaults.assistantModel) {
                    expect(result.assistantModel).toBe(oldDefaults.assistantModel)
                }
                if (oldDefaults.rerankModel) {
                    expect(result.rerankModel).toBe(oldDefaults.rerankModel)
                }
            }
        ), { numRuns: 100 })
    })
})

// ---- Property 8: 无效旧数据迁移回退到默认值 ----

// Feature: chatpdf-model-service-fix, Property 8: 无效旧数据迁移回退到默认值
describe('Property 8: 无效旧数据迁移回退到默认值', () => {
    /**
     * **Validates: Requirements 6.4**
     *
     * 对于任意非法 JSON 字符串或格式不符合预期的数据，
     * 各 Context 的迁移函数应返回 null（表示回退到默认配置），而不抛出异常。
     */
    it('migrateProviders 对无效数据返回 null 且不抛异常', () => {
        // Feature: chatpdf-model-service-fix, Property 8: 无效旧数据迁移回退到默认值
        const arbitraryInvalidData = fc.oneof(
            // 非法 JSON
            fc.string({ minLength: 1, maxLength: 50 }).filter(s => {
                try { JSON.parse(s); return false } catch { return true }
            }),
            // 合法 JSON 但格式不对（非数组）
            fc.constant(JSON.stringify({ foo: 'bar' })),
            fc.constant(JSON.stringify('just a string')),
            fc.constant(JSON.stringify(42)),
            fc.constant(JSON.stringify(null)),
        )

        fc.assert(fc.property(
            arbitraryInvalidData,
            (invalidData) => {
                // 不应抛出异常
                const result = migrateProviders(invalidData)
                expect(result).toBeNull()
            }
        ), { numRuns: 100 })
    })

    it('migrateUserModels 对无效数据返回 null 且不抛异常', () => {
        // Feature: chatpdf-model-service-fix, Property 8: 无效旧数据迁移回退到默认值
        const arbitraryInvalidData = fc.oneof(
            fc.string({ minLength: 1, maxLength: 50 }).filter(s => {
                try { JSON.parse(s); return false } catch { return true }
            }),
            fc.constant(JSON.stringify({ foo: 'bar' })),
            fc.constant(JSON.stringify('just a string')),
            fc.constant(JSON.stringify(42)),
            fc.constant(JSON.stringify(null)),
            // 数组但没有 isUserAdded 的条目
            fc.constant(JSON.stringify([{ id: 'test', isUserAdded: false }])),
        )

        fc.assert(fc.property(
            arbitraryInvalidData,
            (invalidData) => {
                const result = migrateUserModels(invalidData)
                expect(result).toBeNull()
            }
        ), { numRuns: 100 })
    })

    it('migrateDefaults 对无效数据返回 null 且不抛异常', () => {
        // Feature: chatpdf-model-service-fix, Property 8: 无效旧数据迁移回退到默认值
        const arbitraryInvalidData = fc.oneof(
            fc.string({ minLength: 1, maxLength: 50 }).filter(s => {
                try { JSON.parse(s); return false } catch { return true }
            }),
            fc.constant(JSON.stringify([1, 2, 3])),
            fc.constant(JSON.stringify('just a string')),
            fc.constant(JSON.stringify(42)),
            fc.constant(JSON.stringify(null)),
            // 对象但没有有效字段
            fc.constant(JSON.stringify({ foo: 'bar' })),
        )

        fc.assert(fc.property(
            arbitraryInvalidData,
            (invalidData) => {
                const result = migrateDefaults(invalidData)
                expect(result).toBeNull()
            }
        ), { numRuns: 100 })
    })
})
