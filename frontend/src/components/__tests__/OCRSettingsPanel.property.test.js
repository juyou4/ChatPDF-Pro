/**
 * OCR 设置持久化 round-trip 属性测试
 *
 * Feature: chatpdf-ocr-enhancement, Property 7: OCR 设置持久化 round-trip
 * **Validates: Requirements 6.3**
 *
 * 对于任意有效的 OCR 模式值，将其写入 localStorage 后再读取，
 * 得到的值应与写入的值相同。
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import * as fc from 'fast-check'
import { loadOCRSettings, saveOCRSettings } from '../OCRSettingsPanel.jsx'

// 内存模拟 localStorage 实现
function createMockLocalStorage() {
  const store = new Map()
  return {
    getItem: vi.fn((key) => store.get(key) ?? null),
    setItem: vi.fn((key, value) => store.set(key, String(value))),
    removeItem: vi.fn((key) => store.delete(key)),
    clear: vi.fn(() => store.clear()),
    get length() {
      return store.size
    },
    key: vi.fn((index) => [...store.keys()][index] ?? null),
  }
}

describe('Property 7: OCR 设置持久化 round-trip', () => {
  // Feature: chatpdf-ocr-enhancement, Property 7: OCR 设置持久化 round-trip
  let mockStorage

  beforeEach(() => {
    // 每次测试前创建新的 mock localStorage
    mockStorage = createMockLocalStorage()
    globalThis.localStorage = mockStorage
  })

  afterEach(() => {
    // 清理全局 localStorage mock
    delete globalThis.localStorage
  })

  /**
   * 有效 OCR 模式值的生成器
   * 仅生成 'auto'、'always'、'never' 三种合法值
   */
  const validOCRModeArb = fc.oneof(
    fc.constant('auto'),
    fc.constant('always'),
    fc.constant('never')
  )

  it('对于任意有效 OCR 模式，saveOCRSettings 后 loadOCRSettings 应返回相同的模式值', () => {
    // Feature: chatpdf-ocr-enhancement, Property 7: OCR 设置持久化 round-trip
    // **Validates: Requirements 6.3**
    fc.assert(
      fc.property(validOCRModeArb, (mode) => {
        // 清空存储，确保每次迭代独立
        mockStorage.clear()

        // 写入设置
        const settings = { mode }
        saveOCRSettings(settings)

        // 读取设置
        const loaded = loadOCRSettings()

        // 验证 round-trip 一致性
        expect(loaded.mode).toBe(mode)
      }),
      { numRuns: 100 }
    )
  })

  it('未保存任何设置时，loadOCRSettings 应返回默认值 { mode: "auto" }', () => {
    // Feature: chatpdf-ocr-enhancement, Property 7: OCR 设置持久化 round-trip
    // **Validates: Requirements 6.3**
    const loaded = loadOCRSettings()
    expect(loaded).toEqual({ mode: 'auto' })
  })

  it('对于任意有效模式，多次连续写入后读取应返回最后写入的值', () => {
    // Feature: chatpdf-ocr-enhancement, Property 7: OCR 设置持久化 round-trip
    // **Validates: Requirements 6.3**
    fc.assert(
      fc.property(
        fc.array(validOCRModeArb, { minLength: 1, maxLength: 10 }),
        (modes) => {
          mockStorage.clear()

          // 依次写入多个模式值
          for (const mode of modes) {
            saveOCRSettings({ mode })
          }

          // 读取应返回最后一次写入的值
          const loaded = loadOCRSettings()
          const lastMode = modes[modes.length - 1]
          expect(loaded.mode).toBe(lastMode)
        }
      ),
      { numRuns: 100 }
    )
  })
})
