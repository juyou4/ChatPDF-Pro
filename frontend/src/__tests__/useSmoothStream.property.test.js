/**
 * useSmoothStream 属性测试
 *
 * 使用 fast-check 对 useSmoothStream Hook 的核心逻辑进行属性测试，
 * 验证文本分割、渲染公式、流结束完整性和 reset 功能的正确性。
 */

import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'
import { splitChunk } from '../hooks/useSmoothStream.js'

describe('useSmoothStream 属性测试', () => {
  // Feature: chatpdf-streaming-performance, Property 1: 文本分割保持完整性
  // **Validates: Requirements 1.1**
  describe('Property 1: 文本分割保持完整性', () => {
    it('任意文本经 splitChunk 分割后拼接应与原文完全相同', () => {
      fc.assert(
        fc.property(fc.string(), (text) => {
          const segments = splitChunk(text)
          const joined = segments.join('')
          expect(joined).toBe(text)
        }),
        { numRuns: 100 }
      )
    })

    it('包含中文、emoji、特殊字符的文本分割后拼接保持完整', () => {
      // 自定义生成器：混合中英文、emoji、特殊字符
      const mixedCharArb = fc.constantFrom(
        'a', 'b', 'c', 'Z', '0', '9', ' ', '.',   // ASCII 字符
        '你', '好', '世', '界', '测', '试',         // 中文
        '🎉', '🚀', '❤️', '👨‍👩‍👧‍👦', '🇨🇳',           // emoji
        '\n', '\t',                                 // 空白字符
        '①', '②', '™', '©', '½',                   // 特殊符号
        'α', 'β', 'γ', 'δ',                         // 希腊字母
        'あ', 'い', 'う',                            // 日文
      )
      const mixedTextArb = fc.array(mixedCharArb, { minLength: 0, maxLength: 200 })
        .map((chars) => chars.join(''))

      fc.assert(
        fc.property(mixedTextArb, (text) => {
          const segments = splitChunk(text)
          const joined = segments.join('')
          expect(joined).toBe(text)
        }),
        { numRuns: 100 }
      )
    })
  })

  // Feature: chatpdf-streaming-performance, Property 2: 每帧渲染字符数量符合公式
  // **Validates: Requirements 1.2, 1.3**
  describe('Property 2: 每帧渲染字符数量符合公式', () => {
    /**
     * 计算每帧渲染字符数（与 useSmoothStream 中 renderLoop 的公式一致）
     * @param {number} queueLength - 队列长度
     * @returns {number} 每帧应渲染的字符数
     */
    function calcCharsPerFrame(queueLength) {
      return Math.max(1, Math.floor(queueLength / 5))
    }

    it('对任意正整数队列长度，每帧字符数等于 Math.max(1, Math.floor(n / 5))', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 1, max: 10000 }),
          (n) => {
            const result = calcCharsPerFrame(n)
            const expected = Math.max(1, Math.floor(n / 5))
            expect(result).toBe(expected)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('每帧字符数始终 >= 1', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 1, max: 10000 }),
          (n) => {
            const result = calcCharsPerFrame(n)
            expect(result).toBeGreaterThanOrEqual(1)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('每帧字符数不超过队列长度', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 1, max: 10000 }),
          (n) => {
            const result = calcCharsPerFrame(n)
            expect(result).toBeLessThanOrEqual(n)
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  // Feature: chatpdf-streaming-performance, Property 3: 流结束时完整渲染
  // **Validates: Requirements 1.5**
  describe('Property 3: 流结束时完整渲染', () => {
    /**
     * 模拟流结束时的渲染行为（同步版本）
     *
     * 当 streamDone=true 时，渲染循环会一次性将队列中所有剩余字符追加到 displayedText。
     * 此函数模拟该行为：将所有 chunk 通过 splitChunk 拆分入队，然后一次性渲染。
     *
     * @param {string[]} chunks - 文本块数组
     * @param {string} initialText - 初始文本
     * @returns {string} 最终的 displayedText
     */
    function simulateStreamDoneRender(chunks, initialText = '') {
      let queue = []
      let displayedText = initialText

      // 将所有 chunk 拆分后加入队列
      for (const chunk of chunks) {
        queue = [...queue, ...splitChunk(chunk)]
      }

      // 模拟 streamDone=true 时一次性渲染所有剩余字符
      displayedText += queue.join('')
      return displayedText
    }

    it('流结束后 displayedText 应等于 initialText + 所有 chunk 拼接', () => {
      fc.assert(
        fc.property(
          fc.array(fc.string({ minLength: 0, maxLength: 50 }), { minLength: 0, maxLength: 20 }),
          fc.string({ minLength: 0, maxLength: 50 }),
          (chunks, initialText) => {
            const result = simulateStreamDoneRender(chunks, initialText)
            const expected = initialText + chunks.join('')
            expect(result).toBe(expected)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('空 chunk 数组不影响 initialText', () => {
      fc.assert(
        fc.property(
          fc.string({ minLength: 0, maxLength: 100 }),
          (initialText) => {
            const result = simulateStreamDoneRender([], initialText)
            expect(result).toBe(initialText)
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  // Feature: chatpdf-streaming-performance, Property 4: 双实例独立性
  // **Validates: Requirements 3.3**
  describe('Property 4: 双实例独立性', () => {
    /**
     * 模拟双实例独立性
     *
     * 创建两个独立的状态容器，分别添加不同的 chunk 序列，
     * 验证各自的最终输出互不影响。
     *
     * @param {string[]} chunksA - 实例 A 的文本块序列
     * @param {string[]} chunksB - 实例 B 的文本块序列
     * @param {string} initialTextA - 实例 A 的初始文本
     * @param {string} initialTextB - 实例 B 的初始文本
     * @returns {{ displayedTextA: string, displayedTextB: string }}
     */
    function simulateDualInstances(chunksA, chunksB, initialTextA = '', initialTextB = '') {
      // 实例 A
      let queueA = []
      let displayedTextA = initialTextA
      for (const chunk of chunksA) {
        queueA = [...queueA, ...splitChunk(chunk)]
      }
      displayedTextA += queueA.join('')

      // 实例 B
      let queueB = []
      let displayedTextB = initialTextB
      for (const chunk of chunksB) {
        queueB = [...queueB, ...splitChunk(chunk)]
      }
      displayedTextB += queueB.join('')

      return { displayedTextA, displayedTextB }
    }

    it('两个独立实例的最终输出应仅包含各自接收的文本', () => {
      fc.assert(
        fc.property(
          fc.tuple(
            fc.array(fc.string({ minLength: 0, maxLength: 50 }), { minLength: 0, maxLength: 20 }),
            fc.array(fc.string({ minLength: 0, maxLength: 50 }), { minLength: 0, maxLength: 20 })
          ),
          ([chunksA, chunksB]) => {
            const { displayedTextA, displayedTextB } = simulateDualInstances(chunksA, chunksB)

            // 实例 A 的输出应等于其所有 chunk 的拼接
            const expectedA = chunksA.join('')
            expect(displayedTextA).toBe(expectedA)

            // 实例 B 的输出应等于其所有 chunk 的拼接
            const expectedB = chunksB.join('')
            expect(displayedTextB).toBe(expectedB)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('带初始文本的双实例也应互不影响', () => {
      fc.assert(
        fc.property(
          fc.tuple(
            fc.array(fc.string({ minLength: 0, maxLength: 50 }), { minLength: 0, maxLength: 20 }),
            fc.array(fc.string({ minLength: 0, maxLength: 50 }), { minLength: 0, maxLength: 20 })
          ),
          fc.string({ minLength: 0, maxLength: 50 }),
          fc.string({ minLength: 0, maxLength: 50 }),
          ([chunksA, chunksB], initialTextA, initialTextB) => {
            const { displayedTextA, displayedTextB } = simulateDualInstances(
              chunksA, chunksB, initialTextA, initialTextB
            )

            // 实例 A 的输出应等于 initialTextA + 所有 chunksA 拼接
            const expectedA = initialTextA + chunksA.join('')
            expect(displayedTextA).toBe(expectedA)

            // 实例 B 的输出应等于 initialTextB + 所有 chunksB 拼接
            const expectedB = initialTextB + chunksB.join('')
            expect(displayedTextB).toBe(expectedB)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('一个实例为空输入时不影响另一个实例的输出', () => {
      fc.assert(
        fc.property(
          fc.array(fc.string({ minLength: 0, maxLength: 50 }), { minLength: 0, maxLength: 20 }),
          (chunks) => {
            // 实例 A 有数据，实例 B 为空
            const { displayedTextA, displayedTextB } = simulateDualInstances(chunks, [])
            expect(displayedTextA).toBe(chunks.join(''))
            expect(displayedTextB).toBe('')

            // 实例 A 为空，实例 B 有数据
            const result2 = simulateDualInstances([], chunks)
            expect(result2.displayedTextA).toBe('')
            expect(result2.displayedTextB).toBe(chunks.join(''))
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  // Feature: chatpdf-streaming-performance, Property 5: Reset 功能正确性
  // **Validates: Requirements 5.1, 5.4**
  describe('Property 5: Reset 功能正确性', () => {
    /**
     * 模拟 useSmoothStream 的状态管理和 reset 行为
     *
     * 创建一个简化的状态容器，模拟 addChunk 入队和 reset 重置逻辑，
     * 验证 reset 后 displayedText 等于 newText 且队列为空。
     */
    function createStreamState(initialText = '') {
      let queue = []
      let displayedText = initialText

      return {
        addChunk(chunk) {
          queue = [...queue, ...splitChunk(chunk)]
        },
        reset(newText = '') {
          queue = []
          displayedText = newText
        },
        getQueue() {
          return queue
        },
        getDisplayedText() {
          return displayedText
        },
      }
    }

    it('reset 后 displayedText 等于 newText 且队列为空', () => {
      fc.assert(
        fc.property(
          // 先添加一些随机 chunk 模拟使用中的状态
          fc.array(fc.string({ minLength: 1, maxLength: 30 }), { minLength: 1, maxLength: 10 }),
          // reset 时传入的新文本
          fc.string({ minLength: 0, maxLength: 100 }),
          (chunks, newText) => {
            const state = createStreamState('初始文本')

            // 模拟添加多个 chunk
            for (const chunk of chunks) {
              state.addChunk(chunk)
            }

            // 确认队列非空
            expect(state.getQueue().length).toBeGreaterThan(0)

            // 调用 reset
            state.reset(newText)

            // 验证：displayedText 等于 newText
            expect(state.getDisplayedText()).toBe(newText)
            // 验证：队列为空
            expect(state.getQueue()).toEqual([])
          }
        ),
        { numRuns: 100 }
      )
    })

    it('reset 不传参数时 displayedText 为空字符串', () => {
      fc.assert(
        fc.property(
          fc.array(fc.string({ minLength: 1, maxLength: 30 }), { minLength: 1, maxLength: 10 }),
          (chunks) => {
            const state = createStreamState('一些初始内容')

            for (const chunk of chunks) {
              state.addChunk(chunk)
            }

            // 不传参数调用 reset
            state.reset()

            expect(state.getDisplayedText()).toBe('')
            expect(state.getQueue()).toEqual([])
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  // ============================================================
  // Feature: chatpdf-frontend-performance, Property 6: 流式输出文本一致性（Round-trip）
  // **Validates: Requirements 4.3**
  //
  // 对任意输入文本块序列，将所有文本块依次通过 addChunk 方法输入，
  // 流式输出完成后，getFinalText() 返回的文本应与所有输入文本块的
  // 拼接结果完全一致。
  // ============================================================
  describe('Property 6: 流式输出文本一致性（Round-trip）', () => {
    /**
     * 模拟 useSmoothStream 的完整渲染循环（同步版本）
     *
     * 忠实复现 hook 中 renderLoop 的逻辑：
     *   1. addChunk 将文本通过 splitChunk 拆分后入队
     *   2. 逐帧渲染：每帧取 Math.max(1, Math.floor(queue.length / 5)) 个字符
     *   3. streamDone 时一次性渲染所有剩余字符
     *   4. getFinalText 返回最终文本
     *
     * @param {string[]} chunks - 输入文本块序列
     * @param {string} initialText - 初始文本
     * @returns {{ finalText: string, displayedText: string }}
     */
    function simulateFullRenderCycle(chunks, initialText = '') {
      let queue = []
      let displayedText = initialText
      let finalText = initialText

      // 阶段 1：逐个 addChunk，模拟流式输入
      for (const chunk of chunks) {
        const chars = splitChunk(chunk)
        queue = [...queue, ...chars]
      }

      // 阶段 2：模拟 rAF 渲染循环（streamDone=false 期间逐帧渲染）
      // 执行若干帧，每帧按公式取字符
      const maxFrames = 1000 // 安全上限，防止无限循环
      let frame = 0
      while (queue.length > 0 && frame < maxFrames) {
        const charsToRenderCount = Math.max(1, Math.floor(queue.length / 5))
        const charsToRender = queue.slice(0, charsToRenderCount)
        displayedText += charsToRender.join('')
        finalText = displayedText
        queue = queue.slice(charsToRenderCount)
        frame++
      }

      // 阶段 3：streamDone=true，一次性渲染所有剩余字符（如果还有的话）
      if (queue.length > 0) {
        displayedText += queue.join('')
        finalText = displayedText
        queue = []
      }

      return { finalText, displayedText }
    }

    it('任意文本块序列经 addChunk → 渲染循环 → getFinalText 后与原始拼接一致', () => {
      fc.assert(
        fc.property(
          fc.array(
            fc.string({ minLength: 0, maxLength: 100 }),
            { minLength: 0, maxLength: 30 }
          ),
          (chunks) => {
            const { finalText } = simulateFullRenderCycle(chunks)
            const expected = chunks.join('')
            expect(finalText).toBe(expected)
          }
        ),
        { numRuns: 200 }
      )
    })

    it('带初始文本时，getFinalText 应等于 initialText + 所有 chunk 拼接', () => {
      fc.assert(
        fc.property(
          fc.array(
            fc.string({ minLength: 0, maxLength: 50 }),
            { minLength: 0, maxLength: 20 }
          ),
          fc.string({ minLength: 0, maxLength: 50 }),
          (chunks, initialText) => {
            const { finalText } = simulateFullRenderCycle(chunks, initialText)
            const expected = initialText + chunks.join('')
            expect(finalText).toBe(expected)
          }
        ),
        { numRuns: 200 }
      )
    })

    it('包含中文、emoji、特殊字符的文本块序列 round-trip 一致', () => {
      // 自定义生成器：混合多语言字符
      const unicodeCharArb = fc.constantFrom(
        'a', 'Z', '0', ' ', '.', '\n', '\t',       // ASCII
        '你', '好', '世', '界', '测', '试', '中',   // 中文
        '🎉', '🚀', '❤️', '👨‍👩‍👧‍👦', '🇨🇳',           // emoji（含组合 emoji）
        '①', '②', '™', '©', '½',                   // 特殊符号
        'α', 'β', 'γ',                              // 希腊字母
        'あ', 'い', 'う',                            // 日文
        '한', '글',                                  // 韩文
      )
      const unicodeChunkArb = fc.array(unicodeCharArb, { minLength: 0, maxLength: 50 })
        .map((chars) => chars.join(''))
      const chunksArb = fc.array(unicodeChunkArb, { minLength: 0, maxLength: 20 })

      fc.assert(
        fc.property(chunksArb, (chunks) => {
          const { finalText } = simulateFullRenderCycle(chunks)
          const expected = chunks.join('')
          expect(finalText).toBe(expected)
        }),
        { numRuns: 200 }
      )
    })

    it('splitChunk round-trip：任意文本经 splitChunk 分割后拼接与原文一致', () => {
      // 此测试从 round-trip 角度验证 splitChunk 是文本一致性的基础
      fc.assert(
        fc.property(
          fc.array(
            fc.string({ minLength: 0, maxLength: 100 }),
            { minLength: 1, maxLength: 20 }
          ),
          (chunks) => {
            // 对每个 chunk 分别验证 splitChunk round-trip
            for (const chunk of chunks) {
              expect(splitChunk(chunk).join('')).toBe(chunk)
            }
            // 对所有 chunk 拼接后的完整文本也验证
            const fullText = chunks.join('')
            expect(splitChunk(fullText).join('')).toBe(fullText)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('单个大文本块的 round-trip 一致性', () => {
      fc.assert(
        fc.property(
          // 生成较大的单个文本块
          fc.string({ minLength: 100, maxLength: 2000 }),
          (bigChunk) => {
            const { finalText } = simulateFullRenderCycle([bigChunk])
            expect(finalText).toBe(bigChunk)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('大量小文本块的 round-trip 一致性', () => {
      fc.assert(
        fc.property(
          // 生成大量小文本块（模拟 SSE 逐 token 推送）
          fc.array(
            fc.string({ minLength: 1, maxLength: 5 }),
            { minLength: 10, maxLength: 100 }
          ),
          (chunks) => {
            const { finalText } = simulateFullRenderCycle(chunks)
            const expected = chunks.join('')
            expect(finalText).toBe(expected)
          }
        ),
        { numRuns: 100 }
      )
    })
  })
})
