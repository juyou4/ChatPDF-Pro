/**
 * useSmoothStream 单元测试
 *
 * 测试场景：
 * 1. 最小延迟控制（Requirements 1.4）
 * 2. 空文本块处理
 * 3. 组件卸载时 rAF 清理（Requirements 5.3）
 *
 * @vitest-environment jsdom
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useSmoothStream } from '../hooks/useSmoothStream.js'

describe('useSmoothStream 单元测试', () => {
  let rafCallbacks = []
  let rafIdCounter = 0
  let cancelledRafIds = new Set()
  let originalRaf
  let originalCancelRaf
  let originalPerformanceNow

  beforeEach(() => {
    rafCallbacks = []
    rafIdCounter = 0
    cancelledRafIds = new Set()

    // mock requestAnimationFrame：收集回调，手动触发
    originalRaf = globalThis.requestAnimationFrame
    vi.stubGlobal('requestAnimationFrame', (cb) => {
      const id = ++rafIdCounter
      rafCallbacks.push({ id, cb })
      return id
    })

    // mock cancelAnimationFrame：记录被取消的 ID
    originalCancelRaf = globalThis.cancelAnimationFrame
    vi.stubGlobal('cancelAnimationFrame', (id) => {
      cancelledRafIds.add(id)
      rafCallbacks = rafCallbacks.filter((item) => item.id !== id)
    })

    // mock performance.now
    originalPerformanceNow = performance.now
  })

  afterEach(() => {
    vi.restoreAllMocks()
    if (originalRaf) globalThis.requestAnimationFrame = originalRaf
    if (originalCancelRaf) globalThis.cancelAnimationFrame = originalCancelRaf
    if (originalPerformanceNow) performance.now = originalPerformanceNow
  })

  /**
   * 执行所有待处理的 rAF 回调（未被取消的）
   * @param {number} currentTime - 传给回调的时间戳
   */
  function flushRaf(currentTime) {
    const pending = [...rafCallbacks]
    rafCallbacks = []
    for (const { id, cb } of pending) {
      if (!cancelledRafIds.has(id)) {
        cb(currentTime)
      }
    }
  }

  // ============================================================
  // 测试 1：最小延迟控制（Requirements 1.4）
  // ============================================================
  describe('最小延迟控制', () => {
    it('两次渲染之间的间隔不小于 minDelay 时才执行渲染', () => {
      const updates = []
      const onUpdate = vi.fn((text) => updates.push(text))

      const { unmount } = renderHook(() =>
        useSmoothStream({
          onUpdate,
          streamDone: false,
          minDelay: 50, // 设置较大的 minDelay 便于测试
          initialText: '',
        })
      )

      const { addChunk } = renderHook(() =>
        useSmoothStream({
          onUpdate,
          streamDone: false,
          minDelay: 50,
          initialText: '',
        })
      ).result.current

      unmount()

      // 重新渲染以获取正确的 hook 实例
      onUpdate.mockClear()
      updates.length = 0

      const hook = renderHook(() =>
        useSmoothStream({
          onUpdate,
          streamDone: false,
          minDelay: 50,
          initialText: '',
        })
      )

      // 添加足够多的字符
      act(() => {
        hook.result.current.addChunk('abcdefghij')
      })

      // 第一帧：时间 100ms，应该渲染（距上次 0ms > 50ms 的 minDelay）
      flushRaf(100)

      const updatesAfterFirst = onUpdate.mock.calls.length
      expect(updatesAfterFirst).toBeGreaterThan(0)

      // 第二帧：时间 110ms（距上次仅 10ms < 50ms），不应渲染新字符
      const callCountBefore = onUpdate.mock.calls.length
      flushRaf(110)
      // 可能会调用 rAF 但不应有新的 onUpdate（或 onUpdate 参数不变）
      // 由于队列可能为空导致不调用，我们检查是否有新的不同内容的调用
      const callCountAfter = onUpdate.mock.calls.length
      if (callCountAfter > callCountBefore) {
        // 如果有调用，内容应该和上次一样（因为 minDelay 未到，不渲染新字符）
        const lastCallBefore = onUpdate.mock.calls[callCountBefore - 1][0]
        const lastCallAfter = onUpdate.mock.calls[callCountAfter - 1][0]
        // 在 minDelay 内不应有新字符被渲染
        expect(lastCallAfter).toBe(lastCallBefore)
      }

      // 第三帧：时间 200ms（距上次 100ms > 50ms），应该渲染更多字符
      flushRaf(200)
      const callCountFinal = onUpdate.mock.calls.length
      expect(callCountFinal).toBeGreaterThan(callCountBefore)

      hook.unmount()
    })
  })

  // ============================================================
  // 测试 2：空文本块处理
  // ============================================================
  describe('空文本块处理', () => {
    it('addChunk 传入空字符串不应影响队列状态', () => {
      const onUpdate = vi.fn()

      const hook = renderHook(() =>
        useSmoothStream({
          onUpdate,
          streamDone: false,
          minDelay: 10,
          initialText: '',
        })
      )

      // 添加空字符串
      act(() => {
        hook.result.current.addChunk('')
      })

      // 触发 rAF
      onUpdate.mockClear()
      flushRaf(100)

      // 空字符串不应产生有意义的文本更新
      // onUpdate 可能被调用（队列为空 + 流未结束 -> 等待下一帧），
      // 但 displayedText 应保持为空
      const lastCall = onUpdate.mock.calls.length > 0
        ? onUpdate.mock.calls[onUpdate.mock.calls.length - 1][0]
        : ''
      expect(lastCall).toBe('')

      hook.unmount()
    })

    it('addChunk 传入空字符串后再传入正常文本，不影响正常渲染', () => {
      const onUpdate = vi.fn()

      const hook = renderHook(() =>
        useSmoothStream({
          onUpdate,
          streamDone: false,
          minDelay: 0, // 无延迟，简化测试
          initialText: '',
        })
      )

      act(() => {
        hook.result.current.addChunk('')
        hook.result.current.addChunk('hello')
      })

      // 多次 flush 确保所有字符被渲染
      flushRaf(100)
      flushRaf(200)
      flushRaf(300)
      flushRaf(400)
      flushRaf(500)

      // 最终应该包含 "hello" 的部分或全部字符
      const allUpdates = onUpdate.mock.calls.map((c) => c[0])
      const lastUpdate = allUpdates[allUpdates.length - 1]
      expect(lastUpdate.length).toBeGreaterThan(0)
      expect('hello'.startsWith(lastUpdate) || lastUpdate === 'hello').toBe(true)

      hook.unmount()
    })
  })

  // ============================================================
  // 测试 3：组件卸载时 rAF 清理（Requirements 5.3）
  // ============================================================
  describe('组件卸载时 rAF 清理', () => {
    it('组件卸载后应取消所有待执行的 rAF 回调', () => {
      const onUpdate = vi.fn()

      const hook = renderHook(() =>
        useSmoothStream({
          onUpdate,
          streamDone: false,
          minDelay: 10,
          initialText: '',
        })
      )

      // 添加一些内容确保有 rAF 在运行
      act(() => {
        hook.result.current.addChunk('测试内容')
      })

      // 记录卸载前的 cancelAnimationFrame 调用
      const cancelledBefore = cancelledRafIds.size

      // 卸载组件
      hook.unmount()

      // 卸载后应该调用了 cancelAnimationFrame
      expect(cancelledRafIds.size).toBeGreaterThan(cancelledBefore)
    })

    it('组件卸载后 rAF 回调不再执行', () => {
      const onUpdate = vi.fn()

      const hook = renderHook(() =>
        useSmoothStream({
          onUpdate,
          streamDone: false,
          minDelay: 10,
          initialText: '',
        })
      )

      act(() => {
        hook.result.current.addChunk('测试')
      })

      // 卸载组件
      hook.unmount()

      // 记录卸载后的 onUpdate 调用次数
      const callCountAfterUnmount = onUpdate.mock.calls.length

      // 尝试 flush 所有 rAF（已被取消的不应执行）
      flushRaf(1000)

      // 卸载后不应有新的 onUpdate 调用
      expect(onUpdate.mock.calls.length).toBe(callCountAfterUnmount)
    })
  })
})
