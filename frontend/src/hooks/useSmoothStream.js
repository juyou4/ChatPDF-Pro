import { useCallback, useEffect, useRef } from 'react'

/**
 * 创建文本分割器
 * 优先使用 Intl.Segmenter 按语言边界拆分字符，不可用时降级为 Array.from
 */
let segmenter = null
try {
  const languages = ['en-US', 'de-DE', 'es-ES', 'zh-CN', 'zh-TW', 'ja-JP', 'ru-RU', 'el-GR', 'fr-FR', 'pt-PT']
  segmenter = new Intl.Segmenter(languages)
} catch {
  // Intl.Segmenter 不可用（旧浏览器），将在 splitChunk 中降级处理
}

/**
 * 将文本块拆分为字符数组
 * @param {string} chunk - 待拆分的文本块
 * @returns {string[]} 拆分后的字符数组
 */
function splitChunk(chunk) {
  if (segmenter) {
    return Array.from(segmenter.segment(chunk)).map((s) => s.segment)
  }
  // 降级：按 Unicode 码点拆分
  return Array.from(chunk)
}

/**
 * 字符级流式缓冲渲染 Hook
 *
 * 将后端 SSE 推送的文本块拆分为字符队列，通过 requestAnimationFrame
 * 循环按帧动态渲染，实现平滑的打字机效果。
 *
 * @param {Object} options
 * @param {Function} options.onUpdate - 每帧渲染后的回调，参数为当前已渲染的完整文本
 * @param {boolean} options.streamDone - 流式传输是否已完成
 * @param {number} [options.minDelay=10] - 两次渲染之间的最小间隔（毫秒）
 * @param {string} [options.initialText=''] - 初始文本
 * @returns {{ addChunk: Function, reset: Function }}
 */
export { splitChunk }

export const useSmoothStream = ({ onUpdate, streamDone, minDelay = 10, initialText = '' }) => {
  /** @type {React.MutableRefObject<string[]>} 待渲染字符队列 */
  const chunkQueueRef = useRef([])
  /** @type {React.MutableRefObject<number|null>} 当前 rAF ID，用于清理 */
  const animationFrameRef = useRef(null)
  /** @type {React.MutableRefObject<string>} 当前已渲染的完整文本 */
  const displayedTextRef = useRef(initialText)
  /** @type {React.MutableRefObject<number>} 上次渲染时间戳，用于最小延迟控制 */
  const lastUpdateTimeRef = useRef(0)

  /**
   * 将新文本块加入字符队列
   * @param {string} chunk - 新到达的文本块
   */
  const addChunk = useCallback((chunk) => {
    const chars = splitChunk(chunk)
    chunkQueueRef.current = [...chunkQueueRef.current, ...(chars || [])]
  }, [])

  /**
   * 重置所有状态（队列、显示文本、动画帧）
   * @param {string} [newText=''] - 重置后的初始文本
   */
  const reset = useCallback(
    (newText = '') => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      chunkQueueRef.current = []
      displayedTextRef.current = newText
      onUpdate(newText)
    },
    [onUpdate]
  )

  /**
   * rAF 渲染循环
   *
   * 每帧执行逻辑：
   *   1. 队列为空 + 流未结束 -> 等待下一帧
   *   2. 队列为空 + 流已结束 -> 输出最终文本，停止循环
   *   3. 距上次渲染 < minDelay -> 等待下一帧
   *   4. 计算本帧字符数: Math.max(1, Math.floor(queue.length / 5))
   *   5. 流已结束 -> 一次性渲染所有剩余字符
   *   6. 取出字符追加到 displayedText，调用 onUpdate
   *   7. 队列仍有内容 -> 继续下一帧
   */
  const renderLoop = useCallback(
    (currentTime) => {
      try {
        // 1. 队列为空时的处理
        if (chunkQueueRef.current.length === 0) {
          if (streamDone) {
            // 流已结束，输出最终文本并停止循环
            const finalText = displayedTextRef.current
            onUpdate(finalText)
            return
          }
          // 流未结束，等待下一帧
          animationFrameRef.current = requestAnimationFrame(renderLoop)
          return
        }

        // 2. 最小延迟控制
        if (currentTime - lastUpdateTimeRef.current < minDelay) {
          animationFrameRef.current = requestAnimationFrame(renderLoop)
          return
        }
        lastUpdateTimeRef.current = currentTime

        // 3. 动态计算本帧渲染字符数
        let charsToRenderCount = Math.max(1, Math.floor(chunkQueueRef.current.length / 5))

        // 4. 流已结束时一次性渲染所有剩余字符
        if (streamDone) {
          charsToRenderCount = chunkQueueRef.current.length
        }

        // 5. 取出字符并追加到已渲染文本
        const charsToRender = chunkQueueRef.current.slice(0, charsToRenderCount)
        displayedTextRef.current += charsToRender.join('')

        // 6. 调用 onUpdate 回调更新 UI
        onUpdate(displayedTextRef.current)

        // 7. 更新队列，移除已渲染的字符
        chunkQueueRef.current = chunkQueueRef.current.slice(charsToRenderCount)

        // 8. 队列仍有内容，继续下一帧
        if (chunkQueueRef.current.length > 0) {
          animationFrameRef.current = requestAnimationFrame(renderLoop)
        }
      } catch (error) {
        // rAF 回调中发生异常时捕获并记录，继续下一帧
        console.warn('[useSmoothStream] 渲染循环异常:', error)
        animationFrameRef.current = requestAnimationFrame(renderLoop)
      }
    },
    [streamDone, onUpdate, minDelay]
  )

  // 启动渲染循环，组件卸载时取消 rAF 防止内存泄漏
  useEffect(() => {
    animationFrameRef.current = requestAnimationFrame(renderLoop)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [renderLoop])

  return { addChunk, reset }
}
