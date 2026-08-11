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
 * 字符级流式缓冲渲染 Hook（ref 直写模式）
 *
 * 将后端 SSE 推送的文本块拆分为字符队列，通过 requestAnimationFrame
 * 循环按帧动态渲染。流式输出期间通过 contentRef 直接更新 DOM，
 * 避免触发 React 状态更新和重渲染。流结束后通过 getFinalText()
 * 获取最终文本，供调用方同步到 React 状态。
 *
 * @param {Object} options
 * @param {Function} [options.onUpdate] - 每帧渲染后的回调（向后兼容，可选）
 * @param {boolean} options.streamDone - 流式传输是否已完成
 * @param {number} [options.minDelay=10] - 两次渲染之间的最小间隔（毫秒）
 * @param {string} [options.initialText=''] - 初始文本
 * @param {boolean} [options.enableBlurReveal=false] - 是否启用 Blur Reveal 动画
 * @param {string} [options.blurIntensity='medium'] - Blur Reveal 强度（light|medium|strong）
 * @param {number} [options.frameChars=2] - 流式阶段每帧最多渲染字符数
 * @param {number} [options.flushChars=80] - 结束冲刷阶段每帧最多渲染字符数
 * @returns {{ addChunk: Function, reset: Function, replace: Function, flushNow: Function, contentRef: React.RefObject, getFinalText: Function, isFlushComplete: Function, waitForRevealComplete: Function }}
 */
export { splitChunk }

const prefersReducedMotion = () => (
  typeof window !== 'undefined'
  && typeof window.matchMedia === 'function'
  && window.matchMedia('(prefers-reduced-motion: reduce)').matches
)

// 这些值必须和 index.css 中的动画参数保持一致。流结束时 React 会把
// 直写节点切回 Markdown，因此需要知道最后一批字符还要播放多久。
const BLUR_REVEAL_TIMINGS = {
  light: { duration: 280, stagger: 10 },
  medium: { duration: 440, stagger: 16 },
  strong: { duration: 580, stagger: 22 },
}

const getBlurRevealTailMs = (intensity) => {
  const timing = BLUR_REVEAL_TIMINGS[intensity] || BLUR_REVEAL_TIMINGS.medium
  return timing.duration + timing.stagger * 8 + 24
}

// 小队列保持用户选择的逐字速度；积压变大时限定可见输出与后端流之间的
// 最大落后帧数。上限避免单帧创建过多 DOM，仍能让长 reasoning 及时追上。
export const resolveAdaptiveFrameChars = (pendingChars, frameChars = 2, maxFrameChars = 256) => {
  const pending = Math.max(0, Number(pendingChars) || 0)
  const base = Math.max(1, Math.floor(Number(frameChars) || 1))
  const maxChars = Math.max(base, Math.floor(Number(maxFrameChars) || base))

  if (pending <= 256) return base

  const targetFrames = pending > 8192
    ? 36
    : pending > 4096
      ? 48
      : pending > 1024
        ? 72
        : 120

  return Math.min(maxChars, Math.max(base, Math.ceil(pending / targetFrames)))
}

const appendBlurRevealChars = (container, chars, intensity = 'medium') => {
  if (!container || !chars || chars.length === 0) return false

  if (prefersReducedMotion()) {
    container.appendChild(document.createTextNode(chars.join('')))
    return false
  }

  const fragment = document.createDocumentFragment()
  const intensityClass = `blur-intensity-${intensity}`
  const animatedSpans = []
  let visibleIndex = 0

  for (const ch of chars) {
    // 空格和换行无需占用独立合成层，保留为普通文本节点即可。
    if (/^\s+$/u.test(ch)) {
      fragment.appendChild(document.createTextNode(ch))
      continue
    }

    const span = document.createElement('span')
    span.className = `blur-reveal-animate ${intensityClass}`
    span.style.setProperty('--blur-sequence-index', String(Math.min(visibleIndex, 8)))
    span.textContent = ch
    visibleIndex += 1
    animatedSpans.push({ span, ch })
    fragment.appendChild(span)
  }

  animatedSpans.forEach(({ span, ch }, index) => {
    span.addEventListener('animationend', () => {
      if (!span.isConnected) return
      span.replaceWith(document.createTextNode(ch))
      if (index === animatedSpans.length - 1) {
        container.normalize()
      }
    }, { once: true })
  })

  container.appendChild(fragment)
  return animatedSpans.length > 0
}

export const useSmoothStream = ({
  onUpdate,
  streamDone,
  minDelay = 10,
  initialText = '',
  enableBlurReveal = false,
  blurIntensity = 'medium',
  frameChars = 2,
  flushChars = 80,
  smoothFlush = false,
}) => {
  /** @type {React.MutableRefObject<Array<{chars: string[], offset: number}>>} 待渲染字符块队列 */
  const chunkQueueRef = useRef([])
  /** @type {React.MutableRefObject<number>} 当前队首块索引 */
  const queueHeadRef = useRef(0)
  /** @type {React.MutableRefObject<number>} 尚未渲染的字符数 */
  const pendingCharsRef = useRef(0)
  /** @type {React.MutableRefObject<number|null>} 当前 rAF ID，用于清理 */
  const animationFrameRef = useRef(null)
  /** @type {React.MutableRefObject<string>} 当前已渲染的完整文本 */
  const displayedTextRef = useRef(initialText)
  /** @type {React.MutableRefObject<number>} 上次渲染时间戳，用于最小延迟控制 */
  const lastUpdateTimeRef = useRef(0)
  /** @type {React.MutableRefObject<HTMLElement|null>} 指向 DOM 元素的 ref，用于直接更新 DOM */
  const contentRef = useRef(null)
  /** @type {React.MutableRefObject<HTMLElement|null>} 已和 displayedText 同步的 DOM 容器 */
  const boundContentElementRef = useRef(null)
  /** @type {React.MutableRefObject<boolean>} replace/reset 后首帧校验 React 是否提前回填 */
  const verifyBoundContentBeforeAppendRef = useRef(false)
  /** @type {React.MutableRefObject<string>} 流结束后的最终文本 */
  const finalTextRef = useRef(initialText)
  // 时间戳比单纯统计 DOM span 更可靠：后台标签页可能暂停 animationend，
  // 但我们仍然要在切回最终 Markdown 前给前台动画一个完整尾巴。
  const revealTailUntilRef = useRef(0)

  const clearQueue = useCallback(() => {
    chunkQueueRef.current = []
    queueHeadRef.current = 0
    pendingCharsRef.current = 0
  }, [])

  const takeQueuedChars = useCallback((requestedCount) => {
    const count = Math.min(
      pendingCharsRef.current,
      Math.max(0, Math.floor(Number(requestedCount) || 0)),
    )
    if (count <= 0) return []

    const renderedChars = []
    let remaining = count
    while (remaining > 0 && queueHeadRef.current < chunkQueueRef.current.length) {
      const entry = chunkQueueRef.current[queueHeadRef.current]
      const available = entry.chars.length - entry.offset
      const take = Math.min(remaining, available)
      const end = entry.offset + take
      for (let index = entry.offset; index < end; index += 1) {
        renderedChars.push(entry.chars[index])
      }
      entry.offset = end
      remaining -= take
      if (entry.offset >= entry.chars.length) {
        queueHeadRef.current += 1
      }
    }

    pendingCharsRef.current = Math.max(0, pendingCharsRef.current - renderedChars.length)

    // 定期压缩已经消费的块，避免大量 SSE 小分片长期占用数组头部。
    if (
      queueHeadRef.current >= 128
      && queueHeadRef.current * 2 >= chunkQueueRef.current.length
    ) {
      chunkQueueRef.current.splice(0, queueHeadRef.current)
      queueHeadRef.current = 0
    }

    return renderedChars
  }, [])

  const drainQueuedText = useCallback(() => {
    if (pendingCharsRef.current <= 0) {
      clearQueue()
      return ''
    }

    const parts = []
    for (let index = queueHeadRef.current; index < chunkQueueRef.current.length; index += 1) {
      const entry = chunkQueueRef.current[index]
      if (entry.offset < entry.chars.length) {
        parts.push(entry.chars.slice(entry.offset).join(''))
      }
    }
    clearQueue()
    return parts.join('')
  }, [clearQueue])

  const appendMissingText = useCallback((container, targetText, intensity) => {
    if (!container) return
    const currentText = container.textContent || ''
    if (
      enableBlurReveal
      && targetText.startsWith(currentText)
      && targetText.length > currentText.length
    ) {
      const missingChars = splitChunk(targetText.slice(currentText.length))
      const didAnimate = appendBlurRevealChars(container, missingChars, intensity)
      if (didAnimate) {
        revealTailUntilRef.current = Math.max(
          revealTailUntilRef.current,
          Date.now() + getBlurRevealTailMs(intensity),
        )
      }
      return
    }
    if (currentText !== targetText) {
      container.textContent = targetText
    }
  }, [enableBlurReveal])

  /**
   * 将新文本块加入字符队列
   * @param {string} chunk - 新到达的文本块
   */
  const addChunk = useCallback((chunk) => {
    if (!chunk) return
    const chars = splitChunk(chunk)
    if (!chars || chars.length === 0) return
    chunkQueueRef.current.push({ chars, offset: 0 })
    pendingCharsRef.current += chars.length
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
      clearQueue()
      displayedTextRef.current = newText
      finalTextRef.current = newText
      revealTailUntilRef.current = 0
      lastUpdateTimeRef.current = 0
      // 重置 DOM 元素内容
      if (contentRef.current) {
        contentRef.current.textContent = newText
        boundContentElementRef.current = contentRef.current
        verifyBoundContentBeforeAppendRef.current = true
      } else {
        boundContentElementRef.current = null
        verifyBoundContentBeforeAppendRef.current = false
      }
      // 向后兼容：如果提供了 onUpdate 回调，也调用它
      if (onUpdate) {
        onUpdate(newText)
      }
    },
    [clearQueue, onUpdate]
  )

  /**
   * 替换当前已渲染的文本（不中断流式状态）
   * 用于后置引文注入等场景：后端流式完成后发送带 [N] 标记的完整内容替换
   * @param {string} newText - 新的完整文本
   */
  const replace = useCallback(
    (newText) => {
      clearQueue()
      displayedTextRef.current = newText
      finalTextRef.current = newText
      revealTailUntilRef.current = 0
      lastUpdateTimeRef.current = 0
      if (contentRef.current) {
        contentRef.current.textContent = newText
        boundContentElementRef.current = contentRef.current
        verifyBoundContentBeforeAppendRef.current = true
      } else {
        boundContentElementRef.current = null
        verifyBoundContentBeforeAppendRef.current = false
      }
      if (onUpdate) {
        onUpdate(newText)
      }
    },
    [clearQueue, onUpdate]
  )

  /**
   * 超过收尾宽限后立即排空待渲染队列，仅作为后台节流等异常场景的兜底。
   * 可传入后端最终文本，确保 DOM 与随后进入 ReactMarkdown 的内容一致。
   */
  const flushNow = useCallback(
    (finalText) => {
      const queuedText = drainQueuedText()
      const nextText = typeof finalText === 'string'
        ? finalText
        : displayedTextRef.current + queuedText

      displayedTextRef.current = nextText
      finalTextRef.current = nextText
      revealTailUntilRef.current = 0
      lastUpdateTimeRef.current = 0
      if (contentRef.current) {
        contentRef.current.textContent = nextText
        boundContentElementRef.current = contentRef.current
      } else {
        boundContentElementRef.current = null
      }
      verifyBoundContentBeforeAppendRef.current = false
      if (onUpdate) {
        onUpdate(nextText)
      }
      return nextText
    },
    [drainQueuedText, onUpdate]
  )

  /**
   * 获取流结束后的最终文本
   * 供调用方在流结束后同步到 React 状态
   * @returns {string} 最终文本
   */
  const getFinalText = useCallback(() => finalTextRef.current, [])

  /**
   * rAF 渲染循环
   *
   * 每帧执行逻辑：
   *   1. 队列为空 + 流未结束 -> 等待下一帧
   *   2. 队列为空 + 流已结束 -> 记录最终文本，停止循环
   *   3. 距上次渲染 < minDelay -> 等待下一帧
   *   4. 计算本帧字符数: 流式阶段使用固定 frameChars，结束阶段使用 flushChars
   *   5. 流已结束 -> 进入 smoothFlush 渐进冲刷或一次性渲染剩余字符
   *   6. 取出字符追加到 displayedText，直接写入 DOM
   *   7. 队列仍有内容 -> 继续下一帧
   */
  const renderLoop = useCallback(
    (currentTime) => {
      try {
        // 1. 队列为空时的处理
        if (pendingCharsRef.current === 0) {
          // 兼容 ref 延迟挂载：如果文本已渲染到内存但 DOM 还未绑定，
          // 在空队列阶段持续尝试同步，避免出现“流式结束前一直空白，最后一次性显示”。
          if (contentRef.current && boundContentElementRef.current !== contentRef.current) {
            // ref 延迟挂载时，优先把尚未进入 DOM 的后缀补成动画节点；
            // 只有内容发生非追加式变化时才回退为纯文本。
            appendMissingText(contentRef.current, displayedTextRef.current, blurIntensity)
            boundContentElementRef.current = contentRef.current
            verifyBoundContentBeforeAppendRef.current = false
          }
          if (streamDone) {
            // 流已结束，记录最终文本
            finalTextRef.current = displayedTextRef.current
            // 向后兼容：调用 onUpdate 通知最终文本
            if (onUpdate) {
              onUpdate(displayedTextRef.current)
            }
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

        // 3. 小队列遵守用户选择的逐字速度；积压时渐进追赶，防止长 reasoning
        //    让可见输出永久落后并持续占用主线程。
        const pendingChars = pendingCharsRef.current
        let charsToRenderCount = resolveAdaptiveFrameChars(
          pendingChars,
          frameChars,
          enableBlurReveal ? 64 : 256,
        )

        // 4. 流已结束时的渲染策略
        if (streamDone) {
          if (smoothFlush) {
            // 最多约 48 个渲染帧平滑排空：比正常逐字稍快，但仍保留连续渐显。
            charsToRenderCount = Math.max(
              charsToRenderCount,
              flushChars,
              Math.ceil(pendingChars / 48)
            )
          } else {
            // 标准模式：一次性渲染所有剩余字符
            charsToRenderCount = pendingCharsRef.current
          }
        }

        // 5. 取出字符并追加到已渲染文本
        const charsToRender = takeQueuedChars(charsToRenderCount)
        const previousText = displayedTextRef.current
        const renderedText = charsToRender.join('')
        displayedTextRef.current += renderedText

        // 6. 直接更新 DOM 元素（ref 直写模式，避免 React setState）
        if (contentRef.current) {
          if (boundContentElementRef.current !== contentRef.current) {
            appendMissingText(contentRef.current, previousText, blurIntensity)
            boundContentElementRef.current = contentRef.current
            verifyBoundContentBeforeAppendRef.current = false
          } else if (verifyBoundContentBeforeAppendRef.current) {
            // React state 可能在 replace('') 后先把首段内容回填到同一 DOM。
            // 这里只校验一次，随后完全走增量追加，避免长文本逐帧全文扫描。
            if ((contentRef.current.textContent || '') !== previousText) {
              contentRef.current.textContent = previousText
            }
            verifyBoundContentBeforeAppendRef.current = false
          }
          if (enableBlurReveal) {
            const didAnimate = appendBlurRevealChars(contentRef.current, charsToRender, blurIntensity)
            if (didAnimate) {
              revealTailUntilRef.current = Math.max(
                revealTailUntilRef.current,
                Date.now() + getBlurRevealTailMs(blurIntensity),
              )
            }
          } else {
            // 思考区不使用逐字符动画，直接追加到现有文本节点，避免每帧重新
            // 序列化并覆盖整段长文本。
            const lastChild = contentRef.current.lastChild
            if (lastChild?.nodeType === 3) {
              lastChild.appendData(renderedText)
            } else {
              contentRef.current.appendChild(document.createTextNode(renderedText))
            }
          }
        }

        // 7. 向后兼容：如果提供了 onUpdate 回调，也调用它
        //    注意：在纯 ref 直写模式下，调用方不应传入 onUpdate
        if (onUpdate) {
          onUpdate(displayedTextRef.current)
        }

        // 8. 记录最终文本（持续更新，确保 getFinalText 随时可用）
        finalTextRef.current = displayedTextRef.current

        // 9. 继续下一帧：
        // - 只要流未结束，就持续轮询（允许后续新 chunk 随时进入队列）
        // - 流已结束但队列仍有残留，也继续直到清空
        if (!streamDone || pendingCharsRef.current > 0) {
          animationFrameRef.current = requestAnimationFrame(renderLoop)
        }
      } catch (error) {
        // rAF 回调中发生异常时捕获并记录，继续下一帧
        console.warn('[useSmoothStream] 渲染循环异常:', error)
        animationFrameRef.current = requestAnimationFrame(renderLoop)
      }
    },
    [
      appendMissingText,
      streamDone,
      onUpdate,
      minDelay,
      enableBlurReveal,
      blurIntensity,
      frameChars,
      flushChars,
      smoothFlush,
      takeQueuedChars,
    ]
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

  const isFlushComplete = useCallback(() => pendingCharsRef.current === 0, [])

  // 在流结束后等待最后一批字符的 CSS 动画完成。调用方可提供
  // shouldContinue，在用户停止/切换文档时立即放弃等待。
  const waitForRevealComplete = useCallback(async (shouldContinue = null) => {
    while (Date.now() < revealTailUntilRef.current) {
      if (typeof shouldContinue === 'function' && !shouldContinue()) return false
      const remaining = revealTailUntilRef.current - Date.now()
      await new Promise((resolve) => setTimeout(resolve, Math.min(32, Math.max(8, remaining))))
    }
    return true
  }, [])

  return {
    addChunk,
    reset,
    replace,
    flushNow,
    contentRef,
    getFinalText,
    isFlushComplete,
    waitForRevealComplete,
  }
}
