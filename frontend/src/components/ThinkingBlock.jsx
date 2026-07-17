import React, { useState, useEffect, useRef, useCallback, memo } from 'react'
import { Copy, Check, ChevronDown } from 'lucide-react'
import StreamingMarkdown from './StreamingMarkdown'
import { useChatParams } from '../contexts/ChatParamsContext'
import AgentTracePanel from './AgentTracePanel'
import CellsLoader from './CellsLoader'

/**
 * 实时思考计时器组件
 * 思考中时每 100ms 更新一次显示时间，完成后显示最终耗时
 */
const ThinkingTimer = memo(({ isThinking, thinkingMs }) => {
  const [displayMs, setDisplayMs] = useState(thinkingMs || 0)
  const timerRef = useRef(null)

  useEffect(() => {
    if (isThinking) {
      if (!timerRef.current) {
        const startTime = Date.now() - (thinkingMs || 0)
        timerRef.current = setInterval(() => {
          setDisplayMs(Date.now() - startTime)
        }, 100)
      }
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      if (thinkingMs > 0) {
        setDisplayMs(thinkingMs)
      }
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [isThinking, thinkingMs])

  const seconds = (Math.max(100, displayMs) / 1000).toFixed(1)

  return (
    <span className="whitespace-nowrap text-[10px] font-medium tabular-nums text-gray-500 dark:text-gray-400">
      {isThinking ? `思考中 ${seconds}s` : `已深度思考 ${seconds}s`}
    </span>
  )
})

/**
 * 深度思考展示组件
 * 采用全新的胶囊式开关和展开面板 UI 设计
 */
const ThinkingBlock = ({ content, isStreaming, answerStarted = false, darkMode, thinkingMs, streamingRef, agentTrace }) => {
  const [expanded, setExpanded] = useState(true)
  const [copied, setCopied] = useState(false)
  const [hasStreamingText, setHasStreamingText] = useState(false)
  const [progressLineCount, setProgressLineCount] = useState(1)
  const autoCollapsedRef = useRef(false)
  const { thoughtAutoCollapse } = useChatParams()
  const hasAgentTrace = Boolean(agentTrace && agentTrace.enabled)
  const hasThinkingContent = Boolean(content && content.trim())

  // 思考阶段是否已结束：流式中以「正文首 token 到达」为准（answerStarted），
  // 兜底用整条消息结束（!isStreaming），不必等回答全部生成完
  const isThinkingPhase = isStreaming && !answerStarted
  const thinkingFinished = hasThinkingContent && !isThinkingPhase

  const getProgressLineCount = useCallback((text = '') => {
    const lines = String(text)
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0)
    return Math.max(1, lines.length)
  }, [])

  // 思考完成后自动折叠（受 thoughtAutoCollapse 设置控制）。
  // 只自动折叠一次：用户手动展开后不会被再次收起。
  useEffect(() => {
    if (!thoughtAutoCollapse || autoCollapsedRef.current || !thinkingFinished) return undefined
    autoCollapsedRef.current = true
    // 延迟折叠，让用户看到完成状态
    const timer = setTimeout(() => setExpanded(false), 600)
    return () => clearTimeout(timer)
  }, [thinkingFinished, thoughtAutoCollapse])

  // 观察流式 DOM 的直接写入内容，任何检索/思考文本出现后都隐藏占位提示。
  useEffect(() => {
    if (!isStreaming || !streamingRef?.current) {
      setHasStreamingText(false)
      setProgressLineCount(getProgressLineCount(content))
      return
    }

    const el = streamingRef.current
    const syncHasContent = () => {
      const text = el.textContent || ''
      setHasStreamingText(text.trim().length > 0)
      setProgressLineCount((prev) => {
        const next = getProgressLineCount(text || content)
        return prev === next ? prev : next
      })
    }

    syncHasContent()
    const observer = new MutationObserver(syncHasContent)
    observer.observe(el, { childList: true, subtree: true, characterData: true })

    return () => observer.disconnect()
  }, [content, getProgressLineCount, isStreaming, streamingRef])

  // 复制思考内容
  const handleCopy = useCallback((e) => {
    e.stopPropagation()
    if (!content) return
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [content])

  // 深度思考在首个 reasoning_content 到来前，直接给出可见的阶段提示，
  // 避免面板只有三个点、看起来像“卡住了”。
  const shouldShowStreamingHint = isThinkingPhase && !hasThinkingContent && !hasStreamingText
  const shouldShowTimeline = shouldShowStreamingHint || isThinkingPhase || hasThinkingContent
  const timelineDotOffset = Math.min(280, Math.max(0, (progressLineCount - 1) * 23))

  // 普通模型可能不返回 reasoning_content。正文开始后不要继续展示一个
  // “已深度思考 0.1s”的空壳；有真实思考或 Agent 轨迹时仍保留历史过程。
  if (answerStarted && !hasThinkingContent && !hasAgentTrace) return null

  return (
    <div className={`my-2 w-full ${darkMode ? 'dark' : ''}`}>
      <section
        className={`w-full overflow-hidden rounded-[18px] border text-[13px] transition-[transform,box-shadow,border-color] duration-300 ease-out hover:-translate-y-px motion-reduce:transform-none motion-reduce:transition-none ${
          darkMode
            ? 'border-white/[0.09] bg-[#25282f] text-gray-300 shadow-[0_16px_34px_-24px_rgba(0,0,0,0.82),inset_0_1px_0_rgba(255,255,255,0.04)] hover:border-white/[0.13] hover:shadow-[0_20px_38px_-24px_rgba(0,0,0,0.9),inset_0_1px_0_rgba(255,255,255,0.05)]'
            : 'border-[#eadfd8] bg-white text-gray-600 shadow-[0_14px_30px_-23px_rgba(91,65,52,0.52),0_3px_8px_-6px_rgba(91,65,52,0.18),inset_0_1px_0_rgba(255,255,255,0.96)] hover:border-[#e3d4cc] hover:shadow-[0_20px_38px_-24px_rgba(91,65,52,0.58),0_6px_14px_-10px_rgba(91,65,52,0.22),inset_0_1px_0_rgba(255,255,255,0.98)]'
        }`}
      >
        <button
          type="button"
          className={`flex w-full items-center gap-2.5 px-3 py-2.5 text-left transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset ${
            darkMode
              ? 'hover:bg-white/[0.035] focus-visible:ring-[#FFA07A]/45'
              : 'hover:bg-[#fcfaf8] focus-visible:ring-[#D99178]/45'
          }`}
          onClick={() => setExpanded((prev) => !prev)}
          aria-expanded={expanded}
        >
          <span
            className={`flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-[9px] ${
              darkMode ? 'bg-[#FFA07A]/10 text-[#FFA07A]' : 'bg-[#FFF0E9] text-[#B85F47]'
            }`}
          >
            <CellsLoader active={isThinkingPhase} />
          </span>
          <span className={`text-[11px] font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'} ${isThinkingPhase ? 'thinking-text-pulse' : ''}`}>
            Deep Thinking
          </span>
          <span className={`h-3 w-px ${darkMode ? 'bg-white/10' : 'bg-[#e9dfda]'}`} aria-hidden="true" />
          <ThinkingTimer isThinking={isThinkingPhase} thinkingMs={thinkingMs || 0} />
          <ChevronDown
            size={15}
            className={`ml-auto flex-shrink-0 transition-transform duration-300 ${
              darkMode ? 'text-gray-500' : 'text-gray-400'
            } ${expanded ? 'rotate-180' : ''}`}
          />
        </button>

        <div
          className={`grid transition-[grid-template-rows,opacity] duration-300 ease-out motion-reduce:transition-none ${
            expanded
              ? 'visible grid-rows-[1fr] opacity-100'
              : 'invisible pointer-events-none grid-rows-[0fr] opacity-0'
          }`}
          aria-hidden={!expanded}
          inert={expanded ? undefined : ''}
        >
          <div className="min-h-0 overflow-hidden">
            <div className={`relative border-t px-4 pb-4 pt-3 ${darkMode ? 'border-white/[0.07]' : 'border-[#f0e8e3]'}`}>
              {!isStreaming && content && (
                <button
                  className={`absolute right-2.5 top-2.5 z-[1] rounded-[8px] p-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                    darkMode
                      ? 'text-gray-500 hover:bg-white/[0.06] hover:text-gray-200 focus-visible:ring-[#FFA07A]/45'
                      : 'text-gray-400 hover:bg-[#f6f1ee] hover:text-gray-700 focus-visible:ring-[#D99178]/45'
                  }`}
                  onClick={handleCopy}
                  title="复制思考内容"
                  aria-label="复制思考内容"
                >
                  {copied ? <Check size={14} className="text-emerald-500" /> : <Copy size={14} />}
                </button>
              )}

              {shouldShowTimeline && (
                <div className={`ml-1.5 border-l py-0.5 pl-3 pr-7 ${darkMode ? 'border-[#FFA07A]/20' : 'border-[#eaded8]'}`}>
                  <div className="relative">
                    {/* 活动进度点：随检索/思考阶段行数下移，避免一直停在第一行。 */}
                    <span
                      className={`absolute -left-[17px] top-1.5 h-2 w-2 rounded-full transition-transform duration-300 ease-out ${
                        isThinkingPhase
                          ? 'bg-[#B85F47] shadow-[0_0_8px_rgba(184,95,71,0.28)]'
                          : 'bg-[#FFDCCF] dark:bg-[#FFA07A]/30'
                      }`}
                      style={{ transform: `translateY(${timelineDotOffset}px)` }}
                    />
                    {shouldShowStreamingHint && (
                      <div className={`mb-2 text-[12px] italic leading-relaxed ${darkMode ? 'text-[#FFA07A]/90' : 'text-[#B85F47]/90'}`}>
                        正在检索并组织思考内容...
                      </div>
                    )}
                    <div className={`prose prose-sm max-w-none text-[13px] leading-relaxed ${darkMode ? 'prose-invert text-gray-300' : 'text-gray-500'}`}>
                      <StreamingMarkdown
                        content={content}
                        isStreaming={isStreaming}
                        enableBlurReveal={false}
                        blurIntensity="light"
                        streamingRef={streamingRef}
                        suppressInitialDots={shouldShowStreamingHint}
                      />
                    </div>
                  </div>
                </div>
              )}

              {hasAgentTrace && <AgentTracePanel trace={agentTrace} embedded />}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default memo(ThinkingBlock)
