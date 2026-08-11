import React, { useState, useEffect, useRef, useCallback, useMemo, memo } from 'react'
import { Copy, Check, ChevronDown, Lightbulb } from 'lucide-react'
import StreamingMarkdown from './StreamingMarkdown'
import { useChatParams } from '../contexts/ChatParamsContext'
import AgentTracePanel from './AgentTracePanel'
import CellsLoader from './CellsLoader'
import WebSearchActivity from './WebSearchActivity'

/**
 * 实时思考计时器组件
 * 思考中时每 100ms 更新一次显示时间，完成后显示最终耗时
 */
const ThinkingTimer = memo(({ isThinking, thinkingMs, activityOnly = false }) => {
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
  const hasMeasuredDuration = isThinking || Number(thinkingMs) > 0

  return (
    <span className="whitespace-nowrap text-[13.5px] font-medium tabular-nums text-gray-600 dark:text-gray-300">
      {isThinking
        ? `${activityOnly ? '正在检索与整理' : '正在思考'} · ${seconds} 秒`
        : hasMeasuredDuration
          ? `思考了 ${seconds} 秒`
          : '查看处理过程'}
    </span>
  )
})

/**
 * 深度思考展示组件
 * 使用贴近正文的轻量轨迹，避免过程区抢占回答的视觉层级。
 */
const ThinkingBlock = ({
  content,
  isStreaming,
  answerStarted = false,
  darkMode,
  thinkingMs,
  streamingRef,
  agentTrace,
  webSearchActivity = null,
}) => {
  const [expanded, setExpanded] = useState(true)
  const [copied, setCopied] = useState(false)
  const [hasStreamingText, setHasStreamingText] = useState(false)
  const autoCollapsedRef = useRef(false)
  const { thoughtAutoCollapse } = useChatParams()
  const hasAgentTrace = Boolean(agentTrace && agentTrace.enabled)
  const hasWebSearchActivity = Boolean(webSearchActivity)
  const hasThinkingContent = Boolean(content && content.trim())
  const hasProcessDetails = hasThinkingContent || hasAgentTrace || hasWebSearchActivity
  const agentRunning = Boolean(agentTrace?.enabled && agentTrace?.startedAt && !agentTrace?.endedAt)
  const webSearchRunning = Boolean(
    isStreaming && String(webSearchActivity?.status?.phase || '').toLowerCase() === 'searching'
  )
  const activeStageText = useMemo(() => {
    if ((!isStreaming || answerStarted) && !agentRunning && !webSearchRunning) return ''
    if (webSearchRunning) return '正在联网查找可核验的来源...'
    const currentTask = String(agentTrace?.taskStatus?.current || '').trim()
    if (agentRunning && currentTask) return currentTask
    const lines = String(content || '')
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
    const latestLine = lines[lines.length - 1] || ''
    // “检索完成”是已经结束的阶段，不能继续作为“当前正在做什么”展示。
    // 兼容旧文案中带有 total_ms 的情况，避免思考计时旁出现检索耗时。
    if (/^检索完成(?:[，,：:]|\s|$)/.test(latestLine)) {
      return '正在整理上下文...'
    }
    return latestLine || '正在等待模型响应...'
  }, [agentRunning, agentTrace?.taskStatus?.current, answerStarted, content, isStreaming, webSearchRunning])

  // 思考阶段是否已结束：流式中以「正文首 token 到达」为准（answerStarted），
  // 兜底用整条消息结束（!isStreaming），不必等回答全部生成完
  const isThinkingPhase = isStreaming && !answerStarted
  const isProcessRunning = isThinkingPhase || agentRunning || webSearchRunning
  const processFinished = hasProcessDetails && !isProcessRunning

  // 思考完成后自动折叠（受 thoughtAutoCollapse 设置控制）。
  // 只自动折叠一次：用户手动展开后不会被再次收起。
  useEffect(() => {
    if (!thoughtAutoCollapse || autoCollapsedRef.current || !processFinished) return undefined
    autoCollapsedRef.current = true
    // 尽快折叠，避免挡住“思考结束 → 正文开始”的过渡观感。
    const timer = setTimeout(() => setExpanded(false), 180)
    return () => clearTimeout(timer)
  }, [processFinished, thoughtAutoCollapse])

  // 观察流式 DOM 的直接写入内容，任何检索/思考文本出现后都隐藏占位提示。
  useEffect(() => {
    if (!isStreaming || !streamingRef?.current) {
      setHasStreamingText(false)
      return
    }

    const el = streamingRef.current
    const syncHasContent = () => {
      const text = el.textContent || ''
      setHasStreamingText(text.trim().length > 0)
    }

    syncHasContent()
    const observer = new MutationObserver(syncHasContent)
    observer.observe(el, { childList: true, subtree: true, characterData: true })

    return () => observer.disconnect()
  }, [isStreaming, streamingRef])

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
  const shouldShowStreamingHint = isThinkingPhase
    && !hasThinkingContent
    && !hasStreamingText
    && !hasAgentTrace
    && !hasWebSearchActivity
  const shouldShowTimeline = shouldShowStreamingHint || hasThinkingContent || hasStreamingText

  // 普通模型可能不返回 reasoning_content。正文开始后不要继续展示一个
  // “已深度思考 0.1s”的空壳；有真实思考或 Agent 轨迹时仍保留历史过程。
  if (answerStarted && !hasProcessDetails) return null

  return (
    <section
      className={`group/thinking mb-4 mt-1 w-full max-w-[46rem] text-[13.5px] ${darkMode ? 'dark text-gray-300' : 'text-gray-600'}`}
      aria-label="思考过程"
    >
      <button
        type="button"
        className={`-ml-1 flex min-h-9 max-w-full items-center gap-2.5 rounded-[9px] px-2 py-1.5 text-left transition-[background-color,transform] duration-200 active:scale-[0.995] focus-visible:outline-none focus-visible:ring-2 ${
          darkMode
            ? 'hover:bg-white/[0.04] focus-visible:ring-[#FFA07A]/40'
            : 'hover:bg-[#f6f3f1] focus-visible:ring-[#D99178]/40'
        }`}
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span
          className={`grid h-6 w-6 flex-shrink-0 place-items-center ${
            isProcessRunning
              ? darkMode ? 'text-[#FFA07A]' : 'text-[#A85F49]'
              : darkMode ? 'text-gray-400' : 'text-[#8b817b]'
          }`}
          aria-hidden="true"
        >
          <CellsLoader active={isProcessRunning} />
        </span>
        <ThinkingTimer
          isThinking={isProcessRunning}
          thinkingMs={thinkingMs || 0}
          activityOnly={!hasThinkingContent && (hasAgentTrace || hasWebSearchActivity)}
        />
        {!expanded && activeStageText && (
          <>
            <span className={`h-3 w-px flex-shrink-0 ${darkMode ? 'bg-white/10' : 'bg-[#dedad7]'}`} aria-hidden="true" />
            <span
              className={`min-w-0 flex-1 truncate text-[12.5px] ${darkMode ? 'text-gray-400' : 'text-[#8b817b]'}`}
              title={activeStageText}
            >
              {activeStageText}
            </span>
          </>
        )}
        <ChevronDown
          size={15}
          strokeWidth={1.9}
          className={`flex-shrink-0 transition-transform duration-300 ${
            darkMode ? 'text-gray-500' : 'text-gray-400'
          } ${expanded ? 'rotate-180' : ''}`}
          aria-hidden="true"
        />
      </button>

      <div
        className={`grid transition-[grid-template-rows,opacity,visibility] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none ${
          expanded
            ? 'visible grid-rows-[1fr] opacity-100'
            : 'invisible pointer-events-none grid-rows-[0fr] opacity-0'
        }`}
        aria-hidden={!expanded}
        inert={expanded ? undefined : ''}
      >
        <div className="min-h-0 overflow-hidden">
          <div className={`relative ml-[9px] border-l pb-1 pl-5 pt-1.5 ${darkMode ? 'border-white/[0.14]' : 'border-[#d4ccc6]'}`}>
            {!isStreaming && content && (
              <button
                className={`absolute right-0 top-1 z-[1] rounded-[7px] p-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                  darkMode
                    ? 'text-gray-500 hover:bg-white/[0.05] hover:text-gray-200 focus-visible:ring-[#FFA07A]/40'
                    : 'text-gray-400 hover:bg-[#f6f3f1] hover:text-gray-700 focus-visible:ring-[#D99178]/40'
                }`}
                onClick={handleCopy}
                title="复制思考内容"
                aria-label="复制思考内容"
              >
                {copied ? <Check size={14} className="text-emerald-500" /> : <Copy size={14} />}
              </button>
            )}

            {shouldShowTimeline && (
              <div className="relative pr-8">
                <span
                  className={`absolute -left-[31px] top-[2px] z-[1] grid h-[22px] w-[22px] place-items-center rounded-full border ring-[3px] ${
                    isProcessRunning
                      ? darkMode
                        ? 'border-[#806052] bg-[#513f38] text-[#ffc5ae] ring-[#2b2e34]'
                        : 'border-[#d8bdb2] bg-[#f6ece7] text-[#98563f] ring-[#faf8f6]'
                      : darkMode
                        ? 'border-white/15 bg-[#444850] text-gray-100 ring-[#2b2e34]'
                        : 'border-[#d8cec7] bg-[#f2ece7] text-[#5c5049] ring-[#faf8f6]'
                  }`}
                  aria-hidden="true"
                >
                  <Lightbulb className="h-[15px] w-[15px]" strokeWidth={2.15} />
                </span>
                {shouldShowStreamingHint && (
                  <p className={`py-0.5 text-[13px] leading-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    正在分析问题并组织回答...
                  </p>
                )}
                <div className={`prose prose-sm max-w-none text-[13.5px] leading-[1.7] ${darkMode ? 'prose-invert text-gray-300' : 'text-gray-500'}`}>
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
            )}

            {hasAgentTrace && <AgentTracePanel trace={agentTrace} embedded darkMode={darkMode} />}
            {hasWebSearchActivity && (
              <WebSearchActivity
                {...webSearchActivity}
                isStreaming={isStreaming}
                darkMode={darkMode}
                embedded
              />
            )}
          </div>
        </div>
      </div>
    </section>
  )
}

export default memo(ThinkingBlock)
