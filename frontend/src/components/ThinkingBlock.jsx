import React, { useState, useEffect, useLayoutEffect, useRef, useCallback, useMemo, memo } from 'react'
import { Copy, Check, ChevronDown, Lightbulb } from 'lucide-react'
import StreamingMarkdown from './StreamingMarkdown'
import { useChatParams } from '../contexts/ChatParamsContext'
import AgentTracePanel from './AgentTracePanel'
import CellsLoader from './CellsLoader'
import WebSearchActivity from './WebSearchActivity'
import { getReasoningFallbackText } from '../services/reasoningEffortService'

const PROCESS_STAGE_LINE_PATTERNS = [
  /^正在检索(?:文档)?(?:[，,.:：\s…]|$)/,
  /^正在理解问题并确定检索路线/,
  /^正在启动多轮检索代理/,
  /^正在分析问题(?:，|,)?规划(?:多轮)?检索/,
  /^正在分析查询并确定检索策略/,
  /^已生成检索查询/,
  /^查询向量缓存命中/,
  /^正在进行向量召回/,
  /^正在扩展检索问题/,
  /^正在进行关键词召回/,
  /^正在融合语义意群结果/,
  /^语义意群融合完成/,
  /^检索到以下相关片段/,
  /^上下文准备完成/,
  /^思考完成/,
  /^模型正在准备思考内容/,
  /^第\s*\d+\s*轮(?:取材|检索)/,
  /^LLM\s*规划中/i,
  /^执行\s+(?:search_document|web_search|vector_search|keyword_search|grep|regex_search|boolean_search|visual_search|read_blocks|read_section|read_around|fetch|map|analyze_visual_evidence)/,
  /^跳过重复检索\s*[:：]/,
  /^检索完成(?:[，,.:：\s…]|$)/,
  /^正在整理上下文/,
  /^正在等待模型输出思考内容/,
  /^模型仍在处理，正在等待可见输出/,
  /^模型仍在处理，正在等待首段思考内容/,
  /^思考内容已返回，正在等待回答正文/,
  /^(?:规划结果为空|仍有子问题未覆盖|工具调用达到上限|检索规划失败|规划失败)/,
]

export const isProcessStageLine = (line) => {
  const text = String(line || '').trim()
  if (!text) return false
  return PROCESS_STAGE_LINE_PATTERNS.some((pattern) => pattern.test(text))
}

export const isProcessStageOnlyContent = (content) => {
  const lines = String(content || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
  return lines.length > 0 && lines.every(isProcessStageLine)
}

export const extractProcessStageEntries = (content) => (
  String(content || '')
    .split(/\r?\n/)
    .map((text) => text.trim())
    .filter(isProcessStageLine)
    .map((text, index) => ({
      key: `thinking-stage-${index}`,
      text,
    }))
)

export const stripLeadingProcessStageLines = (content) => {
  const lines = String(content || '').split(/\r?\n/)
  let start = 0
  while (start < lines.length) {
    const trimmed = lines[start].trim()
    if (!trimmed || isProcessStageLine(trimmed)) {
      start += 1
      continue
    }
    break
  }
  return lines.slice(start).join('\n').replace(/^\n+/, '')
}

export const shouldHideRetrievalLog = ({
  answerStarted = false,
  hasRealThinking = false,
  retrievalEntries = [],
} = {}) => {
  if (answerStarted || hasRealThinking) return true
  const lastText = String(retrievalEntries[retrievalEntries.length - 1]?.text || '').trim()
  return /思考完成|正在生成回答|上下文准备完成/.test(lastText)
}

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
    <span className="whitespace-nowrap text-[13.5px] font-medium tabular-nums text-[#4a453f] dark:text-gray-200">
      {isThinking
        ? `${activityOnly ? '正在检索与整理' : '正在思考'} · ${seconds} 秒`
        : hasMeasuredDuration
          ? `${activityOnly ? '处理了' : '思考了'} ${seconds} 秒`
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
  answerGenerating = false,
  darkMode,
  thinkingMs,
  streamingRef,
  retrievalProgress = [],
  agentTrace,
  liveMessageId = null,
  webSearchActivity = null,
  reasoningResolution = null,
  thinkingLive = false,
}) => {
  const [expanded, setExpanded] = useState(true)
  const [copied, setCopied] = useState(false)
  const [hasStreamingText, setHasStreamingText] = useState(false)
  const [thinkingRevealComplete, setThinkingRevealComplete] = useState(!isStreaming)
  const autoCollapsedRef = useRef(false)
  const { thoughtAutoCollapse } = useChatParams()
  const hasAgentTrace = Boolean(agentTrace && agentTrace.enabled)
  const hasWebSearchActivity = Boolean(webSearchActivity)
  const visibleRetrievalProgress = useMemo(() => {
    const seen = new Set()
    return (Array.isArray(retrievalProgress) ? retrievalProgress : [])
      .map((entry, index) => ({
        key: String(entry?.key || `retrieval-${index}`),
        text: String(entry?.text || '').trim(),
        phase: String(entry?.phase || ''),
      }))
      .filter((entry) => {
        if (!entry.text || seen.has(entry.key)) return false
        seen.add(entry.key)
        return true
      })
  }, [retrievalProgress])
  const reasoningFallbackText = useMemo(
    () => getReasoningFallbackText(reasoningResolution),
    [reasoningResolution]
  )
  const reasoningOutputMissing = Boolean(
    reasoningResolution?.enabled && reasoningResolution?.output_observed === false
  )
  // 检索阶段文案会进 thinking；真正的模型思考要单独展示，不能和检索日志叠在一起。
  const visibleThinkingContent = useMemo(() => {
    if (isProcessStageOnlyContent(content)) return ''
    return stripLeadingProcessStageLines(String(content || ''))
  }, [content])
  const hasThinkingContent = Boolean(visibleThinkingContent.trim())
  const hasVisibleThinkingText = hasThinkingContent || hasStreamingText
  const hasRealThinking = hasVisibleThinkingText || Boolean(thinkingLive)
  const retrievalEntries = useMemo(() => {
    const seen = new Set(visibleRetrievalProgress.map((entry) => entry.text))
    const extras = extractProcessStageEntries(content).filter((entry) => {
      if (!entry.text || seen.has(entry.text)) return false
      seen.add(entry.text)
      return true
    })
    return extras.length ? [...visibleRetrievalProgress, ...extras] : visibleRetrievalProgress
  }, [content, visibleRetrievalProgress])
  const hideRetrievalLog = shouldHideRetrievalLog({
    answerStarted,
    hasRealThinking,
    retrievalEntries,
  })
  const showRetrievalDetails = retrievalEntries.length > 0 && !hideRetrievalLog
  const latestRetrievalEntry = retrievalEntries[retrievalEntries.length - 1] || null
  const liveModelStageText = useMemo(() => {
    if (!isStreaming || !latestRetrievalEntry) return ''
    const phase = String(latestRetrievalEntry.phase || '')
    const text = String(latestRetrievalEntry.text || '').trim()
    if (
      phase === 'llm_answer_waiting'
      || phase === 'llm_structuring_citations'
      || (phase === 'llm_waiting' && !hasRealThinking)
    ) return text
    return ''
  }, [hasRealThinking, isStreaming, latestRetrievalEntry])
  const hasProcessDetails = hasThinkingContent || retrievalEntries.length > 0 || hasAgentTrace || hasWebSearchActivity || Boolean(reasoningFallbackText)
  const agentRunning = Boolean(agentTrace?.enabled && agentTrace?.startedAt && !agentTrace?.endedAt)
  const webSearchRunning = Boolean(
    isStreaming && String(webSearchActivity?.status?.phase || '').toLowerCase() === 'searching'
  )
  const activeStageText = useMemo(() => {
    if ((!isStreaming || (answerStarted && !answerGenerating)) && !agentRunning && !webSearchRunning) return ''
    if (webSearchRunning) return '正在联网查找可核验的来源...'
    const currentTask = String(agentTrace?.taskStatus?.current || '').trim()
    // Agent 的当前阶段由结构化 taskStatus 提供，不再回退到旧 thinking 文本，
    // 否则历史“跳过重复检索”等状态会与时间线重复出现在标题栏。
    if (agentRunning) return currentTask || '正在检索与整理证据...'
    if (liveModelStageText) return liveModelStageText
    if (hideRetrievalLog) {
      // 检索日志收起后，在真正思考文本到来前把等待说明放在标题里，
      // 不要再挂一个只有灯泡、没有字的空时间线。
      if (isStreaming && !hasRealThinking && !hasAgentTrace && !webSearchRunning) {
        return '正在分析问题并组织回答...'
      }
      return ''
    }
    let lineEnd = visibleThinkingContent.length
    while (lineEnd > 0) {
      const newlineIndex = visibleThinkingContent.lastIndexOf('\n', lineEnd - 1)
      const line = visibleThinkingContent.slice(newlineIndex + 1, lineEnd).trim()
      if (line) {
        // “检索完成”是已经结束的阶段，不能继续作为“当前正在做什么”展示。
        // 兼容旧文案中带有 total_ms 的情况，避免思考计时旁出现检索耗时。
        return /^检索完成(?:[，,：:]|\s|$)/.test(line) ? '正在整理上下文...' : line
      }
      if (newlineIndex < 0) break
      lineEnd = newlineIndex
    }
    if (showRetrievalDetails) {
      return retrievalEntries[retrievalEntries.length - 1]?.text || '正在检索文档...'
    }
    return '正在等待模型响应...'
  }, [agentRunning, agentTrace?.taskStatus?.current, answerGenerating, answerStarted, hasAgentTrace, hasRealThinking, hideRetrievalLog, isStreaming, liveModelStageText, retrievalEntries, showRetrievalDetails, visibleThinkingContent, webSearchRunning])

  // 思考阶段是否已结束：流式中以「可见正文首 token」为准。
  // thinking_complete 只会把 answerStarted 提前打开；若还在 answerGenerating
  // 且没有可见正文，过程标题应继续保持「正在思考」，避免空的「查看处理过程」。
  const isThinkingPhase = isStreaming && (!answerStarted || answerGenerating)
  const isWaitingForThinking = isStreaming
    && !hasRealThinking
    && (!answerStarted || answerGenerating)
    && !hasAgentTrace
    && !hasWebSearchActivity
  const isThinkingRevealRunning = isStreaming
    && answerStarted
    && hasThinkingContent
    && !thinkingRevealComplete
  const isProcessRunning = isThinkingPhase
    || isWaitingForThinking
    || isThinkingRevealRunning
    || agentRunning
    || webSearchRunning
  // 无思考文本时仍以整条流结束作为兜底；有真实思考文本时，则在正文已经
  // 开始且字符队列完整展示后立即收起，不必等待回答正文和引用流全部结束。
  const processFinished = hasProcessDetails && !isProcessRunning && !isStreaming
  const streamedThinkingFinished = Boolean(
    isStreaming
    && answerStarted
    && !answerGenerating
    && hasThinkingContent
    && thinkingRevealComplete
    && !agentRunning
    && !webSearchRunning
  )
  const thinkingFinished = processFinished || streamedThinkingFinished

  // 思考完成后自动折叠（受 thoughtAutoCollapse 设置控制）。
  // 只自动折叠一次：用户手动展开后不会被再次收起。
  useEffect(() => {
    if (!thoughtAutoCollapse || autoCollapsedRef.current || !thinkingFinished) return undefined
    autoCollapsedRef.current = true
    // 尽快折叠，避免挡住“思考结束 → 正文开始”的过渡观感。
    const timer = setTimeout(() => setExpanded(false), 180)
    return () => clearTimeout(timer)
  }, [thinkingFinished, thoughtAutoCollapse])

  // 观察流式 DOM 的直接写入内容。首字符只负责显示时间线；自动折叠必须
  // 等 DOM 文本与最终思考文本完全一致，避免队列尚未播完就提前收起。
  useLayoutEffect(() => {
    if (!isStreaming) {
      setHasStreamingText(false)
      setThinkingRevealComplete(Boolean(visibleThinkingContent.trim()))
      return undefined
    }

    setThinkingRevealComplete(false)
    let observer = null
    let rafId = 0
    let cancelled = false
    const targetText = stripLeadingProcessStageLines(visibleThinkingContent).trim()

    const attach = () => {
      if (cancelled) return
      const el = streamingRef?.current
      if (!el) {
        rafId = requestAnimationFrame(attach)
        return
      }

      const detectRevealProgress = () => {
        if (cancelled) return false
        const renderedText = stripLeadingProcessStageLines(String(el.textContent || '')).trim()
        if (renderedText) setHasStreamingText(true)

        const revealComplete = Boolean(
          answerStarted
          && !answerGenerating
          && targetText
          && renderedText === targetText
        )
        setThinkingRevealComplete((previous) => (
          previous === revealComplete ? previous : revealComplete
        ))
        if (revealComplete) observer?.disconnect()
        return revealComplete
      }

      if (detectRevealProgress()) return
      observer = new MutationObserver(detectRevealProgress)
      observer.observe(el, { childList: true, subtree: true, characterData: true })
    }

    attach()
    return () => {
      cancelled = true
      if (rafId) cancelAnimationFrame(rafId)
      observer?.disconnect()
    }
  }, [answerGenerating, answerStarted, isStreaming, streamingRef, visibleThinkingContent])

  // 复制思考内容
  const handleCopy = useCallback((e) => {
    e.stopPropagation()
    if (!visibleThinkingContent) return
    navigator.clipboard.writeText(visibleThinkingContent).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [visibleThinkingContent])

  // 深度思考在首个 reasoning_content 到来前，把等待说明放在标题里。
  // 流式期间时间线仍要挂载（hidden），避免 StreamingMarkdown 卸掉后 thinkingRef 无处可写。
  const shouldMountTimeline = isStreaming || hasRealThinking
  // 流式期间必须挂着可见时间线，ref 才能写字。灯泡等有正文再出现。
  const showThinkingDetails = hasVisibleThinkingText || Boolean(thinkingLive) || isStreaming
  const showProcessBody = Boolean(
    showRetrievalDetails
    || hasVisibleThinkingText
    || hasAgentTrace
    || hasWebSearchActivity
    || reasoningFallbackText
    || (isStreaming && shouldMountTimeline)
  )

  // 普通模型可能不返回 reasoning_content。正文开始后不要继续展示一个
  // “已深度思考 0.1s”的空壳；流式中仍保留，避免只剩三个等待点。
  if (answerStarted && !hasProcessDetails && !isStreaming) return null

  return (
    <section
      className={`group/thinking mb-4 mt-1 w-full max-w-[46rem] text-[13.5px] ${darkMode ? 'dark text-gray-300' : 'text-gray-600'}`}
      aria-label="思考过程"
    >
      <button
        type="button"
        className={`-ml-1 flex min-h-9 max-w-full items-center gap-2.5 rounded-[9px] px-2 py-1.5 text-left transition-[background-color,transform] duration-200 active:scale-[0.995] focus-visible:outline-none focus-visible:ring-2 ${
          darkMode
            ? 'hover:bg-white/[0.04] focus-visible:ring-gray-100/35'
            : 'hover:bg-[#f6f3f1] focus-visible:ring-[#cfc6bd]/70'
        }`}
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span
          className={`grid h-6 w-6 flex-shrink-0 place-items-center ${
            isProcessRunning
              ? darkMode ? 'text-gray-300' : 'text-[#5c564f]'
              : darkMode ? 'text-gray-500' : 'text-[#8a827b]'
          }`}
          aria-hidden="true"
        >
          <CellsLoader active={isProcessRunning} />
        </span>
        <ThinkingTimer
          isThinking={isProcessRunning}
          thinkingMs={thinkingMs || 0}
          activityOnly={reasoningOutputMissing || (!hasRealThinking && (showRetrievalDetails || hasAgentTrace || hasWebSearchActivity))}
        />
        {reasoningFallbackText && (
          <span
            className={`max-w-[18rem] truncate rounded-full px-2 py-0.5 text-[10.5px] font-normal ${darkMode ? 'bg-white/10 text-gray-300' : 'bg-[#efe8e1] text-[#5c564f]'}`}
            title={reasoningFallbackText}
          >
            {reasoningOutputMissing ? '未返回思考文本' : '已按兼容档位执行'}
          </span>
        )}
        {activeStageText && !expanded && (
          <>
            <span className={`h-3 w-px flex-shrink-0 ${darkMode ? 'bg-white/10' : 'bg-[#dedad7]'}`} aria-hidden="true" />
            <span
              className={`min-w-0 flex-1 truncate text-[12.5px] ${darkMode ? 'text-gray-300' : 'text-[#3f3a35]'}`}
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
          <div className={`relative ${showProcessBody ? `ml-[9px] border-l pb-1 pl-5 pt-1.5 ${darkMode ? 'border-white/[0.14]' : 'border-[#d4ccc6]'}` : ''}`}>
            {reasoningFallbackText && (
              <p
                role="status"
                className={`mb-2 rounded-lg px-3 py-2 text-[12px] leading-5 ${darkMode ? 'bg-white/[0.06] text-gray-200' : 'bg-[#f4f2f0] text-[#2c2723]'}`}
              >
                {reasoningFallbackText}
              </p>
            )}
            {!isStreaming && visibleThinkingContent && (
              <button
                className={`absolute right-0 top-1 z-[1] rounded-[7px] p-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                  darkMode
                    ? 'text-gray-500 hover:bg-white/[0.05] hover:text-gray-200 focus-visible:ring-gray-100/35'
                    : 'text-gray-400 hover:bg-[#f6f3f1] hover:text-gray-700 focus-visible:ring-[#cfc6bd]/70'
                }`}
                onClick={handleCopy}
                title="复制思考内容"
                aria-label="复制思考内容"
              >
                {copied ? <Check size={14} className="text-emerald-500" /> : <Copy size={14} />}
              </button>
            )}

            {showRetrievalDetails && (
              <div className="relative pr-8">
                <span
                  className={`absolute -left-[31px] top-[2px] z-[1] grid h-[22px] w-[22px] place-items-center rounded-full border ring-[3px] ${
                    isProcessRunning
                      ? darkMode
                        ? 'border-white/16 bg-[#454b55] text-gray-100 ring-[#2b2e34]'
                        : 'border-[#cfc6bd] bg-[#efe8e1] text-[#4a453f] ring-[#faf8f6]'
                      : darkMode
                        ? 'border-white/12 bg-[#3d424b] text-gray-300 ring-[#2b2e34]'
                        : 'border-[#ddd5cd] bg-[#f5f1ec] text-[#6a635c] ring-[#faf8f6]'
                  }`}
                  aria-hidden="true"
                >
                  <Lightbulb className="h-[15px] w-[15px]" strokeWidth={2.15} />
                </span>
                <div className={`space-y-1 py-0.5 text-[13px] leading-5 ${darkMode ? 'text-gray-300' : 'text-[#4a453f]'}`}>
                  {retrievalEntries.map((entry) => (
                    <p key={entry.key}>{entry.text}</p>
                  ))}
                </div>
              </div>
            )}

            {shouldMountTimeline && (
              <div className={`relative pr-8 ${showRetrievalDetails ? 'mt-2.5' : ''} ${showThinkingDetails ? '' : 'hidden'}`}>
                <span
                  className={`absolute -left-[31px] top-[2px] z-[1] grid h-[22px] w-[22px] place-items-center rounded-full border ring-[3px] ${
                    hasVisibleThinkingText ? '' : 'hidden'
                  } ${
                    isProcessRunning
                      ? darkMode
                        ? 'border-white/16 bg-[#454b55] text-gray-100 ring-[#2b2e34]'
                        : 'border-[#cfc6bd] bg-[#efe8e1] text-[#4a453f] ring-[#faf8f6]'
                      : darkMode
                        ? 'border-white/12 bg-[#3d424b] text-gray-300 ring-[#2b2e34]'
                        : 'border-[#ddd5cd] bg-[#f5f1ec] text-[#6a635c] ring-[#faf8f6]'
                  }`}
                  aria-hidden="true"
                >
                  <Lightbulb className="h-[15px] w-[15px]" strokeWidth={2.15} />
                </span>
                <div className={`prose prose-sm max-w-none text-[13.5px] leading-[1.7] ${darkMode ? 'prose-invert text-gray-200' : 'text-[#2c2723]'} ${hasRealThinking ? '' : 'hidden'}`}>
                  <StreamingMarkdown
                    content={visibleThinkingContent}
                    isStreaming={isStreaming}
                    enableBlurReveal={false}
                    blurIntensity="light"
                    streamingRef={streamingRef}
                    suppressInitialDots
                    hydrateDirectWriteContent={false}
                  />
                </div>
                {!hasRealThinking && isStreaming && !showRetrievalDetails && (
                  <div
                    role="status"
                    aria-live="polite"
                    className={`min-h-6 text-[12.5px] leading-6 ${darkMode ? 'text-gray-400' : 'text-[#6b645d]'}`}
                  >
                    <span>{liveModelStageText || activeStageText || '正在等待模型开始输出思考内容...'}</span>
                  </div>
                )}
                {hasRealThinking && liveModelStageText && (
                  <div
                    role="status"
                    aria-live="polite"
                    className={`mt-2 min-h-6 text-[12px] leading-6 ${darkMode ? 'text-gray-400' : 'text-[#746d66]'}`}
                  >
                    <span>{liveModelStageText}</span>
                  </div>
                )}
              </div>
            )}

            {hasAgentTrace && (
              <AgentTracePanel
                trace={agentTrace}
                liveMessageId={liveMessageId}
                embedded
                darkMode={darkMode}
              />
            )}
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
