import React, { useState, useEffect, useRef, useCallback, useMemo, memo } from 'react'
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
  const activeStageText = useMemo(() => {
    if (!isStreaming || answerStarted) return ''
    const lines = String(content || '')
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
    return lines[lines.length - 1] || '正在等待模型响应...'
  }, [answerStarted, content, isStreaming])

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
    <div className={`group/thinking relative isolate mb-5 mt-2 w-full ${darkMode ? 'dark' : ''}`}>
      <span
        aria-hidden="true"
        className={`pointer-events-none absolute inset-x-10 -bottom-1 h-5 rounded-[50%] blur-[14px] transition-[opacity,transform] duration-300 ease-out group-hover/thinking:translate-y-0.5 group-hover/thinking:scale-x-[1.03] motion-reduce:transition-none ${
          darkMode ? 'bg-black/35 opacity-65' : 'bg-[#574840]/[0.11] opacity-60'
        }`}
      />
      <span
        aria-hidden="true"
        className={`pointer-events-none absolute inset-x-16 bottom-0 h-2 rounded-[50%] blur-[5px] transition-[opacity,transform] duration-300 ease-out group-hover/thinking:translate-y-0.5 group-hover/thinking:scale-x-[1.02] motion-reduce:transition-none ${
          darkMode ? 'bg-black/40 opacity-60' : 'bg-[#302722]/[0.12] opacity-45'
        }`}
      />
      <section
        className={`relative z-[1] w-full -translate-y-px overflow-hidden rounded-[18px] ring-1 ring-inset text-[13px] transition-[transform,background-color,box-shadow] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] hover:-translate-y-[2px] motion-reduce:transform-none motion-reduce:transition-none ${
          darkMode
            ? 'bg-[#25282f] text-gray-300 ring-white/[0.055] shadow-[inset_0_1px_0_rgba(255,255,255,0.045)] hover:bg-[#272a31] hover:ring-white/[0.075]'
            : 'bg-[#fffefd] text-gray-600 ring-[#66574f]/[0.075] shadow-[inset_0_1px_0_rgba(255,255,255,0.95)] hover:bg-white hover:ring-[#66574f]/[0.1]'
        }`}
      >
        <button
          type="button"
          className={`flex w-full items-center gap-2.5 px-3 py-2.5 text-left transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset ${
            darkMode
              ? 'hover:bg-white/[0.035] focus-visible:ring-[#FFA07A]/45'
              : 'hover:bg-black/[0.012] focus-visible:ring-[#D99178]/45'
          }`}
          onClick={() => setExpanded((prev) => !prev)}
          aria-expanded={expanded}
        >
          <span
            className={`flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-[9px] ${
              darkMode ? 'bg-[#FFA07A]/10 text-[#FFA07A]' : 'bg-[#f8f3f0] text-[#a65d49]'
            }`}
          >
            <CellsLoader active={isThinkingPhase} />
          </span>
          <span className={`text-[11px] font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'} ${isThinkingPhase ? 'thinking-text-pulse' : ''}`}>
            Deep Thinking
          </span>
          <span className={`h-3 w-px ${darkMode ? 'bg-white/10' : 'bg-[#e5e7eb]'}`} aria-hidden="true" />
          <span
            className={`min-w-0 flex-1 truncate text-[11px] ${
              darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}
            title={activeStageText || undefined}
          >
            {activeStageText}
          </span>
          <ThinkingTimer isThinking={isThinkingPhase} thinkingMs={thinkingMs || 0} />
          <ChevronDown
            size={15}
            className={`flex-shrink-0 transition-transform duration-300 ${
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
            <div className={`relative border-t px-4 pb-4 pt-3 ${darkMode ? 'border-white/[0.06]' : 'border-[#eceef1]'}`}>
              {!isStreaming && content && (
                <button
                  className={`absolute right-2.5 top-2.5 z-[1] rounded-[8px] p-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                    darkMode
                      ? 'text-gray-500 hover:bg-white/[0.06] hover:text-gray-200 focus-visible:ring-[#FFA07A]/45'
                      : 'text-gray-400 hover:bg-[#f2f3f5] hover:text-gray-700 focus-visible:ring-[#D99178]/45'
                  }`}
                  onClick={handleCopy}
                  title="复制思考内容"
                  aria-label="复制思考内容"
                >
                  {copied ? <Check size={14} className="text-emerald-500" /> : <Copy size={14} />}
                </button>
              )}

              {shouldShowTimeline && (
                <div className={`ml-1.5 border-l py-0.5 pl-3 pr-7 ${darkMode ? 'border-white/[0.1]' : 'border-[#e3e6ea]'}`}>
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
