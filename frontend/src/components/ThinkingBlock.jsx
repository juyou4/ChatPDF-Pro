import React, { useState, useEffect, useRef, useCallback, memo } from 'react'
import { Copy, Check, ChevronDown, BrainCircuit } from 'lucide-react'
import StreamingMarkdown from './StreamingMarkdown'
import { useChatParams } from '../contexts/ChatParamsContext'
import AgentTracePanel from './AgentTracePanel'

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
    <span className="text-[10px] text-gray-500 bg-gray-200/50 px-1.5 py-0.5 rounded-md ml-1 whitespace-nowrap dark:bg-gray-700/50 dark:text-gray-400">
      {isThinking ? `思考中 ${seconds}s` : `已深度思考 ${seconds}s`}
    </span>
  )
})

/**
 * 深度思考展示组件
 * 采用全新的胶囊式开关和展开面板 UI 设计
 */
const ThinkingBlock = ({ content, isStreaming, darkMode, thinkingMs, streamingRef, agentTrace }) => {
  const [expanded, setExpanded] = useState(true)
  const [copied, setCopied] = useState(false)
  const [hasStreamingText, setHasStreamingText] = useState(false)
  const [progressLineCount, setProgressLineCount] = useState(1)
  const wasStreamingRef = useRef(false)
  const { thoughtAutoCollapse } = useChatParams()
  const hasAgentTrace = Boolean(agentTrace && agentTrace.enabled)

  const getProgressLineCount = useCallback((text = '') => {
    const lines = String(text)
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0)
    return Math.max(1, lines.length)
  }, [])

  // 思考完成后自动折叠（受 thoughtAutoCollapse 设置控制）
  useEffect(() => {
    if (wasStreamingRef.current && !isStreaming && content && thoughtAutoCollapse) {
      // 延迟折叠，让用户看到完成状态
      const timer = setTimeout(() => setExpanded(false), 600)
      return () => clearTimeout(timer)
    }
    wasStreamingRef.current = isStreaming
  }, [isStreaming, content, thoughtAutoCollapse])

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
  const shouldShowStreamingHint = isStreaming && !(content && content.trim()) && !hasStreamingText
  const shouldShowTimeline = shouldShowStreamingHint || isStreaming || Boolean(content && content.trim())
  const timelineDotOffset = Math.min(280, Math.max(0, (progressLineCount - 1) * 23))

  return (
    <div className={`flex flex-col items-start gap-2 my-2 ${darkMode ? 'dark' : ''}`}>
      {/* 紧凑折叠头：展开状态由箭头表达，避免同时出现开关和折叠两套语义。 */}
      <button
        type="button"
        className={`inline-flex items-center gap-2 rounded-[10px] border px-2.5 py-1.5 cursor-pointer transition-colors select-none ${darkMode ? 'bg-white/[0.04] border-white/[0.08] hover:bg-white/[0.07]' : 'bg-[#f6f2ef] border-transparent hover:bg-[#f1ebe7]'}`}
        onClick={() => setExpanded(!expanded)}
      >
        <span className={`flex h-6 w-6 items-center justify-center rounded-[7px] ${darkMode ? 'bg-[#FFA07A]/10 text-[#FFA07A]' : 'bg-[#FFF0E9] text-[#B85F47]'}`}>
          <BrainCircuit size={13} />
        </span>
        <span className={`text-[11px] font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Deep Thinking</span>
        
        <div className="flex items-center">
          <ThinkingTimer isThinking={isStreaming} thinkingMs={thinkingMs || 0} />
          <ChevronDown size={14} className={`ml-1 transition-transform duration-300 ${darkMode ? 'text-gray-500' : 'text-gray-400'} ${expanded ? 'rotate-180' : ''}`} />
        </div>
      </button>

      {/* Expanded Thought Process */}
      <div 
        className={`overflow-hidden transition-all duration-300 ease-in-out origin-top w-full ${
          expanded ? 'max-h-[3000px] opacity-100 scale-100 mb-2' : 'max-h-0 opacity-0 scale-95 mb-0'
        }`}
      >
        <div className={`relative w-full rounded-[14px] border p-4 text-[13px] ${darkMode ? 'border-white/[0.09] bg-[#25282f] text-gray-300 shadow-[0_14px_32px_-20px_rgba(0,0,0,0.68),inset_0_1px_0_rgba(255,255,255,0.04)]' : 'border-[#eadfd8] bg-white text-gray-600 shadow-[0_12px_30px_-20px_rgba(78,64,56,0.38),0_2px_7px_-4px_rgba(78,64,56,0.14),inset_0_1px_0_rgba(255,255,255,0.96)]'}`}>
          
          <div className="flex items-center justify-between mb-3">
            <div className={`flex items-center gap-1.5 font-medium ${darkMode ? 'text-[#FFA07A]' : 'text-[#B85F47]'}`}>
              <BrainCircuit size={15} className={isStreaming ? "animate-pulse" : ""} />
              <span>思考过程</span>
            </div>
            
            {/* 复制按钮 */}
            {!isStreaming && content && (
              <button
                className={`p-1.5 rounded-md transition-colors ${darkMode ? 'text-gray-400 hover:text-gray-200 hover:bg-gray-700' : 'text-gray-400 hover:text-gray-700 hover:bg-gray-100'}`}
                onClick={handleCopy}
                title="复制思考内容"
              >
                {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
              </button>
            )}
          </div>
          
          {shouldShowTimeline && (
            <div className={`ml-1.5 border-l py-0.5 pl-3 ${darkMode ? 'border-[#FFA07A]/20' : 'border-[#eaded8]'}`}>
              <div className="relative">
                {/* 活动进度点：随检索/思考阶段行数下移，避免一直停在第一行。 */}
                <span
                  className={`absolute -left-[17px] top-1.5 w-2 h-2 rounded-full transition-transform duration-300 ease-out ${isStreaming ? 'bg-[#B85F47] shadow-[0_0_8px_rgba(184,95,71,0.28)]' : 'bg-[#FFDCCF] dark:bg-[#FFA07A]/30'}`}
                  style={{ transform: `translateY(${timelineDotOffset}px)` }}
                />
                {shouldShowStreamingHint && (
                  <div className={`mb-2 text-[12px] leading-relaxed italic ${darkMode ? 'text-[#FFA07A]/90' : 'text-[#B85F47]/90'}`}>
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
        </div>

        {hasAgentTrace && (
          <AgentTracePanel trace={agentTrace} embedded />
        )}
      </div>
    </div>
  )
}

export default memo(ThinkingBlock)
