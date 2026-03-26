import React, { useState, useEffect, useRef, useCallback, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Copy, Check, ChevronDown, BrainCircuit } from 'lucide-react'
import StreamingMarkdown from './StreamingMarkdown'
import { useChatParams } from '../contexts/ChatParamsContext'

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
const ThinkingBlock = ({ content, isStreaming, darkMode, thinkingMs, streamingRef }) => {
  const [expanded, setExpanded] = useState(true)
  const [copied, setCopied] = useState(false)
  const [hasStreamingText, setHasStreamingText] = useState(false)
  const wasStreamingRef = useRef(false)
  const { thoughtAutoCollapse } = useChatParams()

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
  const shouldShowStreamingHint = isStreaming && !(content && content.trim()) && !hasStreamingText

  return (
    <div className={`flex flex-col items-start gap-2 my-2 ${darkMode ? 'dark' : ''}`}>
      {/* Expandable Pill (开关样式，与展开状态绑定) */}
      <div 
        className={`flex items-center gap-2 px-3 py-1.5 rounded-full border shadow-sm cursor-pointer transition-colors select-none ${darkMode ? 'bg-gray-800 border-gray-700 hover:bg-gray-700' : 'bg-[#f4f5f9] border-gray-200/60 hover:bg-gray-100'}`}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Toggle Switch 视觉还原 */}
        <div className={`w-8 h-4 rounded-full flex items-center relative shadow-inner transition-colors duration-300 ${expanded ? 'bg-[#7c3aed]' : (darkMode ? 'bg-gray-600' : 'bg-gray-300')}`}>
           <div className={`w-3 h-3 bg-white rounded-full absolute shadow-sm transition-all duration-300 ${expanded ? 'right-0.5' : 'left-0.5'}`}></div>
        </div>

        <span className={`text-[11px] font-medium ${darkMode ? 'text-gray-300' : 'text-gray-800'}`}>Deep Thinking</span>
        
        <div className="flex items-center">
          <ThinkingTimer isThinking={isStreaming} thinkingMs={thinkingMs || 0} />
          <ChevronDown size={14} className={`ml-1 transition-transform duration-300 ${darkMode ? 'text-gray-500' : 'text-gray-400'} ${expanded ? 'rotate-180' : ''}`} />
        </div>
      </div>

      {/* Expanded Thought Process */}
      <div 
        className={`overflow-hidden transition-all duration-300 ease-in-out origin-top w-full ${
          expanded ? 'max-h-[3000px] opacity-100 scale-100 mb-2' : 'max-h-0 opacity-0 scale-95 mb-0'
        }`}
      >
        <div className={`backdrop-blur-sm border shadow-[0_4px_15px_rgba(124,58,237,0.06)] rounded-2xl p-4 w-full text-[13px] relative ml-1 ${darkMode ? 'bg-gray-800/90 border-purple-900/50 text-gray-300' : 'bg-white/90 border-purple-100 text-gray-600'}`}>
          
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-1.5 text-purple-600 font-medium">
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
          
          <div className={`pl-2 border-l-2 py-0.5 ${darkMode ? 'border-purple-900/50' : 'border-purple-100/50'} ml-1.5`}>
            <div className="relative">
              {/* 装饰小圆点 */}
              <span className={`absolute -left-[13px] top-1.5 w-2 h-2 rounded-full ${isStreaming ? 'bg-purple-500 shadow-[0_0_8px_rgba(124,58,237,0.4)]' : 'bg-purple-200 dark:bg-purple-800/50'}`}></span>
              {shouldShowStreamingHint && (
                <div className={`mb-2 text-[12px] leading-relaxed italic ${darkMode ? 'text-purple-300/90' : 'text-purple-500/90'}`}>
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
      </div>
      </div>
    </div>
  )
}

export default memo(ThinkingBlock)
