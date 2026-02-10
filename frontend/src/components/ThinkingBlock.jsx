import React, { useState, useEffect, useRef, useMemo, useCallback, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Copy, Check, ChevronRight } from 'lucide-react'
import StreamingMarkdown from './StreamingMarkdown'

/**
 * 原子图标 SVG 组件
 */
const AtomIcon = ({ className, size = 20 }) => (
  <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <ellipse cx="12" cy="12" rx="10" ry="4" />
    <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)" />
    <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)" />
    <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
  </svg>
)

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

  const seconds = ((displayMs < 100 ? 100 : displayMs) / 1000).toFixed(1)

  return (
    <span className="thinking-timer-text">
      {isThinking ? `思考中 ${seconds}s` : `已深度思考 ${seconds}s`}
    </span>
  )
})

/**
 * 思考过程滚动预览组件（cherry-studio 风格）
 * 在思考进行中且面板收起时，显示最新几行思考内容的滚动动画
 */
const ThinkingPreview = memo(({ content, isThinking, expanded }) => {
  const lines = useMemo(() => {
    if (!content) return []
    const allLines = content.split('\n')
    // 思考中时去掉最后一行（可能不完整）
    const validLines = isThinking ? allLines.slice(0, -1) : allLines
    return validLines.filter(l => l.trim() !== '')
  }, [content, isThinking])

  // 只在思考中且面板收起时显示预览
  const showPreview = isThinking && !expanded

  if (!showPreview || lines.length < 1) return null

  const LINE_HEIGHT = 16
  // 只渲染最后 5 行
  const visibleLines = lines.slice(-5)

  return (
    <div className="thinking-preview-container">
      <motion.div
        className="thinking-preview-messages"
        style={{ height: lines.length * LINE_HEIGHT }}
        initial={{ y: -2 }}
        animate={{ y: -lines.length * LINE_HEIGHT - 2 }}
        transition={{ duration: 0.15, ease: 'linear' }}
      >
        {visibleLines.map((line, i) => (
          <div key={lines.length - 5 + i} className="thinking-preview-line" style={{ lineHeight: `${LINE_HEIGHT}px` }}>
            {line}
          </div>
        ))}
      </motion.div>
    </div>
  )
})

/**
 * 深度思考展示组件
 * 参考 cherry-studio 的 ThinkingBlock + ThinkingEffect 设计
 *
 * 特性：
 * - 实时计时器显示思考耗时
 * - 思考中时原子图标脉冲动画 + 内容滚动预览
 * - 可折叠/展开的思考内容面板
 * - 思考完成后支持一键复制
 * - 思考完成后自动折叠
 * - 圆角卡片边框风格
 */
const ThinkingBlock = ({ content, isStreaming, darkMode, thinkingMs }) => {
  const [expanded, setExpanded] = useState(true)
  const [copied, setCopied] = useState(false)
  const wasStreamingRef = useRef(false)

  // 思考完成后自动折叠
  useEffect(() => {
    if (wasStreamingRef.current && !isStreaming && content) {
      // 延迟折叠，让用户看到完成状态
      const timer = setTimeout(() => setExpanded(false), 600)
      return () => clearTimeout(timer)
    }
    wasStreamingRef.current = isStreaming
  }, [isStreaming, content])

  // 复制思考内容
  const handleCopy = useCallback((e) => {
    e.stopPropagation()
    if (!content) return
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [content])

  // 计算预览区高度
  const headerHeight = useMemo(() => {
    if (!isStreaming || expanded) return 40
    const lines = (content || '').split('\n').filter(l => l.trim()).length
    if (lines < 2) return 40
    return Math.min(78, Math.max(lines + 1, 2) * 16 + 26)
  }, [isStreaming, expanded, content])

  return (
    <div className={`thinking-block ${darkMode ? 'dark' : ''}`}>
      {/* 可点击的头部区域 */}
      <div
        className={`thinking-block-header ${expanded ? 'expanded' : ''}`}
        style={{ height: headerHeight }}
        onClick={() => setExpanded(prev => !prev)}
      >
        {/* 原子图标（思考中时脉冲动画） */}
        <div className="thinking-block-icon">
          <motion.div
            animate={isStreaming ? {
              opacity: [1, 0.3, 1],
              transition: { duration: 1.2, ease: 'easeInOut', times: [0, 0.5, 1], repeat: Infinity }
            } : { opacity: 1 }}
          >
            <AtomIcon
              className={isStreaming ? 'text-amber-500' : 'text-gray-400 dark:text-gray-500'}
              size={isStreaming && !expanded ? 26 : 20}
            />
          </motion.div>
        </div>

        {/* 标题 + 计时器 + 滚动预览 */}
        <div className="thinking-block-text">
          <div className="thinking-block-title">
            <ThinkingTimer isThinking={isStreaming} thinkingMs={thinkingMs || 0} />
          </div>
          <ThinkingPreview content={content} isThinking={isStreaming} expanded={expanded} />
        </div>

        {/* 右侧箭头 */}
        <div className={`thinking-block-arrow ${expanded ? 'expanded' : ''}`}>
          <ChevronRight size={16} />
        </div>
      </div>

      {/* 展开的思考内容 */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className="thinking-block-body-wrapper"
          >
            <div className={`thinking-block-body ${darkMode ? 'dark' : ''}`}>
              {/* 复制按钮（仅思考完成后显示） */}
              {!isStreaming && content && (
                <button
                  className="thinking-copy-btn"
                  onClick={handleCopy}
                  title="复制思考内容"
                >
                  {copied ? <Check size={13} className="text-green-500" /> : <Copy size={13} />}
                </button>
              )}
              <div className="thinking-block-content">
                <StreamingMarkdown
                  content={content}
                  isStreaming={isStreaming}
                  enableBlurReveal={false}
                  blurIntensity="light"
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default memo(ThinkingBlock)
