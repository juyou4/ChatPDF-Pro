import React, { useRef, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import { motion } from 'framer-motion';

/**
 * StreamingMarkdown - 支持实时Markdown渲染和Blur Reveal效果的组件
 *
 * 策略：
 * - 仅对最新新增的文本块做模糊动画，其余内容立即归入稳定文本
 * - 使用 Framer Motion 做逐字符 Staggered Blur Reveal，避免手写 delay
 * - 遇到重置时清空动画，保持顺序正确
 */
const StreamingMarkdown = ({
  content,
  isStreaming,
  enableBlurReveal,
  blurIntensity = 'medium'
}) => {
  // 如果不启用特效或不在流式传输中，直接渲染普通Markdown
  if (!enableBlurReveal || !isStreaming) {
    return (
      <div className="prose prose-sm max-w-none dark:prose-invert">
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight]}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  }

  // 当前正在做动画的文本块 { id, text }
  const [activeChunk, setActiveChunk] = useState(null);

  // 记录已处理的内容长度，用于计算增量
  const processedRef = useRef(0);
  const clearTimerRef = useRef(null);

  const getAnimationDuration = () => {
    switch (blurIntensity) {
      case 'strong': return 320;
      case 'light': return 200;
      case 'medium':
      default: return 250;
    }
  };

  useEffect(() => {
    // 停止或关闭特效时重置
    if (!enableBlurReveal || !isStreaming) {
      if (clearTimerRef.current) {
        clearTimeout(clearTimerRef.current);
        clearTimerRef.current = null;
      }
      processedRef.current = (content || '').length;
      setActiveChunk(null);
      return;
    }

    const fullContent = content || '';

    // 内容回退时重置
    if (fullContent.length < processedRef.current) {
      processedRef.current = fullContent.length;
      setActiveChunk(null);
      return;
    }

    // 无新增内容
    if (fullContent.length === processedRef.current) return;

    // 获取新增文本
    const newText = fullContent.slice(processedRef.current);
    processedRef.current = fullContent.length;

    // 仅针对最新的文本块做动画展示
    const id = Date.now() + Math.random();
    setActiveChunk({ id, text: newText });

    if (clearTimerRef.current) {
      clearTimeout(clearTimerRef.current);
      clearTimerRef.current = null;
    }

    const duration = getAnimationDuration();
    clearTimerRef.current = setTimeout(() => {
      setActiveChunk(current => (current && current.id === id ? null : current));
    }, duration + 80); // 留少许缓冲避免闪烁
  }, [content, enableBlurReveal, isStreaming, blurIntensity]);

  useEffect(() => {
    return () => {
      if (clearTimerRef.current) {
        clearTimeout(clearTimerRef.current);
      }
    };
  }, []);

  // 计算稳定内容：总内容减去队列中的内容
  const activeTextLength = activeChunk?.text?.length || 0;
  const stableContentLength = Math.max(0, (content || '').length - activeTextLength);
  const stableContent = (content || '').slice(0, stableContentLength);

  // Framer Motion 配置
  const blurConfig = {
    strong: { blur: 10, y: 12, stiffness: 120, damping: 14, stagger: 0.04 },
    medium: { blur: 7, y: 10, stiffness: 110, damping: 13, stagger: 0.035 },
    light: { blur: 5, y: 8, stiffness: 100, damping: 12, stagger: 0.03 },
  }[blurIntensity] || { blur: 7, y: 10, stiffness: 110, damping: 13, stagger: 0.035 };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: blurConfig.stagger },
    },
  };

  const childVariants = {
    hidden: {
      opacity: 0,
      y: blurConfig.y,
      filter: `blur(${blurConfig.blur}px)`,
    },
    visible: {
      opacity: 1,
      y: 0,
      filter: 'blur(0px)',
      transition: {
        type: 'spring',
        damping: blurConfig.damping,
        stiffness: blurConfig.stiffness,
      },
    },
  };

  const animatedChars = activeChunk?.text ? Array.from(activeChunk.text) : [];

  return (
    <div className={`streaming-active prose prose-sm max-w-none dark:prose-invert`}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {stableContent}
      </ReactMarkdown>

      {/* 使用 Framer Motion 对新增文本做逐字符模糊显现 */}
      {animatedChars.length > 0 && (
        <motion.span
          key={activeChunk.id}
          className="typing-buffer inline-flex flex-wrap"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          >
          {animatedChars.map((ch, idx) => (
            <motion.span
              key={`${activeChunk.id}-${idx}`}
              variants={childVariants}
              style={{
                marginRight: ch === ' ' ? '4px' : '0px',
                whiteSpace: 'pre',
                display: 'inline-block',
                willChange: 'filter, transform, opacity'
              }}
            >
              {ch === ' ' ? '\u00A0' : ch}
            </motion.span>
          ))}
        </motion.span>
      )}
    </div>
  );
};

export default StreamingMarkdown;
