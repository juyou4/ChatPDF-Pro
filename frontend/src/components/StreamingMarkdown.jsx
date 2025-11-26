import React, { useRef, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';

/**
 * StreamingMarkdown - 支持实时Markdown渲染和微妙Blur Reveal效果的组件
 *
 * 策略：
 * - 始终实时渲染完整的Markdown内容（保证顺序正确）
 * - 流式输出时在整个容器上应用微妙的脉冲效果
 * - 使用节流避免过于频繁的动画触发
 */
const StreamingMarkdown = ({
  content,
  isStreaming,
  enableBlurReveal,
  blurIntensity = 'medium'
}) => {
  const [showPulse, setShowPulse] = useState(false);
  const previousContentRef = useRef('');
  const pulseTimeoutRef = useRef(null);
  const lastUpdateRef = useRef(0);

  // 监听内容变化，触发脉冲效果
  useEffect(() => {
    if (!isStreaming || !enableBlurReveal) {
      setShowPulse(false);
      return;
    }

    const currentContent = content || '';

    // 内容发生变化
    if (currentContent !== previousContentRef.current) {
      previousContentRef.current = currentContent;

      // 节流：至少间隔50ms才触发一次脉冲
      const now = Date.now();
      if (now - lastUpdateRef.current < 50) {
        return;
      }
      lastUpdateRef.current = now;

      // 触发脉冲
      setShowPulse(true);

      // 清除之前的定时器
      if (pulseTimeoutRef.current) {
        clearTimeout(pulseTimeoutRef.current);
      }

      // 100ms后移除脉冲效果
      pulseTimeoutRef.current = setTimeout(() => {
        setShowPulse(false);
      }, 100);
    }

    return () => {
      if (pulseTimeoutRef.current) {
        clearTimeout(pulseTimeoutRef.current);
      }
    };
  }, [content, isStreaming, enableBlurReveal]);

  // 根据强度和状态选择CSS类
  const getClassName = () => {
    let baseClass = 'prose prose-sm max-w-none dark:prose-invert';

    if (showPulse && isStreaming && enableBlurReveal) {
      switch (blurIntensity) {
        case 'strong':
          return `${baseClass} blur-pulse-strong`;
        case 'light':
          return `${baseClass} blur-pulse-light`;
        case 'medium':
        default:
          return `${baseClass} blur-pulse-medium`;
      }
    }

    return baseClass;
  };

  return (
    <div className={getClassName()}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {content || ''}
      </ReactMarkdown>
    </div>
  );
};

export default StreamingMarkdown;
