import React, { useRef, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';

/**
 * StreamingMarkdown - 支持实时Markdown渲染和Blur Reveal效果的组件
 *
 * 策略：
 * - 始终实时渲染Markdown（避免显示##等原始符号）
 * - 流式输出时：使用CSS transition产生微妙的脉冲效果
 * - 通过节流避免频繁触发，防止闪烁
 *
 * @param {string} content - Markdown内容
 * @param {boolean} isStreaming - 是否正在流式输出
 * @param {boolean} enableBlurReveal - 是否启用blur reveal效果
 * @param {string} blurIntensity - 模糊强度 ('strong', 'medium', 'light')
 */
const StreamingMarkdown = ({
  content,
  isStreaming,
  enableBlurReveal,
  blurIntensity = 'medium'
}) => {
  const [isPulsing, setIsPulsing] = useState(false);
  const previousContentRef = useRef('');
  const pulseTimeoutRef = useRef(null);

  // 当内容更新时触发微妙的脉冲效果
  useEffect(() => {
    if (isStreaming && enableBlurReveal && content !== previousContentRef.current) {
      previousContentRef.current = content;

      // 触发脉冲效果
      setIsPulsing(true);

      // 清除之前的定时器
      if (pulseTimeoutRef.current) {
        clearTimeout(pulseTimeoutRef.current);
      }

      // 脉冲效果持续很短时间后自动消失
      pulseTimeoutRef.current = setTimeout(() => {
        setIsPulsing(false);
      }, 100); // 100ms后恢复正常
    } else if (!isStreaming) {
      // 流式结束，确保移除脉冲效果
      setIsPulsing(false);
      previousContentRef.current = '';
    }

    return () => {
      if (pulseTimeoutRef.current) {
        clearTimeout(pulseTimeoutRef.current);
      }
    };
  }, [content, isStreaming, enableBlurReveal]);

  // 根据强度选择CSS类
  const getBlurClass = () => {
    if (!isPulsing || !isStreaming || !enableBlurReveal) return '';

    switch (blurIntensity) {
      case 'strong': return 'blur-pulse-strong';
      case 'medium': return 'blur-pulse-medium';
      case 'light': return 'blur-pulse-light';
      default: return 'blur-pulse-medium';
    }
  };

  return (
    <div
      className={`prose prose-sm max-w-none dark:prose-invert ${getBlurClass()}`}
    >
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default StreamingMarkdown;
