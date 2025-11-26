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
 * - 流式输出时：通过动态添加/移除CSS类触发blur reveal动画
 * - 不重新挂载组件，避免震动
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
  const [showAnimation, setShowAnimation] = useState(false);
  const previousContentRef = useRef('');
  const containerRef = useRef(null);

  // 当内容更新时触发动画
  useEffect(() => {
    if (isStreaming && enableBlurReveal && content !== previousContentRef.current) {
      previousContentRef.current = content;

      // 移除动画类
      setShowAnimation(false);

      // 在下一帧添加动画类，重新触发动画
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setShowAnimation(true);
        });
      });
    } else if (!isStreaming) {
      // 流式结束，移除动画
      setShowAnimation(false);
      previousContentRef.current = '';
    }
  }, [content, isStreaming, enableBlurReveal]);

  // 根据强度选择CSS类
  const getBlurClass = () => {
    if (!showAnimation) return '';

    switch (blurIntensity) {
      case 'strong': return 'blur-reveal-strong';
      case 'medium': return 'blur-reveal-medium';
      case 'light': return 'blur-reveal-light';
      default: return 'blur-reveal-medium';
    }
  };

  return (
    <div
      ref={containerRef}
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
