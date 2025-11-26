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
 * - 流式输出时：如果启用blur reveal，对整个容器应用渐入动画
 * - 流式完成后：移除动画效果
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
  const [contentVersion, setContentVersion] = useState(0);
  const previousContentRef = useRef('');

  // 当内容更新时触发版本号变化（用于重新触发动画）
  useEffect(() => {
    if (isStreaming && content !== previousContentRef.current) {
      previousContentRef.current = content;
      setContentVersion(v => v + 1);
    }
  }, [content, isStreaming]);

  // 根据强度选择动画持续时间和模糊程度
  const getAnimationStyle = () => {
    if (!isStreaming || !enableBlurReveal) return {};

    switch (blurIntensity) {
      case 'strong':
        return {
          animation: 'blurRevealContainer 0.3s ease-out',
        };
      case 'medium':
        return {
          animation: 'blurRevealContainer 0.25s ease-out',
        };
      case 'light':
        return {
          animation: 'blurRevealContainer 0.2s ease-out',
        };
      default:
        return {
          animation: 'blurRevealContainer 0.25s ease-out',
        };
    }
  };

  return (
    <div
      key={contentVersion}
      className="prose prose-sm max-w-none dark:prose-invert"
      style={getAnimationStyle()}
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
