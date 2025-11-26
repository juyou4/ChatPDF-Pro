import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';

/**
 * StreamingMarkdown - 支持逐字符模糊到清晰效果的Markdown渲染组件
 *
 * 策略：
 * - 流式输出时：显示纯文本，每个新字符应用模糊到清晰动画（真正的逐字符效果）
 * - 流式完成后：渲染完整的Markdown（包含语法高亮、LaTeX等）
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
  const previousLengthRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) {
      previousLengthRef.current = 0;
    }
  }, [isStreaming]);

  // 根据强度选择CSS类
  const getBlurClass = () => {
    switch (blurIntensity) {
      case 'strong': return 'blur-reveal-char';
      case 'medium': return 'blur-reveal-char-medium';
      case 'light': return 'blur-reveal-char-light';
      default: return 'blur-reveal-char-medium';
    }
  };

  // 如果流式已完成，渲染完整的Markdown
  if (!isStreaming) {
    previousLengthRef.current = 0;
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

  // 流式输出时，如果不启用blur效果，也直接渲染Markdown
  if (!enableBlurReveal) {
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

  // 流式输出且启用blur效果：渲染纯文本，每个新字符应用动画
  const previousLength = previousLengthRef.current;
  previousLengthRef.current = content.length;

  const existingText = content.slice(0, previousLength);
  const newText = content.slice(previousLength);
  const blurClass = getBlurClass();

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert whitespace-pre-wrap font-mono text-sm">
      {/* 已有的文本（无动画） */}
      <span>{existingText}</span>

      {/* 新增的文本（每个字符应用blur reveal动画） */}
      {newText.split('').map((char, index) => (
        <span
          key={`char-${previousLength + index}-${Date.now()}`}
          className={blurClass}
          style={{
            animationDelay: `${index * 5}ms`
          }}
        >
          {char === '\n' ? <br /> : char}
        </span>
      ))}
    </div>
  );
};

export default StreamingMarkdown;
