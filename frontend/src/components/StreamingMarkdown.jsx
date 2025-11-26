import React, { useRef, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';

/**
 * StreamingMarkdown - 支持实时Markdown渲染和Blur Reveal效果的组件
 *
 * 策略：
 * - 仅对最新新增的文本块做模糊动画，其余内容立即归入稳定文本
 * - 动画持续时间根据强度选择（与 CSS 保持一致）
 * - 遇到Markdown结构符号或换行时，立即清空动画，避免格式错乱
 */
const StreamingMarkdown = ({
  content,
  isStreaming,
  enableBlurReveal,
  blurIntensity = 'medium'
}) => {
  const shouldAnimate = enableBlurReveal && isStreaming;

  // 当前正在模糊动画中的文本块 { id, text }
  const [activeChunk, setActiveChunk] = useState(null);

  // 记录已处理的内容长度，用于计算增量
  const processedRef = useRef(0);
  const clearTimerRef = useRef(null);

  // 根据强度获取动画类名
  const getBlurClass = () => {
    switch (blurIntensity) {
      case 'strong': return 'animate-blur-reveal-strong';
      case 'light': return 'animate-blur-reveal-light';
      case 'medium':
      default: return 'animate-blur-reveal-medium';
    }
  };

  const getAnimationDuration = () => {
    switch (blurIntensity) {
      case 'strong': return 300;
      case 'light': return 200;
      case 'medium':
      default: return 250;
    }
  };

  useEffect(() => {
    // 未启用动画或不在流式过程中，重置内部状态
    if (!shouldAnimate) {
      if (clearTimerRef.current) {
        clearTimeout(clearTimerRef.current);
        clearTimerRef.current = null;
      }
      processedRef.current = (content || '').length;
      setActiveChunk(null);
      return;
    }

    const fullContent = content || '';

    // 处理内容重置或回退的情况
    if (fullContent.length < processedRef.current) {
      if (clearTimerRef.current) {
        clearTimeout(clearTimerRef.current);
        clearTimerRef.current = null;
      }
      processedRef.current = fullContent.length;
      setActiveChunk(null);
      return;
    }

    // 如果没有新内容，保持现有动画
    if (fullContent.length === processedRef.current) return;

    // 获取新增文本
    const newText = fullContent.slice(processedRef.current);
    processedRef.current = fullContent.length;

    // 仅针对最新的文本块做动画展示（保留全部字符，包括换行）
    const id = Date.now() + Math.random();
    setActiveChunk({ id, text: newText });

    if (clearTimerRef.current) {
      clearTimeout(clearTimerRef.current);
      clearTimerRef.current = null;
    }

    const duration = getAnimationDuration();
    clearTimerRef.current = setTimeout(() => {
      setActiveChunk(current => (current && current.id === id ? null : current));
    }, duration); // 必须与CSS动画时长匹配
  }, [content, shouldAnimate, blurIntensity]);

  useEffect(() => {
    return () => {
      if (clearTimerRef.current) {
        clearTimeout(clearTimerRef.current);
      }
    };
  }, []);

  // 如果不启用特效或不在流式传输中，直接渲染普通Markdown
  if (!shouldAnimate) {
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

  // 只保留末尾正在动画的文本，其余立即进入稳定内容
  const activeTextLength = activeChunk?.text?.length || 0;
  const stableContentLength = Math.max(0, (content || '').length - activeTextLength);
  const stableContent = (content || '').slice(0, stableContentLength);

  return (
    <div className="streaming-active prose prose-sm max-w-none dark:prose-invert">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {stableContent}
      </ReactMarkdown>

      {/* 动画部分：只渲染当前新增的文本块 */}
      {/* 使用 inline-block 让它尽可能紧跟在 Markdown 内容后面 */}
      {activeChunk?.text && (
        <span className="typing-buffer">
          <span key={activeChunk.id} className={`inline-block ${getBlurClass()}`}>
            {activeChunk.text}
          </span>
        </span>
      )}
    </div>
  );
};

export default StreamingMarkdown;
