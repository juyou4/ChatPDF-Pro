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

  // 队列状态：存储当前正在动画中的文本块
  // Item format: { id: number, text: string }
  const [queue, setQueue] = useState([]);

  // 记录已处理的内容长度，用于计算增量
  const processedRef = useRef(0);

  // 根据强度获取动画类名
  const getBlurClass = () => {
    switch (blurIntensity) {
      case 'strong': return 'animate-blur-reveal-strong';
      case 'light': return 'animate-blur-reveal-light';
      case 'medium':
      default: return 'animate-blur-reveal-medium';
    }
  };

  useEffect(() => {
    const fullContent = content || '';

    // 处理内容重置或回退的情况
    if (fullContent.length < processedRef.current) {
      processedRef.current = fullContent.length;
      setQueue([]);
      return;
    }

    // 如果没有新内容，直接返回
    if (fullContent.length === processedRef.current) return;

    // 获取新增文本
    const newText = fullContent.slice(processedRef.current);
    processedRef.current = fullContent.length;

    // 检查是否包含可能破坏Markdown结构的字符
    // 包括：换行、加粗、斜体、代码块、列表、引用、标题等标记
    const isStructural = /[\n\*\_\[\]\(\)\#\`\>\-\+\!]/.test(newText);

    if (isStructural) {
      // 如果包含结构性字符，立即清空队列（Flush）
      // 这样所有内容（包括队列中的和新增的）都会立即变为稳定内容被Markdown渲染
      setQueue([]);
    } else {
      // 如果是普通文本，加入队列进行动画
      const id = Date.now() + Math.random();
      const item = { id, text: newText };

      setQueue(prev => [...prev, item]);

      // 设置定时器，动画结束后移除该项
      setTimeout(() => {
        setQueue(prev => prev.filter(i => i.id !== id));
      }, 300); // 必须与CSS动画时长匹配
    }
  }, [content]);

  // 计算稳定内容：总内容减去队列中的内容
  const queueTextLength = queue.reduce((acc, item) => acc + item.text.length, 0);
  const stableContentLength = Math.max(0, (content || '').length - queueTextLength);
  const stableContent = (content || '').slice(0, stableContentLength);

  return (
    <div className={`streaming-active prose prose-sm max-w-none dark:prose-invert`}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {stableContent}
      </ReactMarkdown>

      {/* 队列部分：渲染正在动画的文本块 */}
      {queue.length > 0 && (
        <span className="typing-buffer">
          {queue.map(item => (
            <span key={item.id} className={`inline-block ${getBlurClass()}`}>
              {item.text}
            </span>
          ))}
        </span>
      )}
    </div>
  );
};

export default StreamingMarkdown;
