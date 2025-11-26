import React, { useRef, useEffect, useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';

/**
 * StreamingMarkdown - 支持实时Markdown渲染和Blur Reveal效果的组件
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
 * - 使用队列机制管理新生成的文本块
 * - 每个文本块独立保留在队列中直到动画结束（300ms）
 * - 动画结束后，文本块从队列移除，自然归入稳定内容
 * - 遇到Markdown结构符号或换行时，立即清空队列（Flush），避免格式错乱
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
  // 注意：这里假设队列中的内容总是位于总内容的末尾。
  // 由于我们是按顺序添加和处理的，且Flush会清空队列，这个假设通常成立。
  const queueTextLength = queue.reduce((acc, item) => acc + item.text.length, 0);
  const stableContentLength = Math.max(0, (content || '').length - queueTextLength);
  const stableContent = (content || '').slice(0, stableContentLength);

  // 根据强度获取动画类名
  const getBlurClass = () => {
    switch (blurIntensity) {
      case 'strong': return 'animate-blur-reveal-strong';
      case 'light': return 'animate-blur-reveal-light';
      case 'medium':
      default: return 'animate-blur-reveal-medium';
    }
  };

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      {/* 稳定部分：使用Markdown渲染 */}
      {/* 使用 span 包裹以保持行内布局，但这取决于 Markdown 渲染结果 */}
      {/* 如果 stableContent 包含块级元素，ReactMarkdown 会生成 div/p，这可能会导致换行 */}
      {/* 这是一个权衡：为了 Markdown 正确性，我们允许块级元素换行。*/}
      {/* 对于行内追加的文本，ReactMarkdown 通常会渲染在最后一个 p 标签内... 不，它会闭合 p 标签。*/}
      {/* 这是一个已知限制：Markdown流式渲染很难完美处理行内追加动画。*/}
      {/* 但由于我们 Flush 了所有结构性字符（包括换行），队列中通常只有纯文本。*/}
      {/* 纯文本追加到 Markdown 末尾，通常应该紧跟在最后。*/}
      {/* 为了尽量减少视觉跳动，我们只渲染 Markdown。*/}

      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          // 尝试让段落以 inline 方式渲染，以便队列文本能接在后面？
          // 这会破坏多段落布局。
          // 所以我们保持默认渲染。
          // 队列文本只能作为"独立块"跟在后面，或者绝对定位？
          // 绝对定位很难对齐。
          // inline-block 跟在后面是最佳选择，前提是前一个元素是行内元素。
        }}
      >
        {stableContent}
      </ReactMarkdown>

      {/* 队列部分：渲染正在动画的文本块 */}
      {/* 使用 inline-block 让它们尽可能紧跟在 Markdown 内容后面 */}
      {/* 注意：如果 Markdown 以块级元素结尾（如 </p>），这些 span 会换行显示。*/}
      {/* 这是目前架构下的妥协。要解决这个问题需要深入 ReactMarkdown 的渲染树。*/}
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
