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
import { visit } from 'unist-util-visit';

/**
 * Remark Plugin: Blur Reveal
 * 识别新增的文本内容，并为其添加模糊显现动画
 */
const remarkBlurReveal = (options) => {
  const { stableOffset = 0 } = options;

  return (tree) => {
    visit(tree, 'text', (node, index, parent) => {
      if (!node.position) return;

      const start = node.position.start.offset;
      const end = node.position.end.offset;

      // 完全在稳定区域内，不做处理
      if (end <= stableOffset) return;

      // 完全在新增区域
      if (start >= stableOffset) {
        // 将文本拆分为字符以实现逐字显示效果
        // 注意：为了性能，这里可以优化为按词或按块，但逐字效果最好
        const chars = node.value.split('');
        const nodes = chars.map((char, i) => {
          const charOffset = start + i;
          const delay = Math.min((charOffset - stableOffset) * 0.015, 1.0); // 限制最大延迟

          // 如果是空格或换行，直接返回文本节点，不加动画
          // 这可以避免 inline-block 元素导致的排版间距问题
          if (/\s/.test(char)) {
            return { type: 'text', value: char };
          }

          return {
            type: 'text',
            value: char,
            data: {
              hName: 'span',
              hProperties: {
                className: 'animate-blur-reveal',
                style: `animation-delay: ${delay}s`
              }
            }
          };
        });

        parent.children.splice(index, 1, ...nodes);
        return index + nodes.length; // 跳过新插入的节点
      }

      // 跨越边界：部分稳定，部分新增
      if (start < stableOffset && end > stableOffset) {
        const splitPoint = stableOffset - start;
        const stableText = node.value.slice(0, splitPoint);
        const newText = node.value.slice(splitPoint);

        const stableNode = { type: 'text', value: stableText };

        const chars = newText.split('');
        const newNodes = chars.map((char, i) => {
          const charOffset = stableOffset + i;
          const delay = Math.min((charOffset - stableOffset) * 0.015, 1.0);

          return {
            type: 'text',
            value: char,
            data: {
              hName: 'span',
              hProperties: {
                className: 'animate-blur-reveal',
                style: `animation-delay: ${delay}s`
              }
            }
          };
        });

        parent.children.splice(index, 1, stableNode, ...newNodes);
        return index + 1 + newNodes.length;
      }
    });
  };
};

const StreamingMarkdown = ({
  content,
  isStreaming,
  enableBlurReveal,
  blurIntensity = 'medium'
}) => {
  // 记录“稳定”内容的偏移量
  // 只有超过这个偏移量的内容才会被视为“新增”并应用动画
  const [stableOffset, setStableOffset] = useState(0);
  const contentRef = useRef(content);

  // 每次内容更新时，更新 stableOffset
  useEffect(() => {
    if (!enableBlurReveal) {
      setStableOffset((content || '').length);
      return;
    }

    const currentLength = (content || '').length;

    // 如果内容变短（清空或回退），立即重置
    if (currentLength < stableOffset) {
      setStableOffset(currentLength);
      return;
    }

    // 如果内容增加，设置一个定时器将新增部分标记为稳定
    // 这样在当前渲染周期，新增部分会保留动画类
    // 动画结束后，stableOffset 追上 currentLength，动画类移除（或不再生成）
    const timer = setTimeout(() => {
      setStableOffset(currentLength);
    }, 500); // 500ms 足够播放完大部分动画

    return () => clearTimeout(timer);
  }, [content, enableBlurReveal, stableOffset]);

  // 构造插件配置
  // 使用 useMemo 避免不必要的重绘，但 stableOffset 变化时必须更新
  const plugins = React.useMemo(() => {
    const p = [remarkMath];
    if (enableBlurReveal && isStreaming) {
      p.push([remarkBlurReveal, { stableOffset }]);
    }
    return p;
  }, [enableBlurReveal, isStreaming, stableOffset]);

  return (
    <div
      className="prose prose-sm max-w-none dark:prose-invert message-content leading-7"
    >
      <ReactMarkdown
        remarkPlugins={plugins}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default StreamingMarkdown;
