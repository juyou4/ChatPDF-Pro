import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import { visit } from 'unist-util-visit';

/**
 * StreamingMarkdown - 支持实时Markdown渲染和Blur Reveal效果的组件
 * 
 * 优化策略: 按词拆分 + React.memo 避免不必要的父级重渲染
 */

// Helper to escape HTML special characters
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Remark Plugin: Word-Based Blur Reveal
 * 按词拆分，只对新增的词应用动画
 */
const remarkBlurReveal = (options) => {
  const { stableOffset = 0 } = options;

  return (tree) => {
    visit(tree, 'text', (node, index, parent) => {
      if (!node.position) return;

      const start = node.position.start.offset;
      const end = node.position.end.offset;

      // Completely stable - no modification needed
      if (end <= stableOffset) return;

      // Completely new content - animate all words
      if (start >= stableOffset) {
        // Split by word boundaries (keep whitespace as separators)
        const parts = node.value.split(/(\s+)/);
        let currentOffset = start;

        const nodes = parts.map((part, i) => {
          // Whitespace - just return as text
          if (/^\s+$/.test(part)) {
            return { type: 'text', value: part };
          }

          // Word - wrap in animated span
          // Calculate delay relative to the *start of the new content chunk*
          // This ensures each new "chunk" from the server gets its own mini-sequence
          const relativeIndex = i / 2; // Roughly every other part is a word
          const delay = Math.min(relativeIndex * 0.04, 0.4); // 40ms per word

          return {
            type: 'html',
            value: `<span class="animate-blur-reveal" style="animation-delay: ${delay.toFixed(2)}s;">${escapeHtml(part)}</span>`
          };
        });

        parent.children.splice(index, 1, ...nodes);
        return index + nodes.length;
      }

      // Overlapping: split at stableOffset
      if (start < stableOffset && end > stableOffset) {
        const splitPoint = stableOffset - start;
        const stableText = node.value.slice(0, splitPoint);
        const newText = node.value.slice(splitPoint);

        const stableNode = { type: 'text', value: stableText };

        // Split new text by words
        const parts = newText.split(/(\s+)/);

        const newNodes = parts.map((part, i) => {
          if (/^\s+$/.test(part)) {
            return { type: 'text', value: part };
          }

          const relativeIndex = i / 2;
          const delay = Math.min(relativeIndex * 0.04, 0.4);

          return {
            type: 'html',
            value: `<span class="animate-blur-reveal" style="animation-delay: ${delay.toFixed(2)}s;">${escapeHtml(part)}</span>`
          };
        });

        parent.children.splice(index, 1, stableNode, ...newNodes);
        return index + 1 + newNodes.length;
      }
    });
  };
};

const StreamingMarkdown = React.memo(({
  content,
  isStreaming,
  enableBlurReveal,
  blurIntensity = 'medium'
}) => {
  const lastProcessedLengthRef = useRef(0);

  let stableOffset = lastProcessedLengthRef.current;
  const currentLength = (content || '').length;

  // Reset on content clear (new chat)
  if (currentLength < stableOffset) {
    stableOffset = 0;
    lastProcessedLengthRef.current = 0;
  }

  // Update ref after render
  useEffect(() => {
    if (enableBlurReveal) {
      lastProcessedLengthRef.current = currentLength;
    } else {
      lastProcessedLengthRef.current = 0;
    }
  });

  // Build plugins
  const remarkPlugins = React.useMemo(() => {
    const plugins = [remarkMath];
    if (enableBlurReveal && isStreaming) {
      plugins.push([remarkBlurReveal, { stableOffset }]);
    }
    return plugins;
  }, [enableBlurReveal, isStreaming, stableOffset, content?.length]); // Add content.length to force new plugin instance on update

  const rehypePlugins = React.useMemo(() => {
    return [rehypeRaw, rehypeKatex, rehypeHighlight];
  }, []);

  const intensityClass = enableBlurReveal ? `blur-intensity-${blurIntensity}` : '';

  return (
    <div className={`prose prose-sm max-w-none dark:prose-invert message-content leading-7 ${intensityClass}`}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        remarkRehypeOptions={{ allowDangerousHtml: true }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}, (prevProps, nextProps) => {
  // Only re-render if essential props change
  // This helps performance by preventing re-renders from unrelated parent state changes
  return (
    prevProps.content === nextProps.content &&
    prevProps.isStreaming === nextProps.isStreaming &&
    prevProps.enableBlurReveal === nextProps.enableBlurReveal
  );
});

export default StreamingMarkdown;
