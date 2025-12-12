import React, { useRef, useEffect, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypeMathjax from 'rehype-mathjax/browser';
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

// Math rendering configuration
const MATH_ENGINE = 'katex'; // 'katex' | 'mathjax' | 'none'
const ENABLE_SINGLE_DOLLAR = true; // 允许 $...$ 行内
const USE_LATEX_PREPROCESS = true; // 预处理 \[...\] 和 \(...\)

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

const MATH_HEURISTIC_REGEX = /(\\(frac|int|sum|prod|sqrt|log|exp|pi|rho|eta|theta|alpha|beta|gamma|delta|lambda|mu|tau|sigma|epsilon|phi|psi|omega|partial|Sigma|Delta|Gamma|Lambda|Omega|Phi|Pi|Psi|Theta|Xi)|[∑∏∞≈≠≤≥→←]|[_^][{a-zA-Z0-9]|[A-Za-z]\s*=\s*[A-Za-z0-9])/;

const normalizeMathText = (expr) => {
  let out = expr || '';
  // Normalize underscores and carets to LaTeX braces to avoid KaTeX parse errors on raw text
  out = out.replace(/([A-Za-z\\]+)_([A-Za-z0-9]+)/g, (_m, g1, g2) => `${g1}_{${g2}}`);
  out = out.replace(/([A-Za-z\\]+)\^([A-Za-z0-9]+)/g, (_m, g1, g2) => `${g1}^{${g2}}`);
  return out;
};

const wrapBareMathLines = (segment) => {
  return segment
    .split(/\n/)
    .map((line) => {
      // Skip if line already has math delimiters or code/links
      if (!line.trim()) return line;
      if (line.includes('$') || line.includes('`') || line.includes('http')) return line;

      const trimmed = line.trim();

      // If line has label like "公式: xxx" and math-looking part afterwards, split label and body
      const labelMatch = trimmed.match(/^(.{0,12}?[：:])\s*(.+)$/);
      if (labelMatch && MATH_HEURISTIC_REGEX.test(labelMatch[2])) {
        const normalized = normalizeMathText(labelMatch[2]);
        return `${labelMatch[1]}\n$$\n${normalized}\n$$`;
      }

      // Heuristic: contains math macros/symbols and at least one _ or ^ or backslash keyword
      if (MATH_HEURISTIC_REGEX.test(trimmed)) {
        return `$$\n${normalizeMathText(trimmed)}\n$$`;
      }
      return line;
    })
    .join('\n');
};

// Preprocess LaTeX brackets: \[...\] -> $$...$$ and \(...\) -> $...$
// Protect fenced/inline code blocks from being transformed, and auto-wrap bare math-like lines.
const processLatexBrackets = (text) => {
  if (!text || typeof text !== 'string') return text;

  // Split by fenced code blocks to avoid touching them
  const fencedPattern = /(```[\s\S]*?```|~~~[\s\S]*?~~~)/g;
  const segments = text.split(fencedPattern);
  const fences = text.match(fencedPattern) || [];

  const transformed = segments.map((segment, idx) => {
    // Every odd index corresponds to a fence placeholder
    if (idx % 2 === 1) {
      const fence = fences[(idx - 1) / 2];
      // Check if the fence is actually a math block disguised as code
      // We strip the ``` and check the content
      const innerMatch = fence.match(/^(```|~~~)(?:[\w-]*\n)?([\s\S]*?)\1$/);
      if (innerMatch) {
        const innerContent = innerMatch[2];
        // Logic: if it has significant math markers and no obvious code keywords, let's treat it as math
        if (MATH_HEURISTIC_REGEX.test(innerContent) && !innerContent.includes('import ') && !innerContent.includes('function ') && !innerContent.includes('const ')) {
          return `\n$$\n${normalizeMathText(innerContent.trim())}\n$$\n`;
        }
      }
      return fence;
    }

    // Protect inline code `...`
    const inlineParts = segment.split(/(`[^`]*`)/g);
    const handled = inlineParts
      .map((part, i) => {
        if (i % 2 === 1) return part; // inline code untouched
        let current = part;
        // Block: \[ ... \] -> $$ ... $$
        current = current.replace(/\\\[((?:.|\r?\n)*?)\\\]/g, (_m, p1) => `$$${p1}$$`);
        // Inline: \( ... \) -> $ ... $
        current = current.replace(/\\\(((?:.|\r?\n)*?)\\\)/g, (_m, p1) => `$${p1}$`);
        return current;
      })
      .join('');

    // Heuristic wrap bare math lines
    let finalSegment = handled;

    // fix: unwrap backticks if content looks like math (e.g. `L_{al} = ...`)
    finalSegment = finalSegment.replace(/`([^`]+)`/g, (match, codeContent) => {
      // 1. Explicitly wrapped in $...$ inside backticks? e.g. `$x$` or `$\tau$`
      if (codeContent.trim().startsWith('$') && codeContent.trim().endsWith('$')) {
        return codeContent;
      }
      // 2. Wrapped in \( ... \) inside backticks?
      if (codeContent.trim().startsWith('\\(') && codeContent.trim().endsWith('\\)')) {
        // Convert to $...$ while unwrapping
        return '$' + codeContent.trim().slice(2, -2) + '$';
      }

      // 3. Heuristic check
      // If it looks like a math formula, strip backticks and maybe wrap in $
      if (MATH_HEURISTIC_REGEX.test(codeContent)) {
        // Double check not to unwrap simple variable references or non-math code
        // Math usually has =, \, _, ^, or specific macros
        if (codeContent.includes('=') || codeContent.includes('\\') || codeContent.length > 3) {
          return `$${codeContent}$`;
        }
      }
      return match;
    });

    return wrapBareMathLines(finalSegment);
  });

  return transformed.join('');
};

const StreamingMarkdown = React.memo(({
  content,
  isStreaming,
  enableBlurReveal,
  blurIntensity = 'medium'
}) => {
  const lastProcessedLengthRef = useRef(0);

  const processedContent = useMemo(() => {
    if (!USE_LATEX_PREPROCESS) return content || '';
    return processLatexBrackets(content || '');
  }, [content]);

  let stableOffset = lastProcessedLengthRef.current;
  const currentLength = (processedContent || '').length;

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
    const plugins = [];
    if (MATH_ENGINE !== 'none') {
      plugins.push([remarkMath, { singleDollarTextMath: ENABLE_SINGLE_DOLLAR }]);
    }
    plugins.push(remarkGfm);
    if (enableBlurReveal && isStreaming) {
      plugins.push([remarkBlurReveal, { stableOffset }]);
    }
    return plugins;
  }, [enableBlurReveal, isStreaming, stableOffset, content?.length]); // Add content.length to force new plugin instance on update

  const rehypePlugins = React.useMemo(() => {
    const list = [
      rehypeRaw,
      rehypeHighlight
    ];
    if (MATH_ENGINE === 'katex') {
      list.splice(1, 0, [rehypeKatex, { strict: false, trust: true, output: 'html' }]);
    } else if (MATH_ENGINE === 'mathjax') {
      list.splice(1, 0, [rehypeMathjax, { svg: true }]);
    }
    return list;
  }, []);

  const intensityClass = enableBlurReveal ? `blur-intensity-${blurIntensity}` : '';

  return (
    <div className={`prose prose-sm max-w-full dark:prose-invert message-content leading-7 ${intensityClass}`}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        remarkRehypeOptions={{ allowDangerousHtml: true }}
      >
        {processedContent}
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
