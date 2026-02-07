import React, { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypeMathjax from 'rehype-mathjax/browser';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import mermaid from 'mermaid';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import { visit } from 'unist-util-visit';
import CitationLink from './CitationLink';

// ============================================================
// Mermaid 初始化配置 - 使用 strict 安全级别防止 XSS 攻击
// ============================================================
mermaid.initialize({
  startOnLoad: false,
  // 使用 strict 安全级别，禁止点击事件，防止 XSS 攻击
  securityLevel: 'strict',
  theme: 'default',
});

// Mermaid 渲染计数器，用于生成唯一 ID
let mermaidIdCounter = 0;

/**
 * MermaidBlock - Mermaid 代码块渲染子组件
 *
 * 功能：
 * - 接收 mermaid 代码并渲染为 SVG 流程图
 * - 使用 debounce（400ms）避免流式输出时频繁重渲染导致的抖动
 * - 渲染失败时降级显示原始代码块
 * - 显示加载状态
 */
const MermaidBlock = React.memo(({ code }) => {
  const [svg, setSvg] = useState('');
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);
  const containerRef = useRef(null);
  const timerRef = useRef(null);
  const idRef = useRef(`mermaid-block-${++mermaidIdCounter}`);

  useEffect(() => {
    // 清除上一次的 debounce 定时器
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    // 如果代码为空，不渲染
    if (!code || !code.trim()) {
      setLoading(false);
      setError(true);
      return;
    }

    setLoading(true);

    // 使用 debounce（400ms）延迟渲染，避免流式输出时频繁触发
    timerRef.current = setTimeout(async () => {
      try {
        // 每次渲染使用唯一 ID，避免 mermaid 缓存冲突
        const uniqueId = `${idRef.current}-${Date.now()}`;
        const { svg: renderedSvg } = await mermaid.render(uniqueId, code.trim());
        setSvg(renderedSvg);
        setError(false);
      } catch (err) {
        console.warn('Mermaid 渲染失败，降级显示原始代码:', err);
        setSvg('');
        setError(true);
      } finally {
        setLoading(false);
      }
    }, 400);

    // 组件卸载时清除定时器
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [code]);

  // 渲染失败时降级显示原始代码块
  if (error) {
    return (
      <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
        <code className="language-mermaid">{code}</code>
      </pre>
    );
  }

  // 加载中状态
  if (loading) {
    return (
      <div className="flex items-center justify-center p-6 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 text-sm">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span>正在渲染流程图...</span>
        </div>
      </div>
    );
  }

  // 成功渲染 SVG
  return (
    <div
      ref={containerRef}
      className="mermaid-container my-4 flex justify-center overflow-x-auto bg-white dark:bg-gray-900 rounded-lg p-4 border border-gray-200 dark:border-gray-700"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
});

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
// Optimization: Only keep the "active" animation window wrapped in spans
// Older text is flattened to plain text to improve DOM performance
const ANIMATION_WINDOW = 300; // Characters to keep "active" for animation completion

const remarkBlurReveal = (options) => {
  // Use Ref to avoid changing plugin options on every render (which forces ReactMarkdown to rebuild)
  const { offsetRef } = options;

  return (tree) => {
    // Read the latest stable offset from the ref at transform time
    const stableOffset = offsetRef ? offsetRef.current : (options.stableOffset || 0);
    visit(tree, 'text', (node, index, parent) => {
      if (!node.position) return;

      const start = node.position.start.offset;
      const end = node.position.end.offset;

      // 1. Fully Old/Stable Zone: Flatten to plain text to save performance
      // If the node ends well before the "new" content, we don't need to animate it anymore
      if (end <= Math.max(0, stableOffset - ANIMATION_WINDOW)) {
        return;
      }

      // 2. Active Zone (New content OR Recent content)
      // We wrap words in spans to allow animation (or let animation finish)
      // This covers the area from [stableOffset - ANIMATION_WINDOW] to [End of Text]

      // Split by word boundaries
      const parts = node.value.split(/(\s+)/);

      const nodes = parts.map((part, i) => {
        // Preserve whitespace as text nodes
        if (/^\s+$/.test(part)) {
          return { type: 'text', value: part };
        }

        // Calculate delay.
        // Crucial: The delay logic must be consistent across renders for the same word.
        // Since we are processing the whole text node (which grows at the end),
        // the index 'i' for the same word "Hello" at start remains 0.
        // So the delay remains stable, preventing animation restarts on re-renders (usually).
        const relativeIndex = i / 2;
        const delay = Math.min(relativeIndex * 0.04, 0.4);

        // However, we only strictly *need* the animation class if it's actually new.
        // But if we remove it too soon, it cuts off.
        // Strategy: Keep class for everything in this window. 
        // React's diffing should preserve the DOM node state if delay/key/structure passes.
        return {
          type: 'html',
          value: `<span class="animate-blur-reveal" style="animation-delay: ${delay.toFixed(2)}s;">${escapeHtml(part)}</span>`
        };
      });

      parent.children.splice(index, 1, ...nodes);
      return index + nodes.length;
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

      // 检测是否为 mermaid 代码块，如果是则保留原样不做数学转换
      const langMatch = fence.match(/^(```|~~~)([\w-]*)\n/);
      const fenceLang = langMatch ? langMatch[2].toLowerCase() : '';
      if (fenceLang === 'mermaid') {
        return fence;
      }

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

// ============================================================
// 引文引用预处理 - 将 [n] 模式替换为 <cite> 标签
// ============================================================

/**
 * 预处理 Markdown 内容中的引文引用标记
 * 将 [1]、[2] 等引用编号替换为 <cite data-ref="n">[n]</cite> HTML 标签，
 * 以便 ReactMarkdown 通过自定义组件渲染为可点击链接。
 *
 * 注意：只处理独立的 [n] 模式（n 为 1-99 的数字），
 * 避免误匹配 Markdown 链接 [text](url) 和图片 ![alt](url)。
 *
 * @param {string} text - 原始 Markdown 文本
 * @param {Array} citations - 引文列表，用于验证引用编号是否有效
 * @returns {string} 处理后的文本
 */
const processCitationRefs = (text, citations) => {
  if (!text || !citations || citations.length === 0) return text;

  // 构建有效引用编号集合
  const validRefs = new Set(citations.map(c => c.ref));

  // 匹配 [n] 模式，但排除 Markdown 链接语法 [text](url) 和图片 ![alt](url)
  // 使用负向前瞻排除后面紧跟 ( 的情况（即 Markdown 链接）
  // 使用负向后顾排除前面紧跟 ! 的情况（即 Markdown 图片）
  return text.replace(/(?<!!)\[(\d{1,2})\](?!\()/g, (match, refStr) => {
    const ref = parseInt(refStr, 10);
    if (validRefs.has(ref)) {
      return `<cite data-ref="${ref}">[${ref}]</cite>`;
    }
    return match;
  });
};

const StreamingMarkdown = React.memo(({
  content,
  isStreaming,
  enableBlurReveal,
  blurIntensity = 'medium',
  citations = null,
  onCitationClick = null
}) => {
  const containerRef = useRef(null);
  const lastTextLengthRef = useRef(0);
  const animatedNodesRef = useRef(new WeakSet());

  const processedContent = useMemo(() => {
    let text = content || '';
    if (USE_LATEX_PREPROCESS) {
      text = processLatexBrackets(text);
    }
    // 引文引用预处理：将 [n] 替换为 <cite> 标签
    if (citations && citations.length > 0) {
      text = processCitationRefs(text, citations);
    }
    return text;
  }, [content, citations]);

  // Build plugins (stable, no dependencies that change during streaming)
  const remarkPlugins = React.useMemo(() => {
    const plugins = [];
    if (MATH_ENGINE !== 'none') {
      plugins.push([remarkMath, { singleDollarTextMath: ENABLE_SINGLE_DOLLAR }]);
    }
    plugins.push(remarkGfm);
    return plugins;
  }, []);

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

  // Use MutationObserver to catch genuinely NEW nodes as they're added
  useEffect(() => {
    if (!enableBlurReveal || !containerRef.current) return;

    const container = containerRef.current;

    // MutationObserver fires for every DOM change
    const observer = new MutationObserver((mutations) => {
      if (!isStreaming) return; // Only animate during streaming

      mutations.forEach((mutation) => {
        // Handle added nodes
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            // Apply animation to the new element
            node.classList.add('blur-reveal-animate');

            // Also animate any child elements
            node.querySelectorAll('p, li, h1, h2, h3, h4, h5, h6, span, strong, em, code').forEach(child => {
              child.classList.add('blur-reveal-animate');
            });
          } else if (node.nodeType === Node.TEXT_NODE && node.parentElement) {
            // For text nodes, animate the parent element if it hasn't been animated recently
            const parent = node.parentElement;
            if (!parent.classList.contains('blur-reveal-animate')) {
              parent.classList.add('blur-reveal-animate');
            }
          }
        });

        // Handle character data changes (text content updates)
        if (mutation.type === 'characterData' && mutation.target.parentElement) {
          const parent = mutation.target.parentElement;
          if (!parent.classList.contains('blur-reveal-animate')) {
            parent.classList.add('blur-reveal-animate');
          }
        }
      });
    });

    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
      characterDataOldValue: false
    });

    return () => observer.disconnect();
  }, [enableBlurReveal, isStreaming]);

  // Reset on new conversation
  useEffect(() => {
    if (!content || content.length === 0) {
      // Clear any lingering animation classes
      if (containerRef.current) {
        containerRef.current.querySelectorAll('.blur-reveal-animate').forEach(el => {
          el.classList.remove('blur-reveal-animate');
        });
      }
    }
  }, [content]);

  const streamingClass = isStreaming ? 'streaming-active' : '';

  // 构建引文映射表，用于快速查找引用编号对应的引文数据
  const citationMap = useMemo(() => {
    if (!citations || citations.length === 0) return null;
    const map = {};
    citations.forEach(c => {
      map[c.ref] = c;
    });
    return map;
  }, [citations]);

  // 引文点击处理回调
  const handleCitationClick = useCallback((citation) => {
    if (onCitationClick) {
      onCitationClick(citation);
    }
  }, [onCitationClick]);

  // 自定义代码块渲染组件，用于检测 mermaid 代码块和引文 cite 标签
  const markdownComponents = useMemo(() => ({
    // 自定义 cite 标签渲染 - 将引文引用渲染为可点击链接
    cite({ node, children, ...props }) {
      const refStr = props['data-ref'];
      if (refStr && citationMap) {
        const ref = parseInt(refStr, 10);
        const citation = citationMap[ref];
        if (citation) {
          return (
            <CitationLink
              refNumber={ref}
              citation={citation}
              onClick={handleCitationClick}
            />
          );
        }
      }
      // 无匹配引文时渲染为普通 cite 标签
      return <cite {...props}>{children}</cite>;
    },
    code({ node, inline, className, children, ...props }) {
      // 检测 ```mermaid 代码块
      const match = /language-(\w+)/.exec(className || '');
      const language = match ? match[1] : '';

      if (!inline && language === 'mermaid') {
        // 提取 mermaid 代码内容并渲染为 SVG
        const mermaidCode = String(children).replace(/\n$/, '');
        return <MermaidBlock code={mermaidCode} />;
      }

      // 非 mermaid 代码块使用默认渲染
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
    // 自定义 pre 标签渲染，避免 mermaid 块被包裹在 pre 中
    pre({ node, children, ...props }) {
      // 检查子元素是否为 MermaidBlock（通过检测 code 子元素的 language-mermaid 类名）
      const childArray = React.Children.toArray(children);
      const hasMermaid = childArray.some(child => {
        if (React.isValidElement(child) && child.type === MermaidBlock) {
          return true;
        }
        // 检查 code 元素的 className 是否包含 mermaid
        if (React.isValidElement(child) && child.props?.className?.includes('language-mermaid')) {
          return true;
        }
        return false;
      });

      // 如果包含 mermaid 块，直接渲染子元素（不包裹 pre）
      if (hasMermaid) {
        return <>{children}</>;
      }

      return <pre {...props}>{children}</pre>;
    }
  }), [citationMap, handleCitationClick]);

  return (
    <div
      ref={containerRef}
      className={`prose prose-sm max-w-full dark:prose-invert message-content leading-7 ${streamingClass}`}
    >
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        remarkRehypeOptions={{ allowDangerousHtml: true }}
        components={markdownComponents}
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
    prevProps.enableBlurReveal === nextProps.enableBlurReveal &&
    prevProps.citations === nextProps.citations &&
    prevProps.onCitationClick === nextProps.onCitationClick
  );
});

export default StreamingMarkdown;
