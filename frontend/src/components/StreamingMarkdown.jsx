import React, { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import rehypeKatexCached from '../utils/rehypeKatexCached.js';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import { visit } from 'unist-util-visit';
import CitationLink from './CitationLink';

// mermaid 动态加载：仅在首次遇到 Mermaid 代码块时触发加载，
// 使用单例 Promise 模式避免重复加载（需求 7.1）
let mermaidPromise = null;
const loadMermaid = () => {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then(m => {
      m.default.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'default' });
      return m.default;
    });
  }
  return mermaidPromise;
};

let mermaidIdCounter = 0;

const MermaidBlock = React.memo(({ code, defer }) => {
  const [svg, setSvg] = useState('');
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef(null);
  const idRef = useRef(`mermaid-block-${++mermaidIdCounter}`);

  useEffect(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    if (defer) {
      setLoading(false);
      return;
    }

    if (!code || !code.trim()) {
      setLoading(false);
      setError(true);
      return;
    }

    setLoading(true);

    timerRef.current = setTimeout(async () => {
      try {
        // 动态加载 mermaid，首次调用时触发下载（需求 7.1）
        const mermaidInstance = await loadMermaid();
        const uniqueId = `${idRef.current}-${Date.now()}`;
        const { svg: renderedSvg } = await mermaidInstance.render(uniqueId, code.trim());
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

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [code, defer]);

  if (defer) {
    return (
      <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
        <code className="language-mermaid">{code}</code>
      </pre>
    );
  }

  if (error) {
    return (
      <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
        <code className="language-mermaid">{code}</code>
      </pre>
    );
  }

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

  return (
    <div
      className="mermaid-container my-4 flex justify-center overflow-x-auto bg-white dark:bg-gray-900 rounded-lg p-4 border border-gray-200 dark:border-gray-700"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
});

const MATH_ENGINE = 'katex';
const ENABLE_SINGLE_DOLLAR = true;
const USE_LATEX_PREPROCESS = true;

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

const STREAM_ANIMATION_WINDOW = 600;

const remarkBlurRevealAST = (options) => {
  const { isStreaming, stableOffset, windowSize = STREAM_ANIMATION_WINDOW } = options;

  return (tree) => {
    if (!isStreaming) return;

    visit(tree, 'text', (node, index, parent) => {
      if (!node.position) return;

      const end = node.position.end.offset;
      const activeStart = Math.max(0, stableOffset - windowSize);
      if (end <= activeStart) return;

      const parts = node.value.split(/(\s+)/);
      const nodes = parts.map((part, i) => {
        if (/^\s+$/.test(part)) return { type: 'text', value: part };
        const delay = Math.min((i / 2) * 0.03, 0.35);
        return {
          type: 'html',
          value: `<span class="blur-reveal-animate" style="animation-delay: ${delay.toFixed(2)}s;">${escapeHtml(part)}</span>`
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
  out = out.replace(/([A-Za-z\\]+)_([A-Za-z0-9]+)/g, (_m, g1, g2) => `${g1}_{${g2}}`);
  out = out.replace(/([A-Za-z\\]+)\^([A-Za-z0-9]+)/g, (_m, g1, g2) => `${g1}^{${g2}}`);
  return out;
};

const wrapBareMathLines = (segment) => {
  return segment
    .split(/\n/)
    .map((line) => {
      if (!line.trim()) return line;
      if (line.includes('$') || line.includes('`') || line.includes('http')) return line;

      const trimmed = line.trim();
      const labelMatch = trimmed.match(/^(.{0,12}?[：:])\s*(.+)$/);
      if (labelMatch && MATH_HEURISTIC_REGEX.test(labelMatch[2])) {
        const normalized = normalizeMathText(labelMatch[2]);
        return `${labelMatch[1]}\n$$\n${normalized}\n$$`;
      }

      if (MATH_HEURISTIC_REGEX.test(trimmed)) {
        return `$$\n${normalizeMathText(trimmed)}\n$$`;
      }
      return line;
    })
    .join('\n');
};

const processLatexBrackets = (text) => {
  if (!text || typeof text !== 'string') return text;

  const fencedPattern = /(```[\s\S]*?```|~~~[\s\S]*?~~~)/g;
  const segments = text.split(fencedPattern);
  const fences = text.match(fencedPattern) || [];

  const transformed = segments.map((segment, idx) => {
    if (idx % 2 === 1) {
      const fence = fences[(idx - 1) / 2];

      const langMatch = fence.match(/^(```|~~~)([\w-]*)\n/);
      const fenceLang = langMatch ? langMatch[2].toLowerCase() : '';
      if (fenceLang === 'mermaid') {
        return fence;
      }

      const innerMatch = fence.match(/^(```|~~~)(?:[\w-]*\n)?([\s\S]*?)\1$/);
      if (innerMatch) {
        const innerContent = innerMatch[2];
        if (
          MATH_HEURISTIC_REGEX.test(innerContent) &&
          !innerContent.includes('import ') &&
          !innerContent.includes('function ') &&
          !innerContent.includes('const ')
        ) {
          return `\n$$\n${normalizeMathText(innerContent.trim())}\n$$\n`;
        }
      }
      return fence;
    }

    const inlineParts = segment.split(/(`[^`]*`)/g);
    const handled = inlineParts
      .map((part, i) => {
        if (i % 2 === 1) return part;
        let current = part;
        current = current.replace(/\\\[((?:.|\r?\n)*?)\\\]/g, (_m, p1) => `$$${p1}$$`);
        current = current.replace(/\\\(((?:.|\r?\n)*?)\\\)/g, (_m, p1) => `$${p1}$`);
        return current;
      })
      .join('');

    let finalSegment = handled;

    finalSegment = finalSegment.replace(/`([^`]+)`/g, (match, codeContent) => {
      if (codeContent.trim().startsWith('$') && codeContent.trim().endsWith('$')) {
        return codeContent;
      }
      if (codeContent.trim().startsWith('\\(') && codeContent.trim().endsWith('\\)')) {
        return '$' + codeContent.trim().slice(2, -2) + '$';
      }

      if (MATH_HEURISTIC_REGEX.test(codeContent)) {
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

const processCitationRefs = (text, citations) => {
  if (!text || !citations || citations.length === 0) return text;

  const validRefs = new Set(citations.map((c) => c.ref));

  return text.replace(/(?<!!)\[(\d{1,2})\](?!\()/g, (match, refStr) => {
    const ref = parseInt(refStr, 10);
    if (validRefs.has(ref)) {
      return `<cite data-ref="${ref}">[${ref}]</cite>`;
    }
    return match;
  });
};

const StreamingMarkdown = React.memo(
  ({ content, isStreaming, enableBlurReveal, blurIntensity = 'medium', citations = null, onCitationClick = null, streamingRef = null }) => {
    const containerRef = useRef(null);

    // ref 直写模式：流式输出期间通过 streamingRef 直接更新 DOM，
    // 不经过 React 状态更新和 ReactMarkdown 渲染
    const isRefDirectWrite = isStreaming && streamingRef != null;

    const processedContent = useMemo(() => {
      // ref 直写模式下不需要处理内容（由 DOM 直接显示纯文本）
      if (isRefDirectWrite) return '';
      let text = content || '';
      if (USE_LATEX_PREPROCESS && !isStreaming) {
        text = processLatexBrackets(text);
      }
      if (citations && citations.length > 0) {
        text = processCitationRefs(text, citations);
      }
      return text;
    }, [content, citations, isStreaming, isRefDirectWrite]);

    // 基础 remark 插件数组缓存：配置不变时保持引用稳定，
    // 避免 ReactMarkdown 因插件引用变化而重新初始化（需求 6.3）
    const baseRemarkPlugins = React.useMemo(() => {
      const plugins = [];
      if (MATH_ENGINE !== 'none') {
        plugins.push([remarkMath, { singleDollarTextMath: ENABLE_SINGLE_DOLLAR }]);
      }
      plugins.push(remarkGfm);
      return plugins;
    }, []);

    // 完整 remark 插件数组：仅在需要 blur-reveal 动画时追加动态插件，
    // 非流式输出时直接复用稳定的基础插件数组引用
    const remarkPlugins = React.useMemo(() => {
      if (enableBlurReveal && isStreaming) {
        return [...baseRemarkPlugins, [remarkBlurRevealAST, { isStreaming, stableOffset: content?.length || 0 }]];
      }
      return baseRemarkPlugins;
    }, [baseRemarkPlugins, enableBlurReveal, isStreaming, content?.length]);

    // rehype 插件数组缓存：配置固定，空依赖保持引用稳定（需求 6.3）
    // 使用带缓存的 rehypeKatexCached 替代 rehypeKatex，
    // 相同公式表达式直接返回缓存结果，避免重复渲染（需求 6.2）
    const rehypePlugins = React.useMemo(() => [
      rehypeRaw,
      [rehypeKatexCached, { strict: false, trust: true, output: 'html' }],
      rehypeHighlight,
    ], []);

    useEffect(() => {
      if (!content || content.length === 0) {
        if (containerRef.current) {
          containerRef.current.querySelectorAll('.blur-reveal-animate').forEach((el) => {
            el.classList.remove('blur-reveal-animate');
          });
        }
      }
    }, [content]);

    const streamingClass = isStreaming ? 'streaming-active' : '';

    const showWaitingDots = isStreaming && (!content || content.trim().length === 0);

    const citationMap = useMemo(() => {
      if (!citations || citations.length === 0) return null;
      const map = {};
      citations.forEach((c) => {
        map[c.ref] = c;
      });
      return map;
    }, [citations]);

    const handleCitationClick = useCallback(
      (citation) => {
        if (onCitationClick) {
          onCitationClick(citation);
        }
      },
      [onCitationClick]
    );

    const markdownComponents = useMemo(
      () => ({
        cite({ node, children, ...props }) {
          const refStr = props['data-ref'];
          if (refStr && citationMap) {
            const ref = parseInt(refStr, 10);
            const citation = citationMap[ref];
            if (citation) {
              return <CitationLink refNumber={ref} citation={citation} onClick={handleCitationClick} />;
            }
          }
          return <cite {...props}>{children}</cite>;
        },
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          const language = match ? match[1] : '';

          if (!inline && language === 'mermaid') {
            const mermaidCode = String(children).replace(/\n$/, '');
            return <MermaidBlock code={mermaidCode} defer={isStreaming} />;
          }

          return (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
        pre({ node, children, ...props }) {
          const childArray = React.Children.toArray(children);
          const hasMermaid = childArray.some((child) => {
            if (React.isValidElement(child) && child.type === MermaidBlock) {
              return true;
            }
            if (React.isValidElement(child) && child.props?.className?.includes('language-mermaid')) {
              return true;
            }
            return false;
          });

          if (hasMermaid) {
            return <>{children}</>;
          }

          return <pre {...props}>{children}</pre>;
        }
      }),
      [citationMap, handleCitationClick, isStreaming]
    );

    return (
      <div
        ref={containerRef}
        className={`prose prose-sm max-w-full dark:prose-invert message-content leading-7 ${streamingClass}`}
      >
        {isRefDirectWrite ? (
          // ref 直写模式：流式输出期间显示纯文本容器，
          // useSmoothStream 通过 streamingRef 直接写入 textContent，
          // 避免触发 React 状态更新和 ReactMarkdown 重渲染（需求 4.2）
          <div
            ref={streamingRef}
            className="whitespace-pre-wrap break-words"
          />
        ) : showWaitingDots ? (
          <div className="streaming-dots">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        ) : (
          <ReactMarkdown
            remarkPlugins={remarkPlugins}
            rehypePlugins={rehypePlugins}
            remarkRehypeOptions={{ allowDangerousHtml: true }}
            components={markdownComponents}
          >
            {processedContent}
          </ReactMarkdown>
        )}
      </div>
    );
  },
  // 自定义比较函数：仅在关键 props 变化时重渲染
  // ref 直写模式下，streamingRef 的存在/消失也需要触发重渲染
  (prevProps, nextProps) => {
    return (
      prevProps.content === nextProps.content &&
      prevProps.isStreaming === nextProps.isStreaming &&
      (prevProps.streamingRef != null) === (nextProps.streamingRef != null)
    );
  }
);

// 为 React DevTools 添加显示名称
StreamingMarkdown.displayName = 'StreamingMarkdown';

export { processLatexBrackets };
export default StreamingMarkdown;
