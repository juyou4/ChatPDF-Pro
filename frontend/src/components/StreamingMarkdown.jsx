import React, { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import { replaceInlineCitationRefs, replaceWebCitationRefs } from '../utils/citationUtils';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
// rehype-mathjax/svg 使用 CJS 的 mathjax-full，不能静态 import
// 改为懒加载，失败时 fallback 到 KaTeX
let _rehypeMathjaxSvg = null;
let _mathjaxLoadAttempted = false;
const loadMathjax = () => {
  if (_mathjaxLoadAttempted) return Promise.resolve(_rehypeMathjaxSvg);
  _mathjaxLoadAttempted = true;
  return import('rehype-mathjax/svg')
    .then(m => { _rehypeMathjaxSvg = m.default || m; return _rehypeMathjaxSvg; })
    .catch(() => { _rehypeMathjaxSvg = null; return null; });
};
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import { visit } from 'unist-util-visit';
import CitationLink from './CitationLink';
import WebCitationLink from './WebCitationLink';
import { processLatexBrackets } from '../utils/processLatexBrackets.js';
import remarkDisableConstructs from '../utils/remarkDisableConstructs.js';
import { getCommittedPrefix, getStreamedSourceText, hydrateStreamingMath } from '../utils/streamingMath';
import { useChatParams } from '../contexts/ChatParamsContext';

// 模型回答、网页摘录和文档文本都属于不可信输入。保留现有的 cite/
// wsource 引文标签和常规文本格式，但不允许其创建可执行、可嵌入或可追踪
// 的节点。KaTeX/MathJax 在净化之后生成自己的受控输出。
const UNSAFE_MARKDOWN_TAGS = new Set([
  'audio', 'embed', 'form', 'iframe', 'img', 'input', 'link', 'object',
  'picture', 'script', 'source', 'style', 'svg', 'video',
]);

const withoutClassName = (attrs = []) => (
  attrs.filter((item) => item !== 'className' && !(Array.isArray(item) && item[0] === 'className'))
);

// remark-math 会先变成 .math-inline / .math-display，rehype-katex 靠这些 class 找节点。
// 之前 span 的 className 只放行 blur-reveal，模糊关闭走整篇 ReactMarkdown 时公式会被洗掉。
const MATH_CLASS_NAME = [
  'className',
  'math',
  'math-inline',
  'math-display',
  'language-math',
  'katex',
  'katex-html',
  'katex-mathml',
  'katex-error',
  'katex-display',
  'blur-reveal-animate',
  /^(blur-stagger-[0-8]|katex.*|math.*)$/,
];

export const CHATPDF_MARKDOWN_SCHEMA = {
  ...defaultSchema,
  tagNames: [...new Set([...(defaultSchema.tagNames || []), 'cite', 'wsource'])]
    .filter((tag) => !UNSAFE_MARKDOWN_TAGS.has(tag)),
  attributes: {
    ...defaultSchema.attributes,
    cite: [...(defaultSchema.attributes?.cite || []), 'dataRef', 'data-ref'],
    wsource: [...(defaultSchema.attributes?.wsource || []), 'dataIdx', 'data-idx'],
    span: [...withoutClassName(defaultSchema.attributes?.span), MATH_CLASS_NAME],
    div: [...withoutClassName(defaultSchema.attributes?.div), MATH_CLASS_NAME],
    code: [...withoutClassName(defaultSchema.attributes?.code), MATH_CLASS_NAME],
  },
};

// 与 Cherry Studio 一致：只有正文里真有 HTML 才走 raw。cite / wsource 是我们自己注入的。
const MARKDOWN_HTML_RE = /<(cite|wsource|style|p|div|span|b|i|strong|em|ul|ol|li|table|tr|td|th|thead|tbody|h[1-6]|blockquote|pre|code|br|hr|details|summary)\b/i;

const safeExternalHref = (href) => {
  if (typeof href !== 'string' || !href.trim()) return null;
  try {
    const parsed = new URL(href, window.location.origin);
    return ['http:', 'https:'].includes(parsed.protocol) ? parsed.href : null;
  } catch {
    return null;
  }
};

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


function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

const STREAM_ANIMATION_WINDOW = 160;

const remarkBlurRevealAST = (options) => {
  const {
    isStreaming,
    animationStartOffset,
    animationEndOffset,
    windowSize = STREAM_ANIMATION_WINDOW,
  } = options;

  return (tree) => {
    if (!isStreaming) return;

    const activeStart = Math.max(
      0,
      Math.min(animationStartOffset, animationEndOffset),
      animationEndOffset - windowSize,
    );
    let sequenceIndex = 0;

    visit(tree, 'text', (node, index, parent) => {
      if (!node.position) return;

      const start = node.position.start.offset;
      const end = node.position.end.offset;
      if (end <= activeStart) return;

      const localStart = Math.max(0, Math.min(node.value.length, activeStart - start));
      const stablePrefix = node.value.slice(0, localStart);
      const activeText = node.value.slice(localStart);
      const nodes = stablePrefix ? [{ type: 'text', value: stablePrefix }] : [];

      splitStreamingText(activeText).forEach((part) => {
        if (/^\s+$/u.test(part)) {
          nodes.push({ type: 'text', value: part });
          return;
        }
        const staggerIndex = Math.min(sequenceIndex, 8);
        sequenceIndex += 1;
        nodes.push({
          type: 'html',
          value: `<span class="blur-reveal-animate blur-stagger-${staggerIndex}">${escapeHtml(part)}</span>`,
        });
      });

      parent.children.splice(index, 1, ...nodes);
      return index + nodes.length;
    });
  };
};

const splitStreamingText = (text) => {
  if (typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function') {
    const segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
    return Array.from(segmenter.segment(text), ({ segment }) => segment);
  }
  return Array.from(text);
};

export const processCitationRefs = (text, citations) => {
  if (!text || !citations || citations.length === 0) return text;

  const validRefs = new Set(
    citations
      .map((c) => Number(c?.display_ref ?? c?.ref))
      .filter((ref) => Number.isFinite(ref))
  );

  return replaceInlineCitationRefs(text, (match, halfWidthRef, fullWidthRef) => {
    const ref = parseInt(halfWidthRef ?? fullWidthRef, 10);
    if (validRefs.has(ref)) {
      return `<cite data-ref="${ref}">[${ref}]</cite>`;
    }
    return match;
  });
};

/**
 * 将联网来源的 [Wn] 转换为独立标签；同时兼容没有 PDF 编号冲突的
 * 历史回答 [n]。新回答必须使用 [Wn]。
 */
export const processWebSearchCitationRefs = (text, webSearchSources) => {
  if (!text || !webSearchSources || webSearchSources.length === 0) return text;
  const max = webSearchSources.length;
  let processed = replaceWebCitationRefs(text, (match, halfWidthRef, fullWidthRef) => {
    const n = parseInt(halfWidthRef ?? fullWidthRef, 10);
    if (n < 1 || n > max) return match;
    return `<wsource data-idx="${n}">[W${n}]</wsource>`;
  });
  processed = replaceInlineCitationRefs(processed, (match, halfWidthRef, fullWidthRef, offset, source) => {
    const n = parseInt(halfWidthRef ?? fullWidthRef, 10);
    if (n < 1 || n > max) return match;
    // 如果已被 processCitationRefs 替换为 <cite...> 则跳过
    const before = source.slice(Math.max(0, offset - 20), offset);
    if (before.includes('<cite')) return match;
    return `<wsource data-idx="${n}">[${n}]</wsource>`;
  });
  return processed;
};

/**
 * 可折叠的 pre 代码块组件
 * 超过 10 行的代码块默认折叠，显示前 5 行 + 展开按钮
 */
const CollapsiblePre = ({ children, ...props }) => {
  const [collapsed, setCollapsed] = useState(true);
  const contentRef = useRef(null);

  const lineCount = useMemo(() => {
    const text = React.Children.toArray(children)
      .map(child => {
        if (typeof child === 'string') return child;
        if (React.isValidElement(child) && child.props?.children) {
          return String(child.props.children);
        }
        return '';
      })
      .join('');
    return (text.match(/\n/g) || []).length + 1;
  }, [children]);

  const shouldCollapse = lineCount > 10;

  return (
    <div className="collapsible-code-block relative">
      <pre
        {...props}
        ref={contentRef}
        style={{
          ...(shouldCollapse && collapsed ? { maxHeight: '160px', overflow: 'hidden' } : {}),
        }}
      >
        {children}
      </pre>
      {shouldCollapse && (
        <>
          {collapsed && (
            <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-gray-100 dark:from-gray-800 to-transparent pointer-events-none" />
          )}
          <button
            onClick={() => setCollapsed(prev => !prev)}
            className="w-full text-center text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 py-1 bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 transition-colors"
          >
            {collapsed ? `展开全部 (${lineCount} 行)` : '收起'}
          </button>
        </>
      )}
    </div>
  );
};

export const isRefDirectWriteStreaming = (props) => (
  Boolean(
    props?.isStreaming
    && props?.streamingRef != null
    && (props?.enableBlurReveal || typeof props?.subscribeDisplayedText !== 'function')
  )
);

export const streamingMarkdownAreEqual = (prevProps, nextProps) => {
  // 思考首包会把 content 写成「好的」，但后续 token 只走 ref 直写。
  // 若这里把 content 变化当成需要重绘，直写节点会被拆掉，画面就停在「好的」。
  const ignoreContent = isRefDirectWriteStreaming(prevProps) && isRefDirectWriteStreaming(nextProps);
  return (
    (ignoreContent || prevProps.content === nextProps.content) &&
    prevProps.isStreaming === nextProps.isStreaming &&
    prevProps.enableBlurReveal === nextProps.enableBlurReveal &&
    prevProps.blurIntensity === nextProps.blurIntensity &&
    prevProps.hydrateDirectWriteContent === nextProps.hydrateDirectWriteContent &&
    (prevProps.streamingRef != null) === (nextProps.streamingRef != null) &&
    prevProps.citations === nextProps.citations &&
    prevProps.webSearchSources === nextProps.webSearchSources &&
    prevProps.suppressInitialDots === nextProps.suppressInitialDots &&
    prevProps.subscribeCommittedPrefix === nextProps.subscribeCommittedPrefix &&
    prevProps.subscribeDisplayedText === nextProps.subscribeDisplayedText
  );
};

const StreamingMarkdown = React.memo(
  ({ content, isStreaming, enableBlurReveal, blurIntensity = 'medium', citations = null, onCitationClick = null, streamingRef = null, webSearchSources = null, suppressInitialDots = false, hydrateDirectWriteContent = true, subscribeCommittedPrefix = null, subscribeDisplayedText = null }) => {
    const containerRef = useRef(null);
    const previousAnimatedLengthRef = useRef(0);
    const [hasDirectWriteContent, setHasDirectWriteContent] = useState(false);
    const [committedPrefix, setCommittedPrefix] = useState('');
    const [liveDisplayedText, setLiveDisplayedText] = useState('');
    const { codeCollapsible, codeWrappable, codeShowLineNumbers, mathEngine, mathEnableSingleDollar } = useChatParams();
    const [mathjaxReady, setMathjaxReady] = useState(!!_rehypeMathjaxSvg);

    // MathJax 懒加载：仅在用户选择 MathJax 引擎时触发
    useEffect(() => {
      if (mathEngine === 'MathJax' && !_rehypeMathjaxSvg && !_mathjaxLoadAttempted) {
        loadMathjax().then(mod => { if (mod) setMathjaxReady(true); });
      }
    }, [mathEngine]);

    // 模糊开启，或思考区这种没有全文订阅的流，继续 ref 直写。
    // 正文关闭模糊时走 Cherry Studio 路线：每帧 ReactMarkdown。
    const isRefDirectWrite = Boolean(
      isStreaming
      && streamingRef != null
      && (enableBlurReveal || typeof subscribeDisplayedText !== 'function')
    );

    useEffect(() => {
      if (!isRefDirectWrite || typeof subscribeCommittedPrefix !== 'function') {
        setCommittedPrefix('');
        return undefined;
      }
      return subscribeCommittedPrefix(setCommittedPrefix);
    }, [isRefDirectWrite, subscribeCommittedPrefix]);

    useEffect(() => {
      if (isRefDirectWrite || typeof subscribeDisplayedText !== 'function') {
        setLiveDisplayedText('');
        return undefined;
      }
      return subscribeDisplayedText(setLiveDisplayedText);
    }, [isRefDirectWrite, subscribeDisplayedText]);

    const processedContent = useMemo(() => {
      // ref 直写模式下整篇正文仍走 DOM 尾巴；已完成的标题/段落另见 processedCommitted
      if (isRefDirectWrite) return '';
      let text = (
        isStreaming && typeof subscribeDisplayedText === 'function'
          ? liveDisplayedText
          : (content || '')
      );
      if (mathEngine !== 'none') {
        text = processLatexBrackets(text);
      }
      if (citations && citations.length > 0) {
        text = processCitationRefs(text, citations);
      }
      if (webSearchSources && webSearchSources.length > 0) {
        text = processWebSearchCitationRefs(text, webSearchSources);
      }
      return text;
    }, [content, citations, webSearchSources, isStreaming, isRefDirectWrite, liveDisplayedText, mathEngine, subscribeDisplayedText]);

    const processedCommitted = useMemo(() => {
      if (!isRefDirectWrite || !committedPrefix) return '';
      let text = committedPrefix;
      if (mathEngine !== 'none') {
        text = processLatexBrackets(text);
      }
      if (citations && citations.length > 0) {
        text = processCitationRefs(text, citations);
      }
      if (webSearchSources && webSearchSources.length > 0) {
        text = processWebSearchCitationRefs(text, webSearchSources);
      }
      return text;
    }, [citations, committedPrefix, isRefDirectWrite, mathEngine, webSearchSources]);

    const shouldUseSingleDollarMath = React.useMemo(() => {
      if (mathEnableSingleDollar) return true;
      // 用户关闭单 $ 时，仅在明显 LaTeX 片段场景下兜底开启，避免公式整体失效
      return /(^|[^\\])\$[^$\n]*\\[A-Za-z]{2,}[^$\n]*\$/m.test(processedContent || processedCommitted || '');
    }, [mathEnableSingleDollar, processedCommitted, processedContent]);

    // 基础 remark 插件数组缓存：配置不变时保持引用稳定，
    // 避免 ReactMarkdown 因插件引用变化而重新初始化（需求 6.3）
    const baseRemarkPlugins = React.useMemo(() => {
      const plugins = [
        [remarkGfm, { singleTilde: false }],
        // Cherry Studio：关掉缩进代码块，公式行前导空格不会变成 <pre><code>
        remarkDisableConstructs(['codeIndented']),
      ];
      if (mathEngine !== 'none') {
        plugins.push([remarkMath, { singleDollarTextMath: shouldUseSingleDollarMath }]);
      }
      return plugins;
    }, [mathEngine, shouldUseSingleDollarMath]);

    const animationEndOffset = processedContent.length;
    const animationStartOffset = animationEndOffset < previousAnimatedLengthRef.current
      ? 0
      : previousAnimatedLengthRef.current;

    useEffect(() => {
      previousAnimatedLengthRef.current = isStreaming ? animationEndOffset : 0;
    }, [animationEndOffset, isStreaming]);

    // 非 ref 流式路径只包装本轮新增的字符，避免 Markdown 重渲染时让旧内容反复闪烁。
    const remarkPlugins = React.useMemo(() => {
      if (enableBlurReveal && isStreaming) {
        return [
          ...baseRemarkPlugins,
          [remarkBlurRevealAST, {
            isStreaming,
            animationStartOffset,
            animationEndOffset,
          }],
        ];
      }
      return baseRemarkPlugins;
    }, [
      animationEndOffset,
      animationStartOffset,
      baseRemarkPlugins,
      enableBlurReveal,
      isStreaming,
    ]);

    const rehypePlugins = React.useMemo(() => {
      const plugins = [];
      const markdownSource = processedContent || processedCommitted || '';
      // Cherry Studio：没有 HTML 就不走 raw/sanitize，避免洗掉 math class。
      if (MARKDOWN_HTML_RE.test(markdownSource)) {
        plugins.push(rehypeRaw, [rehypeSanitize, CHATPDF_MARKDOWN_SCHEMA]);
      }
      if (mathEngine === 'KaTeX') {
        plugins.push([rehypeKatex, { strict: false, trust: false, output: 'html' }]);
      } else if (mathEngine === 'MathJax' && _rehypeMathjaxSvg) {
        plugins.push(_rehypeMathjaxSvg);
      } else if (mathEngine === 'MathJax' && !_rehypeMathjaxSvg) {
        plugins.push([rehypeKatex, { strict: false, trust: false, output: 'html' }]);
      }
      plugins.push(rehypeHighlight);
      return plugins;
    }, [mathEngine, mathjaxReady, processedCommitted, processedContent]);

    useEffect(() => {
      if (!content || content.length === 0) {
        if (containerRef.current) {
          containerRef.current.querySelectorAll('.blur-reveal-animate').forEach((el) => {
            el.classList.remove('blur-reveal-animate');
          });
        }
      }
    }, [content]);

    const streamingClass = isStreaming
      ? `streaming-active${enableBlurReveal ? ' blur-reveal-enabled' : ''}`
      : '';
    const showWaitingDots = !suppressInitialDots && isStreaming && (
      isRefDirectWrite
        ? !hasDirectWriteContent && !committedPrefix && (!content || content.trim().length === 0)
        : !String(
          (typeof subscribeDisplayedText === 'function' ? liveDisplayedText : content) || ''
        ).trim()
    );

    useEffect(() => {
      if (!isRefDirectWrite) {
        setHasDirectWriteContent(false);
        return;
      }

      const el = streamingRef?.current;
      if (!el) {
        setHasDirectWriteContent(false);
        return;
      }

      // 检索阶段可能只把 React state 传到这里，流队列还没接管。
      // 思考正文不能靠这个回填：首包「好的」会把后续 token 写到被卸掉的节点上。
      if (
        hydrateDirectWriteContent
        && !getStreamedSourceText(el).trim()
        && content
        && content.trim().length > 0
      ) {
        const committed = getCommittedPrefix(content, {
          enableMath: mathEngine !== 'none',
          enableSingleDollar: mathEnableSingleDollar !== false,
        });
        const tail = content.slice(committed.length);
        el.textContent = tail;
        if (mathEngine !== 'none') {
          hydrateStreamingMath(el, tail, {
            enableSingleDollar: Boolean(
              mathEnableSingleDollar
              || /(^|[^\\])\$[^$\n]*\\[A-Za-z]{2,}[^$\n]*\$/m.test(content)
            ),
          });
        }
      }

      let observer = null;
      const detectFirstContent = () => {
        if (!el.firstChild) return false;
        setHasDirectWriteContent(true);
        observer?.disconnect();
        return true;
      };

      if (detectFirstContent()) return undefined;
      observer = new MutationObserver(detectFirstContent);
      observer.observe(el, { childList: true, subtree: true, characterData: true });

      return () => observer.disconnect();
    }, [isRefDirectWrite, streamingRef, content, hydrateDirectWriteContent, mathEngine, mathEnableSingleDollar]);

    const citationMap = useMemo(() => {
      if (!citations || citations.length === 0) return null;
      const map = {};
      citations.forEach((c) => {
        const ref = Number(c?.display_ref ?? c?.ref);
        if (Number.isFinite(ref)) {
          map[ref] = c;
        }
      });
      return map;
    }, [citations]);

    const webSearchMap = useMemo(() => {
      if (!webSearchSources || webSearchSources.length === 0) return null;
      const map = {};
      webSearchSources.forEach((src, i) => { map[i + 1] = src; });
      return map;
    }, [webSearchSources]);

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
        wsource({ node, children, ...props }) {
          const idxStr = props['data-idx'];
          if (idxStr && webSearchMap) {
            const idx = parseInt(idxStr, 10);
            const source = webSearchMap[idx];
            return <WebCitationLink refNumber={idx} source={source} />;
          }
          return <span>{children}</span>;
        },
        a({ node, href, children, ...props }) {
          const safeHref = safeExternalHref(href);
          if (!safeHref) return <span>{children}</span>;
          return (
            <a {...props} href={safeHref} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          );
        },
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          const language = match ? match[1] : '';

          if (!inline && language === 'mermaid') {
            const mermaidCode = String(children).replace(/\n$/, '');
            return <MermaidBlock code={mermaidCode} defer={isStreaming} />;
          }

          if (!inline && codeShowLineNumbers) {
            const codeStr = String(children).replace(/\n$/, '');
            const lines = codeStr.split('\n');
            return (
              <code className={className} {...props} style={{ ...(codeWrappable ? { whiteSpace: 'pre-wrap', wordBreak: 'break-word' } : {}) }}>
                <table className="code-line-table" style={{ borderCollapse: 'collapse', width: '100%' }}>
                  <tbody>
                    {lines.map((line, i) => (
                      <tr key={i}>
                        <td className="code-line-number" style={{ userSelect: 'none', paddingRight: '12px', textAlign: 'right', color: '#9ca3af', minWidth: '2em', verticalAlign: 'top' }}>{i + 1}</td>
                        <td className="code-line-content" style={{ whiteSpace: codeWrappable ? 'pre-wrap' : 'pre' }}>{line}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </code>
            );
          }

          return (
            <code className={className} {...props} style={{ ...((!inline && codeWrappable) ? { whiteSpace: 'pre-wrap', wordBreak: 'break-word' } : {}) }}>
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

          if (codeCollapsible) {
            return <CollapsiblePre {...props}>{children}</CollapsiblePre>;
          }

          return <pre {...props} style={{ ...(codeWrappable ? { whiteSpace: 'pre-wrap', wordBreak: 'break-word' } : {}) }}>{children}</pre>;
        }
      }),
      [citationMap, webSearchMap, handleCitationClick, isStreaming, codeCollapsible, codeWrappable, codeShowLineNumbers]
    );

    return (
      <div
        ref={containerRef}
        className={`prose prose-sm max-w-full dark:prose-invert message-content leading-7 ${streamingClass}`}
      >
        {isRefDirectWrite ? (
          // 已完成的标题/段落用 Markdown 渲染；当前行仍走 ref 直写 + 闭合公式水合。
          <div className="relative min-h-[20px]">
            <div className="streaming-md-committed">
              {processedCommitted ? (
                <ReactMarkdown
                  remarkPlugins={baseRemarkPlugins}
                  rehypePlugins={rehypePlugins}
                  components={markdownComponents}
                >
                  {processedCommitted}
                </ReactMarkdown>
              ) : null}
            </div>
            <div
              key="streaming-md-tail"
              ref={streamingRef}
              className={`streaming-md-tail whitespace-pre-wrap break-words${enableBlurReveal ? ` blur-intensity-${blurIntensity}` : ''}`}
            />
            {showWaitingDots && (
              <div className="streaming-dots absolute left-0 top-0">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
              </div>
            )}
          </div>
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
  streamingMarkdownAreEqual
);

// 为 React DevTools 添加显示名称
StreamingMarkdown.displayName = 'StreamingMarkdown';

export { processLatexBrackets } from '../utils/processLatexBrackets.js';
export default StreamingMarkdown;
