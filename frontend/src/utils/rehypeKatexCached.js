/**
 * 带缓存的 rehype-katex 插件
 *
 * 拦截 math/inlineMath 节点的渲染过程：
 * - 渲染前先查缓存，命中则直接使用缓存的 HTML
 * - 未命中则调用 KaTeX 渲染，并将结果存入缓存
 *
 * 通过替换 rehype-katex，实现对 KaTeX 渲染结果的透明缓存，
 * 相同公式不重复渲染。
 *
 * @module rehypeKatexCached
 * @see 需求 6.2 - KaTeX 公式渲染缓存
 * @see Property 8 - KaTeX 公式缓存命中
 */

import { visit } from 'unist-util-visit';
import katex from 'katex';
import katexCache from './katexCache.js';

/**
 * 创建带缓存的 rehype-katex 插件
 * @param {Object} options - KaTeX 渲染选项
 * @param {boolean} [options.strict=false] - 严格模式
 * @param {boolean} [options.trust=true] - 信任输入
 * @param {string} [options.output='html'] - 输出格式
 * @returns {Function} rehype 插件函数
 */
function rehypeKatexCached(options = {}) {
  const katexOptions = {
    strict: false,
    trust: true,
    output: 'html',
    throwOnError: false,
    ...options,
  };

  return (tree) => {
    visit(tree, (node) => {
      // 处理 math（块级公式）和 inlineMath（行内公式）节点
      // 这些节点由 remark-math 生成
      if (node.type !== 'element') return;

      const classes = node.properties?.className || [];
      const isMathBlock = classes.includes('math-display') || classes.includes('math');
      const isInlineMath = classes.includes('math-inline');

      if (!isMathBlock && !isInlineMath) return;

      // 提取公式文本：从子节点中获取纯文本内容
      const expression = extractText(node);
      if (!expression) return;

      const displayMode = isMathBlock;
      // 缓存键：公式文本 + 显示模式
      const cacheKey = `${displayMode ? 'D' : 'I'}:${expression}`;

      // 查缓存
      let html = katexCache.get(cacheKey);

      if (html === undefined) {
        // 缓存未命中，调用 KaTeX 渲染
        try {
          html = katex.renderToString(expression, {
            ...katexOptions,
            displayMode,
          });
        } catch (err) {
          // 渲染失败时返回错误提示，不缓存错误结果
          html = `<span class="katex-error" title="${escapeAttr(err.message)}">${escapeHtml(expression)}</span>`;
          // 不缓存错误结果，直接设置到节点
          setNodeHtml(node, html, displayMode);
          return;
        }
        // 存入缓存
        katexCache.set(cacheKey, html);
      }

      // 将渲染结果设置到节点
      setNodeHtml(node, html, displayMode);
    });
  };
}

/**
 * 从 HAST 节点中递归提取纯文本内容
 * @param {Object} node - HAST 节点
 * @returns {string} 纯文本
 */
function extractText(node) {
  if (node.type === 'text') return node.value || '';
  if (node.children) {
    return node.children.map(extractText).join('');
  }
  return '';
}

/**
 * 将渲染后的 HTML 设置到 HAST 节点
 * 替换节点的子节点为 raw HTML 节点
 * @param {Object} node - HAST 节点
 * @param {string} html - 渲染后的 HTML
 * @param {boolean} displayMode - 是否为块级公式
 */
function setNodeHtml(node, html, displayMode) {
  // 将节点转换为包含 raw HTML 的节点
  node.children = [{ type: 'raw', value: html }];
  // 保留原有的 className，移除可能导致 rehype-katex 重复处理的标记
  node.properties = node.properties || {};
  node.properties.className = displayMode
    ? ['katex-display-cached']
    : ['katex-inline-cached'];
}

/**
 * HTML 属性值转义
 * @param {string} str
 * @returns {string}
 */
function escapeAttr(str) {
  return (str || '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * HTML 文本转义
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  return (str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export { rehypeKatexCached, extractText };
export default rehypeKatexCached;
