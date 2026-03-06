/**
 * 引文相关共享工具
 *
 * 统一 INLINE_CITATION_REGEX，避免 StreamingMarkdown 和 useMessageState
 * 使用不同捕获组结构导致的静默解析错误（B6 修复）
 *
 * 正则说明：
 *   - 排除图片语法 ![N] 和链接语法 [N](...)
 *   - 支持半角 [1] 和全角 【1】 两种格式
 *   - 捕获组 1：半角数字，捕获组 2：全角数字
 */
export const INLINE_CITATION_REGEX = /(?<!!)(?:\[(\d{1,3})\](?!\()|【(\d{1,3})】)/g;

/**
 * 从文本中提取按出现顺序去重的引文编号数组
 */
export const extractInlineCitationRefs = (text = '') => {
  const refs = [];
  const seen = new Set();
  for (const m of String(text).matchAll(INLINE_CITATION_REGEX)) {
    const ref = Number(m[1] ?? m[2]);
    if (!Number.isFinite(ref) || seen.has(ref)) continue;
    seen.add(ref);
    refs.push(ref);
  }
  return refs;
};
