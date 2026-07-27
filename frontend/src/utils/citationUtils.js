/**
 * 引文相关共享工具
 *
 * 统一 INLINE_CITATION_REGEX，避免 StreamingMarkdown 和 useMessageState
 * 使用不同捕获组结构导致的静默解析错误（B6 修复）
 *
 * 正则只负责找候选；渲染前会排除 Markdown 代码、数学区间和明显的
 * 公式/数组下标。该规则需与后端引用投影保持一致，避免浏览器在重编号时
 * 篡改数学表达式。
 */
export const INLINE_CITATION_REGEX = /(?<!!)(?:\[(\d{1,3})\](?!\()|【(\d{1,3})】)/g;

const looksProtectedByCodeOrMath = (text, offset) => {
  if (!text || offset <= 0) return false;
  const lineStart = text.lastIndexOf('\n', offset - 1) + 1;
  const prefix = text.slice(lineStart, offset);
  if ((prefix.match(/(?<!\\)`/g) || []).length % 2 === 1 && /(?<!\\)`/.test(text.slice(offset))) return true;

  let mathOpen = false;
  for (const _delimiter of prefix.matchAll(/(?<!\\)\${1,2}/g)) {
    mathOpen = !mathOpen;
  }
  if (mathOpen && /(?<!\\)\${1,2}/.test(text.slice(offset))) return true;

  const latexOpen = Math.max(prefix.lastIndexOf('\\('), prefix.lastIndexOf('\\['));
  const latexClose = Math.max(prefix.lastIndexOf('\\)'), prefix.lastIndexOf('\\]'));
  return latexOpen > latexClose && (text.slice(offset).includes('\\)') || text.slice(offset).includes('\\]'));
};

const looksLikeFormulaSubscript = (text, offset) => {
  if (!text || offset <= 0) return false;
  const prefix = text.slice(0, offset);
  const previous = prefix.at(-1);
  if (previous === ']') {
    const candidateRegex = new RegExp(INLINE_CITATION_REGEX.source, 'g');
    const preceding = [...prefix.matchAll(candidateRegex)].find(
      (candidate) => candidate.index + candidate[0].length === offset,
    );
    return !preceding || !isInlineCitationMatch(text, preceding);
  }
  if (previous === ')' || previous === '}') return true;
  if (/\\[A-Za-z]+$/.test(prefix)) return true;
  const identifier = prefix.match(/([A-Za-z_][A-Za-z0-9_]*)$/)?.[1] || '';
  if (!identifier) return false;
  const normalized = identifier.toLowerCase();
  if (
    identifier.length === 1
    && identifier === identifier.toLowerCase()
    && 'xyzijkmnpqrstuvw'.includes(normalized)
  ) {
    return true;
  }
  return new Set([
    'arr', 'array', 'list', 'dict', 'data', 'tensor', 'matrix',
    'vector', 'values', 'value', 'items', 'index', 'indices', 'row',
    'col', 'column', 'token', 'tokens', 'input', 'output', 'mask',
  ]).has(normalized);
};

export const isInlineCitationMatch = (text, match) => {
  const offset = Number(match?.index);
  if (!Number.isFinite(offset)) return false;
  if (looksProtectedByCodeOrMath(text, offset)) return false;
  // Full-width brackets are reserved for citation syntax in our prompts.
  if (match?.[2]) return true;
  return !looksLikeFormulaSubscript(text, offset);
};

export const getInlineCitationMatches = (text = '') => {
  const source = String(text);
  const candidateRegex = new RegExp(INLINE_CITATION_REGEX.source, 'g');
  return [...source.matchAll(candidateRegex)].filter((match) => isInlineCitationMatch(source, match));
};

export const hasInlineCitationRefs = (text = '') => getInlineCitationMatches(text).length > 0;

export const replaceInlineCitationRefs = (text = '', replacer) => {
  const source = String(text);
  if (typeof replacer !== 'function') return source;
  return source.replace(INLINE_CITATION_REGEX, (match, halfWidthRef, fullWidthRef, offset) => {
    const candidate = { 0: match, 1: halfWidthRef, 2: fullWidthRef, index: offset };
    if (!isInlineCitationMatch(source, candidate)) return match;
    return replacer(match, halfWidthRef, fullWidthRef, offset, source);
  });
};

/**
 * 从文本中提取按出现顺序去重的引文编号数组
 */
export const extractInlineCitationRefs = (text = '') => {
  const refs = [];
  const seen = new Set();
  for (const m of getInlineCitationMatches(text)) {
    const ref = Number(m[1] ?? m[2]);
    if (!Number.isFinite(ref) || seen.has(ref)) continue;
    seen.add(ref);
    refs.push(ref);
  }
  return refs;
};
