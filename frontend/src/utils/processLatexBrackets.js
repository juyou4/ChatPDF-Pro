/**
 * LaTeX 公式括号转换工具
 *
 * 移植自 cherry-studio 的平衡括号匹配算法，
 * 将 \[...\] 和 \(...\) 转换为 $$...$$ 和 $...$
 *
 * 特性：
 * - 保护代码块和链接，避免误转换
 * - 支持嵌套括号的平衡匹配
 * - 转义括号 \\(\\) 或 \\[\\] 不会被处理
 *
 * @module processLatexBrackets
 */

// 检查是否包含潜在的 LaTeX 模式
const containsLatexRegex = /\\\(.*?\\\)|\\\[.*?\\\]|\$[^$\n]*\$|\\[A-Za-z]{2,}/s;

// 常见数学命令（用于识别“无分隔符但明显是公式”的片段）
const MATH_COMMAND_HINTS = new Set([
  'times', 'cdot', 'otimes', 'oplus', 'div', 'pm', 'mp',
  'frac', 'sqrt', 'sum', 'prod', 'int',
  'leq', 'geq', 'neq', 'approx',
  'alpha', 'beta', 'gamma', 'theta', 'lambda', 'mu', 'sigma', 'pi',
  'boldsymbol', 'mathbf', 'mathrm', 'mathit', 'text',
  'uparrow', 'downarrow', 'left', 'right'
]);

const normalizeLatexEscapes = (expr) => {
  if (!expr) return expr;
  // 把过度转义的命令 \\times / \\frac 还原为 \times / \frac
  return expr.replace(/\\\\(?=[A-Za-z])/g, '\\');
};

const splitMathSegments = (text) => {
  return text.split(/(\$\$[\s\S]*?\$\$|\$[^$\n]+\$)/g);
};

const normalizeExistingDollarMath = (content) => {
  const segments = splitMathSegments(content);
  return segments
    .map((seg) => {
      if (!seg) return seg;
      if (seg.startsWith('$$') && seg.endsWith('$$')) {
        const body = seg.slice(2, -2);
        return `$$${normalizeLatexEscapes(body)}$$`;
      }
      if (seg.startsWith('$') && seg.endsWith('$')) {
        const body = seg.slice(1, -1);
        return `$${normalizeLatexEscapes(body)}$`;
      }
      return seg;
    })
    .join('');
};

const AUTO_WRAP_LATEX_FRAGMENT_REGEX = /[A-Za-z0-9{}_^=+\-*/\\(). ]*\\[A-Za-z]{2,}[A-Za-z0-9{}_^=+\-*/\\(). ]*/g;

const maybeWrapLatexFragment = (fragment) => {
  if (!fragment || typeof fragment !== 'string') return fragment;
  const leading = (fragment.match(/^\s*/) || [''])[0];
  const trailing = (fragment.match(/\s*$/) || [''])[0];
  const core = fragment.slice(leading.length, fragment.length - trailing.length);
  const trimmed = core.trim();
  if (!trimmed) return fragment;
  if (trimmed.startsWith('$') && trimmed.endsWith('$')) return fragment;
  if (trimmed.length > 120) return fragment;

  const commandNames = Array.from(trimmed.matchAll(/\\([A-Za-z]{2,})/g))
    .map((m) => m[1].toLowerCase());
  if (commandNames.length === 0) return fragment;

  const hasMathHint = commandNames.some((cmd) => MATH_COMMAND_HINTS.has(cmd));
  if (!hasMathHint) return fragment;

  // 避免把英文说明句整段包进公式（例如 "Input is ...")
  const firstSlash = trimmed.indexOf('\\');
  const prefix = firstSlash > 0 ? trimmed.slice(0, firstSlash).trim() : '';
  if (prefix && /[a-z]{2,}/.test(prefix)) {
    return fragment;
  }

  return `${leading}$${normalizeLatexEscapes(trimmed)}$${trailing}`;
};

const autoWrapStandaloneLatexFragments = (content) => {
  const segments = splitMathSegments(content);
  return segments
    .map((seg) => {
      if (!seg) return seg;
      // 数学分段原样保留，只处理普通文本分段
      if (
        (seg.startsWith('$$') && seg.endsWith('$$')) ||
        (seg.startsWith('$') && seg.endsWith('$'))
      ) {
        return seg;
      }
      return seg.replace(AUTO_WRAP_LATEX_FRAGMENT_REGEX, (m) => maybeWrapLatexFragment(m));
    })
    .join('');
};

/**
 * 查找 LaTeX 数学公式的匹配括号对
 *
 * 使用平衡括号算法处理嵌套结构，正确识别转义字符
 *
 * @param {string} text 要搜索的文本
 * @param {string} openDelim 开始分隔符 (如 '\[' 或 '\(')
 * @param {string} closeDelim 结束分隔符 (如 '\]' 或 '\)')
 * @returns {Object|null} 匹配结果对象或 null
 */
const findLatexMatch = (text, openDelim, closeDelim) => {
  // 统计连续反斜杠：奇数个表示转义，偶数个表示未转义
  const escaped = (i) => {
    let count = 0;
    while (--i >= 0 && text[i] === '\\') count++;
    return count & 1;
  };

  // 查找第一个有效的开始标记
  for (let i = 0, n = text.length; i <= n - openDelim.length; i++) {
    // 没有找到开始分隔符或被转义，跳过
    if (!text.startsWith(openDelim, i) || escaped(i)) continue;

    // 处理嵌套结构
    for (let j = i + openDelim.length, depth = 1; j <= n - closeDelim.length && depth; j++) {
      // 计算当前位置对深度的影响：+1(开始), -1(结束), 0(无关)
      const delta =
        text.startsWith(openDelim, j) && !escaped(j) ? 1 : text.startsWith(closeDelim, j) && !escaped(j) ? -1 : 0;

      if (delta) {
        depth += delta;

        // 找到了匹配的结束位置
        if (!depth)
          return {
            start: i,
            end: j + closeDelim.length,
            pre: text.slice(0, i),
            body: text.slice(i + openDelim.length, j),
            post: text.slice(j + closeDelim.length)
          };

        // 跳过已处理的分隔符字符，避免重复检查
        j += (delta > 0 ? openDelim : closeDelim).length - 1;
      }
    }
  }

  return null;
};

/**
 * 转换 LaTeX 公式括号 \[\] 和 \(\) 为 Markdown 格式 $$...$$ 和 $...$
 *
 * @param {string} text 输入的 Markdown 文本
 * @returns {string} 处理后的字符串
 */
export const processLatexBrackets = (text) => {
  if (!text || typeof text !== 'string') return text;

  // 没有 LaTeX 模式直接返回
  if (!containsLatexRegex.test(text)) {
    return text;
  }

  // 保护代码块和链接
  const protectedItems = [];
  let processedContent = text;

  processedContent = processedContent
    // 保护代码块（包括多行代码块和行内代码）
    .replace(/(```[\s\S]*?```|`[^`]*`)/g, (match) => {
      const index = protectedItems.length;
      protectedItems.push(match);
      return `__CHATPDF_PROTECTED_${index}__`;
    })
    // 保护链接 [text](url)
    .replace(/\[([^[\]]*(?:\[[^\]]*\][^[\]]*)*)\]\([^)]*?\)/g, (match) => {
      const index = protectedItems.length;
      protectedItems.push(match);
      return `__CHATPDF_PROTECTED_${index}__`;
    });

  // LaTeX 括号转换函数
  const processMath = (content, openDelim, closeDelim, wrapper) => {
    let result = '';
    let remaining = content;

    while (remaining.length > 0) {
      const match = findLatexMatch(remaining, openDelim, closeDelim);
      if (!match) {
        result += remaining;
        break;
      }

      result += match.pre;
      result += `${wrapper}${normalizeLatexEscapes(match.body)}${wrapper}`;
      remaining = match.post;
    }

    return result;
  };

  // 先规范已有 $...$/$$...$$ 里的过度转义
  processedContent = normalizeExistingDollarMath(processedContent);

  // 先处理块级公式，再处理内联公式
  let result = processMath(processedContent, '\\[', '\\]', '$$');
  result = processMath(result, '\\(', '\\)', '$');

  // 最后尝试把“无分隔符但明显是公式”的 LaTeX 片段包裹为行内公式
  result = autoWrapStandaloneLatexFragments(result);

  // 还原被保护的内容
  result = result.replace(/__CHATPDF_PROTECTED_(\d+)__/g, (match, indexStr) => {
    const index = parseInt(indexStr, 10);
    if (index >= 0 && index < protectedItems.length) {
      return protectedItems[index];
    }
    return match;
  });

  return result;
};

export default processLatexBrackets;
