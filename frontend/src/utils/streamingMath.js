/**
 * 流式阶段的闭合公式水合。
 *
 * 只替换已经成对的 $...$ / $$...$$ / \(...\) / \[...\] / \begin{env}...\end{env}。
 * 未闭合定界符保持原文，避免把半截公式提前渲染，也避免整段 ReactMarkdown 重绘打断模糊动画。
 */

import katex from 'katex'

const MATH_NODE_CLASS = 'streaming-math'
const MATH_SOURCE_LEN_ATTR = 'data-source-len'
const MATH_SOURCE_TEXT_ATTR = 'data-source-text'

const MATH_ENVIRONMENTS = new Set([
  'equation',
  'equation*',
  'align',
  'align*',
  'aligned',
  'gather',
  'gather*',
  'multline',
  'multline*',
  'split',
  'cases',
  'matrix',
  'pmatrix',
  'bmatrix',
  'vmatrix',
  'Vmatrix',
  'smallmatrix',
])

const isEscaped = (text, index) => {
  let slashes = 0
  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) {
    slashes += 1
  }
  return slashes % 2 === 1
}

const looksLikeTex = (tex, display) => {
  const raw = String(tex || '')
  const trimmed = raw.trim()
  if (!trimmed) return false
  if (display) return true
  if (raw.includes('\n')) return false
  if (/^\d+([.,]\d+)?$/.test(trimmed)) return false
  return /[\\^_{}=A-Za-z+\-*/]/.test(trimmed)
}

const createCodeMask = (text) => {
  const blocked = new Uint8Array(text.length)
  let index = 0
  while (index < text.length) {
    if (text.startsWith('```', index)) {
      const close = text.indexOf('```', index + 3)
      const stop = close === -1 ? text.length : close + 3
      blocked.fill(1, index, stop)
      index = stop
      continue
    }
    if (text[index] === '`') {
      const close = text.indexOf('`', index + 1)
      if (close === -1) {
        index += 1
        continue
      }
      blocked.fill(1, index, close + 1)
      index = close + 1
      continue
    }
    index += 1
  }
  return blocked
}

const pushSpan = (spans, start, end, display, tex) => {
  if (!looksLikeTex(tex, display)) return
  spans.push({ start, end, display, tex })
}

/**
 * 找出源文本里已经闭合、可以立刻渲染的公式区间。
 * @param {string} text
 * @param {{ enableSingleDollar?: boolean }} [options]
 */
export const findClosedMathSpans = (text, options = {}) => {
  if (!text || typeof text !== 'string') return []

  const enableSingleDollar = options.enableSingleDollar !== false
    || /(^|[^\\])\$[^$\n]*\\[A-Za-z]{2,}[^$\n]*\$/m.test(text)
  const blocked = createCodeMask(text)
  const spans = []
  let index = 0
  const length = text.length

  while (index < length) {
    if (blocked[index]) {
      index += 1
      continue
    }

    if (text.startsWith('\\begin{', index) && !isEscaped(index)) {
      const opener = text.slice(index).match(/^\\begin\{([a-zA-Z*]+)\}/)
      if (opener && MATH_ENVIRONMENTS.has(opener[1])) {
        const closer = `\\end{${opener[1]}}`
        const closeAt = text.indexOf(closer, index + opener[0].length)
        if (closeAt !== -1 && !blocked[closeAt]) {
          const end = closeAt + closer.length
          pushSpan(spans, index, end, true, text.slice(index, end))
          index = end
          continue
        }
      }
    }

    if (text.startsWith('\\[', index) && !isEscaped(index)) {
      const closeAt = text.indexOf('\\]', index + 2)
      if (closeAt !== -1 && !blocked[closeAt] && !isEscaped(closeAt)) {
        pushSpan(spans, index, closeAt + 2, true, text.slice(index + 2, closeAt))
        index = closeAt + 2
        continue
      }
      index += 2
      continue
    }

    if (text.startsWith('\\(', index) && !isEscaped(index)) {
      const closeAt = text.indexOf('\\)', index + 2)
      if (closeAt !== -1 && !blocked[closeAt] && !isEscaped(closeAt)) {
        pushSpan(spans, index, closeAt + 2, false, text.slice(index + 2, closeAt))
        index = closeAt + 2
        continue
      }
      index += 2
      continue
    }

    if (text.startsWith('$$', index) && !isEscaped(index)) {
      const closeAt = text.indexOf('$$', index + 2)
      if (closeAt !== -1 && !blocked[closeAt] && !isEscaped(closeAt)) {
        pushSpan(spans, index, closeAt + 2, true, text.slice(index + 2, closeAt))
        index = closeAt + 2
        continue
      }
      index += 2
      continue
    }

    if (enableSingleDollar && text[index] === '$' && !isEscaped(index) && !/\s/.test(text[index + 1] || '')) {
      let cursor = index + 1
      let found = false
      while (cursor < length && !blocked[cursor] && text[cursor] !== '\n') {
        if (text[cursor] === '$' && !isEscaped(cursor)) {
          if (text[cursor + 1] === '$') {
            cursor += 2
            continue
          }
          if (cursor > index + 1 && !/\s/.test(text[cursor - 1] || '')) {
            pushSpan(spans, index, cursor + 1, false, text.slice(index + 1, cursor))
            index = cursor + 1
            found = true
          }
          break
        }
        cursor += 1
      }
      if (!found) index += 1
      continue
    }

    index += 1
  }

  return spans
}

const BLOCK_LINE_RE = /^(?:#{1,6}[ \t]*\S|[-*+][ \t]+\S|\d+\.[ \t]+\S|>[ \t]+\S|\||(?:[-*_]){3,}\s*$)/

const findUnclosedFenceStart = (text) => {
  let index = 0
  let openAt = -1
  while (index < text.length) {
    if (text.startsWith('```', index)) {
      openAt = openAt === -1 ? index : -1
      index += 3
      continue
    }
    index += 1
  }
  return openAt
}

const findUnclosedDisplayMathStart = (text, closed) => {
  const covered = (index) => closed.some((span) => index >= span.start && index < span.end)
  const blocked = createCodeMask(text)
  let index = 0
  while (index < text.length) {
    if (blocked[index] || covered(index) || isEscaped(index)) {
      index += 1
      continue
    }
    if (text.startsWith('\\begin{', index)) {
      const opener = text.slice(index).match(/^\\begin\{([a-zA-Z*]+)\}/)
      if (opener && MATH_ENVIRONMENTS.has(opener[1])) {
        const closer = `\\end{${opener[1]}}`
        if (text.indexOf(closer, index + opener[0].length) === -1) return index
      }
    }
    if (text.startsWith('\\[', index) && text.indexOf('\\]', index + 2) === -1) return index
    if (text.startsWith('$$', index) && text.indexOf('$$', index + 2) === -1) return index
    index += 1
  }
  return -1
}

const isCommitNewline = (text, newlineIndex) => {
  if (text[newlineIndex] !== '\n') return false
  if (newlineIndex > 0 && text[newlineIndex - 1] === '\n') return true
  const lineStart = text.lastIndexOf('\n', newlineIndex - 1) + 1
  return BLOCK_LINE_RE.test(text.slice(lineStart, newlineIndex).trimEnd())
}

/**
 * 已经可以交给 Markdown 渲染的前缀：完整标题/列表/引用/分隔线，或段落空行。
 * 未写完的当前行、未闭合公式和未闭合代码块留在尾巴里，避免半截 ## 提前变成标题。
 */
export const getCommittedPrefix = (text, options = {}) => {
  if (!text || typeof text !== 'string') return ''

  const enableMath = options.enableMath !== false
  const closed = enableMath ? findClosedMathSpans(text, options) : []
  const unclosedMathStart = enableMath ? findUnclosedDisplayMathStart(text, closed) : -1
  const unclosedFenceStart = findUnclosedFenceStart(text)

  let lastCommitNl = -1
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== '\n') continue
    if (unclosedFenceStart !== -1 && index >= unclosedFenceStart) continue
    if (unclosedMathStart !== -1 && index >= unclosedMathStart) continue
    if (closed.some((span) => index >= span.start && index < span.end)) continue
    if (!isCommitNewline(text, index)) continue
    lastCommitNl = index
  }

  return lastCommitNl >= 0 ? text.slice(0, lastCommitNl + 1) : ''
}

const collectSourceSegments = (container) => {
  const segments = []
  let offset = 0

  const walk = (node) => {
    if (!node) return
    if (node.nodeType === Node.TEXT_NODE) {
      const value = node.textContent || ''
      if (value.length > 0) {
        segments.push({ type: 'text', node, start: offset, end: offset + value.length })
        offset += value.length
      }
      return
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return
    if (node.classList?.contains(MATH_NODE_CLASS)) {
      const sourceText = node.getAttribute(MATH_SOURCE_TEXT_ATTR) || ''
      const sourceLen = Number(node.getAttribute(MATH_SOURCE_LEN_ATTR))
      const length = Number.isFinite(sourceLen) && sourceLen > 0 ? sourceLen : sourceText.length
      segments.push({ type: 'math', node, start: offset, end: offset + length })
      offset += length
      return
    }
    const children = Array.from(node.childNodes)
    for (const child of children) walk(child)
  }

  walk(container)
  return { segments, length: offset }
}

const locateSourceOffset = (segments, offset, edge) => {
  if (!Array.isArray(segments) || segments.length === 0) return null

  if (edge === 'end') {
    for (let index = segments.length - 1; index >= 0; index -= 1) {
      const segment = segments[index]
      if (offset < segment.start || offset > segment.end) continue
      if (offset === segment.start && index > 0 && segments[index - 1].end === offset) continue
      if (segment.type === 'math') {
        return { node: segment.node, offset: offset === segment.start ? 0 : 1, math: true }
      }
      return { node: segment.node, offset: offset - segment.start, math: false }
    }
    return null
  }

  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]
    if (offset < segment.start || offset > segment.end) continue
    if (offset === segment.end && index < segments.length - 1 && segments[index + 1].start === offset) continue
    if (segment.type === 'math') {
      return { node: segment.node, offset: offset === segment.start ? 0 : 1, math: true }
    }
    return { node: segment.node, offset: offset - segment.start, math: false }
  }
  return null
}

const renderMathNode = (span, sourceText) => {
  const wrap = document.createElement(span.display ? 'div' : 'span')
  wrap.className = span.display
    ? `${MATH_NODE_CLASS} streaming-math-display`
    : `${MATH_NODE_CLASS} streaming-math-inline`
  wrap.setAttribute(MATH_SOURCE_LEN_ATTR, String(span.end - span.start))
  wrap.setAttribute(MATH_SOURCE_TEXT_ATTR, sourceText)
  try {
    katex.render(span.tex, wrap, {
      displayMode: span.display,
      throwOnError: false,
      strict: 'ignore',
      trust: false,
      output: 'html',
    })
  } catch {
    wrap.textContent = sourceText
  }
  return wrap
}

const pruneEmptyBlurSpans = (container) => {
  container.querySelectorAll('.blur-reveal-animate').forEach((node) => {
    if (!node.textContent) node.remove()
  })
}

export const deleteSourceRange = (container, start, end) => {
  if (!container || !Number.isFinite(start) || !Number.isFinite(end) || end <= start) return false
  const { segments } = collectSourceSegments(container)
  const startPos = locateSourceOffset(segments, start, 'start')
  const endPos = locateSourceOffset(segments, end, 'end')
  if (!startPos || !endPos) return false

  try {
    const range = document.createRange()
    if (startPos.math) {
      if (startPos.offset !== 0) return false
      range.setStartBefore(startPos.node)
    } else {
      range.setStart(startPos.node, startPos.offset)
    }
    if (endPos.math) {
      if (endPos.offset === 0) range.setEndBefore(endPos.node)
      else range.setEndAfter(endPos.node)
    } else {
      range.setEnd(endPos.node, endPos.offset)
    }
    range.deleteContents()
    pruneEmptyBlurSpans(container)
    return true
  } catch {
    return false
  }
}

/**
 * 从流式 DOM 还原源文本。公式节点按 data-source-text 计，而不是 KaTeX 渲染结果。
 */
export const getStreamedSourceText = (container) => {
  if (!container) return ''
  const { segments } = collectSourceSegments(container)
  let output = ''
  for (const segment of segments) {
    if (segment.type === 'math') {
      output += segment.node.getAttribute(MATH_SOURCE_TEXT_ATTR) || ''
    } else {
      output += segment.node.textContent || ''
    }
  }
  return output
}

/**
 * 把已经闭合的公式替换成 KaTeX 节点。未闭合区间保持原文。
 * @returns {number} 本轮新水合的公式数
 */
export const hydrateStreamingMath = (container, sourceText, options = {}) => {
  if (!container || typeof sourceText !== 'string' || !sourceText) return 0
  if (typeof document === 'undefined') return 0

  const spans = findClosedMathSpans(sourceText, {
    enableSingleDollar: options.enableSingleDollar !== false,
  })
  if (spans.length === 0) return 0

  let hydrated = 0
  for (const span of [...spans].reverse()) {
    const { segments } = collectSourceSegments(container)
    const overlapping = segments.filter((segment) => segment.start < span.end && segment.end > span.start)
    if (overlapping.length === 0) continue
    if (
      overlapping.length === 1
      && overlapping[0].type === 'math'
      && overlapping[0].start === span.start
      && overlapping[0].end === span.end
    ) {
      continue
    }
    if (overlapping.some((segment) => segment.type === 'math')) continue

    const startPos = locateSourceOffset(segments, span.start, 'start')
    const endPos = locateSourceOffset(segments, span.end, 'end')
    if (!startPos || !endPos || startPos.math || endPos.math) continue

    try {
      const range = document.createRange()
      range.setStart(startPos.node, startPos.offset)
      range.setEnd(endPos.node, endPos.offset)
      range.deleteContents()
      range.insertNode(renderMathNode(span, sourceText.slice(span.start, span.end)))
      hydrated += 1
    } catch {
      // 某一对公式水合失败时保持原文，不影响后续字符追加。
    }
  }

  if (hydrated > 0) pruneEmptyBlurSpans(container)
  return hydrated
}
