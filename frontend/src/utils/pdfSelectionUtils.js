/*
 * PDF text selection normalization.
 *
 * Browser selection.toString() is presentation dependent: it can include
 * page headers, lose the boundary offsets of the first/last span, and cannot
 * tell consumers which page a fragment came from.  This module reconstructs
 * the selection from PDF text-layer spans and keeps the DOM-specific work in
 * one place for translation, notes, highlights, copy and selected-text chat.
 */

const PAGE_SELECTOR = '[data-pdf-page-number]';
const SPAN_SELECTOR = [
  '.react-pdf__Page__textContent span',
  '.textLayer span',
  '[data-pdf-text-layer] span',
].join(', ');

const CJK_CHAR = /[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]/u;
const LATIN_OR_DIGIT = /[A-Za-z0-9]/u;

const getDocument = (node) => node?.ownerDocument || globalThis.document || null;

const getRangeConstructor = (node) => {
  const documentObject = getDocument(node);
  return documentObject?.defaultView?.Range || globalThis.Range || null;
};

const getTextNodeFilter = (node) => (
  node?.ownerDocument?.defaultView?.NodeFilter?.SHOW_TEXT
  || globalThis.NodeFilter?.SHOW_TEXT
  || 4
);

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const textLength = (node) => String(node?.textContent || '').length;

const isInside = (root, node) => (
  Boolean(root && node && (node === root || root.contains?.(node)))
);

const comparePoints = (left, right, node) => {
  if (!left || !right) return null;
  const documentObject = getDocument(node);
  const RangeConstructor = getRangeConstructor(node);
  if (!documentObject?.createRange || !RangeConstructor) return null;
  try {
    const leftRange = documentObject.createRange();
    const rightRange = documentObject.createRange();
    leftRange.setStart(left.container, left.offset);
    leftRange.collapse(true);
    rightRange.setStart(right.container, right.offset);
    rightRange.collapse(true);
    return leftRange.compareBoundaryPoints(RangeConstructor.START_TO_START, rightRange);
  } catch {
    return null;
  }
};

const getTextOffset = (root, container, offset) => {
  if (!isInside(root, container)) return null;
  const documentObject = getDocument(root);
  if (!documentObject?.createRange) return null;
  try {
    const range = documentObject.createRange();
    range.selectNodeContents(root);
    range.setEnd(container, clamp(Number(offset) || 0, 0, container.nodeType === 3
      ? textLength(container)
      : container.childNodes?.length || 0));
    return range.toString().length;
  } catch {
    return null;
  }
};

const setRangeOffset = (range, method, root, offset) => {
  const targetOffset = clamp(Number(offset) || 0, 0, textLength(root));
  const documentObject = getDocument(root);
  if (!documentObject?.createTreeWalker) {
    range[method](root, targetOffset);
    return;
  }

  const walker = documentObject.createTreeWalker(root, getTextNodeFilter(root));
  let remaining = targetOffset;
  let node = walker.nextNode();
  while (node) {
    const length = textLength(node);
    if (remaining <= length) {
      range[method](node, remaining);
      return;
    }
    remaining -= length;
    node = walker.nextNode();
  }

  range[method](root, root.childNodes?.length || 0);
};

const createTextRange = (span, start, end) => {
  const documentObject = getDocument(span);
  if (!documentObject?.createRange) return null;
  try {
    const range = documentObject.createRange();
    setRangeOffset(range, 'setStart', span, start);
    setRangeOffset(range, 'setEnd', span, end);
    return range;
  } catch {
    return null;
  }
};

const getClientRects = (span, start, end) => {
  const range = createTextRange(span, start, end);
  const rangeRects = range?.getClientRects?.();
  if (rangeRects?.length) return Array.from(rangeRects);
  const spanRect = span?.getBoundingClientRect?.();
  return spanRect && spanRect.width > 0 && spanRect.height > 0 ? [spanRect] : [];
};

const getPageElement = (span) => span?.closest?.(PAGE_SELECTOR) || null;

export const getSelectionText = (selection) => {
  if (!selection) return '';
  try {
    return String(selection.toString?.() || '').trim();
  } catch {
    return '';
  }
};

export const hasSelectionText = (selection) => Boolean(getSelectionText(selection));

const getSpanSelectionOffsets = (selectionRange, span) => {
  const spanLength = textLength(span);
  if (!spanLength) return null;

  const spanStart = { container: span, offset: 0 };
  const spanEnd = { container: span, offset: span.childNodes?.length || 0 };
  const selectionStart = {
    container: selectionRange.startContainer,
    offset: selectionRange.startOffset,
  };
  const selectionEnd = {
    container: selectionRange.endContainer,
    offset: selectionRange.endOffset,
  };

  const startVsEnd = comparePoints(selectionStart, spanEnd, span);
  const endVsStart = comparePoints(selectionEnd, spanStart, span);
  if (startVsEnd === null || endVsStart === null || startVsEnd >= 0 || endVsStart <= 0) {
    return null;
  }

  const startVsStart = comparePoints(selectionStart, spanStart, span);
  const endVsEnd = comparePoints(selectionEnd, spanEnd, span);
  const start = startVsStart !== null && startVsStart <= 0
    ? 0
    : clamp(getTextOffset(span, selectionStart.container, selectionStart.offset) ?? 0, 0, spanLength);
  const end = endVsEnd !== null && endVsEnd >= 0
    ? spanLength
    : clamp(getTextOffset(span, selectionEnd.container, selectionEnd.offset) ?? spanLength, 0, spanLength);
  if (end <= start) return null;
  return { start, end };
};

const getRectLineMeta = (rects) => {
  const first = rects?.[0];
  if (!first) return { left: 0, top: 0, height: 0 };
  return {
    left: Number(first.left) || 0,
    top: Number(first.top) || 0,
    height: Number(first.height) || 0,
  };
};

const inferSeparator = (previous, current) => {
  if (!previous) return '';
  const previousText = String(previous.fragment || '');
  const currentText = String(current.fragment || '');
  if (!previousText || !currentText) return '';
  if (previous.pageNumber !== current.pageNumber) return '\n\n';
  if (/\s$/u.test(previousText) || /^\s/u.test(currentText)) return '';

  const previousLine = getRectLineMeta(previous.clientRects);
  const currentLine = getRectLineMeta(current.clientRects);
  const lineThreshold = Math.max(2, Math.max(previousLine.height, currentLine.height) * 0.65);
  if (Math.abs(previousLine.top - currentLine.top) > lineThreshold) return '\n';

  const previousLast = previousText.slice(-1);
  const currentFirst = currentText[0];
  if (CJK_CHAR.test(previousLast) && CJK_CHAR.test(currentFirst)) return '';
  if (/^[,.;:!?%)\]}，。；：！？）》】]/u.test(currentFirst)) return '';
  if (/^[([{（《【]/u.test(currentFirst)) return '';
  if (LATIN_OR_DIGIT.test(previousLast) && LATIN_OR_DIGIT.test(currentFirst)) return ' ';
  return ' ';
};

export const normalizeSelectionText = (value) => {
  let text = String(value || '')
    .replace(/\u00ad/gu, '')
    .replace(/\r\n?/gu, '\n');

  // Join words split by a PDF line-end hyphen, while retaining mathematical
  // minus signs and hyphens that occur inside a word or at paragraph ends.
  text = text.replace(/([A-Za-z])-[ \t]*\n[ \t]*(?=[A-Za-z])/gu, '$1');
  text = text.replace(/[ \t]+\n/gu, '\n').replace(/\n[ \t]+/gu, '\n');
  text = text.replace(/([A-Za-z0-9)\]])\n(?=[A-Za-z0-9([])/gu, '$1 ');
  text = text.replace(/([\u3400-\u9fff])[ \t]+([\u3400-\u9fff])/gu, '$1$2');
  text = text.replace(/[ \t]{2,}/gu, ' ');
  text = text.replace(/\n{3,}/gu, '\n\n');
  return text.trim();
};

const normalizePageRect = (rect) => {
  if (!rect || !Number.isFinite(rect.left) || !Number.isFinite(rect.top)) return null;
  const width = Number(rect.width);
  const height = Number(rect.height);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return null;
  return {
    left: Number(rect.left),
    top: Number(rect.top),
    width,
    height,
  };
};

const defaultPageContext = (pageElement) => {
  const pageNumber = Number(pageElement?.dataset?.pdfPageNumber);
  return Number.isFinite(pageNumber) ? { pageNumber, pageElement } : null;
};

/**
 * Rebuild a native selection from PDF text-layer spans.
 *
 * `getPageContext` is intentionally injected by PDFViewer because it knows
 * the current scale/rotation and how to convert client rectangles to PDF
 * points.  It may return `null` for non-PDF selections.
 */
export const normalizePdfSelection = ({
  selection,
  root = null,
  getPageContext = defaultPageContext,
  isArtifact = () => false,
  fallbackPage = null,
  fallbackAnchor = null,
} = {}) => {
  const nativeText = getSelectionText(selection);
  const range = selection?.rangeCount > 0 ? selection.getRangeAt(0) : null;
  const selectionRoot = root || range?.commonAncestorContainer?.parentElement || null;
  if (!nativeText && !range) return null;

  const spans = selectionRoot?.querySelectorAll
    ? Array.from(selectionRoot.querySelectorAll(SPAN_SELECTOR))
    : [];
  const records = [];
  if (range && spans.length > 0) {
    spans.forEach((span) => {
      const offsets = getSpanSelectionOffsets(range, span);
      if (!offsets) return;
      const pageElement = getPageElement(span);
      const pageContext = pageElement ? getPageContext(pageElement) : null;
      if (!pageContext?.pageNumber) return;

      const fragment = String(span.textContent || '').slice(offsets.start, offsets.end);
      if (!fragment) return;
      const clientRects = getClientRects(span, offsets.start, offsets.end);
      if (isArtifact({ span, pageElement, pageContext, fragment, clientRects })) return;
      records.push({
        span,
        pageElement,
        pageContext,
        pageNumber: Number(pageContext.pageNumber),
        fragment,
        clientRects,
      });
    });
  }

  if (records.length === 0) {
    if (!nativeText) return null;
    const page = Number(fallbackPage || fallbackAnchor?.page || 0) || null;
    const cleanText = normalizeSelectionText(nativeText);
    if (!cleanText) return null;
    return {
      text: cleanText,
      raw_text: nativeText,
      raw_chars: nativeText.length,
      clean_chars: cleanText.length,
      page_start: page,
      page_end: page,
      page_rects: [],
      anchor: fallbackAnchor
        ? {
          ...fallbackAnchor,
          page: fallbackAnchor.page || page,
          page_start: fallbackAnchor.page_start || page,
          page_end: fallbackAnchor.page_end || page,
          raw_text: nativeText,
          raw_chars: nativeText.length,
          clean_chars: cleanText.length,
          page_rects: fallbackAnchor.page_rects || [],
        }
        : null,
    };
  }

  const rawParts = [];
  records.forEach((record, index) => {
    rawParts.push(inferSeparator(records[index - 1], record));
    rawParts.push(record.fragment);
  });
  const rawText = rawParts.join('');
  const cleanText = normalizeSelectionText(rawText);
  if (!cleanText) return null;

  const pageMap = new Map();
  records.forEach((record) => {
    const pageNumber = record.pageNumber;
    if (!pageMap.has(pageNumber)) {
      pageMap.set(pageNumber, {
        page: pageNumber,
        rects: [],
        page_size: record.pageContext.pageSize || null,
      });
    }
    const target = pageMap.get(pageNumber);
    record.clientRects.forEach((clientRect) => {
      const mapped = record.pageContext.mapClientRect
        ? record.pageContext.mapClientRect(clientRect, record.pageElement)
        : clientRect;
      const normalized = normalizePageRect(mapped);
      if (normalized) target.rects.push(normalized);
    });
  });

  const pageRects = Array.from(pageMap.values())
    .sort((left, right) => left.page - right.page)
    .filter((item) => item.rects.length > 0 || item.page_size);
  const firstPage = pageRects[0] || { page: records[0].pageNumber, rects: [], page_size: null };
  const lastPage = pageRects[pageRects.length - 1] || firstPage;
  const anchor = {
    page: firstPage.page,
    page_start: firstPage.page,
    page_end: lastPage.page,
    rects: firstPage.rects,
    page_rects: pageRects,
    page_size: firstPage.page_size || undefined,
    coordinate_space: 'pdf_top_left_points',
    raw_text: rawText,
    raw_chars: rawText.length,
    clean_chars: cleanText.length,
  };

  return {
    text: cleanText,
    raw_text: rawText,
    raw_chars: rawText.length,
    clean_chars: cleanText.length,
    page_start: firstPage.page,
    page_end: lastPage.page,
    page_rects: pageRects,
    anchor,
  };
};
