const DEFAULT_PAGE_WIDTH = 612;
const DEFAULT_PAGE_HEIGHT = 792;

export const normalizeCitationText = (value) => String(value || '')
  .replace(/\s+/g, '')
  .toLowerCase();

export const normalizeCitationBBox = (value) => {
  if (!Array.isArray(value) || value.length < 4) return null;
  const bbox = value.slice(0, 4).map((item) => Number(item));
  if (bbox.some((item) => !Number.isFinite(item))) return null;
  const [x0, y0, x1, y1] = bbox;
  if (x1 <= x0 || y1 <= y0) return null;
  return bbox;
};

const roundCoordinate = (value) => Math.round(Number(value) * 1000) / 1000;

const resolveAxisPadding = (padding, axis) => Math.max(0, Number(
  typeof padding === 'object'
    ? (axis === 'x'
      ? (padding.x ?? padding.horizontal ?? padding.inline)
      : (padding.y ?? padding.vertical ?? padding.block))
    : padding
) || 0);

const normalizePageSize = (value) => {
  if (Array.isArray(value) && value.length >= 2) {
    const width = Number(value[0]);
    const height = Number(value[1]);
    if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
      return { width, height };
    }
  }
  if (value && typeof value === 'object') {
    const width = Number(value.width ?? value.width_pts);
    const height = Number(value.height ?? value.height_pts);
    if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
      return { width, height };
    }
  }
  return null;
};

export const isCitationGeometryCurrent = (highlightInfo, blockIndex) => {
  const anchorGeneration = String(
    highlightInfo?.citationAnchor?.parseGeneration
      || highlightInfo?.parseGeneration
      || ''
  ).trim();
  if (!anchorGeneration) return true;
  const currentGeneration = String(blockIndex?.parse_generation || '').trim();
  return Boolean(currentGeneration) && currentGeneration === anchorGeneration;
};

const coordinateSpaceName = (value) => String(value || 'pdf_top_left_points')
  .trim()
  .toLowerCase();

const convertBBoxToPagePoints = (bbox, {
  coordinateSpace,
  sourcePageSize,
  targetPageSize,
}) => {
  const normalized = normalizeCitationBBox(bbox);
  if (!normalized) return null;

  const target = targetPageSize || { width: DEFAULT_PAGE_WIDTH, height: DEFAULT_PAGE_HEIGHT };
  const source = sourcePageSize || target;
  const space = coordinateSpaceName(coordinateSpace);
  let [x0, y0, x1, y1] = normalized;

  if (['normalized', 'normalized_0_1', 'ratio', 'relative'].includes(space)) {
    x0 *= target.width;
    x1 *= target.width;
    y0 *= target.height;
    y1 *= target.height;
  } else if (['normalized_0_1000', 'normalized_1000', 'mineru_1000'].includes(space)) {
    x0 = (x0 / 1000) * target.width;
    x1 = (x1 / 1000) * target.width;
    y0 = (y0 / 1000) * target.height;
    y1 = (y1 / 1000) * target.height;
  } else {
    if (['pdf_bottom_left_points', 'pdf_bottom_left'].includes(space)) {
      const oldY0 = y0;
      y0 = source.height - y1;
      y1 = source.height - oldY0;
    }
    x0 = (x0 / source.width) * target.width;
    x1 = (x1 / source.width) * target.width;
    y0 = (y0 / source.height) * target.height;
    y1 = (y1 / source.height) * target.height;
  }

  const clamped = [
    Math.max(0, Math.min(target.width, x0)),
    Math.max(0, Math.min(target.height, y0)),
    Math.max(0, Math.min(target.width, x1)),
    Math.max(0, Math.min(target.height, y1)),
  ];
  return normalizeCitationBBox(clamped);
};

export const citationRectsToRendered = ({
  rects,
  bbox,
  coordinateSpace,
  pageSize,
  pageData,
  renderedPageSize,
  scale = 1,
  padding = 2,
}) => {
  const candidates = Array.isArray(rects) && rects.length
    ? rects
    : (normalizeCitationBBox(bbox) ? [bbox] : []);
  if (!candidates.length) return [];

  const sourcePageSize = normalizePageSize(pageSize);
  const targetPageSize = normalizePageSize(pageData && {
    width: pageData.width_pts,
    height: pageData.height_pts,
  }) || normalizePageSize(renderedPageSize) || sourcePageSize || {
    width: DEFAULT_PAGE_WIDTH,
    height: DEFAULT_PAGE_HEIGHT,
  };
  const effectiveScale = Number.isFinite(Number(scale)) && Number(scale) > 0 ? Number(scale) : 1;
  const horizontalPadding = resolveAxisPadding(padding, 'x');
  const verticalPadding = resolveAxisPadding(padding, 'y');

  return candidates
    .map((candidate) => convertBBoxToPagePoints(candidate, {
      coordinateSpace,
      sourcePageSize,
      targetPageSize,
    }))
    .filter(Boolean)
    .map(([x0, y0, x1, y1]) => ({
      left: roundCoordinate(Math.max(0, x0 * effectiveScale - horizontalPadding)),
      top: roundCoordinate(Math.max(0, y0 * effectiveScale - verticalPadding)),
      width: roundCoordinate((x1 - x0) * effectiveScale + horizontalPadding * 2),
      height: roundCoordinate((y1 - y0) * effectiveScale + verticalPadding * 2),
    }));
};

const charBigrams = (value) => {
  const normalized = normalizeCitationText(value);
  if (normalized.length < 2) return new Set(normalized ? [normalized] : []);
  const result = new Set();
  for (let index = 0; index < normalized.length - 1; index += 1) {
    result.add(normalized.slice(index, index + 2));
  }
  return result;
};

const diceSimilarity = (left, right) => {
  const a = charBigrams(left);
  const b = charBigrams(right);
  if (!a.size || !b.size) return 0;
  let overlap = 0;
  a.forEach((value) => {
    if (b.has(value)) overlap += 1;
  });
  return (2 * overlap) / (a.size + b.size);
};

export const findBestCitationBlock = ({
  blocks,
  text,
  startPhrase,
  endPhrase,
  blockId,
  allowBlockId = true,
}) => {
  const candidates = (Array.isArray(blocks) ? blocks : [])
    .filter((block) => normalizeCitationBBox(block?.bbox));
  if (!candidates.length) return null;

  if (allowBlockId && blockId) {
    const exact = candidates.find((block) => String(block.block_id || '') === String(blockId));
    if (exact) return { block: exact, source: 'block_id', score: Number.POSITIVE_INFINITY };
  }

  const target = normalizeCitationText(text);
  const start = normalizeCitationText(startPhrase);
  const end = normalizeCitationText(endPhrase);
  if (!target && !start && !end) return null;

  let best = null;
  candidates.forEach((block) => {
    const blockText = normalizeCitationText(block?.text);
    if (!blockText) return;
    let score = 0;
    if (target) {
      if (blockText.includes(target)) score += 120 + Math.min(target.length, 240) / 8;
      else if (target.length >= 12 && target.includes(blockText)) score += 75;
      score += diceSimilarity(blockText, target) * 50;
    }
    if (start && blockText.includes(start)) score += 45;
    if (end && blockText.includes(end)) score += 45;
    if (start && end && blockText.includes(start) && blockText.includes(end)) score += 25;
    if (score <= 16) return;
    const lengthPenalty = Math.min(Math.abs(blockText.length - (target.length || blockText.length)), 1200) / 1200;
    score -= lengthPenalty;
    if (!best || score > best.score) best = { block, source: 'text', score };
  });
  return best;
};

const collectMatches = (haystack, needle, limit = 12) => {
  if (!needle) return [];
  const matches = [];
  let from = 0;
  while (from < haystack.length && matches.length < limit) {
    const index = haystack.indexOf(needle, from);
    if (index < 0) break;
    matches.push(index);
    from = index + Math.max(1, Math.floor(needle.length / 2));
  }
  return matches;
};

export const findCitationTextRange = ({ fullText, text, startPhrase, endPhrase }) => {
  const haystack = normalizeCitationText(fullText);
  const target = normalizeCitationText(text);
  const start = normalizeCitationText(startPhrase);
  const end = normalizeCitationText(endPhrase);
  if (!haystack || (!target && !start && !end)) return null;

  if (start || end) {
    const startMatches = collectMatches(haystack, start);
    const endMatches = collectMatches(haystack, end);
    const expectedLength = Math.min(Math.max(target.length || start.length || end.length, 24), 260);
    let best = null;
    for (const startIndex of startMatches.length ? startMatches : [-1]) {
      for (const endIndex of endMatches.length ? endMatches : [-1]) {
        if (startIndex < 0 && endIndex < 0) continue;
        const rangeStart = startIndex >= 0
          ? startIndex
          : Math.max(0, endIndex + end.length - expectedLength);
        const rangeEnd = endIndex >= 0
          ? endIndex + end.length
          : Math.min(haystack.length, rangeStart + expectedLength);
        if (rangeEnd <= rangeStart || rangeEnd - rangeStart > 320) continue;
        const window = haystack.slice(rangeStart, rangeEnd);
        const similarity = target ? diceSimilarity(window, target) : 0;
        const score = (startIndex >= 0 ? 30 : 0)
          + (endIndex >= 0 ? 30 : 0)
          + similarity * 50
          - Math.abs((rangeEnd - rangeStart) - expectedLength) * 0.03;
        if (!best || score > best.score) {
          best = { startIndex: rangeStart, endIndex: rangeEnd, score };
        }
      }
    }
    if (best) return { startIndex: best.startIndex, endIndex: best.endIndex };
  }

  const exactIndex = target ? haystack.indexOf(target) : -1;
  if (exactIndex >= 0) {
    return { startIndex: exactIndex, endIndex: exactIndex + target.length };
  }

  const candidateLength = Math.min(120, target.length);
  if (candidateLength >= 20) {
    const starts = [
      Math.max(0, Math.floor((target.length - candidateLength) / 2)),
      0,
      Math.max(0, target.length - candidateLength),
    ];
    for (const candidateStart of starts) {
      const candidate = target.slice(candidateStart, candidateStart + candidateLength);
      const index = haystack.indexOf(candidate);
      if (index >= 0) return { startIndex: index, endIndex: index + candidate.length };
    }
  }
  return null;
};

export const mergeClientRectsByLine = (rects, pageRect, padding = 3) => {
  const valid = (Array.isArray(rects) ? rects : [])
    .map((rect) => ({
      left: Number(rect.left),
      right: Number(rect.right),
      top: Number(rect.top),
      bottom: Number(rect.bottom),
      width: Number(rect.width ?? (Number(rect.right) - Number(rect.left))),
      height: Number(rect.height ?? (Number(rect.bottom) - Number(rect.top))),
    }))
    .filter((rect) => Object.values(rect).every(Number.isFinite) && rect.width > 1 && rect.height > 1)
    .sort((left, right) => left.top - right.top || left.left - right.left);
  if (!valid.length) return [];

  const lines = [];
  valid.forEach((rect) => {
    const center = (rect.top + rect.bottom) / 2;
    let line = lines.find((candidate) => {
      const candidateCenter = candidate.reduce(
        (sum, item) => sum + (item.top + item.bottom) / 2,
        0
      ) / candidate.length;
      const tolerance = Math.max(2, Math.min(rect.height, candidate[0].height) * 0.55);
      return Math.abs(center - candidateCenter) <= tolerance;
    });
    if (!line) {
      line = [];
      lines.push(line);
    }
    line.push(rect);
  });

  const merged = [];
  lines.forEach((line) => {
    line.sort((left, right) => left.left - right.left);
    let current = null;
    line.forEach((rect) => {
      const gapLimit = Math.max(4, rect.height * 0.6);
      if (!current || rect.left - current.right > gapLimit) {
        if (current) merged.push(current);
        current = { ...rect };
        return;
      }
      current.left = Math.min(current.left, rect.left);
      current.right = Math.max(current.right, rect.right);
      current.top = Math.min(current.top, rect.top);
      current.bottom = Math.max(current.bottom, rect.bottom);
      current.width = current.right - current.left;
      current.height = current.bottom - current.top;
    });
    if (current) merged.push(current);
  });

  const originLeft = Number(pageRect?.left) || 0;
  const originTop = Number(pageRect?.top) || 0;
  const horizontalPadding = resolveAxisPadding(padding, 'x');
  const verticalPadding = resolveAxisPadding(padding, 'y');
  return merged
    .sort((left, right) => left.top - right.top || left.left - right.left)
    .map((rect) => ({
      left: Math.max(0, rect.left - originLeft - horizontalPadding),
      top: Math.max(0, rect.top - originTop - verticalPadding),
      width: rect.width + horizontalPadding * 2,
      height: rect.height + verticalPadding * 2,
    }));
};

export const collectTextRangeClientRects = (
  spans,
  { startIndex, endIndex },
  createRange = () => document.createRange()
) => {
  if (!Array.isArray(spans) || startIndex < 0 || endIndex <= startIndex) return [];
  const rects = [];
  let current = 0;

  spans.forEach((span) => {
    const text = String(span?.textContent || '');
    const characterPositions = [];
    for (let index = 0; index < text.length; index += 1) {
      if (!/\s/.test(text[index])) characterPositions.push(index);
    }
    const spanStart = current;
    const spanEnd = current + characterPositions.length;
    current = spanEnd;
    if (!characterPositions.length || endIndex <= spanStart || startIndex >= spanEnd) return;

    const localStart = Math.max(0, startIndex - spanStart);
    const localEnd = Math.min(characterPositions.length, endIndex - spanStart);
    const node = span.firstChild;
    if (!node || localEnd <= localStart) return;

    const domStart = characterPositions[localStart] ?? text.length;
    const domEnd = localEnd > 0
      ? (characterPositions[localEnd - 1] ?? (text.length - 1)) + 1
      : domStart;
    if (domEnd <= domStart) return;

    try {
      const range = createRange();
      range.setStart(node, domStart);
      range.setEnd(node, domEnd);
      rects.push(...Array.from(range.getClientRects()));
    } catch {
      // 单个异常文本节点不应阻断其余行的定位。
    }
  });

  return rects;
};
