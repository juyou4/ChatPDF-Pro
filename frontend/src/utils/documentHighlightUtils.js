const toFiniteNumber = (value) => {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const stableTextHash = (value) => {
  let hash = 2166136261;
  for (const char of String(value || '')) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
};

export const normalizeDocumentHighlightRect = (value) => {
  const source = Array.isArray(value)
    ? {
      left: value[0],
      top: value[1],
      width: Number(value[2]) - Number(value[0]),
      height: Number(value[3]) - Number(value[1]),
    }
    : value;
  if (!source || typeof source !== 'object') return null;

  const left = toFiniteNumber(source.left);
  const top = toFiniteNumber(source.top);
  const width = toFiniteNumber(source.width);
  const height = toFiniteNumber(source.height);
  if (left === null || top === null || width === null || height === null) return null;
  if (left < 0 || top < 0 || width <= 0 || height <= 0) return null;

  return { left, top, width, height };
};

/** Preset highlighter-ink colors (vivid enough to stay distinct under multiply). */
export const DOCUMENT_HIGHLIGHT_COLORS = [
  { id: 'yellow', label: '黄色', value: '#FFE066' },
  { id: 'green', label: '绿色', value: '#7DDE92' },
  { id: 'blue', label: '蓝色', value: '#7EB6FF' },
  { id: 'pink', label: '粉色', value: '#FF9ECF' },
  { id: 'purple', label: '紫色', value: '#C4B5FD' },
  { id: 'orange', label: '橙色', value: '#FFB86B' },
];

export const DEFAULT_DOCUMENT_HIGHLIGHT_COLOR = DOCUMENT_HIGHLIGHT_COLORS[0].value;

export const DOCUMENT_HIGHLIGHT_STYLES = Object.freeze({
  HIGHLIGHT: 'highlight',
  UNDERLINE: 'underline',
});

export const DOCUMENT_NOTE_ANCHOR_TYPES = Object.freeze({
  PAGE: 'page',
  SELECTION: 'selection',
});

export const normalizeDocumentHighlightStyle = (value) => (
  String(value || '').trim().toLowerCase() === DOCUMENT_HIGHLIGHT_STYLES.UNDERLINE
    ? DOCUMENT_HIGHLIGHT_STYLES.UNDERLINE
    : DOCUMENT_HIGHLIGHT_STYLES.HIGHLIGHT
);

/** Map legacy saved colors onto the current highlighter palette. */
const LEGACY_HIGHLIGHT_COLOR_MAP = {
  '#F2C15C': '#FFE066',
  '#7BC67E': '#7DDE92',
  '#6BA3E8': '#7EB6FF',
  '#E88AB5': '#FF9ECF',
  '#A78BFA': '#C4B5FD',
  '#FB923C': '#FFB86B',
};

export const normalizeDocumentHighlightColor = (value) => {
  const raw = String(value || '').trim();
  let hex = '';
  if (/^#[0-9A-Fa-f]{6}$/.test(raw)) {
    hex = `#${raw.slice(1).toUpperCase()}`;
  } else if (/^#[0-9A-Fa-f]{3}$/.test(raw)) {
    const r = raw[1];
    const g = raw[2];
    const b = raw[3];
    hex = `#${r}${r}${g}${g}${b}${b}`.toUpperCase();
  } else {
    const preset = DOCUMENT_HIGHLIGHT_COLORS.find(
      (item) => item.value.toLowerCase() === raw.toLowerCase() || item.id === raw.toLowerCase()
    );
    return preset?.value || DEFAULT_DOCUMENT_HIGHLIGHT_COLOR;
  }
  return LEGACY_HIGHLIGHT_COLOR_MAP[hex] || hex;
};

/** Convert #RRGGBB to rgba() for PDF overlays. */
export const highlightColorToRgba = (color, alpha = 0.34) => {
  const hex = normalizeDocumentHighlightColor(color).replace('#', '');
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  const a = Math.max(0, Math.min(1, Number(alpha) || 0.34));
  return `rgba(${r}, ${g}, ${b}, ${a})`;
};

export const normalizeDocumentHighlight = (value, index = 0) => {
  if (!value || typeof value !== 'object') return null;
  const text = String(value.text || '').trim();
  const page = Math.max(1, Math.floor(Number(value.page) || 1));
  if (!text) return null;

  const createdAt = Number(value.created_at ?? value.timestamp) || Date.now();
  const rects = (Array.isArray(value.rects) ? value.rects : [])
    .map(normalizeDocumentHighlightRect)
    .filter(Boolean);
  const id = String(
    value.id
    || `legacy-${page}-${createdAt}-${stableTextHash(`${text}:${index}`)}`
  );

  return {
    id,
    text,
    page,
    rects,
    coordinate_space: 'pdf_top_left_points',
    color: normalizeDocumentHighlightColor(value.color),
    style: normalizeDocumentHighlightStyle(value.style),
    created_at: createdAt,
  };
};

export const readDocumentHighlights = (docId, storage = globalThis.localStorage) => {
  if (!docId || !storage) return [];
  try {
    const parsed = JSON.parse(storage.getItem(`highlights_${docId}`) || '[]');
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item, index) => normalizeDocumentHighlight(item, index))
      .filter(Boolean);
  } catch {
    return [];
  }
};

export const writeDocumentHighlights = (docId, highlights, storage = globalThis.localStorage) => {
  if (!docId || !storage) return false;
  try {
    const normalized = (Array.isArray(highlights) ? highlights : [])
      .map((item, index) => normalizeDocumentHighlight(item, index))
      .filter(Boolean);
    storage.setItem(`highlights_${docId}`, JSON.stringify(normalized));
    return true;
  } catch {
    return false;
  }
};

export const createDocumentHighlight = ({
  text,
  page,
  rects = [],
  id,
  color = DEFAULT_DOCUMENT_HIGHLIGHT_COLOR,
  style = DOCUMENT_HIGHLIGHT_STYLES.HIGHLIGHT,
  now = Date.now(),
}) => (
  normalizeDocumentHighlight({
    id: id || `highlight-${now}-${Math.random().toString(36).slice(2, 8)}`,
    text,
    page,
    rects,
    color: normalizeDocumentHighlightColor(color),
    style: normalizeDocumentHighlightStyle(style),
    created_at: now,
  })
);

export const getDocumentHighlightFingerprint = (highlight) => {
  const normalized = normalizeDocumentHighlight(highlight);
  if (!normalized) return '';
  const geometry = normalized.rects
    .map((rect) => [rect.left, rect.top, rect.width, rect.height].map((value) => Math.round(value * 10) / 10).join(','))
    .join(';');
  return `${normalized.page}:${normalized.text}:${geometry}`;
};

export const normalizeDocumentNote = (value, index = 0) => {
  if (!value || typeof value !== 'object') return null;
  const text = String(value.text || '').trim();
  const note = String(value.note || '').trim();
  const page = Math.max(1, Math.floor(Number(value.page) || 1));
  if (!note) return null;

  const createdAt = Number(value.created_at ?? value.timestamp) || Date.now();
  const updatedAt = Number(value.updated_at) || createdAt;
  const rects = (Array.isArray(value.rects) ? value.rects : [])
    .map(normalizeDocumentHighlightRect)
    .filter(Boolean);
  const hasSelectionAnchor = Boolean(text || rects.length > 0);
  const anchorType = value.anchor_type === DOCUMENT_NOTE_ANCHOR_TYPES.PAGE
    ? DOCUMENT_NOTE_ANCHOR_TYPES.PAGE
    : hasSelectionAnchor
      ? DOCUMENT_NOTE_ANCHOR_TYPES.SELECTION
      : DOCUMENT_NOTE_ANCHOR_TYPES.PAGE;
  return {
    id: String(value.id || `note-${page}-${createdAt}-${stableTextHash(`${text}:${note}:${index}`)}`),
    text,
    note,
    page,
    rects,
    anchor_type: anchorType,
    coordinate_space: 'pdf_top_left_points',
    created_at: createdAt,
    updated_at: Math.max(createdAt, updatedAt),
  };
};

export const readDocumentNotes = (docId, storage = globalThis.localStorage) => {
  if (!docId || !storage) return [];
  try {
    const parsed = JSON.parse(storage.getItem(`notes_${docId}`) || '[]');
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item, index) => normalizeDocumentNote(item, index))
      .filter(Boolean);
  } catch {
    return [];
  }
};

export const writeDocumentNotes = (docId, notes, storage = globalThis.localStorage) => {
  if (!docId || !storage) return false;
  try {
    const normalized = (Array.isArray(notes) ? notes : [])
      .map((item, index) => normalizeDocumentNote(item, index))
      .filter(Boolean);
    storage.setItem(`notes_${docId}`, JSON.stringify(normalized));
    return true;
  } catch {
    return false;
  }
};

export const createDocumentNote = ({
  text = '',
  note,
  page,
  rects = [],
  anchorType,
  id,
  now = Date.now(),
}) => (
  normalizeDocumentNote({
    id: id || `note-${now}-${Math.random().toString(36).slice(2, 8)}`,
    text,
    note,
    page,
    rects,
    anchor_type: anchorType,
    created_at: now,
    updated_at: now,
  })
);
