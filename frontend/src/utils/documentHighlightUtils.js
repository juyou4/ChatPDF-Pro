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
    color: String(value.color || '#F2C15C'),
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

export const createDocumentHighlight = ({ text, page, rects = [], id, now = Date.now() }) => (
  normalizeDocumentHighlight({
    id: id || `highlight-${now}-${Math.random().toString(36).slice(2, 8)}`,
    text,
    page,
    rects,
    color: '#F2C15C',
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
  if (!text || !note) return null;

  const createdAt = Number(value.created_at ?? value.timestamp) || Date.now();
  const rects = (Array.isArray(value.rects) ? value.rects : [])
    .map(normalizeDocumentHighlightRect)
    .filter(Boolean);
  return {
    id: String(value.id || `note-${page}-${createdAt}-${stableTextHash(`${text}:${note}:${index}`)}`),
    text,
    note,
    page,
    rects,
    coordinate_space: 'pdf_top_left_points',
    created_at: createdAt,
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

export const createDocumentNote = ({ text, note, page, rects = [], id, now = Date.now() }) => (
  normalizeDocumentNote({
    id: id || `note-${now}-${Math.random().toString(36).slice(2, 8)}`,
    text,
    note,
    page,
    rects,
    created_at: now,
  })
);
