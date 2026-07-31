export const CHAT_VISUAL_ATTACHMENT_VERSION = 'chat_visual_attachment.v1';
export const MAX_CHAT_VISUAL_ATTACHMENTS = 2;

const ATTACHMENT_ID_RE = /^va_[0-9a-f]{32}$/;
const SOURCE_HASH_RE = /^[0-9a-f]{64}$/;
const ALLOWED_KINDS = new Set(['figure', 'table']);
const ALLOWED_EVIDENCE_MODES = new Set(['parser_visual', 'vlm_verified']);

const compactText = (value, limit) => String(value || '').replace(/\s+/g, ' ').trim().slice(0, limit);

const normalizeBBox = (value) => {
  if (!Array.isArray(value) || value.length < 4) return null;
  const bbox = value.slice(0, 4).map(Number);
  if (!bbox.every(Number.isFinite) || bbox[2] <= bbox[0] || bbox[3] <= bbox[1]) return null;
  return bbox;
};

const normalizePositiveInt = (value, max = 1_000_000) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 && parsed <= max ? parsed : 0;
};

const safePathSegment = (value, limit) => {
  const text = compactText(value, limit);
  return text && !/[\\/\0]/.test(text) ? text : '';
};

export const normalizeChatVisualAttachment = (value, expectedIdentity = null) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const version = String(value.version || '');
  const attachmentId = String(value.attachment_id || value.attachmentId || '').trim().toLowerCase();
  const parseGeneration = safePathSegment(
    value.parse_generation || value.parseGeneration,
    160,
  );
  const sourceHash = String(
    value.document_source_hash || value.documentSourceHash || '',
  ).trim().toLowerCase();
  const kind = String(value.kind || '').trim().toLowerCase();
  const page = normalizePositiveInt(value.page);
  const bbox = normalizeBBox(value.bbox);
  const evidenceMode = ALLOWED_EVIDENCE_MODES.has(String(value.evidence_mode || ''))
    ? String(value.evidence_mode)
    : 'parser_visual';
  if (
    version !== CHAT_VISUAL_ATTACHMENT_VERSION
    || !ATTACHMENT_ID_RE.test(attachmentId)
    || !parseGeneration
    || !SOURCE_HASH_RE.test(sourceHash)
    || !ALLOWED_KINDS.has(kind)
    || !page
    || !bbox
  ) {
    return null;
  }
  const expectedGeneration = String(
    expectedIdentity?.parseGeneration || expectedIdentity?.generation || '',
  ).trim();
  const expectedSourceHash = String(
    expectedIdentity?.documentSourceHash || expectedIdentity?.sourceHash || '',
  ).trim().toLowerCase();
  if (
    (expectedGeneration && parseGeneration !== expectedGeneration)
    || (expectedSourceHash && sourceHash !== expectedSourceHash)
  ) {
    return null;
  }
  const citationRefs = Array.isArray(value.citation_refs)
    ? [...new Set(value.citation_refs.map((item) => normalizePositiveInt(item, 999)).filter(Boolean))].slice(0, 8)
    : [];
  return {
    version,
    attachment_id: attachmentId,
    asset_id: compactText(value.asset_id || value.assetId, 240),
    kind,
    label: compactText(value.label, 120) || (kind === 'table' ? '表格' : '图'),
    caption: compactText(value.caption, 360),
    figure_id: compactText(value.figure_id || value.figureId, 120),
    page,
    bbox,
    coordinate_space: 'pdf_top_left_points',
    route: compactText(value.route, 32),
    parse_generation: parseGeneration,
    document_source_hash: sourceHash,
    visual_supplement_revision: compactText(
      value.visual_supplement_revision || value.visualSupplementRevision,
      160,
    ),
    citation_refs: citationRefs,
    evidence_mode: evidenceMode,
    width: normalizePositiveInt(value.width, 20_000),
    height: normalizePositiveInt(value.height, 20_000),
    mime_type: 'image/jpeg',
  };
};

export const normalizeChatVisualAttachments = (value, expectedIdentity = null) => {
  if (!Array.isArray(value)) return [];
  const normalized = [];
  const seen = new Set();
  for (const item of value) {
    const attachment = normalizeChatVisualAttachment(item, expectedIdentity);
    if (!attachment || seen.has(attachment.attachment_id)) continue;
    seen.add(attachment.attachment_id);
    normalized.push(attachment);
    if (normalized.length >= MAX_CHAT_VISUAL_ATTACHMENTS) break;
  }
  return normalized;
};

export const buildChatVisualAttachmentUrl = (docId, attachment) => {
  const safeDocId = compactText(docId, 240);
  const normalized = normalizeChatVisualAttachment(attachment);
  if (!safeDocId || !normalized) return '';
  return `/documents/${encodeURIComponent(safeDocId)}/visual-assets/${encodeURIComponent(normalized.parse_generation)}/${encodeURIComponent(normalized.attachment_id)}`;
};
