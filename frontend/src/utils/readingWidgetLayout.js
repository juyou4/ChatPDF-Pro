export const READING_WIDGET_LAYOUT_STORAGE_KEY = 'chatpdf_reading_widget_layout_v1';

// AI 要点已收进面板头部的「本页对应」面包屑：它只有 0-2 条、且和左侧「总结」共享
// 同一份大纲和同一个跳转函数，不值得再占一个和翻译、笔记平级的卡片位。
export const READING_WIDGET_IDS = Object.freeze([
  'translation',
  'user_notes',
]);

// 双栏布局下的槽位：主列、副列、通栏底部。
// 槽位不由 order 下标硬推——那会把第二个组件永远锁死在右列。
// 现在由用户显式的 fullWidth 决定：标了通栏的进底部，其余的按顺序填主列、副列。
export const READING_WIDGET_SLOTS = Object.freeze(['main', 'side', 'foot']);

const MAX_COLUMN_WIDGETS = 2;

export function resolveReadingWidgetSlots(order, fullWidth) {
  const list = Array.isArray(order) ? order : [];
  const flags = fullWidth && typeof fullWidth === 'object' ? fullWidth : {};
  const slots = {};
  // 通栏卡片必须拿到显式行号。只写 grid-column 让浏览器自动放置的话，
  // 列区那一行为空时（所有卡片都通栏）第一张会被塞进那个 0 高度的行，压在下一张上面。
  const rows = {};
  let columnCount = 0;
  let footRow = 2;

  list.forEach((id) => {
    if (flags[id] || columnCount >= MAX_COLUMN_WIDGETS) {
      slots[id] = 'foot';
      rows[id] = footRow;
      footRow += 1;
      return;
    }
    slots[id] = columnCount === 0 ? 'main' : 'side';
    rows[id] = 1;
    columnCount += 1;
  });

  return { slots, rows, columnCount };
}

// 0 表示「自动高度」，未被用户手动拖过的卡片都是 0。
export const READING_WIDGET_MIN_HEIGHT = 140;
export const READING_WIDGET_MAX_HEIGHT = 1200;

// 双栏模式下主列所占宽度百分比；61 ≈ 原来的 1.55fr : 1fr。
export const READING_WIDGET_DEFAULT_SPLIT = 61;
export const READING_WIDGET_MIN_SPLIT = 40;
export const READING_WIDGET_MAX_SPLIT = 72;

export function clampReadingWidgetSplit(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return READING_WIDGET_DEFAULT_SPLIT;
  return Math.max(
    READING_WIDGET_MIN_SPLIT,
    Math.min(READING_WIDGET_MAX_SPLIT, Math.round(numeric))
  );
}

export const DEFAULT_READING_WIDGET_LAYOUT = Object.freeze({
  version: 2,
  order: Object.freeze([...READING_WIDGET_IDS]),
  collapsed: Object.freeze({
    translation: false,
    user_notes: false,
  }),
  sizes: Object.freeze({
    translation: 0,
    user_notes: 0,
  }),
  // 翻译和笔记左右对照；任意一张都可以拖成底部通栏
  fullWidth: Object.freeze({
    translation: false,
    user_notes: false,
  }),
  split: READING_WIDGET_DEFAULT_SPLIT,
});

export function clampReadingWidgetHeight(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return 0;
  return Math.max(
    READING_WIDGET_MIN_HEIGHT,
    Math.min(READING_WIDGET_MAX_HEIGHT, Math.round(numeric))
  );
}

const VALID_WIDGET_IDS = new Set(READING_WIDGET_IDS);

export function normalizeReadingWidgetLayout(value) {
  const source = value && typeof value === 'object' ? value : {};
  const sourceOrder = Array.isArray(source.order) ? source.order : [];
  const seen = new Set();
  const order = [];

  sourceOrder.forEach((id) => {
    if (!VALID_WIDGET_IDS.has(id) || seen.has(id)) return;
    seen.add(id);
    order.push(id);
  });
  READING_WIDGET_IDS.forEach((id) => {
    if (!seen.has(id)) order.push(id);
  });

  const sourceCollapsed = source.collapsed && typeof source.collapsed === 'object'
    ? source.collapsed
    : {};
  const collapsed = Object.fromEntries(
    READING_WIDGET_IDS.map((id) => [
      id,
      typeof sourceCollapsed[id] === 'boolean'
        ? sourceCollapsed[id]
        : DEFAULT_READING_WIDGET_LAYOUT.collapsed[id],
    ])
  );

  const sourceSizes = source.sizes && typeof source.sizes === 'object' ? source.sizes : {};
  const sizes = Object.fromEntries(
    READING_WIDGET_IDS.map((id) => [id, clampReadingWidgetHeight(sourceSizes[id])])
  );

  const sourceFullWidth = source.fullWidth && typeof source.fullWidth === 'object'
    ? source.fullWidth
    : {};
  const fullWidth = Object.fromEntries(
    READING_WIDGET_IDS.map((id) => [
      id,
      typeof sourceFullWidth[id] === 'boolean'
        ? sourceFullWidth[id]
        : DEFAULT_READING_WIDGET_LAYOUT.fullWidth[id],
    ])
  );

  return {
    version: 2,
    order,
    collapsed,
    sizes,
    fullWidth,
    split: clampReadingWidgetSplit(source.split),
  };
}

export function isDefaultReadingWidgetLayout(value) {
  const normalized = normalizeReadingWidgetLayout(value);
  return normalized.split === READING_WIDGET_DEFAULT_SPLIT
    && READING_WIDGET_IDS.every((id, index) => (
      normalized.order[index] === id
      && normalized.collapsed[id] === DEFAULT_READING_WIDGET_LAYOUT.collapsed[id]
      && normalized.sizes[id] === 0
      && normalized.fullWidth[id] === DEFAULT_READING_WIDGET_LAYOUT.fullWidth[id]
    ));
}
