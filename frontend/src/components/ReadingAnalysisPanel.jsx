import React, { lazy, memo, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronDown,
  FileText,
  GripVertical,
  Languages,
  Loader2,
  MapPin,
  PanelRight,
  Pencil,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  StickyNote,
  Trash2,
  X,
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { useDebouncedLocalStorage } from '../hooks/useDebouncedLocalStorage';
import NoteMarkdown from './NoteMarkdown';
import {
  clampReadingWidgetHeight,
  clampReadingWidgetSplit,
  DEFAULT_READING_WIDGET_LAYOUT,
  isDefaultReadingWidgetLayout,
  normalizeReadingWidgetLayout,
  READING_WIDGET_LAYOUT_STORAGE_KEY,
  resolveReadingWidgetSlots,
} from '../utils/readingWidgetLayout';

// CodeMirror 只在打开笔记编辑器时才需要，懒加载避免拖慢首屏。
const MarkdownNoteEditor = lazy(() => import('./MarkdownNoteEditor'));

const TYPE_LABEL = {
  heading: '标题',
  paragraph: '段落',
  caption: '图注',
  figure: '图表',
  table: '表格',
};

const TRANSLATION_VIEW_STORAGE_KEY = 'chatpdf_translation_view_v1';

const TRANSLATION_VIEWS = [
  { id: 'compare', label: '对照', title: '原文与译文左右对齐，宽度不够时自动上下排' },
  { id: 'target', label: '译文', title: '只看译文，未翻译的块折成一行' },
];

function TranslationViewToggle({ value, onChange, darkMode }) {
  return (
    <div
      role="group"
      aria-label="翻译显示方式"
      className={`inline-flex shrink-0 items-center rounded-full p-0.5 ${darkMode ? 'bg-white/[0.07]' : 'bg-[#f1ece7]'}`}
    >
      {TRANSLATION_VIEWS.map((view) => {
        const active = value === view.id;
        return (
          <button
            key={view.id}
            type="button"
            onClick={() => onChange?.(view.id)}
            aria-pressed={active}
            title={view.title}
            className={`rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/25 ${
              active
                ? (darkMode
                  ? 'bg-white/[0.14] text-gray-100'
                  : 'bg-white text-[#3a332e] shadow-[0_1px_2px_rgba(83,65,55,0.09)]')
                : (darkMode ? 'text-gray-500 hover:text-gray-300' : 'text-[#82766a] hover:text-[#5c534b]')
            }`}
          >
            {view.label}
          </button>
        );
      })}
    </div>
  );
}

function EmptyWidgetState({ icon: Icon, label, hint, action, darkMode }) {
  return (
    <div className={`flex min-h-28 flex-col items-center justify-center gap-2 px-5 py-9 text-center ${darkMode ? 'text-gray-500' : 'text-[#82766a]'}`}>
      <Icon className="h-6 w-6 opacity-55" strokeWidth={1.8} />
      <div className={`text-[12px] font-medium ${darkMode ? 'text-gray-400' : 'text-[#776b60]'}`}>{label}</div>
      {hint && (
        <p className="max-w-[26ch] text-balance text-[11px] leading-[1.7]">{hint}</p>
      )}
      {action}
    </div>
  );
}

// 拖拽换位：三个槽位下「交换」比「插入并顺移」更好预测，
// 命中判定用指针坐标而不是列表主轴，双栏 grid 才不会算错落点。
const IDLE_DRAG = { id: '', dx: 0, dy: 0 };
const IDLE_RESIZE = { id: '', height: 0 };
const WIDGET_SPRING = { type: 'spring', stiffness: 420, damping: 34, mass: 0.8 };
// 与 index.css 里 @container (min-width: 42rem) 保持一致
const TWO_COLUMN_MIN_WIDTH = 672;

function useWidgetSlotDrag({ boardRef, canDrag, onSwap, onDropSlot }) {
  const [drag, setDrag] = useState(IDLE_DRAG);
  const originRef = useRef({ x: 0, y: 0 });
  // 刚落过的目标：指针还停在它上面时不再重复触发，避免在边界来回抖。
  const lastPartnerRef = useRef('');
  // 本次拖拽是否观测到过「按键按下」，用于判断 buttons=0 是不是真的松手
  const sawButtonsRef = useRef(false);

  const startDrag = useCallback((id) => (event) => {
    if (!canDrag || event.button > 0) return;
    event.preventDefault();
    // 捕获指针：否则在窗口外松手时 pointerup 不会送到页面，
    // drag 状态清不掉，那张卡会一直保持放大+偏移，叠在别人身上不动。
    try {
      event.currentTarget?.setPointerCapture?.(event.pointerId);
    } catch {
      // 某些环境（jsdom / 旧浏览器）没有指针捕获，下面的兜底仍然管用
    }
    originRef.current = { x: event.clientX || 0, y: event.clientY || 0 };
    lastPartnerRef.current = '';
    sawButtonsRef.current = false;
    setDrag({ ...IDLE_DRAG, id });
  }, [canDrag]);

  useEffect(() => {
    if (!drag.id) return undefined;

    // 落点既可能是另一张卡片（换位），也可能是空槽位（改成通栏 / 挪回列里）
    const resolveTarget = (clientX, clientY) => {
      const nodes = boardRef.current?.querySelectorAll('[data-reading-widget-id],[data-drop-slot]');
      if (!nodes) return null;
      const hit = Array.from(nodes).find((node) => {
        const rect = node.getBoundingClientRect();
        return clientX >= rect.left && clientX <= rect.right
          && clientY >= rect.top && clientY <= rect.bottom;
      });
      if (!hit) return null;
      const slot = hit.getAttribute('data-drop-slot');
      if (slot) return { key: `slot:${slot}`, slot };
      const id = hit.getAttribute('data-reading-widget-id') || '';
      return id ? { key: `widget:${id}`, id } : null;
    };

    const trackOffset = (event) => {
      setDrag((current) => ({
        ...current,
        dx: event.clientX - originRef.current.x,
        dy: event.clientY - originRef.current.y,
      }));
    };

    const handleMove = (event) => {
      // 兜底：先确认这个环境真的会上报 buttons，再把 buttons=0 当成「漏掉了 pointerup」
      // （窗口外松手、被系统手势打断等）。不这样分两步的话，
      // 万一某个环境始终上报 0，拖拽会一动就结束——比卡住更糟。
      if (event.buttons > 0) {
        sawButtonsRef.current = true;
      } else if (sawButtonsRef.current) {
        handleEnd();
        return;
      }
      const target = resolveTarget(event.clientX, event.clientY);

      if (!target || target.id === drag.id) {
        lastPartnerRef.current = '';
        trackOffset(event);
        return;
      }
      if (target.key === lastPartnerRef.current) {
        trackOffset(event);
        return;
      }

      // 当场生效：被挤开的卡片靠 layout 动画滑到新位置；
      // 同时把原点挪到当前指针，手里这张卡不会因为换槽而突然跳走。
      if (target.slot) {
        onDropSlot?.(drag.id, target.slot);
      } else {
        onSwap?.(drag.id, target.id);
      }
      lastPartnerRef.current = target.key;
      originRef.current = { x: event.clientX, y: event.clientY };
      setDrag((current) => ({ ...current, dx: 0, dy: 0 }));
    };

    const handleEnd = () => {
      lastPartnerRef.current = '';
      setDrag(IDLE_DRAG);
    };
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') handleEnd();
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleEnd);
    window.addEventListener('pointercancel', handleEnd);
    // 指针捕获被系统收走、或窗口失焦（切标签页、Alt+Tab）时同样要收尾
    window.addEventListener('lostpointercapture', handleEnd);
    window.addEventListener('blur', handleEnd);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleEnd);
      window.removeEventListener('lostpointercapture', handleEnd);
      window.removeEventListener('blur', handleEnd);
      window.removeEventListener('pointercancel', handleEnd);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [boardRef, drag.id, onDropSlot, onSwap]);

  return { drag, startDrag };
}

function ReadingWidgetCard({
  id,
  title,
  meta,
  icon: Icon,
  iconClassName,
  slot,
  gridRow = 1,
  collapsed,
  onToggle,
  onMove,
  onToggleFullWidth,
  fullWidth = false,
  onDragStart,
  dragging,
  dragOffset = IDLE_DRAG,
  height = 0,
  resizable = false,
  onResizeStart,
  total,
  headerAction,
  bodyClassName = '',
  darkMode,
  children,
}) {
  const bodyId = `reading-widget-body-${id}`;

  const handleDragKeyDown = (event) => {
    // 上下换位，左右在「入列」和「底部通栏」之间切——键盘也能做到拖拽能做的事
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
      onMove?.(id, event.key === 'ArrowUp' ? -1 : 1);
      return;
    }
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      onToggleFullWidth?.(id);
    }
  };

  return (
    // 外层只负责占格子和换位动画，内层只负责跟手位移；
    // 两件事放在同一个元素上会抢同一个 transform，卡片就会停在错位的地方。
    <motion.section
      layout="position"
      transition={WIDGET_SPRING}
      data-testid={`reading-widget-${id}`}
      data-reading-widget-id={id}
      data-widget-slot={slot}
      data-collapsed={collapsed ? 'true' : 'false'}
      data-dragging={dragging ? 'true' : undefined}
      data-sized={height > 0 ? 'true' : undefined}
      className="reading-widget-shell relative flex min-w-0 flex-col"
      style={{
        zIndex: dragging ? 30 : undefined,
        height: height > 0 ? `${height}px` : undefined,
        // 单栏下这个变量没人用；双栏的容器查询里才会读它来定行号
        '--widget-row': gridRow,
      }}
    >
      <motion.div
        animate={{
          scale: dragging ? 1.025 : 1,
          x: dragging ? dragOffset.dx : 0,
          y: dragging ? dragOffset.dy : 0,
        }}
        transition={dragging ? { duration: 0 } : WIDGET_SPRING}
        className={`reading-widget-card relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-[24px] border transition-[border-color,background-color,box-shadow] duration-200 ${
          darkMode
            ? 'border-white/[0.07] bg-white/[0.085] shadow-[0_1px_2px_rgba(0,0,0,0.22),0_6px_14px_-8px_rgba(0,0,0,0.5),0_18px_38px_-22px_rgba(0,0,0,0.72)]'
            : 'border-white bg-white shadow-[0_1px_2px_rgba(83,65,55,0.05),0_6px_14px_-8px_rgba(83,65,55,0.10),0_18px_38px_-22px_rgba(83,65,55,0.26)]'
        } ${
          dragging
            ? (darkMode
              ? 'border-[#F0653A]/40 shadow-[0_36px_70px_-26px_rgba(0,0,0,0.9)]'
              : 'border-[#f0d2c4] shadow-[0_36px_66px_-24px_rgba(83,65,55,0.38)]')
            : ''
        }`}
      >
      <div className={`flex min-h-12 shrink-0 items-center gap-2 border-b px-3.5 py-3 ${
        darkMode ? 'border-white/[0.06]' : 'border-[#f6f2ed]'
      }`}>
        <button
          type="button"
          onPointerDown={onDragStart}
          onKeyDown={handleDragKeyDown}
          disabled={total < 2}
          className={`reading-widget-drag-handle inline-flex h-8 w-7 shrink-0 touch-none select-none items-center justify-center rounded-[10px] transition-colors active:cursor-grabbing disabled:cursor-default disabled:opacity-35 ${
            darkMode
              ? 'cursor-grab text-gray-500 hover:bg-white/[0.07] hover:text-gray-300 focus-visible:ring-white/20'
              : 'cursor-grab text-[#a99d91] hover:bg-[#f3efeb] hover:text-[#776b60] focus-visible:ring-[#D97A5D]/25'
          } focus-visible:outline-none focus-visible:ring-2`}
          aria-label={`拖动${title}组件`}
          title={`拖到别的组件上交换位置，拖到虚线区域可${fullWidth ? '放回列里' : '改成底部通栏'}；方向键上下换位、左右切换通栏`}
        >
          <GripVertical className="h-4 w-4" strokeWidth={2} />
        </button>

        <div className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[11px] ${
          darkMode ? 'bg-white/[0.07]' : 'bg-[#f7f3ef]'
        } ${iconClassName}`}>
          <Icon className="h-4 w-4" strokeWidth={2} />
        </div>

        <div className="flex min-w-0 flex-1 items-baseline gap-2">
          <h3 className={`shrink-0 text-[12px] font-semibold ${darkMode ? 'text-gray-100' : 'text-[#3a332e]'}`}>{title}</h3>
          <span className={`truncate text-[10px] font-medium tabular-nums ${darkMode ? 'text-gray-500' : 'text-[#82766a]'}`}>{meta}</span>
        </div>

        {headerAction}
        <button
          type="button"
          onClick={() => onToggle?.(id)}
          className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[11px] transition-colors focus-visible:outline-none focus-visible:ring-2 ${
            darkMode
              ? 'text-gray-400 hover:bg-white/[0.07] hover:text-gray-200 focus-visible:ring-white/20'
              : 'text-[#82766a] hover:bg-[#f3efeb] hover:text-[#5c534b] focus-visible:ring-[#D97A5D]/25'
          }`}
          aria-label={collapsed ? `展开${title}` : `收起${title}`}
          aria-expanded={!collapsed}
          aria-controls={bodyId}
          title={collapsed ? '展开' : '收起'}
        >
          <ChevronDown className={`h-4 w-4 transition-transform duration-200 ${collapsed ? '' : 'rotate-180'}`} />
        </button>
      </div>

      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            id={bodyId}
            key="body"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.16, ease: 'easeOut' }}
            className={`reading-widget-body ${bodyClassName}`}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
      </motion.div>

      {resizable && !collapsed && (
        <div
          role="separator"
          aria-orientation="horizontal"
          aria-label={`调整${title}高度`}
          tabIndex={0}
          onPointerDown={onResizeStart}
          onKeyDown={(event) => {
            if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
            event.preventDefault();
            onResizeStart?.(event, event.key === 'ArrowUp' ? -24 : 24);
          }}
          className="reading-widget-resize group/resize absolute inset-x-6 -bottom-1 z-20 flex h-3 cursor-ns-resize touch-none items-center justify-center focus-visible:outline-none"
          title="拖动调整高度，方向键也可以"
        >
          <span
            aria-hidden="true"
            className={`h-[3px] w-12 rounded-full transition-colors duration-200 ${
              darkMode
                ? 'bg-white/0 group-hover/resize:bg-white/25 group-focus-visible/resize:bg-white/35'
                : 'bg-[#cec4b9]/0 group-hover/resize:bg-[#cec4b9] group-focus-visible/resize:bg-[#c9a08c]'
            }`}
          />
        </div>
      )}
    </motion.section>
  );
}

// 吸附栏开着时翻译卡整个退出 widget 板，把主列还给笔记和要点。
// 状态压成头部下面的一条窄条：说明去向 + 预翻译进度（本来就是服务悬浮翻译的）+ 切回入口。
function TranslationDockedStrip({
  pretranslateProgress,
  onPretranslate,
  onReturnToPanel,
  error,
  notice,
  darkMode,
}) {
  const total = Number(pretranslateProgress?.total || 0);
  const done = Math.min(total, Number(pretranslateProgress?.done || 0));
  const percent = total > 0 ? Math.round((done / total) * 100) : 0;

  return (
    <div className={`shrink-0 border-b px-5 py-2 ${darkMode ? 'border-white/[0.06] bg-white/[0.02]' : 'border-[#f3eee9] bg-[#fbf9f7]'}`}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className={`inline-flex shrink-0 items-center gap-1.5 text-[11px] font-medium ${darkMode ? 'text-gray-300' : 'text-[#5c534b]'}`}>
          <PanelRight className={`h-3.5 w-3.5 ${darkMode ? 'text-[#fdc4af]' : 'text-[#c96b50]'}`} strokeWidth={1.9} />
          译文在 PDF 吸附栏
        </span>

        {total > 0 && (
          <>
            <span className={`shrink-0 text-[10px] font-medium tabular-nums ${darkMode ? 'text-gray-500' : 'text-[#82766a]'}`}>
              悬浮预翻译 {done}/{total}
            </span>
            <div
              className={`h-[3px] min-w-[3rem] flex-1 overflow-hidden rounded-full ${darkMode ? 'bg-white/10' : 'bg-[#efe9e3]'}`}
              role="progressbar"
              aria-label="悬浮预翻译缓存进度"
              aria-valuenow={percent}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className="h-full rounded-full bg-[#e0855f] transition-[width] duration-300"
                style={{ width: `${percent}%` }}
              />
            </div>
            <button
              type="button"
              onClick={() => onPretranslate?.()}
              disabled={pretranslateProgress?.running}
              className={`shrink-0 rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 ${
                darkMode
                  ? 'text-gray-400 hover:bg-white/[0.08] hover:text-gray-100 focus-visible:ring-white/20'
                  : 'text-[#776b60] hover:bg-[#f1ece7] hover:text-[#c96b50] focus-visible:ring-[#D97A5D]/25'
              }`}
            >
              {pretranslateProgress?.running ? '处理中' : done > 0 ? '补齐全文' : '开始缓存'}
            </button>
          </>
        )}

        <button
          type="button"
          onClick={onReturnToPanel}
          className={`ml-auto shrink-0 rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors duration-200 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 ${
            darkMode
              ? 'text-gray-300 hover:bg-white/[0.08] hover:text-gray-100 focus-visible:ring-white/20'
              : 'text-[#776b60] hover:bg-[#f1ece7] hover:text-[#c96b50] focus-visible:ring-[#D97A5D]/25'
          }`}
        >
          切回整页对照
        </button>
      </div>

      {/* 整页翻译和预翻译在吸附态下照样能触发，报错必须有落点，
          否则失败信息会跟着翻译卡一起被过滤掉，用户只看到什么都没发生。 */}
      {error && (
        <div
          role="alert"
          className={`mt-1.5 rounded-[9px] px-2.5 py-1.5 text-[11px] font-medium ${
            darkMode ? 'bg-red-500/10 text-red-300' : 'bg-[#fdf1ed] text-[#b4472b]'
          }`}
        >
          {error}
        </div>
      )}
      {notice && !error && (
        <div className={`mt-1.5 text-[10px] font-medium ${darkMode ? 'text-[#fdc4af]' : 'text-[#c96b50]'}`}>
          {notice}
        </div>
      )}
    </div>
  );
}

function TranslationWidgetContent({
  blocks,
  translations,
  translatingSet,
  activeBlockId,
  viewMode = 'compare',
  error,
  notice,
  pretranslateProgress,
  onPretranslate,
  onBackfillSummaries,
  backfillingSummaries,
  onRetranslateBlock,
  onBlockHover,
  onBlockClick,
  darkMode,
}) {
  const pretranslateTotal = Number(pretranslateProgress?.total || 0);
  const pretranslateDone = Math.min(pretranslateTotal, Number(pretranslateProgress?.done || 0));
  const pretranslatePercent = pretranslateTotal > 0
    ? Math.round((pretranslateDone / pretranslateTotal) * 100)
    : 0;

  return (
    <div className="reading-widget-flex min-h-0">
      {(error || pretranslateTotal > 0) && (
        <div className={`shrink-0 border-b px-4 py-2.5 ${
          darkMode
            ? 'border-white/[0.07] bg-white/[0.025]'
            : 'border-[#eee9e4] bg-[#fffdfb]'
        }`}>
          {error && (
            <div className={`mb-2 rounded-[9px] px-3 py-2 text-[11px] font-medium ${darkMode ? 'bg-red-500/10 text-red-300' : 'bg-[#fdf1ed] text-[#b4472b]'}`}>
              {error}
            </div>
          )}
          {pretranslateTotal > 0 && (
            <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
              <span className={`shrink-0 text-[10px] font-medium tabular-nums ${darkMode ? 'text-gray-500' : 'text-[#82766a]'}`}>
                悬浮预翻译 {pretranslateDone}/{pretranslateTotal}
              </span>
              <div
                className={`h-[3px] min-w-0 flex-1 overflow-hidden rounded-full ${darkMode ? 'bg-white/10' : 'bg-[#efe9e3]'}`}
                role="progressbar"
                aria-label="悬浮预翻译缓存进度"
                aria-valuenow={pretranslatePercent}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="h-full rounded-full bg-[#e0855f] transition-[width] duration-300"
                  style={{ width: `${pretranslatePercent}%` }}
                />
              </div>
              <button
                type="button"
                onClick={() => onPretranslate?.()}
                disabled={pretranslateProgress.running}
                className={`shrink-0 rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors duration-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50 disabled:active:scale-100 focus-visible:outline-none focus-visible:ring-2 ${
                  darkMode
                    ? 'text-gray-400 hover:bg-white/[0.08] hover:text-gray-100 focus-visible:ring-white/20'
                    : 'text-[#776b60] hover:bg-[#f1ece7] hover:text-[#c96b50] focus-visible:ring-[#D97A5D]/25'
                }`}
              >
                {pretranslateProgress.running ? '处理中' : pretranslateDone > 0 ? '补齐全文' : '开始缓存'}
              </button>
              {/* 只补要点、不重跑翻译：开关打开之前翻好的块缓存里 summary 是空的 */}
              {onBackfillSummaries && (
                <button
                  type="button"
                  onClick={() => onBackfillSummaries?.()}
                  disabled={backfillingSummaries}
                  title="给已翻译但没有要点的段落补上要点，不会重新翻译"
                  className={`shrink-0 rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors duration-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50 disabled:active:scale-100 focus-visible:outline-none focus-visible:ring-2 ${
                    darkMode
                      ? 'text-gray-400 hover:bg-white/[0.08] hover:text-gray-100 focus-visible:ring-white/20'
                      : 'text-[#776b60] hover:bg-[#f1ece7] hover:text-[#c96b50] focus-visible:ring-[#D97A5D]/25'
                  }`}
                >
                  {backfillingSummaries ? '补要点中' : '补齐要点'}
                </button>
              )}
            </div>
          )}
          {notice && !error && (
            <div className={`mt-1 text-[10px] font-medium ${darkMode ? 'text-[#fdc4af]' : 'text-[#c96b50]'}`}>
              {notice}
            </div>
          )}
        </div>
      )}

      <div
        data-testid="translation-block-list"
        data-view={viewMode}
        className="translation-block-list reading-widget-scroll custom-scrollbar max-h-[min(62vh,720px)] overflow-y-auto overscroll-contain"
      >
        {blocks.length === 0 ? (
          <EmptyWidgetState icon={FileText} label="本页暂无可翻译文本" darkMode={darkMode} />
        ) : (
          <div className={`divide-y ${darkMode ? 'divide-white/[0.05]' : 'divide-[#f4efea]'}`}>
            {blocks.map((block) => {
              const item = translations[block.block_id];
              const isActive = activeBlockId === block.block_id;
              const isTranslatingBlock = translatingSet.has(block.block_id);
              const isHeading = block.type === 'heading';
              const isCompare = viewMode !== 'target';
              // 「译文」模式下未翻译的块折成一行原文，既保持模式差异明显，
              // 又不会因为整页没翻译就变成一片空白。
              const collapseSource = !isCompare && !item;
              // 段落占绝大多数，给每个段落挂一枚「段落」徽章是纯噪音。
              const typeLabel = block.type && block.type !== 'paragraph' ? TYPE_LABEL[block.type] : '';
              const typeChip = typeLabel ? (
                <span className={`mr-1.5 rounded-[5px] px-1.5 py-[1px] align-[0.08em] text-[10px] font-medium ${
                  darkMode ? 'bg-white/[0.09] text-gray-400' : 'bg-[#f4efea] text-[#82766a]'
                }`}>
                  {typeLabel}
                </span>
              ) : null;
              const statusBarClass = isTranslatingBlock
                ? 'bg-[#e0855f] animate-pulse'
                : item
                  ? 'bg-transparent'
                  : (darkMode ? 'bg-white/[0.10]' : 'bg-[#e3ddd6]');
              return (
                <article
                  key={block.block_id}
                  data-block-id={block.block_id}
                  data-compare={isCompare && item ? 'true' : undefined}
                  onMouseEnter={() => onBlockHover?.(block)}
                  onMouseLeave={() => onBlockHover?.(null)}
                  onClick={() => onBlockClick?.(block)}
                  aria-current={isActive ? 'true' : undefined}
                  className={`translation-block group relative cursor-pointer py-3 pl-5 pr-4 transition-colors duration-200 ${
                    isActive
                      ? (darkMode ? 'bg-amber-300/[0.07]' : 'bg-[#fff8f3]')
                      : (darkMode ? 'hover:bg-white/[0.03]' : 'hover:bg-[#fbf9f7]')
                  }`}
                >
                  <span
                    aria-hidden="true"
                    className={`absolute inset-y-3 left-2 w-[2px] rounded-full transition-colors duration-200 ${statusBarClass}`}
                  />
                  <span className="sr-only">
                    {isTranslatingBlock ? '正在翻译' : item ? '已翻译' : '未翻译'}
                  </span>

                  {/* 重译按钮浮在右上角，不再单独占一行把译文往下推 */}
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      onRetranslateBlock?.(block);
                    }}
                    disabled={isTranslatingBlock}
                    className={`absolute right-2.5 top-2 z-10 inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-1 text-[10px] font-medium transition-[opacity,color,background-color] duration-200 disabled:cursor-not-allowed disabled:opacity-100 ${
                      isTranslatingBlock ? 'opacity-100' : 'opacity-0 group-hover:opacity-100 focus-visible:opacity-100'
                    } ${
                      darkMode
                        ? 'bg-white/[0.10] text-gray-200 hover:bg-white/[0.16]'
                        : 'bg-white text-[#776b60] ring-1 ring-[#ebe6e1] hover:text-[#c96b50] hover:ring-[#efc9bb]'
                    }`}
                  >
                    {isTranslatingBlock ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
                    {isTranslatingBlock ? '翻译中' : item ? '重译' : '翻译本段'}
                  </button>

                  {/* 对照模式够宽时由 CSS 变成左右两列，窄时自动回退成上下 */}
                  <div className="translation-pair">
                    {/* 译文模式下已翻译的块不再重复原文；未翻译的折成一行，避免整段留白 */}
                    {(isCompare || !item) && (
                      <p className={`translation-source break-words text-[12.5px] leading-[1.72] ${
                        collapseSource ? 'line-clamp-1' : ''
                      } ${darkMode ? 'text-gray-500' : 'text-[#6f6459]'}`}>
                        {typeChip}
                        {block.text}
                      </p>
                    )}

                    {(item || isTranslatingBlock) && (
                      <div className={`translation-target ${item && !isCompare ? '' : 'mt-2'}`}>
                        {item?.summary && (
                          <p className={`mb-1.5 text-pretty break-words text-[12px] font-medium leading-[1.7] ${darkMode ? 'text-amber-200' : 'text-[#a86a2e]'}`}>
                            {item.summary}
                          </p>
                        )}
                        {item ? (
                          <p className={`text-pretty break-words ${
                            isHeading
                              ? 'text-[15px] font-semibold leading-[1.6]'
                              : 'text-[14.5px] leading-[1.82]'
                          } ${darkMode ? 'text-gray-100' : 'text-[#332d28]'}`}>
                            {item.translation}
                          </p>
                        ) : (
                          <div className="space-y-2" aria-hidden="true">
                            <div className={`h-3.5 w-full animate-pulse rounded-full ${darkMode ? 'bg-white/[0.07]' : 'bg-[#f1ece7]'}`} />
                            <div className={`h-3.5 w-4/5 animate-pulse rounded-full ${darkMode ? 'bg-white/[0.07]' : 'bg-[#f1ece7]'}`} />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function UserNotesWidgetContent({
  userNotes,
  editor,
  editorSaving,
  editorError,
  onEditorChange,
  onEditorSave,
  onEditorCancel,
  onUserNoteClick,
  onEditUserNote,
  onDeleteUserNote,
  onStartUserNote,
  darkMode,
}) {
  // 编辑器打开且本页还没有笔记时，让书写区吃掉整列高度，而不是在下面留一大块空白。
  const editorFillsColumn = editor.open && userNotes.length === 0;

  return (
    <>
      <AnimatePresence initial={false}>
        {editor.open && (
          <motion.div
            key="note-editor"
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            transition={{ duration: 0.16, ease: 'easeOut' }}
            className={`border-b p-4 ${
              editorFillsColumn ? 'flex min-h-0 flex-1 flex-col' : 'shrink-0'
            } ${darkMode ? 'border-white/[0.06] bg-white/[0.02]' : 'border-[#f3eee9] bg-[#fbf9f7]'}`}
          >
            <div className="mb-2.5 flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-baseline gap-2">
                <span className={`shrink-0 text-[12px] font-semibold ${darkMode ? 'text-gray-100' : 'text-[#3a332e]'}`}>
                  {editor.noteId ? '编辑笔记' : '新建笔记'}
                </span>
                <span className={`truncate text-[10px] tabular-nums ${darkMode ? 'text-gray-500' : 'text-[#82766a]'}`}>
                  第 {editor.page} 页
                </span>
              </div>
              <button
                type="button"
                onClick={onEditorCancel}
                disabled={editorSaving}
                className={`inline-flex h-7 w-7 items-center justify-center rounded-[10px] transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                  darkMode
                    ? 'text-gray-500 hover:bg-white/[0.08] hover:text-gray-200 focus-visible:ring-white/20'
                    : 'text-[#82766a] hover:bg-[#eee9e4] hover:text-[#5c534b] focus-visible:ring-[#D97A5D]/25'
                }`}
                aria-label="关闭笔记编辑器"
                title="关闭"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            <Suspense fallback={
              <div className={`h-[104px] animate-pulse rounded-[14px] ${darkMode ? 'bg-white/[0.06]' : 'bg-[#f1ece7]'}`} />
            }>
              <MarkdownNoteEditor
                value={editor.value}
                onChange={onEditorChange}
                onSubmit={onEditorSave}
                onCancel={onEditorCancel}
                darkMode={darkMode}
                fill={editorFillsColumn}
                autoFocus
              />
            </Suspense>

            {editorError && (
              <div className={`mt-2 text-[11px] font-medium ${darkMode ? 'text-rose-300' : 'text-rose-600'}`} role="alert">
                {editorError}
              </div>
            )}

            <div className="mt-3 flex items-center justify-end gap-1">
              <button
                type="button"
                onClick={onEditorCancel}
                disabled={editorSaving}
                className={`h-8 rounded-full px-3 text-[11px] font-medium transition-colors duration-200 disabled:opacity-50 ${
                  darkMode ? 'text-gray-400 hover:bg-white/[0.07] hover:text-gray-200' : 'text-[#776b60] hover:bg-[#f1ece7] hover:text-[#3a332e]'
                }`}
              >
                取消
              </button>
              <button
                type="button"
                onClick={onEditorSave}
                disabled={editorSaving || !editor.value.trim()}
                className="inline-flex h-8 items-center gap-1.5 rounded-full bg-[#F0653A] px-4 text-[11px] font-semibold text-white shadow-[0_6px_16px_-8px_rgba(240,101,58,0.65)] transition-[background-color,transform] duration-200 hover:bg-[#F5713F] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none disabled:active:scale-100"
              >
                {editorSaving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                {editor.noteId ? '保存修改' : '保存笔记'}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className={editorFillsColumn
        ? 'shrink-0'
        : 'reading-widget-scroll custom-scrollbar min-h-0 flex-1 overflow-y-auto overscroll-contain'}
      >
      {userNotes.length === 0 ? (
        !editor.open && (
          <EmptyWidgetState
            icon={StickyNote}
            label="本页暂无笔记"
            hint="在左侧 PDF 上划词可直接摘录成引用笔记，也可以写一条只属于本页的笔记。"
            action={onStartUserNote && (
              <button
                type="button"
                onClick={onStartUserNote}
                className={`mt-1 inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[11px] font-medium transition-colors duration-200 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 ${
                  darkMode
                    ? 'border-white/[0.12] text-gray-300 hover:bg-white/[0.07] hover:text-gray-100 focus-visible:ring-white/20'
                    : 'border-[#e3ddd6] bg-[#fffdfb] text-[#776b60] hover:border-[#e9c3b3] hover:text-[#c96b50] focus-visible:ring-[#D97A5D]/25'
                }`}
              >
                <Plus className="h-3.5 w-3.5" strokeWidth={2} />
                写第一条笔记
              </button>
            )}
            darkMode={darkMode}
          />
        )
      ) : (
        userNotes.map((note, index) => {
          const hasSelectionAnchor = note.anchor_type !== 'page' && Boolean(note.text || note.rects?.length);
          return (
            <article
              key={note.id}
              className={`group relative border-b px-4 py-4 pr-28 transition-colors last:border-b-0 ${
                darkMode ? 'border-white/[0.06] hover:bg-amber-200/[0.025]' : 'border-[#f3eee9] hover:bg-[#fffcf8]'
              }`}
            >
              <div className="flex items-start gap-2.5">
                <span className={`mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-[8px] ${
                  darkMode ? 'bg-amber-300/[0.12] text-amber-300/80' : 'bg-[#fdf2e8] text-[#c98a3f]'
                }`}>
                  <StickyNote className="h-3.5 w-3.5" strokeWidth={1.9} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className={`mb-1.5 text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-[#82766a]'}`}>
                    {hasSelectionAnchor ? '引用笔记' : '页内笔记'}
                  </div>
                  <NoteMarkdown content={note.note} darkMode={darkMode} />
                  {hasSelectionAnchor && (
                    <div className={`mt-2 line-clamp-2 border-l-2 pl-2 text-[11px] leading-relaxed ${
                      darkMode ? 'border-white/10 text-gray-500' : 'border-[#ded5cd] text-[#776b60]'
                    }`}>
                      {note.text}
                    </div>
                  )}
                </div>
              </div>

              <div className="absolute right-3 top-3 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                {hasSelectionAnchor && (
                  <button
                    type="button"
                    onClick={() => onUserNoteClick?.(note)}
                    className={`inline-flex h-7 w-7 items-center justify-center rounded-[8px] transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                      darkMode
                        ? 'text-gray-500 hover:bg-white/[0.08] hover:text-amber-200 focus-visible:ring-white/15'
                        : 'text-[#82766a] hover:bg-amber-50 hover:text-amber-700 focus-visible:ring-amber-200'
                    }`}
                    aria-label={`定位第 ${index + 1} 条笔记原文`}
                    title="定位原文"
                  >
                    <MapPin className="h-3.5 w-3.5" />
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => onEditUserNote?.(note)}
                  className={`inline-flex h-7 w-7 items-center justify-center rounded-[8px] transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                    darkMode
                      ? 'text-gray-500 hover:bg-white/[0.08] hover:text-gray-200 focus-visible:ring-white/15'
                      : 'text-[#82766a] hover:bg-[#f1eeea] hover:text-[#5c534b] focus-visible:ring-[#e3ddd6]'
                  }`}
                  aria-label={`编辑第 ${index + 1} 条笔记`}
                  title="编辑笔记"
                >
                  <Pencil className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => onDeleteUserNote?.(note.id)}
                  className={`inline-flex h-7 w-7 items-center justify-center rounded-[8px] transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                    darkMode
                      ? 'text-gray-500 hover:bg-rose-400/10 hover:text-rose-300 focus-visible:ring-white/15'
                      : 'text-[#82766a] hover:bg-rose-50 hover:text-rose-600 focus-visible:ring-rose-200'
                  }`}
                  aria-label={hasSelectionAnchor ? `删除第 ${index + 1} 条笔记及关联标注` : `删除第 ${index + 1} 条笔记`}
                  title={hasSelectionAnchor ? '删除笔记及关联标注' : '删除笔记'}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </article>
          );
        })
      )}
      </div>
    </>
  );
}

// 「本页对应」面包屑：原来是第三张卡片，但它只有 0-2 条、和左侧「总结」共享同一份
// 大纲和同一个跳转函数，撑不起一个和翻译、笔记平级的卡片位。收进头部后它变成
// 翻页就自动更新的定位信息，回答「我这一页在全文论证结构的哪个位置」。
const OUTLINE_TRAIL_LIMIT = 3;

function OutlineTrail({ notes, activeNodeId, visitedSet, onNoteClick, darkMode }) {
  if (!notes.length) return null;
  const shown = notes.slice(0, OUTLINE_TRAIL_LIMIT);
  const overflow = notes.length - shown.length;

  return (
    <nav
      aria-label="本页对应的大纲位置"
      className="mt-1 flex min-w-0 flex-wrap items-center gap-x-1 gap-y-1"
    >
      <span className={`shrink-0 text-[10px] font-medium ${darkMode ? 'text-gray-600' : 'text-[#9c9186]'}`}>
        本页对应
      </span>
      {shown.map((note, index) => {
        const isActive = activeNodeId === note.id;
        const isVisited = !isActive && visitedSet.has(note.id);
        return (
          <React.Fragment key={note.id}>
            {index > 0 && (
              <span aria-hidden="true" className={darkMode ? 'text-gray-700' : 'text-[#c4b9ad]'}>›</span>
            )}
            <button
              type="button"
              onClick={() => onNoteClick?.(note)}
              title={note.summary || note.title}
              aria-current={isActive ? 'true' : undefined}
              className={`max-w-[18em] truncate rounded-full px-2 py-0.5 text-[11px] font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/25 ${
                isActive
                  ? (darkMode ? 'bg-[#F0653A]/20 text-[#fdc4af]' : 'bg-[#fdece4] text-[#b85f47]')
                  : isVisited
                    ? (darkMode ? 'text-gray-400 hover:bg-white/[0.07]' : 'text-[#82766a] hover:bg-[#f1ece7]')
                    : (darkMode ? 'text-gray-300 hover:bg-white/[0.07]' : 'text-[#5c534b] hover:bg-[#f1ece7]')
              }`}
            >
              {note.title}
            </button>
          </React.Fragment>
        );
      })}
      {overflow > 0 && (
        <span className={`shrink-0 text-[10px] font-medium tabular-nums ${darkMode ? 'text-gray-600' : 'text-[#9c9186]'}`}>
          +{overflow}
        </span>
      )}
    </nav>
  );
}

function ReadingAnalysisPanel({
  blocks = [],
  translations = {},
  translatingBlockIds = [],
  loading = false,
  error = '',
  notice = '',
  pretranslateProgress = { running: false, done: 0, total: 0 },
  onPretranslate,
  onBackfillSummaries,
  backfillingSummaries = false,
  currentPage = 1,
  activeBlockId = null,
  notes = [],
  userNotes = [],
  activeNodeId = null,
  visitedNodeIds = [],
  revealUserNoteId = '',
  onUserNoteReveal,
  onTranslate,
  onRetranslateBlock,
  onBlockHover,
  onBlockClick,
  onNoteClick,
  onUserNoteClick,
  onSaveUserNote,
  onDeleteUserNote,
  documentId = '',
  darkMode = false,
  translationSurface = 'panel',
  onTranslationSurfaceChange,
}) {
  const isTranslationDocked = translationSurface === 'dock';
  const [storedLayout, setStoredLayout] = useDebouncedLocalStorage(
    READING_WIDGET_LAYOUT_STORAGE_KEY,
    DEFAULT_READING_WIDGET_LAYOUT,
    180
  );
  const [noteEditor, setNoteEditor] = useState({
    open: false,
    noteId: '',
    page: currentPage,
    value: '',
  });
  const [translationView, setTranslationView] = useDebouncedLocalStorage(
    TRANSLATION_VIEW_STORAGE_KEY,
    'compare',
    180
  );
  const [noteEditorSaving, setNoteEditorSaving] = useState(false);
  const [noteEditorError, setNoteEditorError] = useState('');
  const handledRevealIdRef = useRef('');
  const boardRef = useRef(null);
  const shellRef = useRef(null);
  const [shellWidth, setShellWidth] = useState(0);
  const [resize, setResize] = useState(IDLE_RESIZE);
  // 拖动中缝时的实时比例；null 表示没在拖，用持久化值。
  const [liveSplit, setLiveSplit] = useState(null);
  const layout = useMemo(() => normalizeReadingWidgetLayout(storedLayout), [storedLayout]);
  // 吸附栏开着时翻译卡整个退出板子，槽位由剩下的组件重新分配，
  // 但 layout.order 里 translation 的下标保持不动，关掉吸附栏能原样回来。
  const visibleOrder = useMemo(
    () => (isTranslationDocked ? layout.order.filter((id) => id !== 'translation') : layout.order),
    [isTranslationDocked, layout.order]
  );
  const { slots, rows, columnCount } = useMemo(
    () => resolveReadingWidgetSlots(visibleOrder, layout.fullWidth),
    [layout.fullWidth, visibleOrder]
  );
  // 列区那一行是 1fr，会吃掉全部剩余高度。里面的卡片如果全是折叠的，
  // 就会出现「一张 58px 的卡占着 780px」的空洞，把通栏卡片全挤到底部。
  const columnRegionOpen = useMemo(
    () => visibleOrder.some((id) => slots[id] !== 'foot' && !layout.collapsed[id]),
    [layout.collapsed, slots, visibleOrder]
  );
  const translatableBlocks = blocks.filter((block) => block?.block_id && block?.text);
  const translatableCount = translatableBlocks.length;
  const pendingCount = translatableBlocks.filter((block) => !translations[block.block_id]).length;
  const visitedSet = useMemo(() => new Set(visitedNodeIds || []), [visitedNodeIds]);
  const translatingSet = useMemo(() => new Set(translatingBlockIds || []), [translatingBlockIds]);
  const isDefaultLayout = isDefaultReadingWidgetLayout(layout);

  const persistLayout = useCallback((nextLayout) => {
    setStoredLayout(normalizeReadingWidgetLayout(nextLayout));
  }, [setStoredLayout]);

  // 换位 = 两张卡完全交换位置，所以「是否通栏」也要跟着换，
  // 否则把底部通栏的卡拖到左列，它还会固执地待在底部。
  const handleSwapWidgets = useCallback((draggedId, targetId) => {
    const order = [...layout.order];
    const from = order.indexOf(draggedId);
    const to = order.indexOf(targetId);
    if (from < 0 || to < 0 || from === to) return;
    order[from] = targetId;
    order[to] = draggedId;
    persistLayout({
      ...layout,
      order,
      fullWidth: {
        ...layout.fullWidth,
        [draggedId]: layout.fullWidth[targetId],
        [targetId]: layout.fullWidth[draggedId],
      },
    });
  }, [layout, persistLayout]);

  const handleDropOnSlot = useCallback((draggedId, slot) => {
    const nextFullWidth = slot === 'foot';
    if (layout.fullWidth[draggedId] === nextFullWidth) return;
    persistLayout({
      ...layout,
      fullWidth: { ...layout.fullWidth, [draggedId]: nextFullWidth },
    });
  }, [layout, persistLayout]);

  const handleToggleFullWidth = useCallback((id) => {
    persistLayout({
      ...layout,
      fullWidth: { ...layout.fullWidth, [id]: !layout.fullWidth[id] },
    });
  }, [layout, persistLayout]);

  const { drag, startDrag } = useWidgetSlotDrag({
    boardRef,
    canDrag: visibleOrder.length > 1,
    onSwap: handleSwapWidgets,
    onDropSlot: handleDropOnSlot,
  });

  // 双栏下 main/side 由行高决定，手动高度只对通栏的 foot 生效，
  // 所以要知道当前实际是几栏——容器查询 CSS 拿不到，只能自己量。
  useEffect(() => {
    const shell = shellRef.current;
    if (!shell || typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver((entries) => {
      setShellWidth(entries[entries.length - 1]?.contentRect?.width || 0);
    });
    observer.observe(shell);
    return () => observer.disconnect();
  }, []);
  const isTwoColumn = shellWidth >= TWO_COLUMN_MIN_WIDTH;

  const persistHeight = useCallback((id, height) => {
    persistLayout({
      ...layout,
      sizes: { ...layout.sizes, [id]: clampReadingWidgetHeight(height) },
    });
  }, [layout, persistLayout]);

  const handleColumnSplitStart = useCallback((event, keyboardDelta) => {
    if (typeof keyboardDelta === 'number') {
      event.preventDefault();
      persistLayout({ ...layout, split: clampReadingWidgetSplit(layout.split + keyboardDelta) });
      return;
    }
    if (event.button > 0) return;
    event.preventDefault();

    // 用左右两列的实际矩形算比例，把中缝和内边距都排除在外。
    const board = boardRef.current;
    const mainRect = board?.querySelector('[data-widget-slot="main"]')?.getBoundingClientRect();
    const sideRect = board?.querySelector('[data-widget-slot="side"]')?.getBoundingClientRect();
    if (!mainRect || !sideRect) return;
    const left = mainRect.left;
    const width = Math.max(1, sideRect.right - left);
    const computeSplit = (clientX) => clampReadingWidgetSplit(((clientX - left) / width) * 100);

    setLiveSplit(computeSplit(event.clientX));
    let sawButtons = false;
    const handleMove = (moveEvent) => {
      // 同上：先确认环境会上报 buttons，再把 0 当成漏掉的 pointerup
      if (moveEvent.buttons > 0) {
        sawButtons = true;
      } else if (sawButtons) {
        handleEnd(moveEvent);
        return;
      }
      setLiveSplit(computeSplit(moveEvent.clientX));
    };
    const handleEnd = (endEvent) => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleEnd);
      window.removeEventListener('pointercancel', handleEnd);
      window.removeEventListener('blur', handleEnd);
      setLiveSplit(null);
      persistLayout({ ...layout, split: computeSplit(endEvent?.clientX ?? event.clientX) });
    };
    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleEnd);
    window.addEventListener('pointercancel', handleEnd);
    window.addEventListener('blur', handleEnd);
  }, [layout, persistLayout]);

  const handleResizeStart = useCallback((id) => (event, keyboardDelta) => {
    const section = boardRef.current?.querySelector(`[data-reading-widget-id="${id}"]`);
    if (!section) return;
    const startHeight = section.getBoundingClientRect().height;

    if (typeof keyboardDelta === 'number') {
      event.preventDefault();
      persistHeight(id, startHeight + keyboardDelta);
      return;
    }
    if (event.button > 0) return;
    event.preventDefault();
    event.stopPropagation();
    setResize({ id, height: clampReadingWidgetHeight(startHeight) });

    const originY = event.clientY;
    let sawButtons = false;
    const handleMove = (moveEvent) => {
      // 同上：先确认环境会上报 buttons，再把 0 当成漏掉的 pointerup
      if (moveEvent.buttons > 0) {
        sawButtons = true;
      } else if (sawButtons) {
        handleEnd(moveEvent);
        return;
      }
      setResize({
        id,
        height: clampReadingWidgetHeight(startHeight + (moveEvent.clientY - originY)),
      });
    };
    const handleEnd = (endEvent) => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleEnd);
      window.removeEventListener('pointercancel', handleEnd);
      window.removeEventListener('blur', handleEnd);
      setResize(IDLE_RESIZE);
      persistHeight(id, startHeight + ((endEvent?.clientY ?? originY) - originY));
    };
    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleEnd);
    window.addEventListener('pointercancel', handleEnd);
    window.addEventListener('blur', handleEnd);
  }, [persistHeight]);

  const handleToggleWidget = useCallback((id) => {
    persistLayout({
      ...layout,
      collapsed: {
        ...layout.collapsed,
        [id]: !layout.collapsed[id],
      },
    });
  }, [layout, persistLayout]);

  // 只在「看得见的组件」之间换位；被吸附栏藏起来的翻译卡不该出现在移动序列里，
  // 否则按一下方向键会像什么都没发生。
  const handleMoveWidget = useCallback((id, delta) => {
    const currentIndex = visibleOrder.indexOf(id);
    const nextIndex = currentIndex + delta;
    if (currentIndex < 0 || nextIndex < 0 || nextIndex >= visibleOrder.length) return;
    const order = [...layout.order];
    const from = order.indexOf(id);
    const to = order.indexOf(visibleOrder[nextIndex]);
    if (from < 0 || to < 0) return;
    order[from] = visibleOrder[nextIndex];
    order[to] = id;
    persistLayout({ ...layout, order });
  }, [layout, persistLayout, visibleOrder]);

  const handleRestoreLayout = useCallback(() => {
    persistLayout(DEFAULT_READING_WIDGET_LAYOUT);
  }, [persistLayout]);

  const expandUserNotesWidget = useCallback(() => {
    if (!layout.collapsed.user_notes) return;
    persistLayout({
      ...layout,
      collapsed: { ...layout.collapsed, user_notes: false },
    });
  }, [layout, persistLayout]);

  const handleStartNewUserNote = useCallback(() => {
    expandUserNotesWidget();
    setNoteEditor({ open: true, noteId: '', page: currentPage, value: '' });
    setNoteEditorError('');
  }, [currentPage, expandUserNotesWidget]);

  const handleStartEditUserNote = useCallback((note) => {
    if (!note) return;
    expandUserNotesWidget();
    setNoteEditor({
      open: true,
      noteId: note.id,
      page: Math.max(1, Number(note.page) || currentPage),
      value: String(note.note || ''),
    });
    setNoteEditorError('');
  }, [currentPage, expandUserNotesWidget]);

  const handleCancelUserNote = useCallback(() => {
    if (noteEditorSaving) return;
    setNoteEditor((current) => ({ ...current, open: false }));
    setNoteEditorError('');
  }, [noteEditorSaving]);

  const handleSaveUserNote = useCallback(async () => {
    const content = noteEditor.value.trim();
    if (!content) {
      setNoteEditorError('笔记内容不能为空');
      return;
    }
    if (typeof onSaveUserNote !== 'function') {
      setNoteEditorError('当前文档无法保存笔记');
      return;
    }

    setNoteEditorSaving(true);
    setNoteEditorError('');
    try {
      await onSaveUserNote({
        id: noteEditor.noteId || undefined,
        note: content,
        page: noteEditor.page,
      });
      setNoteEditor((current) => ({ ...current, open: false, value: '' }));
    } catch (saveError) {
      setNoteEditorError(saveError?.message || '笔记保存失败，请重试');
    } finally {
      setNoteEditorSaving(false);
    }
  }, [noteEditor.noteId, noteEditor.page, noteEditor.value, onSaveUserNote]);

  useEffect(() => {
    setNoteEditor({ open: false, noteId: '', page: 1, value: '' });
    setNoteEditorError('');
    setNoteEditorSaving(false);
  }, [documentId]);

  useEffect(() => {
    if (!revealUserNoteId || handledRevealIdRef.current === revealUserNoteId) return;
    handledRevealIdRef.current = revealUserNoteId;
    if (layout.collapsed.user_notes) {
      persistLayout({
        ...layout,
        collapsed: { ...layout.collapsed, user_notes: false },
      });
    }
    onUserNoteReveal?.(revealUserNoteId);
  }, [layout, onUserNoteReveal, persistLayout, revealUserNoteId]);

  const widgetDefinitions = {
    translation: {
      title: '页面翻译',
      // 明确写出还有多少块没译，否则切到「译文」看不到东西会以为开关坏了
      meta: pendingCount > 0
        ? `${translatableCount} 块 · ${pendingCount} 待译`
        : `${translatableCount} 块 · 已全部翻译`,
      icon: Languages,
      iconClassName: darkMode ? 'text-[#fdc4af]' : 'text-[#c96b50]',
      headerAction: (
        <TranslationViewToggle
          value={translationView === 'target' ? 'target' : 'compare'}
          onChange={setTranslationView}
          darkMode={darkMode}
        />
      ),
      bodyClassName: 'reading-widget-flex overflow-hidden',
      content: (
        <TranslationWidgetContent
          blocks={blocks}
          translations={translations}
          translatingSet={translatingSet}
          activeBlockId={activeBlockId}
          viewMode={translationView === 'target' ? 'target' : 'compare'}
          error={error}
          notice={notice}
          pretranslateProgress={pretranslateProgress}
          onPretranslate={onPretranslate}
          onBackfillSummaries={onBackfillSummaries}
          backfillingSummaries={backfillingSummaries}
          onRetranslateBlock={onRetranslateBlock}
          onBlockHover={onBlockHover}
          onBlockClick={onBlockClick}
          darkMode={darkMode}
        />
      ),
    },
    user_notes: {
      title: '我的笔记',
      meta: `${userNotes.length} 条笔记`,
      icon: StickyNote,
      iconClassName: darkMode ? 'text-amber-300/80' : 'text-amber-600',
      headerAction: (
        <button
          type="button"
          onClick={handleStartNewUserNote}
          className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px] transition-colors focus-visible:outline-none focus-visible:ring-2 ${
            darkMode
              ? 'text-gray-400 hover:bg-white/[0.08] hover:text-gray-100 focus-visible:ring-white/20'
              : 'text-[#82766a] hover:bg-[#f3efeb] hover:text-[#b85f47] focus-visible:ring-[#D97A5D]/25'
          }`}
          aria-label="新建笔记"
          title="新建笔记"
        >
          <Plus className="h-4 w-4" strokeWidth={2} />
        </button>
      ),
      bodyClassName: 'reading-widget-flex flex max-h-[min(58vh,640px)] flex-col overflow-hidden',
      content: (
        <UserNotesWidgetContent
          userNotes={userNotes}
          editor={noteEditor}
          editorSaving={noteEditorSaving}
          editorError={noteEditorError}
          onEditorChange={(value) => setNoteEditor((current) => ({ ...current, value }))}
          onEditorSave={handleSaveUserNote}
          onEditorCancel={handleCancelUserNote}
          onUserNoteClick={onUserNoteClick}
          onEditUserNote={handleStartEditUserNote}
          onDeleteUserNote={onDeleteUserNote}
          onStartUserNote={handleStartNewUserNote}
          darkMode={darkMode}
        />
      ),
    },
  };

  return (
    <div className={`flex h-full flex-col ${darkMode ? 'text-gray-200' : 'text-[#3a332e]'}`}>
      <header className={`shrink-0 border-b px-5 py-3 ${darkMode ? 'border-white/10' : 'border-[#f0ebe6]'}`}>
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex min-w-0 items-baseline gap-2.5">
              <h2 className="shrink-0 text-[15px] font-semibold tracking-[-0.01em]">第 {currentPage} 页</h2>
              <p className={`truncate text-[11px] font-medium tabular-nums ${darkMode ? 'text-gray-500' : 'text-[#82766a]'}`}>
                {translatableCount} 块 · {userNotes.length} 笔记
              </p>
            </div>
            <OutlineTrail
              notes={notes}
              activeNodeId={activeNodeId}
              visitedSet={visitedSet}
              onNoteClick={onNoteClick}
              darkMode={darkMode}
            />
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={handleRestoreLayout}
              disabled={isDefaultLayout}
              className={`inline-flex h-9 w-9 items-center justify-center rounded-[11px] border transition-colors disabled:cursor-default disabled:opacity-35 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/30 ${
                darkMode
                  ? 'border-white/[0.09] text-gray-400 hover:bg-white/[0.07] hover:text-gray-200'
                  : 'border-[#e8e3de] bg-white/80 text-[#82766a] hover:bg-[#f5f1ed] hover:text-[#5c534b]'
              }`}
              aria-label="恢复默认布局"
              title="恢复默认布局"
            >
              <RotateCcw className="h-4 w-4" strokeWidth={2} />
            </button>
            <button
              type="button"
              onClick={onTranslate}
              disabled={loading || translatableCount === 0}
              className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-4 py-2.5 text-xs font-semibold transition-[transform,background-color,box-shadow] duration-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50 disabled:active:scale-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${
                darkMode
                  ? 'bg-[#F0653A] text-white shadow-[0_9px_22px_-8px_rgba(240,101,58,0.45)] hover:-translate-y-0.5 hover:bg-[#F5713F]'
                  : 'accent-cta'
              }`}
            >
              {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Languages className="h-3.5 w-3.5" />}
              翻译当前页
            </button>
          </div>
        </div>
      </header>

      {isTranslationDocked && (
        <TranslationDockedStrip
          pretranslateProgress={pretranslateProgress}
          onPretranslate={onPretranslate}
          onReturnToPanel={() => onTranslationSurfaceChange?.('panel')}
          error={error}
          notice={notice}
          darkMode={darkMode}
        />
      )}

      <div
        ref={shellRef}
        data-testid="reading-widget-board-shell"
        className="reading-widget-board-shell min-h-0 flex-1"
      >
        <div
          ref={boardRef}
          data-testid="reading-widget-board"
          data-dragging={drag.id ? 'true' : undefined}
          data-col-resizing={liveSplit != null ? 'true' : undefined}
          data-columns={columnCount}
          data-columns-open={columnRegionOpen ? 'true' : 'false'}
          className="reading-widget-board custom-scrollbar h-full overflow-y-auto overscroll-contain px-4 py-4"
          style={{
            '--reading-col-main': `${liveSplit ?? layout.split}fr`,
            '--reading-col-side': `${100 - (liveSplit ?? layout.split)}fr`,
          }}
        >
          {isTwoColumn && (
            <div
              role="separator"
              aria-orientation="vertical"
              aria-label="调整左右栏宽度"
              tabIndex={0}
              onPointerDown={handleColumnSplitStart}
              onKeyDown={(event) => {
                if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
                handleColumnSplitStart(event, event.key === 'ArrowLeft' ? -2 : 2);
              }}
              className="reading-col-resize group/colresize relative focus-visible:outline-none"
              title="拖动调整左右栏宽度，方向键也可以"
            >
              {/* 常驻的一道细缝，让人看得出这里可以拖 */}
              <span
                aria-hidden="true"
                className={`absolute inset-y-2 w-px transition-colors duration-200 ${
                  liveSplit != null
                    ? 'bg-[#eab89f]'
                    : darkMode
                      ? 'bg-white/[0.08] group-hover/colresize:bg-white/[0.16]'
                      : 'bg-[#e6dfd7] group-hover/colresize:bg-[#d8cec3]'
                }`}
              />
              {/* 中间的抓手，hover 变长变深 */}
              <span
                aria-hidden="true"
                className={`relative w-[3px] rounded-full transition-all duration-200 ${
                  liveSplit != null
                    ? 'h-20 bg-[#e0a58b]'
                    : darkMode
                      ? 'h-10 bg-white/[0.14] group-hover/colresize:h-20 group-hover/colresize:bg-white/30 group-focus-visible/colresize:bg-white/40'
                      : 'h-10 bg-[#d8cec3] group-hover/colresize:h-20 group-hover/colresize:bg-[#c2b3a6] group-focus-visible/colresize:bg-[#c9a08c]'
                }`}
              />
            </div>
          )}
          {visibleOrder.map((id) => {
            const widget = widgetDefinitions[id];
            const slot = slots[id];
            // 双栏里 main/side 撑满整行，手动高度会和行高打架，所以只让通栏的 foot 可调。
            const canResize = !isTwoColumn || slot === 'foot';
            // 折叠后不再套用手动高度，否则会剩一个撑开的空壳：
            // 外壳有 500px 显式高度，里面只有一条 48px 的标题栏。
            const storedHeight = canResize && !layout.collapsed[id] ? layout.sizes[id] : 0;
            return (
              <ReadingWidgetCard
                key={id}
                id={id}
                title={widget.title}
                meta={widget.meta}
                icon={widget.icon}
                iconClassName={widget.iconClassName}
                slot={slot}
                gridRow={rows[id] || 1}
                collapsed={layout.collapsed[id]}
                height={resize.id === id ? resize.height : storedHeight}
                resizable={canResize}
                onResizeStart={handleResizeStart(id)}
                onToggle={handleToggleWidget}
                onMove={handleMoveWidget}
                onToggleFullWidth={handleToggleFullWidth}
                fullWidth={Boolean(layout.fullWidth[id])}
                onDragStart={startDrag(id)}
                dragging={drag.id === id}
                dragOffset={drag}
                total={visibleOrder.length}
                headerAction={widget.headerAction}
                bodyClassName={widget.bodyClassName}
                darkMode={darkMode}
              >
                {widget.content}
              </ReadingWidgetCard>
            );
          })}

          {/* 拖拽中才出现的落点：空着的列 和 底部通栏。
              有了它们，卡片的位置才是用户选的，而不是按排序下标硬推出来的。 */}
          {drag.id && columnCount < 2 && layout.fullWidth[drag.id] && (
            <div
              data-drop-slot="column"
              className={`reading-drop-zone reading-drop-zone--column flex items-center justify-center rounded-[24px] border-2 border-dashed text-[11px] font-medium ${
                darkMode
                  ? 'border-white/20 bg-white/[0.03] text-gray-400'
                  : 'border-[#e0d4ca] bg-[#fbf7f4] text-[#82766a]'
              }`}
            >
              放到这一列
            </div>
          )}
          {drag.id && !layout.fullWidth[drag.id] && (
            <div
              data-drop-slot="foot"
              // 排在所有通栏卡片之后；不给显式行号会被自动放置塞进列区那一行
              style={{ '--widget-row': visibleOrder.length + 2 }}
              className={`reading-drop-zone reading-drop-zone--foot flex h-16 items-center justify-center rounded-[24px] border-2 border-dashed text-[11px] font-medium ${
                darkMode
                  ? 'border-white/20 bg-white/[0.03] text-gray-400'
                  : 'border-[#e0d4ca] bg-[#fbf7f4] text-[#82766a]'
              }`}
            >
              放到底部通栏
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default memo(ReadingAnalysisPanel);
