import React, {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Copy,
  Highlighter,
  Loader2,
  Globe,
  Search,
  Share2,
  Trash2,
  Underline,
  X,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  DEFAULT_DOCUMENT_HIGHLIGHT_COLOR,
  DOCUMENT_HIGHLIGHT_COLORS,
  normalizeDocumentHighlightColor,
} from '../utils/documentHighlightUtils';
import { FloatingDock, FloatingDockItem } from './ui/FloatingDock';

/**
 * 划词交互工具箱
 * 固定在 PDF 页码栏下方。高亮和下划线是持续标注工具，
 * 其余操作仍在选中文字后启用。
 */
// CodeMirror 只在打开笔记编辑器时才需要，懒加载避免拖慢首屏。
const MarkdownNoteEditor = lazy(() => import('./MarkdownNoteEditor'));

const NOTE_EDITOR_MAX_WIDTH = 620;
const NOTE_EDITOR_GUTTER = 12;
const NOTE_EDITOR_FALLBACK_HEIGHT = 320;

const NoteIcon = ({ className = '' }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    className={className}
    fill="none"
    aria-hidden="true"
    focusable="false"
    data-note-icon="true"
  >
    <path
      stroke="currentColor"
      strokeLinecap="round"
      strokeWidth="1.5"
      d="m16.652 3.455.649-.649A2.753 2.753 0 0 1 21.194 6.7l-.65.649m-3.892-3.893s.081 1.379 1.298 2.595c1.216 1.217 2.595 1.298 2.595 1.298m-3.893-3.893L10.687 9.42c-.404.404-.606.606-.78.829q-.308.395-.524.848c-.121.255-.211.526-.392 1.068L8.412 13.9m12.133-6.552-2.983 2.982m-2.982 2.983c-.404.404-.606.606-.829.78a4.6 4.6 0 0 1-.848.524c-.255.121-.526.211-1.068.392l-1.735.579m0 0-1.123.374a.742.742 0 0 1-.939-.94l.374-1.122m1.688 1.688L8.412 13.9"
    />
    <path
      fill="currentColor"
      d="M22.75 12a.75.75 0 0 0-1.5 0zM12 2.75a.75.75 0 0 0 0-1.5zM7.376 20.013a.75.75 0 1 0-.752 1.298zm-4.687-2.638a.75.75 0 1 0 1.298-.75zM21.25 12A9.25 9.25 0 0 1 12 21.25v1.5c5.937 0 10.75-4.813 10.75-10.75zM12 1.25C6.063 1.25 1.25 6.063 1.25 12h1.5A9.25 9.25 0 0 1 12 2.75zM6.624 21.311A10.7 10.7 0 0 0 12 22.75v-1.5a9.2 9.2 0 0 1-4.624-1.237zM1.25 12a10.7 10.7 0 0 0 1.439 5.375l1.298-.75A9.2 9.2 0 0 1 2.75 12z"
    />
  </svg>
);

const AIExplainIcon = ({ className = '' }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 14 14"
    className={className}
    fill="none"
    aria-hidden="true"
    focusable="false"
    data-ai-explain-icon="true"
  >
    <g stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8.406 7.97c-.386.44-.856.8-1.385 1.061v1.5a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1.5A4.5 4.5 0 0 1 6.875.9M3.021 13.5h4" />
      <path d="M7.395 3.934c-.351-.061-.351-.565 0-.626A3.18 3.18 0 0 0 9.953.858L9.974.76c.076-.347.57-.349.649-.003l.026.113a3.19 3.19 0 0 0 2.565 2.435c.353.062.353.568 0 .63A3.19 3.19 0 0 0 10.65 6.37l-.026.113c-.079.346-.573.344-.649-.003l-.02-.097a3.18 3.18 0 0 0-2.56-2.45" />
    </g>
  </svg>
);

const TextSelectionToolbar = ({
  selectedText,
  position = { x: 0, y: 0 },
  onCopy,
  annotationTool = null,
  annotationColor = DEFAULT_DOCUMENT_HIGHLIGHT_COLOR,
  onAnnotationToolChange,
  onAnnotationColorChange,
  canDeleteAnnotation = false,
  onDeleteAnnotation,
  onAddNote,
  onAIExplain,
  onTranslate,
  onWebSearch,
  onShare,
  size = 'normal', // 新增：支持 'compact', 'normal', 'large'
  darkMode = false,
}) => {
  const toolbarRef = useRef(null);
  const toolbarCapsuleRef = useRef(null);
  const noteEditorRef = useRef(null);
  const feedbackTimerRef = useRef(null);
  const [noteEditorOpen, setNoteEditorOpen] = useState(false);
  const [noteEditorPlacement, setNoteEditorPlacement] = useState('bottom');
  const [noteEditorLayout, setNoteEditorLayout] = useState({
    width: NOTE_EDITOR_MAX_WIDTH,
    offsetX: 0,
  });
  const [noteDraft, setNoteDraft] = useState('');
  const [activeAction, setActiveAction] = useState('');
  const [actionFeedback, setActionFeedback] = useState(null);
  const hasSelection = Boolean(selectedText?.trim());
  const currentAnnotationColor = normalizeDocumentHighlightColor(annotationColor);

  useEffect(() => () => {
    if (feedbackTimerRef.current) window.clearTimeout(feedbackTimerRef.current);
  }, []);

  useEffect(() => {
    setNoteEditorOpen(false);
    setNoteDraft('');
    setActionFeedback(null);
  }, [selectedText]);

  useLayoutEffect(() => {
    if (!noteEditorOpen) return undefined;

    const surface = toolbarRef.current?.closest('[data-pdf-reader-surface]');
    let frame = null;

    const updateLayout = () => {
      const toolbarRect = toolbarRef.current?.getBoundingClientRect();
      if (!toolbarRect) return;

      const measuredAnchorRect = toolbarCapsuleRef.current?.getBoundingClientRect();
      const anchorRect = measuredAnchorRect?.width > 0 ? measuredAnchorRect : toolbarRect;
      const panelHeight = noteEditorRef.current?.getBoundingClientRect().height
        || NOTE_EDITOR_FALLBACK_HEIGHT;
      const requiredSpace = panelHeight + 16;

      const bounds = surface?.getBoundingClientRect();
      const viewportWidth = window.innerWidth || document.documentElement.clientWidth || NOTE_EDITOR_MAX_WIDTH;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
      const hasUsableBounds = Boolean(bounds && bounds.width > 0 && bounds.height > 0);
      const limitLeft = Math.max(hasUsableBounds ? bounds.left : 0, 0);
      const limitRight = Math.min(hasUsableBounds ? bounds.right : viewportWidth, viewportWidth);
      const limitBottom = Math.min(hasUsableBounds ? bounds.bottom : viewportHeight, viewportHeight);
      const limitTop = Math.max(hasUsableBounds ? bounds.top : 0, 0);
      const spaceBelow = limitBottom - toolbarRect.bottom;
      const spaceAbove = toolbarRect.top - limitTop;
      const nextPlacement = spaceBelow >= requiredSpace || spaceBelow >= spaceAbove ? 'bottom' : 'top';
      setNoteEditorPlacement((current) => (current === nextPlacement ? current : nextPlacement));

      const availableWidth = Math.max(1, limitRight - limitLeft - NOTE_EDITOR_GUTTER * 2);
      const width = Math.min(NOTE_EDITOR_MAX_WIDTH, Math.floor(availableWidth));
      const anchorCenter = anchorRect.left + anchorRect.width / 2;
      const minCenter = limitLeft + NOTE_EDITOR_GUTTER + width / 2;
      const maxCenter = limitRight - NOTE_EDITOR_GUTTER - width / 2;
      const clampedCenter = Math.min(Math.max(anchorCenter, minCenter), maxCenter);
      const offsetX = Math.round(clampedCenter - anchorCenter);

      setNoteEditorLayout((current) => (
        current.width === width && current.offsetX === offsetX
          ? current
          : { width, offsetX }
      ));
    };

    const scheduleUpdate = () => {
      if (frame !== null) return;
      frame = window.requestAnimationFrame(() => {
        frame = null;
        updateLayout();
      });
    };

    // 首次打开在浏览器绘制前完成约束，避免先越界再跳回阅读区。
    updateLayout();
    window.addEventListener('resize', scheduleUpdate);

    const resizeObserver = typeof window.ResizeObserver === 'function'
      ? new window.ResizeObserver(scheduleUpdate)
      : null;
    [surface, toolbarCapsuleRef.current, noteEditorRef.current].forEach((element) => {
      if (element) resizeObserver?.observe(element);
    });

    return () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
      resizeObserver?.disconnect();
      window.removeEventListener('resize', scheduleUpdate);
    };
  }, [noteEditorOpen]);

  const showFeedback = (message, tone = 'success') => {
    if (!message) return;
    if (feedbackTimerRef.current) window.clearTimeout(feedbackTimerRef.current);
    setActionFeedback({ message, tone });
    feedbackTimerRef.current = window.setTimeout(() => setActionFeedback(null), 2400);
  };

  const runToolAction = async (tool) => {
    if (!hasSelection || activeAction) return;
    if (tool.kind === 'note') {
      setActionFeedback(null);
      setNoteEditorOpen((open) => !open);
      return;
    }
    if (typeof tool.action !== 'function') return;
    setActiveAction(tool.label);
    try {
      const result = await tool.action();
      if (result?.message) showFeedback(result.message, result.tone || 'success');
    } catch (error) {
      showFeedback(error?.message || `${tool.label}失败，请重试`, 'error');
    } finally {
      setActiveAction('');
    }
  };

  const saveNote = async () => {
    if (!hasSelection || activeAction) return;
    const value = noteDraft.trim();
    if (!value) {
      showFeedback('请输入笔记内容', 'error');
      return;
    }
    if (typeof onAddNote !== 'function') return;
    setActiveAction('笔记');
    try {
      const result = await onAddNote(value);
      setNoteEditorOpen(false);
      setNoteDraft('');
      if (result?.message) showFeedback(result.message, result.tone || 'success');
    } catch (error) {
      showFeedback(error?.message || '笔记保存失败，请重试', 'error');
    } finally {
      setActiveAction('');
    }
  };

  // 根据大小配置样式
  const sizeConfig = {
    compact: {
      iconSize: 'w-4 h-4',
      gap: 'gap-0.5',
      containerPadding: 'px-2 py-1',
      dockSlot: 'h-7 w-7',
      dockButton: 'h-6 w-6 rounded-[8px]',
    },
    normal: {
      iconSize: 'w-[18px] h-[18px]',
      gap: 'gap-0.5',
      containerPadding: 'px-2.5 py-1.5',
      dockSlot: 'h-8 w-8',
      dockButton: 'h-7 w-7 rounded-[9px]',
    },
    large: {
      iconSize: 'w-5 h-5',
      gap: 'gap-1',
      containerPadding: 'px-3 py-2',
      dockSlot: 'h-9 w-9',
      dockButton: 'h-8 w-8 rounded-[10px]',
    }
  };

  const config = sizeConfig[size] || sizeConfig.normal;

  // 没选中文字时，除标注工具外全都是 disabled 状态。与其常驻一排灰按钮占着位置，
  // 不如收起来——划词之后再展开完整工具集。
  const allTools = [
    {
      icon: Copy,
      label: '复制',
      action: onCopy,
      color: 'text-gray-600 hover:text-gray-900'
    },
    {
      icon: Highlighter,
      label: '高亮',
      kind: 'highlight',
      color: 'text-yellow-600 hover:text-yellow-700'
    },
    {
      icon: Underline,
      label: '下划线',
      kind: 'underline',
      color: 'text-[#c96b50] hover:text-[#a64f36]'
    },
    {
      icon: NoteIcon,
      label: '笔记',
      kind: 'note',
      color: 'text-rose-600 hover:text-rose-700'
    },
    ...(canDeleteAnnotation ? [{
      icon: Trash2,
      label: '删除标注',
      action: onDeleteAnnotation,
      color: 'text-rose-500 hover:text-rose-700',
    }] : []),
    {
      icon: AIExplainIcon,
      label: 'AI 解读',
      action: onAIExplain,
      color: 'text-purple-600 hover:text-purple-700'
    },
    {
      icon: Globe,
      label: '翻译',
      action: onTranslate,
      color: 'text-green-600 hover:text-green-700'
    },
    {
      icon: Search,
      label: '搜索',
      action: onWebSearch,
      color: 'text-indigo-600 hover:text-indigo-700'
    },
    {
      icon: Share2,
      label: '分享',
      action: onShare,
      color: 'text-pink-600 hover:text-pink-700'
    }
  ];

  // 标注工具（高亮/下划线）不需要选区就能用，先启用再拖着划；其余的都要先有选区。
  const tools = hasSelection ? allTools : allTools.filter((tool) => tool.kind === 'highlight' || tool.kind === 'underline');

  const panelPlacementClass = noteEditorPlacement === 'top'
    ? 'bottom-full mb-3 origin-bottom'
    : 'top-full mt-3 origin-top';

  return (
    <AnimatePresence>
      <motion.div
        ref={toolbarRef}
        initial={{ opacity: 0, y: -8, scale: 0.98 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6, scale: 0.98 }}
        transition={{ duration: 0.18, ease: 'easeOut' }}
        data-testid="text-selection-toolbar"
        data-docked="true"
        data-has-selection={hasSelection ? 'true' : 'false'}
        // 正常流里的一条窄带：位置完全由 PDF 面板的布局决定，
        // 不再有 fixed + 缓存坐标那套（面板一挪坐标就过期，工具栏会飘到外面去）。
        className={`relative z-20 flex shrink-0 justify-center px-3 pb-1.5 pt-1 ${
          darkMode ? 'text-gray-200' : 'text-gray-600'
        }`}
      >
        {/* 批注操作是离散动作，保留胶囊的稳定边界，只让图标本身跟随鼠标做弹簧反馈。 */}
        <div ref={toolbarCapsuleRef} data-testid="selection-toolbar-capsule" className="relative inline-flex max-w-full">
          <FloatingDock
            darkMode={darkMode}
            ariaLabel="文档批注与笔记工具栏"
            className={`h-auto min-h-[42px] max-w-full rounded-full p-0 shadow-[0_1px_2px_rgba(83,65,55,0.04),0_10px_24px_-14px_rgba(83,65,55,0.28)] ${
              darkMode
                ? 'border-white/[0.09] bg-[#22262c]'
                : 'border-[#ebe4dd] bg-white'
            }`}
          >
          {/* 工具按钮组 */}
          <div className={`${config.containerPadding} flex items-center ${config.gap}`}>
            {tools.map((tool) => {
              const Icon = tool.icon;
              const isHighlight = tool.kind === 'highlight';
              const isUnderline = tool.kind === 'underline';
              const isAnnotationTool = isHighlight || isUnderline;
              const isToolActive = isAnnotationTool && annotationTool === tool.kind;
              const annotationToolTitle = isHighlight
                ? (isToolActive ? '关闭高亮工具' : '启用高亮工具')
                : isUnderline
                  ? (isToolActive ? '关闭下划线工具' : '启用下划线工具')
                  : tool.label;
              return (
                <React.Fragment key={tool.label}>
                  <FloatingDockItem
                    label={isAnnotationTool ? annotationToolTitle : (hasSelection ? tool.label : `选择文字后使用${tool.label}`)}
                    active={isToolActive}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (isAnnotationTool) {
                        setActionFeedback(null);
                        setNoteEditorOpen(false);
                        onAnnotationToolChange?.(isToolActive ? null : tool.kind);
                        return;
                      }
                      void runToolAction(tool);
                    }}
                    disabled={!isAnnotationTool && (!hasSelection || Boolean(activeAction))}
                    slotClassName={config.dockSlot}
                    className={`${config.dockButton} ${
                      isToolActive
                        ? (darkMode
                          ? 'bg-[#d97a5d]/20 text-[#ffb09a] shadow-[inset_0_0_0_1px_rgba(255,176,154,0.18)]'
                          : 'bg-[#f7ded5] text-[#a94d34] shadow-[inset_0_0_0_1px_rgba(201,107,80,0.16)]')
                        : tool.color
                    }`}
                    title={isAnnotationTool ? annotationToolTitle : (hasSelection ? tool.label : `选择文字后使用${tool.label}`)}
                    aria-pressed={isAnnotationTool ? isToolActive : undefined}
                    aria-expanded={tool.kind === 'note' ? noteEditorOpen : undefined}
                  >
                    {activeAction === tool.label ? (
                      <Loader2 className={`${config.iconSize} animate-spin`} strokeWidth={2} />
                    ) : (
                      <span className="relative inline-flex">
                        <Icon className={config.iconSize} strokeWidth={2} />
                        {isAnnotationTool && (
                          <span
                            className="absolute -bottom-0.5 left-1/2 h-1.5 w-1.5 -translate-x-1/2 rounded-full ring-1 ring-white"
                            style={{ backgroundColor: currentAnnotationColor }}
                            aria-hidden="true"
                          />
                        )}
                      </span>
                    )}

                  </FloatingDockItem>

                  {isHighlight && (
                    <div
                      className={`mx-1 flex h-8 items-center gap-1 rounded-[10px] border px-1.5 ${
                        darkMode
                          ? 'border-white/[0.08] bg-white/[0.045]'
                          : 'border-[#e3ddd7] bg-white/75 shadow-[inset_0_1px_0_rgba(255,255,255,0.8)]'
                      }`}
                      role="listbox"
                      aria-label="标注颜色"
                    >
                      {DOCUMENT_HIGHLIGHT_COLORS.map((swatch) => {
                        const selected = currentAnnotationColor === swatch.value;
                        return (
                          <button
                            key={swatch.id}
                            type="button"
                            role="option"
                            aria-selected={selected}
                            aria-label={swatch.label}
                            title={`选择${swatch.label}标注颜色`}
                            onClick={(event) => {
                              event.stopPropagation();
                              setActionFeedback(null);
                              onAnnotationColorChange?.(swatch.value);
                            }}
                            className={`relative h-5 w-5 shrink-0 rounded-full transition-[transform,box-shadow,opacity] duration-150 hover:scale-110 active:scale-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${
                              selected
                                ? (darkMode ? 'ring-2 ring-white/80 ring-offset-1 ring-offset-[#25282e]' : 'ring-2 ring-gray-700/80 ring-offset-1 ring-offset-[#f7f5f2]')
                                : 'ring-1 ring-black/10'
                            }`}
                            style={{ backgroundColor: swatch.value }}
                          >
                            <span className="sr-only">{selected ? '当前颜色' : '选择颜色'}</span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </React.Fragment>
              );
            })}

          </div>
          </FloatingDock>

          <AnimatePresence>
            {noteEditorOpen && (
              <div
                ref={noteEditorRef}
                data-testid="selection-note-editor"
                data-placement={noteEditorPlacement}
                className={`absolute left-1/2 z-40 ${panelPlacementClass}`}
                style={{
                  width: `${noteEditorLayout.width}px`,
                  maxWidth: 'calc(100vw - 24px)',
                  marginLeft: `${noteEditorLayout.offsetX}px`,
                  transform: 'translateX(-50%)',
                }}
                onClick={(event) => event.stopPropagation()}
              >
                <motion.div
                  initial={{ opacity: 0, y: -6, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -4, scale: 0.98 }}
                  transition={{ duration: 0.16, ease: 'easeOut' }}
                  className="w-full rounded-[16px] border border-white/70 bg-white/95 p-3.5 shadow-[0_18px_45px_rgba(45,38,34,0.18)] backdrop-blur-xl"
                >
                  <div className="mb-2.5 flex items-center justify-between gap-3">
                    <div className="text-[12px] font-bold text-gray-800">添加划词笔记</div>
                    <button
                      type="button"
                      onClick={() => setNoteEditorOpen(false)}
                      className="inline-flex h-6 w-6 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-700"
                      aria-label="关闭笔记编辑器"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <div className="mb-2.5 line-clamp-2 rounded-[10px] bg-[#f7f4f1] px-3 py-2 text-[11px] leading-relaxed text-gray-500">
                    {selectedText}
                  </div>

                  <Suspense fallback={<div className="h-[104px] animate-pulse rounded-[14px] bg-[#f1ece7]" />}>
                    <MarkdownNoteEditor
                      value={noteDraft}
                      onChange={setNoteDraft}
                      onSubmit={saveNote}
                      onCancel={() => setNoteEditorOpen(false)}
                      maxLength={20000}
                      autoFocus
                    />
                  </Suspense>

                  <div className="mt-3 flex items-center justify-end gap-3">
                    <button
                      type="button"
                      onClick={() => void saveNote()}
                      disabled={!noteDraft.trim() || activeAction === '笔记'}
                      className="inline-flex h-8 items-center gap-1.5 rounded-full bg-[#F0653A] px-4 text-[11px] font-semibold text-white shadow-[0_6px_16px_-8px_rgba(240,101,58,0.65)] transition-[background-color,transform] duration-200 hover:bg-[#F5713F] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none disabled:active:scale-100"
                    >
                      {activeAction === '笔记' && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                      保存笔记
                    </button>
                  </div>
                </motion.div>
              </div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {actionFeedback && !noteEditorOpen && (
              <motion.div
                role="status"
                initial={{ opacity: 0, y: -5, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -4, scale: 0.98 }}
                className={`absolute left-1/2 top-full mt-3 flex -translate-x-1/2 items-center gap-1.5 whitespace-nowrap rounded-[11px] border px-3 py-2 text-[11px] font-semibold shadow-lg backdrop-blur-xl ${
                  actionFeedback.tone === 'error'
                    ? 'border-rose-200 bg-rose-50/95 text-rose-700'
                    : 'border-emerald-200 bg-emerald-50/95 text-emerald-700'
                }`}
              >
                {actionFeedback.tone === 'error' ? (
                  <AlertCircle className="h-3.5 w-3.5" />
                ) : (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                )}
                {actionFeedback.message}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default TextSelectionToolbar;
