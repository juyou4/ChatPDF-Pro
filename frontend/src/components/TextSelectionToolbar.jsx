import React, { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Copy,
  Highlighter,
  Loader2,
  MessageSquare,
  Sparkles,
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

/**
 * 划词交互工具箱
 * 固定在 PDF 页码栏下方。高亮和下划线是持续标注工具，
 * 其余操作仍在选中文字后启用。
 */
// CodeMirror 只在打开笔记编辑器时才需要，懒加载避免拖慢首屏。
const MarkdownNoteEditor = lazy(() => import('./MarkdownNoteEditor'));

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
  const noteEditorRef = useRef(null);
  const feedbackTimerRef = useRef(null);
  const [noteEditorOpen, setNoteEditorOpen] = useState(false);
  const [noteEditorPlacement, setNoteEditorPlacement] = useState('bottom');
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

  useEffect(() => {
    if (!noteEditorOpen) return undefined;
    const updatePlacement = () => {
      const toolbarRect = toolbarRef.current?.getBoundingClientRect();
      if (!toolbarRect) return;
      const panelHeight = noteEditorRef.current?.getBoundingClientRect().height
        || 320;
      const requiredSpace = panelHeight + 16;
      // 工具栏现在长在 PDF 面板里，可用空间以面板为准而不是整个视口，
      // 否则会算出"下面放得下"然后被面板的 overflow 剪掉。
      const bounds = toolbarRef.current?.closest('[data-pdf-reader-surface]')?.getBoundingClientRect();
      const limitBottom = bounds?.bottom ?? window.innerHeight;
      const limitTop = bounds?.top ?? 0;
      const spaceBelow = limitBottom - toolbarRect.bottom;
      const spaceAbove = toolbarRect.top - limitTop;
      setNoteEditorPlacement(spaceBelow >= requiredSpace || spaceBelow >= spaceAbove ? 'bottom' : 'top');
    };
    const frame = window.requestAnimationFrame(updatePlacement);
    window.addEventListener('resize', updatePlacement);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('resize', updatePlacement);
    };
  }, [noteEditorOpen, noteDraft]);

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
      padding: 'p-1.5',
      gap: 'gap-0.5',
      containerPadding: 'px-2 py-1'
    },
    normal: {
      iconSize: 'w-[18px] h-[18px]',
      padding: 'p-2',
      gap: 'gap-0.5',
      containerPadding: 'px-2.5 py-1.5'
    },
    large: {
      iconSize: 'w-5 h-5',
      padding: 'p-2.5',
      gap: 'gap-1',
      containerPadding: 'px-3 py-2'
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
      icon: MessageSquare,
      label: '笔记',
      kind: 'note',
      color: 'text-purple-600 hover:text-purple-700'
    },
    ...(canDeleteAnnotation ? [{
      icon: Trash2,
      label: '删除标注',
      action: onDeleteAnnotation,
      color: 'text-rose-500 hover:text-rose-700',
    }] : []),
    {
      icon: Sparkles,
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
        {/* 工具栏容器：两端圆润的胶囊，只占内容宽度，不再糊满整条 */}
        <div className={`relative max-w-full rounded-full border px-1.5 shadow-[0_1px_2px_rgba(83,65,55,0.04),0_10px_24px_-14px_rgba(83,65,55,0.28)] transition-colors duration-200 ${
          darkMode
            ? 'border-white/[0.09] bg-[#22262c]'
            : 'border-[#ebe4dd] bg-white'
        }`}>
          {/* 工具按钮组 */}
          <div className={`${config.containerPadding} flex items-center ${config.gap}`}>
            {tools.map((tool, index) => {
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
                  <motion.button
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
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.03 }}
                    className={`group relative ${config.padding} rounded-xl transition-all disabled:cursor-not-allowed disabled:opacity-35 ${
                      isToolActive
                        ? (darkMode
                          ? 'bg-[#d97a5d]/20 text-[#ffb09a] shadow-[inset_0_0_0_1px_rgba(255,176,154,0.18)]'
                          : 'bg-[#f7ded5] text-[#a94d34] shadow-[inset_0_0_0_1px_rgba(201,107,80,0.16)]')
                        : `hover:bg-[var(--color-bg-subtle)]/80 ${tool.color}`
                    }`}
                    title={isAnnotationTool ? annotationToolTitle : (hasSelection ? tool.label : `选择文字后使用${tool.label}`)}
                    aria-label={tool.label}
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

                    {/* 提示气泡朝下弹：工具栏紧贴在阅读器工具栏(z-30)下方，
                        往上弹会被压在它后面，正好是看不见字的原因。下方是 PDF 区，空间充足。 */}
                    <div className="absolute top-full left-1/2 z-10 -translate-x-1/2 mt-2 px-2 py-1 bg-gray-900 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 -mb-1 w-0 h-0 border-l-4 border-r-4 border-b-4 border-transparent border-b-gray-900" />
                      {isAnnotationTool ? annotationToolTitle : tool.label}
                    </div>
                  </motion.button>

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

          <AnimatePresence>
            {noteEditorOpen && (
              <motion.div
                ref={noteEditorRef}
                initial={{ opacity: 0, y: -6, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -4, scale: 0.98 }}
                transition={{ duration: 0.16, ease: 'easeOut' }}
                data-placement={noteEditorPlacement}
                className={`absolute left-1/2 w-[min(620px,calc(100vw-24px))] -translate-x-1/2 rounded-[16px] border border-white/70 bg-white/95 p-3.5 shadow-[0_18px_45px_rgba(45,38,34,0.18)] backdrop-blur-xl ${panelPlacementClass}`}
                onClick={(event) => event.stopPropagation()}
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
