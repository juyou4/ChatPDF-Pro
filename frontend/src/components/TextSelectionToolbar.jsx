import React, { useEffect, useRef, useState } from 'react';
import { AlertCircle, CheckCircle2, Copy, Highlighter, Loader2, MessageSquare, Sparkles, Globe, Search, Share2, X, Move } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * 划词交互工具箱
 * 当用户选中 PDF 文本时弹出的功能菜单
 */
const TextSelectionToolbar = ({
  selectedText,
  position,
  onClose,
  onCopy,
  onHighlight,
  onAddNote,
  onAIExplain,
  onTranslate,
  onWebSearch,
  onShare,
  size = 'normal', // 新增：支持 'compact', 'normal', 'large'
  onPositionChange,
  scale = 1,
  onScaleChange
}) => {
  const toolbarRef = useRef(null);
  const noteEditorRef = useRef(null);
  const [adjustedPosition, setAdjustedPosition] = useState(position);
  const dragState = useRef({ dragging: false, start: { x: 0, y: 0 }, origin: { x: 0, y: 0 } });
  const resizeState = useRef({ resizing: false, start: { x: 0, y: 0 }, originScale: scale });
  const feedbackTimerRef = useRef(null);
  const [hoverCorner, setHoverCorner] = useState('');
  const [noteEditorOpen, setNoteEditorOpen] = useState(false);
  const [noteEditorPlacement, setNoteEditorPlacement] = useState('bottom');
  const [noteDraft, setNoteDraft] = useState('');
  const [activeAction, setActiveAction] = useState('');
  const [actionFeedback, setActionFeedback] = useState(null);

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
      const editorHeight = noteEditorRef.current?.getBoundingClientRect().height || 238;
      const requiredSpace = editorHeight + 16;
      const spaceBelow = window.innerHeight - toolbarRect.bottom;
      const spaceAbove = toolbarRect.top;
      setNoteEditorPlacement(spaceBelow >= requiredSpace || spaceBelow >= spaceAbove ? 'bottom' : 'top');
    };
    const frame = window.requestAnimationFrame(updatePlacement);
    window.addEventListener('resize', updatePlacement);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('resize', updatePlacement);
    };
  }, [adjustedPosition, noteEditorOpen, scale]);

  const showFeedback = (message, tone = 'success') => {
    if (!message) return;
    if (feedbackTimerRef.current) window.clearTimeout(feedbackTimerRef.current);
    setActionFeedback({ message, tone });
    feedbackTimerRef.current = window.setTimeout(() => setActionFeedback(null), 2400);
  };

  const runToolAction = async (tool) => {
    if (tool.kind === 'note') {
      setActionFeedback(null);
      setNoteEditorOpen((open) => !open);
      return;
    }
    if (typeof tool.action !== 'function' || activeAction) return;
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
    const value = noteDraft.trim();
    if (!value) {
      showFeedback('请输入笔记内容', 'error');
      return;
    }
    if (typeof onAddNote !== 'function' || activeAction) return;
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

  // 智能定位：避免超出屏幕边界
  useEffect(() => {
    if (!toolbarRef.current) return;

    const toolbar = toolbarRef.current;
    const rect = toolbar.getBoundingClientRect();
    let { x, y } = position;

    // 检查右边界
    if (rect.right > window.innerWidth - 20) {
      x = window.innerWidth - rect.width / 2 - 20;
    }

    // 检查左边界
    if (rect.left < 20) {
      x = rect.width / 2 + 20;
    }

    // 检查顶部边界
    if (rect.top < 70) {
      y = position.y + rect.height + 60; // 显示在选中文本下方
    }

    setAdjustedPosition({ x, y });
  }, [position]);

  // 同步外部位置变化
  useEffect(() => {
    setAdjustedPosition(position);
  }, [position]);

  if (!selectedText) return null;

  // 根据大小配置样式
  const sizeConfig = {
    compact: {
      iconSize: 'w-4 h-4',
      padding: 'p-2',
      gap: 'gap-0.5',
      containerPadding: 'px-2 py-2'
    },
    normal: {
      iconSize: 'w-5 h-5',
      padding: 'p-2.5',
      gap: 'gap-1',
      containerPadding: 'px-3 py-2.5'
    },
    large: {
      iconSize: 'w-6 h-6',
      padding: 'p-3',
      gap: 'gap-2',
      containerPadding: 'px-4 py-3'
    }
  };

  const config = sizeConfig[size] || sizeConfig.normal;

  const tools = [
    {
      icon: Copy,
      label: '复制',
      action: onCopy,
      color: 'text-gray-600 hover:text-gray-900'
    },
    {
      icon: Highlighter,
      label: '高亮',
      action: onHighlight,
      color: 'text-yellow-600 hover:text-yellow-700'
    },
    {
      icon: MessageSquare,
      label: '笔记',
      kind: 'note',
      color: 'text-purple-600 hover:text-purple-700'
    },
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

  return (
    <AnimatePresence>
      <motion.div
        ref={toolbarRef}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 10 }}
        transition={{ duration: 0.15, ease: 'easeOut' }}
        className="fixed z-50"
        style={{
          left: `${adjustedPosition.x}px`,
          top: `${adjustedPosition.y}px`,
          x: "-50%",
          scale: scale,
          transformOrigin: 'top center'
        }}
      >
        {/* 工具栏容器 */}
        <div className="relative">
          {/* 三角箭头 */}
          <div className="absolute left-1/2 -translate-x-1/2 -bottom-2 w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-white/90 drop-shadow-lg" />

          {/* 工具按钮组 */}
          <div className={`soft-panel backdrop-blur-xl rounded-2xl shadow-2xl border border-white/40 ${config.containerPadding} flex items-center ${config.gap}`}>
            {/* 拖动手柄 */}
            <motion.button
              onMouseDown={(e) => {
                e.preventDefault();
                e.stopPropagation();
                dragState.current = {
                  dragging: true,
                  start: { x: e.clientX, y: e.clientY },
                  origin: { ...adjustedPosition }
                };
                const handleMove = (ev) => {
                  if (!dragState.current.dragging) return;
                  const dx = ev.clientX - dragState.current.start.x;
                  const dy = ev.clientY - dragState.current.start.y;
                  const nextPos = {
                    x: dragState.current.origin.x + dx,
                    y: dragState.current.origin.y + dy
                  };
                  setAdjustedPosition(nextPos);
                  onPositionChange?.(nextPos);
                };
                const handleUp = () => {
                  dragState.current.dragging = false;
                  window.removeEventListener('mousemove', handleMove);
                  window.removeEventListener('mouseup', handleUp);
                };
                window.addEventListener('mousemove', handleMove);
                window.addEventListener('mouseup', handleUp);
              }}
              whileHover={{ scale: 1.05 }}
              className={`${config.padding} mr-1 rounded-xl text-gray-500 hover:text-gray-800 hover:bg-[var(--color-bg-subtle)]/80 cursor-move`}
              title="拖动移动工具栏"
            >
              <Move className={config.iconSize} strokeWidth={2} />
            </motion.button>

            {tools.map((tool, index) => {
              const Icon = tool.icon;
              return (
                <motion.button
                  key={tool.label}
                  onClick={(e) => {
                    e.stopPropagation();
                    void runToolAction(tool);
                  }}
                  disabled={Boolean(activeAction)}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.03 }}
                  className={`group relative ${config.padding} rounded-xl transition-all hover:bg-[var(--color-bg-subtle)]/80 disabled:cursor-wait disabled:opacity-55 ${tool.color}`}
                  title={tool.label}
                  aria-label={tool.label}
                >
                  {activeAction === tool.label ? (
                    <Loader2 className={`${config.iconSize} animate-spin`} strokeWidth={2} />
                  ) : (
                    <Icon className={config.iconSize} strokeWidth={2} />
                  )}

                  {/* Tooltip */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                    {tool.label}
                    <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900" />
                  </div>
                </motion.button>
              );
            })}

            {/* 分隔线 */}
            <div className="w-px h-6 bg-gray-200 mx-1" />

            {/* 关闭按钮 */}
            <motion.button
              onClick={(e) => {
                e.stopPropagation();
                onClose();
              }}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: tools.length * 0.03 }}
              className={`${config.padding} rounded-xl transition-all hover:bg-red-50/80 text-gray-400 hover:text-red-600`}
              title="关闭"
            >
              <X className={config.iconSize} strokeWidth={2} />
            </motion.button>
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
                className={`absolute left-1/2 w-[min(320px,calc(100vw-32px))] -translate-x-1/2 rounded-[16px] border border-white/70 bg-white/95 p-3.5 shadow-[0_18px_45px_rgba(45,38,34,0.18)] backdrop-blur-xl ${
                  noteEditorPlacement === 'top'
                    ? 'bottom-full mb-3 origin-bottom'
                    : 'top-full mt-3 origin-top'
                }`}
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
                <textarea
                  autoFocus
                  value={noteDraft}
                  onChange={(event) => setNoteDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Escape') {
                      event.preventDefault();
                      setNoteEditorOpen(false);
                      return;
                    }
                    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                      event.preventDefault();
                      void saveNote();
                    }
                  }}
                  rows={3}
                  maxLength={2000}
                  placeholder="记录你的理解、疑问或待核实内容"
                  className="w-full resize-none rounded-[11px] border border-gray-200 bg-white px-3 py-2.5 text-[12px] leading-relaxed text-gray-800 outline-none transition-shadow placeholder:text-gray-400 focus:border-[#e9aa94] focus:ring-4 focus:ring-[#ed8c68]/10"
                />
                <div className="mt-2.5 flex items-center justify-between gap-3">
                  <span className="text-[10px] tabular-nums text-gray-400">{noteDraft.length}/2000</span>
                  <button
                    type="button"
                    onClick={() => void saveNote()}
                    disabled={!noteDraft.trim() || activeAction === '笔记'}
                    className="inline-flex h-8 items-center gap-1.5 rounded-[10px] bg-[#D97A5D] px-3.5 text-[11px] font-bold text-white shadow-[0_7px_16px_-7px_rgba(184,95,71,0.62)] transition-all hover:bg-[#c96b50] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-45"
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

        {/* 四角缩放感应区：靠近角落时显示提示，可拖动缩放 */}
        {['top-left', 'top-right', 'bottom-left', 'bottom-right'].map((corner) => {
          const isTop = corner.includes('top');
          const isLeft = corner.includes('left');
          const cursor =
            corner === 'top-left'
              ? 'nwse-resize'
              : corner === 'bottom-right'
                ? 'nwse-resize'
                : 'nesw-resize';

          const handleResizeStart = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const initialScale = scale; // 使用当前 scale prop 而不是 state
            resizeState.current = {
              resizing: true,
              start: { x: e.clientX, y: e.clientY },
              originScale: initialScale
            };

            const handleResize = (ev) => {
              if (!resizeState.current.resizing) return;
              const dx = ev.clientX - resizeState.current.start.x;
              const dy = ev.clientY - resizeState.current.start.y;

              // 水平：左上/左下向左拖变大，右侧向右拖变大
              const horizontal = isLeft ? -dx : dx;
              // 垂直：上侧向上拖变大，下侧向下拖变大
              const vertical = isTop ? -dy : dy;

              const delta = (horizontal + vertical) / 200; // 调整敏感度
              const nextScale = Math.min(1.6, Math.max(0.7, resizeState.current.originScale + delta));
              onScaleChange?.(nextScale);
            };
            const handleResizeEnd = () => {
              resizeState.current.resizing = false;
              window.removeEventListener('mousemove', handleResize);
              window.removeEventListener('mouseup', handleResizeEnd);
            };
            window.addEventListener('mousemove', handleResize);
            window.addEventListener('mouseup', handleResizeEnd);
          };

          return (
            <div
              key={corner}
              className="absolute w-6 h-6"
              style={{
                top: isTop ? -2 : 'auto',
                bottom: isTop ? 'auto' : -2,
                left: isLeft ? -2 : 'auto',
                right: isLeft ? 'auto' : -2,
                cursor
              }}
              onMouseEnter={() => setHoverCorner(corner)}
              onMouseLeave={() => setHoverCorner('')}
              onMouseDown={handleResizeStart}
              aria-label={`resize-${corner}`}
            >
              {hoverCorner === corner && (
                <div
                  className="absolute pointer-events-none"
                  style={{
                    width: 8,
                    height: 8,
                    borderRight: isLeft ? '0' : '2px solid #d1d5db',
                    borderBottom: isTop ? '0' : '2px solid #d1d5db',
                    borderLeft: isLeft ? '2px solid #d1d5db' : '0',
                    borderTop: isTop ? '2px solid #d1d5db' : '0',
                    right: isLeft ? 'auto' : 2,
                    left: isLeft ? 2 : 'auto',
                    bottom: isTop ? 'auto' : 2,
                    top: isTop ? 2 : 'auto'
                  }}
                />
              )}
            </div>
          );
        })}
      </motion.div>
    </AnimatePresence>
  );
};

export default TextSelectionToolbar;
