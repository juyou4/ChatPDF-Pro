import React, { useEffect, useRef, useState } from 'react';
import { Copy, Highlighter, MessageSquare, Sparkles, Globe, Search, Share2, X, Move } from 'lucide-react';
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
  const [adjustedPosition, setAdjustedPosition] = useState(position);
  const dragState = useRef({ dragging: false, start: { x: 0, y: 0 }, origin: { x: 0, y: 0 } });
  const resizeState = useRef({ resizing: false, start: { x: 0, y: 0 }, originScale: scale });
  const [hoverCorner, setHoverCorner] = useState('');

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
      action: onAddNote,
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
                    tool.action();
                  }}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.03 }}
                  className={`group relative ${config.padding} rounded-xl transition-all hover:bg-[var(--color-bg-subtle)]/80 ${tool.color}`}
                  title={tool.label}
                >
                  <Icon className={config.iconSize} strokeWidth={2} />

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
