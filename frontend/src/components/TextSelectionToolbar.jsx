import React from 'react';
import { Copy, Highlighter, MessageSquare, Sparkles, Globe, Search, Share2, X } from 'lucide-react';
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
  onShare
}) => {
  if (!selectedText) return null;

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
      color: 'text-blue-600 hover:text-blue-700'
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
        initial={{ opacity: 0, scale: 0.9, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 10 }}
        transition={{ duration: 0.15, ease: 'easeOut' }}
        className="fixed z-50"
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
          transform: 'translateX(-50%)'
        }}
      >
        {/* 工具栏容器 */}
        <div className="relative">
          {/* 三角箭头 */}
          <div className="absolute left-1/2 -translate-x-1/2 -bottom-2 w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-white/90 drop-shadow-lg" />
          
          {/* 工具按钮组 */}
          <div className="bg-white/90 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/40 px-3 py-2.5 flex items-center gap-1">
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
                  className={`group relative p-2.5 rounded-xl transition-all hover:bg-gray-50/80 ${tool.color}`}
                  title={tool.label}
                >
                  <Icon className="w-5 h-5" strokeWidth={2} />
                  
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
              className="p-2.5 rounded-xl transition-all hover:bg-red-50/80 text-gray-400 hover:text-red-600"
              title="关闭"
            >
              <X className="w-5 h-5" strokeWidth={2} />
            </motion.button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default TextSelectionToolbar;
