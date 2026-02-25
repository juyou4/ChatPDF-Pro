import React, { useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  MessageSquarePlus,
  BookOpen,
  Table,
  Sigma,
  ScanText,
  Languages,
  Copy,
  X,
  Trash2,
} from 'lucide-react'
import { SCREENSHOT_ACTIONS } from '../utils/screenshotUtils'

/**
 * 图标名称到 Lucide 组件的映射
 */
const ICON_MAP = {
  MessageSquarePlus,
  BookOpen,
  Table,
  Sigma,
  ScanText,
  Languages,
  Copy,
}

/**
 * 主要操作按钮的 key 列表
 */
const PRIMARY_ACTION_KEYS = ['ask', 'explain', 'table', 'formula', 'ocr', 'translate']

/**
 * 截图预览列表组件（支持最多 9 张）
 *
 * @param {Array} screenshots - 截图数组 [{id, dataUrl}]
 * @param {function} onAction - 快捷操作回调 (actionKey, screenshotId)
 * @param {function} onClose - 删除单张或清空全部的回调 (id)
 */
function ScreenshotPreview({ screenshots = [], onAction, onClose }) {
  // 处理快捷操作按钮点击
  const handleAction = useCallback(
    (actionKey, screenshotId) => {
      onAction?.(actionKey, screenshotId)
    },
    [onAction]
  )

  if (!screenshots || screenshots.length === 0) return null

  return (
    <AnimatePresence>
      <motion.div
        data-testid="screenshot-preview"
        initial={{ opacity: 0, y: 16, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 16, scale: 0.96 }}
        transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
        className="mx-4 mb-3 p-3 bg-white/90 backdrop-blur-md rounded-2xl shadow-lg border border-gray-100/80 ring-1 ring-black/5"
      >
        <div className="flex flex-col gap-3">
          {/* 标题与清空按钮 */}
          <div className="flex items-center justify-between px-1">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">
              截图附件 ({screenshots.length}/9)
            </span>
            <button
              onClick={() => onClose?.(null)}
              className="text-[11px] text-gray-400 hover:text-red-500 transition-colors flex items-center gap-1"
            >
              <Trash2 className="w-3 h-3" />
              清空全部
            </button>
          </div>

          {/* 缩略图网格 */}
          <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-2">
            {screenshots.map((s) => (
              <div key={s.id} className="relative group aspect-square">
                <img
                  src={s.dataUrl}
                  alt="截图"
                  className="w-full h-full object-cover rounded-lg border border-gray-200 shadow-sm transition-transform group-hover:scale-[1.02]"
                />
                {/* 悬浮删除按钮 */}
                <button
                  onClick={() => onClose?.(s.id)}
                  className="absolute -top-1.5 -right-1.5 p-0.5 bg-red-500 text-white rounded-full 
                    opacity-0 group-hover:opacity-100 transition-opacity shadow-sm hover:bg-red-600"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>

          {/* 针对最后一张图的快捷操作 (如果有的话) */}
          {screenshots.length > 0 && (
            <div className="flex items-start gap-3 pt-2 border-t border-gray-100/50">
              <div className="flex-1 flex flex-col gap-2 min-w-0">
                <div className="flex flex-wrap gap-1.5">
                  {PRIMARY_ACTION_KEYS.map((key) => {
                    const action = SCREENSHOT_ACTIONS[key]
                    const IconComponent = ICON_MAP[action.icon]
                    return (
                      <button
                        key={key}
                        onClick={() => handleAction(key, screenshots[screenshots.length - 1].id)}
                        className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-medium
                          text-gray-700 bg-gray-50 hover:bg-purple-50 hover:text-purple-600
                          border border-gray-200/80 hover:border-purple-200
                          rounded-full transition-all duration-150"
                        title={action.label}
                      >
                        {IconComponent && <IconComponent className="w-3 h-3" />}
                        <span>{action.label}</span>
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

export default ScreenshotPreview
