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
} from 'lucide-react'
import { SCREENSHOT_ACTIONS } from '../utils/screenshotUtils'

/**
 * 图标名称到 Lucide 组件的映射
 *
 * 将 SCREENSHOT_ACTIONS 配置中的字符串图标名称
 * 映射为实际的 Lucide React 图标组件。
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
 * 主要操作按钮的 key 列表（按显示顺序排列）
 * 不包含 'copy'，因为它是次要操作，单独渲染
 */
const PRIMARY_ACTION_KEYS = ['ask', 'explain', 'table', 'formula', 'ocr', 'translate']

/**
 * 截图预览与快捷操作组件
 *
 * 在聊天输入框上方显示截图缩略图和快捷操作按钮。
 * 使用 framer-motion 实现进入/退出动画。
 *
 * @param {string} screenshotData - base64 图片数据（data URL 格式）
 * @param {function} onAction - 快捷操作回调，参数为 action 类型字符串
 *   'ask' | 'explain' | 'table' | 'formula' | 'ocr' | 'translate' | 'copy'
 * @param {function} onClose - 关闭预览回调
 */
function ScreenshotPreview({ screenshotData, onAction, onClose }) {
  // 处理快捷操作按钮点击
  const handleAction = useCallback(
    (actionKey) => {
      onAction?.(actionKey)
    },
    [onAction]
  )

  // 无截图数据时不渲染
  if (!screenshotData) return null

  return (
    <AnimatePresence>
      {screenshotData && (
        <motion.div
          data-testid="screenshot-preview"
          initial={{ opacity: 0, y: 16, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 16, scale: 0.96 }}
          transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
          className="mx-4 mb-3 p-3 bg-white/90 backdrop-blur-md rounded-2xl shadow-lg border border-gray-100/80 ring-1 ring-black/5"
        >
          {/* 上方区域：缩略图 + 关闭按钮 */}
          <div className="flex items-start gap-3">
            {/* 截图缩略图 */}
            <div className="relative flex-shrink-0">
              <img
                src={screenshotData}
                alt="截图预览"
                data-testid="screenshot-thumbnail"
                className="rounded-lg border border-gray-200/60 shadow-sm object-contain"
                style={{ maxHeight: '120px', maxWidth: '200px' }}
              />
            </div>

            {/* 快捷操作按钮区域 */}
            <div className="flex-1 flex flex-col gap-2 min-w-0">
              {/* 主要操作按钮行 */}
              <div className="flex flex-wrap gap-1.5">
                {PRIMARY_ACTION_KEYS.map((key) => {
                  const action = SCREENSHOT_ACTIONS[key]
                  const IconComponent = ICON_MAP[action.icon]
                  return (
                    <button
                      key={key}
                      data-testid={`action-${key}`}
                      onClick={() => handleAction(key)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium
                        text-gray-700 bg-gray-50 hover:bg-blue-50 hover:text-blue-600
                        border border-gray-200/80 hover:border-blue-200
                        rounded-full transition-all duration-150 shadow-sm hover:shadow"
                      title={action.label}
                    >
                      {IconComponent && <IconComponent className="w-3.5 h-3.5" />}
                      <span>{action.label}</span>
                    </button>
                  )
                })}
              </div>

              {/* 次要操作：复制按钮（样式较小） */}
              <div className="flex items-center gap-2">
                <button
                  data-testid="action-copy"
                  onClick={() => handleAction('copy')}
                  className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium
                    text-gray-400 hover:text-gray-600 bg-transparent hover:bg-gray-50
                    rounded-md transition-all duration-150"
                  title={SCREENSHOT_ACTIONS.copy.label}
                >
                  <Copy className="w-3 h-3" />
                  <span>{SCREENSHOT_ACTIONS.copy.label}</span>
                </button>
              </div>
            </div>

            {/* 关闭按钮 */}
            <button
              data-testid="close-button"
              onClick={onClose}
              className="flex-shrink-0 p-1 text-gray-400 hover:text-gray-600
                hover:bg-gray-100 rounded-full transition-colors duration-150"
              title="关闭预览"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default ScreenshotPreview
