import React, { useCallback, useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
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
 * 截图预览（支持最多 9 张）
 *
 * 收起态：纸张式堆叠缩略图从输入卡顶部探出 + 数量胶囊；
 * 悬浮/点击展开：横排缩略图可逐张删除，附快捷操作。
 *
 * @param {Array} screenshots - 截图数组 [{id, dataUrl}]
 * @param {function} onAction - 快捷操作回调 (actionKey, screenshotId)
 * @param {function} onClose - 删除单张（传 id）或清空全部（传 null）
 */
function ScreenshotPreview({ screenshots = [], onAction, onClose }) {
  const [expanded, setExpanded] = useState(false)
  // 收起防抖：展开/收起切换会引起布局变化，瞬时的 mouseleave 不应立即收起
  const collapseTimerRef = useRef(null)
  const rootRef = useRef(null)
  const stackButtonRef = useRef(null)
  const restoreFocusRef = useRef(false)
  const prevCountRef = useRef(0)
  // 提问/新截图后先锁住悬浮展开，避免布局一变又被 mouseenter 撑开
  const ignoreHoverUntilLeaveRef = useRef(false)

  const openPanel = useCallback((options = {}) => {
    const force = options === true || options.force === true
    if (!force && ignoreHoverUntilLeaveRef.current) return
    clearTimeout(collapseTimerRef.current)
    restoreFocusRef.current = false
    ignoreHoverUntilLeaveRef.current = false
    setExpanded(true)
  }, [])

  const collapsePanel = useCallback((lockHover = false) => {
    clearTimeout(collapseTimerRef.current)
    if (lockHover) ignoreHoverUntilLeaveRef.current = true
    setExpanded(false)
  }, [])

  const scheduleCollapse = useCallback(() => {
    clearTimeout(collapseTimerRef.current)
    collapseTimerRef.current = setTimeout(() => {
      if (rootRef.current?.contains(document.activeElement)) return
      setExpanded(false)
    }, 150)
  }, [])

  useEffect(() => () => clearTimeout(collapseTimerRef.current), [])

  useEffect(() => {
    const prevCount = prevCountRef.current
    prevCountRef.current = screenshots.length
    if (screenshots.length === 0) {
      ignoreHoverUntilLeaveRef.current = false
      setExpanded(false)
      return
    }
    if (screenshots.length > prevCount) {
      ignoreHoverUntilLeaveRef.current = true
      setExpanded(false)
    }
  }, [screenshots.length])

  useEffect(() => {
    if (expanded || !restoreFocusRef.current) return
    restoreFocusRef.current = false
    stackButtonRef.current?.focus()
  }, [expanded])

  const closePanel = useCallback((restoreFocus = false) => {
    clearTimeout(collapseTimerRef.current)
    restoreFocusRef.current = restoreFocus
    ignoreHoverUntilLeaveRef.current = true
    setExpanded(false)
  }, [])

  const handleKeyDown = useCallback((event) => {
    if (!expanded || event.key !== 'Escape') return
    event.preventDefault()
    event.stopPropagation()
    closePanel(true)
  }, [closePanel, expanded])

  const handleBlur = useCallback((event) => {
    if (event.currentTarget.contains(event.relatedTarget)) return
    scheduleCollapse()
  }, [scheduleCollapse])

  const handleAction = useCallback(
    (actionKey, screenshotId) => {
      collapsePanel(true)
      onAction?.(actionKey, screenshotId)
    },
    [collapsePanel, onAction]
  )

  const handleMouseLeave = useCallback(() => {
    ignoreHoverUntilLeaveRef.current = false
    scheduleCollapse()
  }, [scheduleCollapse])

  if (!screenshots || screenshots.length === 0) return null

  // 收起态最多露出 4 张，取最新的几张
  const stack = screenshots.slice(-4)

  return (
    <div
      ref={rootRef}
      data-testid="screenshot-preview"
      className="relative mx-1 mb-2 -mt-4"
      onMouseEnter={() => openPanel()}
      onMouseLeave={handleMouseLeave}
      onKeyDown={handleKeyDown}
      onBlur={handleBlur}
    >
      {!expanded ? (
          /* ── 收起态：堆叠探出卡顶 ───────────────────────── */
          <button
            ref={stackButtonRef}
            key="stack"
            type="button"
            aria-label={`${screenshots.length} 张截图，悬浮或点击展开管理`}
            aria-expanded={false}
            onClick={() => openPanel({ force: true })}
            className="flex cursor-pointer select-none items-end gap-2.5 rounded-[10px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D99178]/45 focus-visible:ring-offset-2"
          >
            {/* 堆叠完整落在容器盒内：悬浮判定不会因布局切换而丢失 */}
            <div className="relative h-14 w-[84px] shrink-0">
              {stack.map((s, i) => {
                const centered = i - (stack.length - 1) / 2
                return (
                  <motion.img
                    key={s.id}
                    layoutId={`shot-${s.id}`}
                    src={s.dataUrl}
                    alt="截图"
                    className="absolute bottom-0 h-14 w-12 rounded-[6px] border-2 border-white bg-white object-cover shadow-[0_4px_12px_rgba(30,30,35,0.18)]"
                    style={{
                      left: i * 12,
                      rotate: `${centered * 6}deg`,
                      transformOrigin: 'bottom center',
                      zIndex: i,
                    }}
                  />
                )
              })}
            </div>
            <span className="mb-1 inline-flex items-center rounded-full bg-gray-100 px-2.5 py-1 text-[11px] font-semibold text-gray-500">
              {screenshots.length} 张截图
            </span>
          </button>
        ) : (
          /* ── 展开态：横排缩略图，可逐张删除 ─────────────── */
          <motion.div
            key="expanded"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.14, ease: [0.22, 1, 0.36, 1] }}
            className="rounded-[16px] bg-gray-50/80 p-2.5"
          >
            <div className="flex items-center justify-between px-1 pb-2">
              <span className="text-[11px] font-semibold text-gray-400">
                截图附件 {screenshots.length}/9
              </span>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => onClose?.(null)}
                  className="flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] text-gray-400 transition-colors hover:bg-red-50 hover:text-red-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-300/60"
                >
                  <Trash2 className="h-3 w-3" />
                  清空全部
                </button>
                <button
                  type="button"
                  onClick={() => closePanel(true)}
                  aria-label="收起截图管理"
                  title="收起"
                  className="rounded-full p-1 text-gray-400 transition-colors hover:bg-gray-200/70 hover:text-gray-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D99178]/45"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              {screenshots.map((s) => (
                <div key={s.id} className="group relative">
                  <motion.img
                    layoutId={`shot-${s.id}`}
                    src={s.dataUrl}
                    alt="截图"
                    className="h-14 w-12 rounded-[6px] border border-gray-200 bg-white object-cover shadow-sm transition-transform group-hover:scale-[1.04]"
                  />
                  <button
                    type="button"
                    onClick={() => onClose?.(s.id)}
                    aria-label="删除这张截图"
                    className="absolute -right-1.5 -top-1.5 rounded-full bg-gray-900/85 p-0.5 text-white opacity-0 shadow-sm transition-opacity hover:bg-red-500 group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D99178]/55 focus-visible:ring-offset-1"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>

            {/* 针对最新一张截图的快捷操作 */}
            <div className="mt-2.5 flex flex-wrap gap-1.5 border-t border-gray-200/60 pt-2">
              {PRIMARY_ACTION_KEYS.map((key) => {
                const action = SCREENSHOT_ACTIONS[key]
                const IconComponent = ICON_MAP[action.icon]
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => handleAction(key, screenshots[screenshots.length - 1].id)}
                    className="inline-flex items-center gap-1.5 rounded-full border border-gray-200 bg-white px-2.5 py-1 text-[11px] font-medium text-gray-600 transition-all duration-150 hover:border-[#FFA07A] hover:bg-[#FFF4EF] hover:text-[#B85F47]"
                    title={action.label}
                  >
                    {IconComponent && <IconComponent className="h-3 w-3" />}
                    <span>{action.label}</span>
                  </button>
                )
              })}
            </div>
          </motion.div>
        )}
    </div>
  )
}

export default ScreenshotPreview
