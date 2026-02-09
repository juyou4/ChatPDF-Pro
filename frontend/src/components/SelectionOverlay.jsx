import React, { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { isValidSelection } from '../utils/screenshotUtils'

/**
 * PDF 区域框选遮罩组件
 *
 * 叠加在 PDFViewer 的页面渲染区域上方，提供拖拽框选交互。
 * 框选时显示十字准星光标、暗色半透明遮罩和透明选区矩形。
 *
 * @param {boolean} active - 是否处于框选模式
 * @param {function} onCapture - 框选完成回调，参数为 { x, y, width, height }（相对于容器的 CSS 像素坐标）
 * @param {function} onCancel - 取消框选回调（Escape 键触发）
 */
function SelectionOverlay({ active, onCapture, onCancel }) {
  // 是否正在拖拽
  const [isDragging, setIsDragging] = useState(false)
  // 鼠标按下时的起始点
  const [startPoint, setStartPoint] = useState(null)
  // 鼠标当前位置
  const [currentPoint, setCurrentPoint] = useState(null)
  // 遮罩容器的 ref
  const overlayRef = useRef(null)

  // 计算选区矩形（支持任意方向拖拽）
  const getSelectionRect = useCallback(() => {
    if (!startPoint || !currentPoint) return null
    const left = Math.min(startPoint.x, currentPoint.x)
    const top = Math.min(startPoint.y, currentPoint.y)
    const width = Math.abs(currentPoint.x - startPoint.x)
    const height = Math.abs(currentPoint.y - startPoint.y)
    return { left, top, width, height }
  }, [startPoint, currentPoint])

  // 鼠标按下：开始拖拽
  const handleMouseDown = useCallback((e) => {
    // 仅响应左键
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()

    const rect = overlayRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    setStartPoint({ x, y })
    setCurrentPoint({ x, y })
    setIsDragging(true)
  }, [])

  // 鼠标移动：更新当前位置
  const handleMouseMove = useCallback((e) => {
    if (!isDragging) return
    e.preventDefault()

    const rect = overlayRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    setCurrentPoint({ x, y })
  }, [isDragging])

  // 鼠标释放：完成框选
  const handleMouseUp = useCallback((e) => {
    if (!isDragging || !startPoint || !currentPoint) return
    e.preventDefault()

    setIsDragging(false)

    const selectionRect = {
      x: Math.min(startPoint.x, currentPoint.x),
      y: Math.min(startPoint.y, currentPoint.y),
      width: Math.abs(currentPoint.x - startPoint.x),
      height: Math.abs(currentPoint.y - startPoint.y),
    }

    // 检查选区是否满足最小尺寸要求（宽高均 >= 10px）
    if (isValidSelection(selectionRect)) {
      onCapture?.(selectionRect)
    }

    // 无论是否有效，都重置选区状态
    setStartPoint(null)
    setCurrentPoint(null)
  }, [isDragging, startPoint, currentPoint, onCapture])

  // Escape 键退出框选模式
  useEffect(() => {
    if (!active) return

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        // 重置内部状态
        setIsDragging(false)
        setStartPoint(null)
        setCurrentPoint(null)
        onCancel?.()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [active, onCancel])

  // 非激活状态时重置内部状态
  useEffect(() => {
    if (!active) {
      setIsDragging(false)
      setStartPoint(null)
      setCurrentPoint(null)
    }
  }, [active])

  // 当前选区矩形
  const selectionRect = getSelectionRect()

  // 生成 box-shadow 遮罩样式：选区透明，四周暗色半透明
  const getOverlayBoxShadow = () => {
    if (!selectionRect) return 'none'
    // 使用超大 spread 的 box-shadow 实现选区外暗色遮罩
    return '0 0 0 9999px rgba(0, 0, 0, 0.45)'
  }

  return (
    <div
      ref={overlayRef}
      data-html2canvas-ignore
      data-testid="selection-overlay"
      onMouseDown={active ? handleMouseDown : undefined}
      onMouseMove={active ? handleMouseMove : undefined}
      onMouseUp={active ? handleMouseUp : undefined}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 20,
        cursor: active ? 'crosshair' : 'default',
        pointerEvents: active ? 'auto' : 'none',
        // 非激活且无选区时完全透明
        backgroundColor: active && !selectionRect ? 'rgba(0, 0, 0, 0.15)' : 'transparent',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
    >
      {/* 选区矩形 + 暗色遮罩动画 */}
      <AnimatePresence>
        {isDragging && selectionRect && selectionRect.width > 0 && selectionRect.height > 0 && (
          <motion.div
            data-testid="selection-rect"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.1 }}
            style={{
              position: 'absolute',
              left: selectionRect.left,
              top: selectionRect.top,
              width: selectionRect.width,
              height: selectionRect.height,
              // 选区本身透明，box-shadow 形成四周暗色遮罩
              boxShadow: getOverlayBoxShadow(),
              border: '2px dashed rgba(59, 130, 246, 0.8)',
              borderRadius: '2px',
              backgroundColor: 'transparent',
              pointerEvents: 'none',
            }}
          >
            {/* 选区尺寸提示标签 */}
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              style={{
                position: 'absolute',
                bottom: -24,
                left: '50%',
                transform: 'translateX(-50%)',
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                color: '#fff',
                fontSize: '11px',
                padding: '2px 6px',
                borderRadius: '3px',
                whiteSpace: 'nowrap',
                pointerEvents: 'none',
              }}
            >
              {Math.round(selectionRect.width)} × {Math.round(selectionRect.height)}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default SelectionOverlay
