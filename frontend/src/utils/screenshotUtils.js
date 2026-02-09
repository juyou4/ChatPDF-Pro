/**
 * 截图核心工具函数
 *
 * 提供纯函数用于 PDF 区域截图功能，包括：
 * - 选区坐标裁剪到页面范围内
 * - 最小选区尺寸验证
 * - 截图最大尺寸等比缩放
 * - 双路径截图生成（canvas 裁剪优先，html2canvas 降级）
 * - 快捷操作预设提示词映射
 */

import html2canvas from 'html2canvas'

// ============================================================
// 快捷操作预设提示词映射
// ============================================================

/**
 * 截图快捷操作配置
 *
 * 每个操作包含：
 * - label: 按钮显示文本
 * - icon: Lucide 图标名称
 * - prompt: 预设提示词（null 表示无预设）
 * - autoSend: 是否自动发送（false 表示仅填入输入框）
 */
export const SCREENSHOT_ACTIONS = {
  ask:       { label: '提问',    icon: 'MessageSquarePlus', prompt: null, autoSend: false },
  explain:   { label: '解释',    icon: 'BookOpen',          prompt: '请详细解释这张图片中的内容，如果是图表请分析数据趋势，如果是文本请概括核心观点。', autoSend: true },
  table:     { label: '表格',    icon: 'Table',             prompt: '请将这张图片中的表格或数据转换为 Markdown 格式，以便我可以复制。', autoSend: true },
  formula:   { label: '公式',    icon: 'Sigma',             prompt: '请将图片中的数学公式转换为 LaTeX 代码格式。', autoSend: true },
  ocr:       { label: 'OCR提取', icon: 'ScanText',          prompt: '请仅仅提取这张图片中的所有文字内容，保持原有排版，不要做任何解释。', autoSend: true },
  translate: { label: '翻译',    icon: 'Languages',         prompt: '请将这张图片中的内容翻译成中文（如果是代码或专有名词请保留原样）。', autoSend: true },
  copy:      { label: '复制',    icon: 'Copy',              prompt: null, autoSend: false },
}

// ============================================================
// 选区坐标处理
// ============================================================

/**
 * 将选区矩形裁剪到页面容器范围内
 *
 * 确保选区的 x、y、width、height 完全落在
 * [0, 0, containerWidth, containerHeight] 范围内。
 * 超出边界的部分会被截断。
 *
 * @param {{ x: number, y: number, width: number, height: number }} rect - 原始选区矩形
 * @param {number} containerWidth - 页面容器宽度（CSS 像素）
 * @param {number} containerHeight - 页面容器高度（CSS 像素）
 * @returns {{ x: number, y: number, width: number, height: number }} 裁剪后的选区矩形
 */
export function clampSelectionToPage(rect, containerWidth, containerHeight) {
  // 将起点限制在容器范围内
  const x = Math.max(0, Math.min(rect.x, containerWidth))
  const y = Math.max(0, Math.min(rect.y, containerHeight))

  // 将终点限制在容器范围内，然后计算宽高
  const endX = Math.max(0, Math.min(rect.x + rect.width, containerWidth))
  const endY = Math.max(0, Math.min(rect.y + rect.height, containerHeight))

  return {
    x,
    y,
    width: Math.max(0, endX - x),
    height: Math.max(0, endY - y),
  }
}

// ============================================================
// 选区有效性验证
// ============================================================

/**
 * 检查选区是否有效（宽高均 >= 10px）
 *
 * 用于过滤过小的误触框选操作。
 *
 * @param {{ width: number, height: number }} rect - 选区矩形（至少包含 width 和 height）
 * @returns {boolean} 宽高均 >= 10px 时返回 true
 */
export function isValidSelection(rect) {
  return rect.width >= 10 && rect.height >= 10
}

// ============================================================
// 截图尺寸限制
// ============================================================

/**
 * 限制截图最大尺寸并压缩为 JPEG，超过则等比缩放
 *
 * 若 canvas 的宽高均不超过 maxSize，直接导出。
 * 若任一边超过 maxSize，按最长边等比缩放至 maxSize，保持宽高比。
 * 使用 JPEG 格式压缩（quality=0.8），大幅减小传输体积。
 *
 * @param {HTMLCanvasElement} canvas - 原始截图 canvas
 * @param {number} [maxSize=1200] - 最长边的最大像素值
 * @param {number} [quality=0.8] - JPEG 压缩质量（0-1）
 * @returns {string} JPEG 格式的 base64 data URL
 */
export function applyMaxSizeLimit(canvas, maxSize = 1200, quality = 0.8) {
  const MAX_SIZE = maxSize
  let targetCanvas = canvas

  if (canvas.width > MAX_SIZE || canvas.height > MAX_SIZE) {
    // 按最长边等比缩放
    const scale = MAX_SIZE / Math.max(canvas.width, canvas.height)
    const resized = document.createElement('canvas')
    resized.width = Math.round(canvas.width * scale)
    resized.height = Math.round(canvas.height * scale)
    const ctx = resized.getContext('2d')
    ctx.drawImage(canvas, 0, 0, resized.width, resized.height)
    targetCanvas = resized
  }

  // 使用 JPEG 压缩，大幅减小体积（PNG 截图通常 2-5MB，JPEG 0.8 质量约 200-500KB）
  return targetCanvas.toDataURL('image/jpeg', quality)
}

// ============================================================
// 截图生成（双路径策略）
// ============================================================

/**
 * 从框选区域生成截图
 *
 * 采用双路径策略：
 * - 路径 A（优先）：直接裁剪 pdf.js 渲染的 page canvas 像素数据，性能最优
 * - 路径 B（降级）：使用 html2canvas 对容器进行截图，兼容性更好
 *
 * 两条路径均会通过 applyMaxSizeLimit 限制最大尺寸为 2000px。
 *
 * @param {React.RefObject} pdfContainerRef - PDF 页面容器的 ref 引用
 * @param {{ x: number, y: number, width: number, height: number }} rect - 选区矩形（相对于容器的 CSS 像素坐标）
 * @returns {Promise<string|null>} PNG 格式的 base64 data URL，或 null（两条路径均失败时）
 */
export async function captureArea(pdfContainerRef, rect) {
  const dpr = window.devicePixelRatio || 1

  // 路径 A：直接裁剪 pdf.js page canvas
  const pageCanvas = pdfContainerRef.current?.querySelector('canvas')
  if (pageCanvas) {
    try {
      // 计算 canvas 像素坐标（考虑 DPR 和 CSS 缩放）
      const canvasRect = pageCanvas.getBoundingClientRect()
      const scaleX = pageCanvas.width / canvasRect.width
      const scaleY = pageCanvas.height / canvasRect.height

      const sx = rect.x * scaleX
      const sy = rect.y * scaleY
      const sw = rect.width * scaleX
      const sh = rect.height * scaleY

      // 创建裁剪 canvas
      const cropCanvas = document.createElement('canvas')
      cropCanvas.width = sw
      cropCanvas.height = sh
      const ctx = cropCanvas.getContext('2d')
      ctx.drawImage(pageCanvas, sx, sy, sw, sh, 0, 0, sw, sh)

      return applyMaxSizeLimit(cropCanvas)
    } catch (e) {
      console.warn('Canvas 裁剪失败，降级到 html2canvas:', e)
    }
  }

  // 路径 B：html2canvas 降级
  try {
    const container = pdfContainerRef.current
    const canvas = await html2canvas(container, {
      scale: dpr,
      useCORS: true,
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
      ignoreElements: (el) => el.hasAttribute('data-html2canvas-ignore'),
    })
    return applyMaxSizeLimit(canvas)
  } catch (e) {
    console.error('html2canvas 也失败:', e)
    return null
  }
}
