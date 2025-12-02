import React, { useState } from 'react'
import { getProviderLogo } from '../config/providerLogos'

/**
 * Provider头像组件
 * 支持图片logo、emoji和自动生成的首字母头像
 * @param {Object} props
 * @param {Object} props.provider - Provider对象
 * @param {number} props.size - 头像尺寸（像素）
 * @param {string} props.className - 额外的CSS类名
 */
export default function ProviderAvatar({
  provider,
  providerId,
  size = 32,
  className = ''
}) {
  const [imageError, setImageError] = useState(false)

  /**
   * 生成颜色（基于provider名称）
   */
  const generateColor = (name) => {
    if (!name) return '#6366f1'

    let hash = 0
    for (let i = 0; i < name.length; i++) {
      hash = name.charCodeAt(i) + ((hash << 5) - hash)
    }
    const hue = hash % 360
    return `hsl(${hue}, 65%, 50%)`
  }

  /**
   * 生成对比色（用于文字）
   */
  const getForegroundColor = (bgColor) => {
    // 简单的对比色判断，深色背景用白色文字
    return '#ffffff'
  }

  // 兼容只传 providerId 的场景
  const safeProvider = provider || { id: providerId || 'unknown', name: providerId || '未知', logo: null }

  // 尝试获取本地图标作为fallback
  const localLogo = getProviderLogo(safeProvider.id)
  const displayLogo = safeProvider.logo || localLogo

  const backgroundColor = generateColor(safeProvider.name)
  const color = getForegroundColor(backgroundColor)

  /**
   * 判断是否是图片URL
   * Vite导入的图片会返回一个路径字符串（可能包含hash）
   */
  const isImageUrl = (logo) => {
    if (!logo) return false
    const str = String(logo)

    // Vite导入的图片通常以 / 开头，或者包含文件扩展名
    // 也可能是以 /src/ 或 /assets/ 开头的路径
    return (
      str.startsWith('/') ||
      str.startsWith('http') ||
      str.includes('/assets/') ||
      str.includes('/src/') ||
      str.includes('.png') ||
      str.includes('.webp') ||
      str.includes('.svg') ||
      str.includes('.jpg') ||
      str.includes('.jpeg') ||
      str.includes('data:image') ||
      (str.length > 10 && !str.match(/^[\u{1F300}-\u{1F9FF}]/u)) // 排除emoji（通常只有1-4个字符）
    )
  }

  const logoIsImage = isImageUrl(displayLogo)
  const logoIsEmoji = displayLogo && !logoIsImage && String(displayLogo).length <= 4

  // 调试输出
  if (process.env.NODE_ENV === 'development') {
    console.log(`Provider ${safeProvider.name}:`, {
      logo: displayLogo,
      logoType: logoIsImage ? 'image' : logoIsEmoji ? 'emoji' : 'fallback',
      logoValue: typeof displayLogo
    })
  }

  return (
    <div
      className={`relative flex items-center justify-center rounded-lg overflow-hidden ${className}`}
      style={{
        width: size,
        height: size,
        minWidth: size,
        minHeight: size
      }}
    >
      {logoIsImage && !imageError ? (
        // 图片logo
        <img
          src={displayLogo}
          alt={safeProvider.name}
          className="w-full h-full object-contain"
          onError={() => setImageError(true)}
          style={{ imageRendering: 'crisp-edges' }}
        />
      ) : logoIsEmoji ? (
        // Emoji fallback
        <span style={{ fontSize: size * 0.6 }}>
          {displayLogo}
        </span>
      ) : (
        // 首字母头像fallback
        <div
          className="w-full h-full flex items-center justify-center text-white font-bold"
          style={{
            backgroundColor,
            color,
            fontSize: size * 0.45
          }}
        >
          {safeProvider.name ? safeProvider.name[0].toUpperCase() : '?'}
        </div>
      )}
    </div>
  )
}
