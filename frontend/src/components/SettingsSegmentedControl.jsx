import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

const SLIDE_SPRING = {
  type: 'spring',
  stiffness: 460,
  damping: 34,
  mass: 0.68,
}

const getReducedMotionPreference = () => (
  typeof window !== 'undefined'
  && typeof window.matchMedia === 'function'
  && window.matchMedia('(prefers-reduced-motion: reduce)').matches
)

const useReducedMotionPreference = () => {
  const [reduceMotion, setReduceMotion] = useState(getReducedMotionPreference)

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined

    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const handleChange = (event) => setReduceMotion(event.matches)
    mediaQuery.addEventListener?.('change', handleChange)

    return () => mediaQuery.removeEventListener?.('change', handleChange)
  }, [])

  return reduceMotion
}

/**
 * 设置页共用的滑动分段选择器。
 * 选中背景独立于按钮移动，避免切换时三个选项各自闪烁。
 */
export default function SettingsSegmentedControl({
  ariaLabel,
  value,
  options,
  onChange,
  className = '',
  buttonClassName = 'py-1.5 text-sm font-medium text-center rounded-[16px]',
  indicatorClassName = 'rounded-[16px]',
  renderOption,
}) {
  const reduceMotion = useReducedMotionPreference()
  const activeIndex = options.findIndex((option) => option.value === value)
  const hasSelection = activeIndex >= 0

  return (
    <div
      className={`settings-segment relative grid p-1 rounded-[20px] ${className}`}
      style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
      role="group"
      aria-label={ariaLabel}
    >
      <motion.div
        aria-hidden="true"
        className={`settings-segment-selection pointer-events-none absolute left-1 top-1 bottom-1 ${indicatorClassName}`}
        initial={false}
        animate={{
          x: `${Math.max(activeIndex, 0) * 100}%`,
          opacity: hasSelection ? 1 : 0,
          scale: hasSelection ? 1 : 0.96,
        }}
        transition={reduceMotion ? { duration: 0 } : SLIDE_SPRING}
        style={{ width: `calc((100% - 0.5rem) / ${options.length})` }}
      />

      {options.map((option) => {
        const isActive = option.value === value
        const isDisabled = option.disabled === true

        return (
          <button
            key={String(option.value)}
            type="button"
            aria-pressed={isActive}
            data-active={isActive}
            disabled={isDisabled}
            onClick={() => {
              if (!isDisabled && !isActive) onChange(option.value)
            }}
            className={`settings-segment-option relative z-10 min-w-0 ${buttonClassName} ${isDisabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
          >
            {renderOption ? renderOption(option, isActive) : option.label}
          </button>
        )
      })}
    </div>
  )
}
