import React, {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  AnimatePresence,
  motion,
  useMotionValue,
  useSpring,
  useTransform,
} from 'framer-motion';
import { cn } from '../../utils/cn';

const FloatingDockContext = createContext(null);

// 这组参数刻意保留一点过冲，让鼠标从一个按钮滑到另一个按钮时能看到
// “跟手放大 → 轻微回弹”的过程，而不是只得到一个静态缩放值。
const DOCK_SPRING = {
  mass: 0.36,
  stiffness: 320,
  damping: 11,
};

const usePrefersReducedMotion = () => {
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined;
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const syncPreference = () => setReduceMotion(media.matches);
    syncPreference();
    media.addEventListener?.('change', syncPreference);
    return () => media.removeEventListener?.('change', syncPreference);
  }, []);

  return reduceMotion;
};

export const FloatingDock = ({
  children,
  className = '',
  darkMode = false,
  ariaLabel = '工具栏',
}) => {
  const mouseX = useMotionValue(Number.POSITIVE_INFINITY);
  const reduceMotion = usePrefersReducedMotion();
  const updateMouseX = useCallback((event) => {
    const eventX = Number.isFinite(event.clientX)
      ? event.clientX
      : Number.isFinite(event.pageX)
        ? event.pageX
        : Number.POSITIVE_INFINITY;
    mouseX.set(eventX);
  }, [mouseX]);
  const clearMouseX = useCallback(() => {
    mouseX.set(Number.POSITIVE_INFINITY);
  }, [mouseX]);

  return (
    <FloatingDockContext.Provider value={{ mouseX, reduceMotion, darkMode }}>
      <motion.div
        data-floating-dock="true"
        role="toolbar"
        aria-label={ariaLabel}
        initial={reduceMotion ? false : { opacity: 0, y: -10, scale: 0.94 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={reduceMotion
          ? { duration: 0 }
          : { type: 'spring', mass: 0.62, stiffness: 380, damping: 19 }}
        onPointerEnter={updateMouseX}
        onPointerMove={updateMouseX}
        onPointerLeave={clearMouseX}
        // Electron/旧 Chromium 的鼠标事件兼容兜底，确保桌面端鼠标移动
        // 一定能驱动 mouseX。
        onMouseEnter={updateMouseX}
        onMouseMove={updateMouseX}
        onMouseLeave={clearMouseX}
        className={cn(
          'relative flex h-11 max-w-full items-center gap-0 overflow-visible rounded-[15px] border px-1.5',
          darkMode
            ? 'border-white/[0.09] bg-[#24282e]/95 shadow-[0_10px_28px_rgba(0,0,0,0.24)]'
            : 'border-white/90 bg-white/90 shadow-[0_10px_30px_rgba(91,69,54,0.12)]',
          className,
        )}
      >
        {children}
      </motion.div>
    </FloatingDockContext.Provider>
  );
};

export const FloatingDockItem = forwardRef(({
  children,
  label,
  active = false,
  disabled = false,
  showTooltip = true,
  className = '',
  slotClassName = '',
  onClick,
  type = 'button',
  ...buttonProps
}, forwardedRef) => {
  const dock = useContext(FloatingDockContext);
  const localRef = useRef(null);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const fallbackMouseX = useMotionValue(Number.POSITIVE_INFINITY);
  const mouseX = dock?.mouseX ?? fallbackMouseX;

  const distance = useTransform(mouseX, (value) => {
    if (!Number.isFinite(value)) return 1000;
    const bounds = localRef.current?.getBoundingClientRect();
    if (!bounds) return 1000;
    return value - bounds.left - bounds.width / 2;
  });
  const targetScale = useTransform(
    distance,
    [-112, -72, -38, 0, 38, 72, 112],
    [1, 1.12, 1.3, 1.58, 1.3, 1.12, 1],
  );
  const scale = useSpring(targetScale, {
    ...DOCK_SPRING,
  });
  const targetLift = useTransform(
    distance,
    [-112, -72, -38, 0, 38, 72, 112],
    [0, -1.5, -3.5, -8, -3.5, -1.5, 0],
  );
  const lift = useSpring(targetLift, {
    ...DOCK_SPRING,
  });
  // 让鼠标两侧的按钮轻微向外让位，形成 Aceternity 风格的鱼眼效果，
  // 同时不改变 flex 布局宽度，避免阅读区页码和缩放控件跳动。
  const targetSpread = useTransform(
    distance,
    [-112, -72, -38, 0, 38, 72, 112],
    [0, 2.5, 5, 0, -5, -2.5, 0],
  );
  const spread = useSpring(targetSpread, {
    ...DOCK_SPRING,
  });
  // 额外的入点弹簧让用户把鼠标移入按钮时能立即感知到反馈，
  // 与距离驱动的鱼眼缩放叠加但不改变 slot 尺寸。
  const hoverBoost = useSpring(hovered && !disabled ? 1.06 : 1, {
    mass: 0.28,
    stiffness: 360,
    damping: 10,
  });
  const combinedScale = useTransform(
    [scale, hoverBoost],
    ([distanceScale, entryBoost]) => distanceScale * entryBoost,
  );
  const tooltipVisible = showTooltip && !disabled && (hovered || focused);

  const setRefs = (node) => {
    localRef.current = node;
    if (typeof forwardedRef === 'function') forwardedRef(node);
    else if (forwardedRef) forwardedRef.current = node;
  };

  const syncItemPointer = useCallback((event) => {
    setHovered(true);
    const bounds = localRef.current?.getBoundingClientRect();
    if (bounds) {
      // 进入按钮的瞬间先定位到按钮中心，避免必须再移动一小段鼠标
      // 才出现放大；后续移动由 FloatingDock 的全局监听接管。
      const eventX = Number.isFinite(event.clientX)
        ? event.clientX
        : Number.isFinite(event.pageX)
          ? event.pageX
          : bounds.left + bounds.width / 2;
      mouseX.set(eventX);
    }
  }, [mouseX]);
  const clearItemHover = useCallback(() => {
    setHovered(false);
  }, []);

  return (
    <div
      ref={setRefs}
      className={cn('relative flex h-8 w-8 shrink-0 items-center justify-center', slotClassName)}
      onPointerEnter={syncItemPointer}
      onPointerMove={syncItemPointer}
      onPointerLeave={clearItemHover}
      onMouseEnter={syncItemPointer}
      onMouseMove={syncItemPointer}
      onMouseLeave={clearItemHover}
    >
      <AnimatePresence>
        {tooltipVisible && (
          <motion.div
            initial={{ opacity: 0, y: -4, x: '-50%', scale: 0.96 }}
            animate={{ opacity: 1, y: 0, x: '-50%', scale: 1 }}
            exit={{ opacity: 0, y: -2, x: '-50%', scale: 0.98 }}
            transition={{ duration: 0.14, ease: [0.22, 1, 0.36, 1] }}
            className={cn(
              'pointer-events-none absolute left-1/2 top-[calc(100%+9px)] z-[70] whitespace-nowrap rounded-lg border px-2 py-1 text-[11px] font-medium',
              dock?.darkMode
                ? 'border-white/10 bg-[#30343b] text-gray-100 shadow-lg shadow-black/30'
                : 'border-[#e7ddd6] bg-[#fffdfb] text-[#51473f] shadow-[0_8px_20px_rgba(74,52,38,0.13)]',
            )}
          >
            {label}
          </motion.div>
        )}
      </AnimatePresence>
      <motion.button
        data-floating-dock-item="true"
        type={type}
        aria-label={label}
        disabled={disabled}
        onClick={onClick}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        whileTap={dock?.reduceMotion || disabled ? undefined : { scale: 0.94 }}
        style={{
          scale: dock?.reduceMotion ? 1 : combinedScale,
          x: dock?.reduceMotion ? 0 : spread,
          y: dock?.reduceMotion ? 0 : lift,
          transformOrigin: 'center bottom',
        }}
        className={cn(
          'relative z-10 flex h-7 w-7 items-center justify-center rounded-[9px] border border-transparent outline-none transition-[color,background-color,border-color,box-shadow] duration-200 will-change-transform focus-visible:ring-2 focus-visible:ring-[#d97a5d]/35 focus-visible:ring-offset-1 disabled:cursor-default disabled:opacity-35',
          active
            ? (dock?.darkMode
                ? 'border-white/10 bg-white/[0.12] text-[#ffb49a] shadow-sm'
                : 'border-[#f0d7cd] bg-[#f8e8e1] text-[#a6543e] shadow-[0_4px_12px_rgba(184,95,71,0.12)]')
            : (dock?.darkMode
                ? 'text-gray-400 hover:bg-white/[0.08] hover:text-gray-100'
                : 'text-[#776a62] hover:bg-[#f4efeb] hover:text-[#9f5541]'),
          hovered && !active && !disabled
            ? (dock?.darkMode
                ? 'bg-white/[0.08] text-gray-100 shadow-[0_6px_16px_rgba(0,0,0,0.16)]'
                : 'bg-[#f7efeb] text-[#9f5541] shadow-[0_6px_16px_rgba(184,95,71,0.14)]')
            : '',
          className,
        )}
        {...buttonProps}
      >
        {children}
      </motion.button>
    </div>
  );
});

FloatingDockItem.displayName = 'FloatingDockItem';

export const FloatingDockDivider = ({ darkMode = false, className = '' }) => (
  <span
    aria-hidden="true"
    className={cn(
      'mx-0.5 h-4 w-px shrink-0',
      darkMode ? 'bg-white/[0.09]' : 'bg-[#e9dfd8]',
      className,
    )}
  />
);
