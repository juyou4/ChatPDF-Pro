import React, { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Check, ChevronDown } from 'lucide-react';

/**
 * 暖色系下拉选择器，替代原生 <select>。
 *
 * 换掉原生控件的原因：Electron 28 对应 Chromium 120，弹出的 <option> 列表由浏览器
 * 自行绘制，CSS 只能改文字色和背景色，圆角、内边距、阴影以及选中态的系统蓝高亮都
 * 无法覆盖，展开后会明显脱离本项目的暖色卡片风格。能完全接管原生 select 外观的
 * `appearance: base-select` 要 Chromium 135+，当前版本用不了。
 *
 * 列表通过 portal 挂到 body 并用 fixed 定位：调用点普遍位于 overflow-y-auto /
 * overflow-hidden 的滚动容器内，普通绝对定位会被祖先裁掉。
 */

const SIZE_STYLES = {
  sm: { trigger: 'px-3.5 py-3 text-[13px]', option: 'text-[13px]' },
  md: { trigger: 'px-4 py-3 text-[14px]', option: 'text-[14px]' },
};

const PANEL_MAX_HEIGHT = 280;
const ESTIMATED_OPTION_HEIGHT = 38;
const PANEL_GAP = 6;

export default function SoftSelect({
  value,
  onChange,
  options,
  disabled = false,
  placeholder = '请选择',
  ariaLabel,
  size = 'md',
  className = '',
}) {
  const containerRef = useRef(null);
  const panelRef = useRef(null);
  const activeOptionRef = useRef(null);
  const listboxId = useId();

  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState(null);

  const sizeStyle = SIZE_STYLES[size] || SIZE_STYLES.md;
  const selectedIndex = useMemo(
    () => options.findIndex(option => option.value === value),
    [options, value]
  );
  const selectedOption = selectedIndex >= 0 ? options[selectedIndex] : null;
  const [activeIndex, setActiveIndex] = useState(Math.max(0, selectedIndex));

  useEffect(() => {
    if (!isOpen) setActiveIndex(Math.max(0, selectedIndex));
  }, [isOpen, selectedIndex]);

  useEffect(() => {
    if (disabled) setIsOpen(false);
  }, [disabled]);

  const updatePosition = useCallback(() => {
    const trigger = containerRef.current;
    if (!trigger) return;

    const rect = trigger.getBoundingClientRect();
    // 触发器被滚出视口时继续贴着它没有意义，直接收起。
    if (rect.bottom < 0 || rect.top > window.innerHeight) {
      setIsOpen(false);
      return;
    }

    const needed = Math.min(options.length * ESTIMATED_OPTION_HEIGHT + 12, PANEL_MAX_HEIGHT);
    const spaceBelow = window.innerHeight - rect.bottom - PANEL_GAP;
    const spaceAbove = rect.top - PANEL_GAP;
    const openUp = spaceBelow < needed && spaceAbove > spaceBelow;

    setPosition({
      left: rect.left,
      width: rect.width,
      openUp,
      top: openUp ? undefined : rect.bottom + PANEL_GAP,
      bottom: openUp ? window.innerHeight - rect.top + PANEL_GAP : undefined,
      maxHeight: Math.min(openUp ? spaceAbove : spaceBelow, PANEL_MAX_HEIGHT),
    });
  }, [options.length]);

  useLayoutEffect(() => {
    if (isOpen) updatePosition();
  }, [isOpen, updatePosition]);

  useEffect(() => {
    if (!isOpen) return undefined;

    // 捕获阶段：调用点位于多层滚动容器内，冒泡阶段收不到祖先的 scroll。
    const handleReflow = () => updatePosition();
    window.addEventListener('scroll', handleReflow, true);
    window.addEventListener('resize', handleReflow);

    const handleOutsideMouseDown = event => {
      if (containerRef.current?.contains(event.target)) return;
      if (panelRef.current?.contains(event.target)) return;
      setIsOpen(false);
    };
    document.addEventListener('mousedown', handleOutsideMouseDown);

    return () => {
      window.removeEventListener('scroll', handleReflow, true);
      window.removeEventListener('resize', handleReflow);
      document.removeEventListener('mousedown', handleOutsideMouseDown);
    };
  }, [isOpen, updatePosition]);

  useEffect(() => {
    if (isOpen) activeOptionRef.current?.scrollIntoView({ block: 'nearest' });
  }, [isOpen, activeIndex]);

  const selectOption = option => {
    if (option.value !== value) onChange?.(option.value);
    setIsOpen(false);
    containerRef.current?.querySelector('button')?.focus();
  };

  const handleKeyDown = event => {
    if (disabled) return;

    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (!isOpen) {
        setActiveIndex(Math.max(0, selectedIndex));
        setIsOpen(true);
        return;
      }
      const direction = event.key === 'ArrowDown' ? 1 : -1;
      setActiveIndex(current => (current + direction + options.length) % options.length);
      return;
    }

    if (event.key === 'Home' && isOpen) {
      event.preventDefault();
      setActiveIndex(0);
      return;
    }

    if (event.key === 'End' && isOpen) {
      event.preventDefault();
      setActiveIndex(options.length - 1);
      return;
    }

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (isOpen) selectOption(options[activeIndex]);
      else {
        setActiveIndex(Math.max(0, selectedIndex));
        setIsOpen(true);
      }
      return;
    }

    if (event.key === 'Escape' && isOpen) {
      event.preventDefault();
      // 阻止冒泡：外层弹窗若也监听 Esc，收起下拉不应该顺带把整个弹窗关掉。
      event.stopPropagation();
      setIsOpen(false);
      return;
    }

    if (event.key === 'Tab' && isOpen) setIsOpen(false);
  };

  return (
    <div ref={containerRef} className={`relative w-full ${className}`}>
      <button
        type="button"
        role="combobox"
        aria-label={ariaLabel}
        aria-controls={listboxId}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        disabled={disabled}
        onClick={() => {
          setActiveIndex(Math.max(0, selectedIndex));
          setIsOpen(open => !open);
        }}
        onKeyDown={handleKeyDown}
        className={`flex w-full items-center justify-between gap-2 rounded-[14px] border bg-white font-medium text-gray-700 outline-none transition-colors ${sizeStyle.trigger} ${
          disabled
            ? 'cursor-not-allowed border-gray-200 bg-gray-100 opacity-60'
            : isOpen
              ? 'border-[#FFA07A]/50 ring-2 ring-[#FFA07A]/25'
              : 'border-gray-200 hover:border-gray-300'
        }`}
      >
        <span className={`truncate text-left ${selectedOption ? '' : 'text-gray-400'}`}>
          {selectedOption ? selectedOption.label : placeholder}
        </span>
        <ChevronDown
          size={15}
          className={`shrink-0 transition-transform ${isOpen ? 'rotate-180 text-[#B85F47]' : 'text-gray-400'}`}
        />
      </button>

      {createPortal(
        <AnimatePresence>
          {isOpen && position && (
            <motion.div
              ref={panelRef}
              id={listboxId}
              role="listbox"
              aria-label={ariaLabel}
              initial={{ opacity: 0, y: position.openUp ? 4 : -4, scale: 0.985 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: position.openUp ? 3 : -3, scale: 0.985 }}
              transition={{ duration: 0.14, ease: [0.22, 1, 0.36, 1] }}
              style={{
                position: 'fixed',
                left: position.left,
                width: position.width,
                top: position.top,
                bottom: position.bottom,
                maxHeight: position.maxHeight,
                transformOrigin: position.openUp ? 'bottom left' : 'top left',
              }}
              className="custom-scrollbar z-[130] overflow-y-auto rounded-[14px] border border-gray-200/80 bg-white p-1.5 shadow-[0_16px_40px_rgba(30,30,35,0.14),0_3px_10px_rgba(30,30,35,0.06)]"
            >
              {options.map((option, index) => {
                const isSelected = option.value === value;
                const isActive = index === activeIndex;

                return (
                  <div
                    key={option.value}
                    ref={isActive ? activeOptionRef : undefined}
                    role="option"
                    aria-selected={isSelected}
                    onMouseDown={event => event.preventDefault()}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => selectOption(option)}
                    className={`flex cursor-pointer items-center justify-between gap-2 rounded-[10px] px-2.5 py-2 font-medium transition-colors ${sizeStyle.option} ${
                      isSelected
                        ? 'bg-[#FFF4EF] text-[#B85F47]'
                        : isActive
                          ? 'bg-gray-50 text-gray-800'
                          : 'text-gray-700'
                    }`}
                  >
                    <span className="truncate">{option.label}</span>
                    {isSelected && <Check size={14} strokeWidth={3} className="shrink-0" />}
                  </div>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </div>
  );
}
