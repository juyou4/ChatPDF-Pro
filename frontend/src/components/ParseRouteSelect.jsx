import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Check, ChevronDown, Cloud, Laptop, Sparkles } from 'lucide-react';
import { PARSE_ROUTE_OPTIONS } from '../utils/parseRouteUtils';

const ROUTE_VISUALS = {
  auto: {
    icon: Sparkles,
    hint: '按文档质量自动选择，优先使用本地解析',
  },
  local: {
    icon: Laptop,
    hint: '适合文本型 PDF，解析更快且无需云端',
  },
  mineru: {
    icon: Cloud,
    hint: '适合扫描件与复杂版面，全功能统一解析',
  },
};

export default function ParseRouteSelect({
  value,
  onChange,
  darkMode = false,
  disabled = false,
  className = '',
}) {
  const containerRef = useRef(null);
  const listboxId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const selectedIndex = useMemo(
    () => Math.max(0, PARSE_ROUTE_OPTIONS.findIndex((option) => option.value === value)),
    [value]
  );
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const selectedOption = PARSE_ROUTE_OPTIONS[selectedIndex];

  useEffect(() => {
    if (!isOpen) setActiveIndex(selectedIndex);
  }, [isOpen, selectedIndex]);

  useEffect(() => {
    if (disabled) setIsOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (!isOpen) return undefined;

    const handleOutsideMouseDown = (event) => {
      if (!containerRef.current?.contains(event.target)) setIsOpen(false);
    };

    document.addEventListener('mousedown', handleOutsideMouseDown);
    return () => document.removeEventListener('mousedown', handleOutsideMouseDown);
  }, [isOpen]);

  const selectOption = (option) => {
    if (option.value !== selectedOption.value) onChange?.(option.value);
    setIsOpen(false);
  };

  const handleTriggerKeyDown = (event) => {
    if (disabled) return;

    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (!isOpen) {
        setActiveIndex(selectedIndex);
        setIsOpen(true);
        return;
      }

      const direction = event.key === 'ArrowDown' ? 1 : -1;
      setActiveIndex((current) => (
        (current + direction + PARSE_ROUTE_OPTIONS.length) % PARSE_ROUTE_OPTIONS.length
      ));
      return;
    }

    if (event.key === 'Home' && isOpen) {
      event.preventDefault();
      setActiveIndex(0);
      return;
    }

    if (event.key === 'End' && isOpen) {
      event.preventDefault();
      setActiveIndex(PARSE_ROUTE_OPTIONS.length - 1);
      return;
    }

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (isOpen) selectOption(PARSE_ROUTE_OPTIONS[activeIndex]);
      else {
        setActiveIndex(selectedIndex);
        setIsOpen(true);
      }
      return;
    }

    if (event.key === 'Escape' && isOpen) {
      event.preventDefault();
      setIsOpen(false);
      return;
    }

    if (event.key === 'Tab' && isOpen) setIsOpen(false);
  };

  return (
    <div
      ref={containerRef}
      className={`relative inline-flex h-7 w-[132px] shrink-0 ${className}`}
      title={selectedOption.description}
    >
      <button
        type="button"
        role="combobox"
        aria-label="上传解析路线"
        aria-controls={listboxId}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        aria-activedescendant={isOpen ? `${listboxId}-option-${PARSE_ROUTE_OPTIONS[activeIndex].value}` : undefined}
        disabled={disabled}
        onClick={() => {
          setActiveIndex(selectedIndex);
          setIsOpen((open) => !open);
        }}
        onKeyDown={handleTriggerKeyDown}
        className={`group flex h-full w-full items-center justify-between gap-2 rounded-full border pl-3 pr-2 text-[11px] font-semibold outline-none transition-[background-color,border-color,color,box-shadow] duration-150 focus-visible:ring-2 focus-visible:ring-[#D97A5D]/25 disabled:cursor-not-allowed disabled:opacity-50 ${
          darkMode
            ? isOpen
              ? 'border-white/20 bg-white/[0.09] text-gray-100 shadow-[0_6px_18px_rgba(0,0,0,0.2)]'
              : 'border-white/10 bg-white/[0.04] text-gray-200 hover:border-white/20 hover:bg-white/[0.07]'
            : isOpen
              ? 'border-gray-300 bg-white text-gray-800 shadow-[0_5px_16px_rgba(40,40,45,0.08)]'
              : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
        }`}
      >
        <span className="truncate">{selectedOption.label}</span>
        <motion.span
          aria-hidden="true"
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.16, ease: [0.22, 1, 0.36, 1] }}
          className={`shrink-0 ${isOpen ? (darkMode ? 'text-[#FFA07A]' : 'text-[#B85F47]') : (darkMode ? 'text-gray-500' : 'text-gray-400')}`}
        >
          <ChevronDown className="h-3.5 w-3.5" />
        </motion.span>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            id={listboxId}
            role="listbox"
            aria-label="选择上传解析路线"
            initial={{ opacity: 0, y: -5, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.985 }}
            transition={{ duration: 0.16, ease: [0.22, 1, 0.36, 1] }}
            style={{ transformOrigin: 'top left' }}
            className={`absolute left-0 top-full z-[80] mt-2 w-[292px] overflow-hidden rounded-[18px] p-1.5 shadow-[0_16px_40px_rgba(30,30,35,0.14),0_3px_10px_rgba(30,30,35,0.06)] ${
              darkMode
                ? 'border border-white/[0.12] bg-[#202227]/[0.98]'
                : 'bg-white'
            }`}
          >
            {PARSE_ROUTE_OPTIONS.map((option, index) => {
              const isSelected = option.value === selectedOption.value;
              const isActive = index === activeIndex;
              const routeVisual = ROUTE_VISUALS[option.value] || ROUTE_VISUALS.auto;
              const RouteIcon = routeVisual.icon;

              return (
                <motion.div
                  key={option.value}
                  id={`${listboxId}-option-${option.value}`}
                  role="option"
                  aria-label={option.label}
                  aria-describedby={`${listboxId}-hint-${option.value}`}
                  aria-selected={isSelected}
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => selectOption(option)}
                  whileHover={{ x: 2 }}
                  transition={{ duration: 0.12, ease: 'easeOut' }}
                  className={`flex min-h-[54px] cursor-pointer items-center gap-2.5 rounded-[14px] px-2.5 py-2 outline-none transition-colors ${
                    darkMode
                      ? isSelected || isActive
                        ? 'bg-white/[0.07] text-gray-100'
                        : 'text-gray-200'
                      : isSelected || isActive
                        ? 'bg-[#f5f4f2] text-gray-900'
                        : 'text-gray-700'
                  }`}
                >
                  <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
                    darkMode ? 'bg-white/[0.06] text-gray-400' : 'bg-white text-gray-600 shadow-[0_1px_3px_rgba(30,30,35,0.1)]'
                  }`}>
                    <RouteIcon className="h-3.5 w-3.5" strokeWidth={2} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[12px] font-semibold">
                      {option.label}
                    </span>
                    <span id={`${listboxId}-hint-${option.value}`} className={`mt-0.5 block text-[11px] leading-[1.45] ${
                      darkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      {routeVisual.hint}
                    </span>
                  </span>
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center">
                    <AnimatePresence initial={false}>
                      {isSelected && (
                        <motion.span
                          key="selected"
                          initial={{ opacity: 0, scale: 0.65 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.65 }}
                          transition={{ duration: 0.14 }}
                          className={`flex h-5 w-5 items-center justify-center rounded-full ${
                            darkMode ? 'bg-[#F0653A]/20 text-[#FFA07A]' : 'bg-[#F0653A] text-white'
                          }`}
                        >
                          <Check className="h-3 w-3" strokeWidth={3} />
                        </motion.span>
                      )}
                    </AnimatePresence>
                  </span>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
