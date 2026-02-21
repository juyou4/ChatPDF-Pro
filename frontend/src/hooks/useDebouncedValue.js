import { useState, useEffect, useRef } from 'react';

/**
 * 通用防抖值 Hook
 *
 * 延迟更新返回值，直到输入值在指定时间内不再变化。
 * 适用于 PDF 缩放、搜索输入等需要防抖的场景。
 *
 * @param {*} value - 需要防抖的输入值
 * @param {number} delay - 防抖延迟（毫秒），默认 150ms
 * @returns {*} 防抖后的值
 */
export function useDebouncedValue(value, delay = 150) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  const timerRef = useRef(null);

  useEffect(() => {
    // 设置定时器，延迟更新防抖值
    timerRef.current = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    // 清除上一次的定时器
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [value, delay]);

  return debouncedValue;
}
