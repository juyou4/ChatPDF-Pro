import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * 防抖 localStorage 写入 Hook
 *
 * 将 React 状态与 localStorage 持久化结合，通过防抖机制合并高频写入，
 * 并在页面卸载（beforeunload）时立即 flush 未写入的数据，确保不丢失。
 *
 * @param {string} key - localStorage 键名
 * @param {*} initialValue - 初始值（localStorage 中无数据时使用）
 * @param {number} delay - 防抖间隔（毫秒），默认 300ms
 * @returns {[*, Function]} - [当前值, 设置值并防抖持久化的函数]
 */
export function useDebouncedLocalStorage(key, initialValue, delay = 300) {
  const [value, setValue] = useState(() => {
    try {
      const item = localStorage.getItem(key);
      return item !== null ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  // 待写入的值，null 表示没有待写入操作
  const pendingRef = useRef(null);
  // 防抖定时器
  const timerRef = useRef(null);
  // 保存最新的 key，供 flush 使用
  const keyRef = useRef(key);
  keyRef.current = key;

  const setValueAndPersist = useCallback((newValue) => {
    // 支持函数式更新
    const resolvedValue = typeof newValue === 'function' ? newValue(pendingRef.current ?? undefined) : newValue;
    setValue(resolvedValue);
    pendingRef.current = resolvedValue;

    // 清除上一次定时器，重新计时
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      try {
        localStorage.setItem(keyRef.current, JSON.stringify(pendingRef.current));
      } catch (e) {
        // localStorage 写入失败（如存储满），打印警告但不影响应用运行
        console.warn('[useDebouncedLocalStorage] 写入失败:', e);
      }
      pendingRef.current = null;
      timerRef.current = null;
    }, delay);
  }, [delay]);

  // beforeunload 时立即 flush，组件卸载时也 flush 并清理
  useEffect(() => {
    const flush = () => {
      if (pendingRef.current !== null) {
        try {
          localStorage.setItem(keyRef.current, JSON.stringify(pendingRef.current));
        } catch (e) {
          console.warn('[useDebouncedLocalStorage] flush 写入失败:', e);
        }
        pendingRef.current = null;
      }
    };

    window.addEventListener('beforeunload', flush);
    return () => {
      // 组件卸载时：flush 待写入数据、移除监听、清除定时器
      flush();
      window.removeEventListener('beforeunload', flush);
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [key]);

  return [value, setValueAndPersist];
}
