// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDebouncedValue } from '../useDebouncedValue';

describe('useDebouncedValue', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // --- 基本功能 ---

  it('初始返回值等于输入值', () => {
    const { result } = renderHook(() => useDebouncedValue(1.0, 150));
    expect(result.current).toBe(1.0);
  });

  it('输入值变化后，延迟到期前返回旧值', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebouncedValue(value, delay),
      { initialProps: { value: 1.0, delay: 150 } }
    );

    rerender({ value: 1.5, delay: 150 });

    // 还没到 150ms，应该仍然是旧值
    expect(result.current).toBe(1.0);
  });

  it('延迟到期后返回新值', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebouncedValue(value, delay),
      { initialProps: { value: 1.0, delay: 150 } }
    );

    rerender({ value: 1.5, delay: 150 });

    act(() => {
      vi.advanceTimersByTime(150);
    });

    expect(result.current).toBe(1.5);
  });

  // --- 防抖合并（需求 10.1）---

  it('150ms 内多次变化只更新为最后一次的值', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebouncedValue(value, delay),
      { initialProps: { value: 1.0, delay: 150 } }
    );

    // 快速连续变化
    rerender({ value: 1.2, delay: 150 });
    rerender({ value: 1.5, delay: 150 });
    rerender({ value: 2.0, delay: 150 });

    // 延迟到期前仍为初始值
    expect(result.current).toBe(1.0);

    act(() => {
      vi.advanceTimersByTime(150);
    });

    // 只更新为最后一次的值
    expect(result.current).toBe(2.0);
  });

  // --- 自定义延迟 ---

  it('支持自定义延迟时间', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebouncedValue(value, delay),
      { initialProps: { value: 'a', delay: 300 } }
    );

    rerender({ value: 'b', delay: 300 });

    // 150ms 时还没更新
    act(() => {
      vi.advanceTimersByTime(150);
    });
    expect(result.current).toBe('a');

    // 300ms 时更新
    act(() => {
      vi.advanceTimersByTime(150);
    });
    expect(result.current).toBe('b');
  });

  // --- 默认延迟 ---

  it('默认延迟为 150ms', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebouncedValue(value),
      { initialProps: { value: 10 } }
    );

    rerender({ value: 20 });

    // 149ms 时还没更新
    act(() => {
      vi.advanceTimersByTime(149);
    });
    expect(result.current).toBe(10);

    // 150ms 时更新
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(result.current).toBe(20);
  });

  // --- 支持不同类型 ---

  it('支持字符串类型', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebouncedValue(value, 150),
      { initialProps: { value: 'hello' } }
    );

    rerender({ value: 'world' });

    act(() => {
      vi.advanceTimersByTime(150);
    });

    expect(result.current).toBe('world');
  });

  it('支持对象类型', () => {
    const obj1 = { scale: 1.0 };
    const obj2 = { scale: 2.0 };

    const { result, rerender } = renderHook(
      ({ value }) => useDebouncedValue(value, 150),
      { initialProps: { value: obj1 } }
    );

    rerender({ value: obj2 });

    act(() => {
      vi.advanceTimersByTime(150);
    });

    expect(result.current).toEqual({ scale: 2.0 });
  });

  // --- 组件卸载时清理定时器 ---

  it('组件卸载时不会报错', () => {
    const { rerender, unmount } = renderHook(
      ({ value }) => useDebouncedValue(value, 150),
      { initialProps: { value: 1 } }
    );

    rerender({ value: 2 });

    // 卸载时应清理定时器，不抛异常
    expect(() => unmount()).not.toThrow();
  });
});
