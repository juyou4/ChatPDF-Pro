// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDebouncedLocalStorage } from '../useDebouncedLocalStorage';

describe('useDebouncedLocalStorage', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // --- 基本功能 ---

  it('无已有数据时使用初始值', () => {
    const { result } = renderHook(() =>
      useDebouncedLocalStorage('test-key', 'default')
    );
    expect(result.current[0]).toBe('default');
  });

  it('从 localStorage 读取已有数据', () => {
    localStorage.setItem('test-key', JSON.stringify('saved-value'));
    const { result } = renderHook(() =>
      useDebouncedLocalStorage('test-key', 'default')
    );
    expect(result.current[0]).toBe('saved-value');
  });

  it('localStorage 中数据格式异常时降级为初始值', () => {
    localStorage.setItem('test-key', '{{invalid json');
    const { result } = renderHook(() =>
      useDebouncedLocalStorage('test-key', 'fallback')
    );
    expect(result.current[0]).toBe('fallback');
  });

  it('支持对象类型的值', () => {
    const initial = { a: 1, b: 'hello' };
    const { result } = renderHook(() =>
      useDebouncedLocalStorage('obj-key', initial)
    );
    expect(result.current[0]).toEqual(initial);
  });

  // --- 防抖写入（需求 8.1, 8.2）---

  it('设置值后立即更新 React 状态，但不立即写入 localStorage', () => {
    const { result } = renderHook(() =>
      useDebouncedLocalStorage('test-key', 'init', 300)
    );

    act(() => {
      result.current[1]('new-value');
    });

    // React 状态已更新
    expect(result.current[0]).toBe('new-value');
    // localStorage 尚未写入
    expect(localStorage.getItem('test-key')).toBeNull();
  });

  it('防抖间隔后写入 localStorage', () => {
    const { result } = renderHook(() =>
      useDebouncedLocalStorage('test-key', 'init', 300)
    );

    act(() => {
      result.current[1]('delayed-value');
    });

    // 推进 300ms
    act(() => {
      vi.advanceTimersByTime(300);
    });

    expect(JSON.parse(localStorage.getItem('test-key'))).toBe('delayed-value');
  });

  it('300ms 内多次变更只写入一次 localStorage，且为最后一次的值', () => {
    const spy = vi.spyOn(Storage.prototype, 'setItem');
    const { result } = renderHook(() =>
      useDebouncedLocalStorage('test-key', 'init', 300)
    );

    act(() => {
      result.current[1]('value-1');
      result.current[1]('value-2');
      result.current[1]('value-3');
    });

    act(() => {
      vi.advanceTimersByTime(300);
    });

    // 只调用了一次 setItem（防抖合并）
    const calls = spy.mock.calls.filter(([k]) => k === 'test-key');
    expect(calls).toHaveLength(1);
    expect(JSON.parse(calls[0][1])).toBe('value-3');

    // React 状态也是最后一次的值
    expect(result.current[0]).toBe('value-3');

    spy.mockRestore();
  });

  // --- beforeunload flush（需求 8.3）---

  it('beforeunload 事件触发时立即 flush 待写入数据', () => {
    const { result } = renderHook(() =>
      useDebouncedLocalStorage('test-key', 'init', 300)
    );

    act(() => {
      result.current[1]('pending-value');
    });

    // 还没到 300ms，手动触发 beforeunload
    act(() => {
      window.dispatchEvent(new Event('beforeunload'));
    });

    expect(JSON.parse(localStorage.getItem('test-key'))).toBe('pending-value');
  });

  it('组件卸载时 flush 待写入数据并清理定时器', () => {
    const { result, unmount } = renderHook(() =>
      useDebouncedLocalStorage('test-key', 'init', 300)
    );

    act(() => {
      result.current[1]('unmount-value');
    });

    // 卸载组件
    unmount();

    expect(JSON.parse(localStorage.getItem('test-key'))).toBe('unmount-value');
  });

  it('没有待写入数据时 beforeunload 不写入', () => {
    const spy = vi.spyOn(Storage.prototype, 'setItem');

    renderHook(() =>
      useDebouncedLocalStorage('test-key', 'init', 300)
    );

    act(() => {
      window.dispatchEvent(new Event('beforeunload'));
    });

    const calls = spy.mock.calls.filter(([k]) => k === 'test-key');
    expect(calls).toHaveLength(0);

    spy.mockRestore();
  });

  // --- 自定义 delay ---

  it('支持自定义防抖间隔', () => {
    const { result } = renderHook(() =>
      useDebouncedLocalStorage('test-key', 'init', 500)
    );

    act(() => {
      result.current[1]('custom-delay');
    });

    // 300ms 时还没写入
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(localStorage.getItem('test-key')).toBeNull();

    // 500ms 时写入
    act(() => {
      vi.advanceTimersByTime(200);
    });
    expect(JSON.parse(localStorage.getItem('test-key'))).toBe('custom-delay');
  });
});
