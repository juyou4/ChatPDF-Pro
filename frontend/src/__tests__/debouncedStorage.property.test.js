// @vitest-environment jsdom
/**
 * Feature: chatpdf-frontend-performance, Property 9: localStorage 防抖写入合并
 *
 * **Validates: Requirements 8.1**
 *
 * 属性定义：
 * For any 在 300ms 时间窗口内发生的 N 次（N≥1）状态变更序列，
 * localStorage.setItem 的实际调用次数应不超过 1 次，
 * 且写入的值应为最后一次变更的值。
 *
 * 测试策略：
 * - 使用 renderHook 测试真实的 useDebouncedLocalStorage Hook
 * - 使用 vi.useFakeTimers 精确控制时间
 * - 使用 fast-check 生成随机状态变更序列
 * - 所有变更在 300ms 窗口内完成（每次间隔远小于 300ms）
 * - 验证防抖合并后 localStorage.setItem 只调用一次，且值为最后一次变更
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import * as fc from 'fast-check';
import { useDebouncedLocalStorage } from '../hooks/useDebouncedLocalStorage';

// 生成器：随机值（字符串、数字、布尔、对象、数组）
const arbitraryValue = fc.oneof(
  fc.string({ minLength: 0, maxLength: 50 }),
  fc.integer({ min: -1000, max: 1000 }),
  fc.boolean(),
  fc.record({
    name: fc.string({ minLength: 1, maxLength: 20 }),
    count: fc.integer({ min: 0, max: 100 }),
    enabled: fc.boolean(),
  }),
  fc.array(fc.integer({ min: 0, max: 100 }), { minLength: 0, maxLength: 10 })
);

// 生成器：N 次状态变更序列（至少 1 次）
const valueSequenceArb = fc.array(arbitraryValue, { minLength: 1, maxLength: 30 });

describe('Property 9: localStorage 防抖写入合并', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('300ms 窗口内 N 次变更只触发一次 localStorage 写入，且值为最后一次变更', () => {
    fc.assert(
      fc.property(valueSequenceArb, (values) => {
        // 每次迭代前清理环境
        localStorage.clear();
        vi.clearAllTimers();

        const spy = vi.spyOn(Storage.prototype, 'setItem');
        const testKey = 'pbt-debounce-key';

        const { result, unmount } = renderHook(() =>
          useDebouncedLocalStorage(testKey, null, 300)
        );

        // 在 300ms 窗口内快速连续变更（每次间隔 10ms，远小于 300ms）
        for (const val of values) {
          act(() => {
            result.current[1](val);
          });
          // 推进少量时间，模拟快速连续操作，但不超过防抖阈值
          act(() => {
            vi.advanceTimersByTime(10);
          });
        }

        // 此时最后一次变更刚发生不久，防抖定时器尚未触发
        // 过滤出针对我们 key 的 setItem 调用
        const callsBefore = spy.mock.calls.filter(([k]) => k === testKey);
        // 在 300ms 窗口内不应有写入（最后一次变更距今只有 10ms）
        expect(callsBefore.length).toBe(0);

        // 推进 300ms，触发防抖写入
        act(() => {
          vi.advanceTimersByTime(300);
        });

        const callsAfter = spy.mock.calls.filter(([k]) => k === testKey);

        // 属性验证 1：localStorage.setItem 只调用了 1 次
        expect(callsAfter.length).toBe(1);

        // 属性验证 2：写入的值是最后一次变更的值
        const lastValue = values[values.length - 1];
        const writtenValue = JSON.parse(callsAfter[0][1]);
        expect(writtenValue).toEqual(lastValue);

        // 继续推进时间，确认不会有额外写入
        act(() => {
          vi.advanceTimersByTime(1000);
        });
        const callsFinal = spy.mock.calls.filter(([k]) => k === testKey);
        expect(callsFinal.length).toBe(1);

        // 清理
        spy.mockRestore();
        unmount();
      }),
      { numRuns: 100 }
    );
  });

  it('单次变更在防抖间隔后也只触发一次写入', () => {
    fc.assert(
      fc.property(arbitraryValue, (value) => {
        localStorage.clear();
        vi.clearAllTimers();

        const spy = vi.spyOn(Storage.prototype, 'setItem');
        const testKey = 'pbt-single-key';

        const { result, unmount } = renderHook(() =>
          useDebouncedLocalStorage(testKey, null, 300)
        );

        act(() => {
          result.current[1](value);
        });

        // 299ms 时不应写入
        act(() => {
          vi.advanceTimersByTime(299);
        });
        const callsBefore = spy.mock.calls.filter(([k]) => k === testKey);
        expect(callsBefore.length).toBe(0);

        // 300ms 时触发写入
        act(() => {
          vi.advanceTimersByTime(1);
        });
        const callsAfter = spy.mock.calls.filter(([k]) => k === testKey);
        expect(callsAfter.length).toBe(1);

        const writtenValue = JSON.parse(callsAfter[0][1]);
        expect(writtenValue).toEqual(value);

        // 之后不再有额外写入
        act(() => {
          vi.advanceTimersByTime(1000);
        });
        const callsFinal = spy.mock.calls.filter(([k]) => k === testKey);
        expect(callsFinal.length).toBe(1);

        spy.mockRestore();
        unmount();
      }),
      { numRuns: 100 }
    );
  });

  it('React 状态始终反映最后一次变更的值', () => {
    fc.assert(
      fc.property(valueSequenceArb, (values) => {
        localStorage.clear();
        vi.clearAllTimers();

        const { result, unmount } = renderHook(() =>
          useDebouncedLocalStorage('pbt-state-key', null, 300)
        );

        // 依次设置所有值
        for (const val of values) {
          act(() => {
            result.current[1](val);
          });
        }

        // React 状态应始终为最后一次设置的值
        const lastValue = values[values.length - 1];
        expect(result.current[0]).toEqual(lastValue);

        unmount();
      }),
      { numRuns: 100 }
    );
  });
});
