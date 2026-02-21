// @vitest-environment jsdom
/**
 * Feature: chatpdf-frontend-performance, Property 10: PDF 缩放防抖渲染
 *
 * **Validates: Requirements 10.1**
 *
 * 属性定义：
 * For any 在 150ms 时间窗口内发生的 N 次（N≥1）缩放操作序列，
 * PDF 页面的实际渲染次数应不超过 1 次，
 * 且渲染使用的缩放值应为最后一次操作的值。
 *
 * 测试策略：
 * - 使用 renderHook 测试真实的 useDebouncedValue Hook
 * - 使用 vi.useFakeTimers 精确控制时间
 * - 使用 fast-check 生成随机缩放值序列（0.5 ~ 3.0 的浮点数）
 * - 所有缩放操作在 150ms 窗口内完成（每次间隔远小于 150ms）
 * - 验证防抖后 debouncedValue 只更新一次，且值为最后一次缩放值
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import * as fc from 'fast-check';
import { useDebouncedValue } from '../hooks/useDebouncedValue';

// 生成器：PDF 缩放值（0.5 ~ 3.0 之间的浮点数，保留 2 位小数）
const scaleArb = fc.double({ min: 0.5, max: 3.0, noNaN: true, noDefaultInfinity: true })
  .map(v => Math.round(v * 100) / 100);

// 生成器：N 次缩放操作序列（至少 1 次，最多 30 次）
const scaleSequenceArb = fc.array(scaleArb, { minLength: 1, maxLength: 30 });

describe('Property 10: PDF 缩放防抖渲染', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('150ms 窗口内 N 次缩放操作，防抖值只更新一次且为最后一次缩放值', () => {
    fc.assert(
      fc.property(scaleSequenceArb, (scales) => {
        vi.clearAllTimers();

        const initialScale = 1.0;

        const { result, rerender, unmount } = renderHook(
          ({ value }) => useDebouncedValue(value, 150),
          { initialProps: { value: initialScale } }
        );

        // 记录防抖值的变化次数
        let updateCount = 0;
        const prevValue = { current: result.current };

        // 在 150ms 窗口内快速连续缩放（每次间隔 5ms，远小于 150ms）
        for (const scale of scales) {
          rerender({ value: scale });
          // 推进少量时间，模拟快速连续缩放操作
          act(() => {
            vi.advanceTimersByTime(5);
          });

          // 检查防抖值是否发生了变化
          if (result.current !== prevValue.current) {
            updateCount++;
            prevValue.current = result.current;
          }
        }

        // 在防抖窗口内，防抖值不应更新（仍为初始值）
        // 注意：如果序列中某个值恰好等于初始值，防抖值不变也是正确的
        expect(updateCount).toBe(0);

        // 推进 150ms，触发防抖更新
        act(() => {
          vi.advanceTimersByTime(150);
        });

        // 检查最终更新
        if (result.current !== initialScale) {
          updateCount++;
        }

        // 属性验证 1：防抖值最多更新 1 次
        expect(updateCount).toBeLessThanOrEqual(1);

        // 属性验证 2：防抖后的值应为最后一次缩放值
        const lastScale = scales[scales.length - 1];
        expect(result.current).toBe(lastScale);

        // 继续推进时间，确认不会有额外更新
        const valueAfterDebounce = result.current;
        act(() => {
          vi.advanceTimersByTime(1000);
        });
        expect(result.current).toBe(valueAfterDebounce);

        unmount();
      }),
      { numRuns: 100 }
    );
  });

  it('单次缩放操作在防抖间隔后正确更新', () => {
    fc.assert(
      fc.property(scaleArb, (scale) => {
        vi.clearAllTimers();

        const { result, rerender, unmount } = renderHook(
          ({ value }) => useDebouncedValue(value, 150),
          { initialProps: { value: 1.0 } }
        );

        rerender({ value: scale });

        // 149ms 时不应更新
        act(() => {
          vi.advanceTimersByTime(149);
        });
        expect(result.current).toBe(1.0);

        // 150ms 时触发更新
        act(() => {
          vi.advanceTimersByTime(1);
        });
        expect(result.current).toBe(scale);

        // 之后不再有额外更新
        act(() => {
          vi.advanceTimersByTime(1000);
        });
        expect(result.current).toBe(scale);

        unmount();
      }),
      { numRuns: 100 }
    );
  });

  it('连续缩放序列中间不产生中间值渲染', () => {
    fc.assert(
      fc.property(scaleSequenceArb, (scales) => {
        vi.clearAllTimers();

        const initialScale = 1.0;
        const observedValues = [];

        const { result, rerender, unmount } = renderHook(
          ({ value }) => useDebouncedValue(value, 150),
          { initialProps: { value: initialScale } }
        );

        observedValues.push(result.current);

        // 快速连续缩放
        for (const scale of scales) {
          rerender({ value: scale });
          act(() => {
            vi.advanceTimersByTime(5);
          });
          observedValues.push(result.current);
        }

        // 触发防抖
        act(() => {
          vi.advanceTimersByTime(150);
        });
        observedValues.push(result.current);

        // 去重后的观察值序列：应只包含初始值和最终值
        const uniqueValues = [...new Set(observedValues)];
        const lastScale = scales[scales.length - 1];

        if (lastScale === initialScale) {
          // 如果最后一次缩放值等于初始值，唯一值只有 1 个
          expect(uniqueValues.length).toBe(1);
        } else {
          // 否则唯一值应为 2 个：初始值和最终值
          expect(uniqueValues.length).toBe(2);
          expect(uniqueValues).toContain(initialScale);
          expect(uniqueValues).toContain(lastScale);
        }

        unmount();
      }),
      { numRuns: 100 }
    );
  });
});
