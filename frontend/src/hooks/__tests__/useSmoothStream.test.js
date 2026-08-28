// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useSmoothStream } from '../useSmoothStream';

afterEach(() => {
  vi.useRealTimers();
});

describe('useSmoothStream', () => {
  it('当内容先渲染到内存、ref 后挂载时，应自动补写到 DOM', async () => {
    const { result } = renderHook(() => useSmoothStream({ streamDone: false, minDelay: 0 }));

    act(() => {
      result.current.addChunk('延迟挂载补写测试');
    });

    // 等待队列被消费（此时 contentRef 仍为空）
    await waitFor(() => {
      expect(result.current.getFinalText()).toBe('延迟挂载补写测试');
    });

    const el = document.createElement('div');
    act(() => {
      result.current.contentRef.current = el;
    });

    await waitFor(() => {
      expect(el.textContent).toBe('延迟挂载补写测试');
    });
  });

  it('启用 Blur Reveal 时，应追加带动画类名的字符节点', async () => {
    const { result } = renderHook(() =>
      useSmoothStream({
        streamDone: false,
        minDelay: 0,
        enableBlurReveal: true,
        blurIntensity: 'strong',
      })
    );

    const el = document.createElement('div');
    act(() => {
      result.current.contentRef.current = el;
      result.current.addChunk('动画');
    });

    await waitFor(() => {
      expect(el.textContent).toBe('动画');
    });

    await waitFor(() => {
      expect(el.querySelectorAll('.blur-reveal-animate').length).toBeGreaterThanOrEqual(2);
    });

    expect(el.querySelector('.blur-intensity-strong')).toBeTruthy();
  });

  it('frameChars 应控制流式阶段每帧展开速度', async () => {
    vi.useFakeTimers();
    const originalRaf = global.requestAnimationFrame;
    const originalCancelRaf = global.cancelAnimationFrame;
    global.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 16);
    global.cancelAnimationFrame = (id) => clearTimeout(id);

    try {
      const { result } = renderHook(() =>
        useSmoothStream({ streamDone: false, minDelay: 0, frameChars: 1 })
      );
      const el = document.createElement('div');

      act(() => {
        result.current.contentRef.current = el;
        result.current.addChunk('abcd');
      });

      await vi.advanceTimersByTimeAsync(20);
      expect(el.textContent).toBe('a');

      await vi.advanceTimersByTimeAsync(20);
      expect(el.textContent).toBe('ab');
    } finally {
      global.requestAnimationFrame = originalRaf;
      global.cancelAnimationFrame = originalCancelRaf;
    }
  });

  it('smoothFlush 应按 flushChars 渐进冲刷最终大块内容', async () => {
    vi.useFakeTimers();
    const originalRaf = global.requestAnimationFrame;
    const originalCancelRaf = global.cancelAnimationFrame;
    global.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 16);
    global.cancelAnimationFrame = (id) => clearTimeout(id);

    try {
      const { result, rerender } = renderHook(
        ({ streamDone }) => useSmoothStream({
          streamDone,
          minDelay: 0,
          frameChars: 1,
          flushChars: 2,
          smoothFlush: true,
        }),
        { initialProps: { streamDone: false } }
      );
      const el = document.createElement('div');

      act(() => {
        result.current.contentRef.current = el;
        result.current.addChunk('abcdef');
      });

      act(() => {
        rerender({ streamDone: true });
      });

      await vi.advanceTimersByTimeAsync(20);
      expect(el.textContent).toBe('ab');

      await vi.advanceTimersByTimeAsync(20);
      expect(el.textContent).toBe('abcd');
    } finally {
      global.requestAnimationFrame = originalRaf;
      global.cancelAnimationFrame = originalCancelRaf;
    }
  });

  it('waitForRevealComplete 支持 maxWaitMs 上限，收尾不必等完整动画尾巴', async () => {
    vi.useFakeTimers();
    const originalRaf = global.requestAnimationFrame;
    const originalCancelRaf = global.cancelAnimationFrame;
    global.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 16);
    global.cancelAnimationFrame = (id) => clearTimeout(id);

    try {
      const { result } = renderHook(() =>
        useSmoothStream({
          streamDone: false,
          minDelay: 0,
          enableBlurReveal: true,
          blurIntensity: 'strong',
        })
      );
      const el = document.createElement('div');
      act(() => {
        result.current.contentRef.current = el;
        result.current.addChunk('动画尾巴');
      });
      // 推进若干帧让字符带动画追加，刷新 revealTail 截止时间
      await act(async () => {
        await vi.advanceTimersByTimeAsync(200);
      });
      expect(el.textContent).toBe('动画尾巴');

      let cappedResolved = false;
      let uncappedResolved = false;
      const capped = result.current
        .waitForRevealComplete(null, 50)
        .then(() => { cappedResolved = true; });
      const uncapped = result.current
        .waitForRevealComplete(null)
        .then(() => { uncappedResolved = true; });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(80);
      });
      expect(cappedResolved).toBe(true);
      // strong 的完整尾巴 >700ms，未加上限时此刻仍应在等待
      expect(uncappedResolved).toBe(false);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1200);
      });
      expect(uncappedResolved).toBe(true);
      await Promise.all([capped, uncapped]);
    } finally {
      global.requestAnimationFrame = originalRaf;
      global.cancelAnimationFrame = originalCancelRaf;
    }
  });
});
