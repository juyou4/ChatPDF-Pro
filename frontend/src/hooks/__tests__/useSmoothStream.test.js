// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useSmoothStream } from '../useSmoothStream';

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
});
