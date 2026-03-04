// @vitest-environment jsdom
import React from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { ChatParamsProvider, useChatParams } from '../ChatParamsContext.jsx';

describe('ChatParamsContext - mathEngine 兼容性', () => {
  const wrapper = ({ children }) => <ChatParamsProvider>{children}</ChatParamsProvider>;

  beforeEach(() => {
    localStorage.clear();
  });

  it('读取 chatParamsSettings 时应规范化 mathEngine 值', async () => {
    localStorage.setItem(
      'chatParamsSettings',
      JSON.stringify({
        mathEngine: 'mathjax',
      })
    );

    const { result, unmount } = renderHook(() => useChatParams(), { wrapper });

    await waitFor(() => {
      expect(result.current.mathEngine).toBe('MathJax');
    });

    unmount();
  });

  it('兼容 off/关闭 到 none，兼容 katex 到 KaTeX', async () => {
    localStorage.setItem(
      'chatParamsSettings',
      JSON.stringify({
        mathEngine: 'off',
      })
    );

    const first = renderHook(() => useChatParams(), { wrapper });
    await waitFor(() => {
      expect(first.result.current.mathEngine).toBe('none');
    });
    first.unmount();

    localStorage.setItem(
      'chatParamsSettings',
      JSON.stringify({
        mathEngine: '关闭',
      })
    );
    const second = renderHook(() => useChatParams(), { wrapper });
    await waitFor(() => {
      expect(second.result.current.mathEngine).toBe('none');
    });
    second.unmount();

    localStorage.setItem(
      'chatParamsSettings',
      JSON.stringify({
        mathEngine: 'katex',
      })
    );
    const third = renderHook(() => useChatParams(), { wrapper });
    await waitFor(() => {
      expect(third.result.current.mathEngine).toBe('KaTeX');
    });
    third.unmount();
  });

  it('当 chatParamsSettings 缺失时，应从 globalSettings 迁移并规范化', async () => {
    localStorage.setItem(
      'globalSettings',
      JSON.stringify({
        mathEngine: 'mathjax',
      })
    );

    const { result, unmount } = renderHook(() => useChatParams(), { wrapper });

    await waitFor(() => {
      expect(result.current.mathEngine).toBe('MathJax');
    });

    unmount();
  });

  it('无法识别的值应回退到默认 KaTeX', async () => {
    localStorage.setItem(
      'chatParamsSettings',
      JSON.stringify({
        mathEngine: 'unknown-engine',
      })
    );

    const { result, unmount } = renderHook(() => useChatParams(), { wrapper });

    await waitFor(() => {
      expect(result.current.mathEngine).toBe('KaTeX');
    });

    unmount();
  });
});
