// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

vi.mock('../useSmoothStream', () => ({
  useSmoothStream: () => ({
    addChunk: vi.fn(),
    reset: vi.fn(),
    contentRef: { current: null },
    getFinalText: () => '',
  }),
}));

import {
  useMessageState,
  STREAM_FIRST_EVENT_TIMEOUT_MS,
  normalizeAssistantCitations,
} from '../useMessageState';

const encoder = new TextEncoder();

const createInputEl = (text) => {
  const el = document.createElement('textarea');
  el.value = text;
  el.style.height = '24px';
  Object.defineProperty(el, 'scrollHeight', {
    value: 24,
    configurable: true,
  });
  return el;
};

const buildStreamResponse = (eventTexts) => {
  let idx = 0;
  return {
    ok: true,
    body: {
      getReader: () => ({
        read: async () => {
          if (idx >= eventTexts.length) {
            return { done: true, value: undefined };
          }
          const value = encoder.encode(eventTexts[idx++]);
          return { done: false, value };
        },
      }),
    },
    json: async () => ({}),
  };
};

const createOptions = () => ({
  docId: 'doc-test',
  screenshots: [],
  selectedText: '',
  getChatCredentials: () => ({
    providerId: 'openai',
    modelId: 'deepseek-chat',
    apiKey: 'test-key',
  }),
  getProviderById: () => ({ apiHost: null }),
  streamSpeed: 'normal',
  enableVectorSearch: false,
  globalSettings: {
    maxTokens: null,
    temperature: null,
    topP: null,
    contextCount: 2,
    streamOutput: true,
    enableTemperature: false,
    enableTopP: false,
    enableMaxTokens: false,
    customParams: [],
    reasoningEffort: 'off',
    enableMemory: false,
  },
});

describe('useMessageState streaming regressions', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
    global.alert = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('正常流式：应收敛为非 loading 且输出内容', async () => {
    const events = [
      `data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: '思考中...' } }] })}\n\n`,
      `data: ${JSON.stringify({ choices: [{ delta: { content: '最终回答' } }] })}\n\n`,
      `data: ${JSON.stringify({ done: true })}\n\n`,
    ];
    global.fetch.mockResolvedValue(buildStreamResponse(events));

    const { result } = renderHook(() => useMessageState(createOptions()));
    act(() => {
      result.current.textareaRef.current = createInputEl('你好');
    });

    await act(async () => {
      await result.current.sendMessage();
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.streamingMessageId).toBe(null);
    });

    const assistant = [...result.current.messages].reverse().find((m) => m.type === 'assistant');
    expect(assistant).toBeTruthy();
    expect(assistant.isStreaming).toBe(false);
    expect(assistant.content).toContain('最终回答');
    expect(assistant.thinking || '').toContain('思考中');
  });

  it('首包超时：应终止 loading 并给出超时提示', async () => {
    vi.useFakeTimers();

    global.fetch.mockImplementation((_url, options) => new Promise((_resolve, reject) => {
      const abortError = Object.assign(new Error('aborted'), { name: 'AbortError' });
      options?.signal?.addEventListener('abort', () => reject(abortError), { once: true });
    }));

    const { result } = renderHook(() => useMessageState(createOptions()));
    act(() => {
      result.current.textareaRef.current = createInputEl('超时测试');
    });

    let sendPromise;
    act(() => {
      sendPromise = result.current.sendMessage();
    });

    await vi.advanceTimersByTimeAsync(STREAM_FIRST_EVENT_TIMEOUT_MS + 50);
    await act(async () => {
      await sendPromise;
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.isLoading).toBe(false);
    expect(result.current.streamingMessageId).toBe(null);

    const assistant = [...result.current.messages].reverse().find((m) => m.type === 'assistant');
    expect(assistant).toBeTruthy();
    expect(assistant.isStreaming).toBe(false);
    expect(assistant.content).toContain('首包超时');
  });

  it('并发竞态：前一请求 abort 不应清空后一请求结果', async () => {
    let callCount = 0;
    global.fetch.mockImplementation((_url, options) => {
      callCount += 1;
      if (callCount === 1) {
        return new Promise((_resolve, reject) => {
          const abortError = Object.assign(new Error('aborted'), { name: 'AbortError' });
          options?.signal?.addEventListener('abort', () => reject(abortError), { once: true });
        });
      }
      return Promise.resolve(buildStreamResponse([
        `data: ${JSON.stringify({ choices: [{ delta: { content: '第二次请求成功' } }] })}\n\n`,
        `data: ${JSON.stringify({ done: true })}\n\n`,
      ]));
    });

    const { result } = renderHook(() => useMessageState(createOptions()));

    act(() => {
      result.current.textareaRef.current = createInputEl('第一次');
    });
    let firstPromise;
    act(() => {
      firstPromise = result.current.sendMessage();
    });

    act(() => {
      result.current.textareaRef.current = createInputEl('第二次');
    });
    await act(async () => {
      await result.current.sendMessage();
    });
    await act(async () => {
      await firstPromise;
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.streamingMessageId).toBe(null);
    });

    const assistantMessages = result.current.messages.filter((m) => m.type === 'assistant');
    const latestAssistant = assistantMessages[assistantMessages.length - 1];
    expect(latestAssistant.content).toContain('第二次请求成功');
    expect(latestAssistant.isStreaming).toBe(false);
  });
});

describe('normalizeAssistantCitations', () => {
  it('当回答只使用单一引用且存在多个 citation 时，应按段落重分配引用', () => {
    const content = '语义引导用于保持类别一致。 [1]\n\n3D 渲染用于生成可打印伪装。 [1]';
    const citations = [
      { ref: 1, highlight_text: '语义 引导 类别 一致', group_id: 'g1' },
      { ref: 2, highlight_text: '3d 渲染 可打印 伪装', group_id: 'g2' },
    ];

    const normalized = normalizeAssistantCitations(content, citations);
    const refs = [...normalized.matchAll(/\[(\d{1,3})\]/g)].map(m => Number(m[1]));
    expect(refs).toContain(1);
    expect(refs).toContain(2);
  });

  it('当回答已使用多个不同引用时，应保持原文不变', () => {
    const content = '第一点 [1]\n\n第二点 [2]';
    const citations = [
      { ref: 1, highlight_text: '第一点', group_id: 'g1' },
      { ref: 2, highlight_text: '第二点', group_id: 'g2' },
    ];

    expect(normalizeAssistantCitations(content, citations)).toBe(content);
  });
});
