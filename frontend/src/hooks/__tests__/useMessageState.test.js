// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

const hoisted = vi.hoisted(() => ({
  useSmoothStreamMock: vi.fn(),
}));

vi.mock('../useSmoothStream', () => ({
  useSmoothStream: hoisted.useSmoothStreamMock,
}));

vi.mock('../../contexts/WebSearchContext', () => ({
  useWebSearch: () => ({
    enableWebSearch: false,
    webSearchProvider: 'duckduckgo',
    webSearchApiKey: '',
  }),
}));

import {
  useMessageState,
  STREAM_FIRST_EVENT_TIMEOUT_MS,
  finalizeThinkingDurationMs,
  normalizeAssistantCitations,
  ensureAssistantInlineCitationFallback,
  optimizeAssistantInlineCitations,
  filterCitationsByContentRefs,
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
    answerDetailLevel: 'standard',
    enableMemory: false,
  },
});

describe('useMessageState streaming regressions', () => {
  beforeEach(() => {
    hoisted.useSmoothStreamMock.mockReset();
    hoisted.useSmoothStreamMock.mockImplementation(() => ({
      addChunk: vi.fn(),
      reset: vi.fn(),
      contentRef: { current: null },
      getFinalText: () => '',
      isFlushComplete: () => true,
    }));
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

  it('流式完成事件应保存 memory hits 与 meta', async () => {
    const events = [
      `data: ${JSON.stringify({ choices: [{ delta: { content: '回答正文' } }] })}\n\n`,
      `data: ${JSON.stringify({ done: true, memory_hits: [{ id: 'mem-1', memory_kind: 'doc_fact', title: '文档事实', summary: '当前文档的重要结论' }], memory_meta: { enabled: true, strategy: 'context_injector', selected_count: 1 } })}\n\n`,
    ];
    global.fetch.mockResolvedValue(buildStreamResponse(events));

    const { result } = renderHook(() => useMessageState(createOptions()));
    act(() => {
      result.current.textareaRef.current = createInputEl('测试记忆命中');
    });

    await act(async () => {
      await result.current.sendMessage();
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const assistant = [...result.current.messages].reverse().find((m) => m.type === 'assistant');
    expect(assistant.memoryHits).toEqual([
      expect.objectContaining({ id: 'mem-1', memory_kind: 'doc_fact' }),
    ]);
    expect(assistant.memoryMeta).toEqual(
      expect.objectContaining({ enabled: true, selected_count: 1 })
    );
  });

  it('非流式响应应执行引用纠偏并同步过滤 citations', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        answer: '该方法通过全局光照建模提升物理鲁棒性[1]。',
        reasoning_content: '',
        retrieval_meta: {
          citations: [
            { ref: 1, display_ref: 1, source_ref: 5, highlight_text: '全局光照建模 提升 物理 鲁棒性', group_id: 'group-5' },
            { ref: 2, display_ref: 2, source_ref: 8, highlight_text: '补充证据', group_id: 'group-8' },
          ],
        },
      }),
    });

    const { result } = renderHook(() => useMessageState({
      ...createOptions(),
      globalSettings: {
        ...createOptions().globalSettings,
        streamOutput: false,
      },
    }));

    act(() => {
      result.current.textareaRef.current = createInputEl('非流式引用纠偏');
    });

    await act(async () => {
      await result.current.sendMessage();
    });

    const assistant = [...result.current.messages].reverse().find((m) => m.type === 'assistant');
    expect(assistant.content).toBe('该方法通过全局光照建模提升物理鲁棒性[1]。');
    expect(assistant.citations.map((c) => c.ref)).toEqual([1, 2]);
    expect(assistant.citations.map((c) => c.display_ref)).toEqual([1, 2]);
    expect(assistant.citations.map((c) => c.source_ref)).toEqual([5, 8]);
  });

  it('done 事件携带 final_content 时，应补齐未提前流出的尾部内容', async () => {
    const events = [
      `data: ${JSON.stringify({ content: '前半段', done: false })}\n\n`,
      `data: ${JSON.stringify({ done: true, final_content: '前半段补齐尾部', retrieval_meta: { citations: [{ ref: 1, display_ref: 1, source_ref: 3, group_id: 'g3' }] } })}\n\n`,
    ];
    global.fetch.mockResolvedValue(buildStreamResponse(events));

    const { result } = renderHook(() => useMessageState(createOptions()));
    act(() => {
      result.current.textareaRef.current = createInputEl('补齐尾部');
    });

    await act(async () => {
      await result.current.sendMessage();
    });

    const assistant = [...result.current.messages].reverse().find((m) => m.type === 'assistant');
    expect(assistant.content).toBe('前半段补齐尾部');

    const contentStream = hoisted.useSmoothStreamMock.mock.results[0]?.value;
    expect(contentStream.addChunk).toHaveBeenCalledWith('前半段');
    expect(contentStream.addChunk).toHaveBeenCalledWith('补齐尾部');
  });

  it('streamSpeed 应映射到 useSmoothStream 的真实渲染参数', () => {
    renderHook(() => useMessageState({
      ...createOptions(),
      streamSpeed: 'slow',
    }));

    expect(hoisted.useSmoothStreamMock).toHaveBeenCalledTimes(2);
    expect(hoisted.useSmoothStreamMock.mock.calls[0][0]).toEqual(
      expect.objectContaining({
        minDelay: 48,
        frameChars: 1,
        flushChars: 2,
      })
    );
    expect(hoisted.useSmoothStreamMock.mock.calls[1][0]).toEqual(
      expect.objectContaining({
        minDelay: 48,
        frameChars: 1,
        flushChars: 2,
      })
    );
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

describe('ensureAssistantInlineCitationFallback', () => {
  it('当正文没有任何编号且存在 citations 时，应注入内联引用', () => {
    const content = '这是一个没有编号的回答。';
    const citations = [{ ref: 1 }, { ref: 2 }];
    const result = ensureAssistantInlineCitationFallback(content, citations);

    // 跨语言轮流分配：至少注入一个引用编号
    expect(result).toMatch(/\[\d+\]/);
  });

  it('当正文已包含编号时，不应重复补充参考来源', () => {
    const content = '这是一个已有引用的回答[1]。';
    const citations = [{ ref: 1 }, { ref: 2 }];
    const result = ensureAssistantInlineCitationFallback(content, citations);
    expect(result).toBe(content);
  });

  it('当正文无编号且 citations 含 highlight_text 时，应按段落注入 [N]', () => {
    const content = '固定大小分块是一种传统方法，它按预设大小切分文档而不考虑语义内容。\n\n语义分块通过理解文本含义来切分文档，能保持上下文完整性。';
    const citations = [
      { ref: 1, highlight_text: '固定大小分块 传统方法 预设大小 切分文档', group_id: 'g1' },
      { ref: 2, highlight_text: '语义分块 理解文本含义 上下文完整性', group_id: 'g2' },
    ];
    const result = ensureAssistantInlineCitationFallback(content, citations);
    expect(result).toContain('[1]');
    expect(result).toContain('[2]');
    expect(result).not.toContain('参考来源');
  });
});

describe('optimizeAssistantInlineCitations', () => {
  it('句内引用与证据不相关时，应替换为更相关引用', () => {
    const content = '该方法通过全局光照建模提升物理鲁棒性[1]。';
    const citations = [
      { ref: 1, highlight_text: '社交媒体账号信息', group_id: 'noise' },
      { ref: 2, highlight_text: '全局光照建模 提升 物理 鲁棒性', group_id: 'group-2' },
    ];

    const optimized = optimizeAssistantInlineCitations(content, citations);
    expect(optimized).toContain('[2]');
    expect(optimized).not.toContain('[1]');
  });

  it('句内引用都无法支撑时，若 ref 存在于 citationMap 则保留（跨语言兜底）', () => {
    const content = '这是一句与来源都不相关的话[1][2]。';
    const citations = [
      { ref: 1, highlight_text: '苹果 香蕉 西瓜', group_id: 'g1' },
      { ref: 2, highlight_text: '东京 旅游 酒店', group_id: 'g2' },
    ];
    const optimized = optimizeAssistantInlineCitations(content, citations);
    // 跨语言场景下 token 重叠为 0，但 ref 在 citationMap 中有效，应保留而非剥离
    expect(optimized).toContain('[1]');
  });

  it('句内引用不在 citations 中时，应移除该句引用', () => {
    const content = '这是一句话[5][6]。';
    const citations = [
      { ref: 1, highlight_text: '苹果 香蕉 西瓜', group_id: 'g1' },
    ];
    const optimized = optimizeAssistantInlineCitations(content, citations);
    expect(optimized).not.toContain('[5]');
    expect(optimized).not.toContain('[6]');
  });
});

describe('filterCitationsByContentRefs', () => {
  it('应按正文引用顺序过滤 citations', () => {
    const content = '结论[2]，补充[1]。';
    const citations = [
      { ref: 1, group_id: 'g1' },
      { ref: 2, group_id: 'g2' },
      { ref: 3, group_id: 'g3' },
    ];
    const filtered = filterCitationsByContentRefs(content, citations);
    expect(filtered.map((c) => c.ref)).toEqual([2, 1]);
  });

  it('应优先使用 display_ref 过滤，并保留 source_ref 供证据联动', () => {
    const content = '结论[1]，补充[2]。';
    const citations = [
      { ref: 5, display_ref: 1, source_ref: 5, group_id: 'g5' },
      { ref: 2, display_ref: 2, source_ref: 2, group_id: 'g2' },
      { ref: 9, display_ref: 3, source_ref: 9, group_id: 'g9' },
    ];

    const filtered = filterCitationsByContentRefs(content, citations);

    expect(filtered.map((c) => c.ref)).toEqual([1, 2]);
    expect(filtered.map((c) => c.display_ref)).toEqual([1, 2]);
    expect(filtered.map((c) => c.source_ref)).toEqual([5, 2]);
  });
});

describe('finalizeThinkingDurationMs', () => {
  it('应统计到最后一个思考 chunk，而不是首个正文 chunk', () => {
    const duration = finalizeThinkingDurationMs({
      thinkingStartTime: 1000,
      thinkingLastUpdateTime: 2800,
      contentStartTime: 1800,
    });

    expect(duration).toBe(1800);
  });

  it('只有正文开始时间时，应回退到正文开始时间', () => {
    const duration = finalizeThinkingDurationMs({
      thinkingStartTime: 1000,
      thinkingLastUpdateTime: null,
      contentStartTime: 1600,
    });

    expect(duration).toBe(600);
  });
});

describe('useMessageState non-stream memory hits', () => {
  it('非流式响应应保存 memory hits 与 meta', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        answer: '回答正文',
        reasoning_content: '',
        retrieval_meta: { citations: [] },
        memory_hits: [{ id: 'mem-2', memory_kind: 'profile', title: '用户画像', summary: '偏好中文回答' }],
        memory_meta: { enabled: true, strategy: 'simple', selected_count: 1 },
      }),
    });

    const { result } = renderHook(() => useMessageState({
      ...createOptions(),
      globalSettings: {
        ...createOptions().globalSettings,
        streamOutput: false,
      },
    }));

    act(() => {
      result.current.textareaRef.current = createInputEl('非流式测试');
    });

    await act(async () => {
      await result.current.sendMessage();
    });

    const assistant = [...result.current.messages].reverse().find((m) => m.type === 'assistant');
    expect(assistant.memoryHits).toEqual([
      expect.objectContaining({ id: 'mem-2', memory_kind: 'profile' }),
    ]);
    expect(assistant.memoryMeta).toEqual(
      expect.objectContaining({ enabled: true, selected_count: 1 })
    );
  });
});
