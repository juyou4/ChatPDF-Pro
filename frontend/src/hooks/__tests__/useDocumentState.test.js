// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useDocumentState } from '../useDocumentState';

// 模拟 fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

// 模拟 alert 和 confirm
global.alert = vi.fn();
global.confirm = vi.fn();

describe('useDocumentState', () => {
  let mockSetMessages;
  let mockSetCurrentPage;
  let mockSetScreenshots;
  let mockSetIsLoading;
  let mockSetSelectedText;
  let mockGetCurrentProvider;
  let mockGetCurrentEmbeddingModel;

  beforeEach(() => {
    localStorage.clear();
    mockFetch.mockReset();
    vi.mocked(global.alert).mockReset();
    vi.mocked(global.confirm).mockReset();

    mockSetMessages = vi.fn();
    mockSetCurrentPage = vi.fn();
    mockSetScreenshots = vi.fn();
    mockSetIsLoading = vi.fn();
    mockSetSelectedText = vi.fn();
    mockGetCurrentProvider = vi.fn(() => null);
    mockGetCurrentEmbeddingModel = vi.fn(() => null);

    // 默认 fetch 返回空响应
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });
  });

  const defaultOptions = () => ({
    getCurrentProvider: mockGetCurrentProvider,
    getCurrentEmbeddingModel: mockGetCurrentEmbeddingModel,
    setMessages: mockSetMessages,
    setCurrentPage: mockSetCurrentPage,
    setScreenshots: mockSetScreenshots,
    setIsLoading: mockSetIsLoading,
    setSelectedText: mockSetSelectedText,
  });

  // --- 初始状态 ---

  it('初始状态正确', () => {
    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    expect(result.current.docId).toBeNull();
    expect(result.current.docInfo).toBeNull();
    expect(result.current.isUploading).toBe(false);
    expect(result.current.uploadProgress).toBe(0);
    expect(result.current.uploadStatus).toBe('uploading');
    expect(result.current.history).toEqual([]);
  });

  // --- 会话历史加载 ---

  it('初始化时从 localStorage 加载会话历史', () => {
    const historyData = [
      { id: 'doc1', docId: 'doc1', filename: 'test.pdf', messages: [] },
    ];
    localStorage.setItem('chatHistory', JSON.stringify(historyData));

    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    expect(result.current.history).toEqual(historyData);
  });

  it('localStorage 无历史数据时 history 为空数组', () => {
    const { result } = renderHook(() => useDocumentState(defaultOptions()));
    expect(result.current.history).toEqual([]);
  });

  // --- startNewChat ---

  it('startNewChat 重置文档状态并调用跨域回调', () => {
    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    // 先设置一些状态
    act(() => {
      result.current.setDocId('some-doc');
      result.current.setDocInfo({ filename: 'test.pdf' });
    });

    // 调用 startNewChat
    act(() => {
      result.current.startNewChat();
    });

    expect(result.current.docId).toBeNull();
    expect(result.current.docInfo).toBeNull();
    expect(mockSetMessages).toHaveBeenCalledWith([]);
    expect(mockSetCurrentPage).toHaveBeenCalledWith(1);
    expect(mockSetSelectedText).toHaveBeenCalledWith('');
    expect(mockSetScreenshots).toHaveBeenCalledWith([]);
  });

  // --- deleteSession ---

  it('deleteSession 用户确认后删除会话并更新 localStorage', () => {
    vi.mocked(global.confirm).mockReturnValue(true);
    const historyData = [
      { id: 'doc1', docId: 'doc1', filename: 'a.pdf' },
      { id: 'doc2', docId: 'doc2', filename: 'b.pdf' },
    ];
    localStorage.setItem('chatHistory', JSON.stringify(historyData));

    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    act(() => {
      result.current.deleteSession('doc1');
    });

    // 历史中只剩 doc2
    expect(result.current.history).toEqual([
      { id: 'doc2', docId: 'doc2', filename: 'b.pdf' },
    ]);
    // localStorage 也更新了
    const stored = JSON.parse(localStorage.getItem('chatHistory'));
    expect(stored).toEqual([{ id: 'doc2', docId: 'doc2', filename: 'b.pdf' }]);
  });

  it('deleteSession 用户取消时不执行删除', () => {
    vi.mocked(global.confirm).mockReturnValue(false);
    const historyData = [{ id: 'doc1', docId: 'doc1', filename: 'a.pdf' }];
    localStorage.setItem('chatHistory', JSON.stringify(historyData));

    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    act(() => {
      result.current.deleteSession('doc1');
    });

    // 历史未变
    expect(result.current.history).toEqual(historyData);
  });

  it('deleteSession 删除当前文档时重置 docId 和 docInfo', () => {
    vi.mocked(global.confirm).mockReturnValue(true);
    localStorage.setItem('chatHistory', JSON.stringify([{ id: 'doc1' }]));

    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    // 设置当前文档
    act(() => {
      result.current.setDocId('doc1');
      result.current.setDocInfo({ filename: 'test.pdf' });
    });

    act(() => {
      result.current.deleteSession('doc1');
    });

    expect(result.current.docId).toBeNull();
    expect(result.current.docInfo).toBeNull();
    expect(mockSetMessages).toHaveBeenCalledWith([]);
  });

  // --- saveCurrentSession ---

  it('saveCurrentSession 保存当前会话到 localStorage', () => {
    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    act(() => {
      result.current.setDocId('doc1');
      result.current.setDocInfo({ filename: 'test.pdf' });
    });

    const messages = [{ type: 'user', content: 'hello' }];
    act(() => {
      result.current.saveCurrentSession(messages);
    });

    const stored = JSON.parse(localStorage.getItem('chatHistory'));
    expect(stored).toHaveLength(1);
    expect(stored[0].docId).toBe('doc1');
    expect(stored[0].filename).toBe('test.pdf');
    expect(stored[0].messages).toEqual(messages);
  });

  it('saveCurrentSession 无 docId 时不保存', () => {
    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    act(() => {
      result.current.saveCurrentSession([]);
    });

    expect(localStorage.getItem('chatHistory')).toBeNull();
  });

  it('saveCurrentSession 更新已有会话而非重复添加', () => {
    localStorage.setItem('chatHistory', JSON.stringify([
      { id: 'doc1', docId: 'doc1', filename: 'old.pdf', messages: [], createdAt: 1000 },
    ]));

    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    act(() => {
      result.current.setDocId('doc1');
      result.current.setDocInfo({ filename: 'test.pdf' });
    });

    act(() => {
      result.current.saveCurrentSession([{ type: 'user', content: 'updated' }]);
    });

    const stored = JSON.parse(localStorage.getItem('chatHistory'));
    expect(stored).toHaveLength(1);
    expect(stored[0].messages[0].content).toBe('updated');
    // 保留原始 createdAt
    expect(stored[0].createdAt).toBe(1000);
  });

  // --- loadSession ---

  it('loadSession 加载会话并设置文档状态', async () => {
    const docData = { filename: 'loaded.pdf', total_pages: 5 };

    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    // 等待初始化 effect 完成后再设置 mock
    await act(async () => {});
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => docData,
    });

    await act(async () => {
      await result.current.loadSession({ docId: 'doc1', messages: [{ type: 'user', content: 'hi' }] });
    });

    expect(result.current.docId).toBe('doc1');
    expect(result.current.docInfo).toEqual(docData);
    expect(mockSetMessages).toHaveBeenCalledWith([{ type: 'user', content: 'hi' }]);
    expect(mockSetCurrentPage).toHaveBeenCalledWith(1);
    expect(mockSetIsLoading).toHaveBeenCalledWith(true);
    expect(mockSetIsLoading).toHaveBeenCalledWith(false);
  });

  it('loadSession 请求失败时不更新状态', async () => {
    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    // 等待初始化 effect 完成后再设置 mock
    await act(async () => {});
    mockFetch.mockResolvedValueOnce({ ok: false });

    await act(async () => {
      await result.current.loadSession({ docId: 'doc1', messages: [] });
    });

    expect(result.current.docId).toBeNull();
    expect(mockSetIsLoading).toHaveBeenCalledWith(false);
  });

  // --- fetchStorageInfo ---

  it('fetchStorageInfo 获取存储信息', async () => {
    const storageData = { total: 100, used: 50 };

    const { result } = renderHook(() => useDocumentState(defaultOptions()));

    // 等待初始化 effect 完成后再设置 mock
    await act(async () => {});
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => storageData,
    });

    await act(async () => {
      await result.current.fetchStorageInfo();
    });

    expect(result.current.storageInfo).toEqual(storageData);
  });

  // --- 无回调时不崩溃 ---

  it('不传跨域回调时 startNewChat 不崩溃', () => {
    const { result } = renderHook(() => useDocumentState({
      getCurrentProvider: mockGetCurrentProvider,
      getCurrentEmbeddingModel: mockGetCurrentEmbeddingModel,
    }));

    expect(() => {
      act(() => {
        result.current.startNewChat();
      });
    }).not.toThrow();
  });
});
