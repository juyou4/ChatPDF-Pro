// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePDFState } from '../usePDFState';

describe('usePDFState', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ========== 初始状态 ==========

  it('初始状态值正确', () => {
    const { result } = renderHook(() => usePDFState());

    expect(result.current.currentPage).toBe(1);
    expect(result.current.pdfScale).toBe(1.0);
    expect(result.current.debouncedScale).toBe(1.0);
    expect(result.current.selectedText).toBe('');
    expect(result.current.showTextMenu).toBe(false);
    expect(result.current.menuPosition).toEqual({ x: 0, y: 0 });
    expect(result.current.searchQuery).toBe('');
    expect(result.current.searchResults).toEqual([]);
    expect(result.current.currentResultIndex).toBe(0);
    expect(result.current.isSearching).toBe(false);
    expect(result.current.searchHistory).toEqual([]);
    expect(result.current.activeHighlight).toBeNull();
  });

  // ========== 页码控制 ==========

  it('setCurrentPage 更新当前页码', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.setCurrentPage(5);
    });

    expect(result.current.currentPage).toBe(5);
  });

  // ========== 缩放防抖（需求 10.1）==========

  it('pdfScale 变化后 debouncedScale 延迟 150ms 更新', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.setPdfScale(2.0);
    });

    // 防抖期间 debouncedScale 仍为旧值
    expect(result.current.pdfScale).toBe(2.0);
    expect(result.current.debouncedScale).toBe(1.0);

    act(() => {
      vi.advanceTimersByTime(150);
    });

    // 防抖到期后更新
    expect(result.current.debouncedScale).toBe(2.0);
  });

  it('150ms 内多次缩放只取最后一次值', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => { result.current.setPdfScale(1.5); });
    act(() => { result.current.setPdfScale(2.0); });
    act(() => { result.current.setPdfScale(2.5); });

    // 防抖期间仍为初始值
    expect(result.current.debouncedScale).toBe(1.0);

    act(() => {
      vi.advanceTimersByTime(150);
    });

    // 只更新为最后一次的值
    expect(result.current.debouncedScale).toBe(2.5);
  });

  // ========== 文本选择 ==========

  it('setSelectedText 更新选中文本', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.setSelectedText('测试文本');
    });

    expect(result.current.selectedText).toBe('测试文本');
  });

  it('setMenuPosition 更新菜单位置', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.setMenuPosition({ x: 100, y: 200 });
    });

    expect(result.current.menuPosition).toEqual({ x: 100, y: 200 });
  });

  // ========== 搜索状态 ==========

  it('setSearchQuery 更新搜索关键词', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.setSearchQuery('关键词');
    });

    expect(result.current.searchQuery).toBe('关键词');
  });

  // ========== 文档切换时重置搜索状态 ==========

  it('docId 变化时重置搜索相关状态', () => {
    const { result, rerender } = renderHook(
      ({ docId }) => usePDFState({ docId }),
      { initialProps: { docId: 'doc1' } }
    );

    // 设置一些搜索状态
    act(() => {
      result.current.setSearchQuery('测试');
      result.current.setSearchResults([{ page: 1, chunk: '结果' }]);
      result.current.setCurrentResultIndex(2);
    });

    // 切换文档
    rerender({ docId: 'doc2' });

    expect(result.current.searchQuery).toBe('');
    expect(result.current.searchResults).toEqual([]);
    expect(result.current.currentResultIndex).toBe(0);
    expect(result.current.activeHighlight).toBeNull();
  });

  it('docId 变化时从 localStorage 加载搜索历史', () => {
    localStorage.setItem('search_history_doc1', JSON.stringify(['查询1', '查询2']));

    const { result } = renderHook(() => usePDFState({ docId: 'doc1' }));

    expect(result.current.searchHistory).toEqual(['查询1', '查询2']);
  });

  it('docId 为 null 时清空搜索历史', () => {
    const { result, rerender } = renderHook(
      ({ docId }) => usePDFState({ docId }),
      { initialProps: { docId: 'doc1' } }
    );

    rerender({ docId: null });

    expect(result.current.searchHistory).toEqual([]);
  });

  // ========== 高亮自动消失 ==========

  it('普通高亮 2500ms 后自动消失', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.setActiveHighlight({ page: 1, text: '测试', at: Date.now() });
    });

    expect(result.current.activeHighlight).not.toBeNull();

    act(() => {
      vi.advanceTimersByTime(2500);
    });

    expect(result.current.activeHighlight).toBeNull();
  });

  it('引用高亮 4000ms 后自动消失', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.setActiveHighlight({ page: 1, text: '引用', source: 'citation', at: Date.now() });
    });

    // 2500ms 时还在
    act(() => {
      vi.advanceTimersByTime(2500);
    });
    expect(result.current.activeHighlight).not.toBeNull();

    // 4000ms 时消失
    act(() => {
      vi.advanceTimersByTime(1500);
    });
    expect(result.current.activeHighlight).toBeNull();
  });

  // ========== handleCitationClick ==========

  it('handleCitationClick 跳转到引用页面并设置高亮', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.handleCitationClick({
        page_range: [3],
        highlight_text: '引用文本',
      });
    });

    expect(result.current.currentPage).toBe(3);

    // 等待 400ms 延迟高亮
    act(() => {
      vi.advanceTimersByTime(400);
    });

    expect(result.current.activeHighlight).toEqual(
      expect.objectContaining({
        page: 3,
        text: '引用文本',
        source: 'citation',
      })
    );
  });

  it('handleCitationClick 忽略无效引用', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.handleCitationClick(null);
    });
    expect(result.current.currentPage).toBe(1);

    act(() => {
      result.current.handleCitationClick({});
    });
    expect(result.current.currentPage).toBe(1);
  });

  // ========== focusResult ==========

  it('focusResult 跳转到搜索结果对应页面', () => {
    const { result } = renderHook(() =>
      usePDFState({ docInfo: { total_pages: 10 } })
    );

    const mockResults = [
      { page: 2, chunk: '结果1' },
      { page: 5, chunk: '结果2' },
      { page: 8, chunk: '结果3' },
    ];

    act(() => {
      result.current.focusResult(1, mockResults);
    });

    expect(result.current.currentPage).toBe(5);
    expect(result.current.currentResultIndex).toBe(1);
    expect(result.current.activeHighlight).toEqual(
      expect.objectContaining({ page: 5, text: '结果2' })
    );
  });

  it('focusResult 处理索引循环', () => {
    const { result } = renderHook(() =>
      usePDFState({ docInfo: { total_pages: 10 } })
    );

    const mockResults = [
      { page: 1, chunk: '结果1' },
      { page: 3, chunk: '结果2' },
    ];

    // 索引超出范围时循环
    act(() => {
      result.current.focusResult(2, mockResults);
    });

    expect(result.current.currentResultIndex).toBe(0);
    expect(result.current.currentPage).toBe(1);
  });

  it('focusResult 空结果时不操作', () => {
    const { result } = renderHook(() => usePDFState());

    act(() => {
      result.current.focusResult(0, []);
    });

    expect(result.current.currentPage).toBe(1);
    expect(result.current.activeHighlight).toBeNull();
  });

  // ========== formatSimilarity ==========

  it('formatSimilarity 返回 similarity_percent', () => {
    const { result } = renderHook(() => usePDFState());

    expect(result.current.formatSimilarity({ similarity_percent: 85.5 })).toBe(85.5);
  });

  it('formatSimilarity 从 score 计算百分比', () => {
    const { result } = renderHook(() => usePDFState());

    // score=0 -> 1/(1+0) = 100%
    expect(result.current.formatSimilarity({ score: 0 })).toBe(100);
    // score=1 -> 1/(1+1) = 50%
    expect(result.current.formatSimilarity({ score: 1 })).toBe(50);
  });

  // ========== resetPDFState ==========

  it('resetPDFState 重置所有 PDF 状态', () => {
    const { result } = renderHook(() => usePDFState());

    // 修改各种状态
    act(() => {
      result.current.setCurrentPage(5);
      result.current.setPdfScale(2.0);
      result.current.setSelectedText('选中');
      result.current.setSearchQuery('搜索');
      result.current.setSearchResults([{ page: 1 }]);
      result.current.setActiveHighlight({ page: 1, text: '高亮' });
    });

    // 重置
    act(() => {
      result.current.resetPDFState();
    });

    expect(result.current.currentPage).toBe(1);
    expect(result.current.pdfScale).toBe(1.0);
    expect(result.current.selectedText).toBe('');
    expect(result.current.searchQuery).toBe('');
    expect(result.current.searchResults).toEqual([]);
    expect(result.current.activeHighlight).toBeNull();
  });

  // ========== pdfContainerRef ==========

  it('pdfContainerRef 初始为 null', () => {
    const { result } = renderHook(() => usePDFState());
    expect(result.current.pdfContainerRef.current).toBeNull();
  });
});
