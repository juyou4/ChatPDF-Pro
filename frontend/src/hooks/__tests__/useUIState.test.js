// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useUIState } from '../useUIState';

describe('useUIState', () => {
  // --- 初始状态 ---

  it('初始状态正确', () => {
    const { result } = renderHook(() => useUIState());

    // 侧边栏与布局
    expect(result.current.showSidebar).toBe(true);
    expect(result.current.isHeaderExpanded).toBe(true);
    expect(result.current.pdfPanelWidth).toBe(50);

    // 暗色模式
    expect(result.current.darkMode).toBe(false);

    // 设置面板弹窗
    expect(result.current.showSettings).toBe(false);
    expect(result.current.showEmbeddingSettings).toBe(false);
    expect(result.current.showOCRSettings).toBe(false);
    expect(result.current.showGlobalSettings).toBe(false);
    expect(result.current.showChatSettings).toBe(false);

    // 其他 UI 开关
    expect(result.current.enableThinking).toBe(false);
  });

  // --- 切换方法 ---

  it('toggleSidebar 切换侧边栏状态', () => {
    const { result } = renderHook(() => useUIState());

    expect(result.current.showSidebar).toBe(true);
    act(() => result.current.toggleSidebar());
    expect(result.current.showSidebar).toBe(false);
    act(() => result.current.toggleSidebar());
    expect(result.current.showSidebar).toBe(true);
  });

  it('toggleDarkMode 切换暗色模式', () => {
    const { result } = renderHook(() => useUIState());

    expect(result.current.darkMode).toBe(false);
    act(() => result.current.toggleDarkMode());
    expect(result.current.darkMode).toBe(true);
    act(() => result.current.toggleDarkMode());
    expect(result.current.darkMode).toBe(false);
  });

  it('toggleHeaderExpanded 切换顶栏展开状态', () => {
    const { result } = renderHook(() => useUIState());

    expect(result.current.isHeaderExpanded).toBe(true);
    act(() => result.current.toggleHeaderExpanded());
    expect(result.current.isHeaderExpanded).toBe(false);
    act(() => result.current.toggleHeaderExpanded());
    expect(result.current.isHeaderExpanded).toBe(true);
  });

  // --- 设置面板 ---

  it('可以独立控制各设置面板', () => {
    const { result } = renderHook(() => useUIState());

    act(() => result.current.setShowSettings(true));
    expect(result.current.showSettings).toBe(true);
    expect(result.current.showEmbeddingSettings).toBe(false);

    act(() => result.current.setShowEmbeddingSettings(true));
    expect(result.current.showEmbeddingSettings).toBe(true);

    act(() => result.current.setShowOCRSettings(true));
    expect(result.current.showOCRSettings).toBe(true);

    act(() => result.current.setShowGlobalSettings(true));
    expect(result.current.showGlobalSettings).toBe(true);

    act(() => result.current.setShowChatSettings(true));
    expect(result.current.showChatSettings).toBe(true);
  });

  it('closeAllSettings 关闭所有设置面板', () => {
    const { result } = renderHook(() => useUIState());

    // 先打开所有面板
    act(() => {
      result.current.setShowSettings(true);
      result.current.setShowEmbeddingSettings(true);
      result.current.setShowOCRSettings(true);
      result.current.setShowGlobalSettings(true);
      result.current.setShowChatSettings(true);
    });

    // 一键关闭
    act(() => result.current.closeAllSettings());

    expect(result.current.showSettings).toBe(false);
    expect(result.current.showEmbeddingSettings).toBe(false);
    expect(result.current.showOCRSettings).toBe(false);
    expect(result.current.showGlobalSettings).toBe(false);
    expect(result.current.showChatSettings).toBe(false);
  });

  // --- PDF 面板宽度 ---

  it('setPdfPanelWidth 设置 PDF 面板宽度', () => {
    const { result } = renderHook(() => useUIState());

    act(() => result.current.setPdfPanelWidth(60));
    expect(result.current.pdfPanelWidth).toBe(60);

    act(() => result.current.setPdfPanelWidth(30));
    expect(result.current.pdfPanelWidth).toBe(30);
  });

  // --- 状态隔离验证 ---

  it('UI 状态变更互不影响', () => {
    const { result } = renderHook(() => useUIState());

    // 切换暗色模式不影响侧边栏
    act(() => result.current.toggleDarkMode());
    expect(result.current.darkMode).toBe(true);
    expect(result.current.showSidebar).toBe(true);
    expect(result.current.isHeaderExpanded).toBe(true);

    // 收起侧边栏不影响暗色模式
    act(() => result.current.toggleSidebar());
    expect(result.current.showSidebar).toBe(false);
    expect(result.current.darkMode).toBe(true);

    // 打开设置面板不影响布局状态
    act(() => result.current.setShowSettings(true));
    expect(result.current.showSettings).toBe(true);
    expect(result.current.showSidebar).toBe(false);
    expect(result.current.darkMode).toBe(true);
  });

  // --- enableThinking ---

  it('setEnableThinking 控制思考模式开关', () => {
    const { result } = renderHook(() => useUIState());

    act(() => result.current.setEnableThinking(true));
    expect(result.current.enableThinking).toBe(true);

    act(() => result.current.setEnableThinking(false));
    expect(result.current.enableThinking).toBe(false);
  });
});
