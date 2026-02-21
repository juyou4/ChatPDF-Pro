import { useState, useCallback } from 'react';

/**
 * UI 展示状态管理 Hook
 * 管理侧边栏、暗色模式、面板展开/收起、设置弹窗等 UI 展示状态
 *
 * 这些状态仅影响 UI 展示，不涉及业务逻辑或数据持久化。
 * 将它们从 ChatPDF 主组件中提取出来，使 UI 状态变更仅触发受影响的 UI 区域重渲染。
 */
export function useUIState() {
  // ========== 侧边栏与布局 ==========
  const [showSidebar, setShowSidebar] = useState(true);
  const [isHeaderExpanded, setIsHeaderExpanded] = useState(true);
  const [pdfPanelWidth, setPdfPanelWidth] = useState(50);

  // ========== 暗色模式 ==========
  const [darkMode, setDarkMode] = useState(false);

  // ========== 设置面板弹窗 ==========
  const [showSettings, setShowSettings] = useState(false);
  const [showEmbeddingSettings, setShowEmbeddingSettings] = useState(false);
  const [showOCRSettings, setShowOCRSettings] = useState(false);
  const [showGlobalSettings, setShowGlobalSettings] = useState(false);
  const [showChatSettings, setShowChatSettings] = useState(false);

  // ========== 其他 UI 开关 ==========
  const [enableThinking, setEnableThinking] = useState(false);

  // ========== 便捷方法 ==========

  /**
   * 切换侧边栏显示/隐藏
   */
  const toggleSidebar = useCallback(() => {
    setShowSidebar(prev => !prev);
  }, []);

  /**
   * 切换暗色模式
   */
  const toggleDarkMode = useCallback(() => {
    setDarkMode(prev => !prev);
  }, []);

  /**
   * 切换顶栏展开/收起
   */
  const toggleHeaderExpanded = useCallback(() => {
    setIsHeaderExpanded(prev => !prev);
  }, []);

  /**
   * 关闭所有设置面板
   */
  const closeAllSettings = useCallback(() => {
    setShowSettings(false);
    setShowEmbeddingSettings(false);
    setShowOCRSettings(false);
    setShowGlobalSettings(false);
    setShowChatSettings(false);
  }, []);

  return {
    // 侧边栏与布局
    showSidebar,
    setShowSidebar,
    isHeaderExpanded,
    setIsHeaderExpanded,
    pdfPanelWidth,
    setPdfPanelWidth,

    // 暗色模式
    darkMode,
    setDarkMode,

    // 设置面板弹窗
    showSettings,
    setShowSettings,
    showEmbeddingSettings,
    setShowEmbeddingSettings,
    showOCRSettings,
    setShowOCRSettings,
    showGlobalSettings,
    setShowGlobalSettings,
    showChatSettings,
    setShowChatSettings,

    // 其他 UI 开关
    enableThinking,
    setEnableThinking,

    // 便捷方法
    toggleSidebar,
    toggleDarkMode,
    toggleHeaderExpanded,
    closeAllSettings,
  };
}
