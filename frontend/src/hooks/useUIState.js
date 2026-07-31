import { useState, useCallback, useEffect } from 'react';

const OVERVIEW_DEPTHS = new Set(['brief', 'standard', 'detailed']);
const THEME_STORAGE_KEY = 'chatpdf-theme';
export const NARROW_DESKTOP_MEDIA_QUERY = '(max-width: 1239px)';

const getIsNarrowDesktop = () => (
  typeof window !== 'undefined'
  && typeof window.matchMedia === 'function'
  && window.matchMedia(NARROW_DESKTOP_MEDIA_QUERY).matches
);

const getInitialDarkMode = () => {
  if (typeof window === 'undefined') return false;

  try {
    const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (savedTheme === 'dark') return true;
    if (savedTheme === 'light') return false;
  } catch {
    // 存储不可用时继续读取系统主题。
  }

  return typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-color-scheme: dark)').matches;
};

/**
 * UI 展示状态管理 Hook
 * 管理侧边栏、暗色模式、面板展开/收起、设置弹窗等 UI 展示状态
 *
 * 这些状态仅影响 UI 展示，不涉及业务逻辑或数据持久化。
 * 将它们从 ChatPDF 主组件中提取出来，使 UI 状态变更仅触发受影响的 UI 区域重渲染。
 */
export function useUIState() {
  // ========== 侧边栏与布局 ==========
  const [isNarrowDesktop, setIsNarrowDesktop] = useState(getIsNarrowDesktop);
  const [showSidebar, setShowSidebar] = useState(() => !getIsNarrowDesktop());
  const [isHeaderExpanded, setIsHeaderExpanded] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [pdfPanelWidth, setPdfPanelWidth] = useState(50);

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined;

    const mediaQuery = window.matchMedia(NARROW_DESKTOP_MEDIA_QUERY);
    const handleChange = (event) => {
      setIsNarrowDesktop(event.matches);
      if (event.matches) setShowSidebar(false);
    };

    setIsNarrowDesktop(mediaQuery.matches);
    if (mediaQuery.matches) setShowSidebar(false);
    mediaQuery.addEventListener?.('change', handleChange);
    return () => mediaQuery.removeEventListener?.('change', handleChange);
  }, []);

  // ========== 暗色模式 ==========
  const [darkMode, setDarkMode] = useState(getInitialDarkMode);

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.classList.toggle('dark', darkMode);
      document.documentElement.style.colorScheme = darkMode ? 'dark' : 'light';
    }

    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, darkMode ? 'dark' : 'light');
    } catch {
      // 无痕模式或存储配额异常不影响主题切换。
    }
  }, [darkMode]);

  // ========== 设置面板弹窗 ==========
  const [showSettings, setShowSettings] = useState(false);
  const [showEmbeddingSettings, setShowEmbeddingSettings] = useState(false);
  const [showOCRSettings, setShowOCRSettings] = useState(false);
  const [showGlobalSettings, setShowGlobalSettings] = useState(false);
  const [showChatSettings, setShowChatSettings] = useState(false);

  // ========== 其他 UI 开关 ==========
  const [enableThinking, setEnableThinking] = useState(false);

  // ========== 速览（Overview）功能 ==========
  const [rightPanelMode, setRightPanelMode] = useState('chat'); // 'overview' | 'analysis' | 'chat'
  const [overviewDepth, setOverviewDepthState] = useState(() => {
    try {
      const saved = localStorage.getItem('overviewDepth');
      return OVERVIEW_DEPTHS.has(saved) ? saved : 'standard';
    } catch {
      return 'standard';
    }
  }); // 'brief' | 'standard' | 'detailed'

  const setOverviewDepth = useCallback((value) => {
    const nextValue = OVERVIEW_DEPTHS.has(value) ? value : 'standard';
    setOverviewDepthState(nextValue);
    try {
      localStorage.setItem('overviewDepth', nextValue);
    } catch {
      // 忽略无痕模式或存储不可用场景
    }
  }, []);

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
    isNarrowDesktop,
    showSidebar,
    setShowSidebar,
    isHeaderExpanded,
    setIsHeaderExpanded,
    sidebarWidth,
    setSidebarWidth,
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

    // 速览（Overview）功能
    rightPanelMode,
    setRightPanelMode,
    overviewDepth,
    setOverviewDepth,

    // 便捷方法
    toggleSidebar,
    toggleDarkMode,
    toggleHeaderExpanded,
    closeAllSettings,
  };
}
