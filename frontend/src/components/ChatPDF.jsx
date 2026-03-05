import React, { useState, useRef, useEffect, useMemo, useCallback, lazy, Suspense } from 'react';
import { Upload, Send, FileText, Settings, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Copy, Bot, X, Crop, Image as ImageIcon, History, Moon, Sun, Plus, MessageSquare, Trash2, Menu, Type, ChevronUp, ChevronDown, Search, Loader2, Wand2, Server, Database, ListFilter, ArrowUpRight, SlidersHorizontal, Paperclip, ScanText, Scan, Brain, MessageCircle, ArrowUpDown, Globe } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { supportsVision } from '../utils/visionDetectorUtils';
import ScreenshotPreview from './ScreenshotPreview';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import PDFViewer from './PDFViewer';
import StreamingMarkdown from './StreamingMarkdown';
import TextSelectionToolbar from './TextSelectionToolbar';
import { useProvider } from '../contexts/ProviderContext';
import { useModel } from '../contexts/ModelContext';
import { useDefaults } from '../contexts/DefaultsContext';
import { useCapabilities } from '../contexts/CapabilitiesContext';
const EmbeddingSettings = lazy(() => import('./EmbeddingSettings'));
const OCRSettingsPanel = lazy(() => import('./OCRSettingsPanel'));
const GlobalSettings = lazy(() => import('./GlobalSettings'));
const ChatSettings = lazy(() => import('./ChatSettings'));
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';
import { useChatParams } from '../contexts/ChatParamsContext';
import { useDebouncedLocalStorage } from '../hooks/useDebouncedLocalStorage';
import { useUIState } from '../hooks/useUIState';
import { useDocumentState } from '../hooks/useDocumentState';
import { useMessageState } from '../hooks/useMessageState';
import { usePDFState } from '../hooks/usePDFState';
import { useScreenshotState } from '../hooks/useScreenshotState';
import PresetQuestions from './PresetQuestions';
import ModelQuickSwitch from './ModelQuickSwitch';
import ThinkingBlock from './ThinkingBlock';
import EvidencePanel from './EvidencePanel';
import MindmapView from './MindmapView';
import VirtualMessageList from './VirtualMessageList';
import WebSearchButton from './WebSearchButton';

const WebSearchSourcesBadge = ({ sources }) => {
  const [expanded, setExpanded] = useState(false);
  if (!sources || sources.length === 0) return null;
  return (
    <div className="mt-3 border-t border-gray-100 pt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-purple-600 hover:text-purple-800 transition-colors font-medium"
      >
        <Globe className="w-3.5 h-3.5" />
        <span>联网搜索来源 ({sources.length})</span>
        <ChevronDown className={`w-3 h-3 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>
      {expanded && (
        <div className="mt-2 space-y-1.5">
          {sources.map((src, i) => (
            <a
              key={i}
              href={src.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-start gap-2 p-2 rounded-lg bg-purple-50/50 hover:bg-purple-50 transition-colors group"
            >
              <span className="text-[10px] font-bold text-purple-500 bg-purple-100 rounded px-1 py-0.5 mt-0.5 flex-shrink-0">{i + 1}</span>
              <div className="min-w-0">
                <div className="text-xs font-medium text-gray-800 truncate group-hover:text-purple-700">{src.title}</div>
                {src.snippet && <div className="text-[11px] text-gray-500 line-clamp-2 mt-0.5">{src.snippet}</div>}
              </div>
              <ArrowUpRight className="w-3 h-3 text-gray-400 group-hover:text-purple-500 flex-shrink-0 mt-0.5" />
            </a>
          ))}
        </div>
      )}
    </div>
  );
};

const SendIcon = () => (
  <svg className="glass-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m6.998 10.247l.435.76c.277.485.415.727.415.993s-.138.508-.415.992l-.435.761c-1.238 2.167-1.857 3.25-1.375 3.788c.483.537 1.627.037 3.913-.963l6.276-2.746c1.795-.785 2.693-1.178 2.693-1.832s-.898-1.047-2.693-1.832L9.536 7.422c-2.286-1-3.43-1.5-3.913-.963s.137 1.62 1.375 3.788Z" />
  </svg>
);

const PauseIcon = () => (
  <svg className="glass-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="7" y="6" width="4" height="12" rx="2" />
    <rect x="13" y="6" width="4" height="12" rx="2" />
  </svg>
);

const UPLOAD_RING_CONFIGS = [
  { s: 298, w: 14, c: 'rgba(100, 50, 255, 0.5)',  br: '52% 48% 55% 45% / 48% 52% 48% 52%', dur: 4.2, del: -2.1, dir: 'normal',  mix: 'screen' },
  { s: 302, w: 22, c: 'rgba(50, 150, 255, 0.5)',  br: '45% 55% 48% 52% / 55% 45% 52% 48%', dur: 6.8, del: -4.3, dir: 'reverse', mix: 'screen' },
  { s: 295, w: 17, c: 'rgba(0, 200, 255, 0.4)',   br: '58% 42% 45% 55% / 42% 58% 48% 52%', dur: 3.5, del: -1.7, dir: 'normal',  mix: 'overlay' },
  { s: 304, w: 20, c: 'rgba(255, 100, 50, 0.5)',  br: '48% 52% 52% 48% / 58% 42% 55% 45%', dur: 7.3, del: -3.6, dir: 'reverse', mix: 'screen' },
  { s: 293, w: 13, c: 'rgba(255, 200, 50, 0.4)',  br: '55% 45% 48% 52% / 45% 55% 42% 58%', dur: 5.1, del: -0.8, dir: 'normal',  mix: 'screen' },
  { s: 301, w: 19, c: 'rgba(150, 50, 200, 0.5)',  br: '42% 58% 55% 45% / 52% 48% 58% 42%', dur: 4.7, del: -2.9, dir: 'reverse', mix: 'overlay' },
  { s: 297, w: 16, c: 'rgba(100, 50, 255, 0.4)',  br: '50% 50% 52% 48% / 55% 45% 50% 50%', dur: 6.2, del: -4.8, dir: 'normal',  mix: 'screen' },
  { s: 303, w: 23, c: 'rgba(50, 150, 255, 0.4)',  br: '46% 54% 50% 50% / 48% 52% 45% 55%', dur: 3.8, del: -1.2, dir: 'reverse', mix: 'screen' },
  { s: 299, w: 15, c: 'rgba(0, 200, 255, 0.5)',   br: '53% 47% 46% 54% / 50% 50% 53% 47%', dur: 7.6, del: -3.1, dir: 'normal',  mix: 'overlay' },
  { s: 305, w: 21, c: 'rgba(255, 100, 50, 0.4)',  br: '49% 51% 53% 47% / 46% 54% 49% 51%', dur: 4.9, del: -2.4, dir: 'reverse', mix: 'screen' },
  { s: 292, w: 18, c: 'rgba(255, 200, 50, 0.5)',  br: '57% 43% 49% 51% / 53% 47% 46% 54%', dur: 5.5, del: -0.5, dir: 'normal',  mix: 'screen' },
  { s: 300, w: 12, c: 'rgba(150, 50, 200, 0.4)',  br: '44% 56% 51% 49% / 57% 43% 52% 48%', dur: 6.5, del: -4.0, dir: 'reverse', mix: 'overlay' },
];

const ChatPDF = () => {
  // ========== Context Hooks ==========
  const { getProviderById } = useProvider();
  const { getModelById } = useModel();
  const { getDefaultModel } = useDefaults();
  const { hasLocalRerank } = useCapabilities();
  const globalSettings = useGlobalSettings();
  const { setReasoningEffort, reasoningEffort } = globalSettings;
  const { sendShortcut, confirmDeleteMessage, confirmRegenerateMessage, messageStyle, messageFontSize, codeCollapsible, codeWrappable, codeShowLineNumbers } = useChatParams();

  // ========== 设置状态 - 使用防抖 localStorage 写入（需求 8.1） ==========
  const [apiKey, setApiKey] = useDebouncedLocalStorage('apiKey', '');
  const [apiProvider, setApiProvider] = useDebouncedLocalStorage('apiProvider', 'openai');
  const [model, setModel] = useDebouncedLocalStorage('model', 'gpt-4o');
  const [embeddingApiKey, setEmbeddingApiKey] = useDebouncedLocalStorage('embeddingApiKey', '');
  const [enableVectorSearch, setEnableVectorSearch] = useDebouncedLocalStorage('enableVectorSearch', false);
  const [enableScreenshot, setEnableScreenshot] = useDebouncedLocalStorage('enableScreenshot', true);
  const [streamSpeed, setStreamSpeed] = useDebouncedLocalStorage('streamSpeed', 'normal');
  const [enableBlurReveal, setEnableBlurReveal] = useDebouncedLocalStorage('enableBlurReveal', true);
  const [blurIntensity, setBlurIntensity] = useDebouncedLocalStorage('blurIntensity', 'medium');
  const [searchEngine, setSearchEngine] = useDebouncedLocalStorage('searchEngine', 'google');
  const [searchEngineUrl, setSearchEngineUrl] = useDebouncedLocalStorage('searchEngineUrl', 'https://www.google.com/search?q={query}');
  const [toolbarSize, setToolbarSize] = useDebouncedLocalStorage('toolbarSize', 'normal');
  const [toolbarScale, setToolbarScale] = useDebouncedLocalStorage('toolbarScale', 1);
  const [useRerankSetting, setUseRerankSetting] = useDebouncedLocalStorage('useRerank', true);
  const [rerankerModel, setRerankerModel] = useDebouncedLocalStorage('rerankerModel', 'BAAI/bge-reranker-base');
  const [enableGraphRAG, setEnableGraphRAG] = useDebouncedLocalStorage('enableGraphRAG', false);
  const [enableJiebaBM25, setEnableJiebaBM25] = useDebouncedLocalStorage('enableJiebaBM25', true);
  const [numExpandContextChunk, setNumExpandContextChunk] = useDebouncedLocalStorage('numExpandContextChunk', 1);

  // 不需要持久化的设置状态
  const [availableModels, setAvailableModels] = useState({});
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState({});
  const [toolbarPosition, setToolbarPosition] = useState({ x: 0, y: 0 });

  // ========== UI 状态 Hook（需求 1.3） ==========
  const {
    showSidebar, setShowSidebar,
    isHeaderExpanded, setIsHeaderExpanded,
    pdfPanelWidth, setPdfPanelWidth,
    darkMode, setDarkMode,
    showSettings, setShowSettings,
    showEmbeddingSettings, setShowEmbeddingSettings,
    showOCRSettings, setShowOCRSettings,
    showGlobalSettings, setShowGlobalSettings,
    showChatSettings, setShowChatSettings,
    enableThinking, setEnableThinking,
  } = useUIState();

  // ========== 模型/凭证辅助函数 ==========
  const getEmbeddingConfig = useCallback(() => {
    const emk = getDefaultModel('embeddingModel');
    if (!emk) {
      return { isValid: false, reason: 'not_selected' };
    }

    const [pid, ...rest] = emk.split(':');
    const modelId = rest.join(':');
    const provider = getProviderById(pid);
    if (!provider) {
      return { isValid: false, reason: 'provider_missing', compositeKey: emk, providerId: pid, modelId };
    }

    const modelObj = modelId ? getModelById(modelId, pid) : null;
    if (!modelObj) {
      return { isValid: false, reason: 'model_not_found', compositeKey: emk, providerId: pid, modelId, provider };
    }
    if (modelObj.type !== 'embedding') {
      return {
        isValid: false,
        reason: 'wrong_type',
        compositeKey: emk,
        providerId: pid,
        modelId,
        modelType: modelObj.type,
        provider,
      };
    }

    return {
      isValid: true,
      compositeKey: emk,
      providerId: pid,
      modelId,
      model: modelObj,
      provider,
    };
  }, [getDefaultModel, getProviderById, getModelById]);

  const getCurrentChatModel = useCallback(() => {
    const chatKey = getDefaultModel('assistantModel');
    if (chatKey) {
      const [pid, mid] = chatKey.split(':');
      return { providerId: pid, modelId: mid };
    }
    return { providerId: apiProvider, modelId: model };
  }, [getDefaultModel, apiProvider, model]);

  const getChatCredentials = useCallback(() => {
    const chatKey = getDefaultModel('assistantModel');
    const { providerId, modelId } = getCurrentChatModel();
    const provider = getProviderById(providerId);
    if (chatKey) {
      return { providerId, modelId, apiKey: provider?.apiKey || '' };
    }
    return { providerId, modelId, apiKey: provider?.apiKey || apiKey };
  }, [getDefaultModel, getCurrentChatModel, getProviderById, apiKey]);

  const getCurrentRerankModel = useCallback(() => {
    const rrk = getDefaultModel('rerankModel');
    if (rrk) {
      const [pid, mid] = rrk.split(':');
      return { providerId: pid, modelId: mid };
    }
    // 没有配置 rerank 模型时，仅在本地 rerank 可用时才 fallback 到本地
    if (hasLocalRerank) {
      return { providerId: 'local', modelId: 'BAAI/bge-reranker-base' };
    }
    return null;
  }, [getDefaultModel, hasLocalRerank]);

  const getRerankCredentials = useCallback(() => {
    const rerankModel = getCurrentRerankModel();
    if (!rerankModel) return null;
    const { providerId, modelId } = rerankModel;
    const provider = getProviderById(providerId);
    return { providerId, modelId, apiKey: provider?.apiKey || embeddingApiKey || apiKey };
  }, [getCurrentRerankModel, getProviderById, embeddingApiKey, apiKey]);

  const getDefaultModelLabel = useCallback((key, fallback = '未选择') => {
    if (!key) return fallback;
    const [pid, mid] = key.split(':');
    const p = getProviderById(pid);
    const m = getModelById(mid, pid);
    return `${p?.name || pid} - ${m?.name || mid}`;
  }, [getProviderById, getModelById]);

  const currentChatModelObj = useMemo(() => {
    const chatKey = getDefaultModel('assistantModel');
    if (!chatKey || !chatKey.includes(':')) return null;
    const [pid, mid] = chatKey.split(':');
    return getModelById(mid, pid);
  }, [getDefaultModel, getModelById]);

  const isVisionCapable = useMemo(() => supportsVision(currentChatModelObj), [currentChatModelObj]);

  // ========== 文档状态 Hook（需求 1.1） ==========
  // useDocumentState 内部管理 docId/docInfo，需要其他 Hook 的 setter 函数
  // setter 函数通过 ref 桥接，避免 Hook 调用顺序问题
  const messageSettersRef = useRef({});
  const pdfSettersRef = useRef({});
  const screenshotSettersRef = useRef({});

  const documentState = useDocumentState({
    getEmbeddingConfig,
    setMessages: (...args) => messageSettersRef.current.setMessages?.(...args),
    setCurrentPage: (...args) => pdfSettersRef.current.setCurrentPage?.(...args),
    setScreenshots: (...args) => screenshotSettersRef.current.setScreenshots?.(...args),
    setIsLoading: (...args) => messageSettersRef.current.setIsLoading?.(...args),
    setSelectedText: (...args) => pdfSettersRef.current.setSelectedText?.(...args),
  });
  const {
    docId, setDocId,
    docInfo, setDocInfo,
    isUploading, uploadProgress, uploadStatus,
    history, storageInfo,
    fileInputRef,
    handleFileUpload, startNewChat, loadSession, deleteSession,
    saveCurrentSession, fetchStorageInfo,
  } = documentState;

  // ========== PDF 状态 Hook（需求 1.1） ==========
  const pdfState = usePDFState({
    docId,
    docInfo,
    useRerank: useRerankSetting,
    rerankerModel,
    getRerankCredentials,
    embeddingApiKey,
    apiKey,
  });
  const {
    currentPage, setCurrentPage,
    pdfScale, setPdfScale,
    selectedText, setSelectedText,
    showTextMenu, setShowTextMenu,
    menuPosition, setMenuPosition,
    searchQuery, setSearchQuery,
    searchResults,
    currentResultIndex,
    isSearching,
    searchHistory,
    activeHighlight, setActiveHighlight,
    pdfContainerRef,
    handleSearch, focusResult, handleCitationClick,
    formatSimilarity, renderHighlightedSnippet,
  } = pdfState;

  // ========== 截图状态 Hook（需求 1.1） ==========
  // textareaRef 来自 useMessageState（后续初始化），通过代理 ref 桥接
  const textareaRefProxy = useRef(null);
  const screenshotState = useScreenshotState({
    pdfContainerRef,
    textareaRef: textareaRefProxy,
    isVisionCapable,
    setInputValue: (...args) => messageSettersRef.current.setInputValue?.(...args),
    sendMessage: (...args) => messageSettersRef.current.sendMessage?.(...args),
  });
  const {
    screenshots,
    isSelectingArea, setIsSelectingArea,
    handleAreaSelected, handleSelectionCancel,
    handleScreenshotAction, handleScreenshotClose,
  } = screenshotState;

  // ========== 消息状态 Hook（需求 1.2） ==========
  const messageState = useMessageState({
    docId,
    screenshots,
    selectedText,
    getChatCredentials,
    getCurrentChatModel,
    getProviderById,
    streamSpeed,
    enableVectorSearch,
    enableGraphRAG,
    enableJiebaBM25,
    numExpandContextChunk,
    enableBlurReveal,
    blurIntensity,
    globalSettings,
  });
  const {
    messages, setMessages,
    isLoading, setIsLoading,
    hasInput, setHasInput,
    streamingMessageId,
    lastCallInfo,
    copiedMessageId,
    likedMessages, rememberedMessages,
    messagesEndRef, textareaRef,
    sendMessage, handleStop,
    regenerateMessage, copyMessage, saveToMemory,
    setInputValue,
    // ref 直写模式：流式输出期间直接更新 DOM
    streamingContentRef,
    streamingThinkingRef,
  } = messageState;

  // 双向联动：当前高亮的引文编号
  const [activeCitationRef, setActiveCitationRef] = useState(null);

  // 用户反馈
  const [feedbackTarget, setFeedbackTarget] = useState(null); // {idx, msg}
  const [dislikedMessages, setDislikedMessages] = useState(new Set());

  // 将 setter 函数注册到 ref 桥接对象，供 useDocumentState 和 useScreenshotState 使用
  messageSettersRef.current = { setMessages, setIsLoading, setInputValue, sendMessage };
  pdfSettersRef.current = { setCurrentPage, setSelectedText };
  screenshotSettersRef.current = { setScreenshots: screenshotState.setScreenshots };
  // 同步 textareaRef 代理，使截图 Hook 能正确聚焦输入框
  textareaRefProxy.current = textareaRef?.current ?? null;

  // ========== Refs ==========
  const chatPaneRef = useRef(null);
  const headerContentRef = useRef(null);
  const [headerHeight, setHeaderHeight] = useState(null);

  // ========== 副作用 ==========
  useEffect(() => {
    fetchAvailableModels();
    fetchAvailableEmbeddingModels();
  }, []);

  // 注意：原有的 localStorage 批量写入 useEffect 已被 useDebouncedLocalStorage 替代
  // lastCallInfo 仍需单独处理
  useEffect(() => {
    if (lastCallInfo) localStorage.setItem('lastCallInfo', JSON.stringify(lastCallInfo));
  }, [lastCallInfo]);

  useEffect(() => {
    if (Object.keys(availableModels).length === 0) return;
    const providerModels = availableModels[apiProvider]?.models;
    if (providerModels && !providerModels[model]) {
      const first = Object.keys(providerModels)[0];
      if (first) setModel(first);
    }
  }, [availableModels, apiProvider]);

  // 文档变更时保存会话
  useEffect(() => {
    if (docId && docInfo) saveCurrentSession(messages);
  }, [docId, docInfo, messages]);

  // 顶栏高度测量
  useEffect(() => {
    const el = headerContentRef.current;
    if (!el) return;
    const measure = () => setHeaderHeight(el.getBoundingClientRect().height);
    measure();
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(measure);
      observer.observe(el);
      return () => observer.disconnect();
    }
  }, [docId, docInfo, searchResults.length, useRerankSetting, darkMode]);

  // ========== 数据获取函数（useCallback 包裹，稳定引用） ==========
  const fetchAvailableModels = useCallback(async () => {
    try {
      const res = await fetch('/models');
      const data = await res.json();
      setAvailableModels(data);
    } catch (e) { console.error(e); }
  }, []);

  const fetchAvailableEmbeddingModels = useCallback(async () => {
    try {
      const res = await fetch('/embedding_models');
      if (res.ok) setAvailableEmbeddingModels(await res.json());
    } catch (e) { console.error(e); }
  }, []);

  // ========== 划词工具栏相关函数 ==========
  const handleTextSelection = useCallback(() => {
    const selection = window.getSelection();
    const text = selection.toString().trim();
    if (text) {
      setSelectedText(text);
      setShowTextMenu(true);
      if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        const nextPos = { x: rect.left + rect.width / 2, y: rect.top - 10 };
        setMenuPosition(nextPos);
        setToolbarPosition(nextPos);
      }
    }
  }, [setSelectedText, setShowTextMenu, setMenuPosition]);

  const handleCloseToolbar = useCallback(() => {
    setShowTextMenu(false);
    setSelectedText('');
  }, [setShowTextMenu, setSelectedText]);

  const handleToolbarPositionChange = useCallback((pos) => setToolbarPosition(pos), []);
  const handleToolbarScaleChange = useCallback((scale) => setToolbarScale(scale), [setToolbarScale]);

  // PDFViewer 的文本选择回调（useCallback 稳定引用，避免 PDFViewer 不必要重渲染）
  const handlePdfTextSelect = useCallback((text) => {
    if (text) {
      setSelectedText(text);
      setShowTextMenu(true);
      const selection = window.getSelection();
      if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        const nextPos = { x: rect.left + rect.width / 2, y: rect.top - 10 };
        setMenuPosition(nextPos);
        setToolbarPosition(nextPos);
      }
    }
  }, [setSelectedText, setShowTextMenu, setMenuPosition]);

  // ModelQuickSwitch 的思考模式切换回调（useCallback 稳定引用）
  const handleThinkingChange = useCallback((enabled) => {
    setEnableThinking(enabled);
  }, [setEnableThinking]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(selectedText).then(() => {
      alert('✅ 已复制到剪贴板');
    });
  }, [selectedText]);

  const handleHighlight = useCallback(() => {
    const highlights = JSON.parse(localStorage.getItem(`highlights_${docId}`) || '[]');
    const newHighlight = {
      text: selectedText, page: currentPage,
      timestamp: Date.now(), color: '#fef08a',
    };
    highlights.push(newHighlight);
    localStorage.setItem(`highlights_${docId}`, JSON.stringify(highlights));
    alert('✅ 已添加高亮标注');
  }, [docId, selectedText, currentPage]);

  const handleAddNote = useCallback(() => {
    const note = prompt('请输入您的笔记：', '');
    if (note) {
      const notes = JSON.parse(localStorage.getItem(`notes_${docId}`) || '[]');
      notes.push({
        text: selectedText, note, page: currentPage,
        timestamp: Date.now(),
      });
      localStorage.setItem(`notes_${docId}`, JSON.stringify(notes));
      alert('✅ 笔记已保存');
    }
  }, [docId, selectedText, currentPage]);

  const handleAIExplain = useCallback(() => {
    setInputValue(`请解释这段话：\n\n"${selectedText}"`);
    setShowTextMenu(false);
    setTimeout(() => sendMessage(), 100);
  }, [selectedText, setInputValue, sendMessage, setShowTextMenu]);

  const handleTranslate = useCallback(() => {
    setInputValue(`请将以下内容翻译成中文：\n\n"${selectedText}"`);
    setShowTextMenu(false);
    setTimeout(() => sendMessage(), 100);
  }, [selectedText, setInputValue, sendMessage, setShowTextMenu]);

  const handleWebSearch = useCallback(() => {
    const q = encodeURIComponent(selectedText);
    const searchTemplates = {
      google: `https://www.google.com/search?q=${q}`,
      bing: `https://www.bing.com/search?q=${q}`,
      baidu: `https://www.baidu.com/s?wd=${q}`,
      sogou: `https://www.sogou.com/web?query=${q}`,
      custom: searchEngineUrl.includes('{query}')
        ? searchEngineUrl.replace('{query}', q)
        : `${searchEngineUrl}?q=${q}`,
    };
    window.open(searchTemplates[searchEngine] || searchTemplates.google, '_blank');
  }, [selectedText, searchEngine, searchEngineUrl]);

  const handleShare = useCallback(() => {
    const shareText = `📄 来自《${docInfo?.filename || '文档'}》第 ${currentPage} 页：\n\n"${selectedText}"\n\n--- ChatPDF Pro ---`;
    navigator.clipboard.writeText(shareText).then(() => {
      alert('✅ 引用卡片已复制到剪贴板，可直接粘贴分享');
    });
  }, [docInfo, currentPage, selectedText]);

  // ========== 搜索导航 ==========
  const goToNextResult = useCallback(() => {
    if (!searchResults.length) return;
    focusResult(currentResultIndex + 1);
  }, [searchResults.length, currentResultIndex, focusResult]);

  const goToPrevResult = useCallback(() => {
    if (!searchResults.length) return;
    focusResult(currentResultIndex - 1);
  }, [searchResults.length, currentResultIndex, focusResult]);

  const clearSearchHistory = useCallback(() => {
    if (!docId) return;
    localStorage.removeItem(`search_history_${docId}`);
    // searchHistory 由 usePDFState 管理，需要通过 pdfState 清除
    pdfState.setSearchHistory?.([]);
  }, [docId, pdfState]);

  // ========== 预设问题（useMemo 缓存计算结果） ==========
  const showPresetQuestions = useMemo(() => docId && messages.filter(
    msg => msg.type === 'user' || msg.type === 'assistant'
  ).length === 0, [docId, messages]);

  const handlePresetSelect = useCallback((query) => {
    setInputValue(query);
    requestAnimationFrame(() => sendMessage());
  }, [setInputValue, sendMessage]);

  // ========== 懒加载设置面板关闭回调（useCallback 稳定引用） ==========
  const handleEmbeddingSettingsClose = useCallback(() => setShowEmbeddingSettings(false), [setShowEmbeddingSettings]);
  const handleGlobalSettingsClose = useCallback(() => { setShowGlobalSettings(false); setShowSettings(true); }, [setShowGlobalSettings, setShowSettings]);
  const handleChatSettingsClose = useCallback(() => { setShowChatSettings(false); setShowSettings(true); }, [setShowChatSettings, setShowSettings]);
  const handleOCRSettingsClose = useCallback(() => { setShowOCRSettings(false); setShowSettings(true); }, [setShowOCRSettings, setShowSettings]);

  // ========== 根容器点击回调（useCallback 稳定引用） ==========
  const handleRootClick = useCallback((e) => {
    if (!showTextMenu) return;
    const selection = window.getSelection();
    const hasActiveSelection = selection && selection.toString().trim().length > 0;
    if (hasActiveSelection) return;
    if (!e.target.closest('.text-selection-toolbar-container')) {
      handleCloseToolbar();
    }
  }, [showTextMenu, handleCloseToolbar]);

  // ========== 虚拟消息列表渲染回调（useCallback 稳定引用） ==========
  const renderMessage = useCallback((msg, idx) => {
    const hasThinking = typeof msg.thinking === 'string' && msg.thinking.trim().length > 0;
    const isStreamingCurrentMessage = msg.isStreaming && streamingMessageId === msg.id;
    const shouldShowThinking = hasThinking || (isStreamingCurrentMessage && reasoningEffort !== 'off');
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`flex flex-col ${msg.type === 'user' ? 'items-end' : 'items-start'}`}
        style={{ fontSize: `${messageFontSize}px` }}
      >
        <div className={`${msg.type === 'user'
          ? messageStyle === 'bubble'
            ? 'max-w-[85%] rounded-2xl px-4 py-3 message-bubble-user rounded-tr-sm text-sm'
            : 'max-w-[85%] rounded-2xl px-4 py-3 message-bubble-user rounded-tr-sm text-sm'
          : messageStyle === 'bubble'
            ? 'max-w-[90%] min-w-0 bg-gray-50 dark:bg-gray-800/50 rounded-2xl rounded-tl-sm px-4 py-3 text-gray-800 dark:text-gray-50 overflow-hidden shadow-sm'
            : 'w-full max-w-full min-w-0 bg-transparent shadow-none p-0 text-gray-800 dark:text-gray-50 overflow-hidden'
        }`}
          style={msg.type !== 'user' && messageStyle !== 'bubble' ? { contain: 'inline-size' } : undefined}
        >
          {msg.type === 'assistant' && (
            <div className="flex items-center gap-2 mb-2 select-none">
              <div className="p-1 rounded-lg bg-purple-600 text-white shadow-sm">
                <Bot className="w-4 h-4" />
              </div>
              <span className="font-bold text-sm text-gray-800 dark:text-gray-100">AI Assistant</span>
              {msg.model && <span className="text-xs text-gray-400 border border-gray-200 rounded px-1.5 py-0.5">{msg.model}</span>}
            </div>
          )}
          {shouldShowThinking && (
            <ThinkingBlock
              content={msg.thinking}
              isStreaming={isStreamingCurrentMessage}
              darkMode={darkMode}
              thinkingMs={msg.thinkingMs || 0}
              streamingRef={isStreamingCurrentMessage ? streamingThinkingRef : undefined}
            />
          )}
          {msg.hasImage && (
            <div className="mb-2 rounded-lg overflow-hidden border border-white/20">
              <div className="bg-black/10 p-2 flex items-center gap-2 text-xs">
                <ImageIcon className="w-3 h-3" /> Image attached
              </div>
            </div>
          )}
          {msg.maxRelevanceScore !== null && msg.maxRelevanceScore !== undefined && msg.maxRelevanceScore >= 0 && msg.maxRelevanceScore < 0.3 && !msg.isStreaming && (
            <div className="mb-2 px-3 py-2 rounded-lg bg-amber-50 border border-amber-200 text-amber-700 text-xs flex items-center gap-1.5">
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" /></svg>
              <span>检索到的内容与您的问题相关性较低，回答可能不够准确，请谨慎参考。</span>
            </div>
          )}
          <StreamingMarkdown
            content={msg.content}
            isStreaming={(msg.isStreaming || false) && !(shouldShowThinking && isStreamingCurrentMessage)}
            enableBlurReveal={enableBlurReveal}
            blurIntensity={blurIntensity}
            citations={msg.citations || null}
            onCitationClick={(c) => { setActiveCitationRef(c?.ref ?? null); handleCitationClick(c); }}
            streamingRef={msg.isStreaming && streamingMessageId === msg.id ? streamingContentRef : undefined}
          />
          {/* 联网搜索来源 */}
          {msg.webSearchSources && msg.webSearchSources.length > 0 && !msg.isStreaming && (
            <WebSearchSourcesBadge sources={msg.webSearchSources} />
          )}
        </div>
        {/* 证据面板 */}
        {msg.type === 'assistant' && !msg.isStreaming && msg.citations && msg.citations.length > 0 && (
          <EvidencePanel
            citations={msg.citations}
            onCitationClick={(c) => { setActiveCitationRef(c?.ref ?? null); handleCitationClick(c); }}
            activeRef={activeCitationRef}
            onRefHover={setActiveCitationRef}
          />
        )}
        {/* 思维导图 */}
        {msg.type === 'assistant' && !msg.isStreaming && msg.mindmapMarkdown && (
          <MindmapView markdown={msg.mindmapMarkdown} />
        )}
        {/* 消息操作按钮 */}
        {msg.type === 'assistant' && !msg.isStreaming && (
          <div className="flex items-center gap-1 mt-1 ml-2">
            <button onClick={() => copyMessage(msg.content, msg.id || idx)} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors" title="复制">
              {copiedMessageId === (msg.id || idx) ? (
                <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
              ) : (<Copy className="w-4 h-4" />)}
            </button>
            <button onClick={() => { if (!confirmRegenerateMessage || confirm('确定要重新生成这条回答吗？')) regenerateMessage(idx); }} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors" title="重新生成">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
            </button>
            <button onClick={() => saveToMemory(idx, 'liked')} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${likedMessages.has(idx) ? 'text-pink-500' : 'text-gray-500 hover:text-gray-700'}`} title="点赞并记忆">
              <svg className="w-4 h-4" fill={likedMessages.has(idx) ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" /></svg>
            </button>
            <button onClick={() => setFeedbackTarget({ idx, msg })} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${dislikedMessages.has(idx) ? 'text-orange-500' : 'text-gray-500 hover:text-gray-700'}`} title="点踩并反馈">
              <svg className="w-4 h-4" fill={dislikedMessages.has(idx) ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" /></svg>
            </button>
            <button onClick={() => saveToMemory(idx, 'manual')} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${rememberedMessages.has(idx) ? 'text-purple-500' : 'text-gray-500 hover:text-gray-700'}`} title="记住这个">
              <Brain className={`w-4 h-4 ${rememberedMessages.has(idx) ? 'fill-current' : ''}`} />
            </button>
            {msg.qaScore != null && (
              <span className={`ml-1 text-[10px] px-1.5 py-0.5 rounded-full font-medium ${msg.qaScore >= 0.7 ? 'bg-green-50 text-green-600' : msg.qaScore >= 0.4 ? 'bg-yellow-50 text-yellow-600' : 'bg-red-50 text-red-600'}`} title={`回答置信度: ${(msg.qaScore * 100).toFixed(0)}%`}>
                {(msg.qaScore * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}
        {/* 动态追问建议 */}
        {msg.type === 'assistant' && !msg.isStreaming && msg.followupQuestions && msg.followupQuestions.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2 ml-2">
            {msg.followupQuestions.map((q, qi) => (
              <button
                key={qi}
                onClick={() => {
                  const textarea = document.querySelector('textarea');
                  if (textarea) {
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                    nativeInputValueSetter.call(textarea, q);
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    textarea.focus();
                  }
                }}
                className="text-xs px-2.5 py-1.5 rounded-full border border-blue-200 bg-blue-50 text-blue-600 hover:bg-blue-100 hover:border-blue-300 transition-colors cursor-pointer"
              >
                {q}
              </button>
            ))}
          </div>
        )}
      </motion.div>
    );
  }, [
    streamingMessageId, darkMode, enableBlurReveal, blurIntensity,
    streamingThinkingRef, streamingContentRef, copiedMessageId,
    likedMessages, rememberedMessages,
    handleCitationClick, copyMessage, regenerateMessage, saveToMemory,
    messageStyle, messageFontSize, confirmRegenerateMessage, reasoningEffort,
    activeCitationRef, setActiveCitationRef,
    dislikedMessages, setFeedbackTarget,
  ]);

  // ========== 反馈提交 ==========
  const handleFeedbackSubmit = useCallback(async (issueTypes, detail) => {
    if (!feedbackTarget) return;
    const { idx, msg } = feedbackTarget;
    const prevMsg = messages[idx - 1];
    try {
      await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: docId || '',
          message_idx: idx,
          feedback_type: 'dislike',
          issue_types: issueTypes,
          detail,
          question: prevMsg?.type === 'user' ? prevMsg.content : '',
          answer: (msg.content || '').slice(0, 500),
          model: msg.model || '',
        }),
      });
      setDislikedMessages(prev => new Set(prev).add(idx));
    } catch (e) {
      console.error('反馈提交失败', e);
    }
    setFeedbackTarget(null);
  }, [feedbackTarget, messages, docId]);

  // ========== 渲染 ==========
  return (
    <div
      className={`h-screen w-full flex overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-[#0f1115] text-gray-200' : 'bg-transparent text-[var(--color-text-main)]'}`}
      onClick={handleRootClick}
    >
      {/* 划词工具栏 */}
      {showTextMenu && selectedText && (
        <div className="text-selection-toolbar-container">
          <TextSelectionToolbar
            selectedText={selectedText}
            position={toolbarPosition.x === 0 && toolbarPosition.y === 0 ? menuPosition : toolbarPosition}
            onPositionChange={handleToolbarPositionChange}
            scale={toolbarScale}
            onScaleChange={handleToolbarScaleChange}
            onClose={handleCloseToolbar}
            onCopy={handleCopy}
            onHighlight={handleHighlight}
            onAddNote={handleAddNote}
            onAIExplain={handleAIExplain}
            onTranslate={handleTranslate}
            onWebSearch={handleWebSearch}
            onShare={handleShare}
            size={toolbarSize}
          />
        </div>
      )}

      {/* 侧边栏（历史记录） */}
      <motion.div
        initial={false}
        animate={{ width: showSidebar ? 288 : 0, opacity: showSidebar ? 1 : 0 }}
        transition={{ duration: 0.2, ease: "easeInOut" }}
        style={{ pointerEvents: showSidebar ? 'auto' : 'none' }}
        className={`flex-shrink-0 m-6 mr-0 h-[calc(100vh-3rem)] flex flex-col z-20 overflow-hidden rounded-[var(--radius-panel-lg)] ${darkMode ? 'bg-[#1a1d21]/90 border-white/5 backdrop-blur-3xl backdrop-saturate-150' : 'bg-white/80 border-white/50 backdrop-blur-3xl backdrop-saturate-150 border shadow-xl'}`}
      >
        <div className="w-72 mx-auto flex flex-col h-full items-stretch relative">
          <button
            onClick={() => setShowSidebar(false)}
            className={`absolute top-3 right-3 p-2 rounded-full transition-colors z-10 ${darkMode ? 'hover:bg-white/10 text-gray-500 hover:text-gray-300' : 'hover:bg-black/5 text-gray-400 hover:text-gray-700'}`}
            title="收起侧边栏"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <div className="px-6 py-8 flex items-center justify-between">
            <div className="flex items-center gap-3 font-bold text-2xl text-purple-600 tracking-tight">
              <Bot className="w-9 h-9" />
              <span>ChatPDF</span>
            </div>
            <div className="flex items-center gap-1">
              {!isHeaderExpanded && (
                <button
                  onClick={() => setIsHeaderExpanded(true)}
                  className={`p-2 rounded-full transition-colors ${darkMode ? 'hover:bg-white/10 text-gray-400 hover:text-gray-200' : 'hover:bg-black/5 text-gray-500 hover:text-gray-800'}`}
                  title="展开顶栏"
                >
                  <ChevronDown className="w-4 h-4" />
                </button>
              )}
              <button onClick={() => setDarkMode(!darkMode)} className={`p-2 rounded-full transition-colors ${darkMode ? 'hover:bg-white/10 text-gray-400 hover:text-yellow-400' : 'hover:bg-black/5'}`}>
                {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="px-5 mb-4 flex justify-center">
            <button
              onClick={() => { startNewChat(); fileInputRef.current?.click(); }}
              className="tanya-btn max-w-[260px]"
            >
              <Plus className="w-5 h-5 opacity-70" />
              <span>上传文件/新对话</span>
            </button>
            <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
          </div>

          <div className="flex-1 overflow-y-auto px-5 space-y-2 flex flex-col items-center">
            <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2 px-2 w-full max-w-[260px]">History</div>
            {history.map((item, idx) => (
              <div
                key={idx}
                onClick={() => loadSession(item)}
                className={`w-full max-w-[260px] p-3 rounded-xl cursor-pointer group flex items-center gap-3 transition-all duration-200 ${
                  item.id === docId
                    ? (darkMode ? 'bg-white/10 shadow-md scale-[1.02] text-white ring-1 ring-white/10' : 'bg-white shadow-md scale-[1.02]')
                    : (darkMode ? 'text-gray-400 hover:bg-white/5 hover:text-gray-200' : 'hover:bg-white/40')
                }`}
              >
                <MessageSquare className="w-5 h-5 text-purple-500" />
                <div className="flex-1 truncate text-sm font-medium">{item.filename}</div>
                <button
                  onClick={(e) => { e.stopPropagation(); if (!confirmDeleteMessage || confirm('确定要删除这条对话记录吗？')) deleteSession(item.id); }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-opacity"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>

          <div className="p-4 border-t border-white/20">
            <button onClick={() => { setShowSettings(true); fetchStorageInfo(); }} className="flex items-center gap-3 w-full p-3 rounded-xl hover:bg-white/50 transition-colors text-sm font-medium">
              <Settings className="w-5 h-5" />
              <span>设置 & API Key</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* 主内容区域 */}
      <div className="flex-1 flex flex-col h-full relative transition-all duration-200 ease-in-out">
        {/* 顶栏 - 可折叠 */}
        <motion.header
          layout
          initial={false}
          animate={{
            height: isHeaderExpanded ? (headerHeight ?? 'auto') : 0,
            opacity: isHeaderExpanded ? 1 : 0,
            marginBottom: isHeaderExpanded ? 16 : 0,
            marginTop: isHeaderExpanded ? 24 : 0,
            pointerEvents: isHeaderExpanded ? 'auto' : 'none'
          }}
          transition={{ duration: 0.28, ease: [0.4, 0, 0.2, 1] }}
          style={{ overflow: 'hidden' }}
          className="px-8 soft-panel mx-8 sticky top-4 z-10 flex flex-col justify-center rounded-[var(--radius-panel-lg)]"
        >
          <motion.div
            ref={headerContentRef}
            initial={false}
            animate={{ opacity: isHeaderExpanded ? 1 : 0, y: isHeaderExpanded ? 0 : -6 }}
            transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
          >
            <div className="flex items-center justify-between w-full py-3">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setShowSidebar(!showSidebar)}
                  className="p-2 hover:bg-black/5 rounded-lg transition-colors"
                  title={showSidebar ? "隐藏侧边栏" : "显示侧边栏"}
                >
                  <Menu className="w-6 h-6" />
                </button>
                <div className="flex items-center gap-4">
                  <div className="bg-purple-600 text-white p-2.5 rounded-xl shadow-sm">
                    <FileText className="w-6 h-6" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-[var(--color-text-main)]">
                      ChatPDF Pro <span className="text-xs bg-purple-100 text-purple-600 px-2 py-0.5 rounded-full ml-2 align-middle">v2.0.2</span>
                    </h1>
                    <p className="text-xs text-gray-500 font-medium mt-0.5">智能文档助手</p>
                  </div>
                </div>
              </div>

              {/* 搜索框 */}
              {docId && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                  className="flex-1 max-w-2xl mx-4 flex items-center gap-2"
                >
                  <div className="relative flex-1">
                    <input
                      type="search"
                      placeholder="搜索文档内容..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter' && !isSearching) handleSearch(); }}
                      className="w-full px-4 py-2 pl-11 pr-4 rounded-full soft-input text-sm transition-all focus:ring-2 focus:ring-purple-400"
                      disabled={isSearching}
                    />
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                  </div>
                  <motion.button
                    whileHover={{ scale: isSearching ? 1 : 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleSearch()}
                    disabled={isSearching}
                    className={`px-3 py-2 rounded-full text-sm font-medium shadow-sm flex items-center gap-2 transition-all ${isSearching ? 'bg-purple-200 text-purple-700 cursor-wait' : 'bg-purple-600 text-white hover:shadow-md hover:bg-purple-700'}`}
                  >
                    {isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                    <span>{isSearching ? '搜索中...' : '搜索'}</span>
                  </motion.button>
                  <button
                    onClick={() => setUseRerankSetting(v => !v)}
                    className={`px-3 py-2 rounded-full border text-sm font-medium flex items-center gap-1 transition-colors ${useRerankSetting ? 'bg-purple-50 text-purple-700 border-purple-200' : 'bg-white text-gray-600 border-gray-200'}`}
                    title="使用重排模型提高结果质量"
                  >
                    <Wand2 className="w-4 h-4" />
                    <span>重排</span>
                  </button>
                  <AnimatePresence>
                    {searchResults.length > 0 && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        transition={{ duration: 0.2 }}
                        className="flex items-center gap-1"
                      >
                        <span className="text-xs text-gray-500 px-2 font-medium">
                          {currentResultIndex + 1}/{searchResults.length}
                        </span>
                        <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }} onClick={goToPrevResult} className="p-1.5 hover:bg-black/5 rounded-lg transition-colors" title="上一个结果">
                          <ChevronUp className="w-4 h-4" />
                        </motion.button>
                        <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }} onClick={goToNextResult} className="p-1.5 hover:bg-black/5 rounded-lg transition-colors" title="下一个结果">
                          <ChevronDown className="w-4 h-4" />
                        </motion.button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}

              <div className="flex items-center gap-4">
                {docInfo && (
                  <div className="font-medium text-sm glass-panel px-4 py-1 rounded-full truncate max-w-[200px]">
                    {docInfo.filename}
                  </div>
                )}
                <button
                  onClick={() => setIsHeaderExpanded(false)}
                  className="p-2 hover:bg-black/5 rounded-full transition-colors text-gray-500 hover:text-gray-800"
                  title="收起顶栏"
                >
                  <ChevronUp className="w-5 h-5" />
                </button>
              </div>
            </div>
          </motion.div>
        </motion.header>

        {/* 浮动控制按钮：顶栏收起 + 侧边栏隐藏时显示 */}
        <AnimatePresence>
          {!isHeaderExpanded && !showSidebar && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="absolute top-4 left-2 z-20 flex flex-col gap-1.5"
            >
              <button
                onClick={() => setIsHeaderExpanded(true)}
                className={`p-2 backdrop-blur-md shadow-sm rounded-full hover:scale-105 transition-all border ${darkMode ? 'bg-white/10 text-gray-300 border-white/10 hover:bg-white/20' : 'bg-white/80 text-gray-700 border-white/50 hover:bg-white'}`}
                title="展开顶栏"
              >
                <ChevronDown className="w-4 h-4" />
              </button>
              <button
                onClick={() => setShowSidebar(v => !v)}
                className={`p-2 backdrop-blur-md shadow-sm rounded-full hover:scale-105 transition-all border ${darkMode ? 'bg-white/10 text-gray-300 border-white/10 hover:bg-white/20' : 'bg-white/80 text-gray-700 border-white/50 hover:bg-white'}`}
                title={showSidebar ? '收起侧边栏' : '显示侧边栏'}
              >
                <Menu className="w-4 h-4" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 内容区域 */}
        <div className="flex-1 flex overflow-hidden px-8 pb-8 gap-4 pt-2">
          {/* 左侧：PDF 预览 */}
          {docId ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`soft-panel overflow-hidden flex flex-col relative flex-shrink-0 rounded-[var(--radius-panel)] min-w-0 ${darkMode ? 'bg-gray-800/50' : ''}`}
              style={{ width: `${pdfPanelWidth}%`, minWidth: '350px' }}
            >
              <div className="flex-1 overflow-hidden">
                {docInfo?.pdf_url ? (
                  <PDFViewer
                    ref={pdfContainerRef}
                    pdfUrl={docInfo.pdf_url}
                    page={currentPage}
                    onPageChange={setCurrentPage}
                    highlightInfo={activeHighlight}
                    isSelecting={isSelectingArea}
                    onAreaSelected={handleAreaSelected}
                    onSelectionCancel={handleSelectionCancel}
                    darkMode={darkMode}
                    onTextSelect={handlePdfTextSelect}
                  />
                ) : (docInfo?.pages || docInfo?.data?.pages) ? (
                  <>
                    <div className="h-14 border-b border-black/5 flex items-center justify-between px-6 bg-white/30 backdrop-blur-sm">
                      <div className="flex items-center gap-2">
                        <button onClick={() => setCurrentPage(Math.max(1, currentPage - 1))} className="p-1.5 hover:bg-black/5 rounded-lg"><ChevronLeft className="w-5 h-5" /></button>
                        <span className="text-sm font-medium w-16 text-center">{currentPage} / {docInfo?.total_pages || docInfo?.data?.total_pages || 1}</span>
                        <button onClick={() => setCurrentPage(Math.min(docInfo?.total_pages || docInfo?.data?.total_pages || 1, currentPage + 1))} className="p-1.5 hover:bg-black/5 rounded-lg"><ChevronRight className="w-5 h-5" /></button>
                      </div>
                      <div className="flex items-center gap-2">
                        <button onClick={() => setPdfScale(s => Math.max(0.5, s - 0.1))} className="p-1.5 hover:bg-black/5 rounded-lg"><ZoomOut className="w-5 h-5" /></button>
                        <span className="text-sm font-medium w-12 text-center">{Math.round(pdfScale * 100)}%</span>
                        <button onClick={() => setPdfScale(s => Math.min(2.0, s + 0.1))} className="p-1.5 hover:bg-black/5 rounded-lg"><ZoomIn className="w-5 h-5" /></button>
                      </div>
                    </div>
                    <div ref={pdfContainerRef} className="h-full overflow-auto bg-gray-50/50">
                      <div className="min-h-full flex items-start justify-center p-8" style={{ zoom: pdfScale }}>
                        <div className="bg-white shadow-2xl p-12 rounded-lg max-w-4xl w-full" onMouseUp={handleTextSelection}>
                          <pre className="whitespace-pre-wrap font-serif text-gray-800 leading-relaxed">
                            {(docInfo.pages || docInfo.data?.pages)?.[currentPage - 1]?.content || 'No content'}
                          </pre>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    <p>Loading PDF...</p>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            /* 空状态 */
            <div className="flex-1 flex items-center justify-center relative overflow-hidden">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full max-w-lg max-h-lg pointer-events-none">
                <div className="blob bg-purple-200 w-72 h-72 top-0 left-0 mix-blend-multiply animate-blob"></div>
                <div className="blob bg-cyan-100 w-72 h-72 bottom-0 right-0 mix-blend-multiply animate-blob animation-delay-2000"></div>
              </div>
              <div className="text-center space-y-8 max-w-md relative z-10">
                <div className="w-24 h-24 bg-white/50 backdrop-blur-md rounded-[32px] flex items-center justify-center mx-auto shadow-sm border border-white/60">
                  <Upload className="w-10 h-10 text-purple-500/80" />
                </div>
                <div className="space-y-2">
                  <h2 className="text-3xl font-bold text-gray-800 tracking-tight">Upload a PDF to Start</h2>
                  <p className="text-gray-500 text-lg">Chat with your documents using AI.</p>
                </div>
              </div>
            </div>
          )}

          {/* 可拖拽分隔线 */}
          <div
            className="w-4 cursor-col-resize flex-shrink-0 relative group -ml-2 z-10 flex justify-center"
            onMouseDown={(e) => {
              e.preventDefault();
              const startX = e.clientX;
              const startWidth = pdfPanelWidth;
              const handleMouseMove = (e) => {
                const containerWidth = e.currentTarget?.parentElement?.offsetWidth || window.innerWidth;
                const deltaX = e.clientX - startX;
                const deltaPercent = (deltaX / containerWidth) * 100;
                const newWidth = Math.max(30, Math.min(70, startWidth + deltaPercent));
                setPdfPanelWidth(newWidth);
              };
              const handleMouseUp = () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
              };
              document.addEventListener('mousemove', handleMouseMove);
              document.addEventListener('mouseup', handleMouseUp);
            }}
          >
            <div className="w-1 h-full rounded-full bg-transparent group-hover:bg-purple-500/50 transition-colors duration-200" />
          </div>

          {/* 右侧：聊天区域 */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`soft-panel flex flex-col overflow-hidden rounded-[var(--radius-panel)] min-w-0 ${darkMode ? 'bg-gray-800/50' : ''}`}
            style={{ width: `calc(${100 - pdfPanelWidth}% - 2rem)`, minWidth: '350px' }}
          >
            {/* 消息列表 */}
            <div className="flex-1 overflow-hidden flex flex-col min-w-0">
              {/* 搜索结果面板 - 固定在消息列表上方 */}
              {(searchResults.length > 0 || isSearching || searchHistory.length > 0) && (
                <div className="p-6 pb-0">
                  <div className="rounded-3xl border border-black/5 bg-white/70 backdrop-blur-sm p-4 space-y-3 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Search className="w-4 h-4 text-purple-500" />
                        <span className="font-semibold text-sm text-gray-800">文档搜索</span>
                        {useRerankSetting && (
                          <span className="text-xs text-purple-700 bg-purple-50 px-2 py-0.5 rounded-full border border-purple-100">已开启重排</span>
                        )}
                        {isSearching && <Loader2 className="w-4 h-4 animate-spin text-purple-500" />}
                      </div>
                      {searchResults.length > 0 && (
                        <span className="text-xs text-gray-500">找到 {searchResults.length} 个候选</span>
                      )}
                    </div>

                    {searchHistory.length > 0 && (
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs text-gray-500">历史:</span>
                        {searchHistory.map((item, idx) => (
                          <button key={`history-${idx}`} onClick={() => handleSearch(item)} className="text-xs px-2 py-1 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors">
                            {item}
                          </button>
                        ))}
                        <button onClick={clearSearchHistory} className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1 rounded-full hover:bg-black/5 transition-colors">
                          清除
                        </button>
                      </div>
                    )}

                    <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                      {isSearching && (
                        <div className="text-sm text-gray-500 flex items-center gap-2 px-2">
                          <Loader2 className="w-4 h-4 animate-spin" /> 正在检索匹配片段...
                        </div>
                      )}
                      {!isSearching && !searchResults.length && (
                        <p className="text-sm text-gray-500 px-2">输入查询并点击"搜索"查看匹配片段，支持关键词上下文和匹配度展示。</p>
                      )}
                      {searchResults.map((result, idx) => (
                        <button
                          key={`result-${idx}`}
                          onClick={() => focusResult(idx)}
                          className="w-full text-left p-3 rounded-2xl border border-gray-100 hover:border-purple-200 hover:bg-purple-50/40 transition-all relative"
                        >
                          <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                            <div className="flex items-center gap-1.5">
                              <span>第 {result.page || 1} 页 · #{idx + 1}</span>
                              {result.reranked && (
                                <span className="text-[10px] text-purple-700 bg-purple-50 px-1.5 py-0.5 rounded-full border border-purple-100">Rerank</span>
                              )}
                            </div>
                            <span className={`font-semibold ${formatSimilarity(result) >= 80 ? 'text-green-600' : 'text-purple-600'}`}>
                              匹配度 {formatSimilarity(result)}%
                            </span>
                          </div>
                          <div className="text-sm text-gray-800 leading-relaxed max-h-20 overflow-hidden">
                            {renderHighlightedSnippet(result.snippet || result.chunk || '', result.highlights || [])}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* 预设问题 */}
              {showPresetQuestions && (
                <div className="p-6 pb-0">
                  <PresetQuestions onSelect={handlePresetSelect} disabled={isLoading} />
                </div>
              )}

              {/* 虚拟消息列表 - 替代原有的 messages.map 渲染（需求 3.1） */}
              <VirtualMessageList
                messages={messages}
                renderMessage={renderMessage}
                streamingMessageId={streamingMessageId}
                className="flex-1 overflow-y-auto overflow-x-hidden p-6 space-y-6 min-w-0"
              />
            </div>

            {/* 输入区域 */}
            <div className="p-6 pt-0 bg-transparent">
              {/* 截图预览 */}
              <ScreenshotPreview
                screenshots={screenshots}
                onAction={handleScreenshotAction}
                onClose={handleScreenshotClose}
              />

              <div className="relative bg-white/80 backdrop-blur-[20px] rounded-[36px] shadow-[0_24px_56px_-12px_rgba(0,0,0,0.22),0_8px_24px_-6px_rgba(0,0,0,0.12),inset_0_1px_0_rgba(255,255,255,0.9)] p-1.5 flex items-end gap-2 border border-white/50 ring-1 ring-black/5">
                <div className="flex-1 flex flex-col min-h-[48px] justify-center pl-6 py-1.5">
                  <div className="flex items-center gap-2 mb-1">
                    <textarea
                      ref={textareaRef}
                      onChange={(e) => {
                        e.target.style.height = '24px';
                        e.target.style.height = e.target.scrollHeight + 'px';
                        const newHasInput = !!e.target.value.trim();
                        if (newHasInput !== hasInput) setHasInput(newHasInput);
                      }}
                      onKeyDown={(e) => {
                        if (sendShortcut === 'Ctrl+Enter') {
                          if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendMessage(); }
                        } else {
                          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
                        }
                      }}
                      placeholder="Summarize, rephrase, convert..."
                      className="w-full bg-transparent border-none outline-none text-gray-800 placeholder:text-gray-400 font-medium resize-none h-[24px] overflow-hidden leading-relaxed py-0 focus:ring-0 text-[15px]"
                      rows={1}
                      style={{ minHeight: '24px', maxHeight: '120px' }}
                    />
                  </div>
                  <div className="flex items-center gap-4 text-gray-400 mt-2">
                    <ModelQuickSwitch onThinkingChange={handleThinkingChange} />
                    <button className="hover:text-gray-600 transition-colors p-1 rounded-md hover:bg-gray-50">
                      <SlidersHorizontal className="w-5 h-5" />
                    </button>
                    <button onClick={() => fileInputRef.current?.click()} className="hover:text-gray-600 transition-colors p-1 rounded-md hover:bg-gray-50">
                      <Paperclip className="w-5 h-5" />
                    </button>
                    <WebSearchButton />
                    {isVisionCapable && (
                      <button
                        onClick={() => setIsSelectingArea(true)}
                        disabled={!docId}
                        className={`transition-colors p-1 rounded-md ${docId ? isSelectingArea ? 'text-purple-600 bg-purple-50 hover:bg-purple-100' : 'hover:text-gray-600 hover:bg-gray-50' : 'text-gray-300 cursor-not-allowed'}`}
                        title={!docId ? '请先上传文档' : isSelectingArea ? '框选模式已开启' : '区域截图'}
                      >
                        <Scan className="w-5 h-5" />
                      </button>
                    )}
                  </div>
                </div>
                <motion.button
                  onClick={isLoading ? handleStop : sendMessage}
                  disabled={!isLoading && (!hasInput && screenshots.length === 0)}
                  className="glass-btn-3d relative z-10 flex-shrink-0"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <AnimatePresence initial={false}>
                    {isLoading ? (
                      <motion.div key="pause" initial={{ rotate: -90, scale: 0.5, opacity: 0 }} animate={{ rotate: 0, scale: 1, opacity: 1 }} exit={{ rotate: 90, scale: 0.5, opacity: 0 }} transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }} className="absolute inset-0 flex items-center justify-center">
                        <PauseIcon />
                      </motion.div>
                    ) : (
                      <motion.div key="send" initial={{ rotate: -90, scale: 0.5, opacity: 0 }} animate={{ rotate: 0, scale: 1, opacity: 1 }} exit={{ rotate: 90, scale: 0.5, opacity: 0 }} transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }} className="absolute inset-0 flex items-center justify-center">
                        <SendIcon />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.button>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* 上传进度模态框 */}
      <AnimatePresence>
        {isUploading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 350, damping: 25 }}
              className="flex flex-col items-center"
            >
              <div style={{ position: 'relative', width: 300, height: 300 }}>
                <div style={{ position: 'absolute', inset: 0, filter: 'blur(0.5px) contrast(1.2)' }}>
                  {UPLOAD_RING_CONFIGS.map((cfg, i) => (
                    <div key={i} style={{
                      position: 'absolute', top: '50%', left: '50%',
                      width: cfg.s, height: cfg.s, borderRadius: cfg.br,
                      border: `${cfg.w}px solid ${cfg.c}`, background: 'transparent',
                      mixBlendMode: cfg.mix, pointerEvents: 'none',
                      animation: `chatpdf-spin ${cfg.dur}s linear ${cfg.del}s infinite ${cfg.dir}`,
                    }} />
                  ))}
                </div>
                <div style={{
                  position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center', zIndex: 10, pointerEvents: 'none',
                }}>
                  <span style={{ color: 'rgba(255, 255, 255, 0.9)', fontSize: '2.5rem', fontWeight: 200, letterSpacing: '2px', textShadow: '0 0 15px rgba(255, 255, 255, 0.3)', fontVariantNumeric: 'tabular-nums' }}>
                    {uploadProgress}%
                  </span>
                  <span style={{ color: 'rgba(255, 255, 255, 0.55)', fontSize: '0.7rem', letterSpacing: '4px', textTransform: 'uppercase', marginTop: '6px' }}>
                    {uploadStatus === 'uploading' ? 'Uploading' : 'Processing'}
                  </span>
                </div>
              </div>
              <motion.p
                key={uploadStatus}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.9rem', fontWeight: 300, letterSpacing: '0.5px', marginTop: '8px' }}
              >
                {uploadStatus === 'uploading' ? '正在上传文档...' : 'AI 正在构建知识库索引'}
              </motion.p>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* 设置模态框 */}
      <AnimatePresence initial={false}>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm p-4"
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 10 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30, mass: 0.8 }}
              onClick={(e) => e.stopPropagation()}
              className="soft-panel w-[500px] max-w-full max-h-[90vh] overflow-hidden flex flex-col shadow-2xl"
            >
              <div className="flex justify-between items-center p-8 pb-4 flex-shrink-0">
                <h2 className="text-2xl font-bold text-gray-800">Settings</h2>
                <button onClick={() => setShowSettings(false)} className="p-2 hover:bg-gray-100 rounded-full transition-colors"><X className="w-5 h-5" /></button>
              </div>

              <div className="space-y-4 px-8 overflow-y-auto flex-1">
                {/* 模型服务管理入口 */}
                <div className="relative overflow-hidden rounded-[32px] border border-purple-100/50 bg-gradient-to-br from-white/40 to-purple-50/10 p-1 shadow-sm transition-all hover:shadow-md backdrop-blur-md">
                  <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-purple-500/10 rounded-full blur-3xl"></div>
                  <div className="absolute bottom-0 left-0 -mb-4 -ml-4 w-20 h-20 bg-purple-500/10 rounded-full blur-2xl"></div>
                  <div className="relative bg-white/30 backdrop-blur-sm rounded-[28px] p-5 border border-white/50">
                    <div className="flex flex-col gap-5">
                      <div className="flex items-center justify-between gap-4">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 rounded-[20px] bg-gradient-to-br from-purple-500/90 to-indigo-600/90 shadow-lg shadow-purple-500/20 flex items-center justify-center text-white shrink-0 backdrop-blur-sm">
                            <Server className="w-6 h-6" />
                          </div>
                          <div className="space-y-0.5">
                            <h3 className="text-lg font-bold text-gray-900 tracking-tight">模型服务</h3>
                            <p className="text-xs text-gray-500 font-medium">统一管理 Chat / Embedding / Rerank</p>
                          </div>
                        </div>
                        <button onClick={() => setShowEmbeddingSettings(true)} className="group relative overflow-hidden rounded-[18px] bg-gray-900/90 px-5 py-2.5 text-white shadow-lg transition-all hover:bg-gray-800 hover:shadow-xl hover:-translate-y-0.5 active:scale-95 shrink-0 backdrop-blur-sm">
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out" />
                          <div className="relative flex items-center gap-2 font-medium text-sm">
                            <span>管理模型</span>
                            <Settings className="w-4 h-4 transition-transform duration-500 group-hover:rotate-180" />
                          </div>
                        </button>
                      </div>
                      <div className="flex flex-col gap-3">
                        <div className="group relative overflow-hidden rounded-[18px] border border-gray-100/50 bg-white/40 p-4 transition-all hover:border-purple-300 hover:bg-white/90 shadow-md hover:shadow-xl hover:-translate-y-1 active:scale-[0.98] backdrop-blur-sm cursor-pointer">
                          <div className="flex items-center gap-4">
                            <MessageCircle className="w-5 h-5 text-gray-400 group-hover:text-purple-500 transition-colors" />
                            <div className="flex-1 min-w-0">
                              <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-0.5">Chat Model</div>
                              <div className="font-semibold text-gray-800 text-sm truncate" title={getDefaultModelLabel(getDefaultModel('assistantModel'))}>
                                {getDefaultModelLabel(getDefaultModel('assistantModel')) || '未设置'}
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="group relative overflow-hidden rounded-[18px] border border-gray-100/50 bg-white/40 p-4 transition-all hover:border-purple-300 hover:bg-white/90 shadow-md hover:shadow-xl hover:-translate-y-1 active:scale-[0.98] backdrop-blur-sm cursor-pointer">
                          <div className="flex items-center gap-4">
                            <Database className="w-5 h-5 text-gray-400 group-hover:text-purple-500 transition-colors" />
                            <div className="flex-1 min-w-0">
                              <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-0.5">Embedding</div>
                              <div className="font-semibold text-gray-800 text-sm truncate" title={getDefaultModelLabel(getDefaultModel('embeddingModel'))}>
                                {getDefaultModelLabel(getDefaultModel('embeddingModel')) || '未设置'}
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="group relative overflow-hidden rounded-[18px] border border-gray-100/50 bg-white/40 p-4 transition-all hover:border-amber-300 hover:bg-white/90 shadow-md hover:shadow-xl hover:-translate-y-1 active:scale-[0.98] backdrop-blur-sm cursor-pointer">
                          <div className="flex items-center gap-4">
                            <ArrowUpDown className="w-5 h-5 text-gray-400 group-hover:text-amber-500 transition-colors" />
                            <div className="flex-1 min-w-0">
                              <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-0.5">Rerank</div>
                              <div className="font-semibold text-gray-800 text-sm truncate" title={getDefaultModelLabel(getDefaultModel('rerankModel'))}>
                                {getDefaultModelLabel(getDefaultModel('rerankModel')) || '未设置'}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t border-gray-100">
                  <label className="flex items-center justify-between cursor-pointer p-2 hover:bg-gray-50 rounded-lg">
                    <span className="font-medium">Vector Search</span>
                    <input type="checkbox" checked={enableVectorSearch} onChange={e => setEnableVectorSearch(e.target.checked)} className="accent-purple-600 w-5 h-5" />
                  </label>
                  <label className="flex items-center justify-between cursor-pointer p-2 hover:bg-gray-50 rounded-lg">
                    <span className="font-medium">Screenshot Analysis</span>
                    <input type="checkbox" checked={enableScreenshot} onChange={e => setEnableScreenshot(e.target.checked)} className="accent-purple-600 w-5 h-5" />
                  </label>

                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mt-4 mb-1 px-2">检索增强</h4>
                  <label className="flex items-center justify-between cursor-pointer p-2 hover:bg-gray-50 rounded-lg">
                    <div>
                      <span className="font-medium">GraphRAG 知识图谱</span>
                      <p className="text-xs text-gray-500">实体关系提取 + 社区聚类增强检索</p>
                    </div>
                    <input type="checkbox" checked={enableGraphRAG} onChange={e => setEnableGraphRAG(e.target.checked)} className="accent-purple-600 w-5 h-5" />
                  </label>
                  <label className="flex items-center justify-between cursor-pointer p-2 hover:bg-gray-50 rounded-lg">
                    <div>
                      <span className="font-medium">jieba 中文分词</span>
                      <p className="text-xs text-gray-500">提升 BM25 中文关键词匹配精度</p>
                    </div>
                    <input type="checkbox" checked={enableJiebaBM25} onChange={e => setEnableJiebaBM25(e.target.checked)} className="accent-purple-600 w-5 h-5" />
                  </label>
                  <div className="p-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">邻居上下文扩展</label>
                    <CustomSelect
                      value={numExpandContextChunk}
                      onChange={setNumExpandContextChunk}
                      options={[
                        { value: 0, label: '关闭' },
                        { value: 1, label: '±1 块（前后各 1 个）' },
                        { value: 2, label: '±2 块（前后各 2 个）' },
                        { value: 3, label: '±3 块（前后各 3 个）' },
                      ]}
                    />
                    <p className="text-xs text-gray-500 mt-1">命中 chunk 前后各扩展 N 个邻居块作为上下文</p>
                  </div>

                  {lastCallInfo && (
                    <div className="mt-3 p-3 rounded-[18px] border text-xs text-gray-700 bg-gray-50">
                      <div>调用来源: <strong>{lastCallInfo.provider || '未知'}</strong></div>
                      <div>模型: <strong>{lastCallInfo.model || '未返回'}</strong></div>
                      {lastCallInfo.fallback && <div className="text-amber-700">已切换备用</div>}
                    </div>
                  )}
                  <div className="mt-4">
                    <label className="block text-sm font-medium text-gray-700 mb-2">流式输出速度</label>
                    <CustomSelect
                      value={streamSpeed}
                      onChange={setStreamSpeed}
                      options={[
                        { value: 'fast', label: '快速 (3字符/次, ~20ms)' },
                        { value: 'normal', label: '正常 (2字符/次, ~30ms)' },
                        { value: 'slow', label: '慢速 (1字符/次, ~60ms)' },
                        { value: 'off', label: '关闭流式（直接显示）' }
                      ]}
                    />
                    <p className="text-xs text-gray-500 mt-1">调整AI回复的打字机效果速度</p>
                  </div>
                  <label className="flex items-center justify-between cursor-pointer p-2 hover:bg-gray-50 rounded-lg mt-3">
                    <span className="font-medium">Blur Reveal 效果</span>
                    <input type="checkbox" checked={enableBlurReveal} onChange={e => setEnableBlurReveal(e.target.checked)} className="accent-purple-600 w-5 h-5" />
                  </label>
                  <p className="text-xs text-gray-500 ml-2 mb-2">流式输出时每个新字符从模糊到清晰的渐变效果</p>
                  {enableBlurReveal && (
                    <div className="ml-2 mt-2">
                      <label className="block text-sm font-medium text-gray-700 mb-2">模糊效果强度</label>
                      <CustomSelect
                        value={blurIntensity}
                        onChange={setBlurIntensity}
                        options={[
                          { value: 'light', label: '轻度 (3px blur, 0.2s)' },
                          { value: 'medium', label: '中度 (5px blur, 0.25s)' },
                          { value: 'strong', label: '强烈 (8px blur, 0.3s)' }
                        ]}
                      />
                    </div>
                  )}
                </div>

                {/* 全局设置入口 */}
                <div className="pt-4 border-t border-gray-100">
                  <button onClick={() => { setShowSettings(false); setShowGlobalSettings(true); }} className="soft-card w-full px-4 py-3 rounded-xl font-medium hover:scale-105 transition-transform flex items-center justify-center gap-2">
                    <Type className="w-4 h-4" /> 全局设置（字体、缩放）
                  </button>
                </div>
                <div className="pt-4 border-t border-gray-100">
                  <button onClick={() => { setShowSettings(false); setShowChatSettings(true); }} className="soft-card w-full px-4 py-3 rounded-xl font-medium hover:scale-105 transition-transform flex items-center justify-center gap-2">
                    <SlidersHorizontal className="w-4 h-4" /> 对话设置（温度、Token、流式）
                  </button>
                </div>
                <div className="pt-4 border-t border-gray-100">
                  <button onClick={() => { setShowSettings(false); setShowOCRSettings(true); }} className="soft-card w-full px-4 py-3 rounded-xl font-medium hover:scale-105 transition-transform flex items-center justify-center gap-2">
                    <ScanText className="w-4 h-4" /> OCR 设置（文字识别）
                  </button>
                </div>

                {/* 工具栏设置 */}
                <div className="pt-4 border-t border-gray-100 space-y-3">
                  <h3 className="text-sm font-semibold text-gray-800">划词工具栏</h3>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">默认搜索引擎</label>
                    <CustomSelect
                      value={searchEngine}
                      onChange={setSearchEngine}
                      options={[
                        { value: 'google', label: 'Google' },
                        { value: 'bing', label: 'Bing' },
                        { value: 'baidu', label: '百度' },
                        { value: 'sogou', label: '搜狗' },
                        { value: 'custom', label: '自定义' }
                      ]}
                    />
                    {searchEngine === 'custom' && (
                      <div className="mt-2 space-y-1">
                        <input type="text" value={searchEngineUrl} onChange={(e) => setSearchEngineUrl(e.target.value)} className="w-full p-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-purple-500 outline-none" placeholder="例如：https://www.google.com/search?q={query}" />
                        <p className="text-xs text-gray-500">使用 <code className="font-mono">{'{query}'}</code> 作为搜索词占位符</p>
                      </div>
                    )}
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">工具栏尺寸</label>
                    <CustomSelect
                      value={toolbarSize}
                      onChange={setToolbarSize}
                      options={[
                        { value: 'compact', label: '紧凑' },
                        { value: 'normal', label: '常规' },
                        { value: 'large', label: '大号' }
                      ]}
                    />
                  </div>
                </div>

                {/* 存储位置信息 */}
                <div className="pt-4 border-t border-gray-100">
                  <h3 className="text-sm font-semibold text-gray-800 mb-3">文件存储位置</h3>
                  {storageInfo ? (
                    <div className="space-y-2">
                      <div className="bg-gray-50 p-3 rounded-lg">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-medium text-gray-600">PDF文件</span>
                          <span className="text-xs text-gray-500">{storageInfo.pdf_count} 个文件</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <code className="flex-1 text-xs bg-white px-2 py-1 rounded border border-gray-200 overflow-x-auto whitespace-nowrap">{storageInfo.uploads_dir}</code>
                          <button onClick={() => { navigator.clipboard.writeText(storageInfo.uploads_dir); alert('路径已复制到剪贴板！'); }} className="p-1.5 hover:bg-purple-100 text-purple-600 rounded transition-colors" title="复制路径">
                            <Copy className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                      <div className="bg-gray-50 p-3 rounded-lg">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-medium text-gray-600">对话历史</span>
                          <span className="text-xs text-gray-500">{storageInfo.doc_count} 个文档</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <code className="flex-1 text-xs bg-white px-2 py-1 rounded border border-gray-200 overflow-x-auto whitespace-nowrap">{storageInfo.data_dir}</code>
                          <button onClick={() => { navigator.clipboard.writeText(storageInfo.data_dir); alert('路径已复制到剪贴板！'); }} className="p-1.5 hover:bg-purple-100 text-purple-600 rounded transition-colors" title="复制路径">
                            <Copy className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        点击复制按钮复制路径，然后在{storageInfo.platform === 'Windows' ? '文件资源管理器' : storageInfo.platform === 'Darwin' ? 'Finder' : '文件管理器'}中打开
                      </p>
                    </div>
                  ) : (
                    <div className="text-sm text-gray-500">加载中...</div>
                  )}
                </div>
              </div>

              <div className="p-8 pt-4 flex-shrink-0 border-t border-gray-100">
                <button onClick={() => setShowSettings(false)} className="w-full py-3 soft-button soft-button-primary rounded-xl font-medium hover:shadow-lg transition-all">
                  Save Changes
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 懒加载设置面板（使用 useCallback 稳定的关闭回调） */}
      <Suspense fallback={null}>
        <EmbeddingSettings isOpen={showEmbeddingSettings} onClose={handleEmbeddingSettingsClose} />
      </Suspense>
      <Suspense fallback={null}>
        <GlobalSettings isOpen={showGlobalSettings} onClose={handleGlobalSettingsClose} />
      </Suspense>
      <Suspense fallback={null}>
        <ChatSettings isOpen={showChatSettings} onClose={handleChatSettingsClose} />
      </Suspense>
      <Suspense fallback={null}>
        <OCRSettingsPanel isOpen={showOCRSettings} onClose={handleOCRSettingsClose} />
      </Suspense>

      {/* 反馈 Modal */}
      <AnimatePresence>
        {feedbackTarget && (
          <FeedbackModal
            onSubmit={handleFeedbackSubmit}
            onClose={() => setFeedbackTarget(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

// 反馈 Modal 组件
const ISSUE_OPTIONS = [
  { value: 'wrong_answer', label: '答案错误' },
  { value: 'wrong_citation', label: '引文不对' },
  { value: 'irrelevant', label: '答非所问' },
  { value: 'offensive', label: '内容不当' },
];

const FeedbackModal = ({ onSubmit, onClose }) => {
  const [selectedIssues, setSelectedIssues] = useState([]);
  const [detail, setDetail] = useState('');

  const toggleIssue = (val) => {
    setSelectedIssues(prev =>
      prev.includes(val) ? prev.filter(v => v !== val) : [...prev, val]
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
        className="bg-white rounded-2xl shadow-xl p-5 w-80 max-w-[90vw]"
        onClick={e => e.stopPropagation()}
      >
        <h3 className="text-sm font-semibold text-gray-800 mb-3">反馈问题</h3>
        <div className="flex flex-wrap gap-2 mb-3">
          {ISSUE_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => toggleIssue(opt.value)}
              className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                selectedIssues.includes(opt.value)
                  ? 'border-orange-400 bg-orange-50 text-orange-600'
                  : 'border-gray-200 text-gray-500 hover:border-gray-300'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <textarea
          value={detail}
          onChange={e => setDetail(e.target.value)}
          placeholder="补充说明（可选）"
          className="w-full text-xs border border-gray-200 rounded-lg p-2.5 resize-none h-16 focus:outline-none focus:ring-1 focus:ring-orange-300 mb-3"
        />
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="text-xs px-3 py-1.5 text-gray-500 hover:text-gray-700">取消</button>
          <button
            onClick={() => onSubmit(selectedIssues, detail)}
            disabled={selectedIssues.length === 0}
            className="text-xs px-4 py-1.5 rounded-lg bg-orange-500 text-white hover:bg-orange-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            提交
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

// 自定义下拉选择组件
const CustomSelect = ({ value, onChange, options }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedLabel = options.find(opt => opt.value === value)?.label || value;

  return (
    <div className="relative" ref={containerRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-3 rounded-[18px] border border-gray-200 bg-white/50 backdrop-blur-sm flex items-center justify-between hover:border-purple-300 transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500/20"
      >
        <span className="text-sm font-medium text-gray-700">{selectedLabel}</span>
        <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 400, damping: 25, mass: 0.8 }}
            style={{ transformOrigin: 'top center' }}
            className="absolute top-full left-0 right-0 mt-2 z-50 overflow-hidden rounded-[18px] border border-gray-100 bg-white/90 backdrop-blur-md shadow-xl ring-1 ring-black/5"
          >
            <div className="py-1 max-h-60 overflow-auto custom-scrollbar">
              {options.map((option) => (
                <button
                  key={option.value}
                  onClick={() => { onChange(option.value); setIsOpen(false); }}
                  className={`w-full text-left px-4 py-2.5 text-sm transition-colors flex items-center justify-between ${option.value === value ? 'bg-purple-50 text-purple-600 font-medium' : 'text-gray-700 hover:bg-gray-50'}`}
                >
                  {option.label}
                  {option.value === value && <div className="w-1.5 h-1.5 rounded-full bg-purple-500" />}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ChatPDF;
