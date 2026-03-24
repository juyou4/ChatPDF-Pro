import React, { useState, useRef, useEffect, useMemo, useCallback, lazy, Suspense } from 'react';
import { Upload, Send, Settings, ChevronLeft, ChevronRight, ChevronDown, ZoomIn, ZoomOut, Copy, Bot, X, Crop, Image as ImageIcon, History, Moon, Sun, Plus, MessageSquare, Trash2, Menu, Type, Loader2, Server, Database, ListFilter, ArrowUpRight, SlidersHorizontal, Paperclip, ScanText, Scan, Brain, MessageCircle, ArrowUpDown, Globe, Check } from 'lucide-react';
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
const OverviewPanel = lazy(() => import('./OverviewPanel'));
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

const MEMORY_KIND_LABELS = {
  working: '工作记忆',
  profile: '画像',
  doc_fact: '文档事实',
  episodic: '对话摘要',
  consolidated: '压缩事实',
  graph: '图谱',
};

const MemoryHitsBadge = ({ hits, meta }) => {
  const [expanded, setExpanded] = useState(false);
  if (!Array.isArray(hits) || hits.length === 0) return null;

  return (
    <div className="mt-3 border-t border-gray-100 pt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-emerald-600 hover:text-emerald-800 transition-colors font-medium"
      >
        <Brain className="w-3.5 h-3.5" />
        <span>记忆命中 ({hits.length})</span>
        {meta?.truncated && <span className="rounded-full bg-emerald-50 px-1.5 py-0.5 text-[10px] text-emerald-700">已截断</span>}
        <ChevronDown className={`w-3 h-3 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>
      {expanded && (
        <div className="mt-2 space-y-2">
          {hits.map((hit, i) => (
            <div key={hit.id || `${hit.memory_kind}-${i}`} className="rounded-xl border border-emerald-100 bg-emerald-50/60 p-2.5">
              <div className="flex items-center gap-2 text-[11px] text-emerald-700">
                <span className="rounded-full bg-white px-1.5 py-0.5 font-semibold">{MEMORY_KIND_LABELS[hit.memory_kind] || hit.memory_kind || '记忆'}</span>
                {hit.memory_scope && <span>{hit.memory_scope === 'profile' ? '全局' : '当前文档'}</span>}
                {typeof hit.score === 'number' && <span>score {hit.score.toFixed(2)}</span>}
              </div>
              <div className="mt-1 text-xs font-medium text-gray-800">{hit.title || '记忆条目'}</div>
              <div className="mt-1 text-xs leading-5 text-gray-600">{hit.summary || hit.content}</div>
            </div>
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
  const [enableVectorSearch, setEnableVectorSearch] = useDebouncedLocalStorage('enableVectorSearch', true);
  // 一次性迁移：旧版默认 enableVectorSearch=false，需强制升级为 true
  useEffect(() => {
    if (!localStorage.getItem('_migrated_vectorSearch_v1')) {
      setEnableVectorSearch(true);
      localStorage.setItem('_migrated_vectorSearch_v1', '1');
    }
  }, []);
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
    rightPanelMode, setRightPanelMode,
    overviewDepth, setOverviewDepth,
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

  const getEmbeddingApiKey = useCallback(() => {
    const config = getEmbeddingConfig();
    console.log('[DEBUG getEmbeddingApiKey]', {
      isValid: config.isValid,
      reason: config.reason,
      modelType: config.model?.type,
      providerId: config.providerId,
      hasProviderKey: !!config.provider?.apiKey,
      fallbackKey: embeddingApiKey ? 'embeddingApiKey' : apiKey ? 'apiKey' : 'none',
    });
    if (config.isValid && config.provider?.apiKey) {
      return config.provider.apiKey;
    }
    return embeddingApiKey || apiKey;
  }, [getEmbeddingConfig, embeddingApiKey, apiKey]);

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
    return { providerId, modelId, apiKey: provider?.apiKey || getEmbeddingApiKey() };
  }, [getCurrentRerankModel, getProviderById, getEmbeddingApiKey]);

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
    getChatCredentials,
    getProviderById,
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
    overview, overviewLoading, overviewError, fetchOverview,
  } = documentState;

  // ========== PDF 状态 Hook（需求 1.1） ==========
  const pdfState = usePDFState({
    docId,
    docInfo,
    useRerank: useRerankSetting,
    rerankerModel,
    getRerankCredentials,
    embeddingApiKey: getEmbeddingApiKey(),
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
    embeddingApiKey: getEmbeddingApiKey(),
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
            webSearchSources={msg.webSearchSources || null}
          />
          {/* 联网搜索来源 */}
          {msg.webSearchSources && msg.webSearchSources.length > 0 && !msg.isStreaming && (
            <WebSearchSourcesBadge sources={msg.webSearchSources} />
          )}
          {msg.type === 'assistant' && !msg.isStreaming && msg.memoryHits && msg.memoryHits.length > 0 && (
            <MemoryHitsBadge hits={msg.memoryHits} meta={msg.memoryMeta} />
          )}
        </div>
        {/* 证据面板 */}
        {msg.type === 'assistant' && !msg.isStreaming && msg.citations && msg.citations.length > 0 && (
          <EvidencePanel
            citations={msg.citations}
            docId={docId}
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
    docId,
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
        animate={{ width: showSidebar ? 320 : 0, opacity: showSidebar ? 1 : 0 }}
        transition={{ duration: 0.2, ease: "easeInOut" }}
        style={{ pointerEvents: showSidebar ? 'auto' : 'none' }}
        className={`flex-shrink-0 m-6 mr-0 h-[calc(100vh-3rem)] flex flex-col z-20 overflow-hidden ${darkMode ? 'bg-[#1a1d21]/90 border-white/5 backdrop-blur-3xl backdrop-saturate-150 rounded-[40px]' : 'bg-white/80 backdrop-blur-2xl border border-white/70 rounded-[40px] shadow-2xl shadow-gray-300/40 shadow-[inset_0_1px_1px_rgba(255,255,255,0.8)]'}`}
      >
        <div className="w-[320px] mx-auto flex flex-col h-full items-stretch relative">
          <div className="px-8 py-8 flex items-center justify-between mb-2">
            <div className="flex items-center gap-3 font-bold text-2xl text-purple-600 tracking-tight pl-2">
              <Bot className="w-9 h-9" />
              <span>ChatPDF</span>
            </div>
            <div className="flex items-center gap-1">
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

          <div className="flex-1 overflow-y-auto px-8">
            <h2 className="text-[11px] font-bold text-gray-500 tracking-wider mb-4 pl-4">
              HISTORY
            </h2>
            <ul className="space-y-2 relative">
              {history.map((item, idx) => {
                const isActive = item.id === docId;
                return (
                  <li key={idx}>
                    <div
                      onClick={() => loadSession(item)}
                      className={`w-full flex items-center justify-between px-5 py-4 rounded-3xl cursor-pointer group transition-all duration-300 ${
                        isActive
                          ? (darkMode ? 'bg-white/10 backdrop-blur-md border border-white/10 text-white font-bold shadow-[0_10px_25px_rgba(0,0,0,0.2)]' : 'bg-white/80 backdrop-blur-md border border-white/80 text-gray-900 font-bold shadow-[0_10px_25px_rgba(0,0,0,0.06)]')
                          : (darkMode ? 'text-gray-400 font-medium hover:bg-white/5' : 'text-gray-600 font-medium hover:bg-white/40')
                      }`}
                    >
                      <div className="flex items-center gap-4 overflow-hidden">
                        <MessageSquare 
                          size={22} 
                          className={`${isActive ? 'text-[#9333ea]' : 'text-gray-500'} transition-colors flex-shrink-0`}
                          strokeWidth={isActive ? 2.5 : 2}
                        />
                        <span className="text-[15px] truncate">{item.filename}</span>
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); if (!confirmDeleteMessage || confirm('确定要删除这条对话记录吗？')) deleteSession(item.id); }}
                        className={`opacity-0 group-hover:opacity-100 p-1.5 rounded-full hover:bg-red-50 hover:text-red-500 transition-all flex-shrink-0 ${isActive ? 'text-gray-400' : 'text-gray-400'}`}
                      >
                        <Trash2 size={18} strokeWidth={2} />
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className="px-8 py-6">
            <button 
              onClick={() => { setShowSettings(true); fetchStorageInfo(); }} 
              className={`w-full flex items-center gap-4 px-5 py-4 rounded-3xl transition-all duration-300 ${darkMode ? 'text-gray-400 font-medium hover:bg-white/5' : 'text-gray-600 font-medium hover:bg-white/40'}`}
            >
              <Settings size={22} className="text-gray-500" strokeWidth={2} />
              <span className="text-[15px]">设置 & API Key</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* 主内容区域 */}
      <div className="flex-1 flex flex-col h-full relative transition-all duration-200 ease-in-out">
        {/* 侧边栏展开按钮 (未打开文档时显示) */}
        {!showSidebar && !docId && (
          <button
            onClick={() => setShowSidebar(true)}
            className={`absolute top-4 left-4 z-20 p-2 backdrop-blur-md shadow-sm rounded-full hover:scale-105 transition-all border ${darkMode ? 'bg-white/10 text-gray-300 border-white/10 hover:bg-white/20' : 'bg-white/80 text-gray-700 border-white/50 hover:bg-white'}`}
            title="显示侧边栏"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

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
                    onToggleSidebar={() => setShowSidebar(prev => !prev)}
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

          {/* 右侧：聊天/速览区域 */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`soft-panel flex flex-col overflow-hidden rounded-[var(--radius-panel)] min-w-0 ${darkMode ? 'bg-gray-800/50' : ''}`}
            style={{ width: `calc(${100 - pdfPanelWidth}% - 2rem)`, minWidth: '350px' }}
          >
            {/* 切换按钮：速览 / 对话 */}
            <div className="flex items-center gap-1 px-6 pt-4 pb-2 border-b border-gray-100/50">
              <button
                onClick={() => setRightPanelMode('overview')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  rightPanelMode === 'overview'
                    ? 'bg-purple-100 text-purple-700 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
              >
                速览
              </button>
              <button
                onClick={() => setRightPanelMode('chat')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  rightPanelMode === 'chat'
                    ? 'bg-purple-100 text-purple-700 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
              >
                对话
              </button>
            </div>

            {/* 内容区域：根据模式显示速览或对话 */}
            <div className="flex-1 overflow-hidden flex flex-col min-w-0">
              {rightPanelMode === 'overview' ? (
                <Suspense fallback={
                  <div className="flex-1 flex items-center justify-center text-gray-400">
                    <Loader2 className="w-6 h-6 animate-spin mr-2" />
                    加载中...
                  </div>
                }>
                  <OverviewPanel
                    docId={docId}
                    overview={overview}
                    loading={overviewLoading}
                    error={overviewError}
                    depth={overviewDepth}
                    onDepthChange={setOverviewDepth}
                    onFetch={fetchOverview}
                  />
                </Suspense>
              ) : (
                <>
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
                </>
              )}
            </div>

            {/* 输入区域 */}
            <div className="p-6 pt-0 bg-transparent relative z-10">
              {/* 截图预览 */}
              <ScreenshotPreview
                screenshots={screenshots}
                onAction={handleScreenshotAction}
                onClose={handleScreenshotClose}
              />

              <div className="absolute bottom-5 left-3 right-3 bg-[#f2f3f9] shadow-[0_10px_35px_rgba(0,0,0,0.1),inset_0_1px_0_rgba(255,255,255,0.8)] border border-white/60 rounded-[2rem] p-2.5 z-20">
                {/* 上半部分：模型选择、状态、工具图标 */}
                <div className="flex items-center justify-between mb-2.5 px-1">
                  <ModelQuickSwitch onThinkingChange={handleThinkingChange} />
                  
                  {/* 右侧工具图标 */}
                  <div className="flex items-center gap-2 text-gray-500 shrink-0">
                    <button onClick={() => setShowSettings(true)} className="hover:text-gray-800 transition-colors p-1 rounded-md">
                      <Settings size={15} />
                    </button>
                    <button onClick={() => fileInputRef.current?.click()} className="hover:text-gray-800 transition-colors p-1 rounded-md">
                      <Paperclip size={15} />
                    </button>
                    <WebSearchButton />
                    {isVisionCapable && (
                      <button
                        onClick={() => setIsSelectingArea(true)}
                        disabled={!docId}
                        className={`transition-colors p-1 rounded-md ${docId ? isSelectingArea ? 'text-purple-600' : 'hover:text-gray-800' : 'text-gray-300 cursor-not-allowed'}`}
                        title={!docId ? '请先上传文档' : isSelectingArea ? '框选模式已开启' : '区域截图'}
                      >
                        <Scan size={15} />
                      </button>
                    )}
                  </div>
                </div>

                {/* 下半部分：输入区 */}
                <div className="flex items-center bg-white rounded-full p-1.5 shadow-sm border border-black/5 focus-within:ring-2 focus-within:ring-purple-100 focus-within:border-purple-200 transition-all">
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
                    className="flex-1 bg-transparent outline-none px-4 text-[14px] text-gray-800 min-w-0 resize-none h-[24px] overflow-hidden leading-relaxed py-0"
                    rows={1}
                    style={{ minHeight: '24px', maxHeight: '120px' }}
                  />
                  
                  {/* Send 文字和发送按钮，固定在右侧 */}
                  <div className="flex items-center gap-3 pr-1 shrink-0">
                    <span className="text-[#aba6d1] text-[13px] font-medium select-none pointer-events-none">Send</span>
                    <button
                      onClick={isLoading ? handleStop : sendMessage}
                      disabled={!isLoading && (!hasInput && screenshots.length === 0)}
                      className={`w-9 h-9 rounded-full transition-colors flex items-center justify-center shadow-sm ${
                        isLoading || hasInput || screenshots.length > 0 
                          ? 'bg-[#f0efff] text-[#7c3aed] hover:bg-[#7c3aed] hover:text-white' 
                          : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      }`}
                    >
                      <AnimatePresence initial={false}>
                        {isLoading ? (
                          <motion.div key="pause" initial={{ rotate: -90, scale: 0.5, opacity: 0 }} animate={{ rotate: 0, scale: 1, opacity: 1 }} exit={{ rotate: 90, scale: 0.5, opacity: 0 }} transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }} className="absolute flex items-center justify-center">
                            <PauseIcon />
                          </motion.div>
                        ) : (
                          <motion.div key="send" initial={{ rotate: -90, scale: 0.5, opacity: 0 }} animate={{ rotate: 0, scale: 1, opacity: 1 }} exit={{ rotate: 90, scale: 0.5, opacity: 0 }} transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }} className="absolute flex items-center justify-center ml-0.5">
                            <SendIcon />
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </button>
                  </div>
                </div>
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
              className={`w-[460px] max-w-full max-h-[90vh] overflow-hidden flex flex-col ${darkMode ? 'bg-[#1a1d21]/95 border border-white/5 backdrop-blur-md rounded-[36px] shadow-2xl' : 'bg-white/90 backdrop-blur-md border border-white/60 rounded-[36px] shadow-[0_32px_80px_-20px_rgba(0,0,0,0.12)] relative'}`}
            >
              <div className="p-6 pb-2 flex-shrink-0 flex items-center justify-between mt-1 px-7">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-2xl shadow-sm border ${darkMode ? 'bg-white/10 border-white/10' : 'bg-white/60 border-white/50'}`}>
                    <Settings className="text-[#7c4dff]" size={22} />
                  </div>
                  <h2 className={`text-xl font-bold tracking-tight ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>Settings</h2>
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => { setShowSettings(false); setShowEmbeddingSettings(true); }} className={`transition-all duration-300 hover:shadow-[0_8px_20px_rgba(42,36,66,0.3)] hover:-translate-y-0.5 text-white text-xs font-semibold px-3.5 py-2 rounded-full ${darkMode ? 'bg-[#3a3452] hover:bg-[#2a2442]' : 'bg-[#2a2442] hover:bg-[#1a1528]'}`}>
                    Manage Models
                  </button>
                  <button onClick={() => setShowSettings(false)} className={`p-2 rounded-full transition-colors z-10 ${darkMode ? 'hover:bg-white/10 text-gray-500 hover:text-gray-300' : 'hover:bg-black/5 text-gray-400 hover:text-gray-700'}`}>
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>

              <div className="space-y-5 px-6 overflow-y-auto flex-1 pb-6 custom-scrollbar">
                
                <div className="space-y-3.5 px-1">
                  {/* Chat Model Card */}
                  <div className={`backdrop-blur-md rounded-[24px] p-4 flex items-center space-x-4 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border ${darkMode ? 'bg-white/5 border-white/10' : 'bg-white/80 border-white/60'}`}>
                    <div className="w-[46px] h-[46px] rounded-[16px] bg-[#7c4dff]/10 flex items-center justify-center text-[#7c4dff] shrink-0 border border-white/50 shadow-inner">
                      <MessageSquare size={22} />
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <h3 className={`text-[13px] font-bold uppercase tracking-wider ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        CHAT MODEL
                      </h3>
                      <p className={`text-[12px] mt-0.5 font-medium truncate ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} title={getDefaultModelLabel(getDefaultModel('assistantModel'))}>
                        {getDefaultModelLabel(getDefaultModel('assistantModel')) || '未设置'}
                      </p>
                    </div>
                  </div>

                  {/* Embedding Model Card */}
                  <div className={`backdrop-blur-md rounded-[24px] p-4 flex items-center space-x-4 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border ${darkMode ? 'bg-white/5 border-white/10' : 'bg-white/80 border-white/60'}`}>
                    <div className="w-[46px] h-[46px] rounded-[16px] bg-[#7c4dff]/10 flex items-center justify-center text-[#7c4dff] shrink-0 border border-white/50 shadow-inner">
                      <Database size={22} />
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <h3 className={`text-[13px] font-bold uppercase tracking-wider ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        EMBEDDING
                      </h3>
                      <p className={`text-[12px] mt-0.5 font-medium truncate ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} title={getDefaultModelLabel(getDefaultModel('embeddingModel'))}>
                        {getDefaultModelLabel(getDefaultModel('embeddingModel')) || '未设置'}
                      </p>
                    </div>
                  </div>

                  {/* Rerank Model Card */}
                  <div className={`backdrop-blur-md rounded-[24px] p-4 flex items-center space-x-4 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border ${darkMode ? 'bg-white/5 border-white/10' : 'bg-white/80 border-white/60'}`}>
                    <div className="w-[46px] h-[46px] rounded-[16px] bg-[#7c4dff]/10 flex items-center justify-center text-[#7c4dff] shrink-0 border border-white/50 shadow-inner">
                      <ArrowUpDown size={22} />
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <h3 className={`text-[13px] font-bold uppercase tracking-wider ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        RERANK
                      </h3>
                      <p className={`text-[12px] mt-0.5 font-medium truncate ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} title={getDefaultModelLabel(getDefaultModel('rerankModel'))}>
                        {getDefaultModelLabel(getDefaultModel('rerankModel')) || '未设置'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Features Section - Glass Inner Panel */}
                <div className={`backdrop-blur-md rounded-[28px] p-5 shadow-[0_8px_30px_rgb(0,0,0,0.03)] border space-y-3 mt-2 mx-1 ${darkMode ? 'bg-white/5 border-white/10' : 'bg-white/80 border-white/60'}`}>
                  
                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableVectorSearch ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableVectorSearch && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>Vector Search</h4>
                        <input type="checkbox" checked={enableVectorSearch} onChange={e => setEnableVectorSearch(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        基于向量的语义相似度检索，提供更准确的匹配
                      </p>
                    </div>
                  </label>

                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableScreenshot ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableScreenshot && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>Screenshot Analysis</h4>
                        <input type="checkbox" checked={enableScreenshot} onChange={e => setEnableScreenshot(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        启用截图分析功能，理解视觉内容及图表
                      </p>
                    </div>
                  </label>

                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableGraphRAG ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableGraphRAG && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>GraphRAG 知识图谱</h4>
                        <input type="checkbox" checked={enableGraphRAG} onChange={e => setEnableGraphRAG(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        实体关系提取 + 社区聚类增强检索，提供全局视角
                      </p>
                    </div>
                  </label>

                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableJiebaBM25 ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableJiebaBM25 && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>jieba 中文分词</h4>
                        <input type="checkbox" checked={enableJiebaBM25} onChange={e => setEnableJiebaBM25(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        使用结巴分词提升 BM25 中文关键词匹配精度
                      </p>
                    </div>
                  </label>

                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableBlurReveal ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableBlurReveal && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>Blur Reveal 效果</h4>
                        <input type="checkbox" checked={enableBlurReveal} onChange={e => setEnableBlurReveal(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        流式输出时每个新字符从模糊到清晰的渐变效果
                      </p>
                    </div>
                  </label>
                </div>

                {/* Toolbar and Storage Settings Area */}
                <div className={`backdrop-blur-md rounded-[28px] p-5 shadow-[0_8px_30px_rgb(0,0,0,0.03)] border space-y-4 mt-4 mx-1 ${darkMode ? 'bg-white/5 border-white/10' : 'bg-white/80 border-white/60'}`}>
                  {/* Toolbar Settings */}
                  <div className="space-y-3">
                    <h3 className={`text-[13px] font-bold tracking-wider uppercase ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>工具栏配置</h3>
                    <div className="space-y-3">
                      <div className="flex flex-col gap-1.5">
                        <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>默认搜索引擎</label>
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
                          <div className="mt-1">
                            <input type="text" value={searchEngineUrl} onChange={(e) => setSearchEngineUrl(e.target.value)} className={`w-full p-2.5 rounded-[12px] border text-sm outline-none transition-all ${darkMode ? 'bg-black/20 border-white/10 text-white focus:border-[#7c4dff]/50' : 'bg-white/50 border-gray-200 focus:border-[#7c4dff]/50'}`} placeholder="例如：https://www.google.com/search?q={query}" />
                            <p className="text-[11px] text-gray-500 mt-1">使用 <code className="font-mono bg-black/5 px-1 rounded">{'<query>'}</code> 作为搜索词占位符</p>
                          </div>
                        )}
                      </div>
                      <div className="flex flex-col gap-1.5 pt-1">
                        <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>工具栏尺寸</label>
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
                  </div>

                  {/* Storage Info */}
                  <div className="pt-4 border-t border-gray-200/50">
                    <h3 className={`text-[13px] font-bold tracking-wider uppercase mb-3 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>存储信息</h3>
                    {storageInfo ? (
                      <div className="space-y-2">
                        <div className={`p-3 rounded-[16px] transition-colors ${darkMode ? 'bg-black/20 hover:bg-black/30' : 'bg-white/50 hover:bg-white/80'}`}>
                          <div className="flex items-center justify-between mb-1.5">
                            <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>PDF文件 ({storageInfo.pdf_count})</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className={`flex-1 text-[11px] px-2.5 py-1.5 rounded-[8px] overflow-x-auto whitespace-nowrap font-mono border ${darkMode ? 'bg-black/40 border-white/5 text-gray-400' : 'bg-white border-gray-100 text-gray-500'}`}>
                              {storageInfo.uploads_dir}
                            </div>
                            <button onClick={() => { navigator.clipboard.writeText(storageInfo.uploads_dir); alert('路径已复制到剪贴板！'); }} className="p-1.5 rounded-[8px] bg-[#7c4dff]/10 text-[#7c4dff] hover:bg-[#7c4dff]/20 transition-colors shrink-0" title="复制路径">
                              <Copy size={14} />
                            </button>
                          </div>
                        </div>
                        <div className={`p-3 rounded-[16px] transition-colors ${darkMode ? 'bg-black/20 hover:bg-black/30' : 'bg-white/50 hover:bg-white/80'}`}>
                          <div className="flex items-center justify-between mb-1.5">
                            <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>对话历史 ({storageInfo.doc_count})</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className={`flex-1 text-[11px] px-2.5 py-1.5 rounded-[8px] overflow-x-auto whitespace-nowrap font-mono border ${darkMode ? 'bg-black/40 border-white/5 text-gray-400' : 'bg-white border-gray-100 text-gray-500'}`}>
                              {storageInfo.data_dir}
                            </div>
                            <button onClick={() => { navigator.clipboard.writeText(storageInfo.data_dir); alert('路径已复制到剪贴板！'); }} className="p-1.5 rounded-[8px] bg-[#7c4dff]/10 text-[#7c4dff] hover:bg-[#7c4dff]/20 transition-colors shrink-0" title="复制路径">
                              <Copy size={14} />
                            </button>
                          </div>
                        </div>
                        <p className="text-[11px] text-gray-500 mt-2 px-1">
                          在 {storageInfo.platform === 'Windows' ? '文件资源管理器' : storageInfo.platform === 'Darwin' ? 'Finder' : '文件管理器'} 中打开以管理文件
                        </p>
                      </div>
                    ) : (
                      <div className="text-[12px] text-gray-500 py-2">加载中...</div>
                    )}
                  </div>
                </div>

                {/* Advanced Configuration Section */}
                <div className={`backdrop-blur-md rounded-[28px] p-5 shadow-[0_8px_30px_rgb(0,0,0,0.03)] border space-y-4 mt-4 mx-1 ${darkMode ? 'bg-white/5 border-white/10' : 'bg-white/80 border-white/60'}`}>
                  <h3 className={`text-[13px] font-bold tracking-wider uppercase mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>高级配置</h3>
                  
                  <div className="space-y-3">
                    <div className="flex flex-col gap-1.5">
                      <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>邻居上下文扩展</label>
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
                      <p className="text-[11px] text-gray-500 mt-0.5">命中 chunk 前后各扩展 N 个邻居块作为上下文</p>
                    </div>

                    <div className="flex flex-col gap-1.5 pt-2 border-t border-gray-200/50 mt-2">
                      <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>流式输出速度</label>
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
                      <p className="text-[11px] text-gray-500 mt-0.5">调整AI回复的打字机效果速度</p>
                    </div>

                    {enableBlurReveal && (
                      <div className="flex flex-col gap-1.5 pt-2 border-t border-gray-200/50 mt-2">
                        <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>模糊效果强度</label>
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

                  {lastCallInfo && (
                    <div className={`mt-4 p-3.5 rounded-[16px] border text-[12px] ${darkMode ? 'bg-black/20 border-white/5' : 'bg-white/50 border-gray-100'}`}>
                      <div className="flex justify-between items-center mb-1.5">
                        <span className="text-gray-500">调用来源</span>
                        <strong className={darkMode ? 'text-gray-300' : 'text-gray-700'}>{lastCallInfo.provider || '未知'}</strong>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-500">模型</span>
                        <strong className={darkMode ? 'text-gray-300' : 'text-gray-700'}>{lastCallInfo.model || '未返回'}</strong>
                      </div>
                      {lastCallInfo.fallback && (
                        <div className="mt-2 pt-2 border-t border-gray-200/50 text-amber-600 font-medium flex items-center justify-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span>
                          已切换备用模型
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Other Settings Access */}
                <div className="grid grid-cols-3 gap-3 px-1 mt-4">
                  <button onClick={() => { setShowSettings(false); setShowGlobalSettings(true); }} className={`flex flex-col items-center justify-center p-3 rounded-[20px] border transition-all hover:-translate-y-1 ${darkMode ? 'bg-white/5 border-white/10 hover:bg-white/10' : 'bg-white/60 border-white/50 hover:bg-white/80'}`}>
                    <Type className={`w-5 h-5 mb-1.5 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} />
                    <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>全局设置</span>
                  </button>
                  <button onClick={() => { setShowSettings(false); setShowChatSettings(true); }} className={`flex flex-col items-center justify-center p-3 rounded-[20px] border transition-all hover:-translate-y-1 ${darkMode ? 'bg-white/5 border-white/10 hover:bg-white/10' : 'bg-white/60 border-white/50 hover:bg-white/80'}`}>
                    <SlidersHorizontal className={`w-5 h-5 mb-1.5 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} />
                    <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>对话设置</span>
                  </button>
                  <button onClick={() => { setShowSettings(false); setShowOCRSettings(true); }} className={`flex flex-col items-center justify-center p-3 rounded-[20px] border transition-all hover:-translate-y-1 ${darkMode ? 'bg-white/5 border-white/10 hover:bg-white/10' : 'bg-white/60 border-white/50 hover:bg-white/80'}`}>
                    <ScanText className={`w-5 h-5 mb-1.5 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} />
                    <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>OCR设置</span>
                  </button>
                </div>
              </div>

              <div className="p-6 pt-2 pb-8 flex-shrink-0 relative">
                <button onClick={() => setShowSettings(false)} className="w-full bg-[#7c4dff] hover:bg-[#6836f5] transition-all duration-300 text-white text-[15px] font-semibold py-4 rounded-[22px] shadow-[0_12px_30px_rgba(124,77,255,0.3)] hover:shadow-[0_16px_40px_rgba(124,77,255,0.45)] hover:-translate-y-1 relative overflow-hidden group">
                  <span className="relative z-10">Save Changes</span>
                  <div className="absolute top-0 -inset-full h-full w-1/2 z-5 block transform -skew-x-12 bg-gradient-to-r from-transparent to-white opacity-20 group-hover:animate-shimmer"></div>
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

  const selectedOption = options.find(opt => opt.value === value);

  return (
    <div className="relative w-full" ref={containerRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-2.5 rounded-[12px] bg-white/50 dark:bg-black/20 border border-gray-200 dark:border-white/10 text-sm hover:border-[#7c4dff]/50 transition-all outline-none"
      >
        <span className="text-gray-700 dark:text-gray-300 font-medium">
          {selectedOption ? selectedOption.label : 'Select...'}
        </span>
        <ChevronDown size={14} className={`text-gray-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -5, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -5, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 w-full mt-1.5 p-1.5 bg-white/90 dark:bg-[#1a1d21]/95 backdrop-blur-xl border border-gray-100 dark:border-white/10 rounded-[16px] shadow-[0_8px_30px_rgb(0,0,0,0.08)] max-h-60 overflow-y-auto"
          >
            {options.map((option) => (
              <button
                key={option.value}
                onClick={() => {
                  onChange(option.value);
                  setIsOpen(false);
                }}
                className={`w-full text-left px-3 py-2.5 rounded-[10px] text-[13px] transition-colors flex items-center justify-between ${
                  value === option.value
                    ? 'bg-[#7c4dff]/10 text-[#7c4dff] font-semibold'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100/50 dark:hover:bg-white/5'
                }`}
              >
                {option.label}
                {value === option.value && <Check size={14} className="text-[#7c4dff]" />}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ChatPDF;
































