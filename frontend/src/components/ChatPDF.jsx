import React, { useState, useRef, useEffect, useMemo, useCallback, lazy, Suspense } from 'react';
import { Upload, Send, FileText, Settings, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Copy, Bot, X, Crop, Image as ImageIcon, History, Moon, Sun, Plus, MessageSquare, Trash2, Menu, Type, ChevronUp, ChevronDown, Search, Loader2, Wand2, Server, Database, ListFilter, ArrowUpRight, SlidersHorizontal, Paperclip, ScanText, Scan, Brain, MessageCircle, ArrowUpDown } from 'lucide-react';
import html2canvas from 'html2canvas';
import { motion, AnimatePresence } from 'framer-motion';
import { supportsVision } from '../utils/visionDetectorUtils';
import { captureArea, clampSelectionToPage, SCREENSHOT_ACTIONS } from '../utils/screenshotUtils';
import ScreenshotPreview from './ScreenshotPreview';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import PDFViewer from './PDFViewer';
import StreamingMarkdown from './StreamingMarkdown';
import TextSelectionToolbar from './TextSelectionToolbar';
import { useProvider } from '../contexts/ProviderContext';
import { useModel } from '../contexts/ModelContext';
import { useDefaults } from '../contexts/DefaultsContext';
// 大弹窗懒加载：只在首次打开时才加载对应 chunk，减少初始 bundle 体积
const EmbeddingSettings = lazy(() => import('./EmbeddingSettings'));
const OCRSettingsPanel = lazy(() => import('./OCRSettingsPanel'));
const GlobalSettings = lazy(() => import('./GlobalSettings'));
const ChatSettings = lazy(() => import('./ChatSettings'));
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';

// 内联 OCR 设置读取
const loadOCRSettings = () => {
  try {
    const raw = localStorage.getItem('ocrSettings');
    if (raw) {
      const parsed = JSON.parse(raw);
      const validModes = ['auto', 'always', 'never'];
      const validBackends = ['auto', 'tesseract', 'paddleocr', 'mistral', 'mineru', 'doc2x'];
      return {
        mode: validModes.includes(parsed.mode) ? parsed.mode : 'auto',
        backend: validBackends.includes(parsed.backend) ? parsed.backend : 'auto',
      };
    }
  } catch { /* ignore */ }
  return { mode: 'auto', backend: 'auto' };
};
import PresetQuestions from './PresetQuestions';
import ModelQuickSwitch from './ModelQuickSwitch';
import ThinkingBlock from './ThinkingBlock';
import { useSmoothStream } from '../hooks/useSmoothStream';

// API base URL
const API_BASE_URL = '';

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

const buildChatHistory = (messages, contextCount) => {
  if (!contextCount || contextCount <= 0) return [];
  const validMessages = messages.filter(msg =>
    (msg.type === 'user' || msg.type === 'assistant') && !msg.hasImage
    && !(msg.type === 'assistant' && msg.content && msg.content.startsWith('⚠️ AI未返回内容'))
    && !(msg.type === 'assistant' && msg.content && msg.content.startsWith('❌'))
  );
  const recentMessages = validMessages.slice(-(contextCount * 2));
  return recentMessages.map(msg => ({
    role: msg.type === 'user' ? 'user' : 'assistant',
    content: msg.content
  }));
};

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
  // Core State
  const [docId, setDocId] = useState(null);
  const [docInfo, setDocInfo] = useState(null);
  const [pdfPanelWidth, setPdfPanelWidth] = useState(50);
  const [messages, setMessages] = useState([]);
  const [hasInput, setHasInput] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('uploading');

  // UI State
  const [showSettings, setShowSettings] = useState(false);
  const [showEmbeddingSettings, setShowEmbeddingSettings] = useState(false);
  const [showOCRSettings, setShowOCRSettings] = useState(false);
  const [showGlobalSettings, setShowGlobalSettings] = useState(false);
  const [showChatSettings, setShowChatSettings] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [isHeaderExpanded, setIsHeaderExpanded] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [history, setHistory] = useState([]);

  // PDF State
  const [currentPage, setCurrentPage] = useState(1);
  const [pdfScale, setPdfScale] = useState(1.0);
  const [selectedText, setSelectedText] = useState('');
  const [showTextMenu, setShowTextMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({ x: 0, y: 0 });
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [currentResultIndex, setCurrentResultIndex] = useState(0);
  const [activeHighlight, setActiveHighlight] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [useRerank, setUseRerank] = useState(localStorage.getItem('useRerank') !== 'false');
  const [rerankerModel, setRerankerModel] = useState(localStorage.getItem('rerankerModel') || 'BAAI/bge-reranker-base');
  const [lastCallInfo, setLastCallInfo] = useState(null);
  const [searchHistory, setSearchHistory] = useState([]);

  // Screenshot State (九张截图逻辑)
  const [screenshots, setScreenshots] = useState([]);
  const [isSelectingArea, setIsSelectingArea] = useState(false);

  // Settings State
  const [apiKey, setApiKey] = useState(localStorage.getItem('apiKey') || '');
  const [apiProvider, setApiProvider] = useState(localStorage.getItem('apiProvider') || 'openai');
  const [model, setModel] = useState(localStorage.getItem('model') || 'gpt-4o');
  const [availableModels, setAvailableModels] = useState({});
  const [embeddingApiKey, setEmbeddingApiKey] = useState(localStorage.getItem('embeddingApiKey') || '');
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState({});
  const [enableVectorSearch, setEnableVectorSearch] = useState(localStorage.getItem('enableVectorSearch') === 'true');
  const [enableScreenshot, setEnableScreenshot] = useState(localStorage.getItem('enableScreenshot') !== 'false');
  const [streamSpeed, setStreamSpeed] = useState(localStorage.getItem('streamSpeed') || 'normal');
  const [streamingMessageId, setStreamingMessageId] = useState(null);
  const [storageInfo, setStorageInfo] = useState(null);
  const [enableBlurReveal, setEnableBlurReveal] = useState(localStorage.getItem('enableBlurReveal') !== 'false');
  const [blurIntensity, setBlurIntensity] = useState(localStorage.getItem('blurIntensity') || 'medium');
  const [searchEngine, setSearchEngine] = useState(localStorage.getItem('searchEngine') || 'google');
  const [searchEngineUrl, setSearchEngineUrl] = useState(localStorage.getItem('searchEngineUrl') || 'https://www.google.com/search?q={query}');
  const [toolbarSize, setToolbarSize] = useState(localStorage.getItem('toolbarSize') || 'normal');
  const [toolbarScale, setToolbarScale] = useState(parseFloat(localStorage.getItem('toolbarScale') || '1'));
  const [toolbarPosition, setToolbarPosition] = useState({ x: 0, y: 0 });

  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [likedMessages, setLikedMessages] = useState(new Set());
  const [rememberedMessages, setRememberedMessages] = useState(new Set());
  const [enableThinking, setEnableThinking] = useState(false);

  const [contentStreamDone, setContentStreamDone] = useState(false);
  const [thinkingStreamDone, setThinkingStreamDone] = useState(false);
  const activeStreamMsgIdRef = useRef(null);

  const { getProviderById } = useProvider();
  const { getModelById } = useModel();
  const { getDefaultModel } = useDefaults();
  const {
    maxTokens, temperature, topP, contextCount, streamOutput,
    enableTemperature, enableTopP, enableMaxTokens,
    customParams, reasoningEffort, setReasoningEffort,
    enableMemory,
  } = useGlobalSettings();
  const chatPaneRef = useRef(null);

  const contentStream = useSmoothStream({
    onUpdate: (text) => {
      const msgId = activeStreamMsgIdRef.current;
      if (msgId == null) return;
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last && last.id === msgId) {
          const updated = [...prev];
          updated[updated.length - 1] = { ...last, content: text };
          return updated;
        }
        return prev;
      });
    },
    streamDone: contentStreamDone,
  });

  const thinkingStream = useSmoothStream({
    onUpdate: (text) => {
      const msgId = activeStreamMsgIdRef.current;
      if (msgId == null) return;
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last && last.id === msgId) {
          const updated = [...prev];
          updated[updated.length - 1] = { ...last, thinking: text };
          return updated;
        }
        return prev;
      });
    },
    streamDone: thinkingStreamDone,
  });

  const getCurrentProvider = () => {
    const emk = getDefaultModel('embeddingModel');
    if (!emk) return null;
    const [pid] = emk.split(':');
    return getProviderById(pid);
  };

  const getCurrentEmbeddingModel = () => {
    const emk = getDefaultModel('embeddingModel');
    if (!emk) return null;
    const [pid, mid] = emk.split(':');
    return getModelById(mid, pid);
  };

  const getCurrentChatModel = () => {
    const chatKey = getDefaultModel('assistantModel');
    if (chatKey) {
      const [pid, mid] = chatKey.split(':');
      return { providerId: pid, modelId: mid };
    }
    return { providerId: apiProvider, modelId: model };
  };

  const getChatCredentials = () => {
    const chatKey = getDefaultModel('assistantModel');
    const { providerId, modelId } = getCurrentChatModel();
    const provider = getProviderById(providerId);
    if (chatKey) {
      return { providerId, modelId, apiKey: provider?.apiKey || '' };
    }
    return { providerId, modelId, apiKey: provider?.apiKey || apiKey };
  };

  const getCurrentRerankModel = () => {
    const rrk = getDefaultModel('rerankModel');
    if (rrk) {
      const [pid, mid] = rrk.split(':');
      return { providerId: pid, modelId: mid };
    }
    return { providerId: 'local', modelId: 'BAAI/bge-reranker-base' };
  };

  const getRerankCredentials = () => {
    const { providerId, modelId } = getCurrentRerankModel();
    const provider = getProviderById(providerId);
    return { providerId, modelId, apiKey: provider?.apiKey || embeddingApiKey || apiKey };
  };

  const getDefaultModelLabel = (key, fallback = '未选择') => {
    if (!key) return fallback;
    const [pid, mid] = key.split(':');
    const p = getProviderById(pid);
    const m = getModelById(mid, pid);
    return `${p?.name || pid} - ${m?.name || mid}`;
  };

  const currentChatModelObj = useMemo(() => {
    const chatKey = getDefaultModel('assistantModel');
    if (!chatKey || !chatKey.includes(':')) return null;
    const [pid, mid] = chatKey.split(':');
    return getModelById(mid, pid);
  }, [getDefaultModel, getModelById]);

  const isVisionCapable = useMemo(() => supportsVision(currentChatModelObj), [currentChatModelObj]);

  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const pdfContainerRef = useRef(null);
  const headerContentRef = useRef(null);
  const abortControllerRef = useRef(null);
  const streamingAbortRef = useRef({ cancelled: false });
  const streamCitationsRef = useRef(null);
  const textareaRef = useRef(null);
  const [headerHeight, setHeaderHeight] = useState(null);

  useEffect(() => {
    fetchAvailableModels();
    fetchAvailableEmbeddingModels();
    fetchStorageInfo();
    loadHistory();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    setSearchQuery('');
    setSearchResults([]);
    setCurrentResultIndex(0);
    setActiveHighlight(null);
    if (docId) {
      const stored = JSON.parse(localStorage.getItem(`search_history_${docId}`) || '[]');
      setSearchHistory(stored);
    } else {
      setSearchHistory([]);
    }
  }, [docId]);

  useEffect(() => {
    if (docId && docInfo) saveCurrentSession();
  }, [docId, docInfo, messages]);

  useEffect(() => {
    localStorage.setItem('apiKey', apiKey);
    localStorage.setItem('apiProvider', apiProvider);
    localStorage.setItem('model', model);
    localStorage.setItem('embeddingApiKey', embeddingApiKey);
    localStorage.setItem('enableVectorSearch', enableVectorSearch);
    localStorage.setItem('enableScreenshot', enableScreenshot);
    localStorage.setItem('streamSpeed', streamSpeed);
    localStorage.setItem('enableBlurReveal', enableBlurReveal);
    localStorage.setItem('blurIntensity', blurIntensity);
    localStorage.setItem('searchEngine', searchEngine);
    localStorage.setItem('searchEngineUrl', searchEngineUrl);
    localStorage.setItem('toolbarSize', toolbarSize);
    localStorage.setItem('toolbarScale', toolbarScale);
    localStorage.setItem('useRerank', useRerank);
    localStorage.setItem('rerankerModel', rerankerModel);
    if (lastCallInfo) localStorage.setItem('lastCallInfo', JSON.stringify(lastCallInfo));
  }, [apiKey, apiProvider, model, embeddingApiKey, enableVectorSearch, enableScreenshot, streamSpeed, enableBlurReveal, blurIntensity, searchEngine, searchEngineUrl, toolbarSize, toolbarScale, useRerank, rerankerModel, lastCallInfo]);

  useEffect(() => {
    if (Object.keys(availableModels).length === 0) return;
    const providerModels = availableModels[apiProvider]?.models;
    if (providerModels && !providerModels[model]) {
      const first = Object.keys(providerModels)[0];
      if (first) setModel(first);
    }
  }, [availableModels, apiProvider]);

  useEffect(() => {
    if (!activeHighlight) return;
    const duration = activeHighlight.source === 'citation' ? 4000 : 2500;
    const timer = setTimeout(() => setActiveHighlight(null), duration);
    return () => clearTimeout(timer);
  }, [activeHighlight]);

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
  }, [docId, docInfo, searchResults.length, useRerank, darkMode]);

  const fetchAvailableModels = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/models`);
      const data = await res.json();
      setAvailableModels(data);
    } catch (e) { console.error(e); }
  };

  const fetchStorageInfo = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/storage_info`);
      if (res.ok) setStorageInfo(await res.json());
    } catch (e) { console.error(e); }
  };

  const fetchAvailableEmbeddingModels = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/embedding_models`);
      if (res.ok) setAvailableEmbeddingModels(await res.json());
    } catch (e) { console.error(e); }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setIsUploading(true);
    setUploadProgress(0);
    setUploadStatus('uploading');
    const formData = new FormData();
    formData.append('file', file);
    const provider = getCurrentProvider();
    const emodel = getCurrentEmbeddingModel();
    if (emodel && provider) {
      const compositeKey = `${provider.id}:${emodel.id}`;
      formData.append('embedding_model', compositeKey);
      if (provider.id !== 'local') {
        if (!provider.apiKey) {
          alert(`请先为 ${provider.name} 配置 API Key`);
          setIsUploading(false);
          return;
        }
        formData.append('embedding_api_key', provider.apiKey);
        formData.append('embedding_api_host', provider.apiHost);
      }
    } else {
      formData.append('embedding_model', 'local:all-MiniLM-L6-v2');
    }
    const ocrSettings = loadOCRSettings();
    formData.append('enable_ocr', ocrSettings.mode || 'auto');
    formData.append('ocr_backend', ocrSettings.backend || 'auto');
    try {
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) setUploadProgress(Math.round((e.loaded / e.total) * 70));
        });
        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            setUploadStatus('processing');
            setUploadProgress(75);
            try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(e); }
          } else { reject(new Error('Upload failed')); }
        });
        xhr.addEventListener('error', () => reject(new Error('Network error')));
        xhr.open('POST', `${API_BASE_URL}/upload`);
        xhr.send(formData);
      });
      setDocId(data.doc_id);
      const dres = await fetch(`${API_BASE_URL}/document/${data.doc_id}?t=${Date.now()}`);
      const ddata = await dres.json();
      const full = { ...ddata, ...data };
      setDocInfo(full);
      let uploadMsg = `✅ 文档《${data.filename}》上传成功！共 ${data.total_pages} 页。`;
      if (data.ocr_used) uploadMsg += `\n🔍 已使用 OCR（${data.ocr_backend || '自动'}）处理部分页面。`;
      setMessages([{ type: 'system', content: uploadMsg }]);
    } catch (error) {
      alert(`上传失败: ${error.message}`);
    } finally {
      setTimeout(() => { setIsUploading(false); setUploadProgress(0); setUploadStatus('uploading'); }, 500);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const setInputValue = (val) => {
    if (textareaRef.current) {
      textareaRef.current.value = val;
      textareaRef.current.style.height = '24px';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
    setHasInput(!!(val && val.trim()));
  };

  const sendMessage = async () => {
    const currentInput = textareaRef.current?.value ?? '';
    if (!currentInput.trim() && screenshots.length === 0) return;
    const { providerId: chatProvider, modelId: chatModel, apiKey: chatApiKey } = getChatCredentials();
    if (!docId) { alert('请先上传文档'); return; }
    if (!chatApiKey && chatProvider !== 'ollama' && chatProvider !== 'local') {
      alert('请先配置API Key\n\n请点击左下角"设置 & API Key"按钮进行配置');
      return;
    }
    const userMsg = { type: 'user', content: currentInput, hasImage: screenshots.length > 0 };
    setMessages(prev => [...prev, userMsg]);
    if (textareaRef.current) {
      textareaRef.current.value = '';
      textareaRef.current.style.height = '24px';
    }
    setHasInput(false);
    setIsLoading(true);
    const chatHistory = buildChatHistory(messages, contextCount);
    const { providerId: _pid } = getChatCredentials();
    const chatProviderFull = getProviderById(_pid);
    const requestBody = {
      doc_id: docId,
      question: userMsg.content,
      api_key: chatApiKey,
      model: chatModel,
      api_provider: chatProvider,
      api_host: chatProviderFull?.apiHost || null,
      selected_text: selectedText || null,
      image_base64_list: screenshots.map(s => s.dataUrl.split(',')[1]),
      image_base64: screenshots[0]?.dataUrl ? screenshots[0].dataUrl.split(',')[1] : null,
      enable_thinking: reasoningEffort !== 'off',
      reasoning_effort: reasoningEffort !== 'off' ? reasoningEffort : null,
      max_tokens: enableMaxTokens ? maxTokens : null,
      temperature: enableTemperature ? temperature : null,
      top_p: enableTopP ? topP : null,
      stream_output: streamOutput,
      enable_vector_search: enableVectorSearch,
      chat_history: chatHistory.length > 0 ? chatHistory : null,
      custom_params: customParams.length > 0 ? Object.fromEntries(customParams.filter(p => p.name).map(p => [p.name, p.value])) : null,
      enable_memory: enableMemory,
    };
    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();
    streamingAbortRef.current.cancelled = false;
    streamCitationsRef.current = null;
    const tempMsgId = Date.now();
    setStreamingMessageId(tempMsgId);
    setMessages(prev => [...prev, { id: tempMsgId, type: 'assistant', content: '', model: chatModel, isStreaming: true, thinking: '', thinkingMs: 0 }]);
    try {
      if (streamSpeed !== 'off' && streamOutput) {
        activeStreamMsgIdRef.current = tempMsgId;
        setContentStreamDone(false);
        setThinkingStreamDone(false);
        contentStream.reset('');
        thinkingStream.reset('');
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current.signal
        });
        if (!response.ok) {
          let ed = `HTTP ${response.status}`;
          try { const eb = await response.json(); ed = eb.detail || eb.error?.message || eb.message || JSON.stringify(eb); } catch(e){}
          throw new Error(ed);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let currentText = '';
        let currentThinking = '';
        let thinkingStartTime = null;
        let thinkingEndTime = null;
        let sseBuffer = '';
        let sseDone = false;
        const findSseSeparator = (buf) => {
          const lf = buf.indexOf('\n\n');
          const crlf = buf.indexOf('\r\n\r\n');
          if (lf === -1 && crlf === -1) return { index: -1, length: 0 };
          if (lf === -1) return { index: crlf, length: 4 };
          if (crlf === -1) return { index: lf, length: 2 };
          return lf < crlf ? { index: lf, length: 2 } : { index: crlf, length: 4 };
        };
        const processSseEvent = (et) => {
          const lines = et.split(/\r?\n/);
          const dl = [];
          for (const ln of lines) { if (ln.trim().startsWith('data:')) dl.push(ln.trim().slice(5).trimStart()); }
          if (dl.length === 0) return;
          const data = dl.join('\n');
          if (data === '[DONE]') { sseDone = true; return; }
          try {
            const p = JSON.parse(data);
            if (p.error) { const em = `❌ ${p.error}`; currentText = em; contentStream.addChunk(em); sseDone = true; return; }
            if (p.type === 'retrieval_progress') return;
            const delta = p.choices?.[0]?.delta || {};
            const cc = delta.content || p.content || '';
            const ct = delta.reasoning_content || p.reasoning_content || '';
            if (!p.done && !p.choices?.[0]?.finish_reason) {
              if (cc) { currentText += cc; contentStream.addChunk(cc); if (thinkingStartTime && !thinkingEndTime) thinkingEndTime = Date.now(); }
              if (ct) { if (!thinkingStartTime) thinkingStartTime = Date.now(); currentThinking += ct; thinkingStream.addChunk(ct); }
            } else {
              if (p.retrieval_meta?.citations) streamCitationsRef.current = p.retrieval_meta.citations;
              if (ct) { currentThinking += ct; thinkingStream.addChunk(ct); }
              sseDone = true;
            }
          } catch (e) { console.error(e, data); }
        };
        while (true) {
          const { value, done } = await reader.read();
          if (done || streamingAbortRef.current.cancelled) break;
          sseBuffer += decoder.decode(value, { stream: true });
          while (true) {
            const { index: si, length: sl } = findSseSeparator(sseBuffer);
            if (si === -1) break;
            const re = sseBuffer.slice(0, si);
            sseBuffer = sseBuffer.slice(si + sl);
            if (re.trim()) processSseEvent(re.trim());
            if (sseDone) break;
          }
          if (sseDone) break;
        }
        if (!sseDone && sseBuffer.trim()) processSseEvent(sseBuffer.trim());
        setContentStreamDone(true);
        setThinkingStreamDone(true);
        const finalThinkingMs = thinkingStartTime ? (thinkingEndTime || Date.now()) - thinkingStartTime : 0;
        const finalContent = currentText || (currentThinking ? '' : '⚠️ AI未返回内容');
        setMessages(prev => prev.map(m => m.id === tempMsgId ? { ...m, content: finalContent, thinking: currentThinking, isStreaming: false, thinkingMs: finalThinkingMs, citations: streamCitationsRef.current } : m));
        activeStreamMsgIdRef.current = null;
        setStreamingMessageId(null);
      } else {
        const response = await fetch(`${API_BASE_URL}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestBody), signal: abortControllerRef.current.signal });
        if (!response.ok) {
          let ed = `HTTP ${response.status}`;
          try { const eb = await response.json(); ed = eb.detail || eb.error?.message || eb.message || JSON.stringify(eb); } catch(e){}
          throw new Error(ed);
        }
        const data = await response.json();
        setLastCallInfo({ provider: data.used_provider, model: data.used_model, fallback: data.fallback_used });
        setMessages(prev => prev.map(m => m.id === tempMsgId ? { ...m, content: data.answer, thinking: data.reasoning_content || '', isStreaming: false, citations: data.retrieval_meta?.citations } : m));
        setStreamingMessageId(null);
      }
    } catch (error) {
      if (error.name === 'AbortError') return;
      setContentStreamDone(true); setThinkingStreamDone(true);
      activeStreamMsgIdRef.current = null; setStreamingMessageId(null);
      setMessages(prev => prev.map(m => m.id === tempMsgId ? { ...m, content: '❌ ' + error.message, isStreaming: false } : m));
    } finally { setIsLoading(false); }
  };

  const handleStop = () => {
    if (abortControllerRef.current) { abortControllerRef.current.abort(); abortControllerRef.current = null; setIsLoading(false); }
    streamingAbortRef.current.cancelled = true;
    contentStream.reset(''); thinkingStream.reset('');
    setContentStreamDone(false); setThinkingStreamDone(false);
    activeStreamMsgIdRef.current = null;
    if (streamingMessageId) setMessages(prev => prev.map(m => m.id === streamingMessageId ? { ...m, isStreaming: false } : m));
    setStreamingMessageId(null);
  };

  const copyMessage = (content, messageId) => {
    navigator.clipboard.writeText(content).then(() => { setCopiedMessageId(messageId); setTimeout(() => setCopiedMessageId(null), 2000); });
  };

  const regenerateMessage = async (index) => {
    if (!docId) { alert('请先上传文档'); return; }
    const userMsg = messages.slice(0, index).reverse().find(m => m.type === 'user');
    if (!userMsg) return;
    setMessages(prev => prev.slice(0, index));
    setInputValue(userMsg.content);
    setTimeout(() => sendMessage(), 100);
  };

  const saveToMemory = async (index, type) => {
    const m = messages[index];
    if (!m || m.type !== 'assistant') return;
    const um = messages.slice(0, index).reverse().find(x => x.type === 'user');
    const content = `Q: ${um ? um.content.slice(0, 100) : ''}\nA: ${m.content.slice(0, 200)}`;
    try {
      const res = await fetch('/api/memory/entries', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content, source_type: type, doc_id: docId }) });
      if (res.ok) { if (type === 'liked') setLikedMessages(p => new Set(p).add(index)); else setRememberedMessages(p => new Set(p).add(index)); }
    } catch (e) {}
  };

  const startNewChat = () => {
    setDocId(null); setDocInfo(null); setMessages([]); setCurrentPage(1); setSelectedText(''); setScreenshots([]);
  };

  const handleAreaSelected = async (rect) => {
    const container = pdfContainerRef.current;
    if (!container) { setIsSelectingArea(false); return; }
    const cr = container.getBoundingClientRect();
    const clamped = clampSelectionToPage(rect, cr.width, cr.height);
    try {
      const res = await captureArea(pdfContainerRef, clamped);
      if (res) {
        setScreenshots(prev => {
          if (prev.length >= 9) { alert('最多只能截图 9 张'); return prev; }
          return [...prev, { id: `${Date.now()}-${Math.random().toString(16).slice(2)}`, dataUrl: res }];
        });
      } else { alert('截图生成失败'); }
    } catch (e) { alert('截图生成失败'); }
    finally { setIsSelectingArea(false); }
  };

  const handleSelectionCancel = () => setIsSelectingArea(false);

  const handleScreenshotAction = async (key, sid = null) => {
    const action = SCREENSHOT_ACTIONS[key];
    if (!action) return;
    const target = sid ? screenshots.find(s => s.id === sid) : screenshots[screenshots.length - 1];
    if (!target) return;
    if (key === 'copy') {
      try {
        const res = await fetch(target.dataUrl);
        const blob = await res.blob();
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      } catch (e) { alert('复制失败'); }
      return;
    }
    if (key === 'ask') { setTimeout(() => textareaRef.current?.focus(), 100); return; }
    if (action.autoSend && action.prompt) { setInputValue(action.prompt); requestAnimationFrame(() => sendMessage()); }
  };

  const handleScreenshotClose = (id = null) => {
    if (id) setScreenshots(prev => prev.filter(s => s.id !== id));
    else setScreenshots([]);
  };

  useEffect(() => {
    if (screenshots.length > 0 && !isVisionCapable) setScreenshots([]);
  }, [isVisionCapable, screenshots.length]);

  const loadSession = async (s) => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/document/${s.docId}?t=${Date.now()}`);
      if (res.ok) { setDocId(s.docId); setDocInfo(await res.json()); setMessages(s.messages || []); setCurrentPage(1); }
    } catch (e) {}
    finally { setIsLoading(false); }
  };

  const deleteSession = (sid) => {
    if (!window.confirm('确定要删除这个对话吗？')) return;
    const h = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    const next = h.filter(x => x.id !== sid);
    localStorage.setItem('chatHistory', JSON.stringify(next));
    setHistory(next);
    if (sid === docId) { setDocId(null); setDocInfo(null); setMessages([]); }
  };

  const saveCurrentSession = () => {
    if (!docId) return;
    const h = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    const idx = h.findIndex(x => x.id === docId);
    const data = { id: docId, docId, filename: docInfo.filename, messages, updatedAt: Date.now(), createdAt: idx >= 0 ? h[idx].createdAt : Date.now() };
    if (idx >= 0) h[idx] = data; else h.unshift(data);
    const lim = h.slice(0, 50);
    localStorage.setItem('chatHistory', JSON.stringify(lim));
    setHistory(lim);
  };

  const loadHistory = () => {
    const s = localStorage.getItem('chatHistory');
    if (s) setHistory(JSON.parse(s));
  };

  const handleSearch = async (cq) => {
    if (!docId) { alert('请先上传文档'); return; }
    const q = (cq ?? searchQuery).trim();
    if (!q) { setSearchResults([]); setCurrentResultIndex(0); setActiveHighlight(null); return; }
    setIsSearching(true); setSearchQuery(q);
    const { providerId: rp, modelId: rm, apiKey: rk } = getRerankCredentials();
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), 45000);
    try {
      const res = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: ctrl.signal,
        body: JSON.stringify({ doc_id: docId, query: q, api_key: embeddingApiKey || apiKey, top_k: 5, candidate_k: 20, use_rerank: useRerank, reranker_model: useRerank ? (rm || rerankerModel) : undefined, rerank_provider: useRerank ? rp : undefined, rerank_api_key: useRerank ? rk : undefined })
      });
      if (res.ok) {
        const data = await res.json();
        const results = Array.isArray(data.results) ? data.results : [];
        setSearchResults(results);
        if (results.length) { focusResult(0, results); if (!searchHistory.includes(q)) setSearchHistory(p => [q, ...p.filter(x => x !== q)].slice(0, 8)); }
        else { alert('未找到结果'); }
      }
    } catch (e) {}
    finally { clearTimeout(tid); setIsSearching(false); }
  };

  const focusResult = (idx, res = searchResults) => {
    if (!res.length) return;
    const i = ((idx % res.length) + res.length) % res.length;
    const t = res[i];
    const p = Math.max(1, Math.min(t.page || 1, docInfo?.total_pages || 1));
    setCurrentResultIndex(i); setCurrentPage(p);
    setActiveHighlight({ page: p, text: t.chunk || '', at: Date.now() });
  };

  const formatSimilarity = (r) => {
    if (r?.similarity_percent !== undefined) return r.similarity_percent;
    const s = typeof r?.score === 'number' ? r.score : 0;
    return Math.round((1 / (1 + Math.max(s, 0))) * 10000) / 100;
  };

  const renderHighlightedSnippet = (snip, hls = []) => {
    if (!snip) return '...';
    if (!hls.length) return snip;
    const ord = [...hls].sort((a, b) => a.start - b.start);
    const parts = []; let cur = 0;
    ord.forEach((h, i) => {
      const s = Math.max(0, Math.min(snip.length, h.start || 0));
      const e = Math.max(s, Math.min(snip.length, h.end || 0));
      if (s > cur) parts.push(snip.slice(cur, s));
      parts.push(<mark key={i} className="bg-yellow-200 px-0.5 rounded">{snip.slice(s, e)}</mark>);
      cur = e;
    });
    if (cur < snip.length) parts.push(snip.slice(cur));
    return parts;
  };

  const handleCitationClick = useCallback((c) => {
    if (!c?.page_range) return;
    const tp = c.page_range[0];
    if (typeof tp === 'number' && tp > 0) {
      setActiveHighlight(null); setCurrentPage(tp);
      if (c.highlight_text) setTimeout(() => setActiveHighlight({ page: tp, text: c.highlight_text, source: 'citation' }), 400);
    }
  }, []);

  const handlePresetSelect = (query) => {
    setInputValue(query);
    requestAnimationFrame(() => sendMessage());
  };

  const showPresetQuestions = docId && messages.filter(
    msg => msg.type === 'user' || msg.type === 'assistant'
  ).length === 0;

  return (
    <div className={`h-screen w-full flex overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-[#0f1115] text-gray-200' : 'bg-transparent text-[var(--color-text-main)]'}`}>
      <motion.div
        animate={{ width: showSidebar ? 288 : 0, opacity: showSidebar ? 1 : 0 }}
        className={`flex-shrink-0 m-6 mr-0 h-[calc(100vh-3rem)] flex flex-col z-20 overflow-hidden rounded-[var(--radius-panel-lg)] ${darkMode ? 'bg-[#1a1d21]/90 border-white/5 backdrop-blur-3xl' : 'bg-white/80 border-white/50 backdrop-blur-3xl border shadow-xl'}`}
      >
        <div className="w-72 mx-auto flex flex-col h-full items-stretch relative">
          <button onClick={() => setShowSidebar(false)} className="absolute top-3 right-3 p-2 rounded-full hover:bg-black/5 text-gray-400"><ChevronLeft className="w-4 h-4" /></button>
          <div className="px-6 py-8 flex items-center justify-between">
            <div className="flex items-center gap-3 font-bold text-2xl text-blue-600 tracking-tight">
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
              <button onClick={() => setDarkMode(!darkMode)} className={`p-2 rounded-full hover:bg-black/5 ${darkMode ? 'text-yellow-400' : 'text-gray-400'}`}>{darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}</button>
            </div>
          </div>
          <div className="px-5 mb-4 flex justify-center">
            <button onClick={() => { startNewChat(); fileInputRef.current?.click(); }} className="tanya-btn w-full shadow-lg"><Plus className="w-5 h-5" /><span>上传/新对话</span></button>
            <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
          </div>
          <div className="flex-1 overflow-y-auto px-5 space-y-2">
            <div className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 px-2">History</div>
            {history.map((item, idx) => (
              <div key={idx} onClick={() => loadSession(item)} className={`p-3 rounded-xl cursor-pointer group flex items-center gap-3 transition-all ${item.id === docId ? 'bg-blue-500/10 text-blue-600 ring-1 ring-blue-500/20 shadow-sm' : 'hover:bg-white/40'}`}>
                <MessageSquare className="w-5 h-5" /><div className="flex-1 truncate text-sm font-medium">{item.filename}</div>
                <button onClick={(e) => { e.stopPropagation(); deleteSession(item.id); }} className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500"><Trash2 className="w-4 h-4" /></button>
              </div>
            ))}
          </div>
          <div className="p-4 border-t border-white/20">
            <button onClick={() => { setShowSettings(true); fetchStorageInfo(); }} className="flex items-center gap-3 w-full p-3 rounded-xl hover:bg-white/50 text-sm font-medium transition-colors"><Settings className="w-5 h-5" /><span>设置 & API Key</span></button>
          </div>
        </div>
      </motion.div>

      <div className="flex-1 flex flex-col h-full relative transition-all duration-200">
        <motion.header 
          layout 
          animate={{ height: isHeaderExpanded ? 'auto' : 0, opacity: isHeaderExpanded ? 1 : 0, marginBottom: isHeaderExpanded ? 16 : 0, marginTop: isHeaderExpanded ? 24 : 0 }}
          className="px-8 soft-panel mx-8 mt-6 mb-4 sticky top-4 z-10 rounded-[var(--radius-panel-lg)] overflow-hidden"
        >
          <div ref={headerContentRef} className="flex items-center justify-between w-full py-3">
            <div className="flex items-center gap-4">
              <button onClick={() => setShowSidebar(!showSidebar)} className="p-2 hover:bg-black/5 rounded-lg transition-colors"><Menu className="w-6 h-6" /></button>
              <div className="flex items-center gap-4">
                <div className="bg-blue-600 text-white p-2.5 rounded-xl shadow-sm"><FileText className="w-6 h-6" /></div>
                <div><h1 className="text-2xl font-bold tracking-tight">ChatPDF Pro</h1><p className="text-xs text-gray-500 font-medium">智能文档助手</p></div>
              </div>
            </div>
            {docId && (
              <div className="flex-1 max-w-2xl mx-4 flex items-center gap-2">
                <div className="relative flex-1">
                  <input type="search" placeholder="搜索文档内容..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSearch()} className="w-full px-4 py-2 pl-11 rounded-full soft-input text-sm focus:ring-2 focus:ring-blue-400 transition-all" />
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                </div>
                <button onClick={() => handleSearch()} className="px-4 py-2 rounded-full bg-blue-600 text-white text-sm font-medium flex items-center gap-2 hover:bg-blue-700 transition-all shadow-sm">{isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}<span>搜索</span></button>
                <button onClick={() => setUseRerank(!useRerank)} className={`px-3 py-2 rounded-full border text-sm font-medium transition-colors ${useRerank ? 'bg-purple-50 text-purple-700 border-purple-200' : 'bg-white text-gray-600 border-gray-200'}`} title="使用重排提高质量"><Wand2 className="w-4 h-4" /><span>重排</span></button>
              </div>
            )}
            <div className="flex items-center gap-4">
              {docInfo && <div className="font-medium text-sm glass-panel px-4 py-1 rounded-full truncate max-w-[200px]">{docInfo.filename}</div>}
              <button onClick={() => setIsHeaderExpanded(false)} className="p-2 hover:bg-black/5 rounded-full text-gray-500 transition-colors"><ChevronUp className="w-5 h-5" /></button>
            </div>
          </div>
        </motion.header>

        <AnimatePresence>
          {!isHeaderExpanded && !showSidebar && (
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="absolute top-4 left-2 z-20 flex flex-col gap-1.5">
              <button onClick={() => setIsHeaderExpanded(true)} className="p-2 backdrop-blur-md shadow-sm rounded-full border bg-white/80 text-gray-700 hover:scale-105 transition-all"><ChevronDown className="w-4 h-4" /></button>
              <button onClick={() => setShowSidebar(true)} className="p-2 backdrop-blur-md shadow-sm rounded-full border bg-white/80 text-gray-700 hover:scale-105 transition-all"><Menu className="w-4 h-4" /></button>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="flex-1 flex overflow-hidden px-8 pb-8 gap-4 pt-2">
          {docId ? (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`soft-panel overflow-hidden flex flex-col relative flex-shrink-0 rounded-[var(--radius-panel)] min-w-0 ${darkMode ? 'bg-gray-800/50' : ''}`} style={{ width: `${pdfPanelWidth}%` }}>
              <PDFViewer ref={pdfContainerRef} pdfUrl={docInfo?.pdf_url || (docId ? `/uploads/${docId}.pdf` : undefined)} page={currentPage} onPageChange={setCurrentPage} highlightInfo={activeHighlight} isSelecting={isSelectingArea} onAreaSelected={handleAreaSelected} onSelectionCancel={handleSelectionCancel} darkMode={darkMode} />
            </motion.div>
          ) : (
            <div className="flex-1 flex items-center justify-center relative">
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20"><div className="w-96 h-96 bg-blue-400 rounded-full blur-3xl animate-pulse" /></div>
              <div className="text-center space-y-6 relative z-10">
                <div className="w-24 h-24 bg-white/50 backdrop-blur-md rounded-[32px] flex items-center justify-center mx-auto shadow-lg border border-white/60"><Upload className="w-10 h-10 text-blue-500" /></div>
                <div className="space-y-2"><h2 className="text-3xl font-bold tracking-tight">Upload a PDF to Start</h2><p className="text-gray-500 text-lg">智能文档助手，开启AI对话之旅</p></div>
              </div>
            </div>
          )}

          <div className="w-4 cursor-col-resize relative group flex justify-center" onMouseDown={(e) => {
            const startX = e.clientX; const startW = pdfPanelWidth;
            const move = (em) => setPdfPanelWidth(Math.max(30, Math.min(70, startW + ((em.clientX - startX) / window.innerWidth) * 100)));
            const up = () => { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); };
            document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
          }}><div className="w-1 h-full rounded-full bg-transparent group-hover:bg-blue-500/50 transition-colors" /></div>

          <div className="soft-panel flex flex-col overflow-hidden rounded-[var(--radius-panel)] flex-1 min-w-0">
            <div className="flex-1 overflow-y-auto p-6 space-y-6" ref={chatPaneRef}>
              {(searchResults.length > 0 || isSearching) && (
                <div className="rounded-3xl border border-black/5 bg-white/70 backdrop-blur-sm p-4 space-y-3 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2"><Search className="w-4 h-4 text-blue-500" /><span className="font-semibold text-sm">文档搜索</span>{useRerank && <span className="text-xs text-purple-700 bg-purple-50 px-2 py-0.5 rounded-full border">已开启重排</span>}</div>
                    {searchResults.length > 0 && <span className="text-xs text-gray-500">找到 {searchResults.length} 个候选</span>}
                  </div>
                  <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                    {searchResults.map((result, idx) => (
                      <button key={idx} onClick={() => focusResult(idx)} className="w-full text-left p-3 rounded-2xl border border-gray-100 hover:border-blue-200 hover:bg-blue-50/40 transition-all">
                        <div className="flex justify-between text-[10px] text-gray-500 mb-1"><span>第 {result.page || 1} 页 · #{idx + 1}</span><span className="font-semibold text-blue-600">匹配度 {formatSimilarity(result)}%</span></div>
                        <div className="text-sm text-gray-800 leading-relaxed line-clamp-3">{renderHighlightedSnippet(result.snippet || result.chunk || '', result.highlights || [])}</div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {showPresetQuestions && <PresetQuestions onSelect={handlePresetSelect} disabled={isLoading} />}

              {messages.map((msg, idx) => (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} key={idx} className={`flex flex-col ${msg.type === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`${msg.type === 'user' ? 'max-w-[85%] rounded-2xl px-4 py-3 message-bubble-user rounded-tr-sm text-sm' : 'w-full text-gray-800 dark:text-gray-50'}`}>
                    {msg.thinking && <ThinkingBlock content={msg.thinking} isStreaming={msg.isStreaming && streamingMessageId === msg.id} darkMode={darkMode} thinkingMs={msg.thinkingMs || 0} />}
                    {msg.hasImage && <div className="mb-2 rounded-lg overflow-hidden border border-black/5 bg-black/5 p-2 flex items-center gap-2 text-[10px]"><ImageIcon className="w-3 h-3" /> Images attached</div>}
                    {msg.type === 'assistant' && (
                      <div className="flex items-center gap-2 mb-2 select-none">
                        <div className="p-1 rounded-lg bg-blue-600 text-white shadow-sm"><Bot className="w-4 h-4" /></div>
                        <span className="font-bold text-sm">AI Assistant</span>
                        {msg.model && <span className="text-[10px] text-gray-400 border rounded px-1.5 py-0.5">{msg.model}</span>}
                      </div>
                    )}
                    <StreamingMarkdown content={msg.content} isStreaming={msg.isStreaming} enableBlurReveal={enableBlurReveal} blurIntensity={blurIntensity} citations={msg.citations} onCitationClick={handleCitationClick} />
                  </div>
                  {msg.type === 'assistant' && !msg.isStreaming && (
                    <div className="flex items-center gap-1 mt-1 ml-2">
                      <button onClick={() => copyMessage(msg.content, idx)} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors" title="复制"><Copy className="w-4 h-4" /></button>
                      <button onClick={() => regenerateMessage(idx)} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors" title="重新生成"><History className="w-4 h-4" /></button>
                      <button onClick={() => saveToMemory(idx, 'liked')} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${likedMessages.has(idx) ? 'text-pink-500' : 'text-gray-500'}`} title="点赞"><MessageSquare className={`w-4 h-4 ${likedMessages.has(idx) ? 'fill-current' : ''}`} /></button>
                      <button onClick={() => saveToMemory(idx, 'manual')} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${rememberedMessages.has(idx) ? 'text-purple-500' : 'text-gray-500'}`} title="记忆"><Brain className={`w-4 h-4 ${rememberedMessages.has(idx) ? 'fill-current' : ''}`} /></button>
                    </div>
                  )}
                </motion.div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <div className="p-6 pt-0 bg-transparent">
              <ScreenshotPreview screenshots={screenshots} onAction={handleScreenshotAction} onClose={handleScreenshotClose} />
              <div className="relative bg-white/80 backdrop-blur-[20px] rounded-[36px] shadow-xl p-1.5 flex items-end gap-2 border border-white/50 ring-1 ring-black/5">
                <div className="flex-1 flex flex-col min-h-[48px] justify-center pl-6 py-1.5">
                  <textarea ref={textareaRef} onChange={(e) => { e.target.style.height = '24px'; e.target.style.height = e.target.scrollHeight + 'px'; setHasInput(!!e.target.value.trim()); }} onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())} placeholder="Ask anything, summarize, rephrase..." className="w-full bg-transparent border-none outline-none text-gray-800 placeholder:text-gray-400 font-medium resize-none h-[24px] max-h-[120px] focus:ring-0" rows={1} />
                  <div className="flex items-center gap-4 text-gray-400 mt-2">
                    <ModelQuickSwitch onThinkingChange={setEnableThinking} />
                    <button className="hover:text-gray-600 transition-colors p-1 rounded-md hover:bg-gray-50"><SlidersHorizontal className="w-5 h-5" /></button>
                    <button onClick={() => fileInputRef.current?.click()} className="hover:text-gray-600 transition-colors p-1 rounded-md hover:bg-gray-50"><Paperclip className="w-5 h-5" /></button>
                    {isVisionCapable && (
                      <button onClick={() => setIsSelectingArea(true)} disabled={!docId || screenshots.length >= 9} className={`transition-colors p-1 rounded-md ${docId && screenshots.length < 9 ? (isSelectingArea ? 'text-purple-600 bg-purple-50' : 'hover:text-gray-600') : 'text-gray-200 cursor-not-allowed'}`} title="截图"><Scan className="w-5 h-5" /></button>
                    )}
                  </div>
                </div>
                <motion.button onClick={isLoading ? handleStop : sendMessage} disabled={!isLoading && !hasInput && screenshots.length === 0} className="glass-btn-3d w-12 h-12 flex items-center justify-center bg-blue-600 text-white rounded-full shadow-lg" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  {isLoading ? <PauseIcon /> : <SendIcon />}
                </motion.button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Progress Modal (Rings Animation) */}
      <AnimatePresence>
        {isUploading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90">
            <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }} className="flex flex-col items-center">
              <div style={{ position: 'relative', width: 300, height: 300 }}>
                <div style={{ position: 'absolute', inset: 0, filter: 'blur(0.5px) contrast(1.2)' }}>
                  {UPLOAD_RING_CONFIGS.map((cfg, i) => (
                    <div key={i} style={{ position: 'absolute', top: '50%', left: '50%', width: cfg.s, height: cfg.s, borderRadius: cfg.br, border: `${cfg.w}px solid ${cfg.c}`, mixBlendMode: cfg.mix, animation: `chatpdf-spin ${cfg.dur}s linear ${cfg.del}s infinite ${cfg.dir}` }} />
                  ))}
                </div>
                <div className="absolute inset-0 flex flex-col items-center justify-center z-10 text-white">
                  <span className="text-5xl font-light tracking-tighter">{uploadProgress}%</span>
                  <span className="text-[10px] uppercase tracking-[0.3em] mt-2 opacity-50">{uploadStatus}</span>
                </div>
              </div>
              <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-8 text-white/60 font-light tracking-wide">{uploadStatus === 'uploading' ? '正在上传文档...' : 'AI 正在构建知识库索引'}</motion.p>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Settings Modal */}
      <AnimatePresence initial={false}>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
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
                {/* 模型服务管理入口（对话/嵌入/重排统一管理） */}
                <div className="relative overflow-hidden rounded-[32px] border border-blue-100/50 bg-gradient-to-br from-white/40 to-blue-50/10 p-1 shadow-sm transition-all hover:shadow-md backdrop-blur-md">
                  <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-blue-500/10 rounded-full blur-3xl"></div>
                  <div className="absolute bottom-0 left-0 -mb-4 -ml-4 w-20 h-20 bg-purple-500/10 rounded-full blur-2xl"></div>

                  <div className="relative bg-white/30 backdrop-blur-sm rounded-[28px] p-5 border border-white/50">
                    <div className="flex flex-col gap-5">
                      <div className="flex items-center justify-between gap-4">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 rounded-[20px] bg-gradient-to-br from-blue-500/90 to-indigo-600/90 shadow-lg shadow-blue-500/20 flex items-center justify-center text-white shrink-0 backdrop-blur-sm">
                            <Server className="w-6 h-6" />
                          </div>
                          <div className="space-y-0.5">
                            <h3 className="text-lg font-bold text-gray-900 tracking-tight">模型服务</h3>
                            <p className="text-xs text-gray-500 font-medium">统一管理 Chat / Embedding / Rerank</p>
                          </div>
                        </div>

                        <button
                          onClick={() => setShowEmbeddingSettings(true)}
                          className="group relative overflow-hidden rounded-[18px] bg-gray-900/90 px-5 py-2.5 text-white shadow-lg transition-all hover:bg-gray-800 hover:shadow-xl hover:-translate-y-0.5 active:scale-95 shrink-0 backdrop-blur-sm"
                        >
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out" />
                          <div className="relative flex items-center gap-2 font-medium text-sm">
                            <span>管理模型</span>
                            <Settings className="w-4 h-4 transition-transform duration-500 group-hover:rotate-180" />
                          </div>
                        </button>
                      </div>

                      <div className="flex flex-col gap-3">
                        <div className="group relative overflow-hidden rounded-[18px] border border-gray-100/50 bg-white/40 p-4 transition-all hover:border-blue-200 hover:bg-white/60 hover:shadow-sm backdrop-blur-sm cursor-pointer">
                          <div className="flex items-center gap-4">
                            <MessageCircle className="w-5 h-5 text-gray-400 group-hover:text-blue-500 transition-colors" />
                            <div className="flex-1 min-w-0">
                              <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-0.5">Chat Model</div>
                              <div className="font-semibold text-gray-800 text-sm truncate" title={getDefaultModelLabel(getDefaultModel('assistantModel'))}>
                                {getDefaultModelLabel(getDefaultModel('assistantModel')) || '未设置'}
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="group relative overflow-hidden rounded-[18px] border border-gray-100/50 bg-white/40 p-4 transition-all hover:border-purple-200 hover:bg-white/60 hover:shadow-sm backdrop-blur-sm cursor-pointer">
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

                        <div className="group relative overflow-hidden rounded-[18px] border border-gray-100/50 bg-white/40 p-4 transition-all hover:border-amber-200 hover:bg-white/60 hover:shadow-sm backdrop-blur-sm cursor-pointer">
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
                    <input type="checkbox" checked={enableVectorSearch} onChange={e => setEnableVectorSearch(e.target.checked)} className="accent-blue-600 w-5 h-5" />
                  </label>
                  <label className="flex items-center justify-between cursor-pointer p-2 hover:bg-gray-50 rounded-lg">
                    <span className="font-medium">Screenshot Analysis</span>
                    <input type="checkbox" checked={enableScreenshot} onChange={e => setEnableScreenshot(e.target.checked)} className="accent-blue-600 w-5 h-5" />
                  </label>
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
                    <p className="text-xs text-gray-500 mt-1">调整AI回复的打字机效果速度（已优化为按字符流式）</p>
                  </div>

                  <label className="flex items-center justify-between cursor-pointer p-2 hover:bg-gray-50 rounded-lg mt-3">
                    <span className="font-medium">Blur Reveal 效果</span>
                    <input type="checkbox" checked={enableBlurReveal} onChange={e => setEnableBlurReveal(e.target.checked)} className="accent-blue-600 w-5 h-5" />
                  </label>
                  <p className="text-xs text-gray-500 ml-2 mb-2">流式输出时每个新字符从模糊到清晰的渐变效果（逐字符效果）</p>

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
                      <p className="text-xs text-gray-500 mt-1">调整每个新字符出现时的模糊程度和动画时长</p>
                    </div>
                  )}
                </div>

                <div className="pt-4 border-t border-gray-100">
                  <button
                    onClick={() => {
                      setShowSettings(false);
                      setShowGlobalSettings(true);
                    }}
                    className="soft-card w-full px-4 py-3 rounded-xl font-medium hover:scale-105 transition-transform flex items-center justify-center gap-2"
                  >
                    <Type className="w-4 h-4" />
                    全局设置（字体、缩放）
                  </button>
                </div>

                <div className="pt-4 border-t border-gray-100">
                  <button
                    onClick={() => {
                      setShowSettings(false);
                      setShowChatSettings(true);
                    }}
                    className="soft-card w-full px-4 py-3 rounded-xl font-medium hover:scale-105 transition-transform flex items-center justify-center gap-2"
                  >
                    <SlidersHorizontal className="w-4 h-4" />
                    对话设置（温度、Token、流式）
                  </button>
                </div>

                <div className="pt-4 border-t border-gray-100">
                  <button
                    onClick={() => {
                      setShowSettings(false);
                      setShowOCRSettings(true);
                    }}
                    className="soft-card w-full px-4 py-3 rounded-xl font-medium hover:scale-105 transition-transform flex items-center justify-center gap-2"
                  >
                    <ScanText className="w-4 h-4" />
                    OCR 设置（文字识别）
                  </button>
                </div>

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
                        <input
                          type="text"
                          value={searchEngineUrl}
                          onChange={(e) => setSearchEngineUrl(e.target.value)}
                          className="w-full p-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                          placeholder="例如：https://www.google.com/search?q={query}"
                        />
                        <p className="text-xs text-gray-500">
                          使用 <code className="font-mono">{'{query}'}</code> 作为搜索词占位符（若不填将自动追加 <code className="font-mono">?q=</code>）。
                        </p>
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
                    <p className="text-xs text-gray-500 mt-1">影响划词工具栏按钮尺寸与间距</p>
                  </div>
                </div>

                <div className="pt-4 border-t border-gray-100">
                  <h3 className="text-sm font-semibold text-gray-800 mb-3">文件存储位置</h3>
                  {storageInfo ? (
                    <div className="space-y-2">
                      <div className="bg-gray-50 p-3 rounded-lg">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-medium text-gray-600">PDF文件</span>
                          <span className="text-xs text-gray-500">{storageInfo.pdf_count ?? '-'} 个文件</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <code className="flex-1 text-xs bg-white px-2 py-1 rounded border border-gray-200 overflow-x-auto whitespace-nowrap">
                            {storageInfo.uploads_dir}
                          </code>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(storageInfo.uploads_dir);
                              alert('路径已复制到剪贴板！');
                            }}
                            className="p-1.5 hover:bg-blue-100 text-blue-600 rounded transition-colors"
                            title="复制路径"
                          >
                            <Copy className="w-4 h-4" />
                          </button>
                        </div>
                      </div>

                      <div className="bg-gray-50 p-3 rounded-lg">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-medium text-gray-600">对话历史</span>
                          <span className="text-xs text-gray-500">{storageInfo.doc_count ?? '-'} 个文档</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <code className="flex-1 text-xs bg-white px-2 py-1 rounded border border-gray-200 overflow-x-auto whitespace-nowrap">
                            {storageInfo.data_dir}
                          </code>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(storageInfo.data_dir);
                              alert('路径已复制到剪贴板！');
                            }}
                            className="p-1.5 hover:bg-blue-100 text-blue-600 rounded transition-colors"
                            title="复制路径"
                          >
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
                <button
                  onClick={() => setShowSettings(false)}
                  className="w-full py-3 soft-button soft-button-primary rounded-xl font-medium hover:shadow-lg transition-all"
                >
                  Save Changes
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <Suspense fallback={null}>
        <EmbeddingSettings
          isOpen={showEmbeddingSettings}
          onClose={() => setShowEmbeddingSettings(false)}
        />
      </Suspense>

      <Suspense fallback={null}>
        <GlobalSettings
          isOpen={showGlobalSettings}
          onClose={() => { setShowGlobalSettings(false); setShowSettings(true); }}
        />
      </Suspense>

      <Suspense fallback={null}>
        <ChatSettings
          isOpen={showChatSettings}
          onClose={() => { setShowChatSettings(false); setShowSettings(true); }}
        />
      </Suspense>

      <Suspense fallback={null}>
        <OCRSettingsPanel
          isOpen={showOCRSettings}
          onClose={() => { setShowOCRSettings(false); setShowSettings(true); }}
        />
      </Suspense>
    </div>
  );
};

const CustomSelect = ({ value, onChange, options }) => {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const click = (e) => { if (ref.current && !ref.current.contains(e.target)) setIsOpen(false); };
    document.addEventListener('mousedown', click); return () => document.removeEventListener('mousedown', click);
  }, []);
  const sel = options.find(o => o.value === value)?.label || value;
  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setIsOpen(!isOpen)} className="w-full p-3 rounded-2xl border border-gray-200 bg-white/50 flex items-center justify-between hover:border-blue-300 transition-colors">
        <span className="text-sm font-medium text-gray-700">{sel}</span><ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} className="absolute bottom-full left-0 right-0 mb-2 z-50 bg-white border rounded-2xl shadow-xl overflow-hidden">
            {options.map(o => (
              <button key={o.value} onClick={() => { onChange(o.value); setIsOpen(false); }} className={`w-full text-left px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors ${o.value === value ? 'text-blue-600 bg-blue-50 font-bold' : 'text-gray-700'}`}>{o.label}</button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ChatPDF;
