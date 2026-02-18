import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Upload, Send, FileText, Settings, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Copy, Bot, X, Crop, Image as ImageIcon, History, Moon, Sun, Plus, MessageSquare, Trash2, Menu, Type, ChevronUp, ChevronDown, Search, Loader2, Wand2, Server, Database, ListFilter, ArrowUpRight, SlidersHorizontal, Paperclip, ScanText, Scan, Brain } from 'lucide-react';
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
import EmbeddingSettings from './EmbeddingSettings';
import OCRSettingsPanel, { loadOCRSettings } from './OCRSettingsPanel';
import GlobalSettings from './GlobalSettings';
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';
import ChatSettings from './ChatSettings';
import PresetQuestions from './PresetQuestions';
import ModelQuickSwitch from './ModelQuickSwitch';
import ThinkingBlock from './ThinkingBlock';
import { useSmoothStream } from '../hooks/useSmoothStream';

// API base URL – empty string so that Vite proxy forwards to backend
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

/**
 * 构建对话历史
 * 根据 contextCount 截取最近 N 轮对话，过滤系统消息和含图片的消息
 * @param {Array} messages - 消息列表
 * @param {number} contextCount - 上下文轮数
 * @returns {Array} 对话历史 [{role: 'user'|'assistant', content: '...'}]
 */
const buildChatHistory = (messages, contextCount) => {
  if (!contextCount || contextCount <= 0) return [];

  // 过滤出有效的对话消息（排除 system 和含图片的消息）
  const validMessages = messages.filter(msg =>
    (msg.type === 'user' || msg.type === 'assistant') && !msg.hasImage
  );

  // 取最近 contextCount * 2 条（每轮包含 user + assistant）
  const recentMessages = validMessages.slice(-(contextCount * 2));

  return recentMessages.map(msg => ({
    role: msg.type === 'user' ? 'user' : 'assistant',
    content: msg.content
  }));
};

const ChatPDF = () => {
  // Core State
  const [docId, setDocId] = useState(null);
  const [docInfo, setDocInfo] = useState(null);
  const [pdfPanelWidth, setPdfPanelWidth] = useState(50); // Percentage
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('uploading'); // 'uploading' | 'processing'

  // UI State
  const [showSettings, setShowSettings] = useState(false);
  const [showEmbeddingSettings, setShowEmbeddingSettings] = useState(false);
  const [showOCRSettings, setShowOCRSettings] = useState(false);
  const [showGlobalSettings, setShowGlobalSettings] = useState(false);
  const [showChatSettings, setShowChatSettings] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [isHeaderExpanded, setIsHeaderExpanded] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [history, setHistory] = useState([]); // Mock history for now

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
  const [lastCallInfo, setLastCallInfo] = useState(null); // {provider, model, fallback}
  const [searchHistory, setSearchHistory] = useState([]);

  // Screenshot State
  const [screenshot, setScreenshot] = useState(null);
  const [isSelectingArea, setIsSelectingArea] = useState(false);
  const [selectionBox, setSelectionBox] = useState(null);

  // Settings State
  const [apiKey, setApiKey] = useState(localStorage.getItem('apiKey') || '');
  const [apiProvider, setApiProvider] = useState(localStorage.getItem('apiProvider') || 'openai');
  const [model, setModel] = useState(localStorage.getItem('model') || 'gpt-4o');
  const [availableModels, setAvailableModels] = useState({});
  const [embeddingApiKey, setEmbeddingApiKey] = useState(localStorage.getItem('embeddingApiKey') || '');
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState({});
  const [enableVectorSearch, setEnableVectorSearch] = useState(localStorage.getItem('enableVectorSearch') === 'true');
  const [enableScreenshot, setEnableScreenshot] = useState(localStorage.getItem('enableScreenshot') !== 'false');
  const [streamSpeed, setStreamSpeed] = useState(localStorage.getItem('streamSpeed') || 'normal'); // fast, normal, slow, off
  const [streamingMessageId, setStreamingMessageId] = useState(null);
  const [storageInfo, setStorageInfo] = useState(null);
  const [enableBlurReveal, setEnableBlurReveal] = useState(localStorage.getItem('enableBlurReveal') !== 'false');
  const [blurIntensity, setBlurIntensity] = useState(localStorage.getItem('blurIntensity') || 'medium'); // strong, medium, light
  const [searchEngine, setSearchEngine] = useState(localStorage.getItem('searchEngine') || 'google'); // google, baidu, bing, sogou
  const [searchEngineUrl, setSearchEngineUrl] = useState(
    localStorage.getItem('searchEngineUrl') || 'https://www.google.com/search?q={query}'
  );
  const [toolbarSize, setToolbarSize] = useState(localStorage.getItem('toolbarSize') || 'normal'); // compact, normal, large
  const [toolbarScale, setToolbarScale] = useState(parseFloat(localStorage.getItem('toolbarScale') || '1'));
  const [toolbarPosition, setToolbarPosition] = useState({ x: 0, y: 0 });

  const [copiedMessageId, setCopiedMessageId] = useState(null);
  // 记忆交互状态：记录已点赞和已记住的消息索引
  const [likedMessages, setLikedMessages] = useState(new Set());
  const [rememberedMessages, setRememberedMessages] = useState(new Set());
  // 深度思考模式开关（由 ModelQuickSwitch 控制，向后兼容）
  const [enableThinking, setEnableThinking] = useState(false);

  // 流式缓冲渲染状态
  const [contentStreamDone, setContentStreamDone] = useState(false);
  const [thinkingStreamDone, setThinkingStreamDone] = useState(false);
  // 记录当前流式消息 ID
  const activeStreamMsgIdRef = useRef(null);

  // New three-layer context
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

  // 正文内容缓冲流
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

  // 思考内容缓冲流
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

  // Helper functions to get current provider and model
  const getCurrentProvider = () => {
    const embeddingModelKey = getDefaultModel('embeddingModel');
    if (!embeddingModelKey) return null;
    const [providerId] = embeddingModelKey.split(':');
    return getProviderById(providerId);
  };

  const getCurrentEmbeddingModel = () => {
    const embeddingModelKey = getDefaultModel('embeddingModel');
    if (!embeddingModelKey) return null;
    const [providerId, modelId] = embeddingModelKey.split(':');
    const model = getModelById(modelId, providerId);
    return model;
  };

  const getCurrentChatModel = () => {
    const chatKey = getDefaultModel('assistantModel');
    if (chatKey) {
      const [providerId, modelId] = chatKey.split(':');
      return { providerId, modelId };
    }
    // fallback：使用旧状态
    return { providerId: apiProvider, modelId: model };
  };

  const getChatCredentials = () => {
    const chatKey = getDefaultModel('assistantModel');
    const { providerId, modelId } = getCurrentChatModel();
    const provider = getProviderById(providerId);

    // 新架构路径：DefaultsContext 有 assistantModel 值时，优先使用 Provider 的 apiKey
    if (chatKey) {
      return {
        providerId,
        modelId,
        apiKey: provider?.apiKey || '',
      };
    }

    // 旧架构回退路径：使用全局 apiKey
    return {
      providerId,
      modelId,
      apiKey: provider?.apiKey || apiKey,
    };
  };

  const getCurrentRerankModel = () => {
    const rerankKey = getDefaultModel('rerankModel');
    if (rerankKey) {
      const [providerId, modelId] = rerankKey.split(':');
      return { providerId, modelId };
    }
    return { providerId: 'local', modelId: 'BAAI/bge-reranker-base' };
  };

  const getRerankCredentials = () => {
    const { providerId, modelId } = getCurrentRerankModel();
    const provider = getProviderById(providerId);
    return {
      providerId,
      modelId,
      apiKey: provider?.apiKey || embeddingApiKey || apiKey,
    };
  };

  const getDefaultModelLabel = (key, fallback = '未选择') => {
    if (!key) return fallback;
    const [providerId, modelId] = key.split(':');
    const provider = getProviderById(providerId);
    const modelObj = getModelById(modelId, providerId);
    return `${provider?.name || providerId} - ${modelObj?.name || modelId}`;
  };

  // ========== 当前聊天模型对象（含 tags），用于 supportsVision 判断 ==========
  const currentChatModelObj = useMemo(() => {
    const chatKey = getDefaultModel('assistantModel');
    if (!chatKey || !chatKey.includes(':')) return null;
    const [providerId, modelId] = chatKey.split(':');
    return getModelById(modelId, providerId);
  }, [getDefaultModel, getModelById]);

  // 当前模型是否支持视觉能力
  const isVisionCapable = useMemo(() => supportsVision(currentChatModelObj), [currentChatModelObj]);

  // Refs
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const pdfContainerRef = useRef(null);
  const selectionStartRef = useRef(null);
  const headerContentRef = useRef(null);
  const abortControllerRef = useRef(null);
  const streamingAbortRef = useRef({ cancelled: false });
  // 用于在 SSE 流式响应中暂存 retrieval_meta.citations
  const streamCitationsRef = useRef(null);

  const [headerHeight, setHeaderHeight] = useState(null);
  const API_BASE_URL = ''; // Relative path due to proxy

  // Constants
  const VISION_MODELS = {
    'openai': ['gpt-5.1-2025-11-13', 'gpt-4.1', 'gpt-5-nano', 'o4-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini'],
    'anthropic': ['claude-sonnet-4-5-20250929', 'claude-opus-4-1-20250805', 'claude-haiku-4-5-20250219', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
    'gemini': ['gemini-2.5-pro', 'gemini-2.5-flash-preview-09-2025', 'gemini-2.5-flash-lite-preview-09-2025', 'gemini-2.0-flash', 'gemini-pro-vision'],
    'grok': ['grok-4.1', 'grok-4.1-fast', 'grok-3', 'grok-vision-beta'],
    'doubao': ['doubao-1.5-pro-256k', 'doubao-1.5-pro-32k'],
    'qwen': ['qwen-max-2025-01-25', 'qwen3-235b-a22b-instruct-2507', 'qwen3-coder-plus-2025-09-23'],
    'minimax': ['abab6.5-chat', 'abab6.5s-chat', 'minimax-m2'],
    'ollama': ['llava']
  };

  // Effects
  useEffect(() => {
    fetchAvailableModels();
    fetchAvailableEmbeddingModels();
    fetchStorageInfo();
    loadHistory();  // 加载历史记录
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Reset search state when document changes
  useEffect(() => {
    setSearchQuery('');
    setSearchResults([]);
    setCurrentResultIndex(0);
    setActiveHighlight(null);

    if (docId) {
      const storedHistory = JSON.parse(localStorage.getItem(`search_history_${docId}`) || '[]');
      setSearchHistory(storedHistory);
    } else {
      setSearchHistory([]);
    }
  }, [docId]);

  // 保存当前会话到历史记录
  useEffect(() => {
    if (docId && docInfo) {
      saveCurrentSession();
    }
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
    if (lastCallInfo) {
      localStorage.setItem('lastCallInfo', JSON.stringify(lastCallInfo));
    }
  }, [
    apiKey,
    apiProvider,
    model,
    embeddingApiKey,
    enableVectorSearch,
    enableScreenshot,
    streamSpeed,
    enableBlurReveal,
    blurIntensity,
    searchEngine,
    searchEngineUrl,
    toolbarSize,
    toolbarScale,
    useRerank,
    rerankerModel,
    lastCallInfo
  ]);

  // Validate model when availableModels loads or provider changes
  useEffect(() => {
    if (Object.keys(availableModels).length === 0) return; // Wait for models to load

    const providerModels = availableModels[apiProvider]?.models;
    if (!providerModels) return;

    // Check if current model is valid for current provider
    if (!providerModels[model]) {
      // Model is invalid, select first available model for this provider
      const firstModel = Object.keys(providerModels)[0];
      if (firstModel) {
        console.log(`Model ${model} invalid for provider ${apiProvider}, switching to ${firstModel}`);
        setModel(firstModel);
      }
    }
  }, [availableModels, apiProvider]);

  // Auto-hide highlight overlay after a short delay
  // 高亮自动消失：引文高亮 4 秒，搜索高亮 2.5 秒
  useEffect(() => {
    if (!activeHighlight) return;
    const duration = activeHighlight.source === 'citation' ? 4000 : 2500;
    const timer = setTimeout(() => setActiveHighlight(null), duration);
    return () => clearTimeout(timer);
  }, [activeHighlight]);

  // Smooth header height animation by measuring real height instead of animating 'auto'
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

  // API Functions
  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/models`);
      const data = await response.json();
      setAvailableModels(data);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const fetchStorageInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/storage_info`);
      if (!response.ok) throw new Error('Failed to fetch storage info');
      const data = await response.json();
      setStorageInfo(data);
    } catch (error) {
      console.error('Failed to fetch storage info:', error);
    }
  };

  const fetchAvailableEmbeddingModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/embedding_models`);
      if (!response.ok) throw new Error('Failed to fetch');
      const data = await response.json();
      setAvailableEmbeddingModels(data);
    } catch (error) {
      console.error('Failed to fetch embedding models:', error);
      // Fallback to complete default list if API fails
      setAvailableEmbeddingModels({
        "local-minilm": {
          "name": "Local: MiniLM-L6 (Default)",
          "provider": "local",
          "model_name": "all-MiniLM-L6-v2",
          "dimension": 384,
          "max_tokens": 256,
          "price": "Free (Local)",
          "description": "Fast, general purpose"
        },
        "local-multilingual": {
          "name": "Local: Multilingual",
          "provider": "local",
          "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
          "dimension": 384,
          "max_tokens": 128,
          "price": "Free (Local)",
          "description": "Better for Chinese/multilingual"
        },
        "text-embedding-3-large": {
          "name": "OpenAI: text-embedding-3-large",
          "provider": "openai",
          "base_url": "https://api.openai.com/v1",
          "dimension": 3072,
          "max_tokens": 8191,
          "price": "$0.13/M tokens",
          "description": "Best overall quality"
        },
        "text-embedding-3-small": {
          "name": "OpenAI: text-embedding-3-small",
          "provider": "openai",
          "base_url": "https://api.openai.com/v1",
          "dimension": 1536,
          "max_tokens": 8191,
          "price": "$0.02/M tokens",
          "description": "Best value"
        },
        "text-embedding-v3": {
          "name": "Alibaba: text-embedding-v3",
          "provider": "openai",
          "base_url": "https://dashscope.aliyuncs.com/api/v1",
          "dimension": 1024,
          "max_tokens": 8192,
          "price": "$0.007/M tokens",
          "description": "Chinese optimized, cheapest"
        },
        "moonshot-embedding-v1": {
          "name": "Moonshot: moonshot-embedding-v1",
          "provider": "openai",
          "base_url": "https://api.moonshot.cn/v1",
          "dimension": 1024,
          "max_tokens": 8192,
          "price": "$0.011/M tokens",
          "description": "Kimi, OpenAI compatible"
        },
        "deepseek-embedding-v1": {
          "name": "DeepSeek: deepseek-embedding-v1",
          "provider": "openai",
          "base_url": "https://api.deepseek.com/v1",
          "dimension": 1024,
          "max_tokens": 8192,
          "price": "$0.01/M tokens",
          "description": "Low cost OpenAI compatible"
        },
        "glm-embedding-2": {
          "name": "Zhipu: glm-embedding-2",
          "provider": "openai",
          "base_url": "https://open.bigmodel.cn/api/paas/v4",
          "dimension": 1024,
          "max_tokens": 8192,
          "price": "$0.014/M tokens",
          "description": "GLM series"
        },
        "minimax-embedding-v2": {
          "name": "MiniMax: minimax-embedding-v2",
          "provider": "openai",
          "base_url": "https://api.minimaxi.chat/v1",
          "dimension": 1024,
          "max_tokens": 8192,
          "price": "$0.014/M tokens",
          "description": "ABAB series"
        },
        "BAAI/bge-m3": {
          "name": "SiliconFlow: BAAI/bge-m3",
          "provider": "openai",
          "base_url": "https://api.siliconflow.cn/v1",
          "dimension": 1024,
          "max_tokens": 8192,
          "price": "$0.02/M tokens",
          "description": "Open source, hosted"
        }
      });
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);
    setUploadStatus('uploading');

    const formData = new FormData();
    formData.append('file', file);

    // 使用新的三层架构获取 embedding 模型和 Provider 信息
    const provider = getCurrentProvider();
    const model = getCurrentEmbeddingModel();

    if (model && provider) {
      // 传递 composite key（provider.id:model.id），后端 Resolver 可正确解析
      const compositeKey = `${provider.id}:${model.id}`;
      formData.append('embedding_model', compositeKey);
      console.log('🔵 Using embedding model (composite key):', compositeKey);

      if (provider.id !== 'local') {
        // 非本地 Provider 需要验证 API Key 是否已配置
        if (!provider.apiKey) {
          alert(`请先为 ${provider.name} 配置 API Key`);
          setIsUploading(false);
          return;
        }
        formData.append('embedding_api_key', provider.apiKey);
        formData.append('embedding_api_host', provider.apiHost);
      }
    } else {
      // 回退到本地默认模型
      formData.append('embedding_model', 'local:all-MiniLM-L6-v2');
    }

    // 从 localStorage 读取 OCR 模式设置，默认为 "auto"
    const ocrSettings = loadOCRSettings()
    formData.append('enable_ocr', ocrSettings.mode || 'auto')
    // 传递用户选择的 OCR 后端引擎（如 mistral、tesseract 等）
    formData.append('ocr_backend', ocrSettings.backend || 'auto')

    try {
      console.log('🔵 Uploading file:', file.name);

      // Use XMLHttpRequest for progress tracking
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 70);
            setUploadProgress(percent);
          }
        });

        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            setUploadStatus('processing');
            setUploadProgress(75);
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch (e) {
              reject(new Error('Invalid response'));
            }
          } else {
            reject(new Error('Upload failed'));
          }
        });

        xhr.addEventListener('error', () => reject(new Error('Network error')));
        xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')));

        xhr.open('POST', `${API_BASE_URL}/upload`);
        xhr.send(formData);
      });

      setUploadProgress(80);
      console.log('🟢 Upload response:', data);
      setDocId(data.doc_id);

      setUploadProgress(85);
      console.log('🔵 Fetching document details for:', data.doc_id);
      const docResponse = await fetch(`${API_BASE_URL}/document/${data.doc_id}?t=${new Date().getTime()}`);
      const docData = await docResponse.json();

      setUploadProgress(95);
      const fullDocData = { ...docData, ...data };
      console.log('🟢 Document data received:', fullDocData);

      if (fullDocData.pdf_url) {
        console.log('✅ PDF URL found:', fullDocData.pdf_url);
      } else {
        console.warn('⚠️ No PDF URL found in document data');
      }

      setUploadProgress(100);
      setDocInfo(fullDocData);

      // 构建上传成功消息，包含 OCR 处理结果摘要
      let uploadMsg = `✅ 文档《${data.filename}》上传成功！共 ${data.total_pages} 页。`
      if (data.ocr_used) {
        uploadMsg += `\n🔍 已使用 OCR（${data.ocr_backend || '自动'}）处理部分页面。`
      }
      if (data.ocr_warning) {
        uploadMsg += `\n⚠️ ${data.ocr_warning}`
      }

      setMessages([{
        type: 'system',
        content: uploadMsg
      }]);

    } catch (error) {
      console.error('❌ Upload error:', error);
      const errorMsg = error.message || '未知错误';
      alert(`上传失败: ${errorMsg}\n\n可能原因：\n1. 后端服务未启动\n2. 网络连接问题\n3. PDF文件格式不支持\n\n请检查浏览器控制台查看详细错误信息`);
    } finally {
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
        setUploadStatus('uploading');
      }, 500);

      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() && !screenshot) return;
    
    // 使用新的凭证系统进行验证
    const { providerId: chatProvider, modelId: chatModel, apiKey: chatApiKey } = getChatCredentials();
    
    if (!docId) {
      alert('请先上传文档');
      return;
    }
    
    if (!chatApiKey && chatProvider !== 'ollama' && chatProvider !== 'local') {
      alert('请先配置API Key\n\n请点击左下角"设置 & API Key"按钮进行配置');
      return;
    }

    const userMsg = {
      type: 'user',
      content: inputMessage,
      hasImage: !!screenshot
    };

    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsLoading(true);

    // 构建对话历史（使用当前 messages 状态，不包含刚添加的 userMsg，因为 setMessages 是异步的）
    const chatHistory = buildChatHistory(messages, contextCount);

    const requestBody = {
      doc_id: docId,
      question: userMsg.content,
      api_key: chatApiKey,
      model: chatModel,
      api_provider: chatProvider,
      selected_text: selectedText || null,
      image_base64: screenshot ? screenshot.split(',')[1] : null,
      // 用 reasoningEffort 替代原有的 enable_thinking boolean
      enable_thinking: reasoningEffort !== 'off',
      reasoning_effort: reasoningEffort !== 'off' ? reasoningEffort : null,
      // 根据开关状态决定是否包含参数
      max_tokens: enableMaxTokens ? maxTokens : null,
      temperature: enableTemperature ? temperature : null,
      top_p: enableTopP ? topP : null,
      stream_output: streamOutput,
      chat_history: chatHistory.length > 0 ? chatHistory : null,
      // 自定义参数转换为 dict
      custom_params: customParams.length > 0 ? Object.fromEntries(
        customParams.filter(p => p.name).map(p => [p.name, p.value])
      ) : null,
      // 记忆功能开关
      enable_memory: enableMemory,
    };

    // Add placeholder message for streaming effect
    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();
    streamingAbortRef.current.cancelled = false;
    // 重置流式引文暂存
    streamCitationsRef.current = null;

    const tempMsgId = Date.now();
    setStreamingMessageId(tempMsgId);
    setMessages(prev => [...prev, {
      id: tempMsgId,
      type: 'assistant',
      content: '',
      model: chatModel,
      isStreaming: true,
      thinking: '',
      thinkingMs: 0
    }]);

    try {
      // 使用 SSE 流式传输（截图也支持流式，后端已处理多模态消息）
      if (streamSpeed !== 'off' && streamOutput) {
        // 重置流式缓冲状态
        setContentStreamDone(false);
        setThinkingStreamDone(false);
        contentStream.reset('');
        thinkingStream.reset('');
        activeStreamMsgIdRef.current = tempMsgId;
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current.signal
        });

        if (!response.ok) throw new Error('Failed to get response');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let currentText = '';
        let currentThinking = '';
        // 思考计时：记录首次收到 reasoning_content 的时间
        let thinkingStartTime = null;
        let thinkingEndTime = null;

        // SSE 解析缓冲区：按 "\n\n" 分隔 event，避免 JSON 跨 chunk 时丢字
        let sseBuffer = '';

        const processSseEvent = (eventText) => {
          // eventText 为不含末尾空行的单个 event
          const lines = eventText.split(/\r?\n/);
          const dataLines = [];
          for (const ln of lines) {
            if (ln.startsWith('data:')) {
              dataLines.push(ln.slice(5).trimStart());
            }
          }
          if (dataLines.length === 0) return;

          // SSE 允许一个 event 多行 data:，拼接时用 \n
          const data = dataLines.join('\n');
          if (data === '[DONE]') {
            // 交给外层 while 终止（通过抛出特殊标记）
            throw new Error('__SSE_DONE__');
          }

          const parsed = JSON.parse(data);
          if (parsed.error) {
            throw new Error(parsed.error);
          }

          // 后端可能会插入检索进度事件（非 content/done 结构），这里忽略
          if (parsed.type === 'retrieval_progress') {
            return;
          }

          const chunkContent = parsed.content || '';
          const chunkThinking = parsed.reasoning_content || '';

          if (!parsed.done) {
            if (chunkContent) {
              currentText += chunkContent;
              contentStream.addChunk(chunkContent);
              if (thinkingStartTime && !thinkingEndTime) {
                thinkingEndTime = Date.now();
              }
            }
            if (chunkThinking) {
              if (!thinkingStartTime) thinkingStartTime = Date.now();
              currentThinking += chunkThinking;
              thinkingStream.addChunk(chunkThinking);
            }
          } else {
            if (parsed.retrieval_meta && parsed.retrieval_meta.citations) {
              streamCitationsRef.current = parsed.retrieval_meta.citations;
            }
            if (chunkThinking) {
              currentThinking += chunkThinking;
              thinkingStream.addChunk(chunkThinking);
            }
          }
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (streamingAbortRef.current.cancelled) break;

          sseBuffer += decoder.decode(value, { stream: true });

          // 按 event 边界切分（空行分隔）
          let sepIdx;
          while ((sepIdx = sseBuffer.indexOf('\n\n')) !== -1) {
            const rawEvent = sseBuffer.slice(0, sepIdx);
            sseBuffer = sseBuffer.slice(sepIdx + 2);

            const trimmed = rawEvent.trim();
            if (!trimmed) continue;

            try {
              processSseEvent(trimmed);
            } catch (e) {
              if (e && e.message === '__SSE_DONE__') {
                // 结束整个读取循环
                sseBuffer = '';
                sepIdx = -1;
                break;
              }
              // JSON 解析失败等：不吞掉，直接抛出让外层 catch 处理
              throw e;
            }
          }

          // 如果收到 DONE 标记跳出
          if (sseBuffer === '' && streamingAbortRef.current.cancelled) break;
        }

        // 标记流式传输完成，让 useSmoothStream 一次性渲染剩余字符
        setContentStreamDone(true);
        setThinkingStreamDone(true);

        // 使用 currentText 和 currentThinking 作为最终值确保完整性
        const streamCitations = streamCitationsRef.current;
        const finalThinkingMs = thinkingStartTime ? (thinkingEndTime || Date.now()) - thinkingStartTime : 0;
        setMessages(prev => prev.map(msg =>
          msg.id === tempMsgId
            ? { ...msg, content: currentText, thinking: currentThinking || '', isStreaming: false, thinkingMs: finalThinkingMs, citations: streamCitations || null }
            : msg
        ));
        streamCitationsRef.current = null;
        activeStreamMsgIdRef.current = null;
        setStreamingMessageId(null);
      } else {
        // 非流式回退：统一使用 /chat 端点（后端已支持 image_base64 多模态）
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: abortControllerRef.current.signal,
          body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error('Failed to get response');

        const data = await response.json();
        const fullAnswer = data.answer;
        const reasoningContent = data.reasoning_content || '';
        // 从非流式响应中提取引文数据
        const responseCitations = data.retrieval_meta?.citations || null;
        setLastCallInfo({
          provider: data.used_provider,
          model: data.used_model,
          fallback: data.fallback_used
        });

        setMessages(prev => prev.map(msg =>
          msg.id === tempMsgId
            ? { ...msg, thinking: reasoningContent || '', citations: responseCitations }
            : msg
        ));

        // OPTIMIZED SPEED PROFILES for Performance
        const runClientStream = async (answerText) => {
          if (streamSpeed === 'off') {
            setMessages(prev => prev.map(msg =>
              msg.id === tempMsgId
                ? { ...msg, content: answerText, isStreaming: false }
                : msg
            ));
            setStreamingMessageId(null);
            return true;
          }

          // Increased baseDelay and minChunk to reduce render frequency
          const speedProfiles = {
            fast: { minChunk: 5, maxChunk: 12, maxUpdates: 30, baseDelay: 20, variation: 10 },
            normal: { minChunk: 3, maxChunk: 8, maxUpdates: 40, baseDelay: 35, variation: 15 },
            slow: { minChunk: 2, maxChunk: 5, maxUpdates: 60, baseDelay: 60, variation: 30 }
          };

          const profile = speedProfiles[streamSpeed] || speedProfiles.normal;
          const approxChunk = Math.ceil(answerText.length / profile.maxUpdates) || 1;
          const chunkSize = Math.max(profile.minChunk, Math.min(profile.maxChunk, approxChunk));

          let pointer = 0;
          while (pointer < answerText.length) {
            if (streamingAbortRef.current.cancelled) return false;
            pointer = Math.min(pointer + chunkSize, answerText.length);
            const nextContent = answerText.slice(0, pointer);

            setMessages(prev => prev.map(msg =>
              msg.id === tempMsgId
                ? { ...msg, content: nextContent }
                : msg
            ));

            const delay = profile.baseDelay + Math.random() * profile.variation;
            await new Promise(resolve => setTimeout(resolve, delay));
          }

          return true;
        };

        const completed = await runClientStream(fullAnswer);
        if (completed) {
          setMessages(prev => prev.map(msg =>
            msg.id === tempMsgId
              ? { ...msg, content: fullAnswer, isStreaming: false }
              : msg
          ));
          setStreamingMessageId(null);
        } else {
          setMessages(prev => prev.map(msg =>
            msg.id === tempMsgId
              ? { ...msg, isStreaming: false }
              : msg
          ));
        }
      }

      setScreenshot(null); // Clear screenshot after sending
    } catch (error) {
      if (error.name === 'AbortError') {
        return;
      }
      setMessages(prev => [...prev, {
        type: 'error',
        content: '❌ Error: ' + error.message
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
    }
    streamingAbortRef.current.cancelled = true;
    // 重置流式缓冲
    contentStream.reset('');
    thinkingStream.reset('');
    setContentStreamDone(false);
    setThinkingStreamDone(false);
    activeStreamMsgIdRef.current = null;
    if (streamingMessageId) {
      setMessages(prev => {
        const next = [...prev];
        const idx = next.findIndex(msg => msg.id === streamingMessageId);
        if (idx === -1) return prev;
        next[idx] = { ...next[idx], isStreaming: false };
        return next;
      });
    }
    setStreamingMessageId(null);
  };

  // Message Action Functions
  const copyMessage = (content, messageId) => {
    navigator.clipboard.writeText(content).then(() => {
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    }).catch(err => {
      console.error('Failed to copy:', err);
    });
  };

  const regenerateMessage = async (messageIndex) => {
    // 使用新的凭证系统进行验证
    const { providerId: chatProvider, modelId: chatModel, apiKey: chatApiKey } = getChatCredentials();
    
    if (!docId) {
      alert('请先上传文档');
      return;
    }
    
    if (!chatApiKey && chatProvider !== 'ollama' && chatProvider !== 'local') {
      alert('请先配置API Key\n\n请点击左下角"设置 & API Key"按钮进行配置');
      return;
    }

    // Find the last user message before this assistant message
    const userMessage = messages.slice(0, messageIndex).reverse().find(msg => msg.type === 'user');
    if (!userMessage) return;

    // Remove all messages after the user message
    setMessages(prev => prev.slice(0, messageIndex));

    // Resend the user message
    setInputMessage(userMessage.content);
    // Trigger send in next tick
    setTimeout(() => {
      sendMessage();
    }, 100);
  };

  // 保存消息到记忆系统
  const saveToMemory = async (messageIndex, sourceType) => {
    const msg = messages[messageIndex];
    if (!msg || msg.type !== 'assistant') return;

    // 找到对应的用户消息
    const userMsg = messages.slice(0, messageIndex).reverse().find(m => m.type === 'user');
    const question = userMsg ? userMsg.content.slice(0, 100) : '';
    const answer = msg.content.slice(0, 200);
    const content = `Q: ${question}\nA: ${answer}`;

    try {
      const res = await fetch('/api/memory/entries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content,
          source_type: sourceType,
          doc_id: docId,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      // 更新状态以显示视觉反馈
      if (sourceType === 'liked') {
        setLikedMessages(prev => new Set(prev).add(messageIndex));
      } else {
        setRememberedMessages(prev => new Set(prev).add(messageIndex));
      }
    } catch (err) {
      console.error('保存记忆失败:', err);
    }
  };

  // Helper Functions
  const loadDocumentContent = async (id) => {
    try {
      const response = await fetch(`${API_BASE_URL}/document/${id}`);
      if (!response.ok) throw new Error('Failed to load document');
      const data = await response.json();
      console.log('Loaded document data:', data); // Debug
      setDocInfo(data);
    } catch (error) {
      console.error('Failed to load document:', error);
    }
  };

  const formatSimilarity = (result) => {
    if (result?.similarity_percent !== undefined) {
      return result.similarity_percent;
    }
    const rawScore = typeof result?.score === 'number' ? result.score : 0;
    return Math.round((1 / (1 + Math.max(rawScore, 0))) * 10000) / 100;
  };

  const renderHighlightedSnippet = (snippet, highlights = []) => {
    if (!snippet) return '...';
    if (!highlights.length) return snippet;

    const ordered = [...highlights].sort((a, b) => a.start - b.start);
    const parts = [];
    let cursor = 0;

    ordered.forEach((h, idx) => {
      const start = Math.max(0, Math.min(snippet.length, h.start || 0));
      const end = Math.max(start, Math.min(snippet.length, h.end || 0));

      if (start > cursor) {
        parts.push(snippet.slice(cursor, start));
      }

      parts.push(
        <mark key={`hl-${idx}`} className="bg-yellow-200 px-0.5 rounded">
          {snippet.slice(start, end)}
        </mark>
      );
      cursor = end;
    });

    if (cursor < snippet.length) {
      parts.push(snippet.slice(cursor));
    }
    return parts;
  };

  const persistSearchHistory = (query) => {
    if (!docId || !query) return;
    setSearchHistory((prev) => {
      const next = [query, ...prev.filter(item => item !== query)].slice(0, 8);
      localStorage.setItem(`search_history_${docId}`, JSON.stringify(next));
      return next;
    });
  };

  const clearSearchHistory = () => {
    if (!docId) return;
    localStorage.removeItem(`search_history_${docId}`);
    setSearchHistory([]);
  };

  // Document semantic search
  const focusResult = (index, results = searchResults) => {
    if (!results.length) {
      setCurrentResultIndex(0);
      setActiveHighlight(null);
      return;
    }

    const total = results.length;
    const normalizedIndex = ((index % total) + total) % total;
    const target = results[normalizedIndex];
    const totalPages = docInfo?.total_pages || docInfo?.data?.total_pages || target?.page || 1;
    const nextPage = Math.max(1, Math.min(target?.page || 1, totalPages));

    setCurrentResultIndex(normalizedIndex);
    setCurrentPage(nextPage);
    setActiveHighlight({
      page: nextPage,
      text: target?.chunk || '',
      at: Date.now()
    });
  };

  const handleSearch = async (customQuery) => {
    if (!docId) {
      alert('请先上传文档后再搜索');
      return;
    }

    const query = (customQuery ?? searchQuery).trim();
    if (!query) {
      setSearchResults([]);
      setCurrentResultIndex(0);
      setActiveHighlight(null);
      return;
    }

    setIsSearching(true);
    setSearchQuery(query);

    const { providerId: rerankProvider, modelId: rerankModelId, apiKey: rerankApiKey } = getRerankCredentials();

    // 搜索超时控制：45 秒后自动取消请求
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);

    try {
      const response = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          doc_id: docId,
          query,
          api_key: embeddingApiKey || apiKey || undefined,
          top_k: 5,
          candidate_k: 20,
          use_rerank: useRerank,
          reranker_model: useRerank ? (rerankModelId || rerankerModel) : undefined,
          rerank_provider: useRerank ? rerankProvider : undefined,
          rerank_api_key: useRerank ? rerankApiKey : undefined
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || '搜索失败');
      }

      const data = await response.json();
      const results = Array.isArray(data.results) ? data.results : [];
      setSearchResults(results);
      setLastCallInfo({
        provider: data.used_provider,
        model: data.used_model,
        fallback: data.fallback_used
      });

      if (results.length) {
        focusResult(0, results);
        persistSearchHistory(query);
      } else {
        setCurrentResultIndex(0);
        setActiveHighlight(null);
        alert('未找到匹配结果');
      }
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        console.error('搜索请求超时（45秒）');
        alert('搜索超时，请稍后重试。如果使用了重排序功能，可以尝试关闭后再搜索。');
      } else {
        console.error('Failed to search document:', error);
        alert(`搜索失败：${error.message}`);
      }
      setSearchResults([]);
      setActiveHighlight(null);
    } finally {
      clearTimeout(timeoutId);
      setIsSearching(false);
    }
  };

  const goToNextResult = () => {
    if (!searchResults.length) return;
    focusResult(currentResultIndex + 1);
  };

  const goToPrevResult = () => {
    if (!searchResults.length) return;
    focusResult(currentResultIndex - 1);
  };

  const handleTextSelection = () => {
    const selection = window.getSelection();
    const text = selection.toString().trim();
    if (text) {
      setSelectedText(text);
      setShowTextMenu(true);
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      const nextPos = { x: rect.left + rect.width / 2, y: rect.top - 10 };
      setMenuPosition(nextPos);
      setToolbarPosition(nextPos);
    } else {
      setShowTextMenu(false);
    }
  };

  // ==================== 划词工具箱功能 ====================

  // 1. 复制选中文本
  const handleCopy = () => {
    navigator.clipboard.writeText(selectedText).then(() => {
      alert('✅ 已复制到剪贴板');
    }).catch(err => {
      console.error('复制失败:', err);
    });
  };

  // 2. 高亮标注（保存到 localStorage）
  const handleHighlight = () => {
    const highlights = JSON.parse(localStorage.getItem(`highlights_${docId}`) || '[]');
    const newHighlight = {
      id: Date.now(),
      text: selectedText,
      page: currentPage,
      color: 'yellow',
      createdAt: new Date().toISOString()
    };
    highlights.push(newHighlight);
    localStorage.setItem(`highlights_${docId}`, JSON.stringify(highlights));
    alert('✅ 已添加高亮标注');
    setShowTextMenu(false);
  };

  // 3. 添加笔记
  const handleAddNote = () => {
    const note = prompt('请输入您的笔记：', '');
    if (note) {
      const notes = JSON.parse(localStorage.getItem(`notes_${docId}`) || '[]');
      const newNote = {
        id: Date.now(),
        text: selectedText,
        note: note,
        page: currentPage,
        createdAt: new Date().toISOString()
      };
      notes.push(newNote);
      localStorage.setItem(`notes_${docId}`, JSON.stringify(notes));
      alert('✅ 笔记已保存');
    }
    setShowTextMenu(false);
  };

  // 4. AI 解读
  const handleAIExplain = () => {
    setInputMessage(`请解释这段话：\n\n"${selectedText}"`);
    setShowTextMenu(false);
    // 自动聚焦输入框
    setTimeout(() => {
      document.querySelector('textarea')?.focus();
    }, 100);
  };

  // 5. 翻译
  const handleTranslate = () => {
    setInputMessage(`请将以下内容翻译成中文：\n\n"${selectedText}"`);
    setShowTextMenu(false);
    setTimeout(() => {
      document.querySelector('textarea')?.focus();
    }, 100);
  };

  // 6. 联网搜索
  const handleWebSearch = () => {
    const searchQuery = encodeURIComponent(selectedText);
    const searchTemplates = {
      google: 'https://www.google.com/search?q={query}',
      baidu: 'https://www.baidu.com/s?wd={query}',
      bing: 'https://www.bing.com/search?q={query}',
      sogou: 'https://www.sogou.com/web?query={query}',
      custom: searchEngineUrl
    };
    const template = searchTemplates[searchEngine] || searchTemplates.google;
    const searchUrl = template.includes('{query}')
      ? template.replace('{query}', searchQuery)
      : `${template}${template.includes('?') ? '&' : '?'}q=${searchQuery}`;
    window.open(searchUrl, '_blank', 'noopener,noreferrer');
    setShowTextMenu(false);
  };

  // 7. 分享（生成卡片）
  const handleShare = () => {
    const shareText = `📄 来自《${docInfo?.filename || '文档'}》第 ${currentPage} 页：\n\n"${selectedText}"\n\n--- ChatPDF Pro ---`;
    navigator.clipboard.writeText(shareText).then(() => {
      alert('✅ 引用卡片已复制到剪贴板，可直接粘贴分享');
    });
    setShowTextMenu(false);
  };

  // 关闭工具栏
  const handleCloseToolbar = () => {
    setShowTextMenu(false);
    setSelectedText('');
    window.getSelection()?.removeAllRanges();
  };

  // 手动拖动/缩放时更新位置与缩放
  const handleToolbarPositionChange = (pos) => setToolbarPosition(pos);
  const handleToolbarScaleChange = (scale) => setToolbarScale(scale);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // 预设问题自动发送标记
  const pendingSendRef = useRef(false);

  // ==================== 引文点击跳转处理 ====================
  /**
   * 处理引文引用点击事件
   * 根据 citation 中的 page_range 跳转 PDF 阅读器到对应页码
   * @param {object} citation - 引文数据，包含 ref、group_id、page_range
   */
  const handleCitationClick = (citation) => {
    if (!citation || !citation.page_range) return;
    const targetPage = citation.page_range[0]; // 跳转到页码范围的起始页
    if (typeof targetPage === 'number' && targetPage > 0) {
      // 先清除旧高亮，确保即使同页也能重新触发
      setActiveHighlight(null);
      setCurrentPage(targetPage);
      // 延迟设置高亮，等待 PDFViewer 完成页面切换和文本层渲染
      if (citation.highlight_text) {
        setTimeout(() => {
          setActiveHighlight({
            page: targetPage,
            text: citation.highlight_text,
            source: 'citation'
          });
        }, 400);
      }
    }
  };

  // 处理预设问题选择：填入输入框并标记待发送
  const handlePresetSelect = (query) => {
    setInputMessage(query);
    pendingSendRef.current = true;
  };

  // 当 inputMessage 更新且有待发送标记时，自动触发发送
  useEffect(() => {
    if (pendingSendRef.current && inputMessage.trim()) {
      pendingSendRef.current = false;
      sendMessage();
    }
  }, [inputMessage]);

  // 判断是否显示预设问题：文档已加载且没有用户/助手消息
  const showPresetQuestions = docId && messages.filter(
    msg => msg.type === 'user' || msg.type === 'assistant'
  ).length === 0;

  // ==================== 历史记录管理 ====================
  const loadHistory = () => {
    try {
      const saved = localStorage.getItem('chatHistory');
      if (saved) {
        setHistory(JSON.parse(saved));
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const saveCurrentSession = () => {
    try {
      const sessionId = docId;
      const existingHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');

      const sessionIndex = existingHistory.findIndex(s => s.id === sessionId);
      const sessionData = {
        id: sessionId,
        docId: docId,
        filename: docInfo.filename,
        messages: messages,
        createdAt: sessionIndex >= 0 ? existingHistory[sessionIndex].createdAt : Date.now(),
        updatedAt: Date.now()
      };

      if (sessionIndex >= 0) {
        existingHistory[sessionIndex] = sessionData;
      } else {
        existingHistory.unshift(sessionData);
      }

      // 最多保留 50 个历史记录
      const limitedHistory = existingHistory.slice(0, 50);
      localStorage.setItem('chatHistory', JSON.stringify(limitedHistory));
      setHistory(limitedHistory);
    } catch (error) {
      console.error('Failed to save session:', error);
    }
  };

  const loadSession = async (session) => {
    try {
      console.log('🔵 Loading session:', session);

      // Show loading state
      setIsLoading(true);

      // 加载文档信息
      const docResponse = await fetch(`${API_BASE_URL}/document/${session.docId}?t=${new Date().getTime()}`);
      console.log('🔵 Document response status:', docResponse.status);

      if (!docResponse.ok) {
        if (docResponse.status === 404) {
          alert('无法加载文档：文件不存在。\n\n可能原因：\n1. 这是旧版本的历史记录（未开启持久化存储）\n2. 服务器数据已被清理\n3. 后端服务未启动');
        } else {
          alert(`加载文档失败 (HTTP ${docResponse.status})\n\n请检查：\n1. 后端服务是否正常运行\n2. 网络连接是否正常`);
        }
        setIsLoading(false);
        return;
      }

      const docData = await docResponse.json();
      console.log('🟢 Document data loaded:', docData);

      // 恢复会话状态
      setDocId(session.docId);
      setDocInfo(docData);
      setMessages(session.messages || []);
      setCurrentPage(1);

      console.log('✅ Session loaded successfully');
    } catch (error) {
      console.error('❌ Failed to load session:', error);
      alert(`加载会话失败: ${error.message}\n\n可能原因：\n1. 后端服务未启动\n2. 网络连接问题\n\n请检查浏览器控制台查看详细错误`);
    } finally {
      setIsLoading(false);
    }
  };

  const deleteSession = (sessionId) => {
    try {
      const confirmed = window.confirm('确定要删除这个对话吗？');
      if (!confirmed) return;

      const existingHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');
      const updatedHistory = existingHistory.filter(s => s.id !== sessionId);

      localStorage.setItem('chatHistory', JSON.stringify(updatedHistory));
      setHistory(updatedHistory);

      // 如果删除的是当前会话，清空界面
      if (sessionId === docId) {
        setDocId(null);
        setDocInfo(null);
        setMessages([]);
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  const startNewChat = () => {
    setDocId(null);
    setDocInfo(null);
    setMessages([]);
    setCurrentPage(1);
    setSelectedText('');
    setScreenshot(null);
  };

  /**
   * 处理区域框选完成回调
   * 将选区裁剪到页面范围内，调用 captureArea 生成截图
   */
  const handleAreaSelected = async (rect) => {
    // 获取 PDF 页面容器的尺寸，用于裁剪选区
    const container = pdfContainerRef.current;
    if (!container) {
      setIsSelectingArea(false);
      return;
    }

    const containerRect = container.getBoundingClientRect();
    // 将选区裁剪到页面范围内
    const clampedRect = clampSelectionToPage(rect, containerRect.width, containerRect.height);

    try {
      const result = await captureArea(pdfContainerRef, clampedRect);
      if (result) {
        setScreenshot(result);
      } else {
        alert('截图生成失败，请重试');
      }
    } catch (e) {
      console.error('截图生成异常:', e);
      alert('截图生成失败，请重试');
    } finally {
      // 退出框选模式
      setIsSelectingArea(false);
    }
  };

  /**
   * 处理取消框选回调（Escape 键）
   */
  const handleSelectionCancel = () => {
    setIsSelectingArea(false);
  };

  // ==================== 截图快捷操作分发 ====================

  /**
   * 处理截图预览中的快捷操作
   *
   * - ask: 保留截图作为附件，不自动发送，用户可输入问题后手动发送
   * - explain/table/formula/ocr/translate: 设置预设提示词 + 截图后自动发送
   * - copy: 将截图写入系统剪贴板（Clipboard API）
   *
   * @param {string} actionKey - 操作类型 key
   */
  const handleScreenshotAction = async (actionKey) => {
    const action = SCREENSHOT_ACTIONS[actionKey]
    if (!action) return

    // 复制操作：将截图写入剪贴板
    if (actionKey === 'copy') {
      try {
        // 将 base64 data URL 转换为 Blob
        const response = await fetch(screenshot)
        const blob = await response.blob()
        const clipboardItem = new ClipboardItem({ 'image/png': blob })
        await navigator.clipboard.write([clipboardItem])
        // 可选：提示用户复制成功（不清除截图）
      } catch (e) {
        console.error('复制截图到剪贴板失败:', e)
        alert('复制失败，浏览器不支持此功能')
      }
      return
    }

    // 提问操作：保留截图，不自动发送，聚焦输入框
    if (actionKey === 'ask') {
      // 截图已在 state 中，用户可以输入问题后手动发送
      setTimeout(() => {
        document.querySelector('textarea')?.focus()
      }, 100)
      return
    }

    // 自动发送操作：设置预设提示词后触发发送
    if (action.autoSend && action.prompt) {
      setInputMessage(action.prompt)
      // 使用 pendingSendRef 标记待发送，与预设问题相同的机制
      pendingSendRef.current = true
    }
  }

  /**
   * 处理截图预览关闭
   */
  const handleScreenshotClose = () => {
    setScreenshot(null)
  }

  // ==================== 模型切换时清除不兼容的截图 ====================

  /**
   * 当聊天模型切换时，如果新模型不支持视觉能力，
   * 自动清除已有的截图数据并隐藏预览区域。
   */
  useEffect(() => {
    if (screenshot && !isVisionCapable) {
      setScreenshot(null)
    }
  }, [isVisionCapable])

  // Render Components
  return (
    <div
      className={`h-screen w-full flex overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-gray-900 text-white' : 'bg-transparent text-[var(--color-text-main)]'}`}
      onClick={(e) => {
        if (!showTextMenu) return;
        // 避免刚选中文本时被同一次点击事件立刻关闭
        const selection = window.getSelection();
        const hasActiveSelection = selection && selection.toString().trim().length > 0;
        if (hasActiveSelection) return;

        // 点击工具栏外部才关闭
        if (!e.target.closest('.text-selection-toolbar-container')) {
          handleCloseToolbar();
        }
      }}
    >

      {/* 划词工具栏 */}
      {
        showTextMenu && selectedText && (
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
        )
      }

      {/* Sidebar (History) */}
      <motion.div
        initial={false}
        animate={{
          width: showSidebar ? 288 : 0,
          opacity: showSidebar ? 1 : 0
        }}
        transition={{ duration: 0.2, ease: "easeInOut" }}
        style={{ pointerEvents: showSidebar ? 'auto' : 'none' }}
        className={`flex-shrink-0 m-6 mr-0 h-[calc(100vh-3rem)] flex flex-col z-20 overflow-hidden rounded-[var(--radius-panel-lg)] ${darkMode ? 'bg-gray-900/80 border-white/10 backdrop-blur-3xl backdrop-saturate-150' : 'bg-white/80 border-white/50 backdrop-blur-3xl backdrop-saturate-150 border shadow-xl'}`}
      >
        <div className="w-72 mx-auto flex flex-col h-full items-stretch">
          <div className="px-6 py-8 flex items-center justify-between">
            <div className="flex items-center gap-3 font-bold text-2xl text-blue-600 tracking-tight">
              <Bot className="w-9 h-9" />
              <span>ChatPDF</span>
            </div>
            <div className="flex items-center gap-2">
              {!isHeaderExpanded && (
                <button
                  onClick={() => setIsHeaderExpanded(true)}
                  className="p-2.5 hover:bg-black/5 rounded-full transition-colors text-gray-500 hover:text-gray-800"
                  title="展开顶栏"
                >
                  <ChevronDown className="w-5 h-5" />
                </button>
              )}
              <button onClick={() => setDarkMode(!darkMode)} className="p-2.5 hover:bg-black/5 rounded-full transition-colors">
                {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
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
                className={`w-full max-w-[260px] p-3 rounded-xl cursor-pointer group flex items-center gap-3 transition-all duration-200 ${item.id === docId ? 'bg-white shadow-md scale-[1.02]' : 'hover:bg-white/40'}`}
              >
                <MessageSquare className="w-5 h-5 text-blue-500" />
                <div className="flex-1 truncate text-sm font-medium">{item.filename}</div>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteSession(item.id); }}
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

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full relative transition-all duration-200 ease-in-out">
        {/* Header - Collapsible */}
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
            animate={{
              opacity: isHeaderExpanded ? 1 : 0,
              y: isHeaderExpanded ? 0 : -6
            }}
            transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
          >
            <div className="flex items-center justify-between w-full py-3">
              <div className="flex items-center gap-4">
                {/* 菜单按钮 */}
                <button
                  onClick={() => setShowSidebar(!showSidebar)}
                  className="p-2 hover:bg-black/5 rounded-lg transition-colors"
                  title={showSidebar ? "隐藏侧边栏" : "显示侧边栏"}
                >
                  <Menu className="w-6 h-6" />
                </button>

                <div className="flex items-center gap-4">
                  <div className="bg-blue-600 text-white p-2.5 rounded-xl shadow-sm">
                    <FileText className="w-6 h-6" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-[var(--color-text-main)]">
                      ChatPDF Pro <span className="text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full ml-2 align-middle">v2.0.2</span>
                    </h1>
                    <p className="text-xs text-gray-500 font-medium mt-0.5">智能文档助手</p>
                  </div>
                </div>
              </div>

              {/* Search Box - Center */}
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
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !isSearching) handleSearch();
                      }}
                      className="w-full px-4 py-2 pl-11 pr-4 rounded-full soft-input text-sm transition-all focus:ring-2 focus:ring-blue-400"
                      disabled={isSearching}
                    />
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                  </div>
                  <motion.button
                    whileHover={{ scale: isSearching ? 1 : 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleSearch()}
                    disabled={isSearching}
                    className={`px-3 py-2 rounded-full text-sm font-medium shadow-sm flex items-center gap-2 transition-all ${isSearching
                      ? 'bg-blue-200 text-blue-700 cursor-wait'
                      : 'bg-blue-600 text-white hover:shadow-md hover:bg-blue-700'
                      }`}
                  >
                    {isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                    <span>{isSearching ? '搜索中...' : '搜索'}</span>
                  </motion.button>
                  <button
                    onClick={() => setUseRerank(v => !v)}
                    className={`px-3 py-2 rounded-full border text-sm font-medium flex items-center gap-1 transition-colors ${useRerank ? 'bg-purple-50 text-purple-700 border-purple-200' : 'bg-white text-gray-600 border-gray-200'
                      }`}
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
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={goToPrevResult}
                          className="p-1.5 hover:bg-black/5 rounded-lg transition-colors"
                          title="上一个结果"
                        >
                          <ChevronUp className="w-4 h-4" />
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={goToNextResult}
                          className="p-1.5 hover:bg-black/5 rounded-lg transition-colors"
                          title="下一个结果"
                        >
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

        {/* Floating Controls when Header is Collapsed AND Sidebar is Hidden */}
        <AnimatePresence>
          {!isHeaderExpanded && !showSidebar && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="absolute top-4 left-2 z-20 flex flex-col gap-2"
            >
              <button
                onClick={() => {
                  setShowSidebar(true);
                  setIsHeaderExpanded(true);
                }}
                className="p-2 bg-white/80 backdrop-blur-md shadow-sm rounded-full hover:bg-white hover:scale-105 transition-all text-gray-700 border border-white/50"
                title="显示侧边栏 & 展开顶栏"
              >
                <Menu className="w-4 h-4" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>
        {/* Content Area */}
        <div className="flex-1 flex overflow-hidden px-8 pb-8 gap-4 pt-2">

          {/* Left: PDF Preview (Floating Card) */}
          {docId ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`soft-panel overflow-hidden flex flex-col relative flex-shrink-0 rounded-[var(--radius-panel)] min-w-0 ${darkMode ? 'bg-gray-800/50' : ''}`}
              style={{ width: `${pdfPanelWidth}%`, minWidth: '350px' }}
            >
              {/* PDF Content */}
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
                    onTextSelect={(text) => {
                      if (text) {
                        setSelectedText(text);
                        setShowTextMenu(true);
                        // 获取选中文本的位置
                        const selection = window.getSelection();
                        if (selection.rangeCount > 0) {
                          const range = selection.getRangeAt(0);
                          const rect = range.getBoundingClientRect();
                          const nextPos = {
                            x: rect.left + rect.width / 2,
                            y: rect.top - 10
                          };
                          setMenuPosition(nextPos);
                          setToolbarPosition(nextPos);
                        }
                      }
                    }}
                  />
                ) : (docInfo?.pages || docInfo?.data?.pages) ? (
                  <>
                    {/* Text-based PDF Toolbar */}
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
                      {/* 旧的整页截图按钮已移除，使用 Chat_Toolbar 中的区域截图按钮替代 */}
                    </div>
                    <div ref={pdfContainerRef} className="h-full overflow-auto bg-gray-50/50">
                      <div className="min-h-full flex items-start justify-center p-8" style={{ zoom: pdfScale }}>
                        <div
                          className="bg-white shadow-2xl p-12 rounded-lg max-w-4xl w-full"
                          onMouseUp={handleTextSelection}
                        >
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
            // Empty State
            <div className="flex-1 flex items-center justify-center relative overflow-hidden">
              {/* Background Blobs */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full max-w-lg max-h-lg pointer-events-none">
                <div className="blob bg-purple-200 w-72 h-72 top-0 left-0 mix-blend-multiply animate-blob"></div>
                <div className="blob bg-cyan-100 w-72 h-72 bottom-0 right-0 mix-blend-multiply animate-blob animation-delay-2000"></div>
              </div>

              <div className="text-center space-y-8 max-w-md relative z-10">
                <div className="w-24 h-24 bg-white/50 backdrop-blur-md rounded-[32px] flex items-center justify-center mx-auto shadow-sm border border-white/60">
                  <Upload className="w-10 h-10 text-blue-500/80" />
                </div>

                <div className="space-y-2">
                  <h2 className="text-3xl font-bold text-gray-800 tracking-tight">Upload a PDF to Start</h2>
                  <p className="text-gray-500 text-lg">Chat with your documents using AI.</p>
                </div>

              </div>
            </div>
          )}

          {/* Resizable Divider */}
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
            {/* Invisible trigger area, visible line on hover */}
            <div className="w-1 h-full rounded-full bg-transparent group-hover:bg-blue-500/50 transition-colors duration-200" />
          </div>

          {/* Right: Chat Area (Floating Card) */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`soft-panel flex flex-col overflow-hidden rounded-[var(--radius-panel)] min-w-0 ${darkMode ? 'bg-gray-800/50' : ''}`}
            style={{ width: `calc(${100 - pdfPanelWidth}% - 2rem)`, minWidth: '350px' }}
          >
            {/* Messages */}
            <div className="flex-1 overflow-y-auto overflow-x-hidden p-6 space-y-6 min-w-0" ref={chatPaneRef}>
              {(searchResults.length > 0 || isSearching || searchHistory.length > 0) && (
                <div className="rounded-3xl border border-black/5 bg-white/70 backdrop-blur-sm p-4 space-y-3 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Search className="w-4 h-4 text-blue-500" />
                      <span className="font-semibold text-sm text-gray-800">文档搜索</span>
                      {useRerank && (
                        <span className="text-xs text-purple-700 bg-purple-50 px-2 py-0.5 rounded-full border border-purple-100">
                          已开启重排
                        </span>
                      )}
                      {isSearching && <Loader2 className="w-4 h-4 animate-spin text-blue-500" />}
                    </div>
                    {searchResults.length > 0 && (
                      <span className="text-xs text-gray-500">找到 {searchResults.length} 个候选</span>
                    )}
                  </div>

                  {searchHistory.length > 0 && (
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs text-gray-500">历史:</span>
                      {searchHistory.map((item, idx) => (
                        <button
                          key={`history-${idx}`}
                          onClick={() => handleSearch(item)}
                          className="text-xs px-2 py-1 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors"
                        >
                          {item}
                        </button>
                      ))}
                      <button
                        onClick={clearSearchHistory}
                        className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1 rounded-full hover:bg-black/5 transition-colors"
                      >
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
                      <p className="text-sm text-gray-500 px-2">
                        输入查询并点击“搜索”查看匹配片段，支持关键词上下文和匹配度展示。
                      </p>
                    )}

                    {searchResults.map((result, idx) => (
                      <button
                        key={`result-${idx}`}
                        onClick={() => focusResult(idx)}
                        className="w-full text-left p-3 rounded-2xl border border-gray-100 hover:border-blue-200 hover:bg-blue-50/40 transition-all relative"
                      >
                        <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                          <div className="flex items-center gap-1.5">
                            <span>第 {result.page || 1} 页 · #{idx + 1}</span>
                            {result.reranked && (
                              <span className="text-[10px] text-purple-700 bg-purple-50 px-1.5 py-0.5 rounded-full border border-purple-100">
                                Rerank
                              </span>
                            )}
                          </div>
                          <span className={`font-semibold ${formatSimilarity(result) >= 80 ? 'text-green-600' : 'text-blue-600'}`}>
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
              )}

              {/* 预设问题栏：文档已加载且无对话消息时显示 */}
              {showPresetQuestions && (
                <PresetQuestions
                  onSelect={handlePresetSelect}
                  disabled={isLoading}
                />
              )}

              {messages.map((msg, idx) => {
                const messageKey = msg.id ?? idx;
                const hasThinking = typeof msg.thinking === 'string' && msg.thinking.trim().length > 0;

                return (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={messageKey}
                    className={`flex flex-col ${msg.type === 'user' ? 'items-end' : 'items-start'}`}
                  >
                    <div className={`${msg.type === 'user'
                      ? 'max-w-[85%] rounded-2xl px-4 py-3 shadow-sm message-bubble-user rounded-tr-sm text-sm'
                      : 'w-full max-w-full min-w-0 bg-transparent shadow-none p-0 text-gray-800 dark:text-gray-100 overflow-hidden'
                      }`}
                      style={msg.type !== 'user' ? { contain: 'inline-size' } : undefined}
                    >
                      {hasThinking && (
                        <ThinkingBlock
                          content={msg.thinking}
                          isStreaming={msg.isStreaming && streamingMessageId === msg.id}
                          darkMode={darkMode}
                          thinkingMs={msg.thinkingMs || 0}
                        />
                      )}

                      {msg.hasImage && (
                        <div className="mb-2 rounded-lg overflow-hidden border border-white/20">
                          <div className="bg-black/10 p-2 flex items-center gap-2 text-xs">
                            <ImageIcon className="w-3 h-3" /> Image attached
                          </div>
                        </div>
                      )}

                      {/* AI Avatar/Header for Assistant Messages */}
                      {msg.type === 'assistant' && (
                        <div className="flex items-center gap-2 mb-2 select-none">
                          <div className="p-1 rounded-lg bg-blue-600 text-white shadow-sm">
                            <Bot className="w-4 h-4" />
                          </div>
                          <span className="font-bold text-sm text-gray-800 dark:text-gray-100">AI Assistant</span>
                          {msg.model && <span className="text-xs text-gray-400 border border-gray-200 rounded px-1.5 py-0.5">{msg.model}</span>}
                        </div>
                      )}

                      <StreamingMarkdown
                        content={msg.content}
                        isStreaming={msg.isStreaming || false}
                        enableBlurReveal={enableBlurReveal}
                        blurIntensity={blurIntensity}
                        citations={msg.citations || null}
                        onCitationClick={handleCitationClick}
                      />
                    </div>

                    {/* Action Buttons - Only show for assistant messages that are not streaming */}
                    {msg.type === 'assistant' && !msg.isStreaming && (
                      <div className="flex items-center gap-1 mt-1 ml-2">
                        <button
                          onClick={() => copyMessage(msg.content, msg.id || idx)}
                          className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
                          title="复制"
                        >
                          {copiedMessageId === (msg.id || idx) ? (
                            <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </button>
                        <button
                          onClick={() => regenerateMessage(idx)}
                          className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
                          title="重新生成"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                          </svg>
                        </button>
                        <button
                          onClick={() => saveToMemory(idx, 'liked')}
                          className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${likedMessages.has(idx) ? 'text-pink-500' : 'text-gray-500 hover:text-gray-700'}`}
                          title="点赞并记忆"
                        >
                          <svg className="w-4 h-4" fill={likedMessages.has(idx) ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                          </svg>
                        </button>
                        <button
                          className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
                          title="点踩"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
                          </svg>
                        </button>
                        <button
                          onClick={() => saveToMemory(idx, 'manual')}
                          className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${rememberedMessages.has(idx) ? 'text-violet-500' : 'text-gray-500 hover:text-gray-700'}`}
                          title="记住这个"
                        >
                          <Brain className={`w-4 h-4 ${rememberedMessages.has(idx) ? 'fill-current' : ''}`} />
                        </button>
                      </div>
                    )}
                  </motion.div>
                );
              })
              }
              {/* 等待动画已由 StreamingMarkdown 组件内部的 streaming-dots 处理 */}
              <div ref={messagesEndRef} />
            </div >

            {/* Input Area - Clean Card Style */}
            <div className="p-6 pt-0 bg-transparent">
              {/* 截图预览与快捷操作面板 */}
              <ScreenshotPreview
                screenshotData={screenshot}
                onAction={handleScreenshotAction}
                onClose={handleScreenshotClose}
              />

              <div className="relative bg-white/80 backdrop-blur-[20px] rounded-[36px] shadow-[0_24px_56px_-12px_rgba(0,0,0,0.22),0_8px_24px_-6px_rgba(0,0,0,0.12),inset_0_1px_0_rgba(255,255,255,0.9)] p-1.5 flex items-end gap-2 border border-white/50 ring-1 ring-black/5">
                <div className="flex-1 flex flex-col min-h-[48px] justify-center pl-6 py-1.5">
                  <div className="flex items-center gap-2 mb-1">

                    <textarea
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())}
                      placeholder="Summarize, rephrase, convert..."
                      className="w-full bg-transparent border-none outline-none text-gray-800 placeholder:text-gray-400 font-medium resize-none h-[24px] overflow-hidden leading-relaxed py-0 focus:ring-0 text-[15px]"
                      rows={1}
                      style={{ minHeight: '24px', maxHeight: '120px' }}
                      onInput={(e) => {
                        e.target.style.height = '24px';
                        e.target.style.height = e.target.scrollHeight + 'px';
                      }}
                    />
                  </div>

                  <div className="flex items-center gap-4 text-gray-400 mt-2">
                    {/* 模型快速切换器 */}
                    <ModelQuickSwitch onThinkingChange={(enabled) => {
                      setEnableThinking(enabled);
                      // reasoningEffort 已由 ModelQuickSwitch 直接通过 GlobalSettingsContext 更新
                    }} />
                    <button className="hover:text-gray-600 transition-colors p-1 rounded-md hover:bg-gray-50">
                      <SlidersHorizontal className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="hover:text-gray-600 transition-colors p-1 rounded-md hover:bg-gray-50"
                    >
                      <Paperclip className="w-5 h-5" />
                    </button>
                    {/* 截图按钮 — 仅当模型支持视觉能力时显示 */}
                    {isVisionCapable && (
                      <button
                        onClick={() => setIsSelectingArea(true)}
                        disabled={!docId}
                        className={`transition-colors p-1 rounded-md ${
                          docId
                            ? isSelectingArea
                              ? 'text-purple-600 bg-purple-50 hover:bg-purple-100'
                              : 'hover:text-gray-600 hover:bg-gray-50'
                            : 'text-gray-300 cursor-not-allowed'
                        }`}
                        title={!docId ? '请先上传文档' : isSelectingArea ? '框选模式已开启' : '区域截图'}
                      >
                        <Scan className="w-5 h-5" />
                      </button>
                    )}
                  </div>
                </div>

                <motion.button
                  onClick={isLoading ? handleStop : sendMessage}
                  disabled={!isLoading && (!inputMessage.trim() && !screenshot)}
                  className="glass-btn-3d relative z-10 flex-shrink-0"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <AnimatePresence initial={false}>
                    {isLoading ? (
                      <motion.div
                        key="pause"
                        initial={{ rotate: -90, scale: 0.5, opacity: 0 }}
                        animate={{ rotate: 0, scale: 1, opacity: 1 }}
                        exit={{ rotate: 90, scale: 0.5, opacity: 0 }}
                        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
                        className="absolute inset-0 flex items-center justify-center"
                      >
                        <PauseIcon />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="send"
                        initial={{ rotate: -90, scale: 0.5, opacity: 0 }}
                        animate={{ rotate: 0, scale: 1, opacity: 1 }}
                        exit={{ rotate: 90, scale: 0.5, opacity: 0 }}
                        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
                        className="absolute inset-0 flex items-center justify-center"
                      >
                        <SendIcon />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.button>
              </div>
            </div>
          </motion.div>
        </div >
      </div>



      {/* Upload Progress Modal */}
      <AnimatePresence>
        {isUploading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-[2px] transition-colors">
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 10 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 10 }}
              transition={{ type: 'spring', stiffness: 350, damping: 25 }}
              className="relative p-10 flex flex-col items-center gap-6 min-w-[340px] rounded-[32px] overflow-hidden"
              style={{
                background: 'rgba(255, 255, 255, 0.85)',
                backdropFilter: 'blur(20px)',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.15), inset 0 0 0 1px rgba(255, 255, 255, 0.5)'
              }}
            >
              {/* Glow Effect */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-blue-400/20 blur-[60px] rounded-full pointer-events-none" />

              {/* Animated Icon */}
              <div className="relative w-28 h-28 flex items-center justify-center z-10">
                {/* Background ring */}
                <svg className="w-full h-full rotate-[-90deg]" viewBox="0 0 100 100">
                  <circle
                    cx="50" cy="50" r="42"
                    fill="none"
                    stroke="#F3F4F6"
                    strokeWidth="6"
                    strokeLinecap="round"
                  />
                  {/* Progress ring */}
                  <motion.circle
                    cx="50" cy="50" r="42"
                    fill="none"
                    stroke="url(#progressGradient)"
                    strokeWidth="6"
                    strokeLinecap="round"
                    strokeDasharray={264}
                    strokeDashoffset={264}
                    animate={{ strokeDashoffset: 264 - (264 * uploadProgress) / 100 }}
                    transition={{ duration: 0.4, ease: 'easeOut' }}
                  />
                  <defs>
                    <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#60A5FA" />
                      <stop offset="100%" stopColor="#C084FC" />
                    </linearGradient>
                  </defs>
                </svg>
                {/* Center percentage */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <motion.span
                    key={uploadProgress}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-br from-blue-600 to-purple-600"
                  >
                    {uploadProgress}%
                  </motion.span>
                </div>
              </div>

              {/* Status Text */}
              <div className="text-center space-y-1.5 z-10">
                <motion.h3
                  key={uploadStatus}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-xl font-semibold text-gray-800 tracking-tight"
                >
                  {uploadStatus === 'uploading' ? '正在上传文档...' : '正在智能解析...'}
                </motion.h3>
                <p className="text-sm text-gray-500 font-medium">
                  {uploadStatus === 'uploading'
                    ? '请保持网络连接畅通'
                    : 'AI正在构建知识库索引'}
                </p>
              </div>

              {/* Progress Bar */}
              <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden z-10">
                <motion.div
                  className="h-full bg-gradient-to-r from-blue-400 to-purple-500 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${uploadProgress}%` }}
                  transition={{ duration: 0.4, ease: 'easeOut' }}
                />
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Settings Modal */}
      < AnimatePresence initial={false} >
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
                      {/* Header Section */}
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

                      {/* Model Status Cards */}
                      {/* Model Status Cards */}
                      <div className="flex flex-col gap-3">
                        {/* Chat Model Card */}
                        <div className="group relative overflow-hidden rounded-[18px] border border-gray-100/50 bg-white/40 p-3 transition-all hover:border-blue-200 hover:bg-white/60 hover:shadow-sm backdrop-blur-sm">
                          <div className="flex items-center gap-3 mb-2">
                            <div className="p-1.5 rounded-lg bg-blue-50/80 text-blue-600 backdrop-blur-sm">
                              <MessageSquare className="w-3.5 h-3.5" />
                            </div>
                            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Chat Model</span>
                          </div>
                          <div className="font-bold text-gray-800 text-sm truncate pl-1" title={getDefaultModelLabel(getDefaultModel('assistantModel'))}>
                            {getDefaultModelLabel(getDefaultModel('assistantModel')) || '未设置'}
                          </div>
                        </div>

                        {/* Embedding Model Card */}
                        <div className="group relative overflow-hidden rounded-[18px] border border-gray-100/50 bg-white/40 p-3 transition-all hover:border-purple-200 hover:bg-white/60 hover:shadow-sm backdrop-blur-sm">
                          <div className="flex items-center gap-3 mb-2">
                            <div className="p-1.5 rounded-lg bg-purple-50/80 text-purple-600 backdrop-blur-sm">
                              <Database className="w-3.5 h-3.5" />
                            </div>
                            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Embedding</span>
                          </div>
                          <div className="font-bold text-gray-800 text-sm truncate pl-1" title={getDefaultModelLabel(getDefaultModel('embeddingModel'))}>
                            {getDefaultModelLabel(getDefaultModel('embeddingModel')) || '未设置'}
                          </div>
                        </div>

                        {/* Rerank Model Card */}
                        <div className="group relative overflow-hidden rounded-[18px] border border-gray-100/50 bg-white/40 p-3 transition-all hover:border-amber-200 hover:bg-white/60 hover:shadow-sm backdrop-blur-sm">
                          <div className="flex items-center gap-3 mb-2">
                            <div className="p-1.5 rounded-lg bg-amber-50/80 text-amber-600 backdrop-blur-sm">
                              <ListFilter className="w-3.5 h-3.5" />
                            </div>
                            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Rerank</span>
                          </div>
                          <div className="font-bold text-gray-800 text-sm truncate pl-1" title={getDefaultModelLabel(getDefaultModel('rerankModel'))}>
                            {getDefaultModelLabel(getDefaultModel('rerankModel')) || '未设置'}
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

                {/* 全局设置入口 */}
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

                {/* 对话设置入口 */}
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

                {/* OCR 设置入口 */}
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
                          <span className="text-xs text-gray-500">{storageInfo.doc_count} 个文档</span>
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
      </AnimatePresence >

      <EmbeddingSettings
        isOpen={showEmbeddingSettings}
        onClose={() => setShowEmbeddingSettings(false)}
      />

      <GlobalSettings
        isOpen={showGlobalSettings}
        onClose={() => { setShowGlobalSettings(false); setShowSettings(true); }}
      />

      {/* 对话设置面板 */}
      <ChatSettings
        isOpen={showChatSettings}
        onClose={() => { setShowChatSettings(false); setShowSettings(true); }}
      />

      {/* OCR 设置面板 */}
      <OCRSettingsPanel
        isOpen={showOCRSettings}
        onClose={() => { setShowOCRSettings(false); setShowSettings(true); }}
      />
    </div >
  );
};

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
        className="w-full p-3 rounded-[18px] border border-gray-200 bg-white/50 backdrop-blur-sm flex items-center justify-between hover:border-blue-300 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/20"
      >
        <span className="text-sm font-medium text-gray-700">{selectedLabel}</span>
        <ChevronDown
          className={`w-4 h-4 text-gray-500 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`}
        />
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
                  onClick={() => {
                    onChange(option.value);
                    setIsOpen(false);
                  }}
                  className={`w-full text-left px-4 py-2.5 text-sm transition-colors flex items-center justify-between
                    ${option.value === value
                      ? 'bg-blue-50 text-blue-600 font-medium'
                      : 'text-gray-700 hover:bg-gray-50'
                    }`}
                >
                  {option.label}
                  {option.value === value && <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />}
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
