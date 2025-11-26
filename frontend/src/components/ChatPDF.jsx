import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, FileText, Settings, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Copy, Bot, X, Camera, Crop, Image as ImageIcon, History, Moon, Sun, Plus, MessageSquare, Trash2, Menu } from 'lucide-react';
import html2canvas from 'html2canvas';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import PDFViewer from './PDFViewer';

// API base URL – empty string so that Vite proxy forwards to backend
const API_BASE_URL = '';

const ChatPDF = () => {
  // Core State
  const [docId, setDocId] = useState(null);
  const [docInfo, setDocInfo] = useState(null);
  const [pdfPanelWidth, setPdfPanelWidth] = useState(50); // Percentage
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  // UI State
  const [showSettings, setShowSettings] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [history, setHistory] = useState([]); // Mock history for now

  // PDF State
  const [currentPage, setCurrentPage] = useState(1);
  const [pdfScale, setPdfScale] = useState(1.0);
  const [selectedText, setSelectedText] = useState('');
  const [showTextMenu, setShowTextMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({ x: 0, y: 0 });

  // Screenshot State
  const [screenshot, setScreenshot] = useState(null);
  const [isSelectingArea, setIsSelectingArea] = useState(false);
  const [selectionBox, setSelectionBox] = useState(null);

  // Settings State
  const [apiKey, setApiKey] = useState(localStorage.getItem('apiKey') || '');
  const [apiProvider, setApiProvider] = useState(localStorage.getItem('apiProvider') || 'openai');
  const [model, setModel] = useState(localStorage.getItem('model') || 'gpt-4o');
  const [availableModels, setAvailableModels] = useState({});
  const [embeddingModel, setEmbeddingModel] = useState(localStorage.getItem('embeddingModel') || 'local-minilm');
  const [embeddingApiKey, setEmbeddingApiKey] = useState(localStorage.getItem('embeddingApiKey') || '');
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState({});
  const [enableVectorSearch, setEnableVectorSearch] = useState(localStorage.getItem('enableVectorSearch') === 'true');
  const [enableScreenshot, setEnableScreenshot] = useState(localStorage.getItem('enableScreenshot') !== 'false');
  const [streamSpeed, setStreamSpeed] = useState(localStorage.getItem('streamSpeed') || 'normal'); // fast, normal, slow, off
  const [streamingMessageId, setStreamingMessageId] = useState(null);
  const [storageInfo, setStorageInfo] = useState(null);
  const [enableBlurReveal, setEnableBlurReveal] = useState(localStorage.getItem('enableBlurReveal') !== 'false');
  const [copiedMessageId, setCopiedMessageId] = useState(null);

  // Refs
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const pdfContainerRef = useRef(null);
  const selectionStartRef = useRef(null);

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
    localStorage.setItem('embeddingModel', embeddingModel);
    localStorage.setItem('embeddingApiKey', embeddingApiKey);
    localStorage.setItem('enableVectorSearch', enableVectorSearch);
    localStorage.setItem('enableScreenshot', enableScreenshot);
    localStorage.setItem('streamSpeed', streamSpeed);
    localStorage.setItem('enableBlurReveal', enableBlurReveal);
  }, [apiKey, apiProvider, model, embeddingModel, embeddingApiKey, enableVectorSearch, enableScreenshot, streamSpeed, enableBlurReveal]);

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
    const formData = new FormData();
    formData.append('file', file);
    formData.append('embedding_model', embeddingModel);

    // Use embeddingApiKey if set, otherwise fallback to apiKey for convenience
    const effectiveApiKey = embeddingApiKey || apiKey;
    if (effectiveApiKey) {
      formData.append('embedding_api_key', effectiveApiKey);
    }

    try {
      console.log('🔵 Uploading file:', file.name);
      console.log('🔵 Using embedding model:', embeddingModel);
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');

      const data = await response.json();
      console.log('🟢 Upload response:', data);
      setDocId(data.doc_id);

      // Load full document content with pages
      console.log('🔵 Fetching document details for:', data.doc_id);
      // Add timestamp to prevent browser caching of the GET request
      const docResponse = await fetch(`${API_BASE_URL}/document/${data.doc_id}?t=${new Date().getTime()}`);
      const docData = await docResponse.json();

      // Merge data from upload response (which might be fresher) with document data
      const fullDocData = { ...docData, ...data };

      console.log('🟢 Document data received:', fullDocData);

      // Debug alert to check PDF URL
      if (fullDocData.pdf_url) {
        console.log('✅ PDF URL found:', fullDocData.pdf_url);
      } else {
        console.warn('⚠️ No PDF URL found in document data');
        const keys = Object.keys(fullDocData).join(', ');
        alert(`调试信息 (v2.0.2):\n未找到 PDF URL\n\n返回字段: ${keys}\n\n请截图此弹窗发给我！`);
      }

      console.log('🟢 Pages structure:', fullDocData.pages);
      console.log('🟢 Total pages:', fullDocData.total_pages);
      setDocInfo(fullDocData);

      setMessages([{
        type: 'system',
        content: `✅ 文档《${data.filename}》上传成功！共 ${data.total_pages} 页。`
      }]);

      // Note: History will be auto-saved by the useEffect watching docId/docInfo/messages

    } catch (error) {
      console.error('❌ Upload error:', error);
      const errorMsg = error.message || '未知错误';
      alert(`上传失败: ${errorMsg}\n\n可能原因：\n1. 后端服务未启动\n2. 网络连接问题\n3. PDF文件格式不支持\n\n请检查浏览器控制台查看详细错误信息`);
    } finally {
      setIsUploading(false);
      // Reset file input to allow re-uploading the same file
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() && !screenshot) return;
    if (!docId || !apiKey) {
      alert('请先上传文档并配置API Key');
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

    const requestBody = {
      doc_id: docId,
      question: userMsg.content,
      api_key: apiKey,
      model: model,
      api_provider: apiProvider,
      selected_text: selectedText || null,
      image_base64: screenshot ? screenshot.split(',')[1] : null
    };

    // Add placeholder message for streaming effect
    const tempMsgId = Date.now();
    setStreamingMessageId(tempMsgId);
    setMessages(prev => [...prev, {
      id: tempMsgId,
      type: 'assistant',
      content: '',
      model: model,
      isStreaming: true
    }]);

    try {
      // Temporarily disable SSE due to stability issues - use regular fetch
      // Use SSE streaming if enabled and no screenshot
      if (false && streamSpeed !== 'off' && !screenshot) {
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error('Failed to get response');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let currentText = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') break;

              try {
                const parsed = JSON.parse(data);
                if (parsed.error) {
                  throw new Error(parsed.error);
                }
                if (!parsed.done) {
                  currentText += parsed.content;
                  setMessages(prev => prev.map(msg =>
                    msg.id === tempMsgId
                      ? { ...msg, content: currentText }
                      : msg
                  ));
                }
              } catch (e) {
                // Skip invalid JSON
              }
            }
          }
        }

        // Mark streaming complete
        setMessages(prev => prev.map(msg =>
          msg.id === tempMsgId
            ? { ...msg, isStreaming: false }
            : msg
        ));
        setStreamingMessageId(null);
      } else {
        // Fallback to regular fetch for non-streaming or vision requests
        const endpoint = screenshot ? '/chat/vision' : '/chat';
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error('Failed to get response');

        const data = await response.json();
        const fullAnswer = data.answer;

        if (streamSpeed === 'off') {
          // Show entire message immediately
          setMessages(prev => prev.map(msg =>
            msg.id === tempMsgId
              ? { ...msg, content: fullAnswer, isStreaming: false }
              : msg
          ));
          setStreamingMessageId(null);
        } else {
          // Client-side streaming effect - optimized for smoother display
          let currentText = '';

          const speedConfig = {
            fast: { charsPerChunk: 3, baseDelay: 15, variation: 10 },      // ~3 chars every 15-25ms
            normal: { charsPerChunk: 2, baseDelay: 25, variation: 15 },    // ~2 chars every 25-40ms
            slow: { charsPerChunk: 1, baseDelay: 50, variation: 20 }       // ~1 char every 50-70ms
          };

          const { charsPerChunk, baseDelay, variation } = speedConfig[streamSpeed] || speedConfig.normal;

          // Process in small chunks for smoother effect
          for (let i = 0; i < fullAnswer.length; i += charsPerChunk) {
            currentText = fullAnswer.substring(0, i + charsPerChunk);
            const delay = baseDelay + Math.random() * variation;
            await new Promise(resolve => setTimeout(resolve, delay));

            setMessages(prev => prev.map(msg =>
              msg.id === tempMsgId
                ? { ...msg, content: currentText }
                : msg
            ));
          }

          // Ensure complete text is shown
          setMessages(prev => prev.map(msg =>
            msg.id === tempMsgId
              ? { ...msg, content: fullAnswer, isStreaming: false }
              : msg
          ));
          setStreamingMessageId(null);
        }
      }

      setScreenshot(null); // Clear screenshot after sending
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: '❌ Error: ' + error.message
      }]);
    } finally {
      setIsLoading(false);
    }
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
    if (!docId || !apiKey) {
      alert('请先配置API Key');
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

  const handleTextSelection = () => {
    const selection = window.getSelection();
    const text = selection.toString().trim();
    if (text) {
      setSelectedText(text);
      setShowTextMenu(true);
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      setMenuPosition({ x: rect.left, y: rect.top - 40 });
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

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

  const captureFullPage = async () => {
    if (!pdfContainerRef.current) return;
    setIsLoading(true);
    try {
      const canvas = await html2canvas(pdfContainerRef.current, { scale: 2, useCORS: true });
      setScreenshot(canvas.toDataURL('image/png'));
      alert('📸 整页截图成功！');
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  // Render Components
  return (
    <div className={`h-screen w-full flex overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-gray-900 text-white' : 'bg-gradient-to-br from-[#F6F8FA] to-[#E9F4FF] text-gray-800'}`}>

      {/* Sidebar (History) */}
      <motion.div
        initial={false}
        animate={{
          width: showSidebar ? 288 : 0,
          opacity: showSidebar ? 1 : 0
        }}
        transition={{ duration: 0.2, ease: "easeInOut" }}
        style={{ pointerEvents: showSidebar ? 'auto' : 'none' }}
        className={`flex-shrink-0 glass-panel border-r border-white/40 flex flex-col z-20 overflow-hidden ${darkMode ? 'bg-gray-800/80 border-gray-700' : 'bg-white/60'}`}
      >
        <div className="w-72 flex flex-col h-full">
            <div className="p-6 flex items-center justify-between">
              <div className="flex items-center gap-2 font-bold text-xl text-blue-600">
                <Bot className="w-8 h-8" />
                <span>ChatPDF</span>
              </div>
              <button onClick={() => setDarkMode(!darkMode)} className="p-2 hover:bg-black/5 rounded-full transition-colors">
                {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
            </div>

            <div className="px-4 mb-4">
              <button
                onClick={() => { startNewChat(); fileInputRef.current?.click(); }}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl shadow-lg shadow-blue-500/20 flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <Plus className="w-5 h-5" />
                <span>新对话 / 上传PDF</span>
              </button>
              <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
            </div>

            <div className="flex-1 overflow-y-auto px-4 space-y-2">
              <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2 px-2">History</div>
              {history.map((item, idx) => (
                <div
                  key={idx}
                  onClick={() => loadSession(item)}
                  className={`p-3 rounded-xl hover:bg-white/50 cursor-pointer group flex items-center gap-3 transition-colors ${item.id === docId ? 'bg-blue-50' : ''}`}
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
        {/* Header */}
        <header className="flex items-center justify-between px-8 py-5 bg-white/80 backdrop-blur-md border-b border-white/20 sticky top-0 z-10 shadow-sm transition-all duration-200">
          <div className="flex items-center gap-4">
            {/* 菜单按钮 - 始终在左侧 */}
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="p-2 hover:bg-black/5 rounded-lg transition-colors"
              title={showSidebar ? "隐藏侧边栏" : "显示侧边栏"}
            >
              <Menu className="w-6 h-6" />
            </button>

            <div className="bg-gradient-to-br from-blue-600 to-blue-700 text-white p-2.5 rounded-xl shadow-lg shadow-blue-200">
              <FileText className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-700">
                ChatPDF Pro <span className="text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full ml-2 align-middle">v2.0.2</span>
              </h1>
              <p className="text-xs text-gray-500 font-medium mt-0.5">智能文档助手</p>
            </div>
          </div>
          {docInfo && <div className="font-medium text-sm glass-panel px-4 py-1 rounded-full">{docInfo.filename}</div>}
        </header>
        {/* Content Area */}
        <div className="flex-1 flex overflow-hidden p-6 gap-0 pt-0">

          {/* Left: PDF Preview (Floating Card) */}
          {docId ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`glass-panel rounded-[32px] overflow-hidden flex flex-col relative shadow-xl mr-6 ${darkMode ? 'bg-gray-800/50' : 'bg-white/70'}`}
              style={{ width: `${pdfPanelWidth}%` }}
            >
              {/* PDF Content */}
              <div className="flex-1 overflow-hidden">
                {docInfo?.pdf_url ? (
                  <PDFViewer
                    pdfUrl={docInfo.pdf_url}
                    onTextSelect={(text) => setSelectedText(text)}
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
                      {enableScreenshot && (
                        <div className="flex items-center gap-2">
                          <button onClick={captureFullPage} className="p-1.5 hover:bg-purple-100 text-purple-600 rounded-lg" title="Screenshot"><Camera className="w-5 h-5" /></button>
                        </div>
                      )}
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
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center space-y-6 max-w-md">
                <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto animate-float">
                  <Upload className="w-10 h-10 text-blue-600" />
                </div>
                <h2 className="text-3xl font-bold text-gray-800">Upload a PDF to Start</h2>
                <p className="text-gray-500">Chat with your documents using AI. Supports text analysis, table extraction, and visual understanding.</p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-8 py-4 bg-blue-600 text-white rounded-full font-medium shadow-lg shadow-blue-500/30 hover:shadow-blue-500/40 hover:scale-105 transition-all"
                >
                  Select PDF File
                </button>
              </div>
            </div>
          )}

          {/* Resizable Divider */}
          <div
            className="w-2 cursor-col-resize hover:bg-blue-500/20 transition-colors flex-shrink-0 relative group"
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
            <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-1 bg-gray-300 group-hover:bg-blue-500 transition-colors" />
          </div>

          {/* Right: Chat Area (Floating Card) */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`glass-panel rounded-[32px] flex flex-col overflow-hidden shadow-xl ${darkMode ? 'bg-gray-800/50' : 'bg-white/70'}`}
            style={{ width: `${100 - pdfPanelWidth}%` }}
          >
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {messages.map((msg, idx) => (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  key={idx}
                  className={`flex flex-col ${msg.type === 'user' ? 'items-end' : 'items-start'}`}
                >
                  <div className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${msg.type === 'user'
                    ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-tr-none'
                    : 'bg-white/80 backdrop-blur-sm text-gray-800 rounded-tl-none border border-white/50'
                    }`}>
                    {msg.hasImage && (
                      <div className="mb-2 rounded-lg overflow-hidden border border-white/20">
                        <div className="bg-black/10 p-2 flex items-center gap-2 text-xs">
                          <ImageIcon className="w-3 h-3" /> Image attached
                        </div>
                      </div>
                    )}
                    <div
                      className={`prose prose-sm max-w-none dark:prose-invert ${
                        enableBlurReveal && msg.isStreaming ? 'blur-reveal' : ''
                      }`}
                      style={enableBlurReveal && msg.isStreaming ? {
                        filter: `blur(${Math.max(0, 8 - (msg.content.length / 50))}px)`,
                        transition: 'filter 0.3s ease-out'
                      } : {}}
                    >
                      <ReactMarkdown
                        remarkPlugins={[remarkMath]}
                        rehypePlugins={[rehypeKatex, rehypeHighlight]}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                    {msg.model && <div className="text-xs text-gray-400 mt-2">Model: {msg.model}</div>}
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
                        className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
                        title="点赞"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                    </div>
                  )}
                </motion.div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white/50 rounded-2xl p-4 flex gap-2 items-center">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white/30 backdrop-blur-md border-t border-white/20">
              {screenshot && (
                <div className="mb-2 inline-flex items-center gap-2 bg-purple-100 text-purple-700 px-3 py-1 rounded-full text-xs font-medium">
                  <ImageIcon className="w-3 h-3" />
                  Screenshot ready
                  <button onClick={() => setScreenshot(null)} className="hover:text-purple-900"><X className="w-3 h-3" /></button>
                </div>
              )}
              <div className="relative flex items-end gap-2">
                <div className="flex-1 relative">
                  <textarea
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())}
                    placeholder="Ask anything about the document..."
                    className="w-full glass-input rounded-[24px] py-3 pl-4 pr-12 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50 text-sm min-h-[48px] max-h-32"
                    rows={1}
                  />
                  <button className="absolute right-2 bottom-2 p-2 text-gray-400 hover:text-blue-600 transition-colors">
                    <Bot className="w-5 h-5" />
                  </button>
                </div>
                <button
                  onClick={sendMessage}
                  disabled={isLoading || (!inputMessage.trim() && !screenshot)}
                  className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center shadow-lg shadow-blue-500/30 hover:scale-105 active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send className="w-5 h-5 ml-0.5" />
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      </div >

      {/* Upload Progress Modal */}
      <AnimatePresence>
        {isUploading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              className="bg-white rounded-3xl shadow-2xl p-8 flex flex-col items-center gap-4"
            >
              <div className="relative w-20 h-20">
                <div className="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
              </div>
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-800 mb-1">上传中...</h3>
                <p className="text-sm text-gray-500">正在处理PDF文件，请稍候</p>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Settings Modal */}
      < AnimatePresence >
        {showSettings && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm p-4"
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white rounded-3xl shadow-2xl w-[500px] max-w-full max-h-[90vh] overflow-hidden flex flex-col"
            >
              <div className="flex justify-between items-center p-8 pb-4 flex-shrink-0">
                <h2 className="text-2xl font-bold text-gray-800">Settings</h2>
                <button onClick={() => setShowSettings(false)} className="p-2 hover:bg-gray-100 rounded-full transition-colors"><X className="w-5 h-5" /></button>
              </div>

              <div className="space-y-4 px-8 overflow-y-auto flex-1">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">API Provider</label>
                  <select
                    value={apiProvider}
                    onChange={(e) => {
                      const newProvider = e.target.value;
                      setApiProvider(newProvider);
                      // Auto-select first model for the new provider
                      const providerModels = availableModels[newProvider]?.models;
                      if (providerModels) {
                        const firstModel = Object.keys(providerModels)[0];
                        if (firstModel) setModel(firstModel);
                      }
                    }}
                    className="w-full p-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                  >
                    {Object.keys(availableModels).map(p => (
                      <option key={p} value={p}>{availableModels[p]?.name || p}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full p-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                  >
                    {availableModels[apiProvider]?.models && Object.entries(availableModels[apiProvider].models).map(([k, v]) => (
                      <option key={k} value={k}>{v}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className="w-full p-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                    placeholder="sk-..."
                  />
                </div>

                <div className="pt-4 border-t border-gray-100">
                  <h3 className="text-sm font-semibold text-gray-800 mb-3">🔍 Embedding 配置</h3>

                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Embedding API Key
                      <span className="ml-2 text-xs font-normal text-gray-500">
                        (可选，默认使用上方聊天API Key)
                      </span>
                    </label>
                    <input
                      type="password"
                      value={embeddingApiKey}
                      onChange={(e) => setEmbeddingApiKey(e.target.value)}
                      className="w-full p-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                      placeholder="留空则使用上方API Key"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      💡 如需使用不同提供商的embedding（如OpenAI聊天+阿里云embedding），请在此单独配置
                    </p>
                  </div>

                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Embedding Model
                    <span className="ml-2 text-xs font-normal text-gray-500">
                      (用于文档向量化检索)
                    </span>
                  </label>
                  <select
                    value={embeddingModel}
                    onChange={(e) => setEmbeddingModel(e.target.value)}
                    className="w-full p-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                  >
                    {Object.keys(availableEmbeddingModels).length > 0 ? (
                      <>
                        {/* 本地模型组 */}
                        <optgroup label="🏠 本地模型 (免费, 无需API Key)">
                          {Object.entries(availableEmbeddingModels)
                            .filter(([k, v]) => v.provider === 'local')
                            .map(([k, v]) => (
                              <option key={k} value={k}>
                                {v.name} - {v.description}
                              </option>
                            ))}
                        </optgroup>

                        {/* OpenAI及兼容API模型组 */}
                        <optgroup label="☁️ 云端API模型 (需要API Key)">
                          {Object.entries(availableEmbeddingModels)
                            .filter(([k, v]) => v.provider === 'openai')
                            .map(([k, v]) => (
                              <option key={k} value={k}>
                                {v.name} - {v.price} - {v.description}
                              </option>
                            ))}
                        </optgroup>
                      </>
                    ) : (
                      <option value="">加载中...</option>
                    )}
                  </select>

                  {/* API Key 提示 */}
                  {availableEmbeddingModels[embeddingModel]?.provider === 'openai' && (
                    <div className="mt-2 p-2 bg-blue-50 rounded-lg border border-blue-200">
                      <p className="text-xs text-blue-700">
                        💡 <strong>API Key说明：</strong>
                        {embeddingModel.startsWith('text-embedding-3') && ' 使用上方的OpenAI API Key'}
                        {embeddingModel.startsWith('text-embedding-v3') && ' 需要阿里云DashScope API Key (通义千问)'}
                        {embeddingModel.startsWith('moonshot') && ' 需要Moonshot AI API Key (Kimi)'}
                        {embeddingModel.startsWith('deepseek') && ' 需要DeepSeek API Key'}
                        {embeddingModel.startsWith('glm') && ' 需要智谱AI API Key (ChatGLM)'}
                        {embeddingModel.startsWith('minimax') && ' 需要MiniMax API Key'}
                        {embeddingModel.startsWith('BAAI') && ' 需要SiliconFlow API Key'}
                        {!embeddingModel.startsWith('text-embedding-3') &&
                         !embeddingModel.startsWith('text-embedding-v3') &&
                         !embeddingModel.startsWith('moonshot') &&
                         !embeddingModel.startsWith('deepseek') &&
                         !embeddingModel.startsWith('glm') &&
                         !embeddingModel.startsWith('minimax') &&
                         !embeddingModel.startsWith('BAAI') &&
                         ' 请输入对应提供商的API Key'}
                      </p>
                    </div>
                  )}

                  {availableEmbeddingModels[embeddingModel]?.provider === 'local' && (
                    <div className="mt-2 p-2 bg-green-50 rounded-lg border border-green-200">
                      <p className="text-xs text-green-700">
                        ✅ 本地模型，无需API Key，首次使用会自动下载模型文件
                      </p>
                    </div>
                  )}

                  <p className="text-xs text-gray-500 mt-2">
                    <strong>当前选择：</strong> {availableEmbeddingModels[embeddingModel]?.name || embeddingModel}
                    {availableEmbeddingModels[embeddingModel]?.dimension &&
                      ` | 向量维度: ${availableEmbeddingModels[embeddingModel].dimension}`}
                  </p>
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

                  <div className="mt-4">
                    <label className="block text-sm font-medium text-gray-700 mb-2">流式输出速度</label>
                    <select
                      value={streamSpeed}
                      onChange={(e) => setStreamSpeed(e.target.value)}
                      className="w-full p-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none"
                    >
                      <option value="fast">快速 ⚡ (3字符/次, ~20ms)</option>
                      <option value="normal">正常 ✨ (2字符/次, ~30ms)</option>
                      <option value="slow">慢速 🐢 (1字符/次, ~60ms)</option>
                      <option value="off">关闭流式（直接显示）</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">调整AI回复的打字机效果速度（已优化为按字符流式）</p>
                  </div>

                  <label className="flex items-center justify-between cursor-pointer p-2 hover:bg-gray-50 rounded-lg mt-3">
                    <span className="font-medium">Blur Reveal 效果</span>
                    <input type="checkbox" checked={enableBlurReveal} onChange={e => setEnableBlurReveal(e.target.checked)} className="accent-blue-600 w-5 h-5" />
                  </label>
                  <p className="text-xs text-gray-500 ml-2">流式输出时显示模糊到清晰的渐变效果</p>
                </div>

                {/* 存储位置信息 */}
                <div className="pt-4 border-t border-gray-100">
                  <h3 className="text-sm font-semibold text-gray-800 mb-3">📁 文件存储位置</h3>
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
                        💡 点击复制按钮复制路径，然后在{storageInfo.platform === 'Windows' ? '文件资源管理器' : storageInfo.platform === 'Darwin' ? 'Finder' : '文件管理器'}中打开
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
                  className="w-full py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors"
                >
                  Save Changes
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence >
    </div >
  );
};

export default ChatPDF;
