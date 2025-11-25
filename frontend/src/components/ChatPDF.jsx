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
  const [enableVectorSearch, setEnableVectorSearch] = useState(localStorage.getItem('enableVectorSearch') === 'true');
  const [enableScreenshot, setEnableScreenshot] = useState(localStorage.getItem('enableScreenshot') !== 'false');
  const [streamSpeed, setStreamSpeed] = useState(localStorage.getItem('streamSpeed') || 'normal'); // fast, normal, slow, off
  const [streamingMessageId, setStreamingMessageId] = useState(null);

  // Refs
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const pdfContainerRef = useRef(null);
  const selectionStartRef = useRef(null);

  const API_BASE_URL = ''; // Relative path due to proxy

  // Constants
  const VISION_MODELS = {
    'openai': ['gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini'],
    'anthropic': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'claude-sonnet-4-5-20250929'],
    'gemini': ['gemini-pro-vision', 'gemini-2.5-pro', 'gemini-2.5-flash-preview-09-2025'],
    'grok': ['grok-4.1', 'grok-vision-beta'],
    'ollama': ['llava']
  };

  // Effects
  useEffect(() => {
    fetchAvailableModels();
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
    localStorage.setItem('enableVectorSearch', enableVectorSearch);
    localStorage.setItem('enableScreenshot', enableScreenshot);
    localStorage.setItem('streamSpeed', streamSpeed);
  }, [apiKey, apiProvider, model, enableVectorSearch, enableScreenshot, streamSpeed]);

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

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('🔵 Uploading file:', file.name);
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

      // Add to history
      setHistory(prev => [{ id: data.doc_id, title: data.filename, date: new Date().toLocaleDateString() }, ...prev]);

    } catch (error) {
      console.error('❌ Upload error:', error);
      alert('上传失败: ' + error.message);
    } finally {
      setIsUploading(false);
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
          // Client-side streaming effect
          const words = fullAnswer.split(' ');
          let currentText = '';

          const speedConfig = {
            fast: { baseDelay: 10, variation: 10 },
            normal: { baseDelay: 20, variation: 20 },
            slow: { baseDelay: 40, variation: 30 }
          };

          const { baseDelay, variation } = speedConfig[streamSpeed] || speedConfig.normal;

          for (let i = 0; i < words.length; i++) {
            currentText += (i > 0 ? ' ' : '') + words[i];
            const delay = baseDelay + Math.random() * variation;
            await new Promise(resolve => setTimeout(resolve, delay));

            setMessages(prev => prev.map(msg =>
              msg.id === tempMsgId
                ? { ...msg, content: currentText }
                : msg
            ));
          }

          // Mark streaming complete
          setMessages(prev => prev.map(msg =>
            msg.id === tempMsgId
              ? { ...msg, isStreaming: false }
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
      // 加载文档信息
      const docResponse = await fetch(`${API_BASE_URL}/document/${session.docId}?t=${new Date().getTime()}`);
      if (!docResponse.ok) {
        alert('文档已不存在');
        return;
      }

      const docData = await docResponse.json();

      // 恢复会话状态
      setDocId(session.docId);
      setDocInfo(docData);
      setMessages(session.messages || []);
      setCurrentPage(1);
    } catch (error) {
      console.error('Failed to load session:', error);
      alert('加载会话失败: ' + error.message);
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
      <AnimatePresence mode="wait">
        {showSidebar && (
          <motion.div
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -300, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className={`w-72 flex-shrink-0 glass-panel border-r border-white/40 flex flex-col z-20 ${darkMode ? 'bg-gray-800/80 border-gray-700' : 'bg-white/60'}`}
          >
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
              <button onClick={() => setShowSettings(true)} className="flex items-center gap-3 w-full p-3 rounded-xl hover:bg-white/50 transition-colors text-sm font-medium">
                <Settings className="w-5 h-5" />
                <span>设置 & API Key</span>
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

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
                  className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
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
                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      <ReactMarkdown
                        remarkPlugins={[remarkMath]}
                        rehypePlugins={[rehypeKatex, rehypeHighlight]}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                    {msg.model && <div className="text-xs text-gray-400 mt-2">Model: {msg.model}</div>}
                  </div>
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

      {/* Settings Modal */}
      < AnimatePresence >
        {showSettings && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-3xl shadow-2xl p-8 w-[500px] max-w-[90vw]"
            >
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800">Settings</h2>
                <button onClick={() => setShowSettings(false)} className="p-2 hover:bg-gray-100 rounded-full"><X className="w-5 h-5" /></button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">API Provider</label>
                  <select
                    value={apiProvider}
                    onChange={(e) => setApiProvider(e.target.value)}
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
                      <option value="fast">快速 ⚡</option>
                      <option value="normal">正常 ✨</option>
                      <option value="slow">慢速 🐢</option>
                      <option value="off">关闭流式（直接显示）</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">调整AI回复的打字机效果速度</p>
                  </div>
                </div>
              </div>

              <button
                onClick={() => setShowSettings(false)}
                className="w-full mt-8 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors"
              >
                Save Changes
              </button>
            </motion.div>
          </div>
        )}
      </AnimatePresence >
    </div >
  );
};

export default ChatPDF;
