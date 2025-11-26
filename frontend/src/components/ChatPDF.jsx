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
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState({});
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
    'openai': ['gpt-5.1-2025-11-13', 'gpt-4.1', 'gpt-5-nano', 'o4-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini'],
    'anthropic': ['claude-sonnet-4-5-20250929', 'claude-opus-4-1-20250805', 'claude-haiku-4-5-20250219', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
    'gemini': ['gemini-2.5-pro', 'gemini-2.5-flash-preview-09-2025', 'gemini-2.5-flash-lite-preview-09-2025', 'gemini-2.0-flash', 'gemini-pro-vision'],
    'grok': ['grok-4.1', 'grok-4.1-fast', 'grok-3', 'grok-vision-beta'],
    'doubao': ['doubao-1.5-pro-256k', 'doubao-1.5-pro-32k'],
    'qwen': ['qwen-max-2025-01-25', 'qwen3-235b-a22b-instruct-2507', 'qwen3-coder-plus-2025-09-23'],
                  < label className = "block text-sm font-medium text-gray-700 mb-1" > API Key</label>
    <input
      type="password"
      value={apiKey}
      onChange={(e) => setApiKey(e.target.value)}
                  </label >

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
                </div >
              </div >

  <button
    onClick={() => setShowSettings(false)}
    className="w-full mt-8 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors"
  >
    Save Changes
  </button>
            </motion.div >
    </div >
  )
}
      </AnimatePresence >
    </div >
  );
};

export default ChatPDF;
