import { useState, useRef, useEffect, useCallback } from 'react';
import { useSmoothStream } from './useSmoothStream';

// API base URL
const API_BASE_URL = '';

/**
 * 构建聊天历史记录
 * 过滤无效消息，取最近 contextCount*2 条作为上下文
 *
 * @param {Array} messages - 消息列表
 * @param {number} contextCount - 上下文轮数
 * @returns {Array} 格式化后的聊天历史
 */
export const buildChatHistory = (messages, contextCount) => {
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

/**
 * 消息状态管理 Hook
 * 管理消息列表、流式输出、历史记录等状态和逻辑
 *
 * @param {Object} options - 配置选项
 * @param {string|null} options.docId - 当前文档 ID
 * @param {Array} options.screenshots - 截图列表
 * @param {Function} options.setScreenshots - 设置截图列表
 * @param {string} options.selectedText - 当前选中的文本
 * @param {Function} options.getChatCredentials - 获取聊天凭证
 * @param {Function} options.getCurrentChatModel - 获取当前聊天模型
 * @param {Function} options.getProviderById - 根据 ID 获取 provider
 * @param {string} options.streamSpeed - 流式输出速度设置
 * @param {boolean} options.enableVectorSearch - 是否启用向量搜索
 * @param {Object} options.globalSettings - 全局设置（来自 useGlobalSettings）
 */
export function useMessageState({
  docId = null,
  screenshots = [],
  setScreenshots,
  selectedText = '',
  getChatCredentials,
  getCurrentChatModel,
  getProviderById,
  streamSpeed = 'normal',
  enableVectorSearch = false,
  globalSettings = {},
} = {}) {
  // ========== 消息核心状态 ==========
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasInput, setHasInput] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState(null);
  const [lastCallInfo, setLastCallInfo] = useState(null);

  // 消息交互状态
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [likedMessages, setLikedMessages] = useState(new Set());
  const [rememberedMessages, setRememberedMessages] = useState(new Set());

  // 流式输出控制状态
  const [contentStreamDone, setContentStreamDone] = useState(false);
  const [thinkingStreamDone, setThinkingStreamDone] = useState(false);

  // ========== Refs ==========
  const abortControllerRef = useRef(null);
  const streamingAbortRef = useRef({ cancelled: false });
  const streamCitationsRef = useRef(null);
  const activeStreamMsgIdRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // ========== 从全局设置中解构对话参数 ==========
  const {
    maxTokens, temperature, topP, contextCount, streamOutput,
    enableTemperature, enableTopP, enableMaxTokens,
    customParams, reasoningEffort,
    enableMemory,
  } = globalSettings;

  // ========== 流式输出 Hook（ref 直写模式，需求 4.2） ==========
  // 流式输出期间不调用 setMessages，通过 contentRef 直接更新 DOM
  // 流结束后通过 getFinalText() 一次性同步到 React 状态
  const contentStream = useSmoothStream({
    streamDone: contentStreamDone,
  });

  const thinkingStream = useSmoothStream({
    streamDone: thinkingStreamDone,
  });

  // ========== 副作用 ==========

  // 消息变化时自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ========== 方法 ==========

  /**
   * 设置输入框的值并同步 hasInput 状态
   * @param {string} val - 输入值
   */
  const setInputValue = useCallback((val) => {
    if (textareaRef.current) {
      textareaRef.current.value = val;
      textareaRef.current.style.height = '24px';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
    setHasInput(!!(val && val.trim()));
  }, []);

  /**
   * 发送消息
   * 处理用户输入、构建请求体、发起流式/非流式请求
   */
  const sendMessage = useCallback(async () => {
    const currentInput = textareaRef.current?.value ?? '';
    if (!currentInput.trim() && screenshots.length === 0) return;

    const { providerId: chatProvider, modelId: chatModel, apiKey: chatApiKey } = getChatCredentials?.() || {};
    if (!docId) { alert('请先上传文档'); return; }
    if (!chatApiKey && chatProvider !== 'ollama' && chatProvider !== 'local') {
      alert('请先配置API Key\n\n请点击左下角"设置 & API Key"按钮进行配置');
      return;
    }

    // 构建用户消息
    const userMsg = { type: 'user', content: currentInput, hasImage: screenshots.length > 0 };
    setMessages(prev => [...prev, userMsg]);

    // 清空输入框
    if (textareaRef.current) {
      textareaRef.current.value = '';
      textareaRef.current.style.height = '24px';
    }
    setHasInput(false);
    setIsLoading(true);

    // 构建聊天历史
    const chatHistory = buildChatHistory(messages, contextCount);

    // 获取 provider 完整信息
    const chatProviderFull = getProviderById?.(chatProvider);

    // 构建请求体
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
      custom_params: customParams?.length > 0
        ? Object.fromEntries(customParams.filter(p => p.name).map(p => [p.name, p.value]))
        : null,
      enable_memory: enableMemory,
    };

    // 中止之前的请求
    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();
    streamingAbortRef.current.cancelled = false;
    streamCitationsRef.current = null;

    // 创建临时助手消息
    const tempMsgId = Date.now();
    setStreamingMessageId(tempMsgId);
    setMessages(prev => [...prev, {
      id: tempMsgId, type: 'assistant', content: '', model: chatModel,
      isStreaming: true, thinking: '', thinkingMs: 0,
    }]);

    try {
      if (streamSpeed !== 'off' && streamOutput) {
        // ===== 流式输出模式 =====
        activeStreamMsgIdRef.current = tempMsgId;
        setContentStreamDone(false);
        setThinkingStreamDone(false);
        contentStream.reset('');
        thinkingStream.reset('');

        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          let ed = `HTTP ${response.status}`;
          try {
            const eb = await response.json();
            ed = eb.detail || eb.error?.message || eb.message || JSON.stringify(eb);
          } catch (e) { /* ignore */ }
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

        // SSE 分隔符查找
        const findSseSeparator = (buf) => {
          const lf = buf.indexOf('\n\n');
          const crlf = buf.indexOf('\r\n\r\n');
          if (lf === -1 && crlf === -1) return { index: -1, length: 0 };
          if (lf === -1) return { index: crlf, length: 4 };
          if (crlf === -1) return { index: lf, length: 2 };
          return lf < crlf ? { index: lf, length: 2 } : { index: crlf, length: 4 };
        };

        // SSE 事件处理
        const processSseEvent = (et) => {
          const lines = et.split(/\r?\n/);
          const dl = [];
          for (const ln of lines) {
            if (ln.trim().startsWith('data:')) dl.push(ln.trim().slice(5).trimStart());
          }
          if (dl.length === 0) return;
          const data = dl.join('\n');
          if (data === '[DONE]') { sseDone = true; return; }
          try {
            const p = JSON.parse(data);
            if (p.error) {
              const em = `❌ ${p.error}`;
              currentText = em;
              contentStream.addChunk(em);
              sseDone = true;
              return;
            }
            if (p.type === 'retrieval_progress') return;
            const delta = p.choices?.[0]?.delta || {};
            const cc = delta.content || p.content || '';
            const ct = delta.reasoning_content || p.reasoning_content || '';
            if (!p.done && !p.choices?.[0]?.finish_reason) {
              if (cc) {
                currentText += cc;
                contentStream.addChunk(cc);
                if (thinkingStartTime && !thinkingEndTime) thinkingEndTime = Date.now();
              }
              if (ct) {
                if (!thinkingStartTime) thinkingStartTime = Date.now();
                currentThinking += ct;
                thinkingStream.addChunk(ct);
              }
            } else {
              if (p.retrieval_meta?.citations) streamCitationsRef.current = p.retrieval_meta.citations;
              if (ct) { currentThinking += ct; thinkingStream.addChunk(ct); }
              sseDone = true;
            }
          } catch (e) {
            console.error(e, data);
          }
        };

        // 读取流数据
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

        // 流结束，同步最终状态
        setContentStreamDone(true);
        setThinkingStreamDone(true);
        const finalThinkingMs = thinkingStartTime
          ? (thinkingEndTime || Date.now()) - thinkingStartTime : 0;
        const finalContent = currentText || (currentThinking ? '' : '⚠️ AI未返回内容');
        setMessages(prev => prev.map(m =>
          m.id === tempMsgId
            ? { ...m, content: finalContent, thinking: currentThinking, isStreaming: false, thinkingMs: finalThinkingMs, citations: streamCitationsRef.current }
            : m
        ));
        activeStreamMsgIdRef.current = null;
        setStreamingMessageId(null);
      } else {
        // ===== 非流式输出模式 =====
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          let ed = `HTTP ${response.status}`;
          try {
            const eb = await response.json();
            ed = eb.detail || eb.error?.message || eb.message || JSON.stringify(eb);
          } catch (e) { /* ignore */ }
          throw new Error(ed);
        }

        const data = await response.json();
        setLastCallInfo({ provider: data.used_provider, model: data.used_model, fallback: data.fallback_used });
        setMessages(prev => prev.map(m =>
          m.id === tempMsgId
            ? { ...m, content: data.answer, thinking: data.reasoning_content || '', isStreaming: false, citations: data.retrieval_meta?.citations }
            : m
        ));
        setStreamingMessageId(null);
      }
    } catch (error) {
      if (error.name === 'AbortError') return;
      setContentStreamDone(true);
      setThinkingStreamDone(true);
      activeStreamMsgIdRef.current = null;
      setStreamingMessageId(null);
      setMessages(prev => prev.map(m =>
        m.id === tempMsgId
          ? { ...m, content: '❌ ' + error.message, isStreaming: false }
          : m
      ));
    } finally {
      setIsLoading(false);
    }
  }, [
    docId, screenshots, selectedText, messages, streamSpeed, enableVectorSearch,
    getChatCredentials, getProviderById, contentStream, thinkingStream,
    maxTokens, temperature, topP, contextCount, streamOutput,
    enableTemperature, enableTopP, enableMaxTokens, customParams,
    reasoningEffort, enableMemory,
  ]);

  /**
   * 停止当前流式输出
   */
  const handleStop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
    }
    streamingAbortRef.current.cancelled = true;
    contentStream.reset('');
    thinkingStream.reset('');
    setContentStreamDone(false);
    setThinkingStreamDone(false);
    activeStreamMsgIdRef.current = null;
    if (streamingMessageId) {
      setMessages(prev => prev.map(m =>
        m.id === streamingMessageId ? { ...m, isStreaming: false } : m
      ));
    }
    setStreamingMessageId(null);
  }, [streamingMessageId, contentStream, thinkingStream]);

  /**
   * 重新生成指定位置的消息
   * @param {number} index - 消息索引
   */
  const regenerateMessage = useCallback(async (index) => {
    if (!docId) { alert('请先上传文档'); return; }
    const userMsg = messages.slice(0, index).reverse().find(m => m.type === 'user');
    if (!userMsg) return;
    setMessages(prev => prev.slice(0, index));
    setInputValue(userMsg.content);
    // 延迟发送，确保输入框已更新
    setTimeout(() => sendMessage(), 100);
  }, [docId, messages, setInputValue, sendMessage]);

  /**
   * 复制消息内容到剪贴板
   * @param {string} content - 消息内容
   * @param {*} messageId - 消息 ID
   */
  const copyMessage = useCallback((content, messageId) => {
    navigator.clipboard.writeText(content).then(() => {
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    });
  }, []);

  /**
   * 保存消息到记忆库
   * @param {number} index - 消息索引
   * @param {string} type - 保存类型（'liked' | 'remembered'）
   */
  const saveToMemory = useCallback(async (index, type) => {
    const m = messages[index];
    if (!m || m.type !== 'assistant') return;
    const um = messages.slice(0, index).reverse().find(x => x.type === 'user');
    const content = `Q: ${um ? um.content.slice(0, 100) : ''}\nA: ${m.content.slice(0, 200)}`;
    try {
      const res = await fetch('/api/memory/entries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, source_type: type, doc_id: docId }),
      });
      if (res.ok) {
        if (type === 'liked') setLikedMessages(p => new Set(p).add(index));
        else setRememberedMessages(p => new Set(p).add(index));
      }
    } catch (e) {
      // 静默处理
    }
  }, [messages, docId]);

  return {
    // 消息状态
    messages,
    setMessages,
    isLoading,
    setIsLoading,
    hasInput,
    setHasInput,
    streamingMessageId,
    lastCallInfo,
    setLastCallInfo,

    // 消息交互状态
    copiedMessageId,
    likedMessages,
    rememberedMessages,

    // 流式输出控制
    contentStreamDone,
    thinkingStreamDone,
    contentStream,
    thinkingStream,
    activeStreamMsgIdRef,
    // ref 直写模式：暴露 contentRef 供组件直接挂载 DOM 元素
    streamingContentRef: contentStream.contentRef,
    streamingThinkingRef: thinkingStream.contentRef,

    // Refs
    abortControllerRef,
    messagesEndRef,
    textareaRef,

    // 方法
    sendMessage,
    handleStop,
    regenerateMessage,
    copyMessage,
    saveToMemory,
    setInputValue,
  };
}
