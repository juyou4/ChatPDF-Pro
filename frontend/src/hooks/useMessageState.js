import { useState, useRef, useEffect, useCallback } from 'react';
import { useSmoothStream } from './useSmoothStream';
import { useWebSearch } from '../contexts/WebSearchContext';
import { INLINE_CITATION_REGEX } from '../utils/citationUtils';

// API base URL
const API_BASE_URL = '';
export const STREAM_FIRST_EVENT_TIMEOUT_MS = 15000;

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

const tokenizeForCitation = (text = '') => {
  const lowered = String(text).toLowerCase();
  const tokens = lowered.match(/[a-z0-9]+|[\u4e00-\u9fff]/g);
  return tokens || [];
};

const calcTokenOverlap = (left, right) => {
  if (!left.length || !right.length) return 0;
  const rightSet = new Set(right);
  let score = 0;
  for (const token of left) {
    if (rightSet.has(token)) score += 1;
  }
  return score;
};

const normalizeCitationRecords = (citations = []) => {
  if (!Array.isArray(citations)) return [];
  const normalized = [];
  for (const c of citations) {
    const ref = Number(c?.ref);
    if (!Number.isFinite(ref)) continue;
    normalized.push({ ...c, ref });
  }
  return normalized;
};

export const extractInlineCitationRefs = (content = '') => {
  if (!content) return [];
  const refs = [];
  const seen = new Set();
  for (const m of String(content).matchAll(INLINE_CITATION_REGEX)) {
    const ref = Number(m[1] || m[2]);
    if (!Number.isFinite(ref) || seen.has(ref)) continue;
    seen.add(ref);
    refs.push(ref);
  }
  return refs;
};

const stripInlineCitations = (text = '') =>
  String(text).replace(INLINE_CITATION_REGEX, '').replace(/[ \t]{2,}/g, ' ').trim();

const attachRefsToSentence = (sentence, refs) => {
  if (!sentence || !refs || refs.length === 0) return sentence;
  const refText = refs.map((r) => `[${r}]`).join('');
  const trimmed = sentence.trimEnd();
  const tail = trimmed.match(/([。！？!?；;])$/);
  if (tail) {
    return `${trimmed.slice(0, -1)}${refText}${tail[1]}`;
  }
  return `${trimmed}${refText}`;
};

const calcCitationSupportScore = (sentence = '', citation = null) => {
  if (!sentence || !citation) return 0;
  const sentenceTokens = tokenizeForCitation(sentence);
  if (sentenceTokens.length === 0) return 0;

  const supportText = `${citation.highlight_text || ''} ${citation.group_id || ''}`.trim();
  const citationTokens = tokenizeForCitation(supportText);
  const overlap = calcTokenOverlap(sentenceTokens, citationTokens);
  let score = overlap / Math.max(1, sentenceTokens.length);

  const snippet = String(citation.highlight_text || '').replace(/\s+/g, '').slice(0, 24);
  if (snippet.length >= 6) {
    const compactSentence = String(sentence).replace(/\s+/g, '');
    if (compactSentence.includes(snippet)) {
      score += 0.25;
    } else if (compactSentence.includes(snippet.slice(0, Math.min(10, snippet.length)))) {
      score += 0.1;
    }
  }

  return score;
};

const optimizeSentenceCitations = (sentence, citations) => {
  const refsInSentence = [];
  for (const m of String(sentence).matchAll(INLINE_CITATION_REGEX)) {
    const ref = Number(m[1] || m[2]);
    if (Number.isFinite(ref)) refsInSentence.push(ref);
  }
  if (refsInSentence.length === 0) return sentence;

  const normalized = normalizeCitationRecords(citations);
  if (normalized.length === 0) return stripInlineCitations(sentence);

  const coreSentence = stripInlineCitations(sentence);
  if (!coreSentence) return sentence;

  const citationMap = new Map(normalized.map((c) => [c.ref, c]));
  const scoredAll = normalized
    .map((c) => ({ ref: c.ref, score: calcCitationSupportScore(coreSentence, c) }))
    .sort((a, b) => b.score - a.score);

  const scoredCurrent = [...new Set(refsInSentence)].map((ref) => ({
    ref,
    score: calcCitationSupportScore(coreSentence, citationMap.get(ref)),
  })).sort((a, b) => b.score - a.score);

  const MIN_SUPPORT = 0.08;
  const MIN_REPLACE = 0.14;

  let chosen = scoredCurrent.filter((x) => x.score >= MIN_SUPPORT).map((x) => x.ref);

  if (chosen.length === 0) {
    const better = scoredAll.filter((x) => x.score >= MIN_REPLACE).slice(0, 2).map((x) => x.ref);
    if (better.length > 0) chosen = better;
  }

  if (chosen.length === 0 && scoredCurrent.length > 0 && scoredCurrent[0].score >= 0.02) {
    chosen = [scoredCurrent[0].ref];
  }

  chosen = [...new Set(chosen)].slice(0, 2);
  if (chosen.length === 0) {
    // verifier: 句子没有可支撑来源时移除错误引用
    return coreSentence;
  }

  return attachRefsToSentence(coreSentence, chosen);
};

export const optimizeAssistantInlineCitations = (content, citations) => {
  if (!content || !Array.isArray(citations) || citations.length === 0) return content;

  const lines = String(content).split('\n');
  let inCodeFence = false;
  const optimized = lines.map((line) => {
    if (/^\s*```/.test(line)) {
      inCodeFence = !inCodeFence;
      return line;
    }
    if (inCodeFence) return line;
    return optimizeSentenceCitations(line, citations);
  });

  return optimized.join('\n');
};

export const filterCitationsByContentRefs = (content, citations) => {
  const normalized = normalizeCitationRecords(citations);
  if (normalized.length === 0) return [];

  const refs = extractInlineCitationRefs(content);
  if (refs.length === 0) return normalized;

  const cmap = new Map(normalized.map((c) => [c.ref, c]));
  return refs.filter((r) => cmap.has(r)).map((r) => cmap.get(r));
};

export const normalizeAssistantCitations = (content, citations) => {
  if (!content || !Array.isArray(citations) || citations.length <= 1) return content;

  const refRegex = /(?<!!)(\[(\d{1,3})\](?!\()|【(\d{1,3})】)/g;
  const refsInText = [...String(content).matchAll(refRegex)].map(m => Number(m[2] || m[3]));
  const uniqueRefs = new Set(refsInText);
  if (uniqueRefs.size !== 1) return content;

  const paragraphs = String(content).split(/\n{2,}/);
  const normalized = paragraphs.map((paragraph) => {
    refRegex.lastIndex = 0;
    if (!refRegex.test(paragraph)) return paragraph;
    refRegex.lastIndex = 0;

    const paraTokens = tokenizeForCitation(paragraph);
    let bestRef = Number([...uniqueRefs][0]);
    let bestScore = -1;

    for (const c of citations) {
      const ref = Number(c?.ref);
      if (!Number.isFinite(ref)) continue;
      const citationTokens = tokenizeForCitation(c?.highlight_text || '');
      const score = calcTokenOverlap(paraTokens, citationTokens);
      if (score > bestScore) {
        bestScore = score;
        bestRef = ref;
      }
    }

    return paragraph.replace(refRegex, `[${bestRef}]`);
  });

  return normalized.join('\n\n');
};

export const ensureAssistantInlineCitationFallback = (content, citations) => {
  if (!content || !Array.isArray(citations) || citations.length === 0) return content;

  const hasInlineRefs = /(?<!!)(\[(\d{1,3})\](?!\()|【(\d{1,3})】)/.test(String(content));
  if (hasInlineRefs) return content;

  const refs = [];
  const seen = new Set();
  for (const c of citations) {
    const ref = Number(c?.ref);
    if (!Number.isFinite(ref) || seen.has(ref)) continue;
    seen.add(ref);
    refs.push(ref);
  }
  if (refs.length === 0) return content;

  const tailRefs = refs.slice(0, 2).map((r) => `[${r}]`).join('');
  return `${String(content).trimEnd()}\n\n参考来源：${tailRefs}`;
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
 * @param {boolean} options.enableBlurReveal - 是否启用 Blur Reveal 动画
 * @param {string} options.blurIntensity - Blur Reveal 强度（light|medium|strong）
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
  enableGraphRAG = false,
  enableJiebaBM25 = true,
  numExpandContextChunk = 1,
  enableBlurReveal = false,
  blurIntensity = 'medium',
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
  const streamMaxRelevanceRef = useRef(null);
  const streamFollowupRef = useRef(null);
  const streamQaScoreRef = useRef(null);
  const streamConvNameRef = useRef(null);
  const streamMindmapRef = useRef(null);
  const streamWebSearchRef = useRef(null);
  const activeStreamMsgIdRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // ========== 从全局设置中解构对话参数 ==========
  const {
    maxTokens, temperature, topP, contextCount, streamOutput,
    enableTemperature, enableTopP, enableMaxTokens,
    customParams, reasoningEffort, answerDetailLevel,
    enableMemory,
  } = globalSettings;

  const { enableWebSearch, webSearchProvider, webSearchApiKey } = useWebSearch();

  // ========== 流式输出 Hook（ref 直写模式，需求 4.2） ==========
  // 流式输出期间不调用 setMessages，通过 contentRef 直接更新 DOM
  // 流结束后通过 getFinalText() 一次性同步到 React 状态
  const contentStream = useSmoothStream({
    streamDone: contentStreamDone,
    enableBlurReveal,
    blurIntensity,
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
      answer_detail: answerDetailLevel || 'standard',
      max_tokens: enableMaxTokens ? maxTokens : null,
      temperature: enableTemperature ? temperature : null,
      top_p: enableTopP ? topP : null,
      stream_output: streamOutput,
      enable_vector_search: enableVectorSearch,
      enable_graphrag: enableGraphRAG,
      enable_jieba_bm25: enableJiebaBM25,
      num_expand_context_chunk: numExpandContextChunk,
      chat_history: chatHistory.length > 0 ? chatHistory : null,
      custom_params: customParams?.length > 0
        ? Object.fromEntries(customParams.filter(p => p.name).map(p => [p.name, p.value]))
        : null,
      enable_memory: enableMemory,
      enable_web_search: enableWebSearch,
      web_search_provider: webSearchProvider,
      web_search_api_key: webSearchApiKey || null,
    };

    // 中止之前的请求
    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();
    streamingAbortRef.current.cancelled = false;
    streamCitationsRef.current = null;
    streamMaxRelevanceRef.current = null;
    streamFollowupRef.current = null;
    streamQaScoreRef.current = null;
    streamConvNameRef.current = null;
    streamMindmapRef.current = null;
    streamWebSearchRef.current = null;

    // 创建临时助手消息
    const tempMsgId = Date.now();
    setStreamingMessageId(tempMsgId);
    setMessages(prev => [...prev, {
      id: tempMsgId, type: 'assistant', content: '', model: chatModel,
      isStreaming: true, thinking: '', thinkingMs: 0,
    }]);

    // 每次发送前重置流式状态，确保 rAF 循环重启且无残留数据
    setContentStreamDone(false);
    setThinkingStreamDone(false);
    contentStream.reset('');
    thinkingStream.reset('');

    let firstEventTimeoutTriggered = false;
    try {
      if (streamSpeed !== 'off' && streamOutput) {
        // ===== 流式输出模式 =====
        activeStreamMsgIdRef.current = tempMsgId;

        let firstEventReceived = false;
        let firstEventTimer = null;
        const clearFirstEventTimer = () => {
          if (firstEventTimer) {
            clearTimeout(firstEventTimer);
            firstEventTimer = null;
          }
        };
        const markFirstEventReceived = () => {
          if (!firstEventReceived) {
            firstEventReceived = true;
            clearFirstEventTimer();
          }
        };
        firstEventTimer = setTimeout(() => {
          if (firstEventReceived || streamingAbortRef.current.cancelled) return;
          firstEventTimeoutTriggered = true;
          streamingAbortRef.current.cancelled = true;
          abortControllerRef.current?.abort();
        }, STREAM_FIRST_EVENT_TIMEOUT_MS);

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
          clearFirstEventTimer();
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
          markFirstEventReceived();
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
            if (p.type === 'web_search') {
              streamWebSearchRef.current = p.sources || [];
              return;
            }
            if (p.type === 'followup_questions') {
              streamFollowupRef.current = p.questions || [];
              return;
            }
            if (p.type === 'conv_name') {
              streamConvNameRef.current = p.name || null;
              return;
            }
            if (p.type === 'mindmap') {
              streamMindmapRef.current = p.markdown || null;
              return;
            }
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
              if (p.retrieval_meta?.max_relevance_score !== undefined) streamMaxRelevanceRef.current = p.retrieval_meta.max_relevance_score;
              if (p.qa_score !== undefined) streamQaScoreRef.current = p.qa_score;
              if (p.web_search_sources) streamWebSearchRef.current = p.web_search_sources;
              if (ct) { currentThinking += ct; thinkingStream.addChunk(ct); }
              sseDone = true;
            }
          } catch (e) {
            console.error(e, data);
          }
        };

        // 读取流数据
        let reading = true;
        while (reading) {
          const { value, done } = await reader.read();
          if (done || streamingAbortRef.current.cancelled) break;
          sseBuffer += decoder.decode(value, { stream: true });
          let parsing = true;
          while (parsing) {
            const { index: si, length: sl } = findSseSeparator(sseBuffer);
            if (si === -1) {
              parsing = false;
              continue;
            }
            const re = sseBuffer.slice(0, si);
            sseBuffer = sseBuffer.slice(si + sl);
            if (re.trim()) processSseEvent(re.trim());
            if (sseDone) {
              parsing = false;
              reading = false;
            }
          }
          if (sseDone) reading = false;
        }
        if (!sseDone && sseBuffer.trim()) processSseEvent(sseBuffer.trim());
        clearFirstEventTimer();

        // 流结束，等待一帧让 rAF 渲染循环处理剩余字符，再同步最终状态
        await new Promise(r => requestAnimationFrame(r));
        setContentStreamDone(true);
        setThinkingStreamDone(true);
        const finalThinkingMs = thinkingStartTime
          ? (thinkingEndTime || Date.now()) - thinkingStartTime : 0;
        const finalContent = currentText || (currentThinking ? '' : '⚠️ AI未返回内容');
        const normalizedFinalContent = normalizeAssistantCitations(finalContent, streamCitationsRef.current);
        const optimizedFinalContent = optimizeAssistantInlineCitations(
          normalizedFinalContent,
          streamCitationsRef.current
        );
        const finalContentWithInlineFallback = ensureAssistantInlineCitationFallback(
          optimizedFinalContent,
          streamCitationsRef.current
        );
        const finalCitations = filterCitationsByContentRefs(
          finalContentWithInlineFallback,
          streamCitationsRef.current
        );
        setMessages(prev => prev.map(m =>
          m.id === tempMsgId
            ? { ...m, content: finalContentWithInlineFallback, thinking: currentThinking, isStreaming: false, thinkingMs: finalThinkingMs, citations: finalCitations, maxRelevanceScore: streamMaxRelevanceRef.current, qaScore: streamQaScoreRef.current, followupQuestions: streamFollowupRef.current || null, convName: streamConvNameRef.current || null, mindmapMarkdown: streamMindmapRef.current || null, webSearchSources: streamWebSearchRef.current || null }
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
        const normalizedAnswer = normalizeAssistantCitations(data.answer, data.retrieval_meta?.citations);
        const optimizedAnswer = optimizeAssistantInlineCitations(
          normalizedAnswer,
          data.retrieval_meta?.citations
        );
        const answerWithInlineFallback = ensureAssistantInlineCitationFallback(
          optimizedAnswer,
          data.retrieval_meta?.citations
        );
        const finalCitations = filterCitationsByContentRefs(
          answerWithInlineFallback,
          data.retrieval_meta?.citations
        );
        setLastCallInfo({ provider: data.used_provider, model: data.used_model, fallback: data.fallback_used });
        setMessages(prev => prev.map(m =>
          m.id === tempMsgId
            ? { ...m, content: answerWithInlineFallback, thinking: data.reasoning_content || '', isStreaming: false, citations: finalCitations, webSearchSources: data.web_search_sources || null }
            : m
        ));
        setStreamingMessageId(null);
      }
    } catch (error) {
      if (error.name === 'AbortError' && !firstEventTimeoutTriggered) return;
      const errorMessage = firstEventTimeoutTriggered
        ? `首包超时（${STREAM_FIRST_EVENT_TIMEOUT_MS}ms），请重试或切换模型`
        : error.message;
      setContentStreamDone(true);
      setThinkingStreamDone(true);
      activeStreamMsgIdRef.current = null;
      setStreamingMessageId(null);
      setMessages(prev => prev.map(m =>
        m.id === tempMsgId
          ? { ...m, content: '❌ ' + errorMessage, isStreaming: false }
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
    reasoningEffort, answerDetailLevel, enableMemory,
    enableWebSearch, webSearchProvider, webSearchApiKey,
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
