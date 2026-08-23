import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useSmoothStream } from './useSmoothStream';
import { useWebSearch } from '../contexts/WebSearchContext';
import {
  getInlineCitationMatches,
  hasInlineCitationRefs,
  replaceInlineCitationRefs,
} from '../utils/citationUtils';
import { buildChatHistory, isFailedChatHistoryAssistant } from '../utils/chatContextUsageUtils';
import { normalizeChatVisualAttachments } from '../utils/visualAttachmentUtils';
import { requiresPreservedReasoning } from '../services/reasoningEffortService';
import {
  cloneAgentTrace,
  publishLiveAgentTrace,
  shouldCommitAgentTrace,
} from '../utils/agentTraceLive';

// API base URL
// Web 开发模式下绕过 Vite /chat 代理，避免 SSE 被 dev proxy 缓冲后“最后一股脑显示”。
// Electron 桌面模式保持相对路径，由 config/desktop 注入真实 backend URL 和鉴权 token。
const API_BASE_URL = (() => {
  const isDesktop = typeof window !== 'undefined' && window.chatpdfDesktop?.isDesktop === true;
  if (!isDesktop && typeof import.meta !== 'undefined' && import.meta.env?.DEV) {
    return 'http://127.0.0.1:8000';
  }
  return '';
})();
export const STREAM_FIRST_EVENT_TIMEOUT_MS = 60000;
export const CHAT_PARSE_IDENTITY_UPDATED_MESSAGE = '文档解析结果已更新，请重新提问。';
export const EMBEDDING_IDENTITY_CONFLICT_MESSAGE = '当前 Embedding 配置与文档索引不一致';
const TABLE_VISUAL_VERIFICATION_POLL_INTERVAL_MS = 2000;
const TABLE_VISUAL_VERIFICATION_POLL_MAX_ATTEMPTS = 60;

const TABLE_VISUAL_PENDING_STATES = new Set(['queued', 'running', 'pending']);
// skipped 是可观测的终态：请求进入核验器，但按策略没有向模型发送图片。
const TABLE_VISUAL_TERMINAL_STATES = new Set(['confirmed', 'conflict', 'indeterminate', 'failed', 'stale', 'skipped']);
const CHAT_TURN_STATUSES = new Set([
  'completed',
  'recovered_retry',
  'evidence_fallback',
  'degraded',
  'truncated',
  'failed',
  'interrupted',
]);
const RETRY_CONTROL_INPUTS = new Set([
  '继续', '重新回答', '重新生成', '重答', '再回答一次', '再答一次',
  '请重新回答', '请重答', 'continue', 'retry', 'tryagain', 'regenerate',
]);

/**
 * 正文开始后每个 token 都会走到这里。状态没变就交回原对象，
 * 避免 setMessages 每字换一条新消息把整列重绘一遍。
 */
export function applyAnswerStartedFlags(message, { generating = false, hasVisibleAnswer = false } = {}) {
  const answerGenerating = Boolean(generating) && !hasVisibleAnswer;
  if (message?.answerStarted === true && message?.answerGenerating === answerGenerating) {
    return message;
  }
  return {
    ...message,
    answerStarted: true,
    answerGenerating,
  };
}
const CHAT_INTERACTION_MODES = new Set([
  'default', 'selection', 'image', 'preset', 'retry_failed_turn',
]);

const normalizeRetryControlInput = (value) =>
  String(value || '').replace(/[\s，。！？、,.!?;；:：]+/g, '').toLowerCase();

const normalizeChatTurnStatus = (value, fallback = '') => {
  const status = String(value || '').trim().toLowerCase();
  return CHAT_TURN_STATUSES.has(status) ? status : fallback;
};

const normalizeChatParseIdentity = (value) => {
  const nested = isPlainObject(value?.parse_identity) ? value.parse_identity : {};
  return {
    parseGeneration: String(
      value?.parse_generation
      || value?.parseGeneration
      || nested.parse_generation
      || nested.generation
      || ''
    ).trim(),
    documentSourceHash: String(
      value?.document_source_hash
      || value?.documentSourceHash
      || nested.document_source_hash
      || nested.source_hash
      || ''
    ).trim(),
  };
};

const hasCompleteChatParseIdentity = (identity) => Boolean(
  identity?.parseGeneration && identity?.documentSourceHash
);

const chatParseIdentitiesMatch = (left, right) => (
  hasCompleteChatParseIdentity(left)
  && hasCompleteChatParseIdentity(right)
  && left.parseGeneration === right.parseGeneration
  && left.documentSourceHash === right.documentSourceHash
);

const getChatIdentityFields = (identity) => (
  hasCompleteChatParseIdentity(identity)
    ? {
      parse_generation: identity.parseGeneration,
      document_source_hash: identity.documentSourceHash,
    }
    : {}
);

const messageMatchesChatParseIdentity = (message, identity) => {
  if (!hasCompleteChatParseIdentity(identity)) return message?.parseIdentityStale !== true;
  if (message?.parseIdentityStale === true) return false;
  return chatParseIdentitiesMatch(normalizeChatParseIdentity(message), identity);
};

const getResponseChatParseIdentity = (response, payload) => ({
  parseGeneration: String(
    payload?.parse_generation
    || response?.headers?.get?.('X-Chat-Parse-Generation')
    || ''
  ).trim(),
  documentSourceHash: String(
    payload?.document_source_hash
    || response?.headers?.get?.('X-Chat-Document-Source-Hash')
    || ''
  ).trim(),
});

const responseMatchesChatParseIdentity = (response, payload, expectedIdentity) => {
  const responseIdentity = getResponseChatParseIdentity(response, payload);
  if (!hasCompleteChatParseIdentity(expectedIdentity)) return true;
  return chatParseIdentitiesMatch(responseIdentity, expectedIdentity);
};

const responseHasAnyChatParseIdentity = (response, payload) => {
  const identity = getResponseChatParseIdentity(response, payload);
  return Boolean(identity.parseGeneration || identity.documentSourceHash);
};

const isChatParseIdentityConflict = (response, payload, expectedIdentity, message = '') => {
  if (!hasCompleteChatParseIdentity(expectedIdentity)) return false;
  if (/文档解析结果已(?:在回答期间)?更新|chat_parse_identity_changed/i.test(String(message || ''))) {
    return true;
  }
  const responseIdentity = getResponseChatParseIdentity(response, payload);
  if (hasCompleteChatParseIdentity(responseIdentity)) {
    return !chatParseIdentitiesMatch(responseIdentity, expectedIdentity);
  }
  return false;
};

export const isEmbeddingIdentityConflictDetail = (detail) => {
  const text = String(detail || '');
  return /当前\s*Embedding\s*配置与文档索引不一致|vector_embedding_identity_conflict/i.test(text);
};

const createEmbeddingIdentityConflictError = (detail = '') => {
  const error = new Error(EMBEDDING_IDENTITY_CONFLICT_MESSAGE);
  error.name = 'EmbeddingIdentityConflictError';
  error.detail = String(detail || '');
  return error;
};

const resolveRetryControlQuestion = (input, messages = []) => {
  if (!RETRY_CONTROL_INPUTS.has(normalizeRetryControlInput(input))) return '';
  for (let assistantIndex = messages.length - 1; assistantIndex >= 0; assistantIndex -= 1) {
    const assistant = messages[assistantIndex];
    if (assistant?.type !== 'assistant') continue;
    let user = null;
    for (let userIndex = assistantIndex - 1; userIndex >= 0; userIndex -= 1) {
      const candidateMessage = messages[userIndex];
      if (candidateMessage?.type === 'assistant') break;
      if (candidateMessage?.type === 'user') {
        user = candidateMessage;
        break;
      }
    }
    if (!user) return '';
    const candidate = String(user.content || '').trim();
    if (candidate && !RETRY_CONTROL_INPUTS.has(normalizeRetryControlInput(candidate))) {
      return isFailedChatHistoryAssistant(assistant) ? candidate : '';
    }
    return '';
  }
  return '';
};
const resolveClarificationTicket = (messages = []) => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.type !== 'assistant') continue;
    if (!message.clarificationRequired) return null;
    const ticket = message.intentDecision?.clarification_ticket;
    if (!ticket || typeof ticket !== 'object') return null;
    return {
      version: String(ticket.version || ''),
      ticket_id: String(ticket.ticket_id || ''),
      original_question: String(ticket.original_question || ''),
      parse_generation: String(ticket.parse_generation || ''),
      document_source_hash: String(ticket.document_source_hash || ''),
    };
  }
  return null;
};


const isPlainObject = (value) => Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const normalizeVisualVerificationState = (value) => {
  const state = String(value || '').trim().toLowerCase();
  if (TABLE_VISUAL_PENDING_STATES.has(state) || TABLE_VISUAL_TERMINAL_STATES.has(state)) {
    return state;
  }
  return '';
};

export const isVisualVerificationPending = (verification) =>
  TABLE_VISUAL_PENDING_STATES.has(normalizeVisualVerificationState(verification?.state));

export const isVisualVerificationTerminal = (verification) =>
  TABLE_VISUAL_TERMINAL_STATES.has(normalizeVisualVerificationState(verification?.state));

/**
 * 兼容聊天诊断和状态接口的轻量包装，统一为消息可直接消费的视觉核验状态。
 * 不主动推断未知结果，避免把非终态任务误标记为已确认。
 */
export const normalizeNumericTableVisualVerification = (payload) => {
  if (!isPlainObject(payload)) return null;

  const candidates = [
    payload.verification,
    payload.task,
    payload.data,
    payload.result,
    payload,
  ].filter(isPlainObject);
  const record = candidates.find((candidate) => {
    const state = normalizeVisualVerificationState(
      candidate.state || candidate.verdict || candidate.status || (candidate.pending ? 'pending' : '')
    );
    return Boolean(candidate.task_id || state);
  }) || payload;
  const diagnostics = isPlainObject(record.diagnostics) ? record.diagnostics : {};
  const state = [
    record.state,
    record.verdict,
    record.status,
    diagnostics.state,
    diagnostics.verdict,
    diagnostics.status,
    record.pending ? 'pending' : '',
  ].map(normalizeVisualVerificationState).find(Boolean) || '';

  if (!state && !record.task_id && !payload.task_id) return null;

  return {
    ...payload,
    ...record,
    ...diagnostics,
    task_id: record.task_id || payload.task_id || '',
    state,
  };
};

export const getNumericTableVisualVerification = (retrievalMeta) =>
  normalizeNumericTableVisualVerification(
    retrievalMeta?.diagnostics?.numeric_table_visual_verification
  );

const STREAM_RENDER_PROFILES = {
  // 这些值控制小队列的基础节奏；大队列由 useSmoothStream 自适应提速，
  // 不再让打字机动画落后于模型实际输出。
  fast: { minDelay: 16, frameChars: 4, flushChars: 10 },
  normal: { minDelay: 28, frameChars: 2, flushChars: 4 },
  slow: { minDelay: 48, frameChars: 1, flushChars: 2 },
};

// 思考区使用连续但不过快的独立节奏。大块 reasoning 仍由 useSmoothStream
// 自适应追赶，小分片则保留肉眼可见的逐步输出，避免 64 字/帧造成整段跳显。
const THINKING_STREAM_RENDER_PROFILE = {
  minDelay: 24,
  frameChars: 2,
  flushChars: 4,
};

// 终态包可能携带一整段正文。仅当平滑队列短时间没有任何渲染进展时
// 才退回一次性提交；500ms 足够覆盖正常绘制，也不会在后台标签页 / rAF
// 暂停时留下一个肉眼明显的空白等待。
export const STREAM_FINAL_FLUSH_GRACE_MS = 500;

export const resolveStreamRenderProfile = (streamSpeed = 'normal') =>
  STREAM_RENDER_PROFILES[streamSpeed] || STREAM_RENDER_PROFILES.normal;

export const resolveThinkingStreamProfile = () => THINKING_STREAM_RENDER_PROFILE;

// 这些 phase 由 AgentTracePanel 作为唯一展示入口。把它们同时写入
// thinking 文本会让同一条 SSE 进度在思考区和工具时间线各出现一次。
const AGENT_PHASES = new Set([
  'agent_start',
  'round_start',
  'planning',
  'planner_error',
  'executing',
  'tool_result',
  'tool_skipped',
  'complete',
  'agent_mode',
]);

const AGENT_ONLY_PHASES = new Set([
  'agent_start',
  'round_start',
  'planning',
  'planner_error',
  'executing',
  'tool_result',
  'tool_skipped',
  'agent_mode',
]);

// 部分 OpenAI 兼容服务会在真正思考内容前短暂输出聊天角色标记。该标记是
// 协议元数据而非可展示的思考；若把它当作首个 token，会清空检索过程并让
// 界面只剩“用户”。
const REASONING_ROLE_LABELS = new Set([
  'user',
  'assistant',
  'system',
  'developer',
  'tool',
  '用户',
  '助手',
  '系统',
  '开发者',
  '工具',
]);

const normalizeReasoningRoleLabel = (value) => String(value || '')
  .trim()
  .toLowerCase()
  .replace(/[：:，,、\-—\s]+$/g, '');

const consumeDisplayableReasoningDelta = (delta, pendingRoleLabel = '') => {
  const incoming = String(delta || '');
  const combined = `${pendingRoleLabel}${incoming}`;
  const normalizedCombined = normalizeReasoningRoleLabel(combined);

  // 暂存完整角色标记及其分片（例如“用”+“户”），直到收到真正的思考 token，
  // 既不暴露协议字段，也不会丢掉紧随其后的思考正文。
  if (
    normalizedCombined
    && [...REASONING_ROLE_LABELS].some((label) => label.startsWith(normalizedCombined))
  ) {
    return { text: '', pendingRoleLabel: combined.slice(-32) };
  }

  if (REASONING_ROLE_LABELS.has(normalizeReasoningRoleLabel(pendingRoleLabel))) {
    const text = incoming.replace(/^\s*[：:，,、\-—]*\s*/, '');
    if (!text.trim()) {
      return { text: '', pendingRoleLabel: combined.slice(-32) };
    }
    return { text, pendingRoleLabel: '' };
  }

  return { text: combined, pendingRoleLabel: '' };
};

const isAgentTraceProgressEvent = (payload, { agentTraceEnabled = false } = {}) => {
  if (payload?.type !== 'retrieval_progress') return false;
  const phase = String(payload.phase || '');
  return AGENT_ONLY_PHASES.has(phase)
    // `complete` is also used by the regular retrieval path. It belongs to
    // AgentTracePanel only after this request has actually entered Agent mode.
    || (agentTraceEnabled && phase === 'complete');
};

const isSkippedAgentToolEvent = (payload) => {
  if (!payload || typeof payload !== 'object') return false;
  if (payload.phase === 'tool_skipped' || payload.skipped === true) return true;

  // 兼容旧后端：它曾把“跳过重复检索”错误标为 tool_result。
  return payload.phase === 'tool_result'
    && /^跳过重复检索\s*[:：]/.test(String(payload.message || '').trim());
};

export const formatThinkingStageEvent = (payload, options = {}) => {
  if (!payload || typeof payload !== 'object') return null;

  if (isAgentTraceProgressEvent(payload, options)) return null;

  if (payload.type === 'retrieval_progress') {
    const rawMessage = typeof payload.message === 'string' && payload.message.trim()
      ? payload.message.trim()
      : (payload.phase === 'complete' ? '检索完成，正在组织上下文...' : '正在检索文档...');
    // 旧后端或第三方适配器可能仍把 total_ms 拼进 complete 文案。
    // 耗时已经在 payload.timings 中保留，不应进入用户可见的思考阶段文本。
    const message = payload.phase === 'complete'
      ? '检索完成，正在整理上下文...'
      : rawMessage;
    const stablePhaseKey = ['llm_waiting', 'llm_answer_waiting', 'llm_structuring_citations', 'answer_generating'].includes(payload.phase)
      ? `retrieval:${payload.phase}`
      : null;
    const keyParts = [payload.phase, payload.round, payload.step, message]
      .filter((part) => part !== undefined && part !== null && String(part).trim() !== '');
    return {
      key: stablePhaseKey || `retrieval:${keyParts.join(':')}`,
      text: message,
      phase: String(payload.phase || ''),
    };
  }

  return null;
};

export const upsertRetrievalProgressEntry = (entries, stage, limit = 32) => {
  if (!stage?.text) return Array.isArray(entries) ? entries : [];
  const current = Array.isArray(entries) ? entries : [];
  const key = String(stage.key || stage.text);
  const nextEntry = {
    key,
    text: String(stage.text).trim(),
    phase: String(stage.phase || ''),
  };
  const previousEntry = current.find((entry) => entry.key === key);
  if (
    previousEntry
    && previousEntry.text === nextEntry.text
    && previousEntry.phase === nextEntry.phase
  ) return current;
  return [
    ...current.filter((entry) => entry.key !== key),
    nextEntry,
  ].slice(-Math.max(1, Number(limit) || 32));
};

const extractSseReasoningText = (payload, delta) => {
  const candidates = [
    delta?.reasoning_content,
    delta?.reasoning,
    delta?.reasoning_text,
    payload?.reasoning_content,
    payload?.reasoning,
    payload?.reasoning_text,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate) return candidate;
    if (candidate && typeof candidate === 'object') {
      const text = candidate.text || candidate.content || candidate.thinking || '';
      if (typeof text === 'string' && text) return text;
    }
  }
  return '';
};

const extractSseVisibleContent = (payload, delta) => {
  const raw = delta?.content || payload?.content || '';
  if (typeof raw === 'string') return raw;
  if (!Array.isArray(raw)) {
    if (raw && typeof raw === 'object') {
      return String(raw.text || raw.content || '');
    }
    return '';
  }
  return raw.map((part) => {
    if (typeof part === 'string') return part;
    const partType = String(part?.type || '').toLowerCase();
    if (['thinking', 'reasoning', 'reasoning_text', 'thought'].includes(partType)) {
      return '';
    }
    return String(part?.text || part?.content || '');
  }).join('');
};

export { buildChatHistory } from '../utils/chatContextUsageUtils';

const tokenizeForCitation = (text = '') => {
  const lowered = String(text).toLowerCase();
  // 单字符 + 中文 bigram 以提升中文场景下 overlap 命中率
  const chars = lowered.match(/[a-z0-9]+|[\u4e00-\u9fff]/g) || [];
  const bigrams = [];
  for (let i = 0; i < chars.length - 1; i++) {
    if (chars[i].length === 1 && chars[i + 1]?.length === 1
        && /[\u4e00-\u9fff]/.test(chars[i]) && /[\u4e00-\u9fff]/.test(chars[i + 1])) {
      bigrams.push(chars[i] + chars[i + 1]);
    }
  }
  return [...chars, ...bigrams];
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

const resolveCitationDisplayRef = (citation) => {
  const displayRef = Number(citation?.display_ref);
  if (Number.isFinite(displayRef)) return displayRef;
  const ref = Number(citation?.ref);
  return Number.isFinite(ref) ? ref : null;
};

const normalizeCitationRecords = (citations = []) => {
  if (!Array.isArray(citations)) return [];
  const normalized = [];
  for (const c of citations) {
    const ref = resolveCitationDisplayRef(c);
    if (!Number.isFinite(ref)) continue;
    const sourceRef = Number(c?.source_ref);
    normalized.push({
      ...c,
      ref,
      display_ref: ref,
      source_ref: Number.isFinite(sourceRef) ? sourceRef : ref,
    });
  }
  return normalized.sort((a, b) => {
    const leftRef = Number(a?.display_ref ?? a?.ref);
    const rightRef = Number(b?.display_ref ?? b?.ref);
    if (Number.isFinite(leftRef) && Number.isFinite(rightRef) && leftRef !== rightRef) {
      return leftRef - rightRef;
    }
    return String(a?.group_id || '').localeCompare(String(b?.group_id || ''));
  });
};

export const extractInlineCitationRefs = (content = '') => {
  if (!content) return [];
  const refs = [];
  const seen = new Set();
  for (const m of getInlineCitationMatches(content)) {
    const ref = Number(m[1] || m[2]);
    if (!Number.isFinite(ref) || seen.has(ref)) continue;
    seen.add(ref);
    refs.push(ref);
  }
  return refs;
};

const stripInlineCitations = (text = '') =>
  replaceInlineCitationRefs(text, () => '').replace(/[ \t]{2,}/g, ' ').trim();

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
  for (const m of getInlineCitationMatches(sentence)) {
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

  const MIN_SUPPORT = 0.03;
  const MIN_REPLACE = 0.06;

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
    // 最终兜底：保留 LLM 自身标注的、在 citationMap 中存在的有效 ref，
    // 而非直接丢弃——跨语言场景（中文回答 + 英文文档）token 重叠为 0 但 ref 仍有效
    const preserved = [...new Set(refsInSentence)].filter((r) => citationMap.has(r));
    if (preserved.length > 0) return attachRefsToSentence(coreSentence, preserved.slice(0, 2));
    // 所有 ref 不在 citations 中（如文档自身的参考文献编号），才真正剥离
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

const finalizeAssistantContentAndCitations = (content, citations) => {
  return {
    content: String(content || ''),
    citations: normalizeCitationRecords(citations),
  };
};

export const normalizeAssistantCitations = (content, citations) => {
  if (!content || !Array.isArray(citations) || citations.length <= 1) return content;

  const refsInText = getInlineCitationMatches(content).map(m => Number(m[1] || m[2]));
  const uniqueRefs = new Set(refsInText);
  if (uniqueRefs.size !== 1) return content;

  const paragraphs = String(content).split(/\n{2,}/);
  const normalized = paragraphs.map((paragraph) => {
    if (!hasInlineCitationRefs(paragraph)) return paragraph;

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

    return replaceInlineCitationRefs(paragraph, () => `[${bestRef}]`);
  });

  return normalized.join('\n\n');
};

export const ensureAssistantInlineCitationFallback = (content, citations) => {
  if (!content || !Array.isArray(citations) || citations.length === 0) return content;

  const hasInlineRefs = hasInlineCitationRefs(content);
  if (hasInlineRefs) return content;

  const normalized = normalizeCitationRecords(citations);
  if (normalized.length === 0) return content;

  // 按段落匹配注入 [N] 引文（而非仅在末尾追加）
  const paragraphs = String(content).split('\n');
  let inCodeFence = false;
  const usedRefs = new Set();
  const usedCount = {};
  const eligibleIndices = [];

  // 第一遍：收集可注入段落和匹配分数
  const paraScores = paragraphs.map((para, idx) => {
    if (/^\s*```/.test(para)) { inCodeFence = !inCodeFence; return null; }
    if (inCodeFence) return null;
    const trimmed = para.trim();
    if (!trimmed || trimmed.length < 10 || /^#+\s/.test(trimmed)) return null;
    const paraTokens = tokenizeForCitation(trimmed);
    if (paraTokens.length < 3) return null;
    eligibleIndices.push(idx);

    const scores = [];
    for (const c of normalized) {
      // 优先用 _full_text（完整段落），回退到 highlight_text
      const supportText = (c._full_text || c.highlight_text || '').trim();
      const citTokens = tokenizeForCitation(supportText);
      const overlap = calcTokenOverlap(paraTokens, citTokens);
      const score = overlap / Math.max(1, paraTokens.length);
      if (score >= 0.02) scores.push({ ref: c.ref, score });
    }
    scores.sort((a, b) => b.score - a.score);
    return scores.length > 0 ? scores : null;
  });

  // 第二遍：贪心分配，优先未使用的 citation
  const assignments = new Map();
  paragraphs.forEach((_, idx) => {
    const scores = paraScores[idx];
    if (!scores) return;
    const topScore = scores[0].score;
    const candidates = scores.filter(s => s.score >= topScore * 0.6);
    candidates.sort((a, b) => (usedCount[a.ref] || 0) - (usedCount[b.ref] || 0) || b.score - a.score);
    const chosen = candidates[0].ref;
    assignments.set(idx, chosen);
    usedCount[chosen] = (usedCount[chosen] || 0) + 1;
    usedRefs.add(chosen);
  });

  // 跨语言兜底：全部段落无法匹配时，按顺序轮流分配不同 citation
  if (usedRefs.size === 0 && eligibleIndices.length > 0) {
    eligibleIndices.forEach((idx, i) => {
      const ref = normalized[i % normalized.length].ref;
      assignments.set(idx, ref);
      usedRefs.add(ref);
    });
  }

  if (usedRefs.size > 0) {
    const annotated = paragraphs.map((para, idx) => {
      if (assignments.has(idx)) return `${para}[${assignments.get(idx)}]`;
      return para;
    });
    return annotated.join('\n');
  }

  // 最终兜底：末尾追加所有 ref
  const allRefs = normalized.slice(0, 3).map((c) => `[${c.ref}]`).join('');
  return `${String(content).trimEnd()}\n\n参考来源：${allRefs}`;
};

// 检索代理执行轨迹的初始结构（与 AgentTracePanel 的 props 对齐）
export const createInitialAgentTrace = () => ({
  enabled: false,
  fallback: false,
  startedAt: null,
  endedAt: null,
  rounds: [],
  finalMessage: '',
  searchHistory: [],
  taskStatus: { completed: [], current: '', pending: [] },
  agentDetail: [],
  contextChars: 0,
  agentMode: false,
  agentGate: null,
  routeDiagnosis: null,
  evidenceScoring: null,
  error: '',
  fallbackReason: '',
  diagnostics: null,
  evidenceState: null,
  skippedToolCount: 0,
});

// 在 trace 中找到指定轮次，没有就创建并 push
const normalizeTraceRound = (value) => {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
};

const ensureTraceRound = (trace, round) => {
  let entry = trace.rounds.find((r) => r.round === round);
  if (!entry) {
    entry = { round, message: '', planningMessage: '', operations: [] };
    trace.rounds.push(entry);
    trace.rounds.sort((left, right) => left.round - right.round);
  }
  return entry;
};

// 把 retrieval_progress 事件累积到 agentTrace；只对 agent 相关 phase 生效
export const applyAgentTraceEvent = (trace, payload) => {
  if (!trace || !payload) return trace;
  const phase = payload.phase;
  if (!phase) return trace;
  const explicitRound = normalizeTraceRound(payload.round);
  // 只识别 agent 相关 phase 或显式带 round 字段的事件，避免吞掉普通 retrieval_progress
  if (!AGENT_PHASES.has(phase) && explicitRound === null) return trace;

  if (phase === 'agent_start' || phase === 'agent_mode') {
    trace.enabled = true;
    if (!trace.startedAt) trace.startedAt = Date.now();
    return trace;
  }

  if (phase === 'planner_error') {
    trace.enabled = true;
    trace.error = payload.error || payload.message || '检索规划失败';
  }

  if (phase === 'complete') {
    trace.finalMessage = payload.message || '检索完成';
    if (!trace.endedAt) trace.endedAt = Date.now();
    return trace;
  }

  trace.enabled = true;
  if (isSkippedAgentToolEvent(payload)) {
    // 被重复检索保护拦下的请求没有真正执行，不能被算作工具结果。
    trace.skippedToolCount = (Number(trace.skippedToolCount) || 0) + 1;
    return trace;
  }

  const fallbackRound = trace.rounds.length > 0
    ? trace.rounds[trace.rounds.length - 1].round
    : 1;
  const round = explicitRound ?? fallbackRound;

  if (phase === 'round_start') {
    const entry = ensureTraceRound(trace, round);
    if (!entry.message && payload.message) entry.message = payload.message;
    return trace;
  }

  const entry = ensureTraceRound(trace, round);

  if (phase === 'planning') {
    entry.planningMessage = payload.message || 'LLM 规划中...';
    return trace;
  }

  if (phase === 'executing') {
    entry.operations.push({
      tool: payload.tool || '',
      message: payload.message || '',
      query: payload.query || '',
      status: 'executing',
    });
    return trace;
  }

  if (phase === 'tool_result') {
    // 找到最后一个匹配 tool 且仍在 executing 的 op，回填结果；找不到就直接 push 一条
    let matched = false;
    for (let i = entry.operations.length - 1; i >= 0; i--) {
      const op = entry.operations[i];
      if (op.tool === payload.tool && op.status !== 'done') {
        op.resultMessage = payload.message || '';
        op.resultCount = Number.isFinite(Number(payload.result_count))
          ? Number(payload.result_count)
          : (op.resultCount ?? null);
        op.elapsedMs = Number.isFinite(Number(payload.elapsed_ms))
          ? Number(payload.elapsed_ms)
          : (op.elapsedMs ?? null);
        if (payload.query) op.query = payload.query;
        op.status = 'done';
        matched = true;
        break;
      }
    }
    if (!matched) {
      entry.operations.push({
        tool: payload.tool || '',
        message: '',
        query: payload.query || '',
        resultMessage: payload.message || '',
        resultCount: Number.isFinite(Number(payload.result_count))
          ? Number(payload.result_count)
          : null,
        elapsedMs: Number.isFinite(Number(payload.elapsed_ms))
          ? Number(payload.elapsed_ms)
          : null,
        status: 'done',
      });
    }
  }

  return trace;
};

// 流末事件附带的 retrieval_meta 中的 agent 详情合并到 trace（agent_search_history/task_status/agent_detail 等）
export const mergeAgentMetaIntoTrace = (trace, meta) => {
  if (!trace || !meta) return trace;
  if (meta.agent_mode) {
    trace.enabled = true;
    trace.agentMode = true;
  }
  if (meta.agent_fallback) trace.fallback = true;
  if (meta.agent_error) trace.error = String(meta.agent_error);
  if (meta.agent_fallback_reason) trace.fallbackReason = String(meta.agent_fallback_reason);
  if (Number.isFinite(Number(meta.agent_context_chars))) {
    trace.contextChars = Number(meta.agent_context_chars);
  }
  if (meta.agent_gate && typeof meta.agent_gate === 'object') {
    trace.agentGate = meta.agent_gate;
    // Show the panel for both enabled Agent runs and explicit route denials,
    // so users can see why a path was chosen.
    trace.enabled = true;
    if (meta.agent_gate.agent_mode || meta.agent_gate.use_agent) {
      trace.agentMode = true;
    }
    if (!trace.startedAt && !trace.endedAt) {
      trace.startedAt = Date.now();
      trace.endedAt = Date.now();
      if (!trace.finalMessage) {
        trace.finalMessage = meta.agent_gate.use_agent || meta.agent_gate.agent_mode
          ? 'Agent 检索完成'
          : `未走 Agent（${meta.agent_gate.reason || 'route'}）`;
      }
    }
  }
  if (meta.route_diagnosis && typeof meta.route_diagnosis === 'object') {
    trace.routeDiagnosis = meta.route_diagnosis;
  } else if (meta.intent_decision && typeof meta.intent_decision === 'object') {
    const intent = meta.intent_decision;
    trace.routeDiagnosis = {
      task: intent.task,
      scope: intent.scope,
      query_type: intent.query_type,
      evidence_need: intent.evidence_need,
      agent_policy: intent.agent_policy,
      web_policy: intent.web_policy,
      is_ambiguous: Boolean(intent.is_ambiguous),
      intent_id: intent.intent_id,
      intent_version: intent.version,
      decision_strength: intent.decision_strength ?? intent.confidence,
      matched_rules: intent.matched_rules,
      clarification_llm: meta.clarification_llm || null,
    };
  }
  if (meta.diagnostics?.agent && typeof meta.diagnostics.agent === 'object') {
    trace.diagnostics = meta.diagnostics.agent;
    if (meta.diagnostics.agent.evidence_scoring) {
      trace.evidenceScoring = meta.diagnostics.agent.evidence_scoring;
    }
    if (meta.diagnostics.agent.planner_academic_status) {
      trace.academicStatus = meta.diagnostics.agent.planner_academic_status;
    }
  }
  if (meta.diagnostics?.evidence_scoring && typeof meta.diagnostics.evidence_scoring === 'object') {
    trace.evidenceScoring = meta.diagnostics.evidence_scoring;
  }
  if (meta.agent_evidence_state && typeof meta.agent_evidence_state === 'object') {
    trace.evidenceState = meta.agent_evidence_state;
  } else if (meta.diagnostics?.agent?.evidence_state && typeof meta.diagnostics.agent.evidence_state === 'object') {
    // Compatibility with older responses that only nest this state under diagnostics.
    trace.evidenceState = meta.diagnostics.agent.evidence_state;
  }
  if (Array.isArray(meta.agent_search_history)) {
    trace.searchHistory = meta.agent_search_history;
  }
  if (meta.task_status && typeof meta.task_status === 'object') {
    trace.taskStatus = {
      completed: Array.isArray(meta.task_status.completed) ? meta.task_status.completed : [],
      current: typeof meta.task_status.current === 'string' ? meta.task_status.current : '',
      pending: Array.isArray(meta.task_status.pending) ? meta.task_status.pending : [],
    };
  }
  if (Array.isArray(meta.agent_detail)) {
    trace.agentDetail = meta.agent_detail;
  }
  return trace;
};

export const getEffectiveWebSearchQuery = (meta) => {
  const history = Array.isArray(meta?.agent_search_history)
    ? meta.agent_search_history
    : [];
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const item = history[index];
    if (String(item?.tool || '').trim().toLowerCase() !== 'web_search') continue;
    const query = String(item?.query || '').replace(/\s+/g, ' ').trim();
    if (query) return query.slice(0, 240);
  }
  const directQuery = meta?.web_search_query
    || meta?.web_search_query_meta?.effective_query
    || meta?.web_search_audit?.effective_query;
  if (directQuery) {
    return String(directQuery).replace(/\s+/g, ' ').trim().slice(0, 240);
  }
  const diagnosticQuery = meta?.diagnostics?.agent?.web_search?.effective_query
    || meta?.diagnostics?.web_search?.effective_query;
  return String(diagnosticQuery || '').replace(/\s+/g, ' ').trim().slice(0, 240);
};

export const finalizeThinkingDurationMs = ({
  thinkingStartTime,
  thinkingLastUpdateTime,
  contentStartTime,
}) => {
  if (!Number.isFinite(thinkingStartTime)) return 0;

  const endCandidates = [thinkingLastUpdateTime, contentStartTime]
    .filter((value) => Number.isFinite(value) && value >= thinkingStartTime);

  if (endCandidates.length === 0) return 0;
  return Math.max(...endCandidates) - thinkingStartTime;
};

/**
 * 消息状态管理 Hook
 * 管理消息列表、流式输出、历史记录等状态和逻辑
 *
 * @param {Object} options - 配置选项
 * @param {string|null} options.docId - 当前文档 ID
 * @param {boolean} options.parseIdentityReady - 当前文档解析身份是否已完成初始化
 * @param {Array} options.screenshots - 截图列表
 * @param {Function} options.setScreenshots - 设置截图列表
 * @param {string} options.selectedText - 当前选中的文本
 * @param {Function} options.getChatCredentials - 获取聊天凭证
 * @param {Function} options.getVisualCredentials - 获取独立视觉模型凭证
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
  parseGeneration = '',
  documentSourceHash = '',
  parseIdentityReady = true,
  screenshots = [],
  setScreenshots,
  selectedText = '',
  getChatCredentials,
  getVisualCredentials,
  getCurrentChatModel,
  getProviderById,
  streamSpeed = 'normal',
  enableVectorSearch = false,
  embeddingApiKey = '',
  getEmbeddingConfig,
  enableGraphRAG = false,
  enableAgentRetrieval = false,
  forceAgentRetrieval = false,
  enableJiebaBM25 = true,
  numExpandContextChunk = 1,
  enableBlurReveal = false,
  blurIntensity = 'medium',
  enableMathHydration = true,
  enableSingleDollarMath = true,
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
  const requestEpochRef = useRef(0);
  const normalizedParseGeneration = String(parseGeneration || '').trim();
  const normalizedDocumentSourceHash = String(documentSourceHash || '').trim();
  const chatParseContextRef = useRef({
    docId: docId || '',
    parseGeneration: normalizedParseGeneration,
    documentSourceHash: normalizedDocumentSourceHash,
    epoch: 0,
  });
  if (
    chatParseContextRef.current.docId !== (docId || '')
    || chatParseContextRef.current.parseGeneration !== normalizedParseGeneration
    || chatParseContextRef.current.documentSourceHash !== normalizedDocumentSourceHash
  ) {
    chatParseContextRef.current = {
      docId: docId || '',
      parseGeneration: normalizedParseGeneration,
      documentSourceHash: normalizedDocumentSourceHash,
      epoch: chatParseContextRef.current.epoch + 1,
    };
  }
  const appliedParseContextKeyRef = useRef(
    `${docId || ''}:${normalizedParseGeneration}:${normalizedDocumentSourceHash}`
  );
  const streamCitationsRef = useRef(null);
  const streamCitationBindingsRef = useRef(null);
  const streamMaxRelevanceRef = useRef(null);
  const streamFollowupRef = useRef(null);
  const streamFinalContentRef = useRef(null);
  const streamQaScoreRef = useRef(null);
  const streamConvNameRef = useRef(null);
  const streamMindmapRef = useRef(null);
  const streamAnswerCriticRef = useRef(null);
  const streamAnswerCertaintyRef = useRef(null);
  const streamWebSearchRef = useRef(null);
  const streamWebSearchQueryRef = useRef('');
  const streamWebSearchAuditRef = useRef(null);
  const streamWebSearchStatusRef = useRef(null);
  const streamWebSearchReadsRef = useRef([]);
  const streamMemoryHitsRef = useRef(null);
  const streamMemoryMetaRef = useRef(null);
  const streamAgentTraceRef = useRef(null);
  const streamUsageRef = useRef(null);
  const streamCallInfoRef = useRef(null);
  const streamVisualVerificationRef = useRef(null);
  const streamIntentDecisionRef = useRef(null);
  const streamVisualAttachmentsRef = useRef(null);
  const streamReasoningResolutionRef = useRef(null);
  const activeStreamMsgIdRef = useRef(null);
  const textareaRef = useRef(null);
  const visualVerificationPollersRef = useRef(new Map());
  const visualVerificationEpochRef = useRef(0);

  // ========== 从全局设置中解构对话参数 ==========
  const {
    maxTokens, temperature, topP, contextCount, streamOutput,
    enableTemperature, enableTopP, enableMaxTokens,
    customParams, reasoningEffort, answerDetailLevel,
    enableMemory, memoryTopK, memoryInjectionBudget, memoryPrivacyMode,
    overrideNumericTable, overrideAnswerCritic, overrideAnswerClaimVerifier, overrideLLMQueryRewrite, overrideBM25Synonyms,
    numericTableVisualVerification,
    cheapModel, cheapModelProvider, cheapModelEndpoint,
  } = globalSettings;

  const {
    webSearchMode,
    enableWebSearch,
    webSearchProvider,
    webSearchApiKey,
    webSearchBlacklist,
    webSearchIncludeDocumentContext,
  } = useWebSearch();
  const streamRenderProfile = useMemo(
    () => resolveStreamRenderProfile(streamSpeed),
    [streamSpeed]
  );
  // 兼容旧配置：历史上 streamOutput 与 streamSpeed 是两套独立开关，
  // 容易出现“速度不是 off，但 streamOutput 被旧 localStorage 关掉”的冲突状态。
  // 这里以更直观的 streamSpeed 为准：只要速度档位不是 off，就走流式。
  const shouldUseStreaming = streamSpeed !== 'off' ? true : Boolean(streamOutput);

  // ========== 流式输出 Hook（ref 直写模式，需求 4.2） ==========
  // 流式输出期间不调用 setMessages，通过 contentRef 直接更新 DOM
  // 流结束后通过 getFinalText() 一次性同步到 React 状态
  const contentStream = useSmoothStream({
    streamDone: contentStreamDone,
    minDelay: streamRenderProfile.minDelay,
    frameChars: streamRenderProfile.frameChars,
    flushChars: streamRenderProfile.flushChars,
    enableBlurReveal,
    blurIntensity,
    smoothFlush: true,
    sequencedReveal: enableBlurReveal,
    paragraphPauseMs: 180,
    enableMathHydration,
    enableSingleDollarMath,
    enableMarkdownCommit: enableBlurReveal,
  });

  const thinkingStreamProfile = resolveThinkingStreamProfile();
  const thinkingStream = useSmoothStream({
    streamDone: thinkingStreamDone,
    minDelay: thinkingStreamProfile.minDelay,
    frameChars: thinkingStreamProfile.frameChars,
    flushChars: thinkingStreamProfile.flushChars,
    smoothFlush: true,
    enableMathHydration,
    enableSingleDollarMath,
  });

  const interruptActiveRequest = useCallback(({ staleIdentity = false } = {}) => {
    const targetMessageId = activeStreamMsgIdRef.current || streamingMessageId;
    const renderedContent = String(contentStream.getFinalText?.() || '');
    const renderedThinking = String(thinkingStream.getFinalText?.() || '');

    requestEpochRef.current += 1;
    streamingAbortRef.current.cancelled = true;
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    activeStreamMsgIdRef.current = null;

    if (targetMessageId) {
      setMessages((previous) => previous.map((message) => (
        message.id !== targetMessageId
          ? message
          : {
            ...message,
            ...(renderedContent.trim()
              ? { content: renderedContent }
              : !String(message.content || '').trim() && staleIdentity
                ? { content: CHAT_PARSE_IDENTITY_UPDATED_MESSAGE }
                : {}),
            ...(renderedThinking.trim() ? { thinking: renderedThinking } : {}),
            isStreaming: false,
            turnStatus: 'interrupted',
            ...(staleIdentity ? { parseIdentityStale: true } : {}),
          }
      )));
    }

    setIsLoading(false);
    setStreamingMessageId(null);
    contentStream.reset('');
    thinkingStream.reset('');
    setContentStreamDone(false);
    setThinkingStreamDone(false);
  }, [contentStream, streamingMessageId, thinkingStream]);

  // ========== 副作用 ==========

  useEffect(() => {
    const nextKey = `${docId || ''}:${normalizedParseGeneration}:${normalizedDocumentSourceHash}`;
    if (appliedParseContextKeyRef.current === nextKey) return;
    appliedParseContextKeyRef.current = nextKey;
    interruptActiveRequest({ staleIdentity: true });
  }, [
    docId,
    interruptActiveRequest,
    normalizedDocumentSourceHash,
    normalizedParseGeneration,
  ]);

  // 视觉核验属于回答后的异步旁路任务；切换文档或卸载时必须取消轮询，
  // 以免旧文档结果写入新会话。
  useEffect(() => () => {
    for (const poller of visualVerificationPollersRef.current.values()) {
      poller.cancel();
    }
    visualVerificationPollersRef.current.clear();
  }, [docId, normalizedDocumentSourceHash, normalizedParseGeneration]);

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

  const startVisualVerificationPolling = useCallback((
    messageId,
    initialVerification,
    verificationEpoch = visualVerificationEpochRef.current
  ) => {
    const verification = normalizeNumericTableVisualVerification(initialVerification);
    const taskId = String(verification?.task_id || '').trim();
    if (
      !docId
      || !taskId
      || verificationEpoch !== visualVerificationEpochRef.current
      || !isVisualVerificationPending(verification)
    ) return;

    const pollContext = { ...chatParseContextRef.current };
    const isPollContextCurrent = () => (
      chatParseContextRef.current.docId === pollContext.docId
      && chatParseContextRef.current.parseGeneration === pollContext.parseGeneration
      && chatParseContextRef.current.documentSourceHash === pollContext.documentSourceHash
      && chatParseContextRef.current.epoch === pollContext.epoch
      && visualVerificationEpochRef.current === verificationEpoch
    );

    const pollerKey = `${docId}:${pollContext.parseGeneration}:${messageId}:${taskId}`;
    if (visualVerificationPollersRef.current.has(pollerKey)) return;

    const controller = new AbortController();
    let cancelled = false;
    let timerId = null;
    let attempts = 0;
    let latestVerification = verification;
    const updateMessageVerification = (nextVerification) => {
      if (cancelled || !isPollContextCurrent()) return;
      setMessages((previous) => previous.map((message) => (
        message.id === messageId
          ? { ...message, visualVerification: nextVerification }
          : message
      )));
    };
    const stop = () => {
      if (cancelled) return;
      cancelled = true;
      if (timerId) clearTimeout(timerId);
      controller.abort();
      visualVerificationPollersRef.current.delete(pollerKey);
    };

    const poll = async () => {
      if (cancelled || !isPollContextCurrent()) {
        stop();
        return;
      }
      attempts += 1;
      try {
        const response = await fetch(
          `${API_BASE_URL}/documents/${encodeURIComponent(docId)}/table-visual-verifications/${encodeURIComponent(taskId)}`,
          { signal: controller.signal }
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        const result = normalizeNumericTableVisualVerification(payload);
        if (result) {
          const nextVerification = {
            ...latestVerification,
            ...result,
            task_id: result.task_id || taskId,
          };
          latestVerification = nextVerification;
          updateMessageVerification(nextVerification);
          if (isVisualVerificationTerminal(nextVerification)) {
            stop();
            return;
          }
        }
      } catch (error) {
        if (cancelled || error?.name === 'AbortError') return;
        if (attempts >= TABLE_VISUAL_VERIFICATION_POLL_MAX_ATTEMPTS) {
          updateMessageVerification({
            ...latestVerification,
            state: 'failed',
            polling_error: 'status_unavailable',
          });
          stop();
          return;
        }
      }

      if (attempts >= TABLE_VISUAL_VERIFICATION_POLL_MAX_ATTEMPTS) {
        updateMessageVerification({
          ...latestVerification,
          state: 'failed',
          polling_error: 'status_timeout',
        });
        stop();
        return;
      }
      timerId = setTimeout(poll, TABLE_VISUAL_VERIFICATION_POLL_INTERVAL_MS);
    };

    visualVerificationPollersRef.current.set(pollerKey, { cancel: stop });
    void poll();
  }, [docId]);

  const invalidateVisualVerificationState = useCallback(() => {
    visualVerificationEpochRef.current += 1;
    streamVisualVerificationRef.current = null;
    for (const poller of visualVerificationPollersRef.current.values()) {
      poller.cancel();
    }
    visualVerificationPollersRef.current.clear();
    setMessages((previous) => previous.map((message) => {
      if (!Object.prototype.hasOwnProperty.call(message, 'visualVerification')) return message;
      const nextMessage = { ...message };
      delete nextMessage.visualVerification;
      return nextMessage;
    }));
  }, []);

  /**
   * 发送消息
   * 处理用户输入、构建请求体、发起流式/非流式请求
   */
  const sendMessage = useCallback(async (overrides = {}) => {
    const overrideInput = typeof overrides?.input === 'string' ? overrides.input : null;
    const historyMessages = Array.isArray(overrides?.historyMessages)
      ? overrides.historyMessages
      : messages;
    const currentInput = overrideInput ?? textareaRef.current?.value ?? '';
    if (!currentInput.trim() && screenshots.length === 0) return;

    const { providerId: chatProvider, modelId: chatModel, apiKey: chatApiKey } = getChatCredentials?.() || {};
    const visualCredentials = getVisualCredentials?.() || null;
    if (!docId) { alert('请先上传文档'); return; }
    if (!parseIdentityReady) {
      alert('正在同步文档解析状态，请稍后再试');
      return;
    }
    if (!chatApiKey && chatProvider !== 'ollama' && chatProvider !== 'local') {
      alert('请先配置API Key\n\n请点击左下角"设置 & API Key"按钮进行配置');
      return;
    }

    // Companion documents are always explicit user choices. Keep the current
    // document as the primary route and bound extras to the backend fan-out
    // contract. Agent mode is intentionally single-document, so requesting
    // companions switches this turn to the deterministic multi-doc retriever.
    const requestedDocIds = [...new Set(
      (Array.isArray(overrides?.docIds) ? overrides.docIds : [])
        .map((value) => String(value || '').trim())
        .filter((value) => value && value !== String(docId))
    )].slice(0, 4);
    const isMultiDocumentRequest = requestedDocIds.length > 0;

    if (abortControllerRef.current || activeStreamMsgIdRef.current || streamingMessageId) {
      interruptActiveRequest();
    }

    const requestParseContext = { ...chatParseContextRef.current };
    const requestParseIdentity = {
      parseGeneration: requestParseContext.parseGeneration,
      documentSourceHash: requestParseContext.documentSourceHash,
    };
    const requestIdentityFields = getChatIdentityFields(requestParseIdentity);
    // A retry control must be resolved only from turns belonging to this
    // parse generation. Otherwise an old failed answer can turn a normal
    // "continue" into a hidden replay of a different document revision.
    const identityBoundHistory = historyMessages.filter((message) => (
      messageMatchesChatParseIdentity(message, requestParseIdentity)
    ));
    const retryControlQuestion = resolveRetryControlQuestion(
      currentInput,
      identityBoundHistory,
    );
    // A normal follow-up may be a new question even when the previous answer
    // carried a clarification hint. Resume only through an explicit UI/API
    // action, never by silently attaching the latest ticket.
    const clarificationResponse = Boolean(overrides?.clarificationResponse);
    const clarificationTicket = retryControlQuestion || !clarificationResponse
      ? null
      : resolveClarificationTicket(identityBoundHistory);
    const requestedInteractionMode = String(overrides?.interactionMode || '').trim();
    const interactionMode = CHAT_INTERACTION_MODES.has(requestedInteractionMode)
      ? requestedInteractionMode
      : retryControlQuestion
        ? 'retry_failed_turn'
        : screenshots.length > 0
          ? 'image'
          : String(selectedText || '').trim()
            ? 'selection'
            : 'default';
    const requestEpoch = ++requestEpochRef.current;
    const requestVisualVerificationEpoch = visualVerificationEpochRef.current;
    const requestController = new AbortController();
    const requestAbortState = { cancelled: false };
    abortControllerRef.current = requestController;
    streamingAbortRef.current = requestAbortState;
    const isRequestCurrent = () => (
      requestEpochRef.current === requestEpoch
      && chatParseContextRef.current.docId === requestParseContext.docId
      && chatParseContextRef.current.parseGeneration === requestParseContext.parseGeneration
      && chatParseContextRef.current.documentSourceHash === requestParseContext.documentSourceHash
      && chatParseContextRef.current.epoch === requestParseContext.epoch
    );

    // 构建用户消息
    const userMsg = {
      type: 'user',
      content: currentInput,
      hasImage: screenshots.length > 0,
      interactionMode,
      ...requestIdentityFields,
      ...(retryControlQuestion ? { contextContent: retryControlQuestion } : {}),
    };
    setMessages(prev => [...prev, userMsg]);

    // 清空输入框
    if (textareaRef.current) {
      textareaRef.current.value = '';
      textareaRef.current.style.height = '24px';
    }
    setHasInput(false);
    setIsLoading(true);

    // 构建聊天历史
    const chatHistory = buildChatHistory(identityBoundHistory, contextCount, {
      preserveReasoning: requiresPreservedReasoning({
        providerId: chatProvider,
        modelId: chatModel,
      }),
      providerId: chatProvider,
      modelId: chatModel,
    }).map((message) => ({
      ...message,
      ...requestIdentityFields,
    }));
    const requestQuestion = retryControlQuestion || userMsg.content;

    // 获取 provider 完整信息
    const chatProviderFull = getProviderById?.(chatProvider);
    const useDedicatedVisualModel = visualCredentials?.source === 'dedicated';
    const visualProviderFull = useDedicatedVisualModel
      ? getProviderById?.(visualCredentials?.providerId || '')
      : null;
    const localVisualProviderFull = visualCredentials?.local?.providerId
      ? getProviderById?.(visualCredentials.local.providerId)
      : null;
    const visualRequestParams = {
      visual_strategy: visualCredentials?.strategy || 'balanced',
      visual_enabled: visualCredentials ? visualCredentials.isVisionCapable === true : true,
      ...(useDedicatedVisualModel
        ? {
          visual_provider: visualCredentials?.providerId || '',
          visual_model: visualCredentials?.modelId || '',
          visual_api_key: visualCredentials?.apiKey || '',
          visual_api_host: visualProviderFull?.apiHost || visualCredentials?.apiHost || '',
        }
        : {}),
      ...(visualCredentials?.local?.providerId && visualCredentials?.local?.modelId
        ? {
          local_visual_provider: visualCredentials.local.providerId,
          local_visual_model: visualCredentials.local.modelId,
          local_visual_api_key: visualCredentials.local.apiKey || '',
          local_visual_api_host: localVisualProviderFull?.apiHost || visualCredentials.local.apiHost || '',
        }
        : {}),
    };

    const activeEmbeddingConfig = getEmbeddingConfig?.();
    const embeddingProviderId = activeEmbeddingConfig?.isValid
      ? activeEmbeddingConfig.providerId || ''
      : '';
    const embeddingApiHost = activeEmbeddingConfig?.isValid
      ? activeEmbeddingConfig.provider?.apiHost || ''
      : '';
    // 上传、局部搜索和 RAG 重建都使用 provider:model 的复合身份。
    // 聊天也必须发送同一身份；只发裸 modelId 会让动态 Provider/同名模型
    // 在后端被解析成另一份索引，从而触发 409 embedding identity conflict。
    const embeddingModelId = activeEmbeddingConfig?.isValid
      ? activeEmbeddingConfig.compositeKey || activeEmbeddingConfig.modelId || ''
      : '';

    // 构建请求体
    const requestBody = {
      doc_id: docId,
      doc_ids: requestedDocIds.length > 0 ? requestedDocIds : null,
      parse_generation: requestParseIdentity.parseGeneration || null,
      document_source_hash: requestParseIdentity.documentSourceHash || null,
      clarification_ticket: clarificationTicket,
      clarification_response: Boolean(clarificationTicket && clarificationResponse),
      question: requestQuestion,
      api_key: chatApiKey,
      model: chatModel,
      api_provider: chatProvider,
      api_host: chatProviderFull?.apiHost || null,
      selected_text: selectedText || null,
      image_base64_list: screenshots.map(s => s.dataUrl.split(',')[1]),
      image_base64: screenshots[0]?.dataUrl ? screenshots[0].dataUrl.split(',')[1] : null,
      // off 也是一个需要后端按厂商协议显式序列化的请求值，不能用 null
      // 代替；否则推理优先模型会回到厂商默认并继续思考。
      enable_thinking: reasoningEffort !== 'off',
      reasoning_effort: reasoningEffort,
      answer_detail: answerDetailLevel || 'standard',
      max_tokens: enableMaxTokens ? maxTokens : null,
      temperature: enableTemperature ? temperature : null,
      top_p: enableTopP ? topP : null,
      stream_output: shouldUseStreaming,
      enable_vector_search: enableVectorSearch,
      embedding_api_key: embeddingApiKey || null,
      embedding_model: embeddingModelId || null,
      embedding_provider: embeddingProviderId || null,
      embedding_api_host: embeddingApiHost || null,
      enable_graphrag: enableGraphRAG,
      enable_agent_retrieval: isMultiDocumentRequest ? false : enableAgentRetrieval,
      force_agent_retrieval: isMultiDocumentRequest ? false : forceAgentRetrieval,
      interaction_mode: interactionMode,
      enable_jieba_bm25: enableJiebaBM25,
      num_expand_context_chunk: numExpandContextChunk,
      chat_history: chatHistory.length > 0 ? chatHistory : null,
      custom_params: {
        ...(customParams?.length > 0
          ? Object.fromEntries(customParams.filter(p => p.name).map(p => [p.name, p.value]))
          : {}),
        numeric_table_visual_verification: numericTableVisualVerification || 'auto',
        ...visualRequestParams,
      },
      enable_memory: enableMemory,
      // null 表示跟随后端 config，不做会话级覆盖
      memory_top_k: memoryTopK ?? null,
      memory_injection_budget: memoryInjectionBudget ?? null,
      memory_privacy_mode: memoryPrivacyMode || 'personal',
      enable_web_search: enableWebSearch,
      web_search_mode: webSearchMode,
      web_search_provider: webSearchProvider,
      web_search_api_key: webSearchApiKey || null,
      web_search_blacklist: webSearchBlacklist && webSearchBlacklist.length > 0 ? webSearchBlacklist : null,
      web_search_include_document_context: webSearchIncludeDocumentContext,
      // 检索增强调优 overrides（null 表示跟随后端默认）
      override_numeric_table: overrideNumericTable ?? null,
      override_answer_critic: overrideAnswerCritic ?? null,
      override_answer_claim_verifier: overrideAnswerClaimVerifier ?? null,
      override_llm_query_rewrite: overrideLLMQueryRewrite ?? null,
      override_bm25_synonyms: overrideBM25Synonyms ?? null,
      // 辅助模型（双模型策略；空字符串转 null 避免后端误匹配）
      cheap_model: cheapModel ? cheapModel : null,
      cheap_model_provider: cheapModelProvider ? cheapModelProvider : null,
      cheap_model_endpoint: cheapModelEndpoint ? cheapModelEndpoint : null,
    };

    setScreenshots?.([]);

    streamCitationsRef.current = null;
    streamCitationBindingsRef.current = null;
    streamMaxRelevanceRef.current = null;
    streamFollowupRef.current = null;
    streamFinalContentRef.current = null;
    streamQaScoreRef.current = null;
    streamConvNameRef.current = null;
    streamMindmapRef.current = null;
    streamAnswerCriticRef.current = null;
    streamAnswerCertaintyRef.current = null;
    streamWebSearchRef.current = null;
    streamWebSearchQueryRef.current = '';
    streamWebSearchAuditRef.current = null;
    streamWebSearchStatusRef.current = null;
    streamWebSearchReadsRef.current = [];
    streamMemoryHitsRef.current = null;
    streamMemoryMetaRef.current = null;
    streamAgentTraceRef.current = null;
    streamUsageRef.current = null;
    streamCallInfoRef.current = null;
  streamVisualVerificationRef.current = null;
  streamIntentDecisionRef.current = null;
  streamVisualAttachmentsRef.current = null;
  streamReasoningResolutionRef.current = null;

    // 创建临时助手消息
    const tempMsgId = Date.now();
    setStreamingMessageId(tempMsgId);
    activeStreamMsgIdRef.current = tempMsgId;
    setMessages(prev => [...prev, {
      id: tempMsgId, type: 'assistant', content: '', model: chatModel, provider: chatProvider,
      isStreaming: true, thinking: '', thinkingMs: 0, turnStatus: 'streaming',
      retrievalProgress: [],
      webSearchQuery: String(requestQuestion || '').replace(/\s+/g, ' ').trim().slice(0, 240),
      ...requestIdentityFields,
    }]);

    // 每次发送前重置流式状态，确保 rAF 循环重启且无残留数据
    setContentStreamDone(false);
    setThinkingStreamDone(false);
    contentStream.reset('');
    thinkingStream.reset('');

    let firstEventTimeoutTriggered = false;
    try {
      if (shouldUseStreaming) {
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
          if (firstEventReceived || requestAbortState.cancelled || !isRequestCurrent()) return;
          firstEventTimeoutTriggered = true;
          requestAbortState.cancelled = true;
          requestController.abort();
        }, STREAM_FIRST_EVENT_TIMEOUT_MS);

        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal: requestController.signal,
        });

        if (!response.ok) {
          let ed = `HTTP ${response.status}`;
          try {
            const eb = await response.json();
            ed = eb.detail || eb.error?.message || eb.message || JSON.stringify(eb);
          } catch (e) { /* ignore */ }
          clearFirstEventTimer();
          if (
            response.status === 409
            && isChatParseIdentityConflict(response, null, requestParseIdentity, ed)
          ) {
            const identityError = new Error(ed || '文档解析结果已更新，本次回答已停止，请重新提问');
            identityError.name = 'ChatParseIdentityError';
            throw identityError;
          }
          if (response.status === 409 && isEmbeddingIdentityConflictDetail(ed)) {
            throw createEmbeddingIdentityConflictError(ed);
          }
          throw new Error(ed);
        }
        // 跨域浏览器会隐藏未显式暴露的响应头；两个身份头都不可读时，
        // 延迟到携带同一不可变解析身份的 SSE 终止事件再做严格核验。
        if (
          responseHasAnyChatParseIdentity(response, null)
          && !responseMatchesChatParseIdentity(response, null, requestParseIdentity)
        ) {
          const identityError = new Error('文档解析结果已更新，本次回答已停止，请重新提问');
          identityError.name = 'ChatParseIdentityError';
          throw identityError;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let currentText = '';
        let currentThinking = '';
        let currentRetrievalProgress = [];
        let hasRealThinking = false;
        let hasPublishedRealThinking = false;
        let thinkingStreamFinalized = false;
        let lastThinkingPublishAt = 0;
        let lastThinkingStageKey = null;
        let pendingReasoningRoleLabel = '';
        let thinkingStartTime = null;
        let thinkingLastUpdateTime = null;
        let contentStartTime = null;
        let sseBuffer = '';
        let sseDone = false;
        let streamTerminalFailed = false;
        let streamTerminalReceived = false;
        let streamServerTurnStatus = '';
        let streamFinishReason = '';
        let streamCompletionStatus = '';
        let streamWasTruncated = false;
        let streamResponseIdentityMismatch = false;
        let streamEmbeddingIdentityConflict = false;
        let streamTerminalIdentityVerified = !hasCompleteChatParseIdentity(requestParseIdentity);

        const markThinkingActivity = () => {
          const now = Date.now();
          if (!thinkingStartTime) thinkingStartTime = now;
          thinkingLastUpdateTime = now;
        };

        const appendRetrievalProgress = (stage) => {
          if (!stage?.text || !isRequestCurrent()) return;
          // llm_waiting 等稳定阶段会持续收到心跳。更新并移到队尾，避免首个
          // 状态占住 key 后吞掉后续“等待正文”等更准确的实时状态。
          const nextProgress = upsertRetrievalProgressEntry(currentRetrievalProgress, stage);
          if (nextProgress === currentRetrievalProgress) return;
          currentRetrievalProgress = nextProgress;
          markThinkingActivity();
          setMessages(prev => prev.map(m => (
            m.id === tempMsgId
              ? { ...m, retrievalProgress: currentRetrievalProgress }
              : m
          )));
        };

        const commitAgentTrace = () => {
          if (!isRequestCurrent() || !streamAgentTraceRef.current?.enabled) return;
          const liveTrace = cloneAgentTrace(streamAgentTraceRef.current);
          setMessages(prev => prev.map(m => (
            m.id === tempMsgId ? { ...m, agentTrace: liveTrace } : m
          )));
        };

        const appendThinkingStage = (text, key) => {
          if (!text || hasRealThinking || !isRequestCurrent()) return;
          if (key && lastThinkingStageKey === key) return;
          const addition = currentThinking ? `\n${text}` : text;
          lastThinkingStageKey = key || null;
          markThinkingActivity();
          currentThinking += addition;
          thinkingStream.replace(currentThinking);
          setMessages(prev => prev.map(m =>
            m.id === tempMsgId
              ? { ...m, thinking: currentThinking }
              : m
          ));
        };

        const beginRealThinking = () => {
          if (hasRealThinking) return;
          hasRealThinking = true;
          lastThinkingStageKey = null;
          if (currentThinking) {
            currentThinking = '';
            thinkingStream.replace('');
            setMessages(prev => prev.map(m =>
              m.id === tempMsgId
                ? { ...m, thinking: '' }
                : m
            ));
          }
        };

        const appendRealThinking = (text) => {
          if (!text || !isRequestCurrent()) return;
          const normalized = consumeDisplayableReasoningDelta(text, pendingReasoningRoleLabel);
          pendingReasoningRoleLabel = normalized.pendingRoleLabel;
          if (!normalized.text || (!hasRealThinking && !normalized.text.trim())) return false;
          beginRealThinking();
          markThinkingActivity();
          currentThinking += normalized.text;
          thinkingStream.addChunk(normalized.text);
          // 流式阶段由 ref 增量写入 DOM；React 侧只需要知道“已经有真思考”，
          // 以便 ThinkingBlock 立刻取消 hidden。正文内容仍走 ref，避免整列重绘。
          const now = Date.now();
          if (!hasPublishedRealThinking || now - lastThinkingPublishAt >= 80) {
            hasPublishedRealThinking = true;
            lastThinkingPublishAt = now;
            setMessages(prev => prev.map(m =>
              m.id === tempMsgId
                ? { ...m, thinking: currentThinking, thinkingLive: true }
                : m
            ));
          }
          return true;
        };

        const finalizeThinkingStream = () => {
          if (thinkingStreamFinalized) return;
          thinkingStreamFinalized = true;
          // 正文可以立即开始，但思考队列继续独立、平滑地排空。这里不能
          // flushNow，否则上游最后一个大 reasoning chunk 会整段跳到屏幕上。
          setThinkingStreamDone(true);
          // appendRealThinking 为避免高频重绘会节流 React state。正文首字出现时
          // 必须发布一次完整目标文本，让 ThinkingBlock 能精确判断字符队列何时
          // 展示完毕，并在那个时刻自动收起，而不是用整条 SSE 结束来猜。
          setMessages((prev) => prev.map((item) => (
            item.id === tempMsgId
              ? { ...item, thinking: currentThinking, thinkingLive: hasRealThinking }
              : item
          )));
        };

        const markAnswerStarted = ({ generating = false } = {}) => {
          if (!contentStartTime) contentStartTime = Date.now();
          const hasVisibleAnswer = Boolean(String(currentText || '').trim());
          // thinking_complete 与首个可见正文 token 共用这组状态，避免过程区、
          // “正在生成回答”和正文三者在交接时互相覆盖。
          setMessages((prev) => {
            let changed = false;
            const next = prev.map((item) => {
              if (item.id !== tempMsgId) return item;
              const updated = applyAnswerStartedFlags(item, { generating, hasVisibleAnswer });
              if (updated !== item) changed = true;
              return updated;
            });
            return changed ? next : prev;
          });
        };

        const markThinkingComplete = (message = '') => {
          if (!isRequestCurrent()) return;
          // 后端完成事件可能先于同一模型 chunk 中的最后一段 reasoning 到达。
          // 这里只切换 UI 过渡态；思考流由首个可见正文 token 原子关闭。
          if (message) {
            appendThinkingStage(message, 'model:thinking_complete');
          }
          markAnswerStarted({ generating: true });
        };

        const appendAnswerContent = (text) => {
          if (!text || !isRequestCurrent()) return;
          const hadVisibleAnswer = Boolean(String(currentText || '').trim());
          currentText += text;
          contentStream.addChunk(text);
          // 模型从 reasoning 切到正文时常先发换行。只有真正出现可见字符后
          // 才结束思考状态，避免“已深度思考”与首字之间出现空白等待。
          // 若后端已发 thinking_complete，这里清除 generating 过渡态。
          if (currentText.trim()) {
            if (!hadVisibleAnswer) finalizeThinkingStream();
            markAnswerStarted({ generating: false });
          }
        };

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
          if (!isRequestCurrent() || requestAbortState.cancelled) return;
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
            if (p.error_code === 'chat_parse_identity_changed') {
              streamResponseIdentityMismatch = true;
              requestAbortState.cancelled = true;
              sseDone = true;
              return;
            }
            const payloadIdentity = getResponseChatParseIdentity(null, p);
            const payloadHasIdentity = Boolean(
              payloadIdentity.parseGeneration || payloadIdentity.documentSourceHash
            );
            if (
              payloadHasIdentity
              && !responseMatchesChatParseIdentity(null, p, requestParseIdentity)
            ) {
              streamResponseIdentityMismatch = true;
              requestAbortState.cancelled = true;
              sseDone = true;
              return;
            }
            const isTerminalPayload = Boolean(p.done || p.choices?.[0]?.finish_reason);
            if (isTerminalPayload && hasCompleteChatParseIdentity(requestParseIdentity)) {
              if (!chatParseIdentitiesMatch(payloadIdentity, requestParseIdentity)) {
                streamResponseIdentityMismatch = true;
                requestAbortState.cancelled = true;
                sseDone = true;
                return;
              }
              streamTerminalIdentityVerified = true;
            }
            const payloadTurnStatus = normalizeChatTurnStatus(
              p.turn_status || p.answer_status
            );
            if (payloadTurnStatus) streamServerTurnStatus = payloadTurnStatus;
            if (isTerminalPayload) {
              streamTerminalReceived = true;
              streamFinishReason = String(
                p.finish_reason || p.choices?.[0]?.finish_reason || streamFinishReason || ''
              );
              streamCompletionStatus = String(p.completion_status || streamCompletionStatus || '');
              streamWasTruncated = Boolean(
                p.truncated || streamCompletionStatus === 'truncated' || streamServerTurnStatus === 'truncated'
              );
            }
            const reasoningResolution = p.reasoning_resolution || p.retrieval_meta?.reasoning;
            if (reasoningResolution && typeof reasoningResolution === 'object') {
              streamReasoningResolutionRef.current = reasoningResolution;
            }
            const visualVerification = getNumericTableVisualVerification(p.retrieval_meta);
            if (visualVerification) streamVisualVerificationRef.current = visualVerification;
            const intentDecision = p.retrieval_meta?.intent_decision || p.intent_decision;
            if (intentDecision && typeof intentDecision === 'object') {
              streamIntentDecisionRef.current = intentDecision;
            }
            const recoveredFinalContent = typeof p.final_content === 'string'
              ? p.final_content.trim()
              : '';
            if (p.error && p.type !== 'retrieval_progress' && !recoveredFinalContent) {
              if (isEmbeddingIdentityConflictDetail(p.error)) {
                streamEmbeddingIdentityConflict = true;
                currentText = '';
                contentStream.replace('');
                streamTerminalFailed = true;
                streamTerminalReceived = true;
                streamServerTurnStatus = 'failed';
                sseDone = true;
                return;
              }
              const em = `❌ ${p.error}`;
              currentText = em;
              contentStream.replace(em);
              markAnswerStarted();
              streamTerminalFailed = true;
              streamTerminalReceived = true;
              streamServerTurnStatus = 'failed';
              sseDone = true;
              return;
            }
            if (p.type === 'retrieval_progress') {
              // Agent 进度先进入结构化 trace，再决定是否保留为普通思考阶段文本。
              // 这样 Agent 模式下“完成”等共用 phase 不会和时间线重复展示。
              if (!streamAgentTraceRef.current) {
                streamAgentTraceRef.current = createInitialAgentTrace();
              }
              const wasEnabled = Boolean(streamAgentTraceRef.current.enabled);
              applyAgentTraceEvent(streamAgentTraceRef.current, p);
              if (streamAgentTraceRef.current.enabled) {
                publishLiveAgentTrace(tempMsgId, streamAgentTraceRef.current);
              }
              if (
                p.phase === 'tool_result'
                && String(p.tool || '').trim().toLowerCase() === 'web_search'
                && String(p.query || '').trim()
              ) {
                streamWebSearchQueryRef.current = String(p.query).replace(/\s+/g, ' ').trim().slice(0, 240);
                setMessages(prev => prev.map(m => (
                  m.id === tempMsgId
                    ? { ...m, webSearchQuery: streamWebSearchQueryRef.current }
                    : m
                )));
              }
              const agentTraceEnabled = Boolean(streamAgentTraceRef.current.enabled);
              const thinkingStageEvent = formatThinkingStageEvent(p, { agentTraceEnabled });
              if (thinkingStageEvent) {
                appendRetrievalProgress(thinkingStageEvent);
              }
              // 只有首个可见正文已经开始生成时才结束思考。结构化引文阶段
              // 仍可能只是在生成隐藏协议，不能提前制造“思考完成后的空档”。
              if (p.phase === 'answer_generating') {
                markThinkingComplete(
                  typeof p.message === 'string' && p.message.trim()
                    ? p.message.trim()
                    : '思考完成，正在生成回答...'
                );
              }
              const justEnabled = agentTraceEnabled && !wasEnabled;
              if (agentTraceEnabled && shouldCommitAgentTrace(p.phase, justEnabled)) {
                commitAgentTrace();
              }
              return;
            }
            const thinkingStageEvent = formatThinkingStageEvent(p);
            if (thinkingStageEvent) {
              appendThinkingStage(thinkingStageEvent.text, thinkingStageEvent.key);
            }
            if (p.type === 'web_search_status') {
              if (p.audit && typeof p.audit === 'object') {
                streamWebSearchAuditRef.current = p.audit;
              }
              const executedWebQuery = typeof p.query === 'string'
                ? p.query.replace(/\s+/g, ' ').trim().slice(0, 260)
                : '';
              if (executedWebQuery) {
                streamWebSearchQueryRef.current = executedWebQuery;
              }
              streamWebSearchStatusRef.current = {
                phase: p.phase,
                count: p.count ?? null,
                query: executedWebQuery,
              };
              setMessages(prev => prev.map(m =>
                m.id === tempMsgId
                  ? {
                    ...m,
                    webSearchStatus: streamWebSearchStatusRef.current,
                    webSearchQuery: streamWebSearchQueryRef.current || m.webSearchQuery || '',
                    webSearchAudit: streamWebSearchAuditRef.current,
                  }
                  : m
              ));
              return;
            }
            if (p.type === 'web_search') {
              streamWebSearchRef.current = p.sources || [];
              setMessages(prev => prev.map(m =>
                m.id === tempMsgId
                  ? { ...m, webSearchSources: streamWebSearchRef.current }
                  : m
              ));
              return;
            }
            if (p.type === 'web_search_read') {
              streamWebSearchReadsRef.current = Array.isArray(p.reads) ? p.reads : [];
              setMessages(prev => prev.map(m =>
                m.id === tempMsgId
                  ? { ...m, webSearchReads: streamWebSearchReadsRef.current }
                  : m
              ));
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
            if (p.type === 'answer_critic') {
              const critic = p.critic || null;
              streamAnswerCriticRef.current = critic
                ? {
                    ...critic,
                    // Prefer top-level certainty event payload when present.
                    certainty: p.certainty || critic.certainty || null,
                  }
                : null;
              if (p.certainty) {
                streamAnswerCertaintyRef.current = p.certainty;
              } else if (critic?.certainty) {
                streamAnswerCertaintyRef.current = critic.certainty;
              }
              return;
            }
            if (p.type === 'answer_certainty') {
              streamAnswerCertaintyRef.current = p.certainty || null;
              return;
            }
            if (p.type === 'thinking_complete') {
              markThinkingComplete(
                typeof p.message === 'string' && p.message.trim()
                  ? p.message.trim()
                  : '思考完成，正在生成回答...'
              );
              return;
            }
            if (recoveredFinalContent) {
              streamFinalContentRef.current = p.final_content;
            }
            if (p.type === 'citation_enhanced') {
              if (typeof p.enhanced_answer === 'string' && p.enhanced_answer.trim()) {
                streamFinalContentRef.current = p.enhanced_answer;
              }
              return;
            }
            const delta = p.choices?.[0]?.delta || {};
            const cc = extractSseVisibleContent(p, delta);
            const ct = extractSseReasoningText(p, delta);
            if (!p.done && !p.choices?.[0]?.finish_reason) {
              if (ct) {
                if (!appendRealThinking(ct)) {
                  appendThinkingStage('模型正在准备思考内容...', 'model:preparing_reasoning');
                }
              }
              if (cc) {
                appendAnswerContent(cc);
              } else if (!ct) {
                appendThinkingStage('正在等待模型输出思考内容...', 'model:waiting_reasoning');
              }
            } else {
              if (ct) {
                if (!appendRealThinking(ct)) {
                  appendThinkingStage('模型正在准备思考内容...', 'model:preparing_reasoning');
                }
              }
              const finalContentFromEvent = typeof p.final_content === 'string' ? p.final_content : '';
              if (finalContentFromEvent.trim()) {
                streamFinalContentRef.current = finalContentFromEvent;
                if (finalContentFromEvent.startsWith(currentText)) {
                  const finalDelta = finalContentFromEvent.slice(currentText.length);
                  if (finalDelta) {
                    appendAnswerContent(finalDelta);
                  }
                }
              } else if (cc) {
                appendAnswerContent(cc);
              }
              if (p.retrieval_meta?.citations) streamCitationsRef.current = p.retrieval_meta.citations;
              if (Object.prototype.hasOwnProperty.call(p.retrieval_meta || {}, 'citation_bindings')) {
                streamCitationBindingsRef.current = p.retrieval_meta.citation_bindings || null;
              }
              if (Object.prototype.hasOwnProperty.call(p, 'visual_attachments')) {
                streamVisualAttachmentsRef.current = normalizeChatVisualAttachments(
                  p.visual_attachments,
                  requestParseIdentity,
                );
              }
              if (p.retrieval_meta?.max_relevance_score !== undefined) streamMaxRelevanceRef.current = p.retrieval_meta.max_relevance_score;
              if (p.retrieval_meta && (p.retrieval_meta.agent_mode || p.retrieval_meta.agent_search_history)) {
                if (!streamAgentTraceRef.current) {
                  streamAgentTraceRef.current = createInitialAgentTrace();
                }
                mergeAgentMetaIntoTrace(streamAgentTraceRef.current, p.retrieval_meta);
                const effectiveWebQuery = getEffectiveWebSearchQuery(p.retrieval_meta);
                if (effectiveWebQuery) streamWebSearchQueryRef.current = effectiveWebQuery;
                if (!streamAgentTraceRef.current.endedAt) streamAgentTraceRef.current.endedAt = Date.now();
              }
              if (p.qa_score !== undefined) streamQaScoreRef.current = p.qa_score;
              if (p.web_search_sources) streamWebSearchRef.current = p.web_search_sources;
              if (Array.isArray(p.web_search_reads)) streamWebSearchReadsRef.current = p.web_search_reads;
              if (Object.prototype.hasOwnProperty.call(p, 'web_search_audit')) {
                streamWebSearchAuditRef.current = p.web_search_audit || null;
              } else if (p.retrieval_meta?.web_search_audit) {
                streamWebSearchAuditRef.current = p.retrieval_meta.web_search_audit;
              }
              if (Object.prototype.hasOwnProperty.call(p, 'memory_hits')) streamMemoryHitsRef.current = p.memory_hits;
              if (Object.prototype.hasOwnProperty.call(p, 'memory_meta')) streamMemoryMetaRef.current = p.memory_meta;
              if (p.usage_meta || p.usage) streamUsageRef.current = p.usage_meta || p.usage;
              if (p.used_provider || p.used_model || p.fallback_used !== undefined) {
                const previousCallInfo = streamCallInfoRef.current;
                streamCallInfoRef.current = {
                  provider: p.used_provider || previousCallInfo?.provider || chatProvider,
                  model: p.used_model || previousCallInfo?.model || chatModel,
                  fallback: p.fallback_used ?? previousCallInfo?.fallback,
                  usage: p.usage_meta || p.usage || streamUsageRef.current || null,
                };
                if (
                  streamCallInfoRef.current.provider !== previousCallInfo?.provider
                  || streamCallInfoRef.current.model !== previousCallInfo?.model
                ) {
                  setMessages(prev => prev.map(m => (
                    m.id === tempMsgId
                      ? {
                        ...m,
                        provider: streamCallInfoRef.current.provider,
                        model: streamCallInfoRef.current.model,
                      }
                      : m
                  )));
                }
              }
              // ``done`` 表示主答案已完成；后端还会继续发送追问、会话名、
              // 答案自审等收尾事件。传输层只由 [DONE] 或 reader EOF 结束。
            }
          } catch (e) {
            console.error(e, data);
          }
        };

        // 读取流数据
        let reading = true;
        while (reading) {
          const { value, done } = await reader.read();
          if (done || requestAbortState.cancelled || !isRequestCurrent()) break;
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

        if (!isRequestCurrent()) return;
        if (streamResponseIdentityMismatch || !streamTerminalIdentityVerified) {
          setMessages((previous) => previous.map((message) => (
            message.id === tempMsgId
              ? {
                ...message,
                content: CHAT_PARSE_IDENTITY_UPDATED_MESSAGE,
                isStreaming: false,
                turnStatus: 'interrupted',
                parseIdentityStale: true,
              }
              : message
          )));
          activeStreamMsgIdRef.current = null;
          setStreamingMessageId(null);
          return;
        }
        if (streamEmbeddingIdentityConflict) {
          setContentStreamDone(true);
          setThinkingStreamDone(true);
          setMessages((previous) => previous.map((message) => (
            message.id === tempMsgId
              ? {
                ...message,
                content: '',
                isStreaming: false,
                turnStatus: 'failed',
                embeddingIdentityConflict: true,
                embeddingIdentityConflictDetail: EMBEDDING_IDENTITY_CONFLICT_MESSAGE,
              }
              : message
          )));
          activeStreamMsgIdRef.current = null;
          setStreamingMessageId(null);
          return;
        }

        // 流结束，标记正文 streamDone 触发短暂的自适应冲刷。思考流在
        // 正文首个可见 token 到达时独立完成；异常终止则在此兜底。
        setContentStreamDone(true);
        finalizeThinkingStream();
        const streamedContent = [streamFinalContentRef.current, currentText]
          .find((value) => typeof value === 'string' && value.trim())
          || '❌ AI未返回正文，请重新生成';
        const streamTurnStatus = (
          streamTerminalFailed
          || streamServerTurnStatus === 'failed'
          || streamedContent.startsWith('❌')
            ? 'failed'
            : (
              requestAbortState.cancelled || (!streamTerminalReceived && !sseDone)
                ? 'interrupted'
                : normalizeChatTurnStatus(
                  streamServerTurnStatus,
                  hasCompleteChatParseIdentity(requestParseIdentity) ? 'interrupted' : 'completed'
                )
            )
        );
        const waitForNextPaint = () => new Promise((resolve) => {
          const fallbackTimer = setTimeout(resolve, 64);
          requestAnimationFrame(() => {
            clearTimeout(fallbackTimer);
            resolve();
          });
        });

        // 在切换到最终 Markdown 渲染前，让正文与思考两个 ref 直写队列都排空。
        // 两者并行推进，不阻塞正文首 token；只有队列连续无进展时才强制收尾，
        // 避免后台标签页或异常 rAF 让消息永久停在流式状态。
        {
          const drainTargets = [
            { stream: contentStream, finalText: streamedContent },
            ...(currentThinking
              ? [{ stream: thinkingStream, finalText: currentThinking }]
              : []),
          ];
          const allStreamsFlushed = () => drainTargets.every(
            ({ stream }) => stream.isFlushComplete?.() !== false
          );
          const getPendingStreamChars = () => drainTargets.reduce(
            (total, { stream }) => total + Math.max(0, Number(stream.getPendingChars?.()) || 0),
            0
          );
          let previousPendingChars = getPendingStreamChars();
          let lastProgressAt = Date.now();
          while (
            isRequestCurrent() &&
            !allStreamsFlushed()
          ) {
            await waitForNextPaint();
            const pendingChars = getPendingStreamChars();
            if (pendingChars < previousPendingChars) {
              lastProgressAt = Date.now();
            }
            previousPendingChars = pendingChars;
            if (Date.now() - lastProgressAt >= STREAM_FINAL_FLUSH_GRACE_MS) break;
          }
          drainTargets.forEach(({ stream, finalText }) => {
            if (stream.isFlushComplete?.() === false) {
              stream.flushNow?.(finalText);
            }
          });
        }
        if (!isRequestCurrent()) return;
        // 直写模式下最后一批字符已经进入 DOM，但其模糊渐显仍在播放。
        // 只等待正文自己的动画尾巴；思考动画绝不能延迟最终回答。
        const revealShouldContinue = () => isRequestCurrent();
        await contentStream.waitForRevealComplete?.(revealShouldContinue);
        if (!isRequestCurrent()) return;
        const finalThinkingMs = finalizeThinkingDurationMs({
          thinkingStartTime,
          thinkingLastUpdateTime,
          contentStartTime,
        });
        const { content: finalContent, citations: finalCitations } = finalizeAssistantContentAndCitations(
          streamedContent,
          streamCitationsRef.current
        );
        const streamVisualVerification = (
          visualVerificationEpochRef.current === requestVisualVerificationEpoch
            ? streamVisualVerificationRef.current
            : null
        );
        const streamIntentDecision = streamIntentDecisionRef.current;
        if (streamCallInfoRef.current) {
          setLastCallInfo({
            ...streamCallInfoRef.current,
            usage: streamUsageRef.current || streamCallInfoRef.current.usage || null,
          });
        }
        setMessages(prev => prev.map(m =>
           m.id === tempMsgId
             ? { ...m, content: finalContent, thinking: currentThinking, thinkingLive: hasRealThinking, reasoningContent: hasRealThinking ? currentThinking : '', retrievalProgress: currentRetrievalProgress, isStreaming: false, thinkingMs: finalThinkingMs, turnStatus: streamTurnStatus, finishReason: streamFinishReason, completionStatus: streamCompletionStatus, truncated: streamWasTruncated, citations: finalCitations, citationBindings: streamCitationBindingsRef.current || null, visualAttachments: streamVisualAttachmentsRef.current || [], maxRelevanceScore: streamMaxRelevanceRef.current, qaScore: streamQaScoreRef.current, followupQuestions: streamFollowupRef.current || null, convName: streamConvNameRef.current || null, mindmapMarkdown: streamMindmapRef.current || null, answerCritic: streamAnswerCriticRef.current || null, answerCertainty: streamAnswerCertaintyRef.current || streamAnswerCriticRef.current?.certainty || null, webSearchSources: streamWebSearchRef.current || null, webSearchReads: streamWebSearchReadsRef.current || [], webSearchAudit: streamWebSearchAuditRef.current || null, webSearchQuery: streamWebSearchQueryRef.current || m.webSearchQuery || '', webSearchStatus: null, memoryHits: streamMemoryHitsRef.current || null, memoryMeta: streamMemoryMetaRef.current || null, agentTrace: streamAgentTraceRef.current && streamAgentTraceRef.current.enabled ? cloneAgentTrace(streamAgentTraceRef.current) : null, usage: streamUsageRef.current || null, intentDecision: streamIntentDecision || null, reasoningResolution: streamReasoningResolutionRef.current || null, clarificationRequired: Boolean(streamIntentDecision?.is_ambiguous), ...(streamVisualVerification ? { visualVerification: streamVisualVerification } : {}) }
            : m
        ));
        startVisualVerificationPolling(
          tempMsgId,
          streamVisualVerification,
          requestVisualVerificationEpoch
        );
        activeStreamMsgIdRef.current = null;
        setStreamingMessageId(null);
      } else {
        // ===== 非流式输出模式 =====
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal: requestController.signal,
        });

        if (!response.ok) {
          let ed = `HTTP ${response.status}`;
          try {
            const eb = await response.json();
            ed = eb.detail || eb.error?.message || eb.message || JSON.stringify(eb);
          } catch (e) { /* ignore */ }
          if (
            response.status === 409
            && isChatParseIdentityConflict(response, null, requestParseIdentity, ed)
          ) {
            const identityError = new Error(ed || '文档解析结果已更新，本次回答已停止，请重新提问');
            identityError.name = 'ChatParseIdentityError';
            throw identityError;
          }
          if (response.status === 409 && isEmbeddingIdentityConflictDetail(ed)) {
            throw createEmbeddingIdentityConflictError(ed);
          }
          throw new Error(ed);
        }

        const data = await response.json();
        if (!isRequestCurrent()) return;
        if (!responseMatchesChatParseIdentity(response, data, requestParseIdentity)) {
          const identityError = new Error('文档解析结果已更新，本次回答已停止，请重新提问');
          identityError.name = 'ChatParseIdentityError';
          throw identityError;
        }
        const hasNonStreamAnswer = typeof data.answer === 'string' && data.answer.trim();
        const nonStreamTurnStatus = hasNonStreamAnswer
          ? normalizeChatTurnStatus(
            data.turn_status
            || data.answer_status
            || response?.headers?.get?.('X-Chat-Turn-Status'),
            hasCompleteChatParseIdentity(requestParseIdentity) ? 'interrupted' : 'completed'
          )
          : 'failed';
        const nonStreamAnswer = hasNonStreamAnswer
          ? data.answer
          : '❌ AI未返回正文，请重新生成';
        const { content: finalContent, citations: finalCitations } = finalizeAssistantContentAndCitations(
          nonStreamAnswer,
          data.retrieval_meta?.citations
        );
        const nonStreamVisualVerification = (
          visualVerificationEpochRef.current === requestVisualVerificationEpoch
            ? getNumericTableVisualVerification(data.retrieval_meta)
            : null
        );
        const nonStreamIntentDecision = data.retrieval_meta?.intent_decision || data.intent_decision || null;
        const nonStreamReasoningResolution = data.reasoning_resolution || data.retrieval_meta?.reasoning || null;
        const nonStreamVisualAttachments = normalizeChatVisualAttachments(
          data.visual_attachments,
          requestParseIdentity,
        );
        let nonStreamAgentTrace = null;
        if (data.retrieval_meta && (data.retrieval_meta.agent_mode || data.retrieval_meta.agent_gate)) {
          nonStreamAgentTrace = createInitialAgentTrace();
          mergeAgentMetaIntoTrace(nonStreamAgentTrace, data.retrieval_meta);
          nonStreamAgentTrace.enabled = Boolean(
            data.retrieval_meta.agent_mode ||
            data.retrieval_meta.agent_gate?.use_agent ||
            data.retrieval_meta.agent_gate?.requested_enabled
          );
        }
        const nonStreamWebSearchQuery = getEffectiveWebSearchQuery(data.retrieval_meta)
          || String(data.web_search_query || '').replace(/\s+/g, ' ').trim().slice(0, 240);
        setLastCallInfo({ provider: data.used_provider, model: data.used_model, fallback: data.fallback_used, usage: data.usage_meta || data.usage || null });
        setMessages(prev => prev.map(m =>
          m.id === tempMsgId
             ? { ...m, provider: data.used_provider || m.provider || chatProvider, model: data.used_model || m.model || chatModel, content: finalContent, thinking: data.reasoning_content || '', reasoningContent: data.reasoning_content || '', isStreaming: false, turnStatus: nonStreamTurnStatus, finishReason: data.finish_reason || '', completionStatus: data.completion_status || '', truncated: Boolean(data.truncated || nonStreamTurnStatus === 'truncated'), citations: finalCitations, citationBindings: data.retrieval_meta?.citation_bindings || null, visualAttachments: nonStreamVisualAttachments, webSearchSources: data.web_search_sources || null, webSearchReads: Array.isArray(data.web_search_reads) ? data.web_search_reads : (Array.isArray(data.retrieval_meta?.web_search_reads) ? data.retrieval_meta.web_search_reads : []), webSearchAudit: data.web_search_audit || data.retrieval_meta?.web_search_audit || null, webSearchQuery: nonStreamWebSearchQuery || m.webSearchQuery || '', memoryHits: data.memory_hits || null, memoryMeta: data.memory_meta || null, agentTrace: nonStreamAgentTrace, usage: data.usage_meta || data.usage || null, intentDecision: nonStreamIntentDecision, reasoningResolution: nonStreamReasoningResolution, clarificationRequired: Boolean(data.clarification_required || nonStreamIntentDecision?.is_ambiguous), answerCertainty: data.answer_certainty || data.retrieval_meta?.answer_certainty || null, answerCritic: data.answer_critic || null, ...(nonStreamVisualVerification ? { visualVerification: nonStreamVisualVerification } : {}) }
            : m
        ));
        startVisualVerificationPolling(
          tempMsgId,
          nonStreamVisualVerification,
          requestVisualVerificationEpoch
        );
        activeStreamMsgIdRef.current = null;
        setStreamingMessageId(null);
      }
    } catch (error) {
      if (!isRequestCurrent()) return;
      if (error.name === 'EmbeddingIdentityConflictError' || isEmbeddingIdentityConflictDetail(error.message)) {
        setContentStreamDone(true);
        setThinkingStreamDone(true);
        setMessages((previous) => previous.map((message) => (
          message.id === tempMsgId
            ? {
              ...message,
              content: '',
              isStreaming: false,
              turnStatus: 'failed',
              embeddingIdentityConflict: true,
              embeddingIdentityConflictDetail: String(error.detail || error.message || ''),
            }
            : message
        )));
        activeStreamMsgIdRef.current = null;
        setStreamingMessageId(null);
        return;
      }
      if (error.name === 'ChatParseIdentityError') {
        setMessages((previous) => previous.map((message) => (
          message.id === tempMsgId
            ? {
              ...message,
              content: `❌ ${error.message}`,
              isStreaming: false,
              turnStatus: 'interrupted',
              parseIdentityStale: true,
            }
            : message
        )));
        activeStreamMsgIdRef.current = null;
        setStreamingMessageId(null);
        return;
      }
      if (error.name === 'AbortError' && !firstEventTimeoutTriggered) {
        setMessages(prev => prev.map(m =>
          m.id === tempMsgId ? { ...m, isStreaming: false, turnStatus: 'interrupted' } : m
        ));
        activeStreamMsgIdRef.current = null;
        setStreamingMessageId(null);
        return;
      }
      const errorMessage = firstEventTimeoutTriggered
        ? `首包超时（${STREAM_FIRST_EVENT_TIMEOUT_MS}ms），请重试或切换模型`
        : error.message;
      setContentStreamDone(true);
      setThinkingStreamDone(true);
      activeStreamMsgIdRef.current = null;
      setStreamingMessageId(null);
      setMessages(prev => prev.map(m =>
          m.id === tempMsgId
            ? { ...m, content: '❌ ' + errorMessage, isStreaming: false, turnStatus: 'failed' }
          : m
      ));
    } finally {
      if (abortControllerRef.current === requestController) {
        abortControllerRef.current = null;
      }
      if (isRequestCurrent()) setIsLoading(false);
    }
  }, [
    docId, parseIdentityReady, screenshots, setScreenshots, selectedText, messages, streamSpeed, enableVectorSearch,
    enableGraphRAG, enableAgentRetrieval, forceAgentRetrieval,
    enableJiebaBM25, numExpandContextChunk,
    getChatCredentials, getVisualCredentials, getProviderById, contentStream, thinkingStream,
    maxTokens, temperature, topP, contextCount, streamOutput,
    enableTemperature, enableTopP, enableMaxTokens, customParams,
    reasoningEffort, answerDetailLevel, enableMemory, memoryTopK, memoryInjectionBudget, memoryPrivacyMode,
    webSearchMode, enableWebSearch, webSearchProvider, webSearchApiKey, webSearchBlacklist,
    webSearchIncludeDocumentContext, embeddingApiKey, getEmbeddingConfig,
    streamRenderProfile, shouldUseStreaming,
    overrideNumericTable, overrideAnswerCritic, overrideAnswerClaimVerifier, overrideLLMQueryRewrite, overrideBM25Synonyms,
    numericTableVisualVerification, cheapModel, cheapModelProvider, cheapModelEndpoint,
    interruptActiveRequest, startVisualVerificationPolling, streamingMessageId,
  ]);

  /**
   * 停止当前流式输出
   */
  const handleStop = useCallback(() => {
    interruptActiveRequest();
  }, [interruptActiveRequest]);

  /**
   * 重新生成指定位置的消息
   * @param {number} index - 消息索引
   */
  const regenerateMessage = useCallback(async (index) => {
    if (!docId) { alert('请先上传文档'); return; }
    // 找到 index 前最近一条用户消息的索引
    let userMsgIndex = -1;
    for (let i = index - 1; i >= 0; i--) {
      if (messages[i].type === 'user') { userMsgIndex = i; break; }
    }
    if (userMsgIndex === -1) return;
    const userMsg = messages[userMsgIndex];
    const historyMessages = messages.slice(0, userMsgIndex);
    // 显式把原问题和截断后的历史交给 sendMessage，避免 React 状态更新前
    // 的旧闭包再次携带被删除的空回答或后续消息。
    setMessages(historyMessages);
    setInputValue('');
    await sendMessage({ input: userMsg.content, historyMessages });
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
    subscribeContentCommittedPrefix: contentStream.subscribeCommittedPrefix,
    subscribeContentDisplayedText: contentStream.subscribeDisplayedText,

    // Refs
    abortControllerRef,
    textareaRef,

    // 方法
    sendMessage,
    handleStop,
    regenerateMessage,
    copyMessage,
    saveToMemory,
    setInputValue,
    invalidateVisualVerificationState,
  };
}
