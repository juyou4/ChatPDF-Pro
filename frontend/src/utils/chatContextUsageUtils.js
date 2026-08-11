const CHAT_HISTORY_ERROR_PREFIXES = ['⚠️ AI未返回内容', '❌'];
const RETRIEVAL_FAILURE_PREFIXES = [
  '根据提供的文档片段，无法回答',
  '根据当前检索到的文档片段，无法回答',
  '当前检索到的文档片段不足以回答',
  '当前可用的文档片段不足以回答',
];

const isOperationalRetrievalFailure = (content) => {
  if (RETRIEVAL_FAILURE_PREFIXES.some((prefix) => content.startsWith(prefix))) return true;
  const hasRetrievalScope = /(?:当前|提供的|可用的).{0,12}(?:检索|文档).{0,8}(?:片段|内容|证据)/.test(content);
  const hasCoverageFailure = /(?:仅|只)包含.{0,80}(?:表格|参考文献|标题)|不足以回答|未涉及.{0,40}(?:核心方法|方法原理|训练目标)/.test(content);
  return hasRetrievalScope && hasCoverageFailure;
};

export const isFailedChatHistoryAssistant = (message) => {
  if (message?.type !== 'assistant') return false;
  const turnStatus = String(
    message.turnStatus
    || message.turn_status
    || message.answerStatus
    || message.answer_status
    || ''
  ).toLowerCase();
  if (
    message.parseIdentityStale === true
    || [
      'evidence_fallback',
      'degraded',
      'truncated',
      'failed',
      'interrupted',
      'cancelled',
      'aborted',
      'streaming',
    ].includes(turnStatus)
  ) {
    return true;
  }
  const critic = message.answerCritic || message.answer_critic;
  if (
    critic?.has_hallucination === true
    || String(critic?.citation_risk_level || '').toLowerCase() === 'high'
  ) return true;

  const content = String(message.content || '').trim();
  return !content
    || CHAT_HISTORY_ERROR_PREFIXES.some((prefix) => content.startsWith(prefix))
    || isOperationalRetrievalFailure(content);
};

const isEligibleChatHistoryMessage = (message) => {
  if (!message || (message.type !== 'user' && message.type !== 'assistant')) return false;
  if (message.hasImage) return false;

  const content = String(message.content || '').trim();
  if (!content) return false;
  return !isFailedChatHistoryAssistant(message);
};

const buildEligibleChatTurns = (messages = []) => {
  const turns = [];
  let pendingUser = null;

  for (const message of Array.isArray(messages) ? messages : []) {
    if (message?.type === 'user') {
      pendingUser = message;
      continue;
    }
    if (message?.type !== 'assistant') continue;

    if (isEligibleChatHistoryMessage(message)) {
      if (isEligibleChatHistoryMessage(pendingUser)) {
        turns.push([pendingUser, message]);
      } else if (pendingUser?.hasImage) {
        // 图片本身不能进入文本上下文，但模型已产出的文字结论仍可复用。
        turns.push([{ ...message, historyKind: 'image_summary' }]);
      }
    }
    // 空答/错误答意味着整轮失败，对应 user 也不能孤立进入后续上下文。
    pendingUser = null;
  }

  return turns;
};

/**
 * 构建每次请求会附带的聊天历史，需与接口调用保持一致。
 */
export const buildChatHistory = (messages = [], contextCount, {
  preserveReasoning = false,
  providerId = '',
  modelId = '',
} = {}) => {
  const normalizedCount = Math.max(0, Math.floor(Number(contextCount) || 0));
  if (normalizedCount <= 0 || !Array.isArray(messages)) return [];

  return buildEligibleChatTurns(messages)
    .slice(-normalizedCount)
    .flat()
    .map((message) => {
      const payload = {
        role: message.type === 'user' ? 'user' : 'assistant',
        content: message.type === 'user' && String(message.contextContent || '').trim()
          ? message.contextContent
          : message.content,
      };
      if (message.historyKind) payload.history_kind = message.historyKind;
      const sameProvider = !message.provider || !providerId || message.provider === providerId;
      const sameModel = !message.model || !modelId || message.model === modelId;
      const reasoningContent = String(message.reasoningContent || message.reasoning_content || '').trim();
      if (
        preserveReasoning
        && payload.role === 'assistant'
        && sameProvider
        && sameModel
        && reasoningContent
      ) {
        payload.reasoning_content = reasoningContent;
      }
      return payload;
    });
};

/**
 * 轻量级本地估算，仅用于界面透明展示，不替代供应商返回的真实 token 使用量。
 */
export const estimateContextTokens = (value) => {
  const text = String(value || '').trim();
  if (!text) return 0;

  const cjkChars = (text.match(/[\u3400-\u9fff\uf900-\ufaff]/g) || []).length;
  const otherChars = text.replace(/[\u3400-\u9fff\uf900-\ufaff]/g, '');
  return Math.max(1, Math.ceil(cjkChars * 1.15 + otherChars.length / 4));
};

export const getChatHistoryUsage = (messages = [], contextCount) => {
  const normalizedCount = Math.max(0, Math.floor(Number(contextCount) || 0));
  const eligibleMessages = buildEligibleChatTurns(messages).flat();
  const history = buildChatHistory(messages, normalizedCount);
  const selectedTokens = history.reduce(
    (total, message) => total
      + estimateContextTokens(message.content)
      + estimateContextTokens(message.reasoning_content),
    0,
  );

  return {
    history,
    selectedTokens,
    selectedMessageCount: history.length,
    eligibleMessageCount: eligibleMessages.length,
    selectedTurns: Math.ceil(history.length / 2),
    configuredTurns: normalizedCount,
    omittedMessageCount: Math.max(0, eligibleMessages.length - history.length),
  };
};

export const formatCompactTokenCount = (value) => {
  const tokens = Math.max(0, Number(value) || 0);
  if (tokens < 1000) return `${Math.round(tokens)}`;
  if (tokens < 10000) return `${(tokens / 1000).toFixed(1)}k`;
  return `${Math.round(tokens / 1000)}k`;
};
