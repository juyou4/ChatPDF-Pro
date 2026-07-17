const CHAT_HISTORY_ERROR_PREFIXES = ['⚠️ AI未返回内容', '❌'];

const isEligibleChatHistoryMessage = (message) => {
  if (!message || (message.type !== 'user' && message.type !== 'assistant')) return false;
  if (message.hasImage) return false;

  const content = String(message.content || '');
  return !(
    message.type === 'assistant'
    && CHAT_HISTORY_ERROR_PREFIXES.some((prefix) => content.startsWith(prefix))
  );
};

/**
 * 构建每次请求会附带的聊天历史，需与接口调用保持一致。
 */
export const buildChatHistory = (messages = [], contextCount) => {
  const normalizedCount = Math.max(0, Math.floor(Number(contextCount) || 0));
  if (normalizedCount <= 0 || !Array.isArray(messages)) return [];

  return messages
    .filter(isEligibleChatHistoryMessage)
    .slice(-(normalizedCount * 2))
    .map((message) => ({
      role: message.type === 'user' ? 'user' : 'assistant',
      content: message.content,
    }));
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
  const eligibleMessages = Array.isArray(messages)
    ? messages.filter(isEligibleChatHistoryMessage)
    : [];
  const history = buildChatHistory(messages, normalizedCount);
  const selectedTokens = history.reduce(
    (total, message) => total + estimateContextTokens(message.content),
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
