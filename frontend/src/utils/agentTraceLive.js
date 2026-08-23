/**
 * Agent 检索轨迹的流式总线。
 * 面板用 rAF 订阅；消息列表只在阶段切换时写入 React state。
 */

const listeners = new Map();

export const cloneAgentTrace = (trace) => {
  if (!trace || typeof trace !== 'object') return null;
  return {
    ...trace,
    rounds: Array.isArray(trace.rounds)
      ? trace.rounds.map((round) => ({
        ...round,
        operations: (round?.operations || []).map((operation) => ({ ...operation })),
      }))
      : [],
    searchHistory: Array.isArray(trace.searchHistory) ? [...trace.searchHistory] : [],
    agentDetail: Array.isArray(trace.agentDetail) ? [...trace.agentDetail] : [],
    taskStatus: trace.taskStatus && typeof trace.taskStatus === 'object'
      ? {
        ...trace.taskStatus,
        completed: [...(trace.taskStatus.completed || [])],
        pending: [...(trace.taskStatus.pending || [])],
      }
      : trace.taskStatus,
  };
};

export const subscribeLiveAgentTrace = (messageId, onTrace) => {
  const id = String(messageId || '');
  if (!id || typeof onTrace !== 'function') return () => {};
  let bucket = listeners.get(id);
  if (!bucket) {
    bucket = new Set();
    listeners.set(id, bucket);
  }
  bucket.add(onTrace);
  return () => {
    bucket.delete(onTrace);
    if (bucket.size === 0) listeners.delete(id);
  };
};

export const publishLiveAgentTrace = (messageId, trace) => {
  const bucket = listeners.get(String(messageId || ''));
  if (!bucket || bucket.size === 0) return;
  bucket.forEach((listener) => listener(trace));
};

export const resetLiveAgentTraceListeners = () => {
  listeners.clear();
};

export const AGENT_TRACE_COMMIT_PHASES = new Set([
  'agent_start',
  'agent_mode',
  'round_start',
  'tool_result',
  'complete',
  'planner_error',
]);

export const shouldCommitAgentTrace = (phase, justEnabled = false) => (
  Boolean(justEnabled) || AGENT_TRACE_COMMIT_PHASES.has(String(phase || ''))
);
