import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Check,
  ChevronDown,
  ChevronRight,
  FileText,
  Globe,
  Hash,
  Layers,
  Loader2,
  Map,
  ScanSearch,
  Search,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Wrench,
} from 'lucide-react';

const PlanningThoughtIcon = ({ className = '' }) => (
  <svg
    className={className}
    viewBox="0 0 64 64"
    fill="none"
    aria-hidden="true"
    focusable="false"
  >
    <path
      fill="currentColor"
      d="M25.5 44.2c-4.9 0-9.6-2.5-12.4-6.6C8 35 4.8 29.8 4.8 24.2c0-8.3 6.8-15.1 15.1-15.2c.7 0 1.4 0 2.1.1C24.5 4 29.7.7 35.5.7c6.7 0 12.6 4.5 14.4 10.9c5.6 2.3 9.3 7.7 9.3 13.9c0 8.3-6.7 15-15 15c-2.5 0-4.9-.6-7.1-1.8c-2.9 3.4-7.1 5.4-11.6 5.5m-5.6-30.7c-5.8.1-10.6 4.8-10.6 10.7c0 4.1 2.5 7.9 6.2 9.6c.4.2.7.5 1 .9c1.9 3.1 5.3 5 8.9 5c3.7-.1 7.1-2 9.1-5.1c.3-.5.9-.9 1.5-1s1.3 0 1.8.4c1.9 1.4 4.1 2.2 6.4 2.2c5.8 0 10.5-4.7 10.5-10.5c0-4.7-3-8.7-7.4-10.1c-.8-.2-1.4-.9-1.5-1.7c-1-4.9-5.3-8.5-10.3-8.5c-4.5 0-8.5 2.9-10 7.2c-.4 1.1-1.6 1.8-2.7 1.5c-1-.5-2-.6-2.9-.6M40.7 56c-3.6 0-6.6-3-6.6-6.6s3-6.6 6.6-6.6s6.6 3 6.6 6.6s-2.9 6.6-6.6 6.6m0-8.7c-1.2 0-2.1.9-2.1 2.1s.9 2.1 2.1 2.1s2.1-.9 2.1-2.1c.1-1.2-.9-2.1-2.1-2.1m11.9 16c-3 0-5.4-2.4-5.4-5.4s2.4-5.4 5.4-5.4c1.5 0 2.9.6 3.9 1.6S58 56.5 58 58v.2c-.2 2.8-2.5 5.1-5.4 5.1m0-6.3c-.5 0-.9.4-.9.9s.4.9.9.9s.8-.4.9-.9c0-.3-.1-.4-.2-.5c-.1-.2-.4-.4-.7-.4"
    />
  </svg>
);

const TOOL_META = {
  search_document: { label: '统一检索', icon: ScanSearch, family: 'search' },
  web_search: { label: '联网检索', icon: Globe, family: 'search' },
  vector_search: { label: '向量搜索', icon: Sparkles, family: 'search' },
  keyword_search: { label: 'BM25 关键词', icon: Hash, family: 'search' },
  grep: { label: 'GREP 字面', icon: Search, family: 'search' },
  regex_search: { label: '正则匹配', icon: Search, family: 'search' },
  boolean_search: { label: '布尔逻辑', icon: Search, family: 'search' },
  visual_search: { label: '定位视觉证据', icon: ScanSearch, family: 'search' },
  read_blocks: { label: '读取原文块', icon: Layers, family: 'read' },
  read_section: { label: '读取章节', icon: FileText, family: 'read' },
  read_around: { label: '展开上下文', icon: FileText, family: 'read' },
  fetch: { label: '获取意群', icon: Layers, family: 'read' },
  map: { label: '查看文档地图', icon: Map, family: 'read' },
  analyze_visual_evidence: { label: '分析图表证据', icon: Sparkles, family: 'visual' },
  complete: { label: '结束检索', icon: Check, family: 'complete' },
};

const FAMILY_META = {
  search: { icon: Search, done: (count) => `执行了 ${count} 次搜索`, running: '正在搜索证据' },
  read: { icon: FileText, done: (count) => `读取了 ${count} 处文档内容`, running: '正在读取文档内容' },
  visual: { icon: Sparkles, done: (count) => `分析了 ${count} 个视觉证据`, running: '正在分析视觉证据' },
  complete: { icon: Check, done: () => '证据收集完成', running: '正在结束检索' },
  other: { icon: Wrench, done: (count) => `执行了 ${count} 次工具调用`, running: '正在调用工具' },
};

const AGENT_GATE_REASON_LABELS = {
  matched_query_type: '题型匹配',
  matched_evidence_need: '证据需求匹配',
  matched_visual_intent: '视觉意图匹配',
  route_not_matched: '题型未匹配',
  selected_text_present: '框选文本优先',
  switch_disabled: '开关未启用',
  stream_only: '非流式未执行',
  structural_inventory: '结构化枚举优先',
  numeric_table_exactness: '数值表精确路径',
  page_range_deterministic_scope: '页范围确定性检索',
  forced: '用户强制 Agent',
  force_user: '用户强制 Agent',
};

const TASK_LABELS = {
  qa: '问答',
  summarize: '总结',
  extract: '抽取',
  explain: '解释',
  compare: '比较',
  calculate: '计算',
  translate: '翻译',
  continue: '继续',
  inventory: '枚举',
};

const QUERY_TYPE_LABELS = {
  overview: '概览',
  extraction: '抽取',
  analytical: '分析',
  specific: '具体',
  inventory: '枚举',
};

const EVIDENCE_STATUS_LABELS = {
  answered: '证据已就绪',
  insufficient_evidence: '证据不足',
  budget_exhausted: '工具预算已用尽',
  gathering: '正在收集证据',
};

const formatDecisionStrength = (value) => {
  const strength = Number(value);
  if (!Number.isFinite(strength)) return '';
  if (strength >= 0.85) return '高';
  if (strength >= 0.55) return '中';
  return '低';
};

const normalizeText = (value, limit = 360) => String(value || '')
  .replace(/\s+/g, ' ')
  .trim()
  .slice(0, limit);

const getToolMeta = (tool) => TOOL_META[tool] || {
  label: normalizeText(tool, 80) || '工具',
  icon: Wrench,
  family: 'other',
};

const formatDuration = (startedAt, endedAt) => {
  if (!startedAt || !endedAt || endedAt < startedAt) return '';
  const ms = endedAt - startedAt;
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m${Math.round((ms % 60000) / 1000)}s`;
};

const formatElapsedMs = (value) => {
  const ms = Number(value);
  if (!Number.isFinite(ms) || ms < 0) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
};

const normalizePlanningLabel = (round, message) => {
  const text = normalizeText(message, 180);
  if (!text || /LLM\s*规划中/i.test(text)) return `规划第 ${round} 轮检索`;
  return text;
};

const operationDetailText = (operation) => {
  const query = normalizeText(operation.query, 280);
  if (query) {
    if (operation.tool === 'web_search') return `已搜索网络：${query}`;
    if (operation.tool === 'read_section') return `已读取章节：${query}`;
    if (operation.tool === 'read_around') return `已展开上下文：${query}`;
    if (operation.tool === 'map') return `已查看文档地图：${query}`;
    if (operation.tool === 'visual_search') return `已定位视觉证据：${query}`;
    if (getToolMeta(operation.tool).family === 'read') return `已读取文档：${query}`;
    if (getToolMeta(operation.tool).family === 'visual') return `已分析视觉证据：${query}`;
    return `已检索文档：${query}`;
  }
  return normalizeText(operation.resultMessage || operation.message, 280) || getToolMeta(operation.tool).label;
};

const buildActivities = (rounds, searchHistory) => {
  const history = Array.isArray(searchHistory) ? searchHistory.filter((item) => item && typeof item === 'object') : [];
  let historyCursor = 0;

  const consumeHistory = (tool) => {
    for (let index = historyCursor; index < history.length; index += 1) {
      if (!tool || history[index]?.tool === tool) {
        historyCursor = index + 1;
        return history[index];
      }
    }
    return null;
  };

  const activities = [];
  rounds.forEach((roundData, roundIndex) => {
    const round = roundData.round;
    const operations = (Array.isArray(roundData.operations) ? roundData.operations : []).map((operation) => {
      const historyItem = consumeHistory(operation?.tool);
      return {
        ...operation,
        query: operation?.query || historyItem?.query || '',
        resultCount: Number.isFinite(Number(operation?.resultCount))
          ? Number(operation.resultCount)
          : Number.isFinite(Number(historyItem?.resultCount))
            ? Number(historyItem.resultCount)
            : null,
      };
    });

    if (roundData.planningMessage || roundData.message || operations.length === 0) {
      activities.push({
        id: `round-${round}-plan`,
        kind: 'planning',
        round,
        label: normalizePlanningLabel(round, roundData.planningMessage || roundData.message),
        isLastRound: roundIndex === rounds.length - 1,
      });
    }

    let currentGroup = null;
    operations.forEach((operation, operationIndex) => {
      const family = getToolMeta(operation?.tool).family;
      if (!currentGroup || currentGroup.family !== family) {
        currentGroup = {
          id: `round-${round}-${family}-${operationIndex}`,
          kind: 'operations',
          family,
          round,
          operations: [],
        };
        activities.push(currentGroup);
      }
      currentGroup.operations.push(operation);
    });
  });
  return activities;
};

const buildDiagnosticItems = (trace) => {
  const items = [];
  const gate = trace.agentGate || null;
  const route = trace.routeDiagnosis || null;
  const diagnostics = trace.diagnostics || null;
  const evidenceState = trace.evidenceState || diagnostics?.evidence_state || null;
  const contextBudget = diagnostics?.context_budget || null;

  if (gate) {
    const enabled = gate.enabled || gate.use_agent || gate.agent_mode;
    items.push(`Agent ${enabled ? '启用' : '未启用'}：${AGENT_GATE_REASON_LABELS[gate.reason] || gate.reason || '未知'}`);
    if (Array.isArray(gate.matched_evidence_need) && gate.matched_evidence_need.length > 0) {
      items.push(`证据需求：${gate.matched_evidence_need.join('、')}`);
    }
    if (gate.query_type) items.push(`题型：${QUERY_TYPE_LABELS[gate.query_type] || gate.query_type}`);
  }

  if (route && typeof route === 'object') {
    const taskLabel = TASK_LABELS[route.task] || route.task;
    if (taskLabel) items.push(`意图：${taskLabel}${route.scope ? ` · ${route.scope}` : ''}`);
    if (route.is_ambiguous) items.push('澄清：需要用户补充');
    const strengthLabel = formatDecisionStrength(route.decision_strength ?? route.confidence);
    if (strengthLabel) items.push(`判定强度：${strengthLabel}`);
  }

  if (Number.isFinite(trace.contextChars) && trace.contextChars > 0) items.push(`上下文：${trace.contextChars} 字`);
  if (contextBudget) {
    items.push(`上下文预算：${contextBudget.after_tokens || 0}/${contextBudget.limit_tokens || 0} tokens${contextBudget.truncated ? '（已截断）' : ''}`);
  }

  const scoring = diagnostics?.evidence_scoring || trace.evidenceScoring || null;
  if (scoring?.applied) {
    items.push(`证据评分：高 ${scoring.high_score_count || 0} · 中 ${scoring.mid_score_count || 0} · 丢弃 ${scoring.dropped_count || 0}`);
  }
  if (evidenceState && typeof evidenceState === 'object') {
    const state = EVIDENCE_STATUS_LABELS[String(evidenceState.status || 'gathering')] || evidenceState.status;
    const pieces = [state];
    if (Number(evidenceState.tool_call_count) > 0) pieces.push(`${Number(evidenceState.tool_call_count)} 次工具`);
    if (Number(evidenceState.independent_evidence_count) > 0) pieces.push(`${Number(evidenceState.independent_evidence_count)} 路证据`);
    items.push(`证据状态：${pieces.join(' · ')}`);
  }
  const evidenceDeltas = Array.isArray(diagnostics?.evidence_delta)
    ? diagnostics.evidence_delta
    : [];
  const latestDelta = evidenceDeltas[evidenceDeltas.length - 1];
  if (latestDelta && typeof latestDelta === 'object') {
    items.push(
      `本轮证据：新增 ${Number(latestDelta.unique_delta) || 0} · 重复 ${Number(latestDelta.duplicate_delta) || 0} · 覆盖 +${Number(latestDelta.coverage_delta) || 0}`
    );
    if (latestDelta.state_hash) items.push(`状态哈希：${String(latestDelta.state_hash).slice(0, 12)}`);
  }
  if (diagnostics?.evidence_saturation_stop) {
    items.push('停止原因：连续两轮无证据增量');
  }
  if (trace.fallbackReason) items.push(`降级原因：${trace.fallbackReason}`);
  if (trace.error) items.push(`错误：${trace.error}`);
  return items;
};

const TimelineIcon = ({ children, active = false, complete = false, darkMode = false }) => (
  <span
    className={`absolute -left-[31px] top-[9px] z-[1] grid h-[22px] w-[22px] place-items-center rounded-full border ring-[3px] ${
      active
        ? darkMode
          ? 'border-[#FFA07A] bg-[#FFA07A] text-[#24272d] ring-[#2b2e34]'
          : 'border-[#a8624e] bg-[#a8624e] text-white ring-[#faf8f6]'
        : complete
          ? darkMode
            ? 'border-gray-100 bg-gray-100 text-[#27292d] ring-[#2b2e34]'
            : 'border-[#303238] bg-[#303238] text-white ring-[#faf8f6]'
        : darkMode
          ? 'border-white/15 bg-[#444850] text-gray-100 ring-[#2b2e34]'
          : 'border-[#d8cec7] bg-[#f2ece7] text-[#5c5049] ring-[#faf8f6]'
    } ${active ? 'agent-timeline-node-active' : complete ? 'agent-timeline-node-complete' : ''}`}
    aria-hidden="true"
  >
    {children}
  </span>
);

const AgentProgress = ({ taskStatus, operationCount, darkMode }) => {
  const completedTasks = Array.isArray(taskStatus?.completed) ? taskStatus.completed : [];
  const pendingTasks = Array.isArray(taskStatus?.pending) ? taskStatus.pending : [];
  const hasCurrentTask = Boolean(normalizeText(taskStatus?.current, 180));
  const totalTasks = completedTasks.length + pendingTasks.length + (hasCurrentTask ? 1 : 0);
  const hasKnownTotal = totalTasks > 1 || pendingTasks.length > 0;
  const completedCount = Math.min(completedTasks.length, totalTasks);
  const progress = hasKnownTotal && totalTasks > 0
    ? Math.round((completedCount / totalTasks) * 100)
    : 0;
  const progressText = hasKnownTotal
    ? `${completedCount} / ${totalTasks} 项`
    : operationCount > 0
      ? `${operationCount} 次工具`
      : '准备中';

  return (
    <div className="agent-progress-enter mb-2.5 ml-1 mr-1 pt-1" data-testid="agent-progress">
      <div className={`mb-1.5 flex items-center justify-between gap-3 px-1 text-[12.5px] leading-5 ${
        darkMode ? 'text-gray-300' : 'text-[#5f5a56]'
      }`}>
        <span className="font-medium">检索进度</span>
        <span className={`flex-shrink-0 tabular-nums ${darkMode ? 'text-gray-400' : 'text-[#8b817b]'}`}>
          {progressText}
        </span>
      </div>
      <div
        className={`agent-progress-track h-[5px] w-full rounded-full ${
          darkMode ? 'bg-white/10' : 'bg-[#e8e3df]'
        }`}
        role="progressbar"
        aria-label="检索进度"
        aria-valuemin={hasKnownTotal ? 0 : undefined}
        aria-valuemax={hasKnownTotal ? totalTasks : undefined}
        aria-valuenow={hasKnownTotal ? completedCount : undefined}
        aria-valuetext={hasKnownTotal ? undefined : '正在检索与整理证据'}
      >
        {hasKnownTotal && (
          <span
            className={`agent-progress-fill block h-full rounded-full ${darkMode ? 'bg-[#FFA07A]' : 'bg-[#B56B55]'}`}
            style={{ width: `${progress}%` }}
          />
        )}
        <span className="agent-progress-sweep" aria-hidden="true" />
      </div>
    </div>
  );
};

export default function AgentTracePanel({ trace, embedded = false, darkMode = false }) {
  const isRunning = Boolean(trace?.enabled && trace?.startedAt && !trace?.endedAt);
  const rounds = useMemo(() => {
    const source = Array.isArray(trace?.rounds) ? trace.rounds : [];
    return source
      .map((round) => ({ ...round, round: Number(round?.round) }))
      .filter((round) => Number.isInteger(round.round) && round.round > 0)
      .sort((left, right) => left.round - right.round);
  }, [trace?.rounds]);
  const activities = useMemo(
    () => buildActivities(rounds, trace?.searchHistory),
    [rounds, trace?.searchHistory]
  );
  const activeActivityIds = useMemo(() => new Set(
    activities
      .filter((activity) => activity.kind === 'operations' && activity.operations.some((operation) => operation.status !== 'done'))
      .map((activity) => activity.id)
  ), [activities]);
  const [expandedGroups, setExpandedGroups] = useState(() => new Set(activeActivityIds));
  const [panelExpanded, setPanelExpanded] = useState(() => isRunning);
  const wasRunningRef = useRef(isRunning);
  const [nowTick, setNowTick] = useState(() => Date.now());
  const intentId = String(trace?.routeDiagnosis?.intent_id || '');
  const [intentFeedback, setIntentFeedback] = useState('');
  const [intentFeedbackPending, setIntentFeedbackPending] = useState(false);

  useEffect(() => {
    if (!isRunning) return undefined;
    const timer = setInterval(() => setNowTick(Date.now()), 500);
    return () => clearInterval(timer);
  }, [isRunning]);

  useEffect(() => {
    setIntentFeedback('');
    setIntentFeedbackPending(false);
  }, [intentId]);

  useEffect(() => {
    if (activeActivityIds.size === 0) return;
    setExpandedGroups((current) => new Set([...current, ...activeActivityIds]));
  }, [activeActivityIds]);

  useEffect(() => {
    if (!wasRunningRef.current && isRunning) setPanelExpanded(true);
    if (wasRunningRef.current && !isRunning) setPanelExpanded(false);
    wasRunningRef.current = isRunning;
  }, [isRunning]);

  if (!trace?.enabled) return null;

  const operationCount = activities.reduce(
    (count, activity) => count + (activity.kind === 'operations' ? activity.operations.length : 0),
    0
  );
  const totalResults = activities.reduce(
    (count, activity) => count + (activity.kind === 'operations'
      ? activity.operations.reduce((sum, operation) => sum + (Number(operation.resultCount) || 0), 0)
      : 0),
    0
  );
  const duration = isRunning
    ? formatElapsedMs(Math.max(0, nowTick - trace.startedAt))
    : formatDuration(trace.startedAt, trace.endedAt);
  const diagnosticItems = buildDiagnosticItems(trace);
  const subQuestions = Array.isArray(trace.subQuestions)
    ? trace.subQuestions
    : Array.isArray(trace.diagnostics?.sub_questions)
      ? trace.diagnostics.sub_questions
      : [];
  const coverage = Array.isArray(trace.taskStatus?.sub_question_coverage)
    ? trace.taskStatus.sub_question_coverage
    : [];

  const toggleGroup = (groupId) => {
    setExpandedGroups((current) => {
      const next = new Set(current);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  };

  const submitIntentFeedback = async (verdict) => {
    if (!intentId || intentFeedbackPending || intentFeedback) return;
    const route = trace?.routeDiagnosis || {};
    setIntentFeedbackPending(true);
    try {
      const response = await fetch('/intent/corrections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          intent_id: intentId,
          intent_version: route.intent_version || '',
          verdict,
          predicted_task: route.task || '',
          predicted_scope: route.scope || '',
          predicted_is_ambiguous: Boolean(route.is_ambiguous),
          decision_strength: Number(route.decision_strength ?? route.confidence) || 0,
        }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setIntentFeedback(verdict);
    } catch (_error) {
      setIntentFeedback('error');
    } finally {
      setIntentFeedbackPending(false);
    }
  };

  const renderPlanningActivity = (activity) => {
    const roundHasActiveOperation = activities.some((item) => (
      item.kind === 'operations'
      && item.round === activity.round
      && item.operations.some((operation) => operation.status !== 'done')
    ));
    const active = isRunning && activity.isLastRound && !roundHasActiveOperation;
    return (
      <div key={activity.id} className="agent-op-enter relative min-h-10 py-1 pl-1">
        <TimelineIcon active={active} darkMode={darkMode}>
          {active ? (
            <Loader2 className="h-[15px] w-[15px] animate-spin motion-reduce:animate-none" strokeWidth={2.35} />
          ) : (
            <PlanningThoughtIcon className="h-[15px] w-[15px]" />
          )}
        </TimelineIcon>
        <div className={`min-h-8 rounded-[7px] px-2.5 py-1.5 text-[13.5px] leading-5 ${
          active
            ? darkMode ? 'font-medium text-gray-200' : 'font-medium text-[#4f4a46]'
            : darkMode ? 'text-gray-300' : 'text-[#5c554f]'
        }`}>
          {activity.label}
        </div>
      </div>
    );
  };

  const renderOperationGroup = (activity) => {
    const familyMeta = FAMILY_META[activity.family] || FAMILY_META.other;
    const Icon = familyMeta.icon;
    const active = activity.operations.some((operation) => operation.status !== 'done');
    const resultCount = activity.operations.reduce((sum, operation) => sum + (Number(operation.resultCount) || 0), 0);
    const expanded = expandedGroups.has(activity.id);
    const hasDetails = activity.operations.some((operation) => (
      normalizeText(operation.query)
      || normalizeText(operation.resultMessage || operation.message)
      || Number.isFinite(Number(operation.resultCount))
      || Number.isFinite(Number(operation.elapsedMs))
    ));
    const summary = active ? familyMeta.running : familyMeta.done(activity.operations.length);

    return (
      <div key={activity.id} className="agent-op-enter relative min-h-10 py-0.5 pl-1">
        <TimelineIcon active={active} darkMode={darkMode}>
          {active ? (
            <Loader2 className="h-[15px] w-[15px] animate-spin motion-reduce:animate-none" strokeWidth={2.35} />
          ) : (
            <Icon className="h-[15px] w-[15px]" strokeWidth={2.15} />
          )}
        </TimelineIcon>
        <button
          type="button"
          onClick={() => hasDetails && toggleGroup(activity.id)}
          aria-expanded={hasDetails ? expanded : undefined}
          className={`flex min-h-9 w-full items-center gap-2.5 rounded-[8px] px-2.5 py-1.5 text-left transition-[background-color,transform] duration-200 active:scale-[0.995] focus-visible:outline-none focus-visible:ring-2 ${
            darkMode
              ? 'hover:bg-white/[0.035] focus-visible:ring-[#FFA07A]/35'
              : 'hover:bg-[#f6f3f1] focus-visible:ring-[#D99178]/35'
          }`}
        >
          <span className={`min-w-0 flex-1 truncate text-[13.5px] ${
            active
              ? darkMode ? 'font-medium text-gray-200' : 'font-medium text-[#4f4a46]'
              : darkMode ? 'text-gray-200' : 'text-[#514a45]'
          }`}>
            {summary}
          </span>
          {resultCount > 0 && (
            <span className={`flex-shrink-0 text-[12.5px] tabular-nums ${darkMode ? 'text-gray-300' : 'text-[#7c7069]'}`}>
              {resultCount} 个结果
            </span>
          )}
          {hasDetails && (
            <ChevronDown
              className={`h-4 w-4 flex-shrink-0 transition-transform duration-300 ${
                darkMode ? 'text-gray-300' : 'text-[#7c7069]'
              } ${expanded ? 'rotate-180' : ''}`}
              strokeWidth={1.9}
              aria-hidden="true"
            />
          )}
        </button>

        {hasDetails && (
          <div
            className={`grid transition-[grid-template-rows,opacity] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none ${
              expanded ? 'grid-rows-[1fr] opacity-100' : 'pointer-events-none grid-rows-[0fr] opacity-0'
            }`}
            aria-hidden={!expanded}
          >
            <div className="min-h-0 overflow-hidden">
              <div className="pb-1.5 pl-1 pr-1 pt-0.5">
                {activity.operations.map((operation, index) => {
                  const meta = getToolMeta(operation.tool);
                  const OperationIcon = meta.icon;
                  const elapsed = formatElapsedMs(operation.elapsedMs);
                  return (
                    <div key={`${operation.tool || 'tool'}-${index}`} className="agent-detail-enter flex items-start gap-2.5 rounded-[7px] px-2.5 py-2">
                      <OperationIcon className={`mt-[2px] h-4 w-4 flex-shrink-0 ${darkMode ? 'text-gray-300' : 'text-[#756a63]'}`} strokeWidth={2} aria-hidden="true" />
                      <div className="min-w-0 flex-1">
                        <div className={`break-words text-[13px] leading-5 ${darkMode ? 'text-gray-200' : 'text-[#58514d]'}`}>
                          {operationDetailText(operation)}
                        </div>
                        <div className={`mt-0.5 flex flex-wrap gap-x-2 text-[12px] tabular-nums ${darkMode ? 'text-gray-400' : 'text-[#7f746d]'}`}>
                          <span>{meta.label}</span>
                          {Number.isFinite(Number(operation.resultCount)) && <span>{Number(operation.resultCount)} 个结果</span>}
                          {elapsed && <span>{elapsed}</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const currentTask = normalizeText(trace.taskStatus?.current, 180);
  const hasActivities = activities.length > 0;
  const timeline = (
    <div data-testid="agent-trace-timeline">
      {isRunning && (
        <AgentProgress
          taskStatus={trace.taskStatus}
          operationCount={operationCount}
          darkMode={darkMode}
        />
      )}

      {!hasActivities && isRunning && (
        <div className="agent-op-enter relative min-h-10 py-1 pl-1">
          <TimelineIcon active darkMode={darkMode}>
            <Loader2 className="h-[15px] w-[15px] animate-spin motion-reduce:animate-none" strokeWidth={2.35} />
          </TimelineIcon>
          <div className={`px-2.5 py-1.5 text-[13.5px] leading-5 ${darkMode ? 'text-gray-200' : 'text-[#514a45]'}`}>
            {currentTask || '正在生成检索计划'}
          </div>
        </div>
      )}

      {activities.map((activity) => (
        activity.kind === 'planning'
          ? renderPlanningActivity(activity)
          : renderOperationGroup(activity)
      ))}

      {!isRunning && operationCount > 0 && (
        <div className="agent-op-enter relative min-h-10 py-1 pl-1">
          <TimelineIcon complete darkMode={darkMode}>
            <Check className="h-[15px] w-[15px]" strokeWidth={2.8} />
          </TimelineIcon>
          <div className={`flex min-h-8 flex-wrap items-center gap-x-2 gap-y-0.5 rounded-[7px] px-2.5 py-1.5 text-[13.5px] leading-5 ${darkMode ? 'text-gray-200' : 'text-[#4d4946]'}`}>
            <span className="font-medium">证据收集完成</span>
            <span className={`text-[12px] tabular-nums ${darkMode ? 'text-gray-300' : 'text-[#776d66]'}`}>
              {operationCount} 次工具{totalResults > 0 ? ` · ${totalResults} 个结果` : ''}{duration ? ` · ${duration}` : ''}
            </span>
          </div>
        </div>
      )}

      {(diagnosticItems.length > 0 || subQuestions.length > 0 || trace.agentDetail?.length > 0) && (
        <details className={`agent-diagnostics ml-1 mt-1 rounded-[8px] px-2 py-1 text-[12.5px] ${darkMode ? 'text-gray-300' : 'text-[#746a63]'}`}>
          <summary className={`flex min-h-7 cursor-pointer select-none items-center gap-1.5 rounded-[6px] px-1 py-1 transition-colors focus-visible:outline-none focus-visible:ring-2 ${
            darkMode ? 'hover:text-gray-300 focus-visible:ring-[#FFA07A]/30' : 'hover:text-gray-600 focus-visible:ring-[#D99178]/30'
          }`}>
            <ChevronRight className="agent-diagnostics-chevron h-3.5 w-3.5 flex-shrink-0" strokeWidth={2} aria-hidden="true" />
            <span>检索诊断</span>
          </summary>
          <div className={`mt-1.5 space-y-1 border-l pl-3 ${darkMode ? 'border-white/[0.08]' : 'border-[#e5e0dc]'}`}>
            {diagnosticItems.map((item, index) => (
              <div key={index} className={item.startsWith('错误') ? 'text-rose-600 dark:text-rose-400' : ''}>{item}</div>
            ))}
            {subQuestions.map((question, index) => (
              <div key={`question-${index}`} className="flex items-start gap-1.5">
                <Check className={`mt-0.5 h-3 w-3 flex-shrink-0 ${coverage[index] ? 'opacity-100' : 'opacity-25'}`} strokeWidth={2} />
                <span>{normalizeText(question, 240)}</span>
              </div>
            ))}
            {Array.isArray(trace.agentDetail) && trace.agentDetail.length > 0 && (
              <div>已纳入 {trace.agentDetail.length} 个语义组</div>
            )}
            {intentId && (
              <div className="flex items-center gap-1.5 pt-1" aria-label="意图判定反馈">
                {intentFeedback === 'correct' || intentFeedback === 'incorrect' ? (
                  <span>已记录判定反馈</span>
                ) : intentFeedback === 'error' ? (
                  <button type="button" onClick={() => setIntentFeedback('')} className="rounded-md px-1.5 py-1 hover:bg-black/5 dark:hover:bg-white/5">
                    记录失败，重试
                  </button>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={() => submitIntentFeedback('correct')}
                      disabled={intentFeedbackPending}
                      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 transition-colors hover:bg-black/5 disabled:opacity-50 dark:hover:bg-white/5"
                    >
                      <ThumbsUp className="h-3 w-3" aria-hidden="true" />
                      判定准确
                    </button>
                    <button
                      type="button"
                      onClick={() => submitIntentFeedback('incorrect')}
                      disabled={intentFeedbackPending}
                      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 transition-colors hover:bg-black/5 disabled:opacity-50 dark:hover:bg-white/5"
                    >
                      <ThumbsDown className="h-3 w-3" aria-hidden="true" />
                      判定有误
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        </details>
      )}
    </div>
  );

  if (embedded) return timeline;

  const headerLabel = isRunning ? '正在检索证据' : '检索过程';
  return (
    <section className={`mt-2 w-full max-w-[46rem] text-[13.5px] ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
      <button
        type="button"
        onClick={() => setPanelExpanded((current) => !current)}
        aria-expanded={panelExpanded}
        className={`-ml-1 flex min-h-9 max-w-full items-center gap-2.5 rounded-[9px] px-2 py-1.5 text-left transition-[background-color,transform] duration-200 active:scale-[0.995] focus-visible:outline-none focus-visible:ring-2 ${
          darkMode
            ? 'hover:bg-white/[0.04] focus-visible:ring-[#FFA07A]/40'
            : 'hover:bg-[#f6f3f1] focus-visible:ring-[#D99178]/40'
        }`}
      >
        {isRunning ? (
          <Loader2 className="h-[18px] w-[18px] animate-spin motion-reduce:animate-none" strokeWidth={1.9} aria-hidden="true" />
        ) : (
          <ScanSearch className="h-[18px] w-[18px]" strokeWidth={1.9} aria-hidden="true" />
        )}
        <span className="font-medium">{headerLabel}</span>
        <span className={`truncate text-[12px] tabular-nums ${darkMode ? 'text-gray-400' : 'text-[#8b817b]'}`}>
          {operationCount} 次工具{totalResults > 0 ? ` · ${totalResults} 个结果` : ''}{duration ? ` · ${duration}` : ''}
        </span>
        <ChevronDown className={`h-4 w-4 transition-transform duration-300 ${panelExpanded ? 'rotate-180' : ''}`} strokeWidth={1.9} aria-hidden="true" />
      </button>
      <div className={`grid transition-[grid-template-rows,opacity,visibility] duration-300 ${
        panelExpanded ? 'visible grid-rows-[1fr] opacity-100' : 'invisible grid-rows-[0fr] opacity-0'
      }`} aria-hidden={!panelExpanded}>
        <div className="min-h-0 overflow-hidden">
          <div className={`relative ml-[9px] border-l pb-1 pl-5 pt-1.5 ${darkMode ? 'border-white/[0.14]' : 'border-[#d4ccc6]'}`}>
            {timeline}
          </div>
        </div>
      </div>
    </section>
  );
}
