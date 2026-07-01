import React, { useMemo, useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Bot,
  Search,
  Hash,
  Wrench,
  Layers,
  Map,
  Sparkles,
  CheckCircle2,
  Circle,
  Clock,
} from 'lucide-react';

// 工具到 icon / 中文标签的映射，与后端 retrieval_agent / retrieval_tools 保持一致
const TOOL_META = {
  vector_search: { label: '向量搜索', icon: Sparkles, iconClass: 'text-violet-500' },
  keyword_search: { label: 'BM25 关键词', icon: Hash, iconClass: 'text-amber-500' },
  grep: { label: 'GREP 字面', icon: Search, iconClass: 'text-sky-500' },
  regex_search: { label: '正则匹配', icon: Search, iconClass: 'text-cyan-500' },
  boolean_search: { label: '布尔逻辑', icon: Wrench, iconClass: 'text-indigo-500' },
  fetch: { label: '获取意群', icon: Layers, iconClass: 'text-emerald-500' },
  map: { label: '文档地图', icon: Map, iconClass: 'text-rose-500' },
};

const getToolMeta = (tool) =>
  TOOL_META[tool] || {
    label: tool || '工具',
    icon: Wrench,
    iconClass: 'text-gray-500',
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
  if (!Number.isFinite(ms)) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
};

const AGENT_GATE_REASON_LABELS = {
  matched_query_type: '题型匹配',
  matched_evidence_need: '证据需求匹配',
  route_not_matched: '题型未匹配',
  selected_text_present: '框选文本优先',
  switch_disabled: '开关未启用',
  stream_only: '非流式未执行',
};

/**
 * 检索代理执行轨迹面板。
 * embedded=true 时作为思考面板内的子区域显示，避免完成后跳到回答下方。
 */
export default function AgentTracePanel({ trace, embedded = false }) {
  const [collapsed, setCollapsed] = useState(false);
  const [expandedRounds, setExpandedRounds] = useState(() => new Set([1]));

  const stats = useMemo(() => {
    if (!trace) return null;
    const rounds = Array.isArray(trace.rounds) ? trace.rounds : [];
    const opCount = rounds.reduce(
      (sum, round) => sum + (Array.isArray(round.operations) ? round.operations.length : 0),
      0
    );
    const totalResults = rounds.reduce((sum, round) => {
      if (!Array.isArray(round.operations)) return sum;
      return sum + round.operations.reduce((inner, op) => inner + (Number(op.resultCount) || 0), 0);
    }, 0);
    return { roundCount: rounds.length, opCount, totalResults };
  }, [trace]);

  if (!trace || !trace.enabled) return null;

  const rounds = Array.isArray(trace.rounds) ? trace.rounds : [];
  const taskStatus = trace.taskStatus || { completed: [], current: '', pending: [] };
  const gate = trace.agentGate || null;
  const diagnostics = trace.diagnostics || null;
  const contextBudget = diagnostics?.context_budget || null;
  const subQuestions = trace.subQuestions || trace.diagnostics?.sub_questions || [];
  const coverage = trace.taskStatus?.sub_question_coverage || [];
  const duration = formatDuration(trace.startedAt, trace.endedAt);
  const hasTaskStatus =
    (taskStatus.completed && taskStatus.completed.length > 0) ||
    Boolean(taskStatus.current) ||
    (taskStatus.pending && taskStatus.pending.length > 0);

  const toggleRound = (round) => {
    setExpandedRounds((prev) => {
      const next = new Set(prev);
      if (next.has(round)) next.delete(round);
      else next.add(round);
      return next;
    });
  };

  const renderMetaSummary = () => {
    const items = [];
    if (gate) {
      items.push(
        `触发: ${AGENT_GATE_REASON_LABELS[gate.reason] || gate.reason || '未知'}${
          gate.requested_reason
            ? `（原始: ${AGENT_GATE_REASON_LABELS[gate.requested_reason] || gate.requested_reason}）`
            : ''
        }`
      );
    }
    if (Number.isFinite(trace.contextChars) && trace.contextChars > 0) {
      items.push(`上下文: ${trace.contextChars}字`);
    }
    if (contextBudget) {
      items.push(
        `预算: ${contextBudget.after_tokens || 0}/${contextBudget.limit_tokens || 0} tokens${
          contextBudget.truncated ? '（已截断）' : ''
        }`
      );
    }
    if (trace.fallbackReason) items.push(`降级原因: ${trace.fallbackReason}`);
    if (trace.error) items.push(`错误: ${trace.error}`);
    if (items.length === 0) return null;

    return (
      <div className="flex flex-wrap gap-x-3 gap-y-1 rounded-md border border-gray-200/70 bg-gray-50/70 px-2.5 py-2 text-[11px] text-gray-600 dark:border-gray-700 dark:bg-gray-900/40 dark:text-gray-300">
        {items.map((item, index) => (
          <span key={index} className={item.startsWith('错误') ? 'text-rose-600 dark:text-rose-400' : ''}>
            {item}
          </span>
        ))}
      </div>
    );
  };

  const renderTaskStatus = () => (
    <div className="rounded-md border border-gray-200/70 bg-white/70 p-2.5 text-xs dark:border-gray-700 dark:bg-gray-900/30">
      <div className="mb-2 flex items-center gap-1.5 font-medium text-gray-700 dark:text-gray-200">
        <Bot className="h-3.5 w-3.5 text-violet-500" />
        <span>任务状态</span>
      </div>
      <div className="flex flex-col gap-1.5">
        {taskStatus.completed?.map((task, index) => (
          <div key={`done-${index}`} className="flex items-start gap-1.5 text-emerald-700 dark:text-emerald-400">
            <CheckCircle2 className="mt-0.5 h-3 w-3 flex-shrink-0" />
            <span className="line-clamp-2">{task}</span>
          </div>
        ))}
        {taskStatus.current && (
          <div className="flex items-start gap-1.5 text-blue-700 dark:text-blue-400">
            <Clock className="mt-0.5 h-3 w-3 flex-shrink-0" />
            <span className="line-clamp-2">{taskStatus.current}</span>
          </div>
        )}
        {taskStatus.pending?.map((task, index) => (
          <div key={`pending-${index}`} className="flex items-start gap-1.5 text-gray-500 dark:text-gray-400">
            <Circle className="mt-0.5 h-3 w-3 flex-shrink-0" />
            <span className="line-clamp-2">{task}</span>
          </div>
        ))}
      </div>
    </div>
  );

  const renderOperation = (op, index) => {
    const meta = getToolMeta(op.tool);
    const Icon = meta.icon;
    const isDone = op.status === 'done' || Number.isFinite(op.resultCount);
    const elapsed = formatElapsedMs(op.elapsedMs);

    return (
      <div
        key={index}
        className="flex items-start gap-2 rounded-md border border-gray-200/70 bg-white/75 px-2.5 py-2 text-[11px] dark:border-gray-700 dark:bg-gray-900/30"
      >
        <span className="mt-0.5 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-md bg-gray-50 dark:bg-gray-800">
          <Icon className={`h-3.5 w-3.5 ${meta.iconClass}`} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
            <span className="font-medium text-gray-700 dark:text-gray-200">{meta.label}</span>
            {Number.isFinite(op.resultCount) && (
              <span className="text-gray-500 dark:text-gray-400">→ {op.resultCount} 个结果</span>
            )}
            {elapsed && <span className="text-gray-400 dark:text-gray-500">· {elapsed}</span>}
            {!isDone && <span className="text-blue-500 dark:text-blue-400">执行中</span>}
          </div>
          {(op.resultMessage || op.message) && (
            <div className="mt-1 line-clamp-2 break-words text-gray-500 dark:text-gray-400">
              {op.resultMessage || op.message}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderRound = (roundData, roundIndex) => {
    const round = roundData.round;
    const operations = Array.isArray(roundData.operations) ? roundData.operations : [];
    const isExpanded = expandedRounds.has(round);
    const opCount = operations.length;
    const successCount = operations.filter((op) => Number(op.resultCount) > 0).length;
    const invocationMode = trace.diagnostics?.planner_invocation_mode?.[roundIndex];

    return (
      <div key={round} className="overflow-hidden rounded-md border border-gray-200/80 bg-gray-50/40 dark:border-gray-700 dark:bg-gray-900/20">
        <button
          type="button"
          onClick={() => toggleRound(round)}
          className="flex w-full items-center gap-2 px-2.5 py-2 text-left text-xs transition-colors hover:bg-gray-100/70 dark:hover:bg-gray-800/60"
        >
          {isExpanded ? (
            <ChevronDown className="h-3.5 w-3.5 flex-shrink-0 text-gray-400" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 flex-shrink-0 text-gray-400" />
          )}
          <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-md bg-violet-100 text-[10px] font-bold text-violet-700 dark:bg-violet-900/40 dark:text-violet-300">
            {round}
          </span>
          <span className="min-w-0 flex-1 truncate text-gray-700 dark:text-gray-200">
            第 {round} 轮 · {opCount} 个工具{successCount > 0 ? ` · ${successCount} 命中` : ''}
          </span>
          {invocationMode === 'native_tools' && (
            <span className="rounded border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300">
              原生工具
            </span>
          )}
          {invocationMode === 'json_fallback' && (
            <span className="rounded border border-orange-200 bg-orange-50 px-1.5 py-0.5 text-[10px] font-medium text-orange-700 dark:border-orange-800 dark:bg-orange-950/40 dark:text-orange-300">
              JSON 兜底
            </span>
          )}
          {roundData.planningMessage && !operations.length && (
            <span className="text-[10px] text-gray-400">规划中</span>
          )}
        </button>
        {isExpanded && (
          <div className="space-y-1.5 px-2.5 pb-2.5 pt-0.5">
            {roundData.planningMessage && (
              <div className="px-0.5 text-[11px] italic text-gray-500 dark:text-gray-400">
                {roundData.planningMessage}
              </div>
            )}
            {operations.length === 0 ? (
              <div className="px-0.5 text-[11px] italic text-gray-400">该轮未执行任何工具</div>
            ) : (
              operations.map((op, opIndex) => renderOperation(op, opIndex))
            )}
          </div>
        )}
      </div>
    );
  };

  const rootClass = embedded
    ? 'mt-3 border-t border-purple-100/70 pt-3 dark:border-purple-900/40'
    : 'mt-2 ml-2 max-w-2xl';

  return (
    <div className={rootClass}>
      <button
        type="button"
        onClick={() => setCollapsed((prev) => !prev)}
        className="mb-2 flex w-full items-center gap-1.5 text-left text-xs text-gray-600 transition-colors hover:text-gray-800 dark:text-gray-300 dark:hover:text-gray-100"
      >
        {collapsed ? (
          <ChevronRight className="h-3.5 w-3.5 flex-shrink-0" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 flex-shrink-0" />
        )}
        <Bot className="h-3.5 w-3.5 flex-shrink-0 text-violet-500" />
        <span className="font-medium">检索轨迹</span>
        {stats && (
          <span className="truncate text-gray-400 dark:text-gray-500">
            {stats.roundCount} 轮 · {stats.opCount} 次工具
            {stats.totalResults > 0 ? ` · ${stats.totalResults} 个结果` : ''}
            {duration ? ` · ${duration}` : ''}
          </span>
        )}
        {trace.fallback && (
          <span className="ml-auto rounded border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
            已降级
          </span>
        )}
      </button>

      {!collapsed && (
        <div className="flex flex-col gap-2">
          {renderMetaSummary()}

          {hasTaskStatus && renderTaskStatus()}

          {subQuestions.length > 0 && (
            <div className="rounded-md border border-gray-200/70 bg-white/70 p-2.5 text-xs dark:border-gray-700 dark:bg-gray-900/30">
              <div className="mb-2 flex items-center gap-1.5 font-medium text-gray-700 dark:text-gray-200">
                <Search className="h-3.5 w-3.5 text-violet-500" />
                <span>子问题分解</span>
                <span className="text-gray-400">({subQuestions.length})</span>
              </div>
              <div className="space-y-1.5">
                {subQuestions.map((question, index) => (
                  <div key={index} className="flex items-start gap-1.5 text-[11px] text-gray-600 dark:text-gray-300">
                    {coverage[index] ? (
                      <CheckCircle2 className="mt-0.5 h-3 w-3 flex-shrink-0 text-emerald-500" />
                    ) : (
                      <Circle className="mt-0.5 h-3 w-3 flex-shrink-0 text-gray-400" />
                    )}
                    <span className="line-clamp-2 flex-1">{question}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {rounds.length > 0 && (
            <div className="flex flex-col gap-1.5">
              {rounds.map((roundData, index) => renderRound(roundData, index))}
            </div>
          )}

          {trace.finalMessage && (
            <div className="rounded-r-md border-l-2 border-violet-200 bg-violet-50/50 px-2.5 py-1.5 text-[11px] italic text-gray-500 dark:border-violet-900/60 dark:bg-violet-950/20 dark:text-gray-400">
              {trace.finalMessage}
            </div>
          )}

          {Array.isArray(trace.agentDetail) && trace.agentDetail.length > 0 && (
            <div className="rounded-md border border-gray-200/70 bg-gray-50/70 p-2.5 text-[11px] text-gray-500 dark:border-gray-700 dark:bg-gray-900/30 dark:text-gray-400">
              <div className="mb-1.5 font-medium text-gray-600 dark:text-gray-300">已纳入意群</div>
              <div className="flex max-h-24 flex-wrap gap-1 overflow-y-auto pr-1">
                {trace.agentDetail.map((detail, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center gap-1 rounded border border-gray-200 bg-white px-1.5 py-0.5 text-gray-600 dark:border-gray-700 dark:bg-gray-900/60 dark:text-gray-300"
                  >
                    <Layers className="h-3 w-3 text-emerald-500" />
                    {detail.group_id}
                    {detail.granularity && <span className="text-gray-400">· {detail.granularity}</span>}
                    {Number.isFinite(detail.char_count) && (
                      <span className="text-gray-400">· {detail.char_count}字</span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
