import React, { useState, useMemo } from 'react';
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
  vector_search: { label: '向量搜索', icon: Sparkles, color: 'text-violet-600 bg-violet-50 border-violet-200' },
  keyword_search: { label: 'BM25 关键词', icon: Hash, color: 'text-amber-600 bg-amber-50 border-amber-200' },
  grep: { label: 'GREP 字面', icon: Search, color: 'text-sky-600 bg-sky-50 border-sky-200' },
  regex_search: { label: '正则匹配', icon: Search, color: 'text-cyan-600 bg-cyan-50 border-cyan-200' },
  boolean_search: { label: '布尔逻辑', icon: Wrench, color: 'text-indigo-600 bg-indigo-50 border-indigo-200' },
  fetch: { label: '获取意群', icon: Layers, color: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  map: { label: '文档地图', icon: Map, color: 'text-rose-600 bg-rose-50 border-rose-200' },
};

const getToolMeta = (tool) =>
  TOOL_META[tool] || {
    label: tool || '工具',
    icon: Wrench,
    color: 'text-gray-600 bg-gray-50 border-gray-200',
  };

const formatDuration = (startedAt, endedAt) => {
  if (!startedAt || !endedAt || endedAt < startedAt) return '';
  const ms = endedAt - startedAt;
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m${Math.round((ms % 60000) / 1000)}s`;
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
 * 检索代理执行轨迹面板
 * 与 EvidencePanel 风格保持一致（折叠头 + 卡片列表）
 */
export default function AgentTracePanel({ trace }) {
  const [collapsed, setCollapsed] = useState(false);
  const [expandedRounds, setExpandedRounds] = useState(() => new Set([1]));

  const stats = useMemo(() => {
    if (!trace) return null;
    const rounds = Array.isArray(trace.rounds) ? trace.rounds : [];
    const ops = rounds.reduce(
      (sum, r) => sum + (Array.isArray(r.operations) ? r.operations.length : 0),
      0
    );
    const totalResults = rounds.reduce((sum, r) => {
      if (!Array.isArray(r.operations)) return sum;
      return sum + r.operations.reduce((s, op) => s + (Number(op.resultCount) || 0), 0);
    }, 0);
    return { roundCount: rounds.length, opCount: ops, totalResults };
  }, [trace]);

  if (!trace || !trace.enabled) return null;

  const rounds = Array.isArray(trace.rounds) ? trace.rounds : [];
  const taskStatus = trace.taskStatus || { completed: [], current: '', pending: [] };
  const gate = trace.agentGate || null;
  const diagnostics = trace.diagnostics || null;
  const contextBudget = diagnostics?.context_budget || null;

  // 子问题分解数据
  const subQuestions = trace.subQuestions || trace.diagnostics?.sub_questions || [];
  const coverage = trace.taskStatus?.sub_question_coverage || [];
  const hasTaskStatus =
    (taskStatus.completed && taskStatus.completed.length > 0) ||
    !!taskStatus.current ||
    (taskStatus.pending && taskStatus.pending.length > 0);
  const duration = formatDuration(trace.startedAt, trace.endedAt);

  const toggleRound = (round) => {
    setExpandedRounds((prev) => {
      const next = new Set(prev);
      if (next.has(round)) next.delete(round);
      else next.add(round);
      return next;
    });
  };

  const renderTaskStatus = () => (
    <div className="rounded-lg border border-violet-200 bg-violet-50/40 p-2.5 text-xs">
      <div className="flex items-center gap-1.5 text-violet-700 font-medium mb-1.5">
        <Bot className="w-3.5 h-3.5" />
        <span>任务状态</span>
      </div>
      <div className="flex flex-col gap-1">
        {taskStatus.completed?.map((task, i) => (
          <div key={`done-${i}`} className="flex items-start gap-1.5 text-emerald-700">
            <CheckCircle2 className="w-3 h-3 mt-0.5 flex-shrink-0" />
            <span className="line-clamp-2">{task}</span>
          </div>
        ))}
        {taskStatus.current && (
          <div className="flex items-start gap-1.5 text-blue-700">
            <Clock className="w-3 h-3 mt-0.5 flex-shrink-0" />
            <span className="line-clamp-2">{taskStatus.current}</span>
          </div>
        )}
        {taskStatus.pending?.map((task, i) => (
          <div key={`pending-${i}`} className="flex items-start gap-1.5 text-gray-500">
            <Circle className="w-3 h-3 mt-0.5 flex-shrink-0" />
            <span className="line-clamp-2">{task}</span>
          </div>
        ))}
      </div>
    </div>
  );

  const renderOperation = (op, idx) => {
    const meta = getToolMeta(op.tool);
    const Icon = meta.icon;
    const isDone = op.status === 'done' || Number.isFinite(op.resultCount);
    return (
      <div
        key={idx}
        className={`flex items-start gap-2 px-2 py-1.5 rounded border ${meta.color} text-[11px]`}
      >
        <Icon className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="font-medium">{meta.label}</span>
            {Number.isFinite(op.resultCount) && (
              <span className="text-[10px] text-gray-500">→ {op.resultCount} 个结果</span>
            )}
            {Number.isFinite(op.elapsedMs) && (
              <span className="text-[10px] text-gray-400">· {op.elapsedMs}ms</span>
            )}
            {!isDone && (
              <span className="text-[10px] text-gray-400 italic">执行中...</span>
            )}
          </div>
          {op.resultMessage ? (
            <div className="text-gray-600 mt-0.5 line-clamp-2 break-words">
              {op.resultMessage}
            </div>
          ) : op.message ? (
            <div className="text-gray-500 mt-0.5 line-clamp-2 break-words">{op.message}</div>
          ) : null}
        </div>
      </div>
    );
  };

  const renderRound = (roundData, roundIdx) => {
    const round = roundData.round;
    const operations = Array.isArray(roundData.operations) ? roundData.operations : [];
    const isExpanded = expandedRounds.has(round);
    const opCount = operations.length;
    const successCount = operations.filter((op) => Number(op.resultCount) > 0).length;

    // 获取本轮的调用模式徽标
    const invocationMode = trace.diagnostics?.planner_invocation_mode?.[roundIdx];

    return (
      <div key={round} className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => toggleRound(round)}
          className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs hover:bg-gray-50 transition-colors"
        >
          {isExpanded ? (
            <ChevronDown className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
          )}
          <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-violet-100 text-violet-700 text-[10px] font-bold flex-shrink-0">
            {round}
          </span>
          <span className="text-gray-700 flex-1 truncate">
            第 {round} 轮 · {opCount} 个工具
            {successCount > 0 ? ` · ${successCount} 命中` : ''}
          </span>
          {invocationMode === 'native_tools' && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
              原生工具
            </span>
          )}
          {invocationMode === 'json_fallback' && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-orange-50 text-orange-700 border border-orange-200">
              JSON 兜底
            </span>
          )}
          {roundData.planningMessage && !operations.length && (
            <span className="text-[10px] text-gray-400">规划中</span>
          )}
        </button>
        {isExpanded && (
          <div className="px-3 pb-2.5 pt-1 space-y-1.5">
            {roundData.planningMessage && (
              <div className="text-[11px] text-gray-500 italic">{roundData.planningMessage}</div>
            )}
            {operations.length === 0 ? (
              <div className="text-[11px] text-gray-400 italic">该轮未执行任何工具</div>
            ) : (
              operations.map((op, i) => renderOperation(op, i))
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="mt-2 ml-2">
      <button
        onClick={() => setCollapsed((prev) => !prev)}
        className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors mb-1.5"
      >
        {collapsed ? (
          <ChevronRight className="w-3.5 h-3.5" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5" />
        )}
        <Bot className="w-3.5 h-3.5 text-violet-500" />
        <span className="font-medium">检索代理</span>
        {stats && (
          <span className="text-gray-400">
            ({stats.roundCount} 轮 · {stats.opCount} 次工具调用
            {stats.totalResults > 0 ? ` · ${stats.totalResults} 个结果` : ''}
            {duration ? ` · ${duration}` : ''})
          </span>
        )}
        {trace.fallback && (
          <span className="ml-1 px-1.5 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-200 text-[10px] font-medium">
            已降级
          </span>
        )}
      </button>

      {!collapsed && (
        <div className="flex flex-col gap-2 max-w-2xl">
          {(gate || trace.contextChars || trace.fallbackReason || trace.error || contextBudget) && (
            <div className="text-[11px] text-gray-600 px-2 py-1.5 rounded border border-gray-200 bg-gray-50/60 flex flex-wrap gap-x-3 gap-y-1">
              {gate && (
                <span>
                  触发: {AGENT_GATE_REASON_LABELS[gate.reason] || gate.reason || '未知'}
                  {gate.requested_reason ? `（原始: ${AGENT_GATE_REASON_LABELS[gate.requested_reason] || gate.requested_reason}）` : ''}
                </span>
              )}
              {Number.isFinite(trace.contextChars) && trace.contextChars > 0 && (
                <span>上下文: {trace.contextChars}字</span>
              )}
              {contextBudget && (
                <span>
                  预算: {contextBudget.after_tokens || 0}/{contextBudget.limit_tokens || 0} tokens
                  {contextBudget.truncated ? '（已截断）' : ''}
                </span>
              )}
              {trace.fallbackReason && <span>降级原因: {trace.fallbackReason}</span>}
              {trace.error && <span className="text-rose-600">错误: {trace.error}</span>}
            </div>
          )}
          {/* 子问题分解区块 */}
          {subQuestions.length > 0 && (
            <div className="mb-3 p-2 rounded-lg bg-violet-50/50 dark:bg-violet-900/10 border border-violet-200/40">
              <div className="text-[12px] font-semibold text-violet-700 dark:text-violet-300 mb-1">
                🧩 子问题分解（{subQuestions.length}）
              </div>
              <ul className="space-y-1">
                {subQuestions.map((sq, i) => (
                  <li key={i} className="text-[11px] flex items-start gap-1.5">
                    <span className={coverage[i] ? 'text-emerald-500' : 'text-gray-400'}>
                      {coverage[i] ? '✓' : '○'}
                    </span>
                    <span className="flex-1">{sq}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {hasTaskStatus && renderTaskStatus()}
          {rounds.length > 0 && <div className="flex flex-col gap-1.5">{rounds.map((r, idx) => renderRound(r, idx))}</div>}
          {trace.finalMessage && (
            <div className="text-[11px] text-gray-500 italic px-2 py-1 border-l-2 border-violet-200 bg-violet-50/40 rounded-r">
              {trace.finalMessage}
            </div>
          )}
          {Array.isArray(trace.agentDetail) && trace.agentDetail.length > 0 && (
            <div className="text-[11px] text-gray-500 px-2 py-1.5 rounded border border-gray-200 bg-gray-50/60">
              <div className="font-medium text-gray-600 mb-1">已纳入意群</div>
              <div className="flex flex-wrap gap-1">
                {trace.agentDetail.map((d, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-white border border-gray-200 text-gray-600"
                  >
                    <Layers className="w-3 h-3 text-emerald-500" />
                    {d.group_id}
                    {d.granularity && (
                      <span className="text-gray-400">· {d.granularity}</span>
                    )}
                    {Number.isFinite(d.char_count) && (
                      <span className="text-gray-400">· {d.char_count}字</span>
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
