import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Check,
  ChevronDown,
  ChevronRight,
  Globe,
  ScanSearch,
  Search,
  Hash,
  Wrench,
  Layers,
  Loader2,
  Map,
  Sparkles,
  CheckCircle2,
  Circle,
} from 'lucide-react';

// 工具到 icon / 中文标签的映射，与后端 retrieval_agent / retrieval_tools 保持一致
const TOOL_META = {
  search_document: { label: '统一检索', icon: ScanSearch },
  web_search: { label: '联网检索', icon: Globe },
  read_blocks: { label: '读取原文块', icon: Layers },
  visual_search: { label: '定位视觉证据', icon: ScanSearch },
  analyze_visual_evidence: { label: '分析图表证据', icon: Sparkles },
  complete: { label: '结束检索', icon: Check },
  vector_search: { label: '向量搜索', icon: Sparkles },
  keyword_search: { label: 'BM25 关键词', icon: Hash },
  grep: { label: 'GREP 字面', icon: Search },
  regex_search: { label: '正则匹配', icon: Search },
  boolean_search: { label: '布尔逻辑', icon: Wrench },
  fetch: { label: '获取意群', icon: Layers },
  map: { label: '文档地图', icon: Map },
};

const getToolMeta = (tool) =>
  TOOL_META[tool] || {
    label: tool || '工具',
    icon: Wrench,
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

const EVIDENCE_STATUS_LABELS = {
  answered: '证据已就绪',
  insufficient_evidence: '证据不足',
  budget_exhausted: '工具预算已用尽',
  gathering: '正在收集证据',
};

/**
 * 检索代理执行轨迹面板。
 * embedded=true 时作为思考面板内的子区域显示，避免完成后跳到回答下方。
 */
const BouncingDots = ({ className = 'bg-[#D97A5D]' }) => (
  <span className="ml-1 inline-flex items-center gap-0.5" aria-hidden="true">
    {[0, 1, 2].map((i) => (
      <span
        key={i}
        className={`h-1 w-1 rounded-full ${className} animate-bounce`}
        style={{ animationDelay: `${i * 0.15}s` }}
      />
    ))}
  </span>
);

export default function AgentTracePanel({ trace, embedded = false }) {
  const isRunning = Boolean(trace && trace.enabled && trace.startedAt && !trace.endedAt);
  const rounds = useMemo(() => {
    const sourceRounds = Array.isArray(trace?.rounds) ? trace.rounds : [];
    return sourceRounds
      .map((round) => ({ ...round, round: Number(round?.round) }))
      .filter((round) => Number.isInteger(round.round) && round.round > 0)
      .sort((left, right) => left.round - right.round);
  }, [trace?.rounds]);
  const [collapsed, setCollapsed] = useState(() => !isRunning);
  const [expandedRounds, setExpandedRounds] = useState(() => new Set(isRunning ? [1] : []));
  const wasRunningRef = useRef(isRunning);

  // 运行中每 500ms 触发一次重绘，让头部计时器实时走动
  const [nowTick, setNowTick] = useState(() => Date.now());
  useEffect(() => {
    if (!isRunning) return undefined;
    const id = setInterval(() => setNowTick(Date.now()), 500);
    return () => clearInterval(id);
  }, [isRunning]);

  // 新一轮开始时展示最新进展；用户仍可在当前轮手动收起。
  const latestRound = rounds[rounds.length - 1]?.round;
  useEffect(() => {
    if (!isRunning || latestRound == null) return;
    setCollapsed(false);
    setExpandedRounds((prev) => {
      if (prev.has(latestRound)) return prev;
      const next = new Set(prev);
      next.add(latestRound);
      return next;
    });
  }, [isRunning, latestRound]);

  // 检索开始自动展开，结束后压缩成一行摘要，避免历史过程长期占满对话区。
  useEffect(() => {
    if (!wasRunningRef.current && isRunning) {
      setCollapsed(false);
    } else if (wasRunningRef.current && !isRunning) {
      setCollapsed(true);
      setExpandedRounds(new Set());
    }
    wasRunningRef.current = isRunning;
  }, [isRunning]);

  const stats = useMemo(() => {
    if (!trace) return null;
    const opCount = rounds.reduce(
      (sum, round) => sum + (Array.isArray(round.operations) ? round.operations.length : 0),
      0
    );
    const totalResults = rounds.reduce((sum, round) => {
      if (!Array.isArray(round.operations)) return sum;
      return sum + round.operations.reduce((inner, op) => inner + (Number(op.resultCount) || 0), 0);
    }, 0);
    return { roundCount: rounds.length, opCount, totalResults };
  }, [rounds, trace]);

  if (!trace || !trace.enabled) return null;

  const taskStatus = trace.taskStatus || { completed: [], current: '', pending: [] };
  const gate = trace.agentGate || null;
  const diagnostics = trace.diagnostics || null;
  const evidenceState = trace.evidenceState || diagnostics?.evidence_state || null;
  const contextBudget = diagnostics?.context_budget || null;
  const subQuestions = trace.subQuestions || trace.diagnostics?.sub_questions || [];
  const coverage = trace.taskStatus?.sub_question_coverage || [];
  const duration = formatDuration(trace.startedAt, trace.endedAt);
  const liveDuration = isRunning ? formatElapsedMs(Math.max(0, nowTick - trace.startedAt)) : '';
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
    if (evidenceState && typeof evidenceState === 'object') {
      const status = String(evidenceState.status || 'gathering');
      const pieces = [EVIDENCE_STATUS_LABELS[status] || status];
      const toolCalls = Number(evidenceState.tool_call_count);
      const independentEvidence = Number(evidenceState.independent_evidence_count);
      const selectedBlocks = Number(evidenceState.selected_block_count);
      if (Number.isFinite(toolCalls) && toolCalls > 0) pieces.push(`${toolCalls} 次工具`);
      if (Number.isFinite(independentEvidence) && independentEvidence > 0) {
        pieces.push(`${independentEvidence} 路证据`);
      } else if (Number.isFinite(selectedBlocks) && selectedBlocks > 0) {
        pieces.push(`${selectedBlocks} 个原文块`);
      }
      items.push(`证据: ${pieces.join(' · ')}`);
    }
    if (trace.fallbackReason) items.push(`降级原因: ${trace.fallbackReason}`);
    if (trace.error) items.push(`错误: ${trace.error}`);
    if (items.length === 0) return null;

    return (
      <div className="ml-[9.5px] flex flex-wrap gap-x-3 gap-y-1 border-l border-[#ddd8d4] py-1 pl-[21px] text-[11px] text-gray-500 dark:border-white/10 dark:text-gray-400">
        {items.map((item, index) => (
          <span key={index} className={item.startsWith('错误') ? 'text-rose-600 dark:text-rose-400' : ''}>
            {item}
          </span>
        ))}
      </div>
    );
  };

  const renderProgressMeter = (doneCount, totalCount, unitLabel) => {
    const hasKnownTotal = totalCount > 0;
    const safeDoneCount = hasKnownTotal ? Math.min(Math.max(doneCount, 0), totalCount) : 0;
    const pct = hasKnownTotal ? (safeDoneCount / totalCount) * 100 : 0;

    return (
      <>
        <div className="mb-2.5 text-[11px] font-medium tabular-nums text-gray-600 dark:text-gray-300">
          {hasKnownTotal ? `${safeDoneCount} / ${totalCount} ${unitLabel}完成` : '正在准备检索'}
        </div>
        <div
          className="agent-progress-track h-[5px] w-full rounded-full bg-[#e8e6e3] dark:bg-white/10"
          role="progressbar"
          aria-label="检索进度"
          aria-valuemin={hasKnownTotal ? 0 : undefined}
          aria-valuemax={hasKnownTotal ? totalCount : undefined}
          aria-valuenow={hasKnownTotal ? safeDoneCount : undefined}
          aria-valuetext={hasKnownTotal ? undefined : '正在准备检索'}
        >
          {hasKnownTotal ? (
            <span
              className="block h-full rounded-full bg-[#242629] transition-[width] duration-500 ease-out motion-reduce:transition-none dark:bg-gray-200"
              style={{ width: `${pct}%` }}
            />
          ) : (
            <span className="agent-progress-sweep" />
          )}
        </div>
      </>
    );
  };

  // 任务清单：参考纵向步骤轴，当前步骤直接承载检索轮次，形成一条连续流程。
  const renderTaskStatus = (renderCurrentRounds) => {
    const completedTasks = Array.isArray(taskStatus.completed) ? taskStatus.completed : [];
    const pendingTasks = Array.isArray(taskStatus.pending) ? taskStatus.pending : [];
    const doneCount = completedTasks.length;
    const totalCount = doneCount + (taskStatus.current ? 1 : 0) + pendingTasks.length;
    const steps = [
      ...completedTasks.map((label, index) => ({ key: `done-${index}`, label, status: 'done' })),
      ...(taskStatus.current
        ? [{ key: 'current', label: taskStatus.current, status: 'active' }]
        : []),
      ...pendingTasks.map((label, index) => ({ key: `pending-${index}`, label, status: 'pending' })),
    ];
    const roundHostKey = taskStatus.current
      ? 'current'
      : completedTasks.length > 0
        ? `done-${completedTasks.length - 1}`
        : pendingTasks.length > 0
          ? 'pending-0'
          : '';

    return (
      <div className="rounded-[12px] bg-[#f8f7f5] px-4 pb-4 pt-3.5 text-xs dark:bg-white/[0.035]">
        {renderProgressMeter(doneCount, totalCount, '项')}
        <div className="mt-4">
          {steps.map((step, index) => {
            const isLast = index === steps.length - 1;
            const isActive = step.status === 'active';
            return (
              <div
                key={step.key}
                className={`agent-op-enter relative flex gap-3 ${isLast ? '' : 'min-h-[44px] pb-3'}`}
              >
                {!isLast && (
                  <span
                    className="absolute bottom-[-1px] left-[9.5px] top-5 w-px bg-[#dedbd8] dark:bg-white/10"
                    aria-hidden="true"
                  />
                )}
                {step.status === 'done' ? (
                  <span className="relative z-[1] flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-[#242629] text-white shadow-[0_2px_5px_rgba(0,0,0,0.16)] dark:bg-gray-200 dark:text-gray-900">
                    <Check className="h-3 w-3" strokeWidth={3} aria-hidden="true" />
                  </span>
                ) : isActive ? (
                  <span className="relative z-[1] flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full border border-[#d8d5d2] bg-white dark:border-white/15 dark:bg-[#292c32]">
                    {isRunning ? (
                      <Loader2
                        className="h-4 w-4 animate-spin text-[#242629] motion-reduce:animate-none dark:text-gray-200"
                        strokeWidth={2.2}
                        aria-hidden="true"
                      />
                    ) : (
                      <span className="h-1.5 w-1.5 rounded-full bg-gray-500 dark:bg-gray-400" aria-hidden="true" />
                    )}
                  </span>
                ) : (
                  <span
                    className="relative z-[1] h-5 w-5 flex-shrink-0 rounded-full border border-[#dedbd8] bg-[#f8f7f5] dark:border-white/15 dark:bg-[#30333a]"
                    aria-hidden="true"
                  />
                )}
                <div className="min-w-0 flex-1 pt-px">
                  <div
                    className={`line-clamp-2 leading-[19px] ${
                      step.status === 'done'
                        ? 'text-gray-500 dark:text-gray-400'
                        : isActive
                          ? 'font-medium text-gray-800 dark:text-gray-100'
                          : 'text-gray-400 dark:text-gray-500'
                    }`}
                  >
                    {step.label}
                  </div>
                  {step.key === roundHostKey && renderCurrentRounds && (
                    <div className="mt-2.5 overflow-hidden">
                      {renderCurrentRounds()}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderOperation = (op, index) => {
    const meta = getToolMeta(op.tool);
    const Icon = meta.icon;
    const isDone = op.status === 'done' || Number.isFinite(op.resultCount);
    const isExecuting = !isDone;
    const elapsed = formatElapsedMs(op.elapsedMs);

    return (
      <div
        key={index}
        className="agent-op-enter flex items-start gap-2.5 px-1.5 py-2 text-[11px]"
      >
        <span
          className={`mt-px flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full ${
            isExecuting
              ? 'bg-[#e8e6e3] text-gray-600 dark:bg-white/10 dark:text-gray-300'
              : 'bg-white/75 text-gray-400 dark:bg-white/[0.055] dark:text-gray-500'
          }`}
          aria-hidden="true"
        >
          <Icon className="h-3 w-3" strokeWidth={1.9} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex min-h-5 flex-wrap items-center gap-x-1.5 gap-y-0.5 leading-5">
            <span className="font-medium text-gray-700 dark:text-gray-200">{meta.label}</span>
            {Number.isFinite(op.resultCount) && (
              <span className="agent-op-enter tabular-nums text-gray-500 dark:text-gray-400">
                → {op.resultCount} 个结果
              </span>
            )}
            {elapsed && <span className="tabular-nums text-gray-400 dark:text-gray-500">· {elapsed}</span>}
            {isExecuting && (
              <span className="inline-flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gray-500 motion-reduce:animate-none dark:bg-gray-400" aria-hidden="true" />
                执行中
              </span>
            )}
          </div>
          {(op.resultMessage || op.message) && (
            <div className="mt-0.5 line-clamp-2 break-words leading-[18px] text-gray-500 dark:text-gray-400">
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
    const isCurrentRound = isRunning && roundIndex === rounds.length - 1;

    const isLast = roundIndex === rounds.length - 1;

    return (
      <div key={round} className={`agent-op-enter relative flex gap-3 ${isLast ? '' : 'pb-2'}`}>
        {/* 连接竖线：与任务步骤轴同款，贯穿相邻状态圆 */}
        {!isLast && (
          <span
            className="absolute bottom-[-1px] left-[9.5px] top-6 w-px bg-[#dedbd8] dark:bg-white/10"
            aria-hidden="true"
          />
        )}
        {/* 状态圆：完成 = 深色对勾圆，进行中 = 白底旋转环 */}
        {isCurrentRound ? (
          <span
            className="relative z-[1] mt-1.5 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full border border-[#d8d5d2] bg-white dark:border-white/15 dark:bg-[#292c32]"
            aria-hidden="true"
          >
            <Loader2 className="h-4 w-4 animate-spin text-[#242629] motion-reduce:animate-none dark:text-gray-200" strokeWidth={2.2} />
          </span>
        ) : (
          <span
            className="relative z-[1] mt-1.5 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-[#242629] text-white shadow-[0_2px_5px_rgba(0,0,0,0.16)] dark:bg-gray-200 dark:text-gray-900"
            aria-hidden="true"
          >
            <Check className="h-3 w-3" strokeWidth={3} />
          </span>
        )}

        <div className="min-w-0 flex-1">
          <button
            type="button"
            onClick={() => toggleRound(round)}
            aria-expanded={isExpanded}
            className="flex w-full items-center gap-2 rounded-[8px] px-2 py-2 text-left text-xs transition-colors duration-200 hover:bg-black/[0.025] active:bg-black/[0.045] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gray-400/35 dark:hover:bg-white/[0.035] dark:active:bg-white/[0.055]"
          >
            <span
              className={`min-w-0 flex-1 truncate ${
                isCurrentRound
                  ? 'font-medium text-gray-800 dark:text-gray-100'
                  : 'text-gray-700 dark:text-gray-200'
              }`}
            >
              第 {round} 轮 · {opCount} 个工具{successCount > 0 ? ` · ${successCount} 命中` : ''}
            </span>
            {invocationMode === 'native_tools' && (
              <span className="text-[10px] text-gray-400 dark:text-gray-500">原生工具</span>
            )}
            {invocationMode === 'json_fallback' && (
              <span className="text-[10px] font-medium text-orange-600 dark:text-orange-300">JSON 兜底</span>
            )}
            {roundData.planningMessage && !operations.length && (
              <span className="inline-flex items-center text-[10px] text-gray-500 dark:text-gray-400">
                规划中
                {isCurrentRound && <BouncingDots className="bg-gray-500 dark:bg-gray-400" />}
              </span>
            )}
            <ChevronDown
              className={`h-3.5 w-3.5 flex-shrink-0 text-gray-400 transition-transform duration-300 ${
                isExpanded ? 'rotate-0' : '-rotate-90'
              }`}
            />
          </button>
          {/* 轮次明细：与外层收合同款的 grid-rows 展开动画，消灭瞬时跳变 */}
          <div
            className={`grid transition-[grid-template-rows,opacity] duration-300 ease-out motion-reduce:transition-none ${
              isExpanded ? 'grid-rows-[1fr] opacity-100' : 'pointer-events-none grid-rows-[0fr] opacity-0'
            }`}
            aria-hidden={!isExpanded}
          >
            <div className="min-h-0 overflow-hidden">
              <div className="pb-2 pl-1 pr-0.5 pt-0.5">
                {roundData.planningMessage && (
                  <div className="flex items-center gap-1.5 px-1.5 pb-1 pt-0.5 text-[11px] italic text-gray-500 dark:text-gray-400">
                    <Sparkles className="h-3 w-3 flex-shrink-0 text-gray-400 dark:text-gray-500" strokeWidth={1.8} />
                    <span className="line-clamp-2">{roundData.planningMessage}</span>
                  </div>
                )}
                {operations.length === 0 ? (
                  <div className="px-1.5 py-1 text-[11px] italic text-gray-400">该轮未执行任何工具</div>
                ) : (
                  <div className="space-y-0.5">
                    {operations.map((op, opIndex) => renderOperation(op, opIndex))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderRoundProgress = () => {
    const completedRounds = isRunning ? Math.max(0, rounds.length - 1) : rounds.length;

    return (
      <div className="rounded-[12px] bg-[#f8f7f5] px-4 pb-4 pt-3.5 text-xs dark:bg-white/[0.035]">
        {renderProgressMeter(completedRounds, rounds.length, '轮')}
        <div className="mt-4">
          {rounds.length > 0 ? (
            rounds.map((roundData, index) => renderRound(roundData, index))
          ) : (
            <div className="agent-op-enter flex items-center gap-3 text-gray-600 dark:text-gray-300">
              <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full border border-[#d8d5d2] bg-white dark:border-white/15 dark:bg-[#292c32]">
                <Loader2
                  className="h-4 w-4 animate-spin text-[#242629] motion-reduce:animate-none dark:text-gray-200"
                  strokeWidth={2.2}
                  aria-hidden="true"
                />
              </span>
              <span className="font-medium">正在生成检索计划</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const rootClass = embedded
    ? 'mt-3 border-t border-[#eee5e0] pt-2 dark:border-white/[0.07]'
    : 'mt-2 ml-2 max-w-2xl rounded-[16px] border border-[#eadfd8] bg-white p-2 shadow-[0_14px_30px_-23px_rgba(91,65,52,0.5)] dark:border-white/[0.09] dark:bg-[#25282f]';

  return (
    <div className={rootClass}>
      <button
        type="button"
        onClick={() => setCollapsed((prev) => !prev)}
        className="group flex w-full items-center gap-2 rounded-[8px] px-1.5 py-2 text-left text-xs text-gray-600 transition-colors duration-200 hover:bg-[#faf9f7] hover:text-gray-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400/35 dark:text-gray-300 dark:hover:bg-white/[0.03] dark:hover:text-gray-100"
        aria-expanded={!collapsed}
      >
        <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center text-gray-500 dark:text-gray-400" aria-hidden="true">
          <ScanSearch className="h-4 w-4" strokeWidth={1.8} />
        </span>
        <span className="flex-shrink-0 font-medium">{isRunning ? '检索轨迹' : '检索完成'}</span>
        {isRunning && (
          <span className="inline-flex flex-shrink-0 items-center gap-1.5 text-[10px] font-medium tabular-nums text-gray-500 dark:text-gray-400">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gray-600 motion-reduce:animate-none dark:bg-gray-300" aria-hidden="true" />
            检索中{liveDuration ? ` ${liveDuration}` : ''}
          </span>
        )}
        {stats && (
          <span className="min-w-0 truncate text-gray-400 dark:text-gray-500">
            {stats.roundCount} 轮 · {stats.opCount} 次工具
            {stats.totalResults > 0 ? ` · ${stats.totalResults} 个结果` : ''}
            {!isRunning && duration ? ` · ${duration}` : ''}
          </span>
        )}
        {trace.fallback && (
          <span className="ml-auto flex-shrink-0 rounded-[5px] border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
            已降级
          </span>
        )}
        <ChevronDown
          className={`ml-auto h-3.5 w-3.5 flex-shrink-0 text-gray-400 transition-transform duration-300 ${
            collapsed ? '-rotate-90' : 'rotate-0'
          }`}
        />
      </button>

      <div
        className={`grid transition-[grid-template-rows,opacity,visibility] duration-300 ease-out motion-reduce:transition-none ${
          collapsed ? 'invisible grid-rows-[0fr] opacity-0' : 'visible grid-rows-[1fr] opacity-100'
        }`}
        aria-hidden={collapsed}
      >
        <div className="min-h-0 overflow-hidden">
          <div className="flex flex-col gap-1.5 px-1.5 pb-1 pt-1">
            {renderMetaSummary()}

            {hasTaskStatus &&
              renderTaskStatus(
                rounds.length > 0
                  ? () => rounds.map((roundData, index) => renderRound(roundData, index))
                  : null
              )}

            {!hasTaskStatus && (rounds.length > 0 || isRunning) && renderRoundProgress()}

            {subQuestions.length > 0 && (
              <div className="border-t border-[#eee8e4] px-1 py-2 text-xs dark:border-white/[0.07]">
                <div className="mb-2 flex items-center gap-1.5 font-medium text-gray-700 dark:text-gray-200">
                  <Search className="h-3.5 w-3.5 text-gray-400 dark:text-gray-500" />
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

            {trace.finalMessage && (
              <div className="agent-op-enter ml-[9.5px] border-l border-[#ddd8d4] py-1 pl-[21px] text-[11px] italic text-gray-500 dark:border-white/10 dark:text-gray-400">
                {trace.finalMessage}
              </div>
            )}

            {Array.isArray(trace.agentDetail) && trace.agentDetail.length > 0 && (
              <details className="ml-[9.5px] border-l border-[#ddd8d4] py-1.5 pl-[21px] text-[11px] text-gray-500 dark:border-white/10 dark:text-gray-400">
                <summary className="cursor-pointer select-none font-medium text-gray-600 dark:text-gray-300">
                  已纳入意群 <span className="font-normal text-gray-400">({trace.agentDetail.length})</span>
                </summary>
                <div className="mt-2 flex max-h-24 flex-wrap gap-1 overflow-y-auto pr-1">
                  {trace.agentDetail.map((detail, index) => (
                    <span
                      key={index}
                      className="inline-flex items-center gap-1 rounded-[6px] bg-[#f4f1ee] px-1.5 py-0.5 text-gray-600 dark:bg-white/[0.06] dark:text-gray-300"
                    >
                      <Layers className="h-3 w-3 text-gray-400" />
                      {detail.group_id}
                      {detail.granularity && <span className="text-gray-400">· {detail.granularity}</span>}
                      {Number.isFinite(detail.char_count) && (
                        <span className="text-gray-400">· {detail.char_count}字</span>
                      )}
                    </span>
                  ))}
                </div>
              </details>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
