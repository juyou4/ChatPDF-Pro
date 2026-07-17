import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Brain, History, Info } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  formatCompactTokenCount,
  getChatHistoryUsage,
} from '../utils/chatContextUsageUtils';

const CONTEXT_TONES = {
  low: {
    label: '余量充足',
    description: '历史仍在当前保留范围内',
    color: '#0EAA78',
    lightSurface: '#EDFBF5',
    lightBorder: '#B6E8D4',
    darkSurface: 'rgba(14, 170, 120, 0.13)',
    darkBorder: 'rgba(77, 217, 161, 0.34)',
  },
  warning: {
    label: '接近上限',
    description: '下一轮将接近当前历史保留上限',
    color: '#E58B13',
    lightSurface: '#FFF7E9',
    lightBorder: '#F4D49B',
    darkSurface: 'rgba(229, 139, 19, 0.13)',
    darkBorder: 'rgba(250, 183, 80, 0.34)',
  },
  critical: {
    label: '历史已满',
    description: '下一轮会替换最早的历史消息',
    color: '#E74957',
    lightSurface: '#FFF0F2',
    lightBorder: '#F5BBC3',
    darkSurface: 'rgba(231, 73, 87, 0.13)',
    darkBorder: 'rgba(246, 128, 139, 0.34)',
  },
  truncated: {
    label: '已截断历史',
    description: '更早的消息不会直接随下一轮请求发送',
    color: '#E74957',
    lightSurface: '#FFF0F2',
    lightBorder: '#F5BBC3',
    darkSurface: 'rgba(231, 73, 87, 0.13)',
    darkBorder: 'rgba(246, 128, 139, 0.34)',
  },
  disabled: {
    label: '历史已关闭',
    description: '当前请求不会附带之前的对话消息',
    color: '#8A949F',
    lightSurface: '#F5F6F7',
    lightBorder: '#D9DDE1',
    darkSurface: 'rgba(148, 163, 184, 0.12)',
    darkBorder: 'rgba(148, 163, 184, 0.28)',
  },
};

const getUsageSummary = (usage) => {
  if (!usage || typeof usage !== 'object') return null;

  const prompt = usage.prompt_tokens ?? usage.input_tokens ?? usage.promptTokenCount ?? usage.inputTokenCount ?? null;
  const completion = usage.completion_tokens ?? usage.output_tokens ?? usage.candidatesTokenCount ?? usage.outputTokenCount ?? null;
  const total = usage.total_tokens ?? usage.totalTokenCount ?? (
    Number.isFinite(Number(prompt)) && Number.isFinite(Number(completion))
      ? Number(prompt) + Number(completion)
      : null
  );

  if (!Number.isFinite(Number(prompt)) && !Number.isFinite(Number(completion)) && !Number.isFinite(Number(total))) {
    return null;
  }

  return {
    prompt: Number.isFinite(Number(prompt)) ? Number(prompt) : null,
    completion: Number.isFinite(Number(completion)) ? Number(completion) : null,
    total: Number.isFinite(Number(total)) ? Number(total) : null,
    estimated: Boolean(usage.estimated),
  };
};

const getContextTone = (usage, retentionRatio) => {
  if (usage.configuredTurns <= 0) return { id: 'disabled', ...CONTEXT_TONES.disabled };
  if (usage.omittedMessageCount > 0) return { id: 'truncated', ...CONTEXT_TONES.truncated };
  if (retentionRatio >= 0.9) return { id: 'critical', ...CONTEXT_TONES.critical };
  if (retentionRatio >= 0.6) return { id: 'warning', ...CONTEXT_TONES.warning };
  return { id: 'low', ...CONTEXT_TONES.low };
};

const MetricRow = ({ label, value, hint, darkMode, emphasized = false }) => (
  <div className={`flex items-baseline justify-between gap-4 ${emphasized ? 'pt-2.5' : 'py-1.5'}`}>
    <div className="min-w-0">
      <div className={`text-[11px] font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{label}</div>
      {hint && <div className={`mt-0.5 text-[10px] leading-4 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{hint}</div>}
    </div>
    <div className={`shrink-0 text-right text-[12px] font-semibold tabular-nums ${emphasized ? (darkMode ? 'text-gray-100' : 'text-gray-900') : (darkMode ? 'text-gray-200' : 'text-gray-800')}`}>
      {value}
    </div>
  </div>
);

const formatUsageValue = (value) => (
  Number.isFinite(value) ? `${formatCompactTokenCount(value)} token` : '—'
);

export default function ChatContextIndicator({
  messages,
  contextCount,
  memoryEnabled,
  lastUsage,
  darkMode,
}) {
  const rootRef = useRef(null);
  const [isPinned, setIsPinned] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const usage = useMemo(() => getChatHistoryUsage(messages, contextCount), [messages, contextCount]);
  const lastCallUsage = useMemo(() => getUsageSummary(lastUsage), [lastUsage]);
  const isVisible = isPinned || isHovered;
  const canShow = usage.eligibleMessageCount > 0;
  const retentionRatio = usage.configuredTurns > 0
    ? Math.min(1, usage.selectedMessageCount / (usage.configuredTurns * 2))
    : 0;
  const contextTone = getContextTone(usage, retentionRatio);
  const circumference = 2 * Math.PI * 8;
  const dashOffset = circumference * (1 - retentionRatio);
  const toneSurface = darkMode ? contextTone.darkSurface : contextTone.lightSurface;
  const toneBorder = darkMode ? contextTone.darkBorder : contextTone.lightBorder;
  const ringTrackColor = darkMode ? 'rgba(255,255,255,0.13)' : 'rgba(94, 103, 113, 0.14)';

  useEffect(() => {
    if (!isPinned) return undefined;

    const handlePointerDown = (event) => {
      if (rootRef.current && !rootRef.current.contains(event.target)) setIsPinned(false);
    };
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') setIsPinned(false);
    };

    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isPinned]);

  if (!canShow) return null;

  const historyLabel = usage.configuredTurns > 0
    ? `${usage.selectedTurns}/${usage.configuredTurns} 轮`
    : '未保留';
  const historyPercent = Math.round(retentionRatio * 100);

  return (
    <div
      ref={rootRef}
      className="relative shrink-0"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <button
        type="button"
        data-context-status={contextTone.id}
        aria-label={`对话上下文，${contextTone.label}，${historyLabel}，约 ${formatCompactTokenCount(usage.selectedTokens)} token`}
        aria-expanded={isVisible}
        onClick={() => setIsPinned((value) => !value)}
        className={`group flex h-7 items-center gap-1.5 rounded-full border px-2 transition-[background-color,border-color,box-shadow] duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1 ${
          darkMode ? 'text-gray-200 hover:brightness-110 focus-visible:ring-offset-[#24272d]' : 'text-gray-700 hover:brightness-[0.985] focus-visible:ring-offset-[#fffdfb]'
        }`}
        style={{
          backgroundColor: toneSurface,
          borderColor: toneBorder,
          boxShadow: isVisible ? `0 4px 14px -8px ${contextTone.color}` : 'none',
        }}
        title={`对话上下文：${contextTone.label}`}
      >
        <span className="relative flex h-4 w-4 items-center justify-center" aria-hidden="true">
          <svg className="h-4 w-4 -rotate-90" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" fill="none" stroke={ringTrackColor} strokeWidth="2" />
            <circle
              cx="10"
              cy="10"
              r="8"
              fill="none"
              stroke={contextTone.color}
              strokeWidth="2"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={dashOffset}
              className="transition-[stroke-dashoffset] duration-300"
            />
          </svg>
          <History className="absolute h-2 w-2" style={{ color: contextTone.color }} strokeWidth={2.5} />
        </span>
        <span className="hidden text-[10px] font-semibold tabular-nums lg:inline">
          {formatCompactTokenCount(usage.selectedTokens)}
        </span>
      </button>

      <AnimatePresence>
        {isVisible && (
          <motion.section
            initial={{ opacity: 0, y: 7, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 7, scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 380, damping: 28, mass: 0.62 }}
            className={`absolute bottom-full left-0 z-50 mb-2 w-[292px] border px-4 py-3.5 shadow-[0_18px_34px_-18px_rgba(64,50,42,0.38)] ${
              darkMode
                ? 'border-white/[0.10] bg-[#272a30]/[0.98] shadow-[0_18px_34px_-18px_rgba(0,0,0,0.72)]'
                : 'border-[#e7ddd7] bg-[#fffdfb]/[0.98]'
            }`}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[10px]" style={{ backgroundColor: toneSurface, color: contextTone.color }}>
                  <History className="h-3.5 w-3.5" />
                </span>
                <div className="min-w-0">
                  <h3 className={`text-[12px] font-semibold ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>对话上下文</h3>
                  <p className={`mt-0.5 text-[10px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>下一轮会附带的历史记录</p>
                </div>
              </div>
              <span className="shrink-0 text-[11px] font-semibold" style={{ color: contextTone.color }}>{contextTone.label}</span>
            </div>

            <div className="mt-4">
              <div className="flex items-baseline justify-between gap-3">
                <span className={`text-[11px] font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>历史保留范围</span>
                <span className={`text-[12px] font-semibold tabular-nums ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>
                  {historyLabel}
                  <span className={`ml-1.5 text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>({historyPercent}%)</span>
                </span>
              </div>
              <div className={`mt-2 h-1.5 overflow-hidden rounded-full ${darkMode ? 'bg-white/[0.09]' : 'bg-[#eee9e5]'}`}>
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: contextTone.color }}
                  initial={false}
                  animate={{ width: `${historyPercent}%` }}
                  transition={{ type: 'spring', stiffness: 260, damping: 26, mass: 0.7 }}
                />
              </div>
              <p className={`mt-1.5 text-[10px] leading-4 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{contextTone.description}</p>
            </div>

            <div className={`mt-3.5 divide-y ${darkMode ? 'divide-white/[0.07]' : 'divide-[#eee6e1]'}`}>
              <MetricRow
                label="历史 token"
                value={`${formatCompactTokenCount(usage.selectedTokens)} token`}
                hint={`${usage.selectedMessageCount} 条会随下一轮请求发送`}
                darkMode={darkMode}
              />
              <MetricRow
                label="更早消息"
                value={usage.omittedMessageCount > 0 ? `${usage.omittedMessageCount} 条` : '无'}
                hint={usage.omittedMessageCount > 0 ? '不会直接发送' : '当前没有超出保留范围'}
                darkMode={darkMode}
              />
              <MetricRow
                label="长对话记忆"
                value={memoryEnabled ? '已启用' : '关闭'}
                hint={memoryEnabled ? '相关事实可由后台记忆补充' : '不额外注入已提炼记忆'}
                darkMode={darkMode}
              />
            </div>

            {lastCallUsage && (
              <div className={`mt-2.5 border-t ${darkMode ? 'border-white/[0.10]' : 'border-[#e8e0db]'}`}>
                <MetricRow
                  label="上一轮输入"
                  value={formatUsageValue(lastCallUsage.prompt)}
                  hint={lastCallUsage.estimated ? '客户端估算，含检索与系统提示' : '模型返回，含检索与系统提示'}
                  darkMode={darkMode}
                />
                <MetricRow
                  label="上一轮输出"
                  value={formatUsageValue(lastCallUsage.completion)}
                  darkMode={darkMode}
                />
                <MetricRow
                  label="上一轮总计"
                  value={formatUsageValue(lastCallUsage.total)}
                  darkMode={darkMode}
                  emphasized
                />
              </div>
            )}

            <div className={`mt-3 flex gap-1.5 border-t pt-2.5 text-[10px] leading-4 ${darkMode ? 'border-white/[0.07] text-gray-500' : 'border-[#eee6e1] text-gray-500'}`}>
              <Info className="mt-0.5 h-3 w-3 shrink-0" />
              <p>这里只统计对话历史；文档检索、系统提示和模型输出会按问题动态加入，不等同于模型总窗口。</p>
            </div>
          </motion.section>
        )}
      </AnimatePresence>
    </div>
  );
}
