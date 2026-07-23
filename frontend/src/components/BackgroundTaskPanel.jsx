import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  MoreHorizontal,
  RotateCcw,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import MinerUScanLoader from './MinerUScanLoader';

const VISIBLE_TASK_STATES = new Set(['running', 'failed', 'recommended']);

const getTaskProgress = (progress) => {
  if (!progress || typeof progress !== 'object') return null;
  const percent = Number(progress.percent);
  if (!Number.isFinite(percent)) return null;
  const normalizedPercent = Math.max(0, Math.min(100, Math.round(percent)));
  return {
    percent: normalizedPercent,
    label: progress.label || `${normalizedPercent}%`,
    ariaLabel: progress.ariaLabel || `进度 ${normalizedPercent}%`,
  };
};

export const TASK_PILL_PANEL_MOTION = {
  initial: {
    opacity: 0,
    x: 0,
    y: -52,
    scaleX: 0.2,
    scaleY: 0.12,
    borderRadius: 999,
  },
  animate: {
    opacity: 1,
    x: 0,
    y: 0,
    scaleX: 1,
    scaleY: 1,
    borderRadius: 20,
  },
  exit: {
    opacity: 0,
    x: 0,
    y: -52,
    scaleX: 0.2,
    scaleY: 0.12,
    borderRadius: 999,
  },
  transition: {
    type: 'spring',
    stiffness: 500,
    damping: 40,
    mass: 0.72,
    opacity: { duration: 0.14, ease: [0.22, 1, 0.36, 1] },
    borderRadius: { duration: 0.2, ease: [0.22, 1, 0.36, 1] },
  },
};

export const getVisibleBackgroundTasks = (items, hasDocument) => {
  if (!hasDocument || !Array.isArray(items)) return [];
  return items.filter((item) => VISIBLE_TASK_STATES.has(item?.state));
};

export const getBackgroundTaskSummary = (items) => {
  const tasks = Array.isArray(items) ? items : [];
  const failedCount = tasks.filter((item) => item.state === 'failed').length;
  const runningCount = tasks.filter((item) => item.state === 'running').length;
  const recommendedCount = tasks.filter((item) => item.state === 'recommended').length;
  const minerUProgress = getTaskProgress(
    tasks.find((item) => item?.id === 'deep_parse' && item?.state === 'running')?.progress
  );

  if (failedCount > 0) return { state: 'failed', label: '任务异常', count: failedCount };
  if (minerUProgress) return { state: 'running', label: `解析 ${minerUProgress.percent}%`, count: runningCount };
  if (runningCount > 0) return { state: 'running', label: `任务 ${runningCount}`, count: runningCount };
  if (recommendedCount > 0) return { state: 'recommended', label: `建议 ${recommendedCount}`, count: recommendedCount };
  return { state: 'idle', label: '任务', count: 0 };
};

const TASK_STYLES = {
  running: {
    Icon: Loader2,
    lightIconClassName: 'animate-spin text-[#ed8c68]',
    darkIconClassName: 'animate-spin text-[#FFA07A]',
    lightStatusClassName: 'text-[#B85F47]',
    darkStatusClassName: 'text-[#FFA07A]',
  },
  failed: {
    Icon: AlertCircle,
    lightIconClassName: 'text-rose-500',
    darkIconClassName: 'text-rose-300',
    lightStatusClassName: 'text-rose-600',
    darkStatusClassName: 'text-rose-300',
  },
  recommended: {
    Icon: Sparkles,
    lightIconClassName: 'text-amber-500',
    darkIconClassName: 'text-amber-300',
    lightStatusClassName: 'text-amber-700',
    darkStatusClassName: 'text-amber-300',
  },
};

const BackgroundTaskPanel = ({
  items = [],
  autoEnabled,
  onAutoEnabledChange,
  onClose,
  onClearCache,
  canClearCache,
  onRollbackRagIndex,
  canRollbackRagIndex,
  darkMode = false,
}) => {
  const [showMaintenanceMenu, setShowMaintenanceMenu] = useState(false);
  const summary = useMemo(() => getBackgroundTaskSummary(items), [items]);

  return (
    <motion.div
      onClick={(event) => event.stopPropagation()}
      initial={TASK_PILL_PANEL_MOTION.initial}
      animate={TASK_PILL_PANEL_MOTION.animate}
      exit={TASK_PILL_PANEL_MOTION.exit}
      transition={TASK_PILL_PANEL_MOTION.transition}
      style={{ transformOrigin: 'top right' }}
      className={`absolute right-0 top-[3.25rem] w-[330px] will-change-transform overflow-hidden rounded-[20px] border shadow-[0_24px_60px_rgba(15,23,42,0.16)] backdrop-blur-xl ${
        darkMode
          ? 'border-white/10 bg-[#1f2329]/95 text-gray-100'
          : 'border-white/80 bg-white/95 text-gray-900'
      }`}
      role="dialog"
      aria-label="后台任务"
    >
      <div className={`flex items-start justify-between gap-3 border-b px-4 pb-3 pt-4 ${darkMode ? 'border-white/[0.08]' : 'border-[#eee8e3]'}`}>
        <div>
          <div className="text-[14px] font-bold">后台任务</div>
          <div className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            {summary.state === 'idle' ? '当前没有运行或待处理任务' : summary.label}
          </div>
        </div>
        <div className="relative flex items-center gap-1">
          <button
            type="button"
            onClick={() => setShowMaintenanceMenu((current) => !current)}
            aria-label="后台任务更多操作"
            title="更多操作"
            className={`rounded-full p-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-600'}`}
          >
            <MoreHorizontal className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭后台任务"
            title="关闭"
            className={`rounded-full p-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-600'}`}
          >
            <X className="h-4 w-4" />
          </button>

          {showMaintenanceMenu && (
            <div className={`absolute right-8 top-9 z-10 w-44 rounded-[12px] border p-1.5 shadow-lg ${darkMode ? 'border-white/10 bg-[#292d34]' : 'border-gray-100 bg-white'}`}>
              {canRollbackRagIndex && onRollbackRagIndex && (
                <button
                  type="button"
                  onClick={() => {
                    setShowMaintenanceMenu(false);
                    onRollbackRagIndex();
                  }}
                  className={`flex w-full items-center gap-2 rounded-[9px] px-3 py-2 text-left text-[11px] font-semibold transition-colors ${darkMode ? 'text-gray-300 hover:bg-white/10' : 'text-gray-600 hover:bg-gray-50'}`}
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  回退到本地问答索引
                </button>
              )}
              <button
                type="button"
                onClick={() => {
                  setShowMaintenanceMenu(false);
                  onClearCache?.();
                }}
                disabled={!canClearCache}
                className={`flex w-full items-center gap-2 rounded-[9px] px-3 py-2 text-left text-[11px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${darkMode ? 'text-gray-300 hover:bg-white/10' : 'text-gray-600 hover:bg-gray-50'}`}
              >
                <Trash2 className="h-3.5 w-3.5" />
                清理当前文档缓存
              </button>
            </div>
          )}
        </div>
      </div>

      <div className={`flex items-center justify-between px-4 py-3 ${darkMode ? 'bg-white/[0.025]' : 'bg-[#fbfaf8]'}`}>
        <div>
          <div className="text-[12px] font-semibold">自动后台处理</div>
          <div className={`mt-0.5 text-[10px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>总结、大纲与翻译按当前阅读设置执行</div>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={autoEnabled}
          aria-label="自动后台处理"
          onClick={() => onAutoEnabledChange?.(!autoEnabled)}
          className={`relative h-6 w-11 shrink-0 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${autoEnabled ? 'bg-[#D97A5D]' : darkMode ? 'bg-white/15' : 'bg-gray-200'}`}
        >
          <span className={`absolute left-0 top-1 h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${autoEnabled ? 'translate-x-6' : 'translate-x-1'}`} />
        </button>
      </div>

      <div className={`max-h-[360px] overflow-y-auto px-4 ${darkMode ? 'divide-white/[0.08]' : 'divide-[#eee8e3]'} divide-y`}>
        {items.length === 0 ? (
          <div className="flex flex-col items-center px-4 py-8 text-center">
            <CheckCircle2 className={`h-7 w-7 ${darkMode ? 'text-gray-600' : 'text-gray-300'}`} />
            <div className={`mt-2 text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>没有后台任务</div>
            <div className={`mt-1 max-w-[220px] text-[10px] leading-4 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              手动生成请前往总结、大纲、速览或阅读面板
            </div>
          </div>
        ) : items.map((item) => {
          const taskStyle = TASK_STYLES[item.state] || TASK_STYLES.running;
          const TaskIcon = taskStyle.Icon;
          const showAction = item.actionLabel && (!item.busy || item.actionLabel === '取消');
          const taskProgress = getTaskProgress(item.progress);
          const showMinerUScanLoader = item.id === 'deep_parse' && item.state === 'running';
          return (
            <div key={item.id} className="flex items-start gap-3 py-3.5">
              {showMinerUScanLoader ? (
                <MinerUScanLoader
                  size={20}
                  className={`mt-0.5 ${darkMode ? 'text-[#FFA07A]' : 'text-[#ed8c68]'}`}
                />
              ) : (
                <TaskIcon className={`mt-0.5 h-4 w-4 shrink-0 ${darkMode ? taskStyle.darkIconClassName : taskStyle.lightIconClassName}`} />
              )}
              <div className="min-w-0 flex-1">
                <div className="flex min-w-0 items-center gap-2">
                  <span className="truncate text-[12px] font-bold">{item.title}</span>
                  <span className={`shrink-0 text-[10px] font-semibold ${darkMode ? taskStyle.darkStatusClassName : taskStyle.lightStatusClassName}`}>{item.status}</span>
                </div>
                <div className={`mt-1 text-[10px] leading-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{item.desc}</div>
                {taskProgress && (
                  <div className="mt-2 flex items-center gap-2">
                    <div
                      role="progressbar"
                      aria-label={`${item.title}：${taskProgress.ariaLabel}`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={taskProgress.percent}
                      className={`relative h-1.5 min-w-0 flex-1 overflow-hidden rounded-full ${darkMode ? 'bg-white/10' : 'bg-[#eee8e3]'}`}
                    >
                      <motion.div
                        initial={false}
                        animate={{ scaleX: taskProgress.percent / 100 }}
                        transition={{ duration: 0.35, ease: 'easeOut' }}
                        className="absolute inset-y-0 left-0 w-full origin-left rounded-full bg-[#D97A5D]"
                      />
                    </div>
                    <span className={`shrink-0 text-[10px] font-semibold tabular-nums ${darkMode ? 'text-[#FFA07A]' : 'text-[#B85F47]'}`}>
                      {taskProgress.label}
                    </span>
                  </div>
                )}
              </div>
              {showAction && (
                <button
                  type="button"
                  onClick={() => item.onAction?.()}
                  disabled={item.disabled}
                  className={`shrink-0 rounded-[8px] px-2.5 py-1.5 text-[10px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 disabled:cursor-not-allowed disabled:opacity-40 ${darkMode ? 'bg-white/[0.07] text-gray-200 hover:bg-white/10' : 'bg-[#fff4ef] text-[#B85F47] hover:bg-[#ffe8df]'}`}
                >
                  {item.actionLabel}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </motion.div>
  );
};

export default BackgroundTaskPanel;
