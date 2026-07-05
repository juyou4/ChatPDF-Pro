import React, { useEffect, useState } from 'react';
import { FileText, BookOpen, FlaskConical, Image, Award, Loader2, RefreshCw } from 'lucide-react';

const DEPTH_OPTIONS = [
  { id: 'brief', label: '简略' },
  { id: 'standard', label: '标准' },
  { id: 'detailed', label: '详细' },
];

const getFigureMetaLabel = (meta) => {
  if (!meta?.source || meta.source === 'none') return '';
  const sourceLabels = {
    mineru: 'MinerU 增强',
    yolo: 'YOLO 识别',
    pdf_native: 'PDF 原生',
    caption_only: '矢量兜底',
    fallback: '图片兜底',
  };
  const sourceLabel = sourceLabels[meta.source] || meta.source;
  if (meta.render_mode === 'yolo' && meta.source !== 'yolo') {
    return `YOLO 裁剪 · ${sourceLabel}`;
  }
  return sourceLabel;
};

const getFigureModeLabel = (mode) => {
  if (mode === 'yolo') return 'YOLO 模式';
  return '原始模式';
};

/**
 * 速览（Overview）面板组件
 * 展示结构化的学术导读五卡片
 */
const OverviewPanel = ({
  overview,
  loading,
  error,
  depth,
  figureMode = 'raw',
  onDepthChange,
  onFetch,
  docId,
}) => {
  const figureMetaLabel = getFigureMetaLabel(overview?.figure_meta);
  const figureModeLabel = getFigureModeLabel(overview?.figure_meta?.render_mode || figureMode);
  const activeDepth = depth || 'standard';
  const [regenerating, setRegenerating] = useState(false);
  const busy = loading || regenerating;
  const handleRegenerate = async () => {
    if (!docId || busy) return;
    setRegenerating(true);
    try {
      await onFetch?.(activeDepth, { force: true });
    } catch {
      // fetchOverview 会写入 error 状态，这里只负责恢复按钮反馈。
    } finally {
      setRegenerating(false);
    }
  };

  // 只在当前面板没有结果时自动获取；已有结果时切换 tab 不再触发额外请求。
  useEffect(() => {
    if (docId && onFetch && !overview && !loading && !error) {
      onFetch(activeDepth);
    }
  }, [docId, activeDepth, onFetch, overview, loading, error]);

  const toolbar = (
    <div className="shrink-0 px-4 pb-3">
      <div className="flex items-center justify-between gap-3 rounded-[22px] border border-white/70 bg-white/60 px-3 py-2 shadow-[0_12px_32px_rgba(148,163,184,0.16),inset_0_1px_0_rgba(255,255,255,0.9)] backdrop-blur-xl">
        <div className="flex min-w-0 items-center gap-2">
          <span className="hidden text-[12px] font-semibold text-gray-400 sm:inline">速览深度</span>
          <div className="relative grid w-[150px] grid-cols-3 rounded-[18px] bg-gray-100/80 p-1">
            <div
              className="absolute bottom-1 top-1 rounded-[14px] bg-white shadow-[0_8px_20px_rgba(139,124,200,0.14),0_1px_4px_rgba(31,41,55,0.08)] transition-transform duration-300"
              style={{
                width: 'calc((100% - 0.5rem) / 3)',
                transform: `translateX(${Math.max(0, DEPTH_OPTIONS.findIndex((item) => item.id === activeDepth)) * 100}%)`,
              }}
            />
            {DEPTH_OPTIONS.map((item) => {
              const isActive = item.id === activeDepth;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onDepthChange?.(item.id)}
                  className={`relative z-10 h-7 rounded-[14px] text-[12px] font-semibold transition-colors ${
                    isActive ? 'text-[#8b7cc8]' : 'text-gray-400 hover:text-gray-600'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </div>
          <span className="hidden shrink-0 rounded-full bg-white px-2.5 py-1 text-[11px] font-semibold text-gray-400 shadow-sm sm:inline-flex">
            {figureModeLabel}
          </span>
        </div>

        <button
          type="button"
          onClick={handleRegenerate}
          disabled={busy || !docId}
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-[16px] bg-white px-3 text-[12px] font-semibold text-gray-600 shadow-[0_8px_18px_rgba(148,163,184,0.16),inset_0_1px_0_rgba(255,255,255,0.95)] transition-all hover:-translate-y-0.5 hover:text-[#8b7cc8] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0"
        >
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          {busy ? '重新生成中' : '重新生成'}
        </button>
      </div>
      {activeDepth === 'detailed' && (
        <div className="px-3 pt-2 text-[11px] font-medium text-gray-400">
          详细速览会读取更多上下文并消耗更多 token
        </div>
      )}
    </div>
  );

  // 加载状态
  if (loading) {
    return (
      <div className="flex h-full flex-col">
        {toolbar}
        <div className="flex flex-1 flex-col items-center justify-center text-gray-400">
          <Loader2 className="w-8 h-8 animate-spin mb-3 text-purple-500" />
          <p className="text-sm">正在生成速览...</p>
          <p className="text-xs mt-1 text-gray-300">根据文档内容进行分析</p>
        </div>
      </div>
    );
  }

  // 错误状态
  if (error) {
    return (
      <div className="flex h-full flex-col">
        {toolbar}
        <div className="flex flex-1 flex-col items-center justify-center text-gray-400">
          <div className="text-red-400 mb-2 text-sm">{error}</div>
          <button
            onClick={handleRegenerate}
            className="text-xs px-3 py-1.5 rounded-lg bg-purple-100 text-purple-600 hover:bg-purple-200 transition-colors"
          >
            重试
          </button>
        </div>
      </div>
    );
  }

  // 无数据状态
  if (!overview) {
    return (
      <div className="flex h-full flex-col">
        {toolbar}
        <div className="flex flex-1 flex-col items-center justify-center text-gray-400">
          <FileText className="w-12 h-12 mb-3 opacity-30" />
          <p className="text-sm">暂无速览数据</p>
          <button
            onClick={handleRegenerate}
            className="mt-3 text-xs px-4 py-2 rounded-lg bg-purple-600 text-white hover:bg-purple-700 transition-colors"
          >
            生成速览
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {toolbar}
      <div className="space-y-6 pb-4 overflow-y-auto custom-scrollbar px-2">
        {/* 全文概述卡片 */}
        <div className="bg-white rounded-3xl p-6 shadow-[0_6px_28px_-4px_rgba(0,0,0,0.13),0_2px_8px_rgba(0,0,0,0.07)] border border-gray-100/80 transition-shadow duration-300 hover:shadow-[0_8px_32px_-4px_rgba(0,0,0,0.12),0_2px_8px_rgba(0,0,0,0.06)]">
        <div className="flex items-center gap-2 mb-3">
          <FileText className="w-5 h-5 text-gray-700" strokeWidth={2} />
          <h3 className="text-lg font-semibold text-gray-900">全文概述</h3>
        </div>
        <p className="text-gray-600 leading-relaxed text-[15px]">
          {overview.full_text_summary}
        </p>
      </div>

      {/* 术语解释卡片 */}
      <div className="bg-white rounded-3xl p-6 shadow-[0_6px_28px_-4px_rgba(0,0,0,0.13),0_2px_8px_rgba(0,0,0,0.07)] border border-gray-100/80 transition-shadow duration-300 hover:shadow-[0_8px_32px_-4px_rgba(0,0,0,0.12),0_2px_8px_rgba(0,0,0,0.06)]">
        <div className="flex items-center gap-2 mb-5">
          <BookOpen className="w-5 h-5 text-gray-700" strokeWidth={2} />
          <h3 className="text-lg font-semibold text-gray-900">术语解释</h3>
        </div>
        {overview.terminology && overview.terminology.length > 0 ? (
          <div className="space-y-4">
            {overview.terminology.map((item, idx) => (
              <div key={idx} className="flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-4">
                <span className="inline-flex w-fit bg-[#F3E8FF] text-[#6B21A8] px-2.5 py-1 rounded-md text-sm font-medium shrink-0">
                  {item.term}
                </span>
                <span className="text-gray-600 text-[15px] leading-relaxed sm:pt-1">
                  {item.explanation}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-400 text-[15px]">暂无术语解释</p>
        )}
      </div>

      {/* 论文速读卡片 */}
      <div className="bg-white rounded-3xl p-6 shadow-[0_6px_28px_-4px_rgba(0,0,0,0.13),0_2px_8px_rgba(0,0,0,0.07)] border border-gray-100/80 transition-shadow duration-300 hover:shadow-[0_8px_32px_-4px_rgba(0,0,0,0.12),0_2px_8px_rgba(0,0,0,0.06)]">
        <div className="flex items-center gap-2 mb-4">
          <FlaskConical className="w-5 h-5 text-gray-700" strokeWidth={2} />
          <h3 className="text-lg font-semibold text-gray-900">论文速读</h3>
        </div>
        {overview.speed_read && (
          <div className="space-y-4 text-[15px] text-gray-600 leading-relaxed">
            <p>
              <strong className="text-gray-900 font-semibold">论文方法：</strong>
              {overview.speed_read.method}
            </p>
            <p>
              <strong className="text-gray-900 font-semibold">实验设计：</strong>
              {overview.speed_read.experiment_design}
            </p>
            <p>
              <strong className="text-gray-900 font-semibold">解决的问题：</strong>
              {overview.speed_read.problems_solved}
            </p>
          </div>
        )}
      </div>

      {/* 关键图表解读卡片 */}
      <div className="bg-white rounded-3xl p-6 shadow-[0_6px_28px_-4px_rgba(0,0,0,0.13),0_2px_8px_rgba(0,0,0,0.07)] border border-gray-100/80 transition-shadow duration-300 hover:shadow-[0_8px_32px_-4px_rgba(0,0,0,0.12),0_2px_8px_rgba(0,0,0,0.06)]">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Image className="w-5 h-5 text-gray-700" strokeWidth={2} />
            <h3 className="text-lg font-semibold text-gray-900">关键图表解读</h3>
          </div>
          {(figureMetaLabel || figureModeLabel) && (
            <span className="bg-[#EEF2FF] text-[#4F46E5] px-3 py-1 rounded-full text-xs font-semibold tracking-wide">
              {[figureModeLabel, figureMetaLabel].filter(Boolean).join(' · ')}
            </span>
          )}
        </div>
        
        {overview.key_figures && overview.key_figures.length > 0 ? (
          <div className="space-y-6">
            {overview.key_figures.map((figure, idx) => (
              <div key={idx}>
                <p className="text-center font-medium text-gray-800 mb-3">{figure.caption}</p>
                <div className="rounded-lg overflow-hidden border border-gray-200 bg-gray-100 mb-3">
                  {figure.image_base64 && (
                    <img
                      src={figure.image_base64}
                      alt={figure.caption}
                      className="w-full h-auto opacity-90"
                    />
                  )}
                </div>
                <p className="text-[15px] text-gray-600 leading-relaxed">{figure.analysis}</p>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col gap-3 rounded-2xl border border-dashed border-gray-200 bg-gray-50/70 p-4">
            <p className="text-gray-400 text-[15px]">
              暂无图表解读（重新生成速览可能获取）
            </p>
            <p className="text-xs leading-relaxed text-gray-400">
              当前使用{figureModeLabel}；图表来源为{figureMetaLabel || '未识别'}。如果正文有复杂架构图，可在设置中切换到 YOLO 模式后重新生成。
            </p>
            <button
              type="button"
              onClick={handleRegenerate}
              disabled={busy || !docId}
              className="inline-flex w-fit items-center gap-1.5 rounded-full bg-white px-3 py-1.5 text-xs font-semibold text-gray-600 shadow-sm transition-colors hover:text-[#8b7cc8] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              {busy ? '重新生成中' : '重新生成速览'}
            </button>
          </div>
        )}
      </div>

      {/* 论文总结卡片 */}
      <div className="bg-white rounded-3xl p-6 shadow-[0_6px_28px_-4px_rgba(0,0,0,0.13),0_2px_8px_rgba(0,0,0,0.07)] border border-gray-100/80 transition-shadow duration-300 hover:shadow-[0_8px_32px_-4px_rgba(0,0,0,0.12),0_2px_8px_rgba(0,0,0,0.06)]">
        <div className="flex items-center gap-2 mb-4">
          <Award className="w-5 h-5 text-gray-700" strokeWidth={2} />
          <h3 className="text-lg font-semibold text-gray-900">论文总结</h3>
        </div>
        {overview.paper_summary && (
          <div className="space-y-4 text-[15px] text-gray-600 leading-relaxed">
            <p>
              <strong className="text-gray-900 font-semibold">优点与创新：</strong>
              {overview.paper_summary.strengths || overview.paper_summary.innovations}
            </p>
            <p>
              <strong className="text-gray-900 font-semibold">未来展望：</strong>
              {overview.paper_summary.future_work}
            </p>
          </div>
        )}
      </div>
      </div>
    </div>
  );
};

export default OverviewPanel;
