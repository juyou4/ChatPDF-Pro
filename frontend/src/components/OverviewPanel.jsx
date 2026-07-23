import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, FileText, BookOpen, FlaskConical, Image, Award, Loader2, RefreshCw } from 'lucide-react';

const DEPTH_OPTIONS = [
  { id: 'brief', label: '简略' },
  { id: 'standard', label: '标准' },
  { id: 'detailed', label: '详细' },
];

const FIGURE_SOURCE_META = {
  mineru: {
    label: '结构化图表',
    detail: '来自 MinerU 主解析结果',
    className: 'border-[#f1d5ca] bg-[#fff4ef] text-[#a8553f]',
  },
  mineru_deep_parse: {
    label: '结构化图表',
    detail: '来自 MinerU 主解析结果',
    className: 'border-[#f1d5ca] bg-[#fff4ef] text-[#a8553f]',
  },
  yolo: {
    label: '本地图表定位',
    detail: '主解析未定位图表时使用 DocLayout-YOLO 定位与裁切',
    className: 'border-[#dbe5e1] bg-[#f2f7f5] text-[#4f7065]',
  },
  pdf_native: {
    label: 'PDF 图表',
    detail: '来自 PDF 原生图像与图注结构',
    className: 'border-[#dde3eb] bg-[#f4f7fa] text-[#5b6878]',
  },
  caption_only: {
    label: '图注兜底',
    detail: '根据图注位置恢复图表区域',
    className: 'border-[#e4e0db] bg-[#f8f6f3] text-[#756c64]',
  },
  fallback: {
    label: '图片兜底',
    detail: '使用文档内嵌图片作为基础结果',
    className: 'border-[#e4e0db] bg-[#f8f6f3] text-[#756c64]',
  },
};

const getFigureSourceMeta = (meta) => FIGURE_SOURCE_META[meta?.source] || null;

const getOverviewGenerationStatus = (overview) => {
  if (overview?.ai_meta?.fallback) return 'fallback';
  return overview?.ai_meta?.generation_status || 'completed';
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
  parseRoute = '',
  onDepthChange,
  onFetch,
  docId,
  availabilityState = '',
  availabilityMessage = '',
  availabilityActionLabel = '',
  onAvailabilityAction,
}) => {
  const figureSourceMeta = getFigureSourceMeta(overview?.figure_meta);
  const generationStatus = getOverviewGenerationStatus(overview);
  const degradedOverview = generationStatus === 'fallback' || generationStatus === 'partial';
  const generationError = String(
    overview?.ai_meta?.generation_error || overview?.figure_meta?.generation_error || '',
  ).trim();
  const activeDepth = depth || 'standard';
  const [regenerating, setRegenerating] = useState(false);
  const busy = loading || regenerating;
  const handleRegenerate = async () => {
    if (!docId || busy || !onFetch) return;
    setRegenerating(true);
    try {
      await onFetch(activeDepth, { force: true });
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
      <div className="flex items-center justify-between gap-3 rounded-[16px] border border-[#ebe5e0] bg-white/75 px-3 py-2 shadow-[0_8px_24px_rgba(104,88,79,0.08),inset_0_1px_0_rgba(255,255,255,0.95)] backdrop-blur-xl">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-[12px] font-semibold text-gray-400">速览深度</span>
          <div className="relative grid w-[150px] grid-cols-3 rounded-[12px] bg-[#f3f2f1] p-1">
            <div
              className="absolute bottom-1 top-1 rounded-[9px] bg-white shadow-[0_5px_14px_rgba(184,95,71,0.1),0_1px_3px_rgba(31,41,55,0.07)] transition-transform duration-300"
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
                  aria-pressed={isActive}
                  className={`relative z-10 h-7 rounded-[9px] text-[12px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/25 ${
                    isActive ? 'text-[#ce8e76]' : 'text-gray-400 hover:text-gray-600'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>

        <button
          type="button"
          onClick={handleRegenerate}
          disabled={busy || !docId || !onFetch}
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-[10px] border border-[#ebe5e0] bg-white px-3 text-[12px] font-semibold text-gray-600 shadow-[0_4px_12px_rgba(104,88,79,0.07)] transition-[border-color,color,box-shadow] hover:border-[#e2c8bc] hover:text-[#b85f47] hover:shadow-[0_6px_16px_rgba(104,88,79,0.1)] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          {busy ? '重新生成中' : availabilityState ? '等待解析' : '重新生成'}
        </button>
      </div>
      {activeDepth === 'detailed' && (
        <div className="px-3 pt-2 text-[11px] font-medium text-gray-400">
          详细速览会读取更多上下文并消耗更多 token
        </div>
      )}
    </div>
  );

  if (availabilityState) {
    const failed = availabilityState === 'failed' || availabilityState === 'cancelled';
    return (
      <div className="flex h-full flex-col">
        {toolbar}
        <div className="flex flex-1 flex-col items-center justify-center px-8 text-center text-gray-400">
          {failed ? (
            <AlertCircle className="mb-3 h-9 w-9 text-rose-400" />
          ) : (
            <Loader2 className="mb-3 h-9 w-9 animate-spin text-[#ed8c68]" />
          )}
          <p className={`text-sm font-semibold ${failed ? 'text-rose-500' : 'text-gray-600'}`}>
            {failed ? '主解析未完成' : '正在准备速览内容'}
          </p>
          <p className="mt-1.5 max-w-md text-xs leading-5 text-gray-400">
            {availabilityMessage || '主解析完成后会自动开放速览生成'}
          </p>
          {availabilityActionLabel && onAvailabilityAction && (
            <button
              type="button"
              onClick={onAvailabilityAction}
              className={`mt-4 inline-flex items-center gap-1.5 rounded-[11px] px-4 py-2 text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                failed
                  ? 'bg-rose-50 text-rose-700 hover:bg-rose-100 focus-visible:ring-rose-200'
                  : 'bg-[#FFF4EF] text-[#B85F47] hover:bg-[#FFE8DF] focus-visible:ring-[#D97A5D]/30'
              }`}
            >
              <RefreshCw className="h-3.5 w-3.5" />
              {availabilityActionLabel}
            </button>
          )}
        </div>
      </div>
    );
  }

  // 加载状态
  if (loading) {
    return (
      <div className="flex h-full flex-col">
        {toolbar}
        <div className="flex flex-1 flex-col items-center justify-center text-gray-400">
          <Loader2 className="w-8 h-8 animate-spin mb-3 text-[#F0653A]" />
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
            disabled={!onFetch}
            className="text-xs px-3.5 py-2 rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
          >
            重试
          </button>
        </div>
      </div>
    );
  }

  // 无数据状态：组合式引导，预告速览会生成哪些内容
  if (!overview) {
    return (
      <div className="flex h-full flex-col">
        {toolbar}
        <div className="flex flex-1 flex-col items-center justify-center px-8">
          <div className="w-full max-w-[320px]">
            <div className="flex flex-col items-center text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-gray-100/80 text-gray-400">
                <FileText className="h-6 w-6" />
              </div>
              <p className="mt-4 text-[14px] font-bold text-gray-700">还没有生成速览</p>
              <p className="mt-1 text-[12px] leading-relaxed text-gray-400">
                AI 会通读全文，提炼出下面几部分内容；详略可用上方「速览深度」调整
              </p>
            </div>
            <div className="mt-5 space-y-2">
              {[
                { Icon: BookOpen, label: '全文概述', bar: 'w-20' },
                { Icon: FlaskConical, label: '研究方法', bar: 'w-14' },
                { Icon: Award, label: '关键结论', bar: 'w-24' },
                { Icon: Image, label: '图表解读', bar: 'w-12' },
              ].map(({ Icon, label, bar }) => (
                <div key={label} className="flex items-center gap-3 rounded-[12px] bg-gray-50/80 px-3.5 py-2.5">
                  <Icon className="h-4 w-4 shrink-0 text-gray-400" />
                  <span className="text-[12px] font-semibold text-gray-500">{label}</span>
                  <span aria-hidden="true" className={`ml-auto h-1.5 rounded-full bg-gray-200/80 ${bar}`} />
                </div>
              ))}
            </div>
            <div className="mt-6 flex justify-center">
              <motion.button
                type="button"
                onClick={handleRegenerate}
                disabled={!onFetch}
                whileTap={{ scale: 0.98 }}
                className="accent-cta shrink-0 rounded-full px-5 py-2.5 text-[12px] font-bold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 disabled:cursor-not-allowed disabled:opacity-45"
              >
                生成速览
              </motion.button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {toolbar}
      <div className="space-y-6 pb-4 overflow-y-auto custom-scrollbar px-2">
        {degradedOverview && (
          <motion.div
            role="status"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            className="mx-1 flex items-start gap-3 rounded-[16px] border border-[#efd9c6] bg-[#fff8f1] px-4 py-3 text-[#8b5b3f]"
          >
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-[#d17a4f]" />
            <div className="min-w-0 flex-1">
              <p className="text-[12px] font-bold">
                {generationStatus === 'fallback' ? '当前展示基础速览' : '本次速览不完整'}
              </p>
              <p className="mt-0.5 text-[11px] leading-5 text-[#9b735d]">
                {generationStatus === 'fallback'
                  ? 'AI 深度总结未完成，这份内容不会作为正式速览缓存。'
                  : '部分核心内容缺失，这份结果不会作为正式速览缓存。'}
                {generationError ? ` 原因：${generationError.slice(0, 160)}` : ' 可稍后重新生成。'}
              </p>
            </div>
            <button
              type="button"
              onClick={handleRegenerate}
              disabled={busy || !onFetch}
              className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-[10px] bg-white px-3 text-[11px] font-semibold text-[#a65d40] shadow-[0_4px_12px_rgba(139,91,63,0.1)] transition-colors hover:bg-[#fff1e7] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              重试
            </button>
          </motion.div>
        )}
        {/* 全文概述卡片 */}
        <div className="bg-white rounded-3xl p-6 shadow-[0_6px_28px_-4px_rgba(0,0,0,0.13),0_2px_8px_rgba(0,0,0,0.07)] border border-gray-100/80 transition-shadow duration-300 hover:shadow-[0_8px_32px_-4px_rgba(0,0,0,0.12),0_2px_8px_rgba(0,0,0,0.06)]">
        <div className="flex items-center gap-2 mb-3">
          <FileText className="w-5 h-5 text-gray-700" strokeWidth={2} />
          <h3 className="text-lg font-semibold text-gray-900">
            {generationStatus === 'fallback' ? '文档内容摘录' : '全文概述'}
          </h3>
        </div>
        <p className="text-gray-600 leading-relaxed text-[15px]">
          {overview.full_text_summary}
        </p>
      </div>

      {generationStatus !== 'fallback' && (
      <>
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
                <span className="inline-flex w-fit bg-[#ffeee8] text-[#b24017] px-2.5 py-1 rounded-md text-sm font-medium shrink-0">
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
          {figureSourceMeta && (
            <span
              title={figureSourceMeta.detail}
              className={`inline-flex items-center gap-1.5 rounded-[8px] border px-2.5 py-1 text-[11px] font-semibold ${figureSourceMeta.className}`}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" />
              {figureSourceMeta.label}
            </span>
          )}
        </div>
        
        {overview.key_figures && overview.key_figures.length > 0 ? (
          <div className="space-y-6">
            {overview.key_figures.map((figure, idx) => (
              <div key={figure.figure_id || idx}>
                <p className="text-center font-medium text-gray-800 mb-3">{figure.caption}</p>
                <div className="mb-3 flex min-h-[180px] max-h-[560px] items-center justify-center overflow-hidden rounded-xl border border-gray-200 bg-gray-50 p-3">
                  {figure.image_base64 && (
                    <img
                      src={figure.image_base64}
                      alt={figure.caption}
                      className="block h-auto max-h-[536px] w-auto max-w-full object-contain opacity-95"
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
              当前解析结果未定位到可用图表
            </p>
            <p className="text-xs leading-relaxed text-gray-400">
              {parseRoute === 'mineru'
                ? '重新生成会继续复用 MinerU 主解析结果；图表内容解读由已选视觉模型单独完成。'
                : '重新生成会优先检查 PDF 原生结构，再使用本地图表定位；图表内容解读由已选视觉模型单独完成。'}
            </p>
            <button
              type="button"
              onClick={handleRegenerate}
              disabled={busy || !docId || !onFetch}
              className="inline-flex w-fit items-center gap-1.5 rounded-full bg-white px-3 py-1.5 text-xs font-semibold text-gray-600 shadow-sm transition-colors hover:text-[#ce8e76] disabled:cursor-not-allowed disabled:opacity-50"
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
      </>
      )}
      </div>
    </div>
  );
};

export default OverviewPanel;
