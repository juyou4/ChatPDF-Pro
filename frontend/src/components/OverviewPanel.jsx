import React, { useEffect } from 'react';
import { FileText, BookOpen, FlaskConical, Image, Award, Loader2 } from 'lucide-react';

/**
 * 速览（Overview）面板组件
 * 展示结构化的学术导读五卡片
 */
const OverviewPanel = ({
  overview,
  loading,
  error,
  depth,
  onDepthChange,
  onFetch,
  docId,
}) => {
  // 当 docId 或 depth 变化时重新获取速览
  useEffect(() => {
    if (docId && onFetch) {
      onFetch(depth);
    }
  }, [docId, depth]);

  // 加载状态
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400">
        <Loader2 className="w-8 h-8 animate-spin mb-3 text-purple-500" />
        <p className="text-sm">正在生成速览...</p>
        <p className="text-xs mt-1 text-gray-300">根据文档内容进行分析</p>
      </div>
    );
  }

  // 错误状态
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400">
        <div className="text-red-400 mb-2 text-sm">{error}</div>
        <button
          onClick={() => onFetch?.(depth)}
          className="text-xs px-3 py-1.5 rounded-lg bg-purple-100 text-purple-600 hover:bg-purple-200 transition-colors"
        >
          重试
        </button>
      </div>
    );
  }

  // 无数据状态
  if (!overview) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400">
        <FileText className="w-12 h-12 mb-3 opacity-30" />
        <p className="text-sm">暂无速览数据</p>
        <button
          onClick={() => onFetch?.(depth)}
          className="mt-3 text-xs px-4 py-2 rounded-lg bg-purple-600 text-white hover:bg-purple-700 transition-colors"
        >
          生成速览
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-4 overflow-y-auto h-full custom-scrollbar px-2">
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
          {(overview.figure_meta?.source === 'mineru' || overview.figure_meta?.source === 'pdf_native') && (
            <span className="bg-[#EEF2FF] text-[#4F46E5] px-3 py-1 rounded-full text-xs font-semibold tracking-wide">
              {overview.figure_meta?.source === 'mineru' ? 'MinerU 增强' : 'PDF 原生'}
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
          <p className="text-gray-400 text-[15px]">
            暂无图表解读（可通过切换到「详细」模式获取）
          </p>
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
  );
};

export default OverviewPanel;
