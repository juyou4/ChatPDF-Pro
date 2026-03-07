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
    <div className="space-y-4 overflow-y-auto pb-4">
      {/* 全文概述 */}
      <OverviewCard
        icon={<FileText className="w-5 h-5" />}
        title="全文概述"
        color="blue"
      >
        <p className="text-gray-700 leading-relaxed">
          {overview.full_text_summary}
        </p>
      </OverviewCard>

      {/* 术语解释 */}
      <OverviewCard
        icon={<BookOpen className="w-5 h-5" />}
        title="术语解释"
        color="green"
      >
        {overview.terminology && overview.terminology.length > 0 ? (
          <div className="space-y-2">
            {overview.terminology.map((item, idx) => (
              <div key={idx} className="flex items-start gap-2">
                <span className="font-medium text-purple-600 bg-purple-50 px-2 py-0.5 rounded text-sm whitespace-nowrap">
                  {item.term}
                </span>
                <span className="text-gray-600 text-sm">{item.explanation}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-400 text-sm">暂无术语解释</p>
        )}
      </OverviewCard>

      {/* 论文速读 */}
      <OverviewCard
        icon={<FlaskConical className="w-5 h-5" />}
        title="论文速读"
        color="orange"
      >
        {overview.speed_read && (
          <div className="space-y-3">
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                论文方法
              </h4>
              <p className="text-gray-700 text-sm leading-relaxed">
                {overview.speed_read.method}
              </p>
            </div>
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                实验设计
              </h4>
              <p className="text-gray-700 text-sm leading-relaxed">
                {overview.speed_read.experiment_design}
              </p>
            </div>
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                解决的问题
              </h4>
              <p className="text-gray-700 text-sm leading-relaxed">
                {overview.speed_read.problems_solved}
              </p>
            </div>
          </div>
        )}
      </OverviewCard>

      {/* 关键图表解读 */}
      <OverviewCard
        icon={<Image className="w-5 h-5" />}
        title="关键图表解读"
        color="purple"
      >
        {overview.key_figures && overview.key_figures.length > 0 ? (
          <div className="space-y-3">
            {overview.key_figures.map((figure, idx) => (
              <div key={idx} className="border border-gray-100 rounded-lg overflow-hidden">
                {figure.image_base64 && (
                  <img
                    src={figure.image_base64}
                    alt={figure.caption}
                    className="w-full h-auto"
                  />
                )}
                <div className="p-3 bg-gray-50">
                  <p className="text-xs font-medium text-gray-500 mb-1">
                    {figure.caption}
                  </p>
                  <p className="text-gray-700 text-sm">{figure.analysis}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-400 text-sm">
            暂无图表解读（可通过切换到「详细」模式获取）
          </p>
        )}
      </OverviewCard>

      {/* 论文总结 */}
      <OverviewCard
        icon={<Award className="w-5 h-5" />}
        title="论文总结"
        color="yellow"
      >
        {overview.paper_summary && (
          <div className="space-y-3">
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                优点与创新
              </h4>
              <p className="text-gray-700 text-sm leading-relaxed">
                {overview.paper_summary.strengths || overview.paper_summary.innovations}
              </p>
            </div>
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                未来展望
              </h4>
              <p className="text-gray-700 text-sm leading-relaxed">
                {overview.paper_summary.future_work}
              </p>
            </div>
          </div>
        )}
      </OverviewCard>
    </div>
  );
};

/**
 * 速览卡片组件
 */
const OverviewCard = ({ icon, title, color, children }) => {
  const colorClasses = {
    blue: 'bg-blue-50 border-blue-100 text-blue-600',
    green: 'bg-green-50 border-green-100 text-green-600',
    orange: 'bg-orange-50 border-orange-100 text-orange-600',
    purple: 'bg-purple-50 border-purple-100 text-purple-600',
    yellow: 'bg-yellow-50 border-yellow-100 text-yellow-600',
  };

  const bgClasses = {
    blue: 'bg-white',
    green: 'bg-white',
    orange: 'bg-white',
    purple: 'bg-white',
    yellow: 'bg-white',
  };

  return (
    <div className={`rounded-xl border border-gray-100 overflow-hidden ${bgClasses[color] || 'bg-white'}`}>
      <div className={`flex items-center gap-2 px-4 py-3 border-b border-gray-100 ${colorClasses[color] || ''}`}>
        {icon}
        <h3 className="font-semibold text-sm">{title}</h3>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
};

export default OverviewPanel;
