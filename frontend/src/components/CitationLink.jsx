import React from 'react';

/**
 * CitationLink - 引文引用链接组件
 *
 * 渲染一个可点击的引用编号徽章（如 [1]、[2]），
 * 点击时触发 onCitationClick 回调，用于跳转 PDF 阅读器到对应页码。
 *
 * @param {number} refNumber - 引用编号
 * @param {object|null} citation - 引文数据，包含 ref、group_id、page_range
 * @param {function} onClick - 点击回调，参数为 citation 对象
 */
const CitationLink = React.memo(({ refNumber, citation, onClick }) => {
  // 处理点击事件
  const handleClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (citation && onClick) {
      onClick(citation);
    }
  };

  // 如果没有引文数据，渲染为普通文本
  if (!citation) {
    return <span className="text-gray-500">[{refNumber}]</span>;
  }

  // 构建 tooltip 提示文本
  const pageRange = citation.page_range;
  const tooltipText = pageRange
    ? `点击跳转并高亮：第 ${pageRange[0]}${pageRange[1] !== pageRange[0] ? `-${pageRange[1]}` : ''} 页`
    : `引用 [${refNumber}]`;

  return (
    <button
      type="button"
      onClick={handleClick}
      title={tooltipText}
      className="inline-flex items-center justify-center min-w-[1.5em] px-1 py-0 mx-0.5 text-xs font-semibold text-blue-600 bg-blue-50 border border-blue-200 rounded hover:bg-blue-100 hover:text-blue-700 hover:border-blue-300 cursor-pointer transition-colors duration-150 align-baseline leading-tight no-underline"
      style={{ fontSize: '0.8em', verticalAlign: 'super' }}
    >
      {refNumber}
    </button>
  );
});

CitationLink.displayName = 'CitationLink';

export default CitationLink;
