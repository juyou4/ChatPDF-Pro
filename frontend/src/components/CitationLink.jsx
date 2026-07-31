import React from 'react';

/**
 * CitationLink - 引文引用链接组件
 *
 * 渲染一个可点击的上标引用编号（如 1、2），
 * 点击时触发 onCitationClick 回调，用于跳转 PDF 阅读器到对应页码。
 *
 * @param {number} refNumber - 引用编号
 * @param {object|null} citation - 引文数据，包含 ref、group_id、page_range
 * @param {function} onClick - 点击回调，参数为 citation 对象
 */
const CitationLink = React.memo(({ refNumber, citation, onClick }) => {
  const descriptionId = React.useId();

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
  const documentLabel = citation.doc_name || citation.document_name || '';
  const tooltipText = pageRange
    ? `点击${documentLabel ? `打开 ${documentLabel} 并` : ''}跳转高亮：第 ${pageRange[0]}${pageRange[1] !== pageRange[0] ? `-${pageRange[1]}` : ''} 页`
    : `引用 [${refNumber}]`;

  return (
    <>
      <button
        type="button"
        onClick={handleClick}
        title={tooltipText}
        aria-describedby={descriptionId}
        className="mx-0.5 inline-flex min-w-[0.9em] items-center justify-center align-super text-[0.68em] font-semibold leading-none tabular-nums text-stone-400 transition-colors duration-150 hover:text-[#B85F47] focus-visible:rounded-[3px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#F5B49C]/70 dark:text-stone-500 dark:hover:text-[#F2B29A]"
      >
        {refNumber}
      </button>
      <span id={descriptionId} className="sr-only">{tooltipText}</span>
    </>
  );
});

CitationLink.displayName = 'CitationLink';

export default CitationLink;
