import React, { lazy, Suspense } from 'react';
import { ChevronLeft, ChevronRight, Scan, X, ZoomIn, ZoomOut } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { pdfWorkspacePanePropsAreEqual } from '../utils/pdfWorkspacePaneMemo';
import BreatheLoader from './BreatheLoader';

export const loadPDFViewer = () => import('./PDFViewer');
export const preloadPDFViewer = () => {
  void loadPDFViewer().catch(() => {});
};
const PDFViewer = lazy(loadPDFViewer);
const EMPTY_ID_LIST = Object.freeze([]);

export { pdfWorkspacePanePropsAreEqual };

function PdfWorkspacePaneInner({
  darkMode,
  pdfPanelWidth,
  docInfo,
  currentPage,
  onPageChange,
  pdfScale,
  onPdfScaleChange,
  pdfContainerRef,
  activeHighlight,
  documentHighlights,
  onSavedHighlightClick,
  isSelectingArea,
  onAreaSelected,
  onSelectionCancel,
  onTextSelect,
  onToggleSidebar,
  blockIndex,
  activeReadingBlockId,
  focusedReadingBlockIds,
  focusPulseToken,
  navigationRequest,
  onBlockHover,
  onBlockClick,
  blockTranslations,
  translatingBlockIds,
  hasDockedSelectionToolbar,
  selectionToolbar,
  translationSurface,
  onTranslationSurfaceChange,
  onTextSelection,
  searchStatus,
  onDismissSearchStatus,
}) {
  const totalPages = docInfo?.total_pages || docInfo?.data?.total_pages || 1;
  const pageContent = (docInfo?.pages || docInfo?.data?.pages)?.[currentPage - 1]?.content || 'No content';

  return (
    <motion.div
      initial={false}
      animate={{ opacity: 1, y: 0 }}
      className={`workspace-pane workspace-pane-pdf overflow-hidden flex flex-col relative flex-shrink-0 min-w-0 ${darkMode ? 'workspace-pane-dark' : ''}`}
      style={{ width: `${pdfPanelWidth}%`, minWidth: '350px' }}
    >
      <div className="flex-1 overflow-hidden">
        {docInfo?.pdf_url ? (
          <Suspense fallback={
            <div
              className={`flex h-full flex-col items-center justify-center gap-4 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
              role="status"
              aria-live="polite"
            >
              <BreatheLoader className={darkMode ? 'text-[#FFA07A]' : 'text-[#D97A5D]'} />
              <span>加载 PDF 中...</span>
            </div>
          }>
            <PDFViewer
              ref={pdfContainerRef}
              pdfUrl={docInfo.pdf_url}
              page={currentPage}
              onPageChange={onPageChange}
              highlightInfo={activeHighlight}
              savedHighlights={documentHighlights}
              onSavedHighlightClick={onSavedHighlightClick}
              isSelecting={isSelectingArea}
              onAreaSelected={onAreaSelected}
              onSelectionCancel={onSelectionCancel}
              darkMode={darkMode}
              onTextSelect={onTextSelect}
              onToggleSidebar={onToggleSidebar}
              blockIndex={blockIndex}
              activeBlockId={activeReadingBlockId}
              focusedBlockIds={focusedReadingBlockIds}
              focusPulseToken={focusPulseToken}
              navigationRequest={navigationRequest}
              visitedBlockIds={EMPTY_ID_LIST}
              inlineTranslationBlockIds={EMPTY_ID_LIST}
              onBlockHover={onBlockHover}
              onBlockClick={onBlockClick}
              blockTranslations={blockTranslations}
              translatingBlockIds={translatingBlockIds}
              hasDockedSelectionToolbar={hasDockedSelectionToolbar}
              selectionToolbar={selectionToolbar}
              translationSurface={translationSurface}
              onTranslationSurfaceChange={onTranslationSurfaceChange}
            />
          </Suspense>
        ) : (docInfo?.pages || docInfo?.data?.pages) ? (
          <>
            <div data-pdf-reader-toolbar className={`relative z-30 flex h-14 items-center justify-between border-b px-6 backdrop-blur-sm ${darkMode ? 'border-white/[0.08] bg-[#1a1d21] text-gray-200' : 'border-black/5 bg-white/30 text-gray-600'}`}>
              <div className="flex items-center gap-2" role="toolbar" aria-label="页码导航">
                <button
                  type="button"
                  onClick={() => onPageChange(Math.max(1, currentPage - 1))}
                  disabled={currentPage <= 1}
                  className={`rounded-lg p-1.5 transition-[background-color,color,transform] duration-200 active:scale-95 disabled:cursor-default disabled:opacity-45 disabled:active:scale-100 ${darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-100' : 'text-gray-500 hover:bg-black/5 hover:text-gray-800'}`}
                  title="上一页"
                  aria-label="上一页"
                >
                  <ChevronLeft className="h-5 w-5" />
                </button>
                <span className={`w-16 text-center text-sm font-medium tabular-nums ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} aria-live="polite">
                  {currentPage} / {totalPages}
                </span>
                <button
                  type="button"
                  onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage >= totalPages}
                  className={`rounded-lg p-1.5 transition-[background-color,color,transform] duration-200 active:scale-95 disabled:cursor-default disabled:opacity-45 disabled:active:scale-100 ${darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-100' : 'text-gray-500 hover:bg-black/5 hover:text-gray-800'}`}
                  title="下一页"
                  aria-label="下一页"
                >
                  <ChevronRight className="h-5 w-5" />
                </button>
              </div>
              <div className="flex items-center gap-2" role="toolbar" aria-label="缩放控制">
                <button
                  type="button"
                  onClick={() => onPdfScaleChange((s) => Math.max(0.5, s - 0.1))}
                  disabled={pdfScale <= 0.5}
                  className={`rounded-lg p-1.5 transition-[background-color,color,transform] duration-200 active:scale-95 disabled:cursor-default disabled:opacity-45 disabled:active:scale-100 ${darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-100' : 'text-gray-500 hover:bg-black/5 hover:text-gray-800'}`}
                  title="缩小"
                  aria-label="缩小"
                >
                  <ZoomOut className="h-5 w-5" />
                </button>
                <span className={`w-12 text-center text-sm font-medium tabular-nums ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} aria-live="polite">
                  {Math.round(pdfScale * 100)}%
                </span>
                <button
                  type="button"
                  onClick={() => onPdfScaleChange((s) => Math.min(2.0, s + 0.1))}
                  disabled={pdfScale >= 2.0}
                  className={`rounded-lg p-1.5 transition-[background-color,color,transform] duration-200 active:scale-95 disabled:cursor-default disabled:opacity-45 disabled:active:scale-100 ${darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-100' : 'text-gray-500 hover:bg-black/5 hover:text-gray-800'}`}
                  title="放大"
                  aria-label="放大"
                >
                  <ZoomIn className="h-5 w-5" />
                </button>
              </div>
            </div>
            {selectionToolbar}
            <div ref={pdfContainerRef} className="h-full overflow-auto bg-gray-50/50">
              <div
                className="min-h-full flex items-start justify-center p-8 transition-[padding] duration-200"
                style={{ zoom: pdfScale }}
              >
                <div className="bg-white shadow-2xl p-12 rounded-lg max-w-4xl w-full" onMouseUp={onTextSelection}>
                  <pre className="whitespace-pre-wrap font-serif text-gray-800 leading-relaxed">
                    {pageContent}
                  </pre>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            <p>Loading PDF...</p>
          </div>
        )}
      </div>

      <AnimatePresence initial={false}>
        {['degraded', 'error', 'stale'].includes(searchStatus?.state) && (
          <motion.div
            key={`${searchStatus.state}:${searchStatus.errorCode || ''}`}
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 360, damping: 30 }}
            role="status"
            aria-live="polite"
            className="pointer-events-none absolute left-4 right-4 top-[72px] z-40 flex justify-end"
          >
            <div className={`pointer-events-auto flex w-full max-w-[430px] items-start gap-3 rounded-[16px] border px-4 py-3 shadow-[0_16px_36px_rgba(15,23,42,0.16)] backdrop-blur-xl ${
              darkMode
                ? 'border-amber-300/15 bg-[#25282f]/95 text-gray-100'
                : 'border-amber-200/80 bg-white/95 text-gray-700'
            }`}>
              <div className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
                searchStatus.state === 'degraded'
                  ? (darkMode ? 'bg-amber-300/10 text-amber-300' : 'bg-amber-50 text-amber-600')
                  : (darkMode ? 'bg-rose-300/10 text-rose-300' : 'bg-rose-50 text-rose-500')
              }`}>
                <Scan size={15} strokeWidth={2.2} />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-[13px] font-semibold leading-5">
                  {searchStatus.state === 'degraded'
                    ? (searchStatus.resultCount > 0 ? '检索已降级' : '检索暂时不可用')
                    : searchStatus.state === 'stale' ? '搜索结果已失效' : '搜索未完成'}
                </p>
                <p className={`mt-0.5 text-[12px] leading-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {searchStatus.message}
                </p>
              </div>
              <button
                type="button"
                onClick={onDismissSearchStatus}
                className={`mt-0.5 rounded-full p-1 transition-colors ${darkMode ? 'text-gray-500 hover:bg-white/10 hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-700'}`}
                aria-label="关闭搜索状态提示"
                title="关闭"
              >
                <X size={15} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

const PdfWorkspacePane = React.memo(PdfWorkspacePaneInner, pdfWorkspacePanePropsAreEqual);
PdfWorkspacePane.displayName = 'PdfWorkspacePane';

export default PdfWorkspacePane;
