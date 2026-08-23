function documentViewerIdentity(docInfo) {
  return [
    docInfo?.pdf_url || '',
    docInfo?.pages || docInfo?.data?.pages || null,
    docInfo?.total_pages || docInfo?.data?.total_pages || 0,
  ];
}

export function pdfWorkspacePanePropsAreEqual(prev, next) {
  const [prevUrl, prevPages, prevTotal] = documentViewerIdentity(prev.docInfo);
  const [nextUrl, nextPages, nextTotal] = documentViewerIdentity(next.docInfo);
  return prev.darkMode === next.darkMode
    && prev.pdfPanelWidth === next.pdfPanelWidth
    && prevUrl === nextUrl
    && prevPages === nextPages
    && prevTotal === nextTotal
    && prev.currentPage === next.currentPage
    && prev.pdfScale === next.pdfScale
    && prev.activeHighlight === next.activeHighlight
    && prev.documentHighlights === next.documentHighlights
    && prev.isSelectingArea === next.isSelectingArea
    && prev.blockIndex === next.blockIndex
    && prev.activeReadingBlockId === next.activeReadingBlockId
    && prev.focusedReadingBlockIds === next.focusedReadingBlockIds
    && prev.focusPulseToken === next.focusPulseToken
    && prev.navigationRequest === next.navigationRequest
    && prev.blockTranslations === next.blockTranslations
    && prev.translatingBlockIds === next.translatingBlockIds
    && prev.hasDockedSelectionToolbar === next.hasDockedSelectionToolbar
    && prev.selectionToolbar === next.selectionToolbar
    && prev.translationSurface === next.translationSurface
    && prev.searchStatus === next.searchStatus
    && prev.pdfContainerRef === next.pdfContainerRef
    && prev.onPageChange === next.onPageChange
    && prev.onPdfScaleChange === next.onPdfScaleChange
    && prev.onSavedHighlightClick === next.onSavedHighlightClick
    && prev.onAreaSelected === next.onAreaSelected
    && prev.onSelectionCancel === next.onSelectionCancel
    && prev.onTextSelect === next.onTextSelect
    && prev.onToggleSidebar === next.onToggleSidebar
    && prev.onBlockHover === next.onBlockHover
    && prev.onBlockClick === next.onBlockClick
    && prev.onTranslationSurfaceChange === next.onTranslationSurfaceChange
    && prev.onTextSelection === next.onTextSelection
    && prev.onDismissSearchStatus === next.onDismissSearchStatus;
}
