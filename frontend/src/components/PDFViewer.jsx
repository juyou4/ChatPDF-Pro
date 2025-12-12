import React, { useState, useEffect, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

// Configure worker - 直接指定版本以确保匹配
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

const PDFViewer = ({ pdfUrl, onTextSelect, highlightInfo = null, page = 1, onPageChange }) => {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(page || 1);
    const [scale, setScale] = useState(1.0);
    const [selectedText, setSelectedText] = useState('');
    const [error, setError] = useState(null);

    useEffect(() => {
        if (typeof page === 'number' && page > 0 && page !== pageNumber) {
            setPageNumber(page);
        }
    }, [page, pageNumber]);

    // 确保 PDF URL 是完整路径
    const fullPdfUrl = pdfUrl?.startsWith('http') ? pdfUrl : `${window.location.origin}${pdfUrl}`;

    console.log('📄 PDFViewer - Loading PDF:', fullPdfUrl);

    function onDocumentLoadSuccess({ numPages }) {
        console.log('✅ PDF loaded successfully, pages:', numPages);
        setNumPages(numPages);
        setError(null);
        setPageNumber(prev => {
            const safePage = Math.min(Math.max(prev, 1), numPages);
            if (onPageChange && safePage !== prev) {
                onPageChange(safePage);
            }
            return safePage;
        });
    }

    function onDocumentLoadError(error) {
        console.error('❌ PDF load error:', error);
        setError(error.message || 'Failed to load PDF');
    }

    const handleTextSelection = () => {
        const selection = window.getSelection();
        const text = selection.toString().trim();
        if (text) {
            setSelectedText(text);
            if (onTextSelect) {
                onTextSelect(text);
            }
        }
    };

    const changePage = (offset) => {
        setPageNumber(prevPageNumber => {
            const nextPage = Math.max(1, Math.min(prevPageNumber + offset, numPages || prevPageNumber || 1));
            if (onPageChange) {
                onPageChange(nextPage);
            }
            return nextPage;
        });
    };

    const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0));
    const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5));

    const [highlightRect, setHighlightRect] = useState(null);
    const pageRef = useRef(null);

    useEffect(() => {
        let isMounted = true;
        let retryTimer = null;

        if (!highlightInfo || highlightInfo.page !== pageNumber || !highlightInfo.text) {
            setHighlightRect(null);
            return;
        }

        const findHighlight = () => {
            if (!isMounted) return;

            const pageElement = pageRef.current;
            if (!pageElement) return;

            const textLayer = pageElement.querySelector('.react-pdf__Page__textContent');
            if (!textLayer || textLayer.children.length === 0) {
                // Retry if text layer is not ready
                retryTimer = setTimeout(findHighlight, 100);
                return;
            }

            try {
                const spans = Array.from(textLayer.querySelectorAll('span'));
                let fullText = '';

                // Build full text
                spans.forEach(span => {
                    fullText += span.textContent;
                });

                if (!fullText) return;

                // Normalize strings for comparison (remove all whitespace)
                const searchStr = String(highlightInfo.text).replace(/\s+/g, '').toLowerCase();
                const pageStr = fullText.replace(/\s+/g, '').toLowerCase();

                // Strategy 1: Exact match
                let startIndex = pageStr.indexOf(searchStr);
                let endIndex = -1;

                if (startIndex !== -1) {
                    endIndex = startIndex + searchStr.length;
                } else {
                    // Strategy 2: Multi-anchor matching with flexible sizes
                    const anchorSize = Math.min(12, Math.floor(searchStr.length * 0.15));
                    const startAnchor = searchStr.substring(0, anchorSize);
                    const endAnchor = searchStr.substring(searchStr.length - anchorSize);

                    const startAnchorIndex = pageStr.indexOf(startAnchor);

                    if (startAnchorIndex !== -1) {
                        // Try to find end anchor
                        const endAnchorIndex = pageStr.indexOf(endAnchor, startAnchorIndex + anchorSize);

                        if (endAnchorIndex !== -1 && endAnchorIndex > startAnchorIndex) {
                            // Both anchors found
                            startIndex = startAnchorIndex;
                            endIndex = endAnchorIndex + endAnchor.length;
                        } else {
                            // Try middle anchor as fallback
                            const midPoint = Math.floor(searchStr.length / 2);
                            const midAnchor = searchStr.substring(midPoint, midPoint + anchorSize);
                            const midAnchorIndex = pageStr.indexOf(midAnchor, startAnchorIndex);

                            if (midAnchorIndex !== -1) {
                                // Use start to estimated end based on original length
                                startIndex = startAnchorIndex;
                                endIndex = Math.min(startIndex + Math.floor(searchStr.length * 1.3), pageStr.length);
                            } else {
                                // Last resort: character-by-character match from start
                                startIndex = startAnchorIndex;
                                let matchLen = anchorSize;
                                while (matchLen < searchStr.length && startIndex + matchLen < pageStr.length) {
                                    if (pageStr[startIndex + matchLen] === searchStr[matchLen]) {
                                        matchLen++;
                                    } else {
                                        break;
                                    }
                                }
                                endIndex = startIndex + matchLen;
                            }
                        }
                    }
                }

                if (startIndex === -1 || endIndex === -1) return;

                // Map string indices to DOM nodes
                let startNode = null;
                let startOffset = 0;
                let endNode = null;
                let endOffset = 0;

                let currentCharCount = 0;
                let foundStart = false;
                let foundEnd = false;

                for (const span of spans) {
                    const text = span.textContent;
                    const cleanText = text.replace(/\s+/g, '');
                    const spanLength = cleanText.length;

                    if (!foundStart) {
                        if (currentCharCount + spanLength > startIndex) {
                            foundStart = true;
                            // Find exact offset in this span
                            let localCount = 0;
                            for (let i = 0; i < text.length; i++) {
                                if (!/\s/.test(text[i])) {
                                    if (currentCharCount + localCount === startIndex) {
                                        startNode = span.firstChild;
                                        startOffset = i;
                                        break;
                                    }
                                    localCount++;
                                }
                            }
                        }
                    }

                    if (foundStart && !foundEnd) {
                        if (currentCharCount + spanLength >= endIndex) {
                            foundEnd = true;
                            // Find exact end offset
                            let localCount = 0;
                            for (let i = 0; i < text.length; i++) {
                                if (!/\s/.test(text[i])) {
                                    localCount++;
                                    if (currentCharCount + localCount === endIndex) {
                                        endNode = span.firstChild;
                                        endOffset = i + 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    currentCharCount += spanLength;
                    if (foundEnd) break;
                }

                if (startNode && endNode) {
                    const range = document.createRange();
                    range.setStart(startNode, startOffset);
                    range.setEnd(endNode, endOffset);
                    const rects = Array.from(range.getClientRects());

                    if (rects.length > 0) {
                        const pageRect = pageElement.getBoundingClientRect();

                        // Calculate union rect with padding
                        const padding = 4;
                        const unionRect = {
                            top: Math.min(...rects.map(r => r.top)) - pageRect.top - padding,
                            left: Math.min(...rects.map(r => r.left)) - pageRect.left - padding,
                            right: Math.max(...rects.map(r => r.right)) - pageRect.left + padding,
                            bottom: Math.max(...rects.map(r => r.bottom)) - pageRect.top + padding
                        };

                        if (isMounted) {
                            setHighlightRect({
                                top: unionRect.top,
                                left: unionRect.left,
                                width: unionRect.right - unionRect.left,
                                height: unionRect.bottom - unionRect.top
                            });
                        }
                    }
                }
            } catch (e) {
                console.error('Error calculating highlight:', e);
            }
        };

        // Debounce slightly to allow rendering to settle
        const initialTimer = setTimeout(findHighlight, 300);

        return () => {
            isMounted = false;
            clearTimeout(initialTimer);
            if (retryTimer) clearTimeout(retryTimer);
        };

    }, [highlightInfo, pageNumber, scale, numPages]);

    return (
        <div className="relative h-full flex flex-col bg-[var(--color-bg-base)] rounded-2xl overflow-hidden">
            <div className="flex items-center justify-between p-4 bg-white border-b border-gray-200">
                <div className="flex items-center gap-2">
                    <button onClick={() => changePage(-1)} disabled={pageNumber <= 1} className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50">
                        <ChevronLeft className="w-5 h-5" />
                    </button>
                    <span className="text-sm font-medium px-3">{pageNumber} / {numPages || '--'}</span>
                    <button onClick={() => changePage(1)} disabled={pageNumber >= (numPages || 1)} className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50">
                        <ChevronRight className="w-5 h-5" />
                    </button>
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={zoomOut} className="p-2 rounded-lg hover:bg-gray-100">
                        <ZoomOut className="w-5 h-5" />
                    </button>
                    <span className="text-sm font-medium px-2">{Math.round(scale * 100)}%</span>
                    <button onClick={zoomIn} className="p-2 rounded-lg hover:bg-gray-100">
                        <ZoomIn className="w-5 h-5" />
                    </button>
                </div>
            </div>
            <div className="flex-1 overflow-auto p-6 flex items-start justify-center bg-[var(--color-bg-base)] pdf-scroll" onMouseUp={handleTextSelection}>
                {error ? (
                    <div className="flex flex-col items-center justify-center h-full text-center p-8">
                        <div className="text-red-500 text-6xl mb-4">⚠️</div>
                        <div className="text-lg font-semibold text-gray-700 mb-2">PDF加载失败</div>
                        <div className="text-sm text-gray-500 mb-4">{error}</div>
                        <div className="text-xs text-gray-400 bg-gray-100 p-3 rounded-lg max-w-md">
                            <div className="font-mono break-all">URL: {fullPdfUrl}</div>
                        </div>
                    </div>
                ) : (
                    <Document
                        file={fullPdfUrl}
                        onLoadSuccess={onDocumentLoadSuccess}
                        onLoadError={onDocumentLoadError}
                        loading={
                            <div className="flex items-center justify-center h-full">
                                <div className="text-center">
                                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
                                    <div className="text-gray-500">加载PDF中...</div>
                                </div>
                            </div>
                        }
                    >
                        <div className="relative">
                            <Page
                                inputRef={pageRef}
                                pageNumber={pageNumber}
                                scale={scale}
                                renderTextLayer={true}
                                renderAnnotationLayer={true}
                            />
                            {/* Bounding Box Highlight with Spring Animation */}
                            <AnimatePresence>
                                {highlightRect && (
                                    <motion.div
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{
                                            opacity: 1,
                                            scale: 1,
                                            top: highlightRect.top,
                                            left: highlightRect.left,
                                            width: highlightRect.width,
                                            height: highlightRect.height
                                        }}
                                        exit={{ opacity: 0, scale: 0.9 }}
                                        transition={{
                                            type: "spring",
                                            stiffness: 300,
                                            damping: 30,
                                            mass: 1
                                        }}
                                        className="absolute border-2 border-blue-500 bg-blue-500/20 rounded-lg pointer-events-none z-10"
                                        style={{
                                            boxShadow: '0 0 0 2px rgba(59, 130, 246, 0.1), 0 4px 6px -1px rgba(59, 130, 246, 0.1)'
                                        }}
                                    >
                                        <div className="absolute -top-3 -right-3 bg-blue-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full shadow-sm">
                                            匹配
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </Document>
                )}
            </div>
        </div>
    );
};

export default PDFViewer;
