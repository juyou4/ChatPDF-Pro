import React, { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

// Configure worker - 直接指定版本以确保匹配
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.8.69/pdf.worker.min.mjs`;

const PDFViewer = ({ pdfUrl, onTextSelect }) => {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(1);
    const [scale, setScale] = useState(1.0);
    const [selectedText, setSelectedText] = useState('');
    const [error, setError] = useState(null);

    // 确保 PDF URL 是完整路径
    const fullPdfUrl = pdfUrl?.startsWith('http') ? pdfUrl : `${window.location.origin}${pdfUrl}`;

    console.log('📄 PDFViewer - Loading PDF:', fullPdfUrl);

    function onDocumentLoadSuccess({ numPages }) {
        console.log('✅ PDF loaded successfully, pages:', numPages);
        setNumPages(numPages);
        setError(null);
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
        setPageNumber(prevPageNumber => Math.max(1, Math.min(prevPageNumber + offset, numPages || 1)));
    };

    const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0));
    const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5));

    return (
        <div className="h-full flex flex-col bg-gray-50 rounded-2xl overflow-hidden">
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
            <div className="flex-1 overflow-auto p-6 flex items-start justify-center bg-gray-50" onMouseUp={handleTextSelection}>
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
                        <Page pageNumber={pageNumber} scale={scale} renderTextLayer={true} renderAnnotationLayer={true} />
                    </Document>
                )}
            </div>
            {selectedText && (
                <div className="p-3 bg-blue-50 border-t border-blue-100">
                    <div className="text-xs text-blue-600 font-medium">已选择文本</div>
                    <div className="text-sm text-gray-700 mt-1 line-clamp-2">{selectedText}</div>
                </div>
            )}
        </div>
    );
};

export default PDFViewer;
