import React from 'react';
import { Check, Loader2, ScanText, TriangleAlert } from 'lucide-react';
import MinerUScanLoader from './MinerUScanLoader';

const STATUS_PRESENTATION = {
  processing: {
    Icon: Loader2,
    iconClass: 'text-[#8B6A52]',
    animate: false,
  },
  complete: {
    Icon: ScanText,
    iconClass: 'text-emerald-600 dark:text-emerald-400',
    animate: false,
  },
  warning: {
    Icon: TriangleAlert,
    iconClass: 'text-amber-600 dark:text-amber-400',
    animate: false,
  },
  failed: {
    Icon: TriangleAlert,
    iconClass: 'text-red-600 dark:text-red-400',
    animate: false,
  },
};

const clampPercent = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, Math.round(numeric)));
};

const isMinerUStatusItem = (item) => (
  item?.id === 'mineru_parse' || String(item?.title || '').includes('MinerU')
);

const stripLegacyStatusPrefix = (value) => String(value || '')
  .replace(/^(?:\s|\u2705|\u23F3|\u{1F50D}|\u26A0|\uFE0F)+/u, '')
  .trim();

export function resolveDocumentUploadNotice(message) {
  if (message?.notice?.type === 'document_upload') return message.notice;

  const content = String(message?.content || '').trim();
  const header = content.match(/文档《(.+?)》(?:上传成功[！!]?|已上传)[，,]?共\s*(\d+)\s*页/u);
  if (!header) return null;

  const lines = content.split('\n').slice(1).map(stripLegacyStatusPrefix).filter(Boolean);
  const items = [];
  const hasMinerU = lines.some((line) => line.includes('MinerU'));
  const ocrWarning = lines.find((line) => (
    line.includes('OCR')
    && !line.startsWith('已使用 OCR')
    && /(未完成|失败|不可用|部分|警告|注意)/u.test(line)
  ));
  const ocrComplete = lines.find((line) => line.startsWith('已使用 OCR'));

  if (hasMinerU) {
    items.push({
      id: 'mineru_parse',
      status: 'processing',
      title: 'MinerU 全程解析中',
      description: 'PDF 可先阅读；正文、速览、大纲、翻译和问答将在同一解析结果发布后启用。',
    });
  }
  if (ocrWarning) {
    items.push({
      id: 'ocr',
      status: 'warning',
      title: 'OCR 需要注意',
      description: ocrWarning.replace(/^OCR\s*/u, ''),
    });
  } else if (ocrComplete) {
    items.push({
      id: 'ocr',
      status: 'complete',
      title: 'OCR 处理完成',
      description: ocrComplete.replace(
        /^已使用\s*OCR(?:（([^）]+)）)?/u,
        (_match, backend) => (backend ? `使用 ${backend} ` : ''),
      ),
    });
  }
  if (!hasMinerU && lines.some((line) => line.includes('检索索引'))) {
    items.push({
      id: 'rag_index',
      status: 'processing',
      title: '问答索引准备中',
      description: 'PDF 与正文阅读可先使用；首次问答可能需要稍等。',
    });
  }

  return {
    type: 'document_upload',
    filename: header[1],
    pageCount: Number(header[2]),
    items,
  };
}

export default function DocumentUploadNotice({ notice, liveParseStatus = null }) {
  const filename = String(notice?.filename || '未命名文档');
  const pageCount = Number(notice?.pageCount || 0);
  const items = Array.isArray(notice?.items) ? notice.items : [];

  return (
    <section
      aria-label="文档上传状态"
      aria-live="polite"
      className="w-full max-w-[760px] overflow-hidden rounded-[18px] border border-[#E7E1D9] bg-white/95 shadow-[0_10px_28px_rgba(76,60,43,0.07)] dark:border-white/10 dark:bg-[#20201F] dark:shadow-none"
    >
      <header className="flex items-start gap-3 px-[18px] py-4">
        <span className="mt-0.5 flex h-8 w-8 flex-none items-center justify-center rounded-[10px] bg-emerald-50 text-emerald-600 dark:bg-emerald-400/10 dark:text-emerald-300">
          <Check className="h-4 w-4" strokeWidth={2.2} aria-hidden="true" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-baseline gap-2.5">
            <h2 className="flex-none text-[14px] font-semibold text-[#262421] dark:text-gray-100">
              文档已上传
            </h2>
            {pageCount > 0 && (
              <span className="text-[12px] tabular-nums text-[#8A8279] dark:text-gray-400">
                {pageCount} 页
              </span>
            )}
          </div>
          <p
            className="mt-0.5 truncate text-[12.5px] leading-5 text-[#625D56] dark:text-gray-300"
            title={filename}
          >
            {filename}
          </p>
        </div>
      </header>

      {items.length > 0 && (
        <div className="border-t border-[#EEE9E3] bg-[#FCFBF9]/80 px-[18px] py-3.5 dark:border-white/10 dark:bg-white/[0.025]">
          <div className="space-y-3">
            {items.map((item, index) => {
              const isMinerUItem = isMinerUStatusItem(item);
              const effectiveItem = isMinerUItem && liveParseStatus
                ? { ...item, ...liveParseStatus }
                : item;
              // 上传消息会被持久化，旧的 processing 只能当作历史快照。
              // 只有当前文档的实时状态明确仍在处理时才播放动画。
              const isLiveMinerUProcessing = Boolean(
                isMinerUItem
                && liveParseStatus
                && effectiveItem?.status === 'processing'
              );
              const presentation = STATUS_PRESENTATION[effectiveItem?.status] || STATUS_PRESENTATION.processing;
              const StatusIcon = presentation.Icon;
              const progress = isLiveMinerUProcessing
                ? effectiveItem?.progress
                : null;
              const progressPercent = clampPercent(progress?.percent);
              return (
                <div
                  key={`${item?.id || item?.title || 'status'}-${index}`}
                  className="grid grid-cols-[20px_minmax(0,1fr)] items-start gap-3"
                >
                  {isLiveMinerUProcessing ? (
                    <MinerUScanLoader size={19} className="mt-0.5 text-[#D97A5D] dark:text-[#FFA07A]" />
                  ) : (
                    <StatusIcon
                      className={`mt-0.5 h-4 w-4 ${presentation.iconClass} ${presentation.animate ? 'animate-spin' : ''}`}
                      strokeWidth={1.9}
                      aria-hidden="true"
                    />
                  )}
                  <div className="min-w-0 text-[12.5px] leading-[1.6]">
                    <div className="flex min-w-0 items-baseline justify-between gap-3">
                      <span className="truncate font-semibold text-[#37332F] dark:text-gray-200">
                        {effectiveItem?.title}
                      </span>
                      {progressPercent !== null && (
                        <span className="shrink-0 text-[11px] font-semibold tabular-nums text-[#B85F47] dark:text-[#FFA07A]">
                          {progress?.label || `${progressPercent}%`}
                        </span>
                      )}
                    </div>
                    {effectiveItem?.description && (
                      <p className="mt-0.5 text-[12px] leading-5 text-[#777068] dark:text-gray-400">
                        {effectiveItem.description}
                      </p>
                    )}
                    {progressPercent !== null && (
                      <div
                        role="progressbar"
                        aria-label={`MinerU 解析：${progress?.ariaLabel || `${progressPercent}%`}`}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={progressPercent}
                        className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-[#EDE6DF] dark:bg-white/10"
                      >
                        <div
                          className="h-full w-full origin-left rounded-full bg-[#D97A5D] transition-transform duration-500 ease-out motion-reduce:transition-none"
                          style={{ transform: `scaleX(${progressPercent / 100})` }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
