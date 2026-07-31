import React, { useMemo } from 'react';
import { ArrowUpRight } from 'lucide-react';

export const resolveEvidenceRef = (citation) => {
  const displayRef = Number(citation?.display_ref);
  if (Number.isFinite(displayRef)) return displayRef;
  const ref = Number(citation?.ref);
  return Number.isFinite(ref) ? ref : null;
};

export const partitionEvidenceCitations = (citations = []) => {
  const sorted = [...(Array.isArray(citations) ? citations : [])]
    .filter(Boolean)
    .sort((a, b) => {
      const leftRef = resolveEvidenceRef(a);
      const rightRef = resolveEvidenceRef(b);
      if (Number.isFinite(leftRef) && Number.isFinite(rightRef) && leftRef !== rightRef) {
        return leftRef - rightRef;
      }
      return String(a?.group_id || '').localeCompare(String(b?.group_id || ''));
    });

  return {
    cited: sorted.filter((citation) => Boolean(citation?.highlight_text)),
    uncited: sorted.filter((citation) => !citation?.highlight_text),
  };
};

const compactText = (value, limit = 72) => {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (!text) return '';
  return text.length > limit ? `${text.slice(0, limit).trimEnd()}...` : text;
};

const isOpaqueCitationId = (value) => {
  const text = String(value || '').trim();
  return !text
    || /^(?:g|group|chunk|semantic|block|visual)[\s:_-]*[\w-]+$/i.test(text)
    || /^[a-f0-9]{12,}$/i.test(text);
};

export const getCitationPageLabel = (citation) => {
  const range = Array.isArray(citation?.page_range) ? citation.page_range : [];
  const first = Number(range[0]);
  const last = Number(range[1] ?? range[0]);
  if (!Number.isFinite(first) || first <= 0) return '';
  if (!Number.isFinite(last) || last <= 0 || last === first) return `第 ${first} 页`;
  return `第 ${first}-${last} 页`;
};

export const getCitationSourceLabel = (citation) => {
  const candidates = [
    citation?.doc_name,
    citation?.document_name,
    citation?.section_title,
    citation?.section,
    citation?.heading,
    citation?.title,
    citation?.source_title,
    citation?.filename,
  ];
  const namedSource = candidates
    .map((value) => compactText(value, 88))
    .find((value) => value && !isOpaqueCitationId(value));
  if (namedSource) return namedSource;

  const groupId = compactText(citation?.group_id, 64);
  if (!isOpaqueCitationId(groupId)) return groupId;

  const excerpt = compactText(
    citation?.highlight_text || citation?.display_text || citation?.source_text,
    72,
  );
  return excerpt || '文档原文';
};

const getCitationTooltip = (citation, ref) => {
  const source = getCitationSourceLabel(citation);
  const page = getCitationPageLabel(citation);
  const excerpt = compactText(citation?.highlight_text || citation?.display_text || citation?.source_text, 180);
  return [
    `引用 ${ref}${source ? `：${source}` : ''}`,
    page,
    excerpt && excerpt !== source ? excerpt : '',
  ].filter(Boolean).join('\n');
};

/**
 * 回答下方的简洁引用脚注。
 *
 * 正文上标和这里的编号保持一一对应。点击整行仍会跳到 PDF 并高亮原文，
 * 只去掉会打断阅读节奏的卡片、缩略图和二级展开层。
 */
export default function EvidencePanel({ citations, onCitationClick, activeRef, onRefHover }) {
  const entries = useMemo(() => {
    const { cited, uncited } = partitionEvidenceCitations(citations || []);
    return [...cited, ...uncited];
  }, [citations]);

  if (entries.length === 0) return null;

  return (
    <section className="mt-4 max-w-[44rem]" aria-label="引用来源" data-testid="citation-footer">
      <div className="h-px bg-stone-200/90 dark:bg-white/10" />
      <div className="space-y-px pt-2">
        <span className="sr-only">引用来源</span>
        {entries.map((citation, index) => {
          const ref = resolveEvidenceRef(citation);
          const fallbackRef = index + 1;
          const displayRef = Number.isFinite(ref) ? ref : fallbackRef;
          const source = getCitationSourceLabel(citation);
          const page = getCitationPageLabel(citation);
          const isActive = activeRef === displayRef || activeRef === Number(citation?.ref);

          return (
            <button
              key={`${displayRef}-${citation?.evidence_id || citation?.group_id || index}`}
              type="button"
              title={getCitationTooltip(citation, displayRef)}
              onClick={() => onCitationClick?.(citation)}
              onMouseEnter={() => onRefHover?.(displayRef)}
              onMouseLeave={() => onRefHover?.(null)}
              onFocus={() => onRefHover?.(displayRef)}
              onBlur={() => onRefHover?.(null)}
              className={`group -mx-1 flex w-full items-center gap-1.5 rounded-[6px] px-1 py-1 text-left text-[12px] leading-5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#F5B49C]/70 ${
                isActive
                  ? 'bg-[#FFF5EF] text-[#9B513C] dark:bg-[#3D2922] dark:text-[#F2B29A]'
                  : 'text-stone-500 hover:bg-[#FFF8F4] hover:text-stone-700 dark:text-stone-400 dark:hover:bg-white/5 dark:hover:text-stone-200'
              }`}
              aria-label={`跳转到引用 ${displayRef}${page ? `，${page}` : ''}`}
            >
              <span className="w-3.5 shrink-0 text-right text-[11px] font-semibold tabular-nums text-stone-400 group-hover:text-[#B85F47] dark:text-stone-500">
                {displayRef}
              </span>
              <span className="min-w-0 truncate">{source}</span>
              {page && <span className="shrink-0 text-stone-400 dark:text-stone-500">· {page}</span>}
              <ArrowUpRight className="h-3 w-3 shrink-0 text-stone-300 opacity-0 transition-[opacity,transform] duration-150 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:opacity-100 group-focus-visible:opacity-100 dark:text-stone-600" aria-hidden="true" />
            </button>
          );
        })}
      </div>
    </section>
  );
}
